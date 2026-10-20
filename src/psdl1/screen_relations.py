"""학습 전 1차 스크리닝: 전극(층)별 사용 패턴과 형상으로 제품 쌍의 유사도를 수치로 매기고 relations.csv 초안을 만든다.

비교 단위는 모델이 보는 것과 같은 (role, layer) 요소 마스크와 CNT 합집합이다. 프레임 크기가 다른 제품도 공정하게 비교하려고
프레임에 맞춰 줄이지 않고, 모든 제품을 같은 정수 배율(`factor`, 픽셀 피치 × 배율 = 래스터 해상도)로 줄인 물리 래스터에서 비교한다.
제품 쌍마다
  1. 기준(CNT 합집합, 없거나 정보가 없으면 전극 합집합) 마스크의 교차상관으로 어긋남 후보를 찾고,
  2. 후보마다 모든 층에 같은 이동을 적용해 겹친 영역(ROI)에서 층별 거리를 구하고,
  3. 전체 유사도가 가장 큰 후보를 고른다.
층별 거리는 한쪽에서만 쓰이는 전극이면 1, 양쪽에서 쓰이면 이동을 허용한 형상 거리(1 - tolerant F1)다. 유사도는 1에서 층별 거리의 평균을
뺀 값(0~1)이다. 같은 계열(lineage)의 쌍은 similar, 같은 계열 쌍 중 가장 낮은 유사도보다 확실히 낮은 다른 계열 쌍만 different, 나머지는
모름으로 두고 검토 파일에 남긴다. 정답이 아니라 판단을 돕는 지표다."""
from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from .data import _iter_manifest_products
from .losses import validate_relations
from .pretrain import check_split_ids
from .schema import binary_mask
from .split import validate_split

PARTITIONS = ("train", "validation", "test")
CNT = "CNT"
CHECK_GRID = 64  # manifest 읽기용 grid 검사 해상도. 스크리닝 결과와 무관하다
SCORE_FIELDS = ("product_a", "product_b", "partition", "group_a", "group_b", "lineage_a", "lineage_b", "similarity", "usage_jaccard",
                "geometry_distance", "anchor", "shift_y_um", "shift_x_um", "alignment_score", "roi_overlap", "alignment_flag", "tier", "reason")


def block_reduce(mask, factor, how):
    h, w = mask.shape
    mask = np.pad(mask, ((0, -h % factor), (0, -w % factor)))
    blocks = mask.reshape(mask.shape[0] // factor, factor, mask.shape[1] // factor, factor)
    return blocks.any(axis=(1, 3)) if how == "any" else blocks.mean(axis=(1, 3)) > 0.5


def dilate(mask, radius):
    if radius <= 0:
        return mask
    return F.max_pool2d(torch.from_numpy(mask.astype(np.float32))[None, None], 2 * radius + 1, 1, radius)[0, 0].numpy() > 0


def product_signature(product, factor, tolerance):
    """binding 이름 → 물리 래스터 마스크. 프레임에 맞춰 줄이지 않고 모든 제품에 같은 정수 배율을 쓴다(얇은 선이 남도록 any 축소)."""
    shape = next(e.mask.shape for e in product.elements)
    valid = np.ones(shape, dtype=bool) if product.valid is None else binary_mask(product.valid) > 0
    masks = {f"{e.role}/{e.layer}": (binary_mask(e.mask) > 0) & valid for e in product.elements}
    cnt_sources = [binary_mask(c.mask) > 0 for c in product.connections if c.mask is not None]
    if product.cnt is not None:
        cnt_sources.append(binary_mask(product.cnt) > 0)
    masks[CNT] = np.logical_or.reduce(cnt_sources) & valid if cnt_sources else np.zeros(shape, dtype=bool)
    rasters = {}
    for key, mask in masks.items():
        coarse = block_reduce(mask, factor, "any")
        rasters[key] = {"mask": coarse, "dilated": dilate(coarse, tolerance), "used": bool(coarse.any())}
    electrodes = np.logical_or.reduce([r["mask"] for k, r in rasters.items() if k != CNT])
    valid_raster = block_reduce(valid, factor, "mean")
    pitch = product.frame.pixel_pitch_um if product.frame is not None else 1.0
    return {"rasters": rasters, "electrodes": electrodes, "valid": valid_raster, "shape": valid_raster.shape, "um_per_px": float(pitch * factor)}


def tolerant_distance(a, b):
    """경계 F-measure 기반 거리: 한쪽 형상이 다른 쪽의 허용 범위 안에 얼마나 들어가는가. 0=같음, 1=겹침 없음."""
    precision = (a["mask"] & b["dilated"]).sum() / a["mask"].sum()
    recall = (b["mask"] & a["dilated"]).sum() / b["mask"].sum()
    return 1.0 - (0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))


def roi_box(shape_a, shape_b, dy, dx):
    """B[y, x]가 A[y + dy, x + dx]에 놓일 때 두 프레임이 겹치는 영역. A 좌표의 (y0, y1, x0, x1)이고 겹침이 없으면 None."""
    y0, y1 = max(0, dy), min(shape_a[0], shape_b[0] + dy)
    x0, x1 = max(0, dx), min(shape_a[1], shape_b[1] + dx)
    return (y0, y1, x0, x1) if y1 > y0 and x1 > x0 else None


def roi_views(sa, sb, dy, dx):
    box = roi_box(sa["shape"], sb["shape"], dy, dx)
    if box is None:
        return None
    y0, y1, x0, x1 = box
    a_slice, b_slice = (slice(y0, y1), slice(x0, x1)), (slice(y0 - dy, y1 - dy), slice(x0 - dx, x1 - dx))
    return box, a_slice, b_slice, sa["valid"][a_slice] & sb["valid"][b_slice]


def anchor_layers(sa, sb, mode):
    """기준 마스크 고르기. CNT 합집합이 비었거나 프레임 대부분을 덮어(흰색 전체 마스크 등) 위치 정보가 없으면 전극 합집합을 쓴다."""
    def informative(sig, mask):
        valid = sig["valid"]
        return mask.any() and (mask & valid).sum() <= 0.9 * max(valid.sum(), 1)
    kinds = [CNT, "ELECTRODES"] if mode == "auto" else [mode]
    for kind in kinds:
        get = (lambda s: s["rasters"][CNT]["mask"]) if kind == CNT else (lambda s: s["electrodes"])
        if informative(sa, get(sa)) and informative(sb, get(sb)):
            return kind, get(sa), get(sb)
    return None, None, None


def candidate_shifts(a, b, center, limit, tolerance, count, suppress):
    """교차상관 상위 후보(이동량 dy, dx와 상관값). center는 두 프레임 중심을 맞췄을 때의 이동, limit는 그 주변 탐색 반경(픽셀)."""
    (ha, wa), (hb, wb) = a.shape, b.shape
    size = (ha + hb, wa + wb)
    fa, fb = dilate(a, tolerance).astype(np.float32), dilate(b, tolerance).astype(np.float32)
    corr = np.fft.irfft2(np.fft.rfft2(fa, size) * np.conj(np.fft.rfft2(fb, size)), size)
    dys, dxs = np.arange(-(hb - 1), ha), np.arange(-(wb - 1), wa)
    window = corr[np.ix_(dys % size[0], dxs % size[1])].copy()
    window[np.abs(dys - center[0]) > limit[0], :] = -np.inf
    window[:, np.abs(dxs - center[1]) > limit[1]] = -np.inf
    found = []
    for _ in range(count):
        iy, ix = np.unravel_index(np.argmax(window), window.shape)
        value = window[iy, ix]
        if not np.isfinite(value) or value <= 0:
            break
        found.append((int(dys[iy]), int(dxs[ix]), float(value)))
        window[max(0, iy - suppress):iy + suppress + 1, max(0, ix - suppress):ix + suppress + 1] = -np.inf
    cy, cx = (np.argwhere(a).mean(0) - np.argwhere(b).mean(0))  # 기준 마스크 무게중심을 맞추는 후보도 하나 더 본다
    near = lambda c: any(abs(c[0] - f[0]) <= suppress and abs(c[1] - f[1]) <= suppress for f in found)  # 이미 있는 후보와 겹치면 추가하지 않는다
    if not found or not near((round(cy), round(cx))):
        found.append((int(round(cy)), int(round(cx)), 0.0))
    return found


def evaluate_shift(sa, sb, dy, dx, anchor_pair=None):
    views = roi_views(sa, sb, dy, dx)
    if views is None:
        return None
    box, a_slice, b_slice, valid = views
    parts = []
    for key in sorted(sa["rasters"].keys() | sb["rasters"].keys()):
        ra, rb = sa["rasters"].get(key), sb["rasters"].get(key)
        a = np.zeros(valid.shape, bool) if ra is None else ra["mask"][a_slice] & valid
        b = np.zeros(valid.shape, bool) if rb is None else rb["mask"][b_slice] & valid
        if not a.any() and not b.any():
            continue
        if a.any() and b.any():
            da, db = (ra["dilated"][a_slice] & valid), (rb["dilated"][b_slice] & valid)
            parts.append((key, "both", float(tolerant_distance({"mask": a, "dilated": da}, {"mask": b, "dilated": db}))))
        else:
            parts.append((key, "only_a" if a.any() else "only_b", 1.0))
    both = [d for _, status, d in parts if status == "both"]
    anchor_score = None
    if anchor_pair is not None:
        am, bm = anchor_pair[0][a_slice] & valid, anchor_pair[1][b_slice] & valid
        anchor_score = None if not (am.any() and bm.any()) else 1.0 - float(tolerant_distance(
            {"mask": am, "dilated": dilate(am, 1)}, {"mask": bm, "dilated": dilate(bm, 1)}))
    return {"similarity": 1.0 - (float(np.mean([d for _, _, d in parts])) if parts else 0.0),
            "usage_jaccard": sum(s == "both" for _, s, _ in parts) / len(parts) if parts else 1.0,
            "geometry_distance": float(np.mean(both)) if both else None, "bindings": parts,
            "shift": (dy, dx), "roi": box, "anchor_score": anchor_score,
            "roi_overlap": (box[1] - box[0]) * (box[3] - box[2]) / min(np.prod(sa["shape"]), np.prod(sb["shape"]))}


def compare(sa, sb, *, anchor="auto", max_shift=0.35, candidates=5, tolerance=1):
    """제품 쌍을 정렬하고 비교한다. 후보 이동마다 전체 유사도를 구해 가장 높은 것을 쓴다."""
    center = ((sa["shape"][0] - sb["shape"][0]) // 2, (sa["shape"][1] - sb["shape"][1]) // 2)
    kind, a_anchor, b_anchor = anchor_layers(sa, sb, anchor)
    flags = []
    if kind is None:
        options, kind = [(center[0], center[1], 0.0)], "none"
        flags.append("no_anchor")
    else:
        limit = tuple(math.ceil(max_shift * max(sa["shape"][i], sb["shape"][i])) for i in range(2))
        suppress = max(3, round(0.03 * max(sa["shape"] + sb["shape"])))
        options = candidate_shifts(a_anchor, b_anchor, center, limit, tolerance, candidates, suppress)
        if kind != CNT:
            flags.append("anchor_fallback_electrodes")
    scored = [(evaluate_shift(sa, sb, dy, dx, None if kind == "none" else (a_anchor, b_anchor)), corr) for dy, dx, corr in options]
    scored = [(r, c) for r, c in scored if r is not None]
    if not scored:
        raise ValueError("the two frames do not overlap for any candidate shift")
    scored.sort(key=lambda rc: (-rc[0]["similarity"], -rc[1]))
    best = scored[0][0]
    if len(scored) > 1 and best["similarity"] - scored[1][0]["similarity"] < 0.02:
        flags.append("ambiguous_alignment")
    if best["anchor_score"] is not None and best["anchor_score"] < 0.5:
        flags.append("alignment_unreliable")
    if best["roi_overlap"] < 0.5:
        flags.append("low_overlap")
    return {**best, "anchor": kind, "flags": flags, "um_per_px": sa["um_per_px"]}


def auc(higher, lower):
    """무작위로 뽑은 higher 쪽 값이 lower 쪽 값보다 클 확률(동점은 절반). 1에 가까울수록 두 집단이 잘 갈린다."""
    if not higher or not lower:
        return None
    h, l = np.asarray(higher)[:, None], np.asarray(lower)[None, :]
    return float(((h > l).sum() + 0.5 * (h == l).sum()) / (h.size * l.size))


def load_groups(path, product_ids, needed):
    groups = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(groups, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not v.strip() for k, v in groups.items()):
        raise ValueError("groups.json must map product_id to a group name")
    unknown = sorted(set(groups) - set(product_ids))
    missing = sorted(set(needed) - set(groups))
    if unknown or missing:
        raise ValueError(f"groups.json does not match the manifest; unknown IDs: {unknown[:10]}; missing for screened products: {missing[:10]}")
    return groups


def screen(manifest, split, *, groups=None, partitions=("train",), factor=8, tolerance=1, margin=0.05, floor=None, cross_group_fraction=0.2,
           min_group_pairs=3, anchor="auto", max_shift=0.35, candidates=5, review_band=0.15):
    if type(factor) is not int or factor <= 0 or type(tolerance) is not int or tolerance < 0 or not 0 <= margin < 1 or not 0 <= cross_group_fraction <= 1:
        raise ValueError("factor > 0, tolerance >= 0, margin in [0,1) and cross_group_fraction in [0,1] required")
    if anchor not in {"auto", CNT, "ELECTRODES"} or not 0 < max_shift <= 1 or type(candidates) is not int or candidates < 1:
        raise ValueError("anchor must be auto/CNT/ELECTRODES, max_shift in (0,1] and candidates >= 1")
    if not partitions or any(p not in PARTITIONS for p in partitions):
        raise ValueError(f"partitions must be chosen from {PARTITIONS}")
    document = json.loads(Path(manifest).read_text(encoding="utf-8"))
    ids = [p["product_id"] for p in document["products"]]
    check_split_ids(split, ids)
    wanted = {pid for part in partitions for pid in split[part]}
    group_of = {pid: "-" for pid in wanted} if groups is None else load_groups(groups, ids, wanted)
    stubs, signatures, lineage = [], {}, {}
    for product in _iter_manifest_products(manifest, grid=CHECK_GRID):  # 끝까지 읽어 manifest 전체를 검증한다
        stubs.append(SimpleNamespace(product_id=product.product_id, lineage_id=product.lineage_id))
        lineage[product.product_id] = product.lineage_id
        if product.product_id in wanted:
            signatures[product.product_id] = product_signature(product, factor, tolerance)
    validate_split(stubs, split)
    pairs = []
    for part in partitions:
        names = sorted(split[part])
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                result = compare(signatures[a], signatures[b], anchor=anchor, max_shift=max_shift, candidates=candidates, tolerance=tolerance)
                pairs.append({"product_a": a, "product_b": b, "partition": part, "group_a": group_of[a], "group_b": group_of[b],
                              "lineage_a": lineage[a], "lineage_b": lineage[b], "same_lineage": lineage[a] == lineage[b], "result": result,
                              "similarity": round(result["similarity"], 6), "usage_jaccard": round(result["usage_jaccard"], 6),
                              "geometry_distance": None if result["geometry_distance"] is None else round(result["geometry_distance"], 6),
                              "anchor": result["anchor"], "shift_y_um": round(result["shift"][0] * result["um_per_px"], 3), "shift_x_um": round(result["shift"][1] * result["um_per_px"], 3),
                              "alignment_score": None if result["anchor_score"] is None else round(result["anchor_score"], 4), "roi_overlap": round(float(result["roi_overlap"]), 4),
                              "alignment_flag": ";".join(result["flags"])})
    same = [p for p in pairs if p["same_lineage"]]
    global_floor = floor if floor is not None else (min(p["similarity"] for p in same) if same else None)
    if global_floor is None:
        raise ValueError("no same-lineage pair to calibrate with; pass an explicit --floor")
    group_floor = {}
    for g in sorted(set(group_of.values())):
        within = [p["similarity"] for p in same if p["group_a"] == g and p["group_b"] == g]
        if floor is None and len(within) >= min_group_pairs:
            group_floor[g] = min(within)
    for p in pairs:
        p["floor"] = group_floor.get(p["group_a"], global_floor) if p["group_a"] == p["group_b"] else global_floor
        if p["same_lineage"]:
            p["tier"], p["reason"] = "similar", "same_lineage"
        elif p["similarity"] < p["floor"] - margin:
            p["tier"], p["reason"] = "different", "below_same_lineage_floor"
        elif p["similarity"] >= p["floor"]:
            p["tier"], p["reason"] = "unknown", "close_to_same_lineage"  # 이름만 다른 같은 회로일 수 있다
        else:
            p["tier"], p["reason"] = "unknown", "near_boundary"
    within_different = sum(p["tier"] == "different" and p["group_a"] == p["group_b"] for p in pairs)
    allowed = math.ceil(cross_group_fraction * within_different)
    cross = sorted((p for p in pairs if p["tier"] == "different" and p["group_a"] != p["group_b"]), key=lambda p: (-p["similarity"], p["product_a"], p["product_b"]))
    for p in cross[allowed:]:
        p["tier"], p["reason"] = "unknown", "cross_group_capped"  # 쉬운 쌍이라 일부만 쓴다
    weakest = sorted(same, key=lambda p: p["similarity"])[:max(1, math.ceil(0.2 * len(same)))] if same else []
    for p in weakest:
        p["flag"] = "weakest_same_lineage"
    for p in pairs:  # 사람이 봐야 하는 쌍과 그 이유를 모두 모은다
        p["review"] = []
        if p["tier"] == "unknown" and p["reason"] in {"close_to_same_lineage", "near_boundary"}:
            p["review"].append(p["reason"])
        if p.get("flag") == "weakest_same_lineage":
            p["review"].append("weakest_same_lineage")
        if p["tier"] == "different" and p["similarity"] >= p["floor"] - margin - review_band:
            p["review"].append("different_near_boundary")
        p["review"] += [f for f in p["alignment_flag"].split(";") if f in {"ambiguous_alignment", "alignment_unreliable", "low_overlap", "no_anchor"}]
    return {"pairs": pairs, "signatures": signatures, "floor": {"global": global_floor, "groups": group_floor, "margin": margin},
            "summary": summarize(pairs, split, partitions, group_of, global_floor, group_floor, margin)}


def summarize(pairs, split, partitions, group_of, global_floor, group_floor, margin):
    cross = [p for p in pairs if not p["same_lineage"]]
    same = [p for p in pairs if p["same_lineage"]]

    def separation(key):
        return auc([key(p) for p in same], [key(p) for p in cross])
    tiers = {t: sum(p["tier"] == t for p in pairs) for t in ("similar", "different", "unknown")}
    uncovered = {}
    for part in partitions:
        have = {pid for p in pairs if p["partition"] == part and p["tier"] == "different" for pid in (p["product_a"], p["product_b"])}
        uncovered[part] = sorted(set(split[part]) - have)
    return {"pairs_by_tier": tiers, "unknown_by_reason": {r: sum(p["reason"] == r and p["tier"] == "unknown" for p in pairs) for r in sorted({p["reason"] for p in pairs if p["tier"] == "unknown"})},
            "floor": {"global": global_floor, "per_group": group_floor, "margin": margin},
            "separation_auc_same_vs_other_lineage": {"similarity": separation(lambda p: p["similarity"]), "usage_jaccard": separation(lambda p: p["usage_jaccard"]),
                                                      "geometry": separation(lambda p: 0.0 if p["geometry_distance"] is None else 1 - p["geometry_distance"])},
            "alignment": {"anchors": {k: sum(p["anchor"] == k for p in pairs) for k in sorted({p["anchor"] for p in pairs})},
                          "flags": {f: sum(f in p["alignment_flag"].split(";") for p in pairs) for f in ("no_anchor", "anchor_fallback_electrodes", "ambiguous_alignment", "alignment_unreliable", "low_overlap")},
                          "median_roi_overlap": float(np.median([p["roi_overlap"] for p in pairs])) if pairs else None},
            "review": {"pairs_to_review": sum(bool(p["review"]) for p in pairs),
                       "by_reason": {r: sum(r in p["review"] for p in pairs) for r in sorted({r for p in pairs for r in p["review"]})}},
            "products_without_different_pair": uncovered, "groups": {g: sum(v == g for v in group_of.values()) for g in sorted(set(group_of.values()))}}


# ---- 출력 -------------------------------------------------------------------------------------------------------
def write_csvs(result, out, scores_path):
    for path in (out, scores_path):
        if Path(path).exists():
            raise FileExistsError(f"refusing to overwrite {path}; choose a new path")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pairs = sorted(result["pairs"], key=lambda p: (p["partition"], p["product_a"], p["product_b"]))
    with open(out, "w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows([p["product_a"], p["product_b"], p["tier"]] for p in pairs if p["tier"] in {"similar", "different"})
    with open(scores_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, SCORE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(pairs, key=lambda p: (-p["similarity"], p["product_a"], p["product_b"])))
    similar = {frozenset((p["product_a"], p["product_b"])) for p in pairs if p["tier"] == "similar"}
    different = {frozenset((p["product_a"], p["product_b"])) for p in pairs if p["tier"] == "different"}
    validate_relations(similar, different)


def overlay_png(a, b, size=168):
    """겹친 영역(ROI)을 비율을 유지한 채 size 안에 그린다. 남는 부분은 투명하다."""
    from PIL import Image
    rgba = np.zeros(a.shape + (4,), dtype=np.uint8)
    rgba[a & ~b] = (57, 135, 229, 255)    # A만: 파랑
    rgba[b & ~a] = (235, 104, 52, 255)    # B만: 주황
    rgba[a & b] = (122, 122, 118, 255)    # 둘 다: 회색
    scale = size / max(a.shape)
    image = Image.fromarray(rgba, "RGBA").resize((max(1, round(a.shape[1] * scale)), max(1, round(a.shape[0] * scale))), Image.NEAREST)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(image, (0, 0))
    buffer = io.BytesIO()
    canvas.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


REVIEW_CSS = """
:root{color-scheme:light dark;--surface:#fcfcfb;--text:#0b0b0b;--text2:#52514e;--grid:#e3e2de;--a:#3987e5;--b:#eb6834;--both:#7a7a76}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--text:#f0efec;--text2:#c3c2b7;--grid:#2e2e2c}}
body{margin:0;background:var(--surface);color:var(--text);font:14px/1.5 system-ui,'Segoe UI','Malgun Gothic',sans-serif;padding:24px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:0}p.sub{color:var(--text2);max-width:900px;margin:0 0 12px}
.legend{display:flex;gap:18px;color:var(--text2);margin:8px 0 18px}.legend i{display:inline-block;width:12px;height:12px;margin-right:6px;border-radius:2px;vertical-align:-1px}
.card{border:1px solid var(--grid);border-radius:8px;padding:14px;margin:0 0 16px}.card header{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap;margin-bottom:8px}
.badge{color:var(--text2);font-size:12px;border:1px solid var(--grid);border-radius:10px;padding:1px 8px}
.tiles{display:flex;gap:10px;flex-wrap:wrap}.tile{width:168px}.tile img{display:block;width:168px;height:168px;background:var(--surface);border:1px solid var(--grid)}
.tile span{display:block;font-size:11px;color:var(--text2);word-break:break-all}table{border-collapse:collapse;font-size:12px;margin-bottom:12px}
td,th{padding:3px 10px;border-bottom:1px solid var(--grid);text-align:left}th{color:var(--text2);font-weight:500}td.n{text-align:right;font-variant-numeric:tabular-nums}
"""
REASON_TEXT = {"close_to_same_lineage": "다른 계열인데 같은 계열 쌍만큼 비슷함: 같은 회로일 수 있음", "near_boundary": "경계 부근: 판단 필요",
               "weakest_same_lineage": "같은 계열 쌍 중 유사도가 가장 낮은 쌍: 정말 비슷한지 확인", "different_near_boundary": "자동으로 different가 됐지만 기준에 가까움: 정말 다른지 확인",
               "ambiguous_alignment": "정렬 후보가 여러 개(반복 패턴 등): 유사도를 덜 믿을 것", "alignment_unreliable": "기준 마스크가 잘 맞지 않음: 유사도를 덜 믿을 것",
               "low_overlap": "두 프레임이 겹치는 영역이 작음: 유사도를 덜 믿을 것", "no_anchor": "정렬 기준이 없어 프레임 중심으로 맞춤"}
REASON_ORDER = list(REASON_TEXT)
TIER_TEXT = {"similar": "similar", "different": "different", "unknown": "모름"}


def render_review(result, top_pairs=60, top_bindings=6):
    """검토 대상 쌍(p["review"]가 비어 있지 않은 모든 쌍)을 위쪽 표에 전부 나열하고, 경계에 가까운 순으로 top_pairs쌍은 층별 겹침 이미지 카드로 보여 준다."""
    pairs = result["pairs"]
    chosen = sorted((p for p in pairs if p["review"]), key=lambda p: (abs(p["similarity"] - p["floor"]), p["product_a"], p["product_b"]))
    summary = result["summary"]
    counts = "".join(f'<li>{html.escape(r)}: {n}쌍 — {html.escape(REASON_TEXT[r])}</li>' for r, n in summary["review"]["by_reason"].items())
    table = "".join(f'<tr><td>{i}</td><td>{html.escape(p["product_a"])} ↔ {html.escape(p["product_b"])}</td><td>{TIER_TEXT[p["tier"]]}</td><td class="n">{p["similarity"]:.3f}</td>'
                    f'<td class="n">{p["floor"]:.3f}</td><td>{html.escape(", ".join(p["review"]))}</td><td>{"아래 카드" if i <= top_pairs else "scores.csv"}</td></tr>' for i, p in enumerate(chosen, 1))
    cards = []
    for p in chosen[:top_pairs]:
        sa, sb = result["signatures"][p["product_a"]], result["signatures"][p["product_b"]]
        dy, dx = p["result"]["shift"]
        _, a_slice, b_slice, valid = roi_views(sa, sb, dy, dx)
        parts = sorted(p["result"]["bindings"], key=lambda item: (-item[2], item[0]))[:top_bindings]

        def roi_mask(sig, key, view):
            return np.zeros(valid.shape, bool) if key not in sig["rasters"] else sig["rasters"][key]["mask"][view] & valid
        rows = "".join(f'<tr><td>{html.escape(k)}</td><td>{ {"both": "양쪽 사용", "only_a": "A만 사용", "only_b": "B만 사용"}[s] }</td><td class="n">{d:.3f}</td></tr>' for k, s, d in parts)
        tiles = "".join(f'<div class="tile"><img alt="{html.escape(k)}" src="{overlay_png(roi_mask(sa, k, a_slice), roi_mask(sb, k, b_slice))}"><span>{html.escape(k)} · 거리 {d:.2f}</span></div>' for k, s, d in parts)
        reasons = "".join(f'<li>{html.escape(REASON_TEXT[r])}</li>' for r in p["review"])
        cards.append(f'<section class="card"><header><h2>A {html.escape(p["product_a"])} ↔ B {html.escape(p["product_b"])}</h2><span class="badge">유사도 {p["similarity"]:.3f}</span>'
                     f'<span class="badge">기준 {p["floor"]:.3f}</span><span class="badge">전극 사용 일치 {p["usage_jaccard"]:.2f}</span><span class="badge">{html.escape(p["group_a"])}/{html.escape(p["group_b"])}</span>'
                     f'<span class="badge">정렬 기준 {html.escape(p["anchor"])} · 이동 ({p["shift_y_um"]:.1f}, {p["shift_x_um"]:.1f})µm · 겹침 {p["roi_overlap"]:.0%}</span>'
                     f'{"".join(f"<span class=badge>{html.escape(f)}</span>" for f in p["alignment_flag"].split(";") if f)}'
                     f'<span class="badge">현재: {TIER_TEXT[p["tier"]]}</span></header><ul class="sub">{reasons}</ul>'
                     f'<table><tr><th>binding (role/layer)</th><th>사용</th><th>거리</th></tr>{rows}</table><div class="tiles">{tiles}</div></section>')
    omitted = len(chosen) - min(len(chosen), top_pairs)
    return (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>DL1 스크리닝 검토</title><style>{REVIEW_CSS}</style></head><body>'
            f'<h1>DL1 스크리닝 검토</h1><p class="sub">검토 대상 <b>{len(chosen)}쌍</b>입니다(경계에 가까운 순). 유사도 = 1 − binding별 거리의 평균이며, 한쪽에서만 쓰이는 전극은 거리 1입니다. '
            f'이미지는 B를 A에 정렬한 뒤 두 제품이 겹치는 영역(ROI)만 그린 것입니다. 확정할 쌍은 relations.csv에 similar/different로 직접 추가하고, 틀린 자동 줄은 지우세요.</p>'
            f'<ul class="sub">{counts or "<li>검토 대상이 없습니다.</li>"}</ul>'
            f'<div class="legend"><span><i style="background:var(--a)"></i>A만</span><span><i style="background:var(--b)"></i>B만</span><span><i style="background:var(--both)"></i>둘 다</span>'
            f'<span>기준(같은 계열 최저 유사도) {summary["floor"]["global"]:.3f}</span></div>'
            f'<h2>검토 대상 전체</h2><table><tr><th>#</th><th>쌍</th><th>현재 판정</th><th>유사도</th><th>기준</th><th>이유</th><th>이미지</th></tr>{table}</table>'
            + (f'<p class="sub">카드는 {top_pairs}쌍까지만 그렸습니다. 나머지 {omitted}쌍은 위 표와 scores.csv로 보세요(--review-pairs로 늘릴 수 있음).</p>' if omitted else "")
            + f'<h2>카드</h2>{"".join(cards) or "<p>검토할 쌍이 없습니다.</p>"}</body></html>')


def main(argv=None):
    ap = argparse.ArgumentParser(description="First-pass screening before training: per-electrode usage and shape similarity -> relations.csv draft, scores and a review page")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True, help="new relations.csv (similar/different only, header-free)")
    ap.add_argument("--groups", help="JSON {product_id: group}, e.g. mobile/it/auto")
    ap.add_argument("--partitions", nargs="+", default=["train"], choices=PARTITIONS, help="pairs are made inside each partition only")
    ap.add_argument("--factor", type=int, default=8, help="integer downsampling of every product (raster pitch = pixel pitch x factor); the same for all products")
    ap.add_argument("--tolerance", type=int, default=1, help="raster pixels of residual misalignment tolerated after alignment")
    ap.add_argument("--anchor", default="auto", choices=("auto", "CNT", "ELECTRODES"), help="mask used to align a pair; auto = CNT union, else the electrode union")
    ap.add_argument("--max-shift", type=float, default=0.35, help="search radius around the centred position, as a fraction of the larger frame side")
    ap.add_argument("--candidates", type=int, default=5, help="alignment candidates (correlation peaks) verified with the full similarity")
    ap.add_argument("--margin", type=float, default=0.05, help="a cross-lineage pair is 'different' only this far below the weakest same-lineage similarity")
    ap.add_argument("--floor", type=float, help="explicit similarity floor instead of the weakest same-lineage pair")
    ap.add_argument("--cross-group-fraction", type=float, default=0.2, help="cross-group 'different' pairs kept, as a fraction of within-group ones")
    ap.add_argument("--review-pairs", type=int, default=60, help="pairs drawn as image cards; the table always lists every pair to review")
    ap.add_argument("--review-band", type=float, default=0.15, help="auto-different pairs within margin + band of the floor are listed for review")
    ap.add_argument("--review-bindings", type=int, default=6)
    a = ap.parse_args(argv)
    split = json.loads(Path(a.split).read_text(encoding="utf-8"))
    result = screen(a.manifest, split, groups=a.groups, partitions=tuple(a.partitions), factor=a.factor, tolerance=a.tolerance, margin=a.margin,
                    floor=a.floor, cross_group_fraction=a.cross_group_fraction, anchor=a.anchor, max_shift=a.max_shift, candidates=a.candidates, review_band=a.review_band)
    out = Path(a.out)
    scores, review, summary = (out.with_name(out.name + suffix) for suffix in (".scores.csv", ".review.html", ".summary.json"))
    for path in (review, summary):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}; choose a new path")
    write_csvs(result, out, scores)
    review.write_text(render_review(result, a.review_pairs, a.review_bindings), encoding="utf-8")
    summary.write_text(json.dumps(result["summary"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"relations": str(out), "scores": str(scores), "review": str(review), "summary_file": str(summary), **result["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
