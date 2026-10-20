"""학습 전 1차 스크리닝: 전극(층)별 사용 패턴과 형상으로 제품 쌍의 유사도를 수치로 매기고 relations.csv 초안을 만든다.

비교 단위는 모델이 보는 것과 같은 (role, layer) 요소 이미지와 CNT 합집합이다. 쌍마다
  - 한쪽에서만 쓰이는 전극(마스크가 비어 있지 않은 binding)은 거리 1,
  - 양쪽에서 쓰이면 이동을 허용한 형상 거리(1 - tolerant F1)
를 구하고 binding 평균을 1에서 뺀 값을 유사도(0~1)로 쓴다. 같은 계열(lineage)의 쌍은 similar, 같은 계열 쌍 중 가장 낮은
유사도보다 확실히 낮은 다른 계열 쌍만 different, 나머지는 모름으로 두고 검토 파일에 남긴다. 정답이 아니라 판단을 돕는 지표다."""
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
from .pretrain import check_split_ids, element_images
from .split import validate_split

PARTITIONS = ("train", "validation", "test")
CNT = "CNT"
SCORE_FIELDS = ("product_a", "product_b", "partition", "group_a", "group_b", "lineage_a", "lineage_b", "similarity",
                "usage_jaccard", "geometry_distance", "tier", "reason")


def product_signature(product, grid, tolerance):
    """binding 이름 → 마스크, 이동 허용 범위로 부풀린 마스크. 모델 입력과 같은 정규화(element_images)를 쓴다."""
    images, _ = element_images(product, grid)
    masks = {f"{e.role}/{e.layer}": images[i, 0] > 0 for i, e in enumerate(product.elements)}
    masks[CNT] = images[0, 1] > 0
    signature = {}
    for key, mask in masks.items():
        if tolerance:
            dilated = F.max_pool2d(torch.from_numpy(mask.astype(np.float32))[None, None], 2 * tolerance + 1, 1, tolerance)[0, 0].numpy() > 0
        else:
            dilated = mask
        signature[key] = {"mask": mask, "dilated": dilated, "used": bool(mask.any())}
    return signature


def tolerant_distance(a, b):
    """경계 F-measure 기반 거리: 한쪽 형상이 다른 쪽의 허용 범위 안에 얼마나 들어가는가. 0=같음, 1=겹침 없음."""
    precision = (a["mask"] & b["dilated"]).sum() / a["mask"].sum()
    recall = (b["mask"] & a["dilated"]).sum() / b["mask"].sum()
    return 1.0 - (0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))


def compare(sa, sb):
    used_a = {k for k, v in sa.items() if v["used"]}
    used_b = {k for k, v in sb.items() if v["used"]}
    parts = []
    for key in sorted(used_a | used_b):
        if key in used_a and key in used_b:
            parts.append((key, "both", float(tolerant_distance(sa[key], sb[key]))))
        else:
            parts.append((key, "only_a" if key in used_a else "only_b", 1.0))
    both = [d for _, status, d in parts if status == "both"]
    return {"similarity": 1.0 - (float(np.mean([d for _, _, d in parts])) if parts else 0.0),
            "usage_jaccard": len(used_a & used_b) / len(used_a | used_b) if parts else 1.0,
            "geometry_distance": float(np.mean(both)) if both else None, "bindings": parts}


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


def screen(manifest, split, *, groups=None, partitions=("train",), grid=128, tolerance=2, margin=0.05, floor=None,
           cross_group_fraction=0.2, min_group_pairs=3):
    if type(grid) is not int or grid <= 0 or type(tolerance) is not int or tolerance < 0 or not 0 <= margin < 1 or not 0 <= cross_group_fraction <= 1:
        raise ValueError("grid > 0, tolerance >= 0, margin in [0,1) and cross_group_fraction in [0,1] required")
    if not partitions or any(p not in PARTITIONS for p in partitions):
        raise ValueError(f"partitions must be chosen from {PARTITIONS}")
    document = json.loads(Path(manifest).read_text(encoding="utf-8"))
    ids = [p["product_id"] for p in document["products"]]
    check_split_ids(split, ids)
    wanted = {pid for part in partitions for pid in split[part]}
    group_of = {pid: "-" for pid in wanted} if groups is None else load_groups(groups, ids, wanted)
    stubs, signatures, lineage = [], {}, {}
    for product in _iter_manifest_products(manifest, grid=grid):  # 끝까지 읽어 manifest 전체를 검증한다
        stubs.append(SimpleNamespace(product_id=product.product_id, lineage_id=product.lineage_id))
        lineage[product.product_id] = product.lineage_id
        if product.product_id in wanted:
            signatures[product.product_id] = product_signature(product, grid, tolerance)
    validate_split(stubs, split)
    pairs = []
    for part in partitions:
        names = sorted(split[part])
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                result = compare(signatures[a], signatures[b])
                pairs.append({"product_a": a, "product_b": b, "partition": part, "group_a": group_of[a], "group_b": group_of[b],
                              "lineage_a": lineage[a], "lineage_b": lineage[b], "same_lineage": lineage[a] == lineage[b], "result": result,
                              "similarity": round(result["similarity"], 6), "usage_jaccard": round(result["usage_jaccard"], 6),
                              "geometry_distance": None if result["geometry_distance"] is None else round(result["geometry_distance"], 6)})
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
    from PIL import Image
    rgba = np.zeros(a.shape + (4,), dtype=np.uint8)
    rgba[a & ~b] = (57, 135, 229, 255)    # A만: 파랑
    rgba[b & ~a] = (235, 104, 52, 255)    # B만: 주황
    rgba[a & b] = (122, 122, 118, 255)    # 둘 다: 회색
    buffer = io.BytesIO()
    Image.fromarray(rgba, "RGBA").resize((size, size), Image.NEAREST).save(buffer, "PNG")
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
REASON_TEXT = {"close_to_same_lineage": "다른 계열인데 같은 계열 쌍만큼 비슷함: 같은 회로일 수 있음", "near_boundary": "경계 부근: 판단 필요", "same_lineage": "같은 계열 쌍 중 유사도가 가장 낮은 쌍: 정말 비슷한지 확인",
               "cross_group_capped": "제품군이 다른 쉬운 쌍(일부만 사용)", "below_same_lineage_floor": "확실히 다름"}


def render_review(result, top_pairs=40, top_bindings=6):
    pairs = result["pairs"]
    chosen = [p for p in pairs if p["tier"] == "unknown" and p["reason"] in {"close_to_same_lineage", "near_boundary"} or p.get("flag") == "weakest_same_lineage"]
    chosen.sort(key=lambda p: (abs(p["similarity"] - p["floor"]), p["product_a"], p["product_b"]))
    chosen = chosen[:top_pairs]
    cards = []
    for p in chosen:
        a, b = result["signatures"][p["product_a"]], result["signatures"][p["product_b"]]
        parts = sorted(p["result"]["bindings"], key=lambda item: (-item[2], item[0]))[:top_bindings]
        rows = "".join(f'<tr><td>{html.escape(k)}</td><td>{ {"both": "양쪽 사용", "only_a": "A만 사용", "only_b": "B만 사용"}[s] }</td><td class="n">{d:.3f}</td></tr>' for k, s, d in parts)
        tiles = "".join(f'<div class="tile"><img alt="{html.escape(k)}" src="{overlay_png(a[k]["mask"] if k in a else np.zeros_like(b[k]["mask"]), b[k]["mask"] if k in b else np.zeros_like(a[k]["mask"]))}"><span>{html.escape(k)} · 거리 {d:.2f}</span></div>' for k, s, d in parts)
        reason = REASON_TEXT.get("same_lineage" if p.get("flag") else p["reason"], p["reason"])
        cards.append(f'<section class="card"><header><h2>A {html.escape(p["product_a"])} ↔ B {html.escape(p["product_b"])}</h2><span class="badge">유사도 {p["similarity"]:.3f}</span>'
                     f'<span class="badge">기준 {p["floor"]:.3f}</span><span class="badge">전극 사용 일치 {p["usage_jaccard"]:.2f}</span><span class="badge">{html.escape(p["group_a"])}/{html.escape(p["group_b"])}</span>'
                     f'<span class="badge">현재: {p["tier"]}</span></header><p class="sub">{html.escape(reason)}</p>'
                     f'<table><tr><th>binding (role/layer)</th><th>사용</th><th>거리</th></tr>{rows}</table><div class="tiles">{tiles}</div></section>')
    summary = result["summary"]
    return (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>DL1 스크리닝 검토</title><style>{REVIEW_CSS}</style></head><body>'
            f'<h1>DL1 스크리닝 검토</h1><p class="sub">판단이 필요한 쌍입니다(경계에 가까운 순). 유사도 = 1 − binding별 거리의 평균이며, 한쪽에서만 쓰이는 전극은 거리 1입니다. '
            f'확정할 쌍은 relations.csv에 similar/different로 직접 추가하세요. 아래 {len(chosen)}쌍 외에는 scores.csv에 있습니다.</p>'
            f'<div class="legend"><span><i style="background:var(--a)"></i>A만</span><span><i style="background:var(--b)"></i>B만</span><span><i style="background:var(--both)"></i>둘 다</span>'
            f'<span>기준(같은 계열 최저 유사도) {summary["floor"]["global"]:.3f}</span></div>{"".join(cards) or "<p>검토할 쌍이 없습니다.</p>"}</body></html>')


def main(argv=None):
    ap = argparse.ArgumentParser(description="First-pass screening before training: per-electrode usage and shape similarity -> relations.csv draft, scores and a review page")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True, help="new relations.csv (similar/different only, header-free)")
    ap.add_argument("--groups", help="JSON {product_id: group}, e.g. mobile/it/auto")
    ap.add_argument("--partitions", nargs="+", default=["train"], choices=PARTITIONS, help="pairs are made inside each partition only")
    ap.add_argument("--grid", type=int, default=128, help="working resolution of the normalised element images")
    ap.add_argument("--tolerance", type=int, default=2, help="pixels of shift tolerated at --grid")
    ap.add_argument("--margin", type=float, default=0.05, help="a cross-lineage pair is 'different' only this far below the weakest same-lineage similarity")
    ap.add_argument("--floor", type=float, help="explicit similarity floor instead of the weakest same-lineage pair")
    ap.add_argument("--cross-group-fraction", type=float, default=0.2, help="cross-group 'different' pairs kept, as a fraction of within-group ones")
    ap.add_argument("--review-pairs", type=int, default=40)
    ap.add_argument("--review-bindings", type=int, default=6)
    a = ap.parse_args(argv)
    split = json.loads(Path(a.split).read_text(encoding="utf-8"))
    result = screen(a.manifest, split, groups=a.groups, partitions=tuple(a.partitions), grid=a.grid, tolerance=a.tolerance, margin=a.margin,
                    floor=a.floor, cross_group_fraction=a.cross_group_fraction)
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
