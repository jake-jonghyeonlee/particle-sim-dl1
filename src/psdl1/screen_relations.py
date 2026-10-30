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
import csv
import hashlib
import io
import json
import math
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from . import screen_align
from .data import _iter_manifest_products
from .import_png import SCREENING_LAYERS_DIR, SCREENING_LAYERS_FILE, SCREENING_LAYERS_SCHEMA_VERSION, _read_json, bundle_path
from .losses import validate_relations
from .png_contract import mask_hash
from .pretrain import check_split_ids
from .schema import binary_mask
from .split import validate_split

PARTITIONS = ("train", "validation", "test")
CNT = "CNT"
CHECK_GRID = 64  # manifest 읽기용 grid 검사 해상도. 스크리닝 결과와 무관하다
SCORE_FIELDS = ("product_a", "product_b", "partition", "group_a", "group_b", "lineage_a", "lineage_b", "similarity", "usage_jaccard",
                "geometry_distance", "anchor", "shift_y_um", "shift_x_um", "alignment_score", "roi_overlap", "alignment_flag", "tier", "reason",
                "metrics_anchor", "similarity_layer", "layer_shift_y_um", "layer_shift_x_um", "layer_align_dice", "layer_align_second_dice",
                "layer_align_decided", "layer_align_disagreeing", "layer_align_flag",
                "usage_jaccard_electrodes", "electrode_similarity", "pattern_similarity", "cnt_similarity", "layer_align_decided_dice")


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
    # 판단 지표용 원본 해상도 자료(면적은 원본 px 로 센다). 마스크는 비트로 묶어 보관한다.
    all_valid = bool(valid.all())
    exact = all_valid and shape[0] % factor == 0 and shape[1] % factor == 0  # 개수 래스터(block-sum)가 원본 직접 합과 같은 경우
    return {"rasters": rasters, "electrodes": electrodes, "valid": valid_raster, "shape": valid_raster.shape, "um_per_px": float(pitch * factor),
            "factor": factor, "full_shape": tuple(shape), "full": {k: screen_align.pack_mask(m) for k, m in masks.items()},
            "full_valid": None if all_valid else screen_align.pack_mask(valid), "roles": {f"{e.role}/{e.layer}": e.role for e in product.elements},
            "counts": {k: screen_align.block_sum(m, factor) for k, m in masks.items()} if exact else {}}


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


def load_screening_layers(manifest, needed, factor, originals=None):
    """manifest 폴더의 screening-layers.json 에서 `needed` 제품의 Layer 를 읽어 {product_id: {casefold 층 이름: 래스터(bool)}} 로 돌려준다.

    사이드카 파일이 없으면 {} (모든 제품에 Layer 없음). 파일이 있는데 손상됐으면 ValueError 다: 스키마, 경로 containment, manifest 의 product_id·frame 연결,
    casefold 중복 층 이름, npy 의 dtype bool·shape==frame, mask_sha256. 마스크는 필요한 제품만 읽고 `block_reduce(any, factor)` 로 줄여 보관한다.
    `originals` dict 를 주면 원본 해상도 마스크를 비트로 묶어 {product_id: {이름: packed}} 로 채운다."""
    root = Path(manifest).parent
    if not (root / SCREENING_LAYERS_FILE).exists() and not (root / SCREENING_LAYERS_FILE).is_symlink():
        return {}
    frames = {p["product_id"]: p.get("frame") for p in json.loads(Path(manifest).read_text(encoding="utf-8"))["products"]}

    def bad(reason):
        return ValueError(f"{SCREENING_LAYERS_FILE} is corrupt: {reason}")
    try:
        document = _read_json(bundle_path(root, SCREENING_LAYERS_FILE))
    except (OSError, ValueError) as exc:
        raise bad(str(exc)) from exc
    if not isinstance(document, dict) or set(document) != {"schema_version", "products"} or not isinstance(document["products"], dict):
        raise bad("fields")
    if document["schema_version"] != SCREENING_LAYERS_SCHEMA_VERSION:
        raise bad("unsupported schema_version")
    loaded, seen_paths = {}, set()
    for product_id, entry in document["products"].items():
        frame = frames.get(product_id)
        if not isinstance(frame, dict):
            raise bad(f"product {product_id!r} is not in the manifest")
        if (not isinstance(entry, dict) or set(entry) != {"frame_id", "layers"} or entry["frame_id"] != frame.get("frame_id") or
                not isinstance(entry["layers"], list) or not entry["layers"]):
            raise bad(f"product {product_id!r} entry or frame_id")
        folded, rasters = set(), {}
        for layer in entry["layers"]:
            if not isinstance(layer, dict) or set(layer) != {"layer", "filename_number", "relative_path", "mask", "mask_sha256"}:
                raise bad(f"product {product_id!r} layer fields")
            name, path = layer["layer"], layer["mask"]
            if not isinstance(name, str) or not name or not isinstance(layer["mask_sha256"], str):
                raise bad(f"product {product_id!r} layer values")
            if name.casefold() in folded:
                raise bad(f"product {product_id!r} has duplicate layer name {name!r}")
            folded.add(name.casefold())
            if not isinstance(path, str) or not path.startswith(SCREENING_LAYERS_DIR + "/") or path in seen_paths:
                raise bad(f"product {product_id!r} layer mask path")
            seen_paths.add(path)
            if product_id not in needed:
                continue
            try:
                mask = np.load(bundle_path(root, path), allow_pickle=False)
            except (OSError, ValueError, EOFError) as exc:
                raise bad(f"{path}: {exc}") from exc
            if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or list(mask.shape) != list(frame["shape_yx"]):
                raise bad(f"{path}: dtype bool and the frame shape are required")
            if mask_hash(mask) != layer["mask_sha256"]:
                raise bad(f"{path}: mask_sha256 mismatch")
            rasters[name.casefold()] = block_reduce(mask, factor, "any")
            if originals is not None:
                originals.setdefault(product_id, {})[name.casefold()] = screen_align.pack_mask(mask)
        if product_id in needed:
            loaded[product_id] = rasters
    return loaded


def search_window(sa, sb, max_shift):
    """compare 가 쓰는 시작점(center)과 탐색 반경(limit) — 같은 식이다."""
    center = ((sa["shape"][0] - sb["shape"][0]) // 2, (sa["shape"][1] - sb["shape"][1]) // 2)
    return center, tuple(math.ceil(max_shift * max(sa["shape"][i], sb["shape"][i])) for i in range(2))


def layer_columns(sa, sb, result, layers_a, layers_b, *, max_shift, min_overlap, min_support, min_dice):
    """Layer 정렬을 따로 한 번 호출하고 scores.csv 의 새 컬럼 값을 만든다. compare 의 결과는 바꾸지 않는다."""
    center, limit = search_window(sa, sb, max_shift)
    align = screen_align.align_layers(layers_a, layers_b, sa["valid"], sb["valid"], center=center, limit=limit, min_overlap=min_overlap,
                                      min_support=min_support, min_dice=min_dice)
    shown = align["shift"] is not None
    similarity_layer = result["similarity"]
    if align["applied"]:
        moved = evaluate_shift(sa, sb, align["shift"][0], align["shift"][1])
        if moved is not None:
            similarity_layer = moved["similarity"]
    um = sa["um_per_px"]
    return align, {"metrics_anchor": "LAYER" if align["applied"] else result["anchor"], "similarity_layer": round(similarity_layer, 6),
                   "layer_shift_y_um": round(align["shift"][0] * um, 3) if shown else None, "layer_shift_x_um": round(align["shift"][1] * um, 3) if shown else None,
                   "layer_align_dice": align["dice"], "layer_align_second_dice": align["second_dice"], "layer_align_decided": align["decided"],
                   "layer_align_decided_dice": align["decided_dice"],
                   "layer_align_disagreeing": align["disagreeing"], "layer_align_flag": ";".join(align["flags"])}


BINDING_FIELDS = ("product_a", "product_b", "partition", "metrics_anchor", "kind", "key", "status", "area_a_px", "area_b_px", "area_ratio", "dice",
                  "boundary_f1", "boundary_precision", "boundary_recall", "fill_f1")
PAIR_METRICS = ("usage_jaccard_electrodes", "electrode_similarity", "pattern_similarity", "cnt_similarity")
PRESENCE_FRACTION = 0.01


def signature_item(sig, key):
    """제품 서명의 binding 하나를 지표 계산용 항목으로 바꾼다. 없으면 None."""
    raster = sig["rasters"].get(key)
    if raster is None:
        return None
    return {"mask": raster["mask"], "dilated": raster["dilated"], "full": lambda: screen_align.unpack_mask(sig["full"][key], sig["full_shape"]),
            "counts": sig["counts"].get(key)}


def union_item(items):
    """role 합집합 항목. 개수 래스터는 합집합에서 정확하지 않으므로 쓰지 않는다."""
    items = [i for i in items if i is not None]
    if not items:
        return None
    return {"mask": np.logical_or.reduce([i["mask"] for i in items]), "dilated": np.logical_or.reduce([i["dilated"] for i in items]),
            "full": lambda: np.logical_or.reduce([i["full"]() for i in items]), "counts": None}


def layer_item(raster, packed, shape, tolerance):
    return {"mask": raster, "dilated": dilate(raster, tolerance), "full": lambda: screen_align.unpack_mask(packed, shape), "counts": None}


def metric_context(sa, sb, shift, *, tolerance, presence_fraction, method="auto"):
    """지표를 계산할 위치(래스터 이동 shift)의 Ω. 래스터 Ω 는 형상 지표에, 원본 Ω 는 면적에 쓴다. 겹침이 없으면 None."""
    views = roi_views(sa, sb, shift[0], shift[1])
    if views is None:
        return None
    _, a_slice, b_slice, omega = views
    factor = sa["factor"]
    window = screen_align.original_window(sa["full_shape"], sb["full_shape"], shift[0] * factor, shift[1] * factor)
    cache = {}

    def omega_full():  # None = 겹침 전체가 valid
        if "omega" not in cache:
            parts = []
            for sig, part in ((sa, window[0]), (sb, window[1])):
                if sig["full_valid"] is not None:
                    parts.append(screen_align.unpack_mask(sig["full_valid"], sig["full_shape"])[part])
            cache["omega"] = np.logical_and.reduce(parts) if parts else None
        return cache["omega"]
    return {"a_slice": a_slice, "b_slice": b_slice, "omega": omega, "window": window, "omega_full": omega_full, "tolerance": tolerance,
            "presence_fraction": presence_fraction, "block_sum": method == "auto" and bool(sa["counts"]) and bool(sb["counts"])}


def original_area(ctx, item, side):
    """원본 해상도 px 면적: item ∧ 원본 Ω. side 0 = A, 1 = B."""
    if item is None or ctx["window"] is None:
        return 0
    if ctx["block_sum"] and item["counts"] is not None:
        return int(item["counts"][ctx["a_slice"] if side == 0 else ctx["b_slice"]].sum())
    mask = item["full"]()[ctx["window"][side]]
    omega = ctx["omega_full"]()
    return int(np.count_nonzero(mask if omega is None else mask & omega))


def metric_row(ctx, kind, key, item_a, item_b):
    """binding 하나의 지표 행(PLAN_v1.2 X4 표). 양쪽 모두 문턱에 못 미치면 None."""
    area_a, area_b = original_area(ctx, item_a, 0), original_area(ctx, item_b, 1)
    top = max(area_a, area_b)
    if top == 0:
        return None
    present_a, present_b = area_a > 0 and area_a >= ctx["presence_fraction"] * top, area_b > 0 and area_b >= ctx["presence_fraction"] * top
    if not present_a and not present_b:
        return None
    row = {"kind": kind, "key": key, "area_a_px": area_a, "area_b_px": area_b, "dice": None, "boundary_f1": None, "boundary_precision": None,
           "boundary_recall": None, "fill_f1": None}
    if present_a != present_b:
        row.update(status="only_a" if present_a else "only_b", area_ratio=0.0, dice=0.0, boundary_f1=0.0, fill_f1=0.0)
        return row  # 한쪽에 없는 binding 은 tolerant_distance 를 부르지 않는다(0 나눗셈)
    row.update(status="both", area_ratio=round(min(area_a, area_b) / top, 6))
    omega = ctx["omega"]
    a, b = item_a["mask"][ctx["a_slice"]] & omega, item_b["mask"][ctx["b_slice"]] & omega
    if not a.any() and not b.any():
        return row  # 원본에는 있지만 래스터 Ω 안에는 없다: 형상 지표를 낼 수 없다
    if not (a.any() and b.any()):
        row.update(dice=0.0, boundary_f1=0.0, fill_f1=0.0)
        return row
    row["dice"] = round(2 * int(np.count_nonzero(a & b)) / (int(np.count_nonzero(a)) + int(np.count_nonzero(b))), 6)
    row["fill_f1"] = round(1.0 - float(tolerant_distance({"mask": a, "dilated": item_a["dilated"][ctx["a_slice"]] & omega},
                                                         {"mask": b, "dilated": item_b["dilated"][ctx["b_slice"]] & omega})), 6)
    edge_a, edge_b = screen_align.inner_boundary(a, omega), screen_align.inner_boundary(b, omega)
    if edge_a.any() and edge_b.any():
        precision = np.count_nonzero(edge_a & dilate(edge_b, ctx["tolerance"])) / np.count_nonzero(edge_a)
        recall = np.count_nonzero(edge_b & dilate(edge_a, ctx["tolerance"])) / np.count_nonzero(edge_b)
        row.update(boundary_precision=round(float(precision), 6), boundary_recall=round(float(recall), 6),
                   boundary_f1=round(float(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)), 6))
    return row


def pair_metric_rows(sa, sb, shift, *, tolerance, presence_fraction=PRESENCE_FRACTION, use_layers=False, method="auto"):
    """제품 쌍의 binding·role·layer·CNT 지표 행 목록. shift 는 metrics_anchor 위치의 래스터 이동이다."""
    ctx = metric_context(sa, sb, shift, tolerance=tolerance, presence_fraction=presence_fraction, method=method)
    if ctx is None:
        return []
    rows = []

    def add(kind, key, item_a, item_b):
        row = metric_row(ctx, kind, key, item_a, item_b)
        if row is not None:
            rows.append(row)
    for key in sorted(sa["rasters"].keys() | sb["rasters"].keys()):
        add("cnt" if key == CNT else "binding", key, signature_item(sa, key), signature_item(sb, key))
    for role in sorted(set(sa["roles"].values()) | set(sb["roles"].values())):
        members = lambda sig: union_item([signature_item(sig, k) for k, r in sorted(sig["roles"].items()) if r == role])
        add("role", role, members(sa), members(sb))
    if use_layers:
        la, lb = sa.get("layers", {}), sb.get("layers", {})
        for name in sorted(la.keys() | lb.keys()):
            add("layer", name, la.get(name), lb.get(name))
    rows.sort(key=lambda r: (r["kind"], r["key"]))
    return rows


def pair_metric_columns(rows, anchor):
    """scores.csv 의 쌍 단위 지표. 빈칸(None)인 항목은 평균에서 빼고, 유효 항목이 없으면 빈칸이다."""
    def mean(values):
        values = [v for v in values if v is not None]
        return round(float(np.mean(values)), 6) if values else None
    bindings = [r for r in rows if r["kind"] == "binding"]
    cnt = [r for r in rows if r["kind"] == "cnt"]
    return {"usage_jaccard_electrodes": round(sum(r["status"] == "both" for r in bindings) / len(bindings), 6) if bindings else None,
            "electrode_similarity": mean([r["boundary_f1"] for r in rows if r["kind"] == "role"]),
            "pattern_similarity": mean([r["boundary_f1"] for r in rows if r["kind"] == "layer"]) if anchor == "LAYER" else None,
            "cnt_similarity": cnt[0]["boundary_f1"] if cnt else None}


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


def load_floor_spec(path):
    """이전 라운드의 summary.json에서 고정할 기준(floor.global, floor.per_group)을 읽는다. summary는 per_group, screen() 결과는 groups라는 키를 쓴다."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    floor = document.get("floor") if isinstance(document, dict) else None
    number = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    if not isinstance(floor, dict) or not number(floor.get("global")) or not isinstance(floor.get("per_group"), dict) \
            or any(not isinstance(k, str) or not number(v) for k, v in floor["per_group"].items()):
        raise ValueError(f"{path} is not a screening summary with floor.global and floor.per_group")
    return {"global": float(floor["global"]), "groups": {k: float(v) for k, v in floor["per_group"].items()}}


def read_base_relations(path, product_ids):
    """합친 relations.csv를 읽는다(사용자가 손으로 고친 파일). 매니페스트에 없는 ID가 있으면 거부한다."""
    raw = Path(path).read_bytes()
    known, rows = set(product_ids), []
    for row in csv.reader(io.StringIO(raw.decode("utf-8"), newline="")):
        if len(row) != 3:
            raise ValueError("base relations CSV requires product_a,product_b,similar|different")
        a, b, kind = (v.strip() for v in row)
        if kind not in {"similar", "different"}:
            raise ValueError("base relation must be similar or different")
        if a not in known or b not in known:
            raise ValueError(f"base relation references a product not in the manifest: {sorted({a, b} - known)}")
        rows.append((a, b, kind))
    return rows, hashlib.sha256(raw).hexdigest()


def base_old_sets(base_split, split, product_ids):
    """구획마다 이전 라운드 제품(old). 매니페스트에 없거나 현재 구획에서 빠진 제품이 있으면 거부한다."""
    if not isinstance(base_split, dict):
        raise ValueError("base split must be a split manifest")
    known, old_of = set(product_ids), {}
    for part in PARTITIONS:
        old = set(base_split.get(part, []))
        if old - known:
            raise ValueError(f"base split product not in the manifest: {sorted(old - known)[:10]}")
        if old - set(split[part]):
            raise ValueError(f"base split product left the partition {part}: {sorted(old - set(split[part]))[:10]}")
        old_of[part] = old
    return old_of


def floor_index(q, n):
    """lower index floor(q·(n−1)). q 의 십진 표기(repr)를 정확한 분수로 곱해 float 반올림(0.29·100 = 28.999…)을 피한다."""
    return math.floor(Fraction(repr(float(q))) * (n - 1))


def screen(manifest, split, *, groups=None, partitions=("train",), factor=8, tolerance=1, margin=0.05, floor=None, cross_group_fraction=0.2,
           min_group_pairs=3, anchor="auto", max_shift=0.35, candidates=5, review_band=0.15, base_split=None, base_relations=None, floor_spec=None,
           align_min_overlap=screen_align.ALIGN_MIN_OVERLAP, align_min_support=screen_align.ALIGN_MIN_SUPPORT, align_min_dice=screen_align.ALIGN_MIN_DICE,
           presence_fraction=PRESENCE_FRACTION, floor_quantile=None):
    """base_split과 base_relations를 함께 주면 증분 모드: 새 제품이 낀 쌍(새×새, 새×기존)만 계산한다. floor_spec은 load_floor_spec() 결과다.
    align_*는 Layer 정렬(scores.csv 의 새 컬럼)에만 쓰이고 tier·floor 판정은 바꾸지 않는다.
    floor_quantile=q 이면 같은 계열 유사도를 정렬한 index floor(q*(n-1)) 의 값을 floor 로 쓴다(None 이면 최솟값). floor·floor_spec 과 배타다."""
    if floor_quantile is not None:
        if isinstance(floor_quantile, bool) or not isinstance(floor_quantile, (int, float)) or not 0 <= floor_quantile <= 1:
            raise ValueError("floor_quantile in [0,1] required")
        if floor is not None or floor_spec is not None:
            raise ValueError("floor_quantile is exclusive with floor and floor_spec")
    if (base_split is None) != (base_relations is None):
        raise ValueError("base_split and base_relations go together")
    if floor is not None and floor_spec is not None:
        raise ValueError("floor and floor_spec are exclusive")
    incremental = base_split is not None
    if incremental and floor is None and floor_spec is None:
        raise ValueError("incremental screening needs exactly one of floor and floor_spec")
    if type(factor) is not int or factor <= 0 or type(tolerance) is not int or tolerance < 0 or not 0 <= margin < 1 or not 0 <= cross_group_fraction <= 1:
        raise ValueError("factor > 0, tolerance >= 0, margin in [0,1) and cross_group_fraction in [0,1] required")
    if anchor not in {"auto", CNT, "ELECTRODES"} or not 0 < max_shift <= 1 or type(candidates) is not int or candidates < 1:
        raise ValueError("anchor must be auto/CNT/ELECTRODES, max_shift in (0,1] and candidates >= 1")
    if not partitions or any(p not in PARTITIONS for p in partitions):
        raise ValueError(f"partitions must be chosen from {PARTITIONS}")
    if not 0 <= align_min_overlap <= 1 or type(align_min_support) is not int or align_min_support < 1 or not 0 <= align_min_dice <= 1:
        raise ValueError("align_min_overlap in [0,1], align_min_support >= 1 and align_min_dice in [0,1] required")
    if not 0 <= presence_fraction <= 1:
        raise ValueError("presence_fraction in [0,1] required")
    document = json.loads(Path(manifest).read_text(encoding="utf-8"))
    ids = [p["product_id"] for p in document["products"]]
    check_split_ids(split, ids)
    wanted = {pid for part in partitions for pid in split[part]}
    if incremental:
        old_of = base_old_sets(base_split, split, ids)
        base_rows, base_sha = read_base_relations(base_relations, ids)
    group_of = {pid: "-" for pid in wanted} if groups is None else load_groups(groups, ids, wanted)
    stubs, signatures, lineage = [], {}, {}
    for product in _iter_manifest_products(manifest, grid=CHECK_GRID):  # 끝까지 읽어 manifest 전체를 검증한다
        stubs.append(SimpleNamespace(product_id=product.product_id, lineage_id=product.lineage_id))
        lineage[product.product_id] = product.lineage_id
        if product.product_id in wanted:
            signatures[product.product_id] = product_signature(product, factor, tolerance)
    validate_split(stubs, split)
    layer_originals = {}
    layers = load_screening_layers(manifest, wanted, factor, layer_originals)  # 사이드카 파일이 없으면 {}, 손상됐으면 오류
    for pid, named in layers.items():
        signatures[pid]["layers"] = {name: layer_item(raster, layer_originals[pid][name], signatures[pid]["full_shape"], tolerance) for name, raster in named.items()}
    pairs = []
    for part in partitions:
        names = sorted(split[part])
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if incremental and a in old_of[part] and b in old_of[part]:
                    continue  # 기존×기존 쌍은 이전 라운드 결과를 쓴다
                result = compare(signatures[a], signatures[b], anchor=anchor, max_shift=max_shift, candidates=candidates, tolerance=tolerance)
                pairs.append({"product_a": a, "product_b": b, "partition": part, "group_a": group_of[a], "group_b": group_of[b],
                              "lineage_a": lineage[a], "lineage_b": lineage[b], "same_lineage": lineage[a] == lineage[b], "result": result,
                              "similarity": round(result["similarity"], 6), "usage_jaccard": round(result["usage_jaccard"], 6),
                              "geometry_distance": None if result["geometry_distance"] is None else round(result["geometry_distance"], 6),
                              "anchor": result["anchor"], "shift_y_um": round(result["shift"][0] * result["um_per_px"], 3), "shift_x_um": round(result["shift"][1] * result["um_per_px"], 3),
                              "alignment_score": None if result["anchor_score"] is None else round(result["anchor_score"], 4), "roi_overlap": round(float(result["roi_overlap"]), 4),
                              "alignment_flag": ";".join(result["flags"])})
                align, columns = layer_columns(signatures[a], signatures[b], result, layers.get(a), layers.get(b), max_shift=max_shift,
                                               min_overlap=align_min_overlap, min_support=align_min_support, min_dice=align_min_dice)
                pairs[-1].update(columns)
                pairs[-1]["layer_flags"], pairs[-1]["layer_excluded"] = align["flags"], align["excluded"]
                on_layer = columns["metrics_anchor"] == "LAYER"  # 판단 지표는 metrics_anchor 위치에서 계산한다
                rows = pair_metric_rows(signatures[a], signatures[b], align["shift"] if on_layer else result["shift"], tolerance=tolerance,
                                        presence_fraction=presence_fraction, use_layers=on_layer)
                pairs[-1]["metric_rows"] = rows
                pairs[-1].update(pair_metric_columns(rows, columns["metrics_anchor"]))
    if incremental:
        new_of = {part: sorted(set(split[part]) - old_of[part]) for part in partitions}
        expected = {part: len(new_of[part]) * (len(new_of[part]) - 1) // 2 + len(new_of[part]) * len(old_of[part]) for part in partitions}
        counts = {part: sum(p["partition"] == part for p in pairs) for part in partitions}
        if counts != expected:
            raise ValueError(f"delta pair count {counts} differs from the expected n(n-1)/2 + n*m {expected}")
    same = [p for p in pairs if p["same_lineage"]]
    def pick(values):  # 정렬한 값의 lower index. q=None 이면 최솟값
        ordered = sorted(values)
        return ordered[0 if floor_quantile is None else floor_index(floor_quantile, len(ordered))]

    global_floor = floor if floor is not None else floor_spec["global"] if floor_spec is not None else (pick([p["similarity"] for p in same]) if same else None)
    if global_floor is None:
        raise ValueError("no same-lineage pair to calibrate with; pass an explicit --floor")
    group_floor = {}
    if floor_spec is not None:
        group_floor = {g: v for g, v in floor_spec["groups"].items() if g in set(group_of.values())}
    for g in sorted(set(group_of.values())):
        within = [p["similarity"] for p in same if p["group_a"] == g and p["group_b"] == g]
        if floor is None and floor_spec is None and len(within) >= min_group_pairs:
            group_floor[g] = pick(within)
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
    floor_info = {"global": global_floor, "groups": group_floor, "margin": margin}
    if floor_spec is not None:
        floor_info["source"] = "base_summary"
    if not incremental:
        summary = summarize(pairs, split, partitions, group_of, global_floor, group_floor, margin)
    else:
        # 판정에는 두 ID가 모두 같은 구획의 기존 제품인 행만 쓴다. 다른 구획·구획을 넘는 행은 무시한다(거부하지 않는다).
        used = [(a, b, k, part) for a, b, k in base_rows for part in partitions if a in old_of[part] and b in old_of[part]]
        validate_relations({frozenset((a, b)) for a, b, k, _ in used if k == "similar"}, {frozenset((a, b)) for a, b, k, _ in used if k == "different"})
        delta = {frozenset((p["product_a"], p["product_b"])): p["tier"] for p in pairs}
        for a, b, kind in base_rows:  # 무시한 행도 delta 쌍과 겹치면 같은 쌍을 두 번 판정하게 된다
            tier = delta.get(frozenset((a, b)))
            if tier in {"similar", "different"}:
                raise ValueError(f"base relation {kind} {sorted((a, b))} " + ("contradicts" if tier != kind else "duplicates") + f" a delta pair judged {tier}")
        summary = summarize(pairs, split, partitions, group_of, global_floor, group_floor, margin,
                            extra_different={part: {pid for a, b, k, q in used if k == "different" and q == part for pid in (a, b)} for part in partitions})
        summary.update({"mode": "incremental", "base": {"relations": str(base_relations), "sha256": base_sha, "pairs": len(used)},
                        "new_products": new_of, "delta_pairs": counts, "expected_delta_pairs": expected})
    if floor_spec is not None:
        summary["floor"]["source"] = "base_summary"
    if floor_quantile is not None:
        summary["floor"]["quantile"] = floor_quantile
        summary["floor"]["quantile_method"] = "lower_index"
    return {"pairs": pairs, "signatures": signatures, "floor": floor_info, "summary": summary}


def summarize(pairs, split, partitions, group_of, global_floor, group_floor, margin, extra_different=None):
    cross = [p for p in pairs if not p["same_lineage"]]
    same = [p for p in pairs if p["same_lineage"]]

    def separation(key):
        return auc([key(p) for p in same], [key(p) for p in cross])

    def separation_valid(name):  # 빈칸인 쌍은 뺀다
        return auc([p[name] for p in same if p[name] is not None], [p[name] for p in cross if p[name] is not None])
    new_metrics = ("similarity_layer",) + PAIR_METRICS
    tiers = {t: sum(p["tier"] == t for p in pairs) for t in ("similar", "different", "unknown")}
    uncovered = {}
    for part in partitions:
        have = {pid for p in pairs if p["partition"] == part and p["tier"] == "different" for pid in (p["product_a"], p["product_b"])}
        have |= (extra_different or {}).get(part, set())
        uncovered[part] = sorted(set(split[part]) - have)
    return {"pairs_by_tier": tiers, "unknown_by_reason": {r: sum(p["reason"] == r and p["tier"] == "unknown" for p in pairs) for r in sorted({p["reason"] for p in pairs if p["tier"] == "unknown"})},
            "floor": {"global": global_floor, "per_group": group_floor, "margin": margin},
            "separation_auc_same_vs_other_lineage": {"similarity": separation(lambda p: p["similarity"]), "usage_jaccard": separation(lambda p: p["usage_jaccard"]),
                                                      "geometry": separation(lambda p: 0.0 if p["geometry_distance"] is None else 1 - p["geometry_distance"]),
                                                      **{name: separation_valid(name) for name in new_metrics},
                                                      "n_pairs_with_value": {name: sum(p[name] is not None for p in pairs) for name in new_metrics}},
            "alignment": {"anchors": {k: sum(p["anchor"] == k for p in pairs) for k in sorted({p["anchor"] for p in pairs})},
                          "flags": {f: sum(f in p["alignment_flag"].split(";") for p in pairs) for f in ("no_anchor", "anchor_fallback_electrodes", "ambiguous_alignment", "alignment_unreliable", "low_overlap")},
                          "metrics_anchors": {k: sum(p["metrics_anchor"] == k for p in pairs) for k in sorted({p["metrics_anchor"] for p in pairs})},
                          "median_roi_overlap": float(np.median([p["roi_overlap"] for p in pairs])) if pairs else None},
            "layer_alignment": {"flags": {f: sum(f in p["layer_flags"] for p in pairs) for f in screen_align.FLAGS},
                                "excluded_layers": {c: sum(c == cause for p in pairs for cause in p["layer_excluded"].values()) for c in screen_align.EXCLUDE_CAUSES}},
            "review": {"pairs_to_review": sum(bool(p["review"]) for p in pairs),
                       "by_reason": {r: sum(r in p["review"] for p in pairs) for r in sorted({r for p in pairs for r in p["review"]})}},
            "products_without_different_pair": uncovered, "groups": {g: sum(v == g for v in group_of.values()) for g in sorted(set(group_of.values()))}}


# ---- 출력 -------------------------------------------------------------------------------------------------------
def write_csvs(result, out, scores_path, bindings_path=None):
    """bindings_path 가 None 이면 `<out>.bindings.csv`. 쓰기 전에 모든 경로의 충돌을 검사하고, bindings·scores → relations 순으로 쓴다."""
    if bindings_path is None:
        bindings_path = Path(out).with_name(Path(out).name + ".bindings.csv")
    for path in (bindings_path, out, scores_path):
        if Path(path).exists():
            raise FileExistsError(f"refusing to overwrite {path}; choose a new path")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pairs = sorted(result["pairs"], key=lambda p: (p["partition"], p["product_a"], p["product_b"]))
    with open(bindings_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, BINDING_FIELDS)
        writer.writeheader()
        for p in pairs:
            for row in p["metric_rows"]:  # 쌍 안의 행은 (kind, key) 순이다
                writer.writerow({"product_a": p["product_a"], "product_b": p["product_b"], "partition": p["partition"], "metrics_anchor": p["metrics_anchor"], **row})
    with open(scores_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, SCORE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(pairs, key=lambda p: (-p["similarity"], p["product_a"], p["product_b"])))
    with open(out, "w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows([p["product_a"], p["product_b"], p["tier"]] for p in pairs if p["tier"] in {"similar", "different"})
    similar = {frozenset((p["product_a"], p["product_b"])) for p in pairs if p["tier"] == "similar"}
    different = {frozenset((p["product_a"], p["product_b"])) for p in pairs if p["tier"] == "different"}
    validate_relations(similar, different)


def render_review(result, top_pairs=60, top_bindings=6, top_layers=4):
    """review.html 문자열. 렌더링은 screen_report 에 있다(그 모듈이 이 모듈을 import 하므로 여기서는 호출할 때 가져온다)."""
    from . import screen_report
    return screen_report.render_review(result, top_pairs, top_bindings, top_layers)


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
    ap.add_argument("--floor-quantile", type=float, help="use the lower-index quantile q in [0,1] of the same-lineage similarities as the floor instead of the minimum (exclusive with --floor and --floor-json)")
    ap.add_argument("--cross-group-fraction", type=float, default=0.2, help="cross-group 'different' pairs kept, as a fraction of within-group ones")
    ap.add_argument("--floor-json", help="summary.json of an earlier screening: reuse its floor.global and floor.per_group (exclusive with --floor)")
    ap.add_argument("--base-split", help="split.json of the previous round; with --base-relations only pairs that include a new product are screened")
    ap.add_argument("--base-relations", help="merged relations.csv of the previous round (hand edits kept); --out then holds the delta rows only")
    ap.add_argument("--align-min-overlap", type=float, default=screen_align.ALIGN_MIN_OVERLAP, help="Layer alignment: a shift needs |valid overlap| / min(|valid A|, |valid B|) at least this")
    ap.add_argument("--align-min-support", type=int, default=screen_align.ALIGN_MIN_SUPPORT, help="Layer alignment: raster pixels a layer needs in A and in B inside the overlap of a shift")
    ap.add_argument("--align-min-dice", type=float, default=screen_align.ALIGN_MIN_DICE, help="Layer alignment: L1(first-layer) Dice at the chosen shift below this is alignment_unreliable (shift not used)")
    ap.add_argument("--presence-fraction", type=float, default=PRESENCE_FRACTION, help="a binding counts as present when its original-resolution area is at least this fraction of the larger side's area")
    ap.add_argument("--bindings-out", help="new per-binding metrics CSV (default <out>.bindings.csv)")
    ap.add_argument("--review-pairs", type=int, default=60, help="pairs drawn as image cards; the table always lists every pair to review")
    ap.add_argument("--review-band", type=float, default=0.15, help="auto-different pairs within margin + band of the floor are listed for review")
    ap.add_argument("--review-bindings", type=int, default=6, help="deprecated and ignored: cards now list every binding in a metrics table")
    ap.add_argument("--review-layers", type=int, default=4, help="per card: layers (or bindings, without Layer) shown expanded as A | B | difference tiles; the rest are folded")
    a = ap.parse_args(argv)
    if a.floor_json is not None and a.floor is not None:
        ap.error("--floor and --floor-json are exclusive")
    if a.floor_quantile is not None:
        if a.floor is not None or a.floor_json is not None:
            ap.error("--floor-quantile is exclusive with --floor and --floor-json")
        if not 0 <= a.floor_quantile <= 1:  # NaN 도 여기서 걸린다
            ap.error("--floor-quantile must be in [0,1]")
    if (a.base_split is None) != (a.base_relations is None):
        ap.error("--base-split and --base-relations go together")
    if a.base_split is not None and a.floor is None and a.floor_json is None:
        ap.error("incremental screening needs exactly one of --floor and --floor-json")
    split = json.loads(Path(a.split).read_text(encoding="utf-8"))
    extra = {} if a.base_split is None else {"base_split": json.loads(Path(a.base_split).read_text(encoding="utf-8")), "base_relations": a.base_relations}
    if a.floor_json is not None:
        extra["floor_spec"] = load_floor_spec(a.floor_json)
    out = Path(a.out)
    scores, review, summary = (out.with_name(out.name + suffix) for suffix in (".scores.csv", ".review.html", ".summary.json"))
    bindings = Path(a.bindings_out) if a.bindings_out else out.with_name(out.name + ".bindings.csv")
    resolved = [str(p.resolve()).casefold() for p in (out, scores, review, summary, bindings)]  # 대소문자만 다른 경로도 같은 경로로 본다(모든 플랫폼 같은 규칙)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"output paths must be distinct: out={out}, scores={scores}, review={review}, summary={summary}, bindings={bindings}")
    for path in (bindings, review, summary):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}; choose a new path")
    result = screen(a.manifest, split, groups=a.groups, partitions=tuple(a.partitions), factor=a.factor, tolerance=a.tolerance, margin=a.margin,
                    floor=a.floor, cross_group_fraction=a.cross_group_fraction, anchor=a.anchor, max_shift=a.max_shift, candidates=a.candidates, review_band=a.review_band,
                    align_min_overlap=a.align_min_overlap, align_min_support=a.align_min_support, align_min_dice=a.align_min_dice, presence_fraction=a.presence_fraction, floor_quantile=a.floor_quantile, **extra)
    write_csvs(result, out, scores, bindings)
    # 쓰기 순서: bindings·scores → relations → review → summary(필수 3개는 마지막이라 중단되면 부분 출력 검사가 잡는다)
    page = render_review(result, a.review_pairs, a.review_bindings, a.review_layers)
    review.write_text(page, encoding="utf-8")
    result["summary"]["review"]["html_bytes"] = review.stat().st_size
    summary.write_text(json.dumps(result["summary"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"relations": str(out), "scores": str(scores), "review": str(review), "summary_file": str(summary), **result["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
