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
# three-axis 판정에서만 scores.csv 끝에 붙는 열(legacy 는 SCORE_FIELDS 그대로)
JUDGE_SCORE_FIELDS = ("axis_p", "axis_p_layer", "axis_u", "size_ratio_y", "size_ratio_x", "size_delta_unit_px_y", "size_delta_unit_px_x", "size_tag",
                      "judge_case", "judge_p_floor", "judge_u_floor")
# --judge-size-u-floor 가 켜졌을 때만 scores.csv 끝에 붙는 열(tile 열 이름은 screen_tile.tile_pattern_axis 반환 키와 같다)
JUDGE_SIZE_SCORE_FIELDS = ("judge_size_u_floor", "tile_status", "tile_source", "tile_p", "tile_size_y_cells", "tile_size_x_cells", "tile_count", "tile_applied",
                           "tile_step_replaced", "tile_radius_y_cells", "tile_radius_x_cells", "tile_shift_min_y", "tile_shift_max_y", "tile_shift_min_x", "tile_shift_max_x")
JUDGE_REVIEW_REASONS = frozenset({"close_to_same_lineage", "near_boundary", "pattern_same_usage_differs", "size_differs_out_of_support",
                                  "pattern_same_usage_unknown", "no_pattern_value", "near_boundary_tiled"})  # three-axis 에서 unknown 쌍이 검토 목록에 들어가는 reason
JUDGE_MODES = ("legacy", "three-axis")
JUDGE_SCHEMA = "dl1_judge_v1"
JUDGE_SIZE_SUPPORT_PX = 7  # PLAN v1.2 G-size: LAYER 정렬에서 |ΔU| 가 이 값 이하인 크기 차만 P 가 믿을 만하다(T2 측정 근거)
JUDGE_TILE_MAX_PAIRS = 20000  # --judge-tile-max-pairs 기본값
JUDGE_SIZE_FIXED = {"support_px": 40, "s_max_cells": 2, "tile_min_cells": 7, "xy_support_px": 50, "xy_ratio_min": 0.15, "tile_min_cells_xy": 5, "factor": 8,
                    "tolerance": 1}  # screen_tile 상수(MAX_DELTA_U_PX·S_MAX_CELLS·TILE_MIN_CELLS·XY_SUPPORT_PX·XY_RATIO_MIN·TILE_MIN_CELLS_XY)와 같은 값(순환 import 라 값으로 적고 테스트가 같음을 단정한다). summary.judge.size.config 에 고정해 기록한다
TILE_FAILURE_STATUSES = frozenset({"not_run", "capped", "geometry_unsupported", "out_of_support", "tile_too_small", "too_many_tiles", "unit_align_failed", "low_coverage"})
TILE_NOT_EXECUTED = frozenset({None, "not_run", "capped", "not_candidate"})  # 이 밖의 tile_status 는 tile_pattern_axis 를 실제로 부른 쌍이다


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

    사이드카 파일이 없으면 {} (모든 제품에 Layer 없음). 파일이 있는데 손상됐으면(빈 products 포함) ValueError 다: 스키마, 경로 containment, manifest 의 product_id·frame 연결,
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
    if not document["products"]:
        raise bad(f"no products in {bundle_path(root, SCREENING_LAYERS_FILE)}")
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


def boundary_f1(a, b, omega, tolerance):
    """Ω 안 두 마스크의 안쪽 경계 정밀도·재현율·F1(반올림 전 값). 어느 한쪽 경계가 비면 None."""
    edge_a, edge_b = screen_align.inner_boundary(a, omega), screen_align.inner_boundary(b, omega)
    if not (edge_a.any() and edge_b.any()):
        return None
    precision = np.count_nonzero(edge_a & dilate(edge_b, tolerance)) / np.count_nonzero(edge_a)
    recall = np.count_nonzero(edge_b & dilate(edge_a, tolerance)) / np.count_nonzero(edge_b)
    return precision, recall, 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


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
    edges = boundary_f1(a, b, omega, ctx["tolerance"])
    if edges is not None:
        precision, recall, f1 = edges
        row.update(boundary_precision=round(float(precision), 6), boundary_recall=round(float(recall), 6), boundary_f1=round(float(f1), 6))
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


def pair_axes(rows, anchor, sa, sb):
    """판정 세 축 입력값(PLAN v1.1 A1). 어떤 출력에도 쓰지 않는 순수 계산이다.
    axis_p = binding 행 중 status=both 이고 boundary_f1 이 유한한 행의 평균, axis_p_layer = LAYER anchor 일 때 layer 행의 같은 평균(진단),
    axis_u = usage_jaccard_electrodes 와 같은 값, size_* = 원본 frame shape 의 축별 비교(delta 는 (B−A)/2 원본 px, 홀수 차이는 x.5 float)."""
    def mean(kind):
        values = [r["boundary_f1"] for r in rows if r["kind"] == kind and r["status"] == "both" and r["boundary_f1"] is not None and np.isfinite(r["boundary_f1"])]
        return round(float(np.mean(values)), 6) if values else None
    ratio = [round(min(a, b) / max(a, b), 6) for a, b in zip(sa["full_shape"], sb["full_shape"])]
    delta = [(b - a) / 2 for a, b in zip(sa["full_shape"], sb["full_shape"])]
    delta = [int(d) if d == int(d) else d for d in delta]
    return {"axis_p": mean("binding"), "axis_p_layer": mean("layer") if anchor == "LAYER" else None,
            "axis_u": pair_metric_columns(rows, anchor)["usage_jaccard_electrodes"],
            "size_ratio_y": ratio[0], "size_ratio_x": ratio[1], "size_delta_unit_px_y": delta[0], "size_delta_unit_px_x": delta[1],
            "size_tag": "same" if ratio == [1.0, 1.0] else "size_differs"}


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
    judge = document.get("judge") if isinstance(document, dict) else None
    if judge is None:  # legacy summary: 지금 규칙 그대로
        if not isinstance(floor, dict) or not number(floor.get("global")) or not isinstance(floor.get("per_group"), dict) \
                or any(not isinstance(k, str) or not number(v) for k, v in floor["per_group"].items()):
            raise ValueError(f"{path} is not a screening summary with floor.global and floor.per_group")
        return {"global": float(floor["global"]), "groups": {k: float(v) for k, v in floor["per_group"].items()}}
    # three-axis summary: similarity floor.global 은 null 일 수 있다(같은 계열이 없었던 실행)
    if not isinstance(floor, dict) or not (floor.get("global") is None or number(floor["global"])) or not isinstance(floor.get("per_group"), dict) \
            or any(not isinstance(k, str) or not number(v) for k, v in floor["per_group"].items()):
        raise ValueError(f"{path} is not a screening summary with floor.global (number or null) and floor.per_group")

    def bad(why):
        return ValueError(f"{path} has an invalid judge block: {why}")
    if not isinstance(judge, dict) or judge.get("schema") != JUDGE_SCHEMA or judge.get("mode") != "three-axis":
        raise bad(f"schema {JUDGE_SCHEMA} and mode three-axis are required")
    if not number(judge.get("margin")):
        raise bad("margin must be a finite number")
    if not (judge.get("quantile") is None or number(judge["quantile"])):
        raise bad("quantile must be a finite number or null")
    axes = {}
    for axis in ("p", "u"):
        block = judge.get(axis)
        if not isinstance(block, dict) or not number(block.get("global")) or not isinstance(block.get("per_group"), dict) \
                or any(not isinstance(k, str) or not number(v) for k, v in block["per_group"].items()):
            raise bad(f"{axis} needs a finite global and a per_group map of finite numbers")
        if not isinstance(block.get("source"), str) or isinstance(block.get("n_values"), bool) or not isinstance(block.get("n_values"), int) or block["n_values"] < 0:
            raise bad(f"{axis} needs a string source and an integer n_values >= 0")
        axes[axis] = {"global": float(block["global"]), "groups": {k: float(v) for k, v in block["per_group"].items()}, "source": block["source"],
                      "n_values": block["n_values"]}
    size = None
    if "size" in judge:  # 선택 블록: 있을 때만 설정(config)을 검증해 읽는다. stats 는 증분에서 다시 계산하므로 읽지 않는다
        block = judge["size"]
        config = block.get("config") if isinstance(block, dict) else None
        if not isinstance(config, dict) or set(config) != {"u_floor", "enabled", "max_pairs", *JUDGE_SIZE_FIXED}:
            raise bad("size needs a config with u_floor, enabled, max_pairs, support_px, s_max_cells, tile_min_cells, xy_support_px, xy_ratio_min, tile_min_cells_xy, "
                      "factor and tolerance (a summary written before the xy support range lacks the xy keys: move the relations and run again)")
        if not number(config["u_floor"]) or isinstance(config["u_floor"], bool) or not 0 <= config["u_floor"] <= 1:
            raise bad("size.config.u_floor must be a number in [0,1]")
        if config["enabled"] is not True:
            raise bad("size.config.enabled must be true")
        if isinstance(config["max_pairs"], bool) or not isinstance(config["max_pairs"], int) or config["max_pairs"] < 0:
            raise bad("size.config.max_pairs must be an integer >= 0")
        for key, fixed in JUDGE_SIZE_FIXED.items():
            if isinstance(config[key], bool) or config[key] != fixed:
                raise bad(f"size.config.{key} must be {fixed}")
        size = {"u_floor": float(config["u_floor"]), "enabled": True, "max_pairs": config["max_pairs"], **JUDGE_SIZE_FIXED}
    return {"global": None if floor["global"] is None else float(floor["global"]), "groups": {k: float(v) for k, v in floor["per_group"].items()},
            "judge": {"margin": float(judge["margin"]), "quantile": None if judge.get("quantile") is None else float(judge["quantile"]), **axes,
                      **({} if size is None else {"size": size})}}


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


def make_pick(floor_quantile):
    def pick(values):  # 정렬한 값의 lower index. q=None 이면 최솟값
        ordered = sorted(values)
        return ordered[0 if floor_quantile is None else floor_index(floor_quantile, len(ordered))]
    return pick


def judge_pairs_legacy(pairs, group_of, *, floor, floor_spec, pick, margin, min_group_pairs):
    """기존 similarity 판정(이동만 했다: 동작·순서 불변). 쌍마다 floor·tier·reason 을 채우고 (global_floor, group_floor) 를 돌려준다."""
    same = [p for p in pairs if p["same_lineage"]]
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
    return global_floor, group_floor


def size_supported(metrics_anchor, axes):
    """PLAN v1.2 G-size 지원 범위: metrics_anchor 가 LAYER 이고 max|size_delta_unit_px| <= JUDGE_SIZE_SUPPORT_PX."""
    return metrics_anchor == "LAYER" and max(abs(axes["size_delta_unit_px_y"]), abs(axes["size_delta_unit_px_x"])) <= JUDGE_SIZE_SUPPORT_PX


def judge_verdict(axes, metrics_anchor, same_lineage, p_floor, u_floor, margin, size_u_floor=None, tile_p=None, tiled=False):
    """three-axis 판정표(PLAN v1.1 A2·X2, v1.2 G-size, T4d 혼합)의 초기 (tier, reason, judge_case). cross-group cap 은 이 뒤에 따로 적용한다.
    size_u_floor(dl1-size B)가 있으면 단일 P < p_floor−margin 이고 크기가 달라 G-size 밖인 쌍만 다시 가른다. None 이면 현행과 같다.
    tiled=False 는 1차 판정(U 가 있는 쌍은 모두 tile 후보라 unknown). tiled=True 는 tile 단계의 판정: tile_p 가 있으면 tile_p 로 가르고
    (U 가 기준보다 낮아도 tile_p 가 높으면 different 로 보내지 않는다), tile_p 가 없으면(실패·capped·not_run) U 가 기준보다 낮은 쌍만 different 다."""
    differs = axes["size_tag"] == "size_differs"
    if same_lineage:
        return "similar", "same_lineage", 3 if differs else None
    p, u = axes["axis_p"], axes["axis_u"]
    if p is None:
        return "unknown", "no_pattern_value", None
    exact = lambda value: Fraction(repr(float(value)))  # 십진 표기 그대로 비교한다: 0.52 - 0.05 = 0.47000000000000003 같은 float 오차가 경계를 옮기지 않는다
    if exact(p) < exact(p_floor) - exact(margin):
        if differs and not size_supported(metrics_anchor, axes):
            if size_u_floor is None or u is None:
                return "unknown", "size_differs_out_of_support", 3
            if not tiled:  # tile 단계가 다시 판정한다
                return "unknown", "size_differs_out_of_support", 3
            if tile_p is None:  # 미지원·실패·cap·미실행
                if exact(u) < exact(size_u_floor):
                    return "different", "size_differs_usage_below_floor", 2  # tile 로 확인 못 한 쌍은 사용(U)이 크기 기준보다 낮으면 다른 회로다
                return "unknown", "size_differs_out_of_support", 3
            if exact(tile_p) < exact(p_floor) - exact(margin):
                return "different", "size_differs_below_pattern_floor_tiled", 2
            if exact(tile_p) < exact(p_floor):
                return "unknown", "near_boundary_tiled", None
            if u is not None and exact(u) < exact(u_floor):
                return "unknown", "pattern_same_usage_differs", 1
            return "unknown", "close_to_same_lineage", 3
        return "different", "below_pattern_floor", 2
    if exact(p) < exact(p_floor):
        return "unknown", "near_boundary", None
    if u is None:
        return "unknown", "pattern_same_usage_unknown", None
    if exact(u) < exact(u_floor):
        return "unknown", "pattern_same_usage_differs", 1
    return "unknown", "close_to_same_lineage", 3 if differs else None


def judge_axis_floor(axis, name, explicit, spec, same, group_of, pick, min_group_pairs):
    """한 축(p 또는 u)의 기준값. 증분이면 저장값, --judge-floor-<axis> 가 있으면 그 값(그룹 기준 없음), 아니면 같은 계열 유효 값으로 보정한다."""
    values = [p["judge_axes"][name] for p in same if p["judge_axes"][name] is not None]
    if spec is not None:
        return {"global": spec["global"], "per_group": {g: v for g, v in spec["groups"].items() if g in set(group_of.values())},
                "source": spec["source"], "n_values": spec["n_values"]}
    if explicit is not None:
        return {"global": float(explicit), "per_group": {}, "source": "explicit", "n_values": len(values)}
    if not values:
        raise ValueError(f"no same-lineage pair with a valid {name} value to calibrate with; pass --judge-floor-{axis}")
    per_group = {}
    for g in sorted(set(group_of.values())):
        within = [p["judge_axes"][name] for p in same if p["group_a"] == g and p["group_b"] == g and p["judge_axes"][name] is not None]
        if len(within) >= min_group_pairs:
            per_group[g] = pick(within)
    return {"global": pick(values), "per_group": per_group, "source": "calibrated", "n_values": len(values)}


def judge_pairs_three_axis(pairs, group_of, *, floor, floor_spec, pick, margin, min_group_pairs, judge_floor_p, judge_floor_u, size_u_floor=None):
    """three-axis 판정. 쌍마다 축 값·기준·tier·reason·judge_case 를 채우고 (similarity global, similarity groups, {"p": ..., "u": ...}) 를 돌려준다.
    similarity floor 는 계산되면 기록하고 없어도 오류가 아니다(L5)."""
    same = [p for p in pairs if p["same_lineage"]]
    spec_judge = None if floor_spec is None else floor_spec["judge"]
    axes_floor = {axis: judge_axis_floor(axis, name, explicit, None if spec_judge is None else spec_judge[axis], same, group_of, pick, min_group_pairs)
                  for axis, name, explicit in (("p", "axis_p", judge_floor_p), ("u", "axis_u", judge_floor_u))}
    global_floor = floor if floor is not None else floor_spec["global"] if floor_spec is not None else (pick([p["similarity"] for p in same]) if same else None)
    group_floor = {}
    if floor_spec is not None:
        group_floor = {g: v for g, v in floor_spec["groups"].items() if g in set(group_of.values())}
    for g in sorted(set(group_of.values())):
        within = [p["similarity"] for p in same if p["group_a"] == g and p["group_b"] == g]
        if floor is None and floor_spec is None and len(within) >= min_group_pairs:
            group_floor[g] = pick(within)

    def floor_of(info, p):
        return info["per_group"].get(p["group_a"], info["global"]) if p["group_a"] == p["group_b"] else info["global"]
    for p in pairs:
        axes = p["judge_axes"]
        p.update({key: axes[key] for key in JUDGE_SCORE_FIELDS if key in axes})
        p["axis_p_eff"] = axes["axis_p"]  # tile_p 가 나오면 tile 단계가 바꾼다. cap·weakest·검토 이유의 P 정렬은 이 값을 쓴다(기능 off 면 axis_p 와 같다)
        p["floor"] = group_floor.get(p["group_a"], global_floor) if p["group_a"] == p["group_b"] else global_floor
        p["judge_p_floor"], p["judge_u_floor"] = floor_of(axes_floor["p"], p), floor_of(axes_floor["u"], p)
        p["tier"], p["reason"], p["judge_case"] = judge_verdict(axes, p["metrics_anchor"], p["same_lineage"], p["judge_p_floor"], p["judge_u_floor"], margin,
                                                                size_u_floor=size_u_floor)
        if size_u_floor is not None:  # tile 은 이 뒤의 judge_tile_stage 가 돌린다: U 가 있는 size_differs_out_of_support 쌍은 모두 not_run(후보), 나머지는 후보가 아니다
            p.update({key: None for key in JUDGE_SIZE_SCORE_FIELDS})
            p["judge_size_u_floor"] = size_u_floor
            if p["reason"] == "size_differs_out_of_support" and axes["axis_u"] is not None:
                p["tile_status"] = "not_run"
            else:
                p["tile_status"] = "not_candidate"
    return global_floor, group_floor, axes_floor


def tile_inputs(sa, sb, metrics_anchor):
    """tile 정렬 소스와 P 를 낼 binding 이름. metrics_anchor LAYER 이고 공통 층이 있으면 층 래스터(tile_source=layer), 아니면 공통 binding 래스터(CNT 포함, binding)."""
    common_layers = sorted(sa.get("layers", {}).keys() & sb.get("layers", {}).keys()) if metrics_anchor == "LAYER" else []
    if common_layers:
        sources, source = {k: (sa["layers"][k]["mask"], sb["layers"][k]["mask"]) for k in common_layers}, "layer"
    else:
        sources, source = {k: (sa["rasters"][k]["mask"], sb["rasters"][k]["mask"]) for k in sorted(sa["rasters"].keys() & sb["rasters"].keys())}, "binding"
    return sources, source, sorted(k for k in sa["rasters"].keys() & sb["rasters"].keys() if k != CNT)


def judge_tile_stage(pairs, signatures, shifts, size, margin):
    """1차 판정 뒤 tile_status == not_run 인 쌍(U 가 있는 크기 차 쌍 전부)에 tile 정렬을 돌려 tile_p 로 판정표를 다시 적용한다.
    (−U, product_a, product_b) 순으로 앞 max_pairs 개만 돌리고 나머지는 capped, max_pairs 0 이면 모두 not_run 으로 둔다.
    돌리지 못한 쌍(capped·not_run)과 tile 실패 쌍은 tile_p 없음 행으로 판정한다(U 가 기준보다 낮으면 different).
    shifts[(a, b)] = metrics_anchor 위치의 래스터 이동(B[y,x] → A[y+dy,x+dx]). compare·layer 정렬처럼 두 프레임 중심을 맞춘 기준이므로
    screen_tile 이 쓰는 원점 이동으로 바꿔서(screen_tile.origin_shift) 넘긴다."""
    from . import screen_tile  # screen_tile 이 이 모듈의 boundary_f1 을 가져오므로 늦게 가져온다
    candidates = sorted((p for p in pairs if p.get("tile_status") == "not_run"), key=lambda p: (-p["axis_u"], p["product_a"], p["product_b"]))
    for rank, p in enumerate(candidates):
        if rank >= size["max_pairs"]:
            if size["max_pairs"] > 0:
                p["tile_status"] = "capped"
            p["tier"], p["reason"], p["judge_case"] = judge_verdict(p["judge_axes"], p["metrics_anchor"], p["same_lineage"], p["judge_p_floor"], p["judge_u_floor"], margin,
                                                                    size_u_floor=size["u_floor"], tiled=True)
            continue
        sa, sb = signatures[p["product_a"]], signatures[p["product_b"]]
        sources, source, binding_keys = tile_inputs(sa, sb, p["metrics_anchor"])
        shift = screen_tile.origin_shift(sa, sb, shifts[(p["product_a"], p["product_b"])])
        out = screen_tile.tile_pattern_axis(sa, sb, shift, sources, binding_keys, factor=size["factor"], tolerance=size["tolerance"])
        p.update(out)
        p["tile_source"] = source
        p["tier"], p["reason"], p["judge_case"] = judge_verdict(p["judge_axes"], p["metrics_anchor"], p["same_lineage"], p["judge_p_floor"], p["judge_u_floor"], margin,
                                                                size_u_floor=size["u_floor"], tile_p=out["tile_p"], tiled=True)
        if out["tile_p"] is not None:
            p["axis_p_eff"] = out["tile_p"]


def judge_summary(pairs, axes_floor, margin, floor_quantile, spec_judge, size=None):
    """summary.json 의 judge 블록(스키마 dl1_judge_v1, three-axis 에서만). size = 설정 dict(기능 on 일 때만), 있으면 size 블록과 cases 새 키가 붙는다."""
    cross = [p for p in pairs if not p["same_lineage"]]
    same = [p for p in pairs if p["same_lineage"]]
    names = ("axis_p", "axis_u")
    quantile = floor_quantile if spec_judge is None else spec_judge["quantile"]
    judge = {"schema": JUDGE_SCHEMA, "mode": "three-axis", "margin": margin, "quantile": quantile, "p": axes_floor["p"], "u": axes_floor["u"],
             "cases": {"case_1": sum(p["judge_case"] == 1 for p in pairs), "case_2": sum(p["judge_case"] == 2 for p in pairs),
                       "case_2_capped": sum(p["reason"] == "cross_group_capped" for p in pairs),
                       "case_3": sum(p["judge_case"] == 3 for p in pairs),
                       "size_differs_out_of_support": sum(p["reason"] == "size_differs_out_of_support" for p in pairs)},
             "auc": {**{name: auc([p[name] for p in same if p[name] is not None], [p[name] for p in cross if p[name] is not None]) for name in names},
                     "n_pairs_with_value": {name: sum(p[name] is not None for p in pairs) for name in names}}}
    if size is not None:
        status = lambda p: p.get("tile_status")
        judge["cases"].update({"size_differs_usage_below_floor": sum(p["reason"] == "size_differs_usage_below_floor" for p in pairs),
                               "size_differs_tiled": sum(status(p) == "ok" for p in pairs),
                               "size_differs_tiled_different": sum(p["reason"] == "size_differs_below_pattern_floor_tiled" for p in pairs),
                               "size_differs_tile_unsupported": sum(status(p) in TILE_FAILURE_STATUSES for p in pairs)})
        judge["size"] = {"config": dict(size),
                         "stats": {"pairs_usage_below_floor": sum(p["reason"] == "size_differs_usage_below_floor" for p in pairs), "pairs_tiled": sum(status(p) not in TILE_NOT_EXECUTED for p in pairs),
                                   "pairs_capped": sum(status(p) == "capped" for p in pairs),
                                   "tile_status": {s: sum(status(p) == s for p in pairs) for s in sorted({status(p) for p in pairs if status(p) is not None})}}}
    return judge


def screen(manifest, split, *, groups=None, partitions=("train",), factor=8, tolerance=1, margin=None, floor=None, cross_group_fraction=0.2,
           min_group_pairs=3, anchor="auto", max_shift=0.35, candidates=5, review_band=0.15, base_split=None, base_relations=None, floor_spec=None,
           align_min_overlap=screen_align.ALIGN_MIN_OVERLAP, align_min_support=screen_align.ALIGN_MIN_SUPPORT, align_min_dice=screen_align.ALIGN_MIN_DICE,
           presence_fraction=PRESENCE_FRACTION, floor_quantile=None, judge=None, judge_floor_p=None, judge_floor_u=None,
           judge_size_u_floor=None, judge_tile_max_pairs=None):
    """base_split과 base_relations를 함께 주면 증분 모드: 새 제품이 낀 쌍(새×새, 새×기존)만 계산한다. floor_spec은 load_floor_spec() 결과다.
    align_*는 Layer 정렬(scores.csv 의 새 컬럼)에만 쓰이고 tier·floor 판정은 바꾸지 않는다.
    floor_quantile=q 이면 같은 계열 유사도를 정렬한 index floor(q*(n-1)) 의 값을 floor 로 쓴다(None 이면 최솟값). floor·floor_spec 과 배타다.
    judge: "legacy"(기본, similarity 판정) | "three-axis"(P·U 판정). None 이면 비증분은 legacy, 증분(floor_spec)은 base summary 의 모드를 잇고 다르면 거부한다.
    margin=None 이면 0.05, 단 three-axis 증분은 base summary 에 저장된 margin 을 쓰고 다른 값을 주면 거부한다. judge_floor_p/u 는 three-axis 의 축별 명시 기준이다.
    judge_size_u_floor=X([0,1])는 크기가 다른 쌍의 B 판정(U 기준)을 켠다. three-axis 에서만, factor 8·tolerance 1 에서만 허용하고 floor_spec 과는 배타다.
    judge_tile_max_pairs(정수 >= 0, 기본 20000)는 size 기준과 함께만 준다. 증분은 base summary 의 size 설정을 그대로 쓴다."""
    if judge is not None and judge not in JUDGE_MODES:
        raise ValueError(f"judge must be one of {JUDGE_MODES}")
    spec_judge = None if floor_spec is None else floor_spec.get("judge")
    if floor_spec is not None:
        base_mode = "three-axis" if spec_judge is not None else "legacy"
        if judge is not None and judge != base_mode:
            raise ValueError("judge mode differs from the base round summary")
        judge = base_mode
    elif judge is None:
        judge = "legacy"
    if judge_floor_p is not None or judge_floor_u is not None:
        if judge == "legacy":
            raise ValueError("judge_floor_p and judge_floor_u need the three-axis judge")
        if floor_spec is not None:
            raise ValueError("judge_floor_p and judge_floor_u are exclusive with floor_spec")
        for value in (judge_floor_p, judge_floor_u):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1):
                raise ValueError("judge_floor_p and judge_floor_u in [0,1] required")
    size = None
    if judge_size_u_floor is not None or judge_tile_max_pairs is not None:
        if judge == "legacy":
            raise ValueError("judge_size_u_floor and judge_tile_max_pairs need the three-axis judge")
        if floor_spec is not None:
            raise ValueError("judge_size_u_floor and judge_tile_max_pairs are exclusive with floor_spec")
        if judge_size_u_floor is None:
            raise ValueError("judge_tile_max_pairs needs judge_size_u_floor")
        if isinstance(judge_size_u_floor, bool) or not isinstance(judge_size_u_floor, (int, float)) or not 0 <= judge_size_u_floor <= 1:
            raise ValueError("judge_size_u_floor in [0,1] required")
        if judge_tile_max_pairs is not None and (type(judge_tile_max_pairs) is not int or judge_tile_max_pairs < 0):
            raise ValueError("judge_tile_max_pairs must be an integer >= 0")
        size = {"u_floor": float(judge_size_u_floor), "enabled": True, "max_pairs": JUDGE_TILE_MAX_PAIRS if judge_tile_max_pairs is None else judge_tile_max_pairs,
                **JUDGE_SIZE_FIXED}
    elif spec_judge is not None and spec_judge.get("size") is not None:
        size = dict(spec_judge["size"])  # 증분: 설정만 고정하고 stats 는 이 라운드 쌍으로 다시 센다
    if size is not None and (factor != JUDGE_SIZE_FIXED["factor"] or tolerance != JUDGE_SIZE_FIXED["tolerance"]):
        raise ValueError(f"the size judge needs factor {JUDGE_SIZE_FIXED['factor']} and tolerance {JUDGE_SIZE_FIXED['tolerance']}")
    if spec_judge is not None:
        if margin is not None and margin != spec_judge["margin"]:
            raise ValueError("margin differs from the base round summary")
        margin = spec_judge["margin"]
    if margin is None:
        margin = 0.05
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
    if incremental and judge == "three-axis" and floor_spec is None:
        raise ValueError("three-axis incremental screening needs --floor-json (round 1 criteria)")
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
    pairs, shifts = [], {}  # shifts: 판단 지표 위치(metrics_anchor)의 래스터 이동. tile 정렬이 쓴다
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
                shifts[(a, b)] = align["shift"] if on_layer else result["shift"]
                rows = pair_metric_rows(signatures[a], signatures[b], shifts[(a, b)], tolerance=tolerance,
                                        presence_fraction=presence_fraction, use_layers=on_layer)
                pairs[-1]["metric_rows"] = rows
                pairs[-1].update(pair_metric_columns(rows, columns["metrics_anchor"]))
                pairs[-1]["judge_axes"] = pair_axes(rows, columns["metrics_anchor"], signatures[a], signatures[b])
    if incremental:
        new_of = {part: sorted(set(split[part]) - old_of[part]) for part in partitions}
        expected = {part: len(new_of[part]) * (len(new_of[part]) - 1) // 2 + len(new_of[part]) * len(old_of[part]) for part in partitions}
        counts = {part: sum(p["partition"] == part for p in pairs) for part in partitions}
        if counts != expected:
            raise ValueError(f"delta pair count {counts} differs from the expected n(n-1)/2 + n*m {expected}")
    same = [p for p in pairs if p["same_lineage"]]
    pick = make_pick(floor_quantile)
    axes_floor = None
    if judge == "legacy":
        global_floor, group_floor = judge_pairs_legacy(pairs, group_of, floor=floor, floor_spec=floor_spec, pick=pick, margin=margin, min_group_pairs=min_group_pairs)
        strength, strength_floor = (lambda p: p["similarity"]), "floor"
    else:  # 이 뒤의 cap·weakest·review 는 similarity 대신 P 와 P 기준값을 쓴다
        global_floor, group_floor, axes_floor = judge_pairs_three_axis(pairs, group_of, floor=floor, floor_spec=floor_spec, pick=pick, margin=margin,
                                                                       min_group_pairs=min_group_pairs, judge_floor_p=judge_floor_p, judge_floor_u=judge_floor_u,
                                                                       size_u_floor=None if size is None else size["u_floor"])
        if size is not None:
            judge_tile_stage(pairs, signatures, shifts, size, margin)
        strength, strength_floor = (lambda p: p["axis_p_eff"]), "judge_p_floor"
    within_different = sum(p["tier"] == "different" and p["group_a"] == p["group_b"] for p in pairs)
    allowed = math.ceil(cross_group_fraction * within_different)
    cross = sorted((p for p in pairs if p["tier"] == "different" and p["group_a"] != p["group_b"]), key=lambda p: (-strength(p), p["product_a"], p["product_b"]))
    for p in cross[allowed:]:
        p["tier"], p["reason"] = "unknown", "cross_group_capped"  # 쉬운 쌍이라 일부만 쓴다
        if judge == "three-axis":
            p["judge_case"] = None  # case 2 는 최종 different 만 센다. cap 으로 떨어진 쌍(모두 case 2 였다)은 judge.cases.case_2_capped 로 센다
    ranked = same if judge == "legacy" else [p for p in same if p["axis_p"] is not None]
    weakest = sorted(ranked, key=strength)[:max(1, math.ceil(0.2 * len(ranked)))] if ranked else []
    for p in weakest:
        p["flag"] = "weakest_same_lineage"
    for p in pairs:  # 사람이 봐야 하는 쌍과 그 이유를 모두 모은다
        p["review"] = []
        if p["tier"] == "unknown" and p["reason"] in ({"close_to_same_lineage", "near_boundary"} if judge == "legacy" else JUDGE_REVIEW_REASONS):
            p["review"].append(p["reason"])
        if p.get("flag") == "weakest_same_lineage":
            p["review"].append("weakest_same_lineage")
        if p["tier"] == "different":
            if judge == "legacy":
                near = strength(p) >= p[strength_floor] - margin - review_band
            else:  # three-axis 는 판정 경계와 같은 십진 정확 비교를 쓴다(표시용)
                exact = lambda value: Fraction(repr(float(value)))
                near = exact(strength(p)) >= exact(p[strength_floor]) - exact(margin) - exact(review_band)
            if near:
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
    if judge == "three-axis":
        summary["judge"] = judge_summary(pairs, axes_floor, margin, floor_quantile, spec_judge, size)
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
        writer = csv.DictWriter(stream, SCORE_FIELDS + (JUDGE_SCORE_FIELDS if "judge" in result["summary"] else ())
                                + (JUDGE_SIZE_SCORE_FIELDS if "size" in result["summary"].get("judge", {}) else ()), extrasaction="ignore")
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
    ap.add_argument("--margin", type=float, default=None, help="a cross-lineage pair is 'different' only this far below the weakest same-lineage similarity (default 0.05; "
                    "a three-axis incremental run uses the margin stored in the base summary and rejects a different value)")
    ap.add_argument("--judge", choices=JUDGE_MODES, default=None, help="judging mode: legacy (similarity) or three-axis (pattern P, usage U, size). Omitted = legacy; "
                    "with --floor-json it inherits the base summary's mode and a different explicit value is rejected")
    ap.add_argument("--judge-floor-p", type=float, help="three-axis: explicit pattern (P) floor instead of calibrating it on the same-lineage pairs (exclusive with --floor-json)")
    ap.add_argument("--judge-floor-u", type=float, help="three-axis: explicit usage (U) floor instead of calibrating it on the same-lineage pairs (exclusive with --floor-json)")
    ap.add_argument("--judge-size-u-floor", type=float, help="three-axis: usage (U) floor in [0,1] for pairs whose unit size differs beyond the P-trusted range; "
                    "below it the pair is 'different' (factor 8 and tolerance 1 only; exclusive with --floor-json, which carries the setting from round 1)")
    ap.add_argument("--judge-tile-max-pairs", type=int, help=f"three-axis: most size-differing pairs to tile-align, integer >= 0 (default {JUDGE_TILE_MAX_PAIRS}; needs --judge-size-u-floor)")
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
    for flag, value in (("--review-pairs", a.review_pairs), ("--review-layers", a.review_layers)):
        if value < 0:
            ap.error(f"{flag} must be 0 or more")
    if a.floor_json is not None and a.floor is not None:
        ap.error("--floor and --floor-json are exclusive")
    if a.floor_json is not None and (a.judge_floor_p is not None or a.judge_floor_u is not None):
        ap.error("--judge-floor-p and --judge-floor-u are exclusive with --floor-json")
    if a.floor_json is not None and (a.judge_size_u_floor is not None or a.judge_tile_max_pairs is not None):
        ap.error("--judge-size-u-floor and --judge-tile-max-pairs are exclusive with --floor-json")
    if a.judge == "legacy" and (a.judge_size_u_floor is not None or a.judge_tile_max_pairs is not None):
        ap.error("--judge-size-u-floor and --judge-tile-max-pairs need --judge three-axis")
    if a.judge_size_u_floor is not None and not 0 <= a.judge_size_u_floor <= 1:  # NaN 도 여기서 걸린다
        ap.error("--judge-size-u-floor must be in [0,1]")
    if a.judge_tile_max_pairs is not None and (a.judge_tile_max_pairs < 0 or a.judge_size_u_floor is None):
        ap.error("--judge-tile-max-pairs must be 0 or more and needs --judge-size-u-floor")
    if a.floor_quantile is not None:
        if a.floor is not None or a.floor_json is not None:
            ap.error("--floor-quantile is exclusive with --floor and --floor-json")
        if not 0 <= a.floor_quantile <= 1:  # NaN 도 여기서 걸린다
            ap.error("--floor-quantile must be in [0,1]")
    if (a.base_split is None) != (a.base_relations is None):
        ap.error("--base-split and --base-relations go together")
    if a.base_split is not None and a.floor is None and a.floor_json is None:
        ap.error("incremental screening needs exactly one of --floor and --floor-json")
    if a.base_split is not None and a.judge == "three-axis" and a.floor_json is None:
        ap.error("three-axis incremental screening needs --floor-json (round 1 criteria)")
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
    for path in (out, scores, bindings, review, summary):  # 5개 모두 screen() 전에 검사한다
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}; choose a new path")
    result = screen(a.manifest, split, groups=a.groups, partitions=tuple(a.partitions), factor=a.factor, tolerance=a.tolerance, margin=a.margin,
                    floor=a.floor, cross_group_fraction=a.cross_group_fraction, anchor=a.anchor, max_shift=a.max_shift, candidates=a.candidates, review_band=a.review_band,
                    align_min_overlap=a.align_min_overlap, align_min_support=a.align_min_support, align_min_dice=a.align_min_dice, presence_fraction=a.presence_fraction, floor_quantile=a.floor_quantile,
                    judge=a.judge, judge_floor_p=a.judge_floor_p, judge_floor_u=a.judge_floor_u,
                    judge_size_u_floor=a.judge_size_u_floor, judge_tile_max_pairs=a.judge_tile_max_pairs, **extra)
    write_csvs(result, out, scores, bindings)
    # 쓰기 순서: bindings·scores → relations → review → summary(필수 3개는 마지막이라 중단되면 부분 출력 검사가 잡는다)
    page = render_review(result, a.review_pairs, a.review_bindings, a.review_layers)
    review.write_text(page, encoding="utf-8")
    result["summary"]["review"]["html_bytes"] = review.stat().st_size
    summary.write_text(json.dumps(result["summary"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"relations": str(out), "scores": str(scores), "review": str(review), "summary_file": str(summary), **result["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
