"""dl1-judge/T2: unit 크기 차(ΔU)가 판정 축 P(axis_p)를 얼마나 깎는지 합성 제품으로 잰다.

합성 제품: 프레임 = 2×2 unit + 사방 MARGIN px(0.1 µm/px). 선폭(CD)은 고정이고 배선 길이·간격만 unit 크기 U 에 비례해 늘어난다
(rescale 하지 않는다). 공개 함수(product_signature, compare, layer_columns, pair_metric_rows, pair_axes)만 쓴다.

실행: PYTHONPATH=$PWD/src python scripts/measure_unit_stretch.py [--unit 1000] [--phases 0-7] [--quick]
dl1-size/T4·T4b: --tile 이면 tile 정렬 합성 게이트(같은 설계·거짓 일치 B′·지원 범위)를 잰다.
"""
import argparse
import math
import sys
import time
from fractions import Fraction

import numpy as np

from psdl1 import screen_align, screen_tile
from psdl1.schema import Element, Product
from psdl1.screen_relations import (block_reduce, compare, judge_verdict, layer_columns, layer_item, pair_axes, pair_metric_rows, product_signature,
                                    tile_inputs)

UNIT = 1000
MARGIN = 500
CD = 12  # 선폭(px). unit 크기와 무관하다
FACTOR = 8
TOLERANCE = 1
MARGIN_P = 0.05  # three-axis 판정 margin 기본값(PLAN v1.1)
JUDGE_SIZE_SUPPORT_PX = 7  # PLAN v1.2 G-size 지원 범위(T2 측정 근거)
GATE_DELTAS = tuple(range(8))
RECORD_DELTAS = (8, 12, 20, 40)
AXES = ("x", "y", "xy")
PATHS = ("anchor", "layer")
LAYERS = ("M1", "M2", "M3")


def fill(mask, y0, y1, x0, x1, value=True):
    mask[max(0, int(y0)):max(0, int(y1)), max(0, int(x0)):max(0, int(x1))] = value


# unit 안 배선·판·점의 위치 비율. 기본이 A 설계이고 MIXED 는 같은 요소를 다른 비율에 놓은 다른 패턴이다.
POS = {"gate_y": (0.2, 0.6), "stub_x": 0.5, "data_x": (0.3, 0.7), "pixel": ((0.4, 0.5, 100), (0.8, 0.8, 60)), "cnt": ((0.2, 0.3), (0.6, 0.7))}
MIXED = {"gate_y": (0.35, 0.75), "stub_x": 0.35, "data_x": (0.2, 0.8), "pixel": ((0.65, 0.3, 100), (0.25, 0.6, 60)), "cnt": ((0.45, 0.2), (0.8, 0.45))}
# dl1-size/T4d(R1 adv:1): P_var0 가 낮은 쪽(0.15–0.40)에 퍼지는 변형. keep_pixel = 배선만 다른 패턴(판은 같다), dense = 배선 수·간격이 다르다, cutdrift = 국소 이동 + 배선 일부 삭제
KEEP_PIXEL = {**MIXED, "pixel": POS["pixel"], "cnt": POS["cnt"]}
DENSE = {**POS, "gate_y": (0.12, 0.34, 0.56, 0.78), "data_x": (0.18, 0.4, 0.62, 0.84), "stub_x": 0.62}
# dl1-size/T8(R2 adv:2): P_var0 가 0.25–0.30 인 변형. half_pixel = MIXED 인데 판 두 개 중 하나만 A 와 같은 자리에 둔다(keep_pixel 은 둘 다 같다)
HALF_PIXEL = {**MIXED, "pixel": (POS["pixel"][0], MIXED["pixel"][1])}
POS_OF = {"mixed": MIXED, "keep_pixel": KEEP_PIXEL, "dense": DENSE, "half_pixel": HALF_PIXEL}
VARIANTS = ("mixed", "half", "drift", "seam", "keep_pixel", "dense", "cutdrift", "half_pixel")  # 거짓 일치 검증의 다른 패턴 B′ 변형(build 의 variant 인수)
DRIFT_BLOCK, DRIFT_REACH, DRIFT_STEP = 128, 16, 8  # px. 국소 이동은 block 마다 ±2 칸(16 px) 안에서 8 px(1 칸)씩 누적된다


def draw_unit(m, oy, ox, uy, ux, pos=POS):
    """unit 하나(원점 oy, ox · 크기 uy × ux)의 배선. 위치는 unit 크기의 비율이고 선폭·패드는 고정이다."""
    r = lambda f, size: int(round(f * size))
    for f in pos["gate_y"]:  # gate/M1: 가로 배선 + 왼쪽 패드
        y = oy + r(f, uy)
        fill(m["gate/M1"], y, y + CD, ox + r(0.1, ux), ox + r(0.9, ux))
        fill(m["gate/M1"], y - 14, y + CD + 14, ox + r(0.1, ux), ox + r(0.1, ux) + 40)
    x = ox + r(pos["stub_x"], ux)  # gate/M2: 세로 stub
    fill(m["gate/M2"], oy + r(0.2, uy), oy + r(0.8, uy), x, x + CD)
    for f in pos["data_x"]:  # data/M2: 세로 배선 + 위쪽 패드
        x = ox + r(f, ux)
        fill(m["data/M2"], oy + r(0.1, uy), oy + r(0.9, uy), x, x + CD)
        fill(m["data/M2"], oy + r(0.1, uy), oy + r(0.1, uy) + 40, x - 14, x + CD + 14)
    for (fy, fx, size) in pos["pixel"]:  # pixel/M3: 고정 크기 판
        y, x = oy + r(fy, uy), ox + r(fx, ux)
        fill(m["pixel/M3"], y, y + size, x, x + size)
    for (fy, fx) in pos["cnt"]:  # CNT: 고정 크기 점
        y, x = oy + r(fy, uy), ox + r(fx, ux)
        fill(m["cnt"], y - 4, y + 56, x - 4, x + 16)


def draw_half(m, oy, ox, uy, ux):
    """A 설계를 unit 안에서 반 unit 순환 이동한 다른 패턴(배선이 unit 경계에서 접혀 이어진다)."""
    tmp = {k: np.zeros((uy, ux), dtype=bool) for k in m}
    draw_unit(tmp, 0, 0, uy, ux)
    for k in m:
        m[k][oy:oy + uy, ox:ox + ux] |= np.roll(tmp[k], (uy // 2, ux // 2), axis=(0, 1))


def seam(m, oy, ox, uy, ux):
    """이음새 변형: unit 사이 버스를 3배 굵게 하고, 버스를 두 곳에서 끊는다(선폭과 연결이 A 와 다르다)."""
    fill(m["gate/M1"], oy + uy - 18, oy + uy + 18, ox, ox + 2 * ux)
    fill(m["data/M2"], oy, oy + 2 * uy, ox + ux - 18, ox + ux + 18)
    for f in (0.5, 1.5):
        fill(m["gate/M1"], oy + uy - 20, oy + uy + 20, ox + int(f * ux) - 12, ox + int(f * ux) + 12, False)
        fill(m["data/M2"], oy + int(f * uy) - 12, oy + int(f * uy) + 12, ox + ux - 20, ox + ux + 20, False)


def drift(m, seed=5):
    """국소 이동 누적 변형: block 마다 모든 층을 같은 (dy, dx) 만큼 옮긴다. 이동은 이웃 block 과 한 걸음(1 칸)씩만 달라 ±2 칸을 넘지 않는다."""
    rng = np.random.RandomState(seed)
    shape = next(iter(m.values())).shape
    padded = {k: np.pad(v, DRIFT_REACH) for k, v in m.items()}
    out = {k: np.zeros_like(v) for k, v in m.items()}
    dy = dx = 0
    for y0 in range(0, shape[0], DRIFT_BLOCK):
        for x0 in range(0, shape[1], DRIFT_BLOCK):
            dy = int(np.clip(dy + DRIFT_STEP * rng.randint(-1, 2), -DRIFT_REACH, DRIFT_REACH))
            dx = int(np.clip(dx + DRIFT_STEP * rng.randint(-1, 2), -DRIFT_REACH, DRIFT_REACH))
            y1, x1 = min(y0 + DRIFT_BLOCK, shape[0]), min(x0 + DRIFT_BLOCK, shape[1])
            for k in m:
                out[k][y0:y1, x0:x1] = padded[k][DRIFT_REACH + y0 + dy:DRIFT_REACH + y1 + dy, DRIFT_REACH + x0 + dx:DRIFT_REACH + x1 + dx]
    return out


def cut(m, oy, ox, uy, ux):
    """배선 일부 삭제: gate 층(M1·M2)은 왼쪽 위 unit 만, data/M2 와 판은 오른쪽 아래 unit 만 남기고 지운다(binding 마다 A 의 4분의 1만 남는다)."""
    for i in range(2):
        for j in range(2):
            y0, x0 = oy + i * uy, ox + j * ux
            for key, keep in (("gate/M1", (i, j) == (0, 0)), ("gate/M2", (i, j) == (0, 0)), ("data/M2", (i, j) == (1, 1)), ("pixel/M3", (i, j) == (1, 1))):
                if not keep:
                    fill(m[key], y0, y0 + uy, x0, x0 + ux, False)


def build(name, uy, ux, offset=0, variant=None):
    """2×2 unit 제품. offset 은 내용을 프레임 안에서 (offset, offset) px 옮긴다(프레임 크기는 그대로 2*U+2*MARGIN).
    variant: None = A 설계, VARIANTS = 다른 패턴 B′(같은 binding 집합을 쓴다)."""
    shape = (2 * uy + 2 * MARGIN, 2 * ux + 2 * MARGIN)
    m = {k: np.zeros(shape, dtype=bool) for k in ("gate/M1", "gate/M2", "data/M2", "pixel/M3", "cnt")}
    oy, ox = MARGIN + offset, MARGIN + offset
    for i in range(2):
        for j in range(2):
            if variant == "half":
                draw_half(m, oy + i * uy, ox + j * ux, uy, ux)
            else:
                draw_unit(m, oy + i * uy, ox + j * ux, uy, ux, POS_OF.get(variant, POS))
    fill(m["gate/M1"], oy + uy - 6, oy + uy + 6, ox, ox + 2 * ux)  # unit 사이 가로 버스: ΔU 만큼 길어진다
    fill(m["data/M2"], oy, oy + 2 * uy, ox + ux - 6, ox + ux + 6)  # unit 사이 세로 버스
    if variant == "seam":
        seam(m, oy, ox, uy, ux)
    if variant in ("drift", "cutdrift"):
        m = drift(m)
    if variant == "cutdrift":
        cut(m, oy, ox, uy, ux)
    elements = tuple(Element(*key.split("/"), m[key]) for key in ("gate/M1", "gate/M2", "data/M2", "pixel/M3"))
    product = Product(name, (shape[0] * 0.1, shape[1] * 0.1), elements, m["cnt"].astype(np.float32), lineage_id="L", design_version="v")
    layer_masks = {"m1": m["gate/M1"], "m2": m["gate/M2"] | m["data/M2"], "m3": m["pixel/M3"]}
    return product, layer_masks


def signature(name, uy, ux, offset=0, variant=None):
    product, layer_masks = build(name, uy, ux, offset, variant)
    sig = product_signature(product, FACTOR, TOLERANCE)
    rasters = {k: block_reduce(v, FACTOR, "any") for k, v in layer_masks.items()}
    sig["layers"] = {k: layer_item(rasters[k], screen_align.pack_mask(layer_masks[k]), sig["full_shape"], TOLERANCE) for k in layer_masks}
    return sig, rasters


def measure_pair(sa, rasters_a, sb, rasters_b, path):
    """정렬 경로 하나로 쌍을 비교하고 판정 축·유사도·Dice 를 돌려준다. path: anchor = 기존 compare 위치, layer = Layer 사이드카 정렬."""
    result = compare(sa, sb, tolerance=TOLERANCE)
    out = {"similarity": result["similarity"], "layer_applied": None}
    if path == "layer":
        align, columns = layer_columns(sa, sb, result, rasters_a, rasters_b, max_shift=0.35, min_overlap=screen_align.ALIGN_MIN_OVERLAP,
                                       min_support=screen_align.ALIGN_MIN_SUPPORT, min_dice=screen_align.ALIGN_MIN_DICE)
        on_layer = columns["metrics_anchor"] == "LAYER"
        out["layer_applied"] = on_layer
        out["similarity"] = columns["similarity_layer"]
    else:
        on_layer = False
    anchor = "LAYER" if on_layer else result["anchor"]
    shift = align["shift"] if on_layer else result["shift"]
    rows = pair_metric_rows(sa, sb, shift, tolerance=TOLERANCE, use_layers=on_layer)
    dice = [r["dice"] for r in rows if r["kind"] == "binding" and r["status"] == "both" and r["dice"] is not None]
    axes = pair_axes(rows, anchor, sa, sb)
    out.update(axes, dice=round(float(np.mean(dice)), 6) if dice else None, shift=tuple(int(v) for v in shift), anchor=anchor)
    return out


def supported(anchor, size_delta_y, size_delta_x):
    """PLAN v1.2 지원 범위: metrics_anchor 가 LAYER 이고 max|size_delta_unit_px| ≤ JUDGE_SIZE_SUPPORT_PX."""
    return anchor == "LAYER" and max(abs(size_delta_y), abs(size_delta_x)) <= JUDGE_SIZE_SUPPORT_PX


def tier(p, p_floor, margin=MARGIN_P, size_tag="same", in_support=True):
    """PLAN v1.1 판정표(다른 계보 쌍의 초기 tier, P 만 본다) + v1.2 G-size: P < p_floor − margin → different, 그 밖은 unknown. P 없음 → unknown.
    size_differs 이고 지원 범위 밖이면 different 대신 unknown(size_differs_out_of_support)."""
    if p is None:
        return "unknown", "no_pattern_value"
    exact = lambda value: Fraction(repr(float(value)))  # src judge_verdict 와 같은 십진 정확 비교(0.52 - 0.05 = 0.47000000000000003 오차 방지)
    if exact(p) < exact(p_floor) - exact(margin):
        if size_tag == "size_differs" and not in_support:
            return "unknown", "size_differs_out_of_support"
        return "different", "below_pattern_floor"
    return "unknown", "close_to_same_lineage" if exact(p) >= exact(p_floor) else "near_boundary"


def classify(records, base):
    """base = {(경로, 위상): ΔU=0 의 axis_p}. 레코드마다 p0, 지원 범위 안/밖, tier 를 채운다."""
    for r in records:
        r["p0"] = base.get((r["path"], r["phase"]))
        r["in_support"] = supported(r["anchor"], r["size_delta_unit_px_y"], r["size_delta_unit_px_x"])
        r["tier"], r["reason"] = (tier(r["axis_p"], r["p0"], size_tag=r["size_tag"], in_support=r["in_support"]) if r["p0"] is not None
                                  else ("unknown", "no_baseline"))


def gate_failures(records):
    """(a) 지원 범위 안(size_differs 가 아니거나 LAYER 이고 |ΔU| ≤ 7): axis_p ≥ P0 − 0.05 이고 tier ≠ different. (b) 밖: tier ≠ different."""
    bad = []
    for r in records:
        inside = r["size_tag"] == "same" or r["in_support"]
        if r["axis_p"] is None or r["p0"] is None or r["tier"] == "different" or (inside and r["axis_p"] < r["p0"] - MARGIN_P):
            bad.append(r)
    return bad


def delta_pair(delta, axis):
    """축 이름과 ΔU 에서 (ΔU_y, ΔU_x). x·y·xy 는 한 축 또는 같은 값이고, "r0.50" 같은 비율 축은 y 가 ΔU 이고 x 가 ceil(비율·ΔU)이다(올림이라 실제 비율이 이름의 비율보다 작아지지 않는다).
    그 밖(ΔU=0 의 "-")은 (0, 0)."""
    if not delta:
        return 0, 0
    if axis.startswith("r"):
        return delta, math.ceil(Fraction(repr(float(axis[1:]))) * delta)
    return (delta if "y" in axis else 0), (delta if "x" in axis else 0)


def dims(unit, delta, axis):
    dy, dx = delta_pair(delta, axis)
    return unit + dy, unit + dx


def measure(unit=UNIT, phases=range(8), deltas=GATE_DELTAS, axes=AXES, paths=PATHS, progress=None):
    """레코드 목록. A 는 ΔU=0 · offset 0 제품이고 B 가 (ΔU, 축, 위상)마다 바뀐다. 같은 (경로, 위상)의 ΔU=0 쌍이 기준(P0)이다."""
    sa, ra = signature("A", unit, unit, 0)
    records = []
    for delta in deltas:
        for axis in (AXES[:1] if delta == 0 else axes):
            uy, ux = dims(unit, delta, axis)
            for phase in phases:
                sb, rb = signature("B", uy, ux, phase)
                for path in paths:
                    rec = measure_pair(sa, ra, sb, rb, path)
                    rec.update(delta=delta, axis="-" if delta == 0 else axis, phase=phase, path=path)
                    records.append(rec)
                if progress:
                    progress(delta, axis, phase)
    base = {(r["path"], r["phase"]): r["axis_p"] for r in records if r["delta"] == 0}
    classify(records, base)
    return records


def fmt(v):
    return "—" if v is None else f"{v:.4f}"


def table(records):
    lines = ["| ΔU px | 축 | 경로 | n 위상 | P min | P max | P0 min | (P−P0) min | 유사도 min | Dice min | size_delta_unit_px (y,x) | layer 적용 | 지원 범위 | tier(최악, G-size 포함) | tier(G-size 없이) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    keys = sorted({(r["delta"], r["axis"], r["path"]) for r in records}, key=lambda k: (k[0], k[1], k[2]))
    for delta, axis, path in keys:
        group = [r for r in records if (r["delta"], r["axis"], r["path"]) == (delta, axis, path)]
        ps = [r["axis_p"] for r in group if r["axis_p"] is not None]
        diffs = [r["axis_p"] - r["p0"] for r in group if r["axis_p"] is not None and r["p0"] is not None]
        sizes = {(r["size_delta_unit_px_y"], r["size_delta_unit_px_x"]) for r in group}
        applied = "" if path == "anchor" else f"{sum(bool(r['layer_applied']) for r in group)}/{len(group)}"
        worst = "different" if any(r["tier"] == "different" for r in group) else "unknown"
        raw = "different" if any(tier(r["axis_p"], r["p0"])[0] == "different" for r in group if r["p0"] is not None) else "unknown"
        inside = "안" if all(r["size_tag"] == "same" or r["in_support"] for r in group) else "밖"
        mn = lambda key: fmt(min((r[key] for r in group if r[key] is not None), default=None))
        lines.append(f"| {delta} | {axis} | {path} | {len(group)} | {fmt(min(ps, default=None))} | {fmt(max(ps, default=None))} | {mn('p0')} | "
                     f"{fmt(min(diffs, default=None))} | {mn('similarity')} | {mn('dice')} | {', '.join(str(s) for s in sorted(sizes))} | {applied} | {inside} | {worst} | {raw} |")
    return "\n".join(lines)


# ---- dl1-size/T4: tile 합성 게이트 --------------------------------------------------------------------------------------
TILE_UNITS = (552, 560, 1000)  # 552 = 실자료(ETX r01) 최소 U
TILE_DELTAS = (8, 12, 20, 30, 38, 40, 41, 42, 45, 50, 51, 60)  # 41–50 은 두 축이 함께 늘어난 쌍(xy·비율 축)만 지원, 한 축만 늘어난 쌍과 51 이상은 out_of_support 여야 한다(dl1-size/T9)
TILE_PHASES = (0, 4)
FALSE_DELTAS = (8, 20, 30, 38, 40, 42, 45, 50, 60)
FALSE_AXES = ("x", "xy")
XY_DELTAS = (42, 45, 50)  # 40 위 xy 구간의 비율 축 ΔU
GATE_RATIOS = (0.9, screen_tile.XY_RATIO_MIN, round(screen_tile.XY_RATIO_MIN - 0.05, 2))  # 비율 1.0 은 axis "xy". 마지막은 지원 비율 바로 아래(out_of_support 여야 한다)
FALSE_RATIOS = (screen_tile.XY_RATIO_MIN,)  # (b): 지원 비율 끝에서 한 번 더 본다
SWEEP_RATIOS = tuple(round(1 - 0.05 * k, 2) for k in range(20))  # --xy-sweep: XY_RATIO_MIN 을 정하는 표(1.0 에서 0.05 까지 0.05 간격, 내림차순)
JUDGE_P_FLOOR, JUDGE_U_FLOOR, JUDGE_SIZE_U_FLOOR = 0.45, 0.9, 0.75  # 실제 screen() 실행 인수와 같다
FALSE_P = 0.35  # P_var0(ΔU=0 에서 같은 변형을 단일 이동으로 잰 unit 영역 P)가 이보다 낮은 변형은 모든 ΔU 에서 tier == different 또는 tile_p None 이어야 한다
LOW_U = 0.5  # 크기 기준(0.75)보다 낮은 U: 혼합 판정은 이 쌍도 tile 을 돌리므로 (b) 에서 같은 tile_p 로 U 가 낮은 판정도 본다
RECORD_P = 0.40  # P_var0 가 FALSE_P–RECORD_P 이면 기록만 한다
FALSE_SLACK = 0.07  # tile_p ≤ P_var0 + FALSE_SLACK: tile 정렬이 원래 차이보다 더 맞추지 않는다. 0.05 는 ΔU 8 에서도 mixed 가 +0.061 이라 못 지킨다(반경과 무관한 우연 수준 변동, 아래 표)
SHIFT_TOL_CELLS = 16  # 입력 이동(compare·layer 정렬)이 참값에서 이만큼 넘게 어긋난 쌍(주기 구조에서 한 unit 옆에 맞춘 경우)은 (a) 에서 tile 성공을 요구하지 않는다


def tile_pair(sa, ra, sb, rb, path, shift=None):
    """measure_pair + 같은 이동으로 tile 정렬. tile_p·단일 unit 영역 P·시간·실제 판정 헬퍼 결과·같은 조합 2회 동일 여부를 채운다.
    shift 를 주면 compare·layer 정렬이 낸 이동(중심 정렬 기준) 대신 그 값을 tile 단계에 넘긴다(반복 unit 오인 반례용)."""
    out = measure_pair(sa, ra, sb, rb, path)
    if shift is not None:
        out["shift"] = tuple(int(v) for v in shift)
    sources, source, keys = tile_inputs(sa, sb, out["anchor"])
    start = time.perf_counter()
    origin = screen_tile.origin_shift(sa, sb, out["shift"])  # screen() 의 judge_tile_stage 와 같은 변환(중심 정렬 이동 → 원점 이동)
    first = screen_tile.tile_pattern_axis(sa, sb, origin, sources, keys, factor=FACTOR, tolerance=TOLERANCE)
    out["origin_shift"] = origin
    out["seconds"] = time.perf_counter() - start
    out["repeat_equal"] = first == screen_tile.tile_pattern_axis(sa, sb, origin, sources, keys, factor=FACTOR, tolerance=TOLERANCE)
    out.update(first)
    out["tile_source"], out["n_layers"] = source, len(sources)
    out["single_p"] = screen_tile.single_shift_unit_p(sa, sb, origin, keys)
    # 단일 P 가 낮아 tile 로 넘어온 쌍의 판정(실제 judge_verdict). 단일 P 가 이미 높은 쌍은 tile 을 거치지 않으므로 P 를 0 으로 눌러 tile 행을 강제로 본다.
    # T4d 혼합 판정: tile 단계 판정(tiled=True). U 가 크기 기준보다 낮은 쌍도 tile 을 돌리므로 U 를 낮춘 판정(tier_low_u)도 함께 잰다.
    forced = {**out, "axis_p": 0.0}
    out["tier"], out["reason"], _ = judge_verdict(forced, out["anchor"], False, JUDGE_P_FLOOR, JUDGE_U_FLOOR, MARGIN_P, size_u_floor=JUDGE_SIZE_U_FLOOR, tile_p=out["tile_p"],
                                                  tiled=True)
    low = {**forced, "axis_u": LOW_U}
    out["tier_low_u"], out["reason_low_u"], _ = judge_verdict(low, out["anchor"], False, JUDGE_P_FLOOR, JUDGE_U_FLOOR, MARGIN_P, size_u_floor=JUDGE_SIZE_U_FLOOR,
                                                              tile_p=out["tile_p"], tiled=True)
    return out


def ratio_axis(ratio):
    return "xy" if ratio == 1.0 else f"r{ratio:.2f}"


def axes_for(delta, axes, ratios):
    """ΔU 의 축 목록: ΔU=0 은 하나, 40 위 xy 구간 ΔU(XY_DELTAS)는 비율 축을 더한다."""
    if delta == 0:
        return AXES[:1]
    return tuple(axes) + (tuple(ratio_axis(r) for r in ratios) if delta in XY_DELTAS else ())


def tile_measure(units=TILE_UNITS, deltas=TILE_DELTAS, phases=TILE_PHASES, axes=AXES, paths=PATHS, progress=None, ratios=GATE_RATIOS):
    """(a) 같은 설계. U·ΔU·축·위상·경로마다 tile_p 를 재고, 같은 (U, 경로, 위상)의 ΔU=0 쌍 tile_p 를 P0 로 붙인다. 40 위 xy 구간 ΔU 는 비율 축(ratios)을 더한다."""
    records = []
    for unit in units:
        sa, ra = signature("A", unit, unit, 0)
        for delta in (0,) + tuple(deltas):
            for axis in axes_for(delta, axes, ratios):
                uy, ux = dims(unit, delta, axis)
                for phase in phases:
                    sb, rb = signature("B", uy, ux, phase)
                    for path in paths:
                        rec = tile_pair(sa, ra, sb, rb, path)
                        rec.update(unit=unit, delta=delta, axis="-" if delta == 0 else axis, phase=phase, path=path, variant=None)
                        rec["shift_error"] = max(abs(rec["origin_shift"][ax] + phase / FACTOR) for ax in (0, 1))  # 합성 B 는 내용이 phase px 아래·오른쪽으로 밀려 있어 참 원점 이동은 −phase/8 칸
                        records.append(rec)
                    if progress:
                        progress("same", unit, delta, axis, phase)
    base = {(r["unit"], r["path"], r["phase"]): r["tile_p"] for r in records if r["delta"] == 0}
    for r in records:
        r["p0"] = base.get((r["unit"], r["path"], r["phase"]))
    return records


def false_measure(units=TILE_UNITS, deltas=FALSE_DELTAS, variants=VARIANTS, axes=FALSE_AXES, paths=PATHS, progress=None, ratios=FALSE_RATIOS):
    """(b) 거짓 일치. 같은 binding 집합을 쓰는 다른 패턴 B′ 와 A 를 잰다. 같은 (U, 변형, 경로)의 ΔU=0 쌍 단일 이동 unit 영역 P 를 p_var0 으로 붙인다
    (다른 정도의 기준: 늘어난 상태의 단일 P 는 늘어남 때문에 낮아서 쓰지 않는다). 40 위 xy 구간 ΔU 는 비율 축(ratios)을 더한다."""
    records = []
    for unit in units:
        sa, ra = signature("A", unit, unit, 0)
        for variant in variants:
            for delta in (0,) + tuple(deltas):
                for axis in axes_for(delta, axes, ratios):
                    uy, ux = dims(unit, delta, axis)
                    sb, rb = signature("B", uy, ux, 0, variant)
                    for path in paths:
                        rec = tile_pair(sa, ra, sb, rb, path)
                        rec.update(unit=unit, delta=delta, axis="-" if delta == 0 else axis, phase=0, path=path, variant=variant, p0=None)
                        records.append(rec)
                    if progress:
                        progress(variant, unit, delta, axis, 0)
    base = {(r["unit"], r["variant"], r["path"]): r["single_p"] for r in records if r["delta"] == 0}
    best = {}
    for (unit, variant, path), p in base.items():
        if p is not None:
            best[(unit, variant)] = max(best.get((unit, variant), p), p)
    for r in records:
        r["p_var0"] = base.get((r["unit"], r["variant"], r["path"]))
        r["p_var0_best"] = best.get((r["unit"], r["variant"]))  # 두 경로 중 높은 단일 P: anchor 이동 오차로 낮아진 값 대신 변형의 정렬된 수준(T8, half_pixel)
    return records


SUPPORT_DELTA = screen_tile.MAX_DELTA_U_PX
SUPPORT_UNITS = (int(SUPPORT_DELTA * screen_tile.TILE_DIVISOR * screen_tile.TILE_MIN_CELLS), int(SUPPORT_DELTA * screen_tile.TILE_DIVISOR * screen_tile.TILE_MIN_CELLS) - 8)
# tile 폭 = U // (TILE_DIVISOR·ΔU) 칸: 최소 폭이면 ok, 한 칸(8 px)이 모자라면 too_small


XY_SUPPORT_DELTA = screen_tile.XY_SUPPORT_PX
XY_SUPPORT_UNITS = (552, int(XY_SUPPORT_DELTA * screen_tile.TILE_DIVISOR * screen_tile.TILE_MIN_CELLS_XY), int(XY_SUPPORT_DELTA * screen_tile.TILE_DIVISOR * screen_tile.TILE_MIN_CELLS_XY) - 8)
# 40 위 xy 구간: U 552(실자료 최소 U)·ΔU 50 → tile 폭 5 칸 ok, U = 2·50·5 = 500 도 5 칸 ok, 거기서 8 px 작은 U → 4 칸이라 tile_too_small


def support_measure(delta=SUPPORT_DELTA, units=SUPPORT_UNITS, xy_delta=XY_SUPPORT_DELTA, xy_units=XY_SUPPORT_UNITS):
    """지원 범위 확인: 지원 상한 ΔU 에서 U = TILE_DIVISOR·ΔU·최소 폭 → tile 폭이 최소(7 칸)로 ok, 거기서 8 px 작은 U → tile_too_small. 같은 설계·같은 합성기.
    40 위 xy 구간(ΔU 50, 두 축)은 최소 폭 5 칸: U 552·500 은 ok, 492 는 tile_too_small."""
    records = []
    for d, us in ((delta, units), (xy_delta, xy_units)):
        for unit in us:
            sa, ra = signature("A", unit, unit, 0)
            sb, rb = signature("B", unit + d, unit + d, 0)
            rec = tile_pair(sa, ra, sb, rb, "layer")
            rec.update(unit=unit, delta=d, axis="xy", phase=0, path="layer", variant=None, p0=None)
            records.append(rec)
    return records


def rule_deltas(r):
    """레코드의 (ΔU_y, ΔU_x). axis 가 없으면 두 축이 같다(xy)."""
    return delta_pair(r["delta"], r.get("axis", "xy"))


def xy_ratio_ok(dy, dx):
    """게이트 쪽 지원 비율 판정(screen_tile.is_xy_stretch 와 따로 적은 같은 규칙): 같은 부호이고 min/max ≥ XY_RATIO_MIN."""
    return dy * dx > 0 and Fraction(min(abs(dy), abs(dx)), max(abs(dy), abs(dx))) >= Fraction(repr(screen_tile.XY_RATIO_MIN))


def out_of_support_pair(r):
    """ΔU 쌍이 지원 범위 밖인가: max|ΔU| > 40 인데 두 축이 함께 늘어난 쌍(비율 ≥ XY_RATIO_MIN)이 아니거나, max|ΔU| > 50."""
    dy, dx = rule_deltas(r)
    top = max(abs(dy), abs(dx))
    return top > screen_tile.XY_SUPPORT_PX or (top > screen_tile.MAX_DELTA_U_PX and not xy_ratio_ok(dy, dx))


def in_tile_support(r):
    """ΔU 쌍이 tile 지원 범위 안(out_of_support_pair 가 아님)이고 tile 폭이 최소 칸 이상인가. 밖이면 out_of_support 또는 tile_too_small 이어야 한다(→ unknown, 안전 쪽).
    tile 폭은 U_min // (TILE_DIVISOR·ΔU) 이고 축마다 구해 작은 쪽이 정한다(레코드에 unit 이 없으면 폭은 보지 않는다). 최소 칸은 max|ΔU| ≤ 40 이면 7, 그 위(xy 구간)는 5 이다."""
    if out_of_support_pair(r):
        return False
    dy, dx = rule_deltas(r)
    if not r.get("unit") or not (dy or dx):
        return True
    min_cells = screen_tile.TILE_MIN_CELLS if max(abs(dy), abs(dx)) <= screen_tile.MAX_DELTA_U_PX else screen_tile.TILE_MIN_CELLS_XY
    return min(r["unit"] // (screen_tile.TILE_DIVISOR * abs(d)) for d in (dy, dx) if d) >= min_cells


def expected_out_status(r):
    return "out_of_support" if out_of_support_pair(r) else "tile_too_small"


def same_failures(records):
    """(a): 지원 범위 안은 tile_status ok, tile_p ≥ P0 − margin, 판정 tier ≠ different, 같은 조합 2회 동일. 범위 밖(ΔU 초과, 또는 tile 폭이 최소 칸 미만)은 out_of_support·tile_too_small 과 tile_p None.
    ΔU=0 은 P0 를 재는 쌍이라 tile_status·재현성만 본다(강제 P 0 의 tier 는 게이트 산출물이라 단정하지 않는다).
    입력 이동이 SHIFT_TOL_CELLS 넘게 어긋난 쌍은 tile 이 고칠 수 없으므로 tile_p 가 있으면 P0 − margin 이상, 없으면 tier ≠ different 만 요구한다."""
    bad = []
    for r in records:
        if not in_tile_support(r):
            if r["tile_status"] != expected_out_status(r) or r["tile_p"] is not None:
                bad.append(r)
        elif r.get("shift_error", 0) > SHIFT_TOL_CELLS:
            if not r["repeat_equal"] or r["tier"] == "different" or (r["tile_p"] is not None and r["p0"] is not None and r["tile_p"] < r["p0"] - MARGIN_P):
                bad.append(r)
        elif r["tile_status"] != "ok" or r["tile_p"] is None or r["p0"] is None or not r["repeat_equal"]:
            bad.append(r)
        elif r["delta"] and (r["tile_p"] < r["p0"] - MARGIN_P or r["tier"] == "different"):
            bad.append(r)
    return bad


def false_failures(records):
    """(b): P_var0 < 0.35 인 변형은 ΔU>0 의 모든 쌍에서 tile_p 가 None 이거나 (tier == different 이고 tile_p ≤ P_var0 + 0.07) 여야 한다. 같은 조합 2회 동일도 본다.
    P_var0 는 같은 경로의 값이고, 두 경로 중 높은 값(p_var0_best)이 더 크면 tile_p 상한은 그 값 기준이다(anchor 경로는 단일 이동 오차로 P_var0 가 낮아질 수 있다)."""
    bad = []
    for r in records:
        p_var0 = r["p_var0"]
        if not r["repeat_equal"]:
            bad.append(r)
        elif r["delta"] and p_var0 is not None and p_var0 < FALSE_P:
            if r["tile_p"] is not None and (r["tier"] != "different" or r["tier_low_u"] != "different" or r["tile_p"] > max(p_var0, r.get("p_var0_best") or p_var0) + FALSE_SLACK):
                bad.append(r)
            elif r["tile_p"] is None and r["tier_low_u"] != "different":  # tile 실패 + U 낮음 → 표의 tile_p 없음 행이 different 여야 한다
                bad.append(r)
    return bad


def support_failures(records):
    """지원 상한 레코드: tile 폭이 최소 칸 이상이면 ok 이고 tile 폭이 U // (TILE_DIVISOR·ΔU) 칸, 미만이면 tile_too_small 과 tile_p None 이어야 한다."""
    bad = []
    for r in records:
        width = r["unit"] // (screen_tile.TILE_DIVISOR * r["delta"])
        ok = in_tile_support(r)
        if (r["tile_status"] != ("ok" if ok else "tile_too_small") or (ok and (r["tile_size_y_cells"] != width or r["tile_p"] is None))
                or (not ok and r["tile_p"] is not None)):
            bad.append(r)
    return bad


# ---- dl1-size/T8: 반복 unit 오인 반례 -------------------------------------------------------------------------------------
REPEAT_CASES = ((552, 38), (560, 40), (552, 45), (560, 50))  # (U_A, ΔU): 실자료 최소 U 의 M13·WF3 형, 지원 범위 끝(tile 폭 최소 7 칸), 40 위 xy 구간(ΔU 45·50 두 축). 지원 범위 안이어야 한다
REAL_SHIFT = (-82, -42)  # 실자료 M13·WF3 의 layer_shift y −65.6 µm, x −33.6 µm (0.1 µm/px, 8 px/칸 → 칸). compare·layer 정렬이 낸 중심 정렬 기준 이동이다


def repeat_shifts(sa, sb, unit):
    """(이름, 중심 정렬 기준 이동 또는 None). 참 원점 이동은 0 이다. 같은 설계는 unit 이 반복이라 한 unit(y) + 반 unit(x) 옆에도 같은 패턴이 놓여 있다."""
    center = tuple((sa["shape"][ax] - sb["shape"][ax]) // 2 for ax in (0, 1))
    one_y, half_x = int(round(unit / FACTOR)), int(round(unit / FACTOR / 2))
    return (("none", None), ("minus", (center[0] - one_y, center[1] - half_x)), ("plus", (center[0] + one_y, center[1] + half_x)), ("real", REAL_SHIFT))


def tile_layers(sa, sb, anchor):
    """tile_pattern_axis 와 같은 정렬 소스."""
    sources, _, keys = tile_inputs(sa, sb, anchor)
    layers = {k: (np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)) for k, (a, b) in sorted(dict(sources or {}).items())}
    if not layers:
        layers = {k: (np.asarray(sa["rasters"][k]["mask"], dtype=bool), np.asarray(sb["rasters"][k]["mask"], dtype=bool)) for k in sorted(keys) if k in sa["rasters"] and k in sb["rasters"]}
    return layers


def repeat_measure(cases=REPEAT_CASES):
    """같은 설계 쌍(A U → B U+ΔU)에 입력 이동을 한 unit(y)+반 unit(x) 어긋나게 넣어도 (0,0) 시작점이 골라지고 tile 이 맞는지 잰다.
    레코드마다 고른 시작점·coverage·tile_p·P0(같은 U 의 ΔU=0 tile_p)를 채운다."""
    records = []
    for unit, delta in cases:
        sa, ra = signature("A", unit, unit, 0)
        sb, rb = signature("B", unit + delta, unit + delta, 0)
        sz, rz = signature("B", unit, unit, 0)
        p0 = tile_pair(sa, ra, sz, rz, "layer")["tile_p"]
        for name, shift in repeat_shifts(sa, sb, unit):
            rec = tile_pair(sa, ra, sb, rb, "layer", shift)
            plan = screen_tile.tile_plan(screen_tile.unit_geometry(sa["full_shape"], FACTOR), screen_tile.unit_geometry(sb["full_shape"], FACTOR), FACTOR, TOLERANCE)
            aligned = screen_tile.align_tiles(plan, tile_layers(sa, sb, rec["anchor"]), np.asarray(sa["valid"], dtype=bool), np.asarray(sb["valid"], dtype=bool), rec["origin_shift"])
            rec.update(unit=unit, delta=delta, axis="xy", phase=0, path="layer", variant=None, p0=p0, shift_name=name, input_shift=rec["shift"], origin=aligned["origin"],
                       coverage=aligned["coverage"])
            records.append(rec)
    return records


def repeat_failures(records):
    """원점 (0,0) 이 골라지고 tile ok, tile_p ≥ P0 − margin, tier ≠ different, 같은 조합 2회 동일."""
    return [r for r in records if r["origin"] != "zero" or r["tile_status"] != "ok" or r["tile_p"] is None or r["p0"] is None or r["tile_p"] < r["p0"] - MARGIN_P
            or r["tier"] == "different" or not r["repeat_equal"]]


def repeat_table(records):
    lines = ["| U | ΔU | 입력 이동 | 입력 이동 (y,x 칸; 중심 기준) | 원점 이동 | 고른 시작점 | coverage | tile 상태 | tile_p | P0 | tile_p−P0 | tier | 시간(s) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        diff = r["tile_p"] - r["p0"] if r["tile_p"] is not None and r["p0"] is not None else None
        lines.append(f"| {r['unit']} | {r['delta']} | {r['shift_name']} | {r['input_shift']} | {r['origin_shift']} | {r['origin']} | {r['coverage']:.4f} | {r['tile_status']} | {fmt(r['tile_p'])} | "
                     f"{fmt(r['p0'])} | {fmt(diff)} | {r['tier']}/{r['reason']} | {r['seconds']:.2f} |")
    return "\n".join(lines)


def seconds(group):
    values = [r["seconds"] for r in group]
    return f"{sum(values) / len(values):.2f}/{max(values):.2f}" if values else "—"


def tile_table(records):
    lines = ["| U | 변형 | ΔU | 축 | 경로 | n | tile 상태 | tile_p min | P0 min | (tile_p−P0) min | P_var0 min | (tile_p−P_var0) max | 단일 P min | 단일 unit P min | tier(강제) | 층 수 | tile 수 | tile 폭(칸) | 시간 평균/최대(s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    keyf = lambda r: (r["unit"], r["variant"] or "same", r["delta"], r["axis"], r["path"])
    for key in sorted({keyf(r) for r in records}):
        g = [r for r in records if keyf(r) == key]
        tiled = [r["tile_p"] for r in g if r["tile_p"] is not None]
        diffs = [r["tile_p"] - r["p0"] for r in g if r["tile_p"] is not None and r["p0"] is not None]
        over = [r["tile_p"] - r["p_var0"] for r in g if r["tile_p"] is not None and r.get("p_var0") is not None]
        mn = lambda vals: fmt(min(vals, default=None))
        statuses = ",".join(sorted({str(r["tile_status"]) for r in g}))
        tiers = ",".join(sorted({f"{r['tier']}/{r['reason']}" + (f" (U낮음 {r['tier_low_u']})" if r["tier_low_u"] != r["tier"] else "") for r in g}))
        widths = ",".join(sorted({f"{r['tile_size_y_cells']}x{r['tile_size_x_cells']}" for r in g}))
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {key[4]} | {len(g)} | {statuses} | {mn(tiled)} | {mn([r['p0'] for r in g if r['p0'] is not None])} | {mn(diffs)} | "
                     f"{mn([r['p_var0'] for r in g if r.get('p_var0') is not None])} | {fmt(max(over, default=None))} | "
                     f"{mn([r['axis_p'] for r in g if r['axis_p'] is not None])} | {mn([r['single_p'] for r in g if r['single_p'] is not None])} | {tiers} | "
                     f"{','.join(sorted({str(r['n_layers']) for r in g}))} | {','.join(sorted({str(r['tile_count']) for r in g}))} | {widths} | {seconds(g)} |")
    return "\n".join(lines)


def delta_table(same, false):
    """ΔU 별 요약. (a) 같은 설계: 통과율·최소 tile_p−P0. (b) 거짓 일치: P_var0 < 0.35 변형 쌍의 실패 수·최대 tile_p−P_var0. 시간 평균/최대."""
    lines = ["| ΔU | 범위 | (a) n | (a) 통과 | (a) 통과율 | (a) 이동 오류 제외 | (a) tile ok | (a) min(tile_p−P0) | (b) n(P_var0<0.35) | (b) tile_p 있음 | (b) 실패 | (b) max(tile_p−P_var0) | 시간 평균/최대(s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    bad_same, bad_false = {id(r) for r in same_failures(same)}, {id(r) for r in false_failures(false)}
    for delta in sorted({r["delta"] for r in same + false}):
        a = [r for r in same if r["delta"] == delta]
        b = [r for r in false if r["delta"] == delta and r["p_var0"] is not None and r["p_var0"] < FALSE_P]
        diffs = [r["tile_p"] - r["p0"] for r in a if r["tile_p"] is not None and r["p0"] is not None]
        over = [r["tile_p"] - r["p_var0"] for r in b if r["tile_p"] is not None]
        everything = a + [r for r in false if r["delta"] == delta]
        a_pass = sum(id(r) not in bad_same for r in a)
        excluded = sum(r.get("shift_error", 0) > SHIFT_TOL_CELLS for r in a)
        scope = "안" if delta <= screen_tile.MAX_DELTA_U_PX else ("xy만" if delta <= screen_tile.XY_SUPPORT_PX else "밖")
        lines.append(f"| {delta} | {scope} | {len(a)} | {a_pass} | {a_pass / len(a) if a else 0:.3f} | {excluded} | "
                     f"{sum(r['tile_status'] == 'ok' for r in a)} | {fmt(min(diffs, default=None))} | {len(b)} | {len(over)} | "
                     f"{sum(id(r) in bad_false for r in false if r['delta'] == delta)} | {fmt(max(over, default=None))} | {seconds(everything)} |")
    return "\n".join(lines)


def ratio_table(same, false=()):
    """40 위 xy 구간(ΔU 42·45·50)의 축 비율별 요약: 실제 (ΔU_y, ΔU_x)·비율, 규칙상 지원 여부, (a) 통과 수·tile ok 수·최소 tile_p−P0 (U 별 최솟값 포함), (b) 있으면 tile_p 있음·최대 tile_p−P_var0·실패 수."""
    lines = ["| ΔU | 축 | (ΔU_y, ΔU_x) | min/max | 지원(규칙) | (a) n | (a) 통과 | (a) tile ok | (a) min(tile_p−P0) | U 별 min(tile_p−P0) | (b) n(P_var0<0.35) | (b) tile_p 있음 | (b) 실패 | (b) max(tile_p−P_var0) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    bad_same, bad_false = {id(r) for r in same_failures(same)}, {id(r) for r in false_failures(list(false))}
    keys = sorted({(r["delta"], r["axis"]) for r in list(same) + list(false) if r["delta"] in XY_DELTAS and r["axis"] not in ("x", "y")},
                  key=lambda k: (k[0], -delta_pair(k[0], k[1])[1]))
    for delta, axis in keys:
        a = [r for r in same if (r["delta"], r["axis"]) == (delta, axis)]
        b = [r for r in false if (r["delta"], r["axis"]) == (delta, axis) and r["p_var0"] is not None and r["p_var0"] < FALSE_P]
        dy, dx = delta_pair(delta, axis)
        diffs = lambda g: [r["tile_p"] - r["p0"] for r in g if r["tile_p"] is not None and r["p0"] is not None]
        per_unit = ",".join(f"{u}:{fmt(min(diffs([r for r in a if r['unit'] == u]), default=None))}" for u in sorted({r["unit"] for r in a}))
        over = [r["tile_p"] - r["p_var0"] for r in b if r["tile_p"] is not None]
        supported_by_rule = not out_of_support_pair({"delta": delta, "axis": axis})
        lines.append(f"| {delta} | {axis} | ({dy}, {dx}) | {min(dy, dx) / max(dy, dx):.3f} | {'예' if supported_by_rule else '아니오'} | {len(a)} | {sum(id(r) not in bad_same for r in a)} | "
                     f"{sum(r['tile_status'] == 'ok' for r in a)} | {fmt(min(diffs(a), default=None))} | {per_unit} | {len(b)} | {len(over)} | "
                     f"{sum(id(r) in bad_false for r in false if (r['delta'], r['axis']) == (delta, axis))} | {fmt(max(over, default=None))} |")
    return "\n".join(lines)


def variant_table(false):
    """변형별 요약(R1 adv:1): P_var0 범위, ΔU>0 쌍 수, tile_p 가 있는 쌍 수, 최대 tile_p−P_var0, (b) 실패 수, U 낮음 판정이 different 가 아닌 쌍 수."""
    lines = ["| 변형 | P_var0 min | P_var0 max | ΔU>0 n | tile_p 있음 | max(tile_p−P_var0) | 최대 tile_p | (b) 실패 | U낮음 tier≠different |", "|---|---|---|---|---|---|---|---|---|"]
    bad = {id(r) for r in false_failures(false)}
    for variant in sorted({r["variant"] for r in false}):
        g = [r for r in false if r["variant"] == variant and r["delta"]]
        zero = [r["p_var0"] for r in false if r["variant"] == variant and r["p_var0"] is not None]
        tiled = [r for r in g if r["tile_p"] is not None]
        over = [r["tile_p"] - r["p_var0"] for r in tiled if r["p_var0"] is not None]
        lines.append(f"| {variant} | {fmt(min(zero, default=None))} | {fmt(max(zero, default=None))} | {len(g)} | {len(tiled)} | {fmt(max(over, default=None))} | "
                     f"{fmt(max((r['tile_p'] for r in tiled), default=None))} | {sum(id(r) in bad for r in false if r['variant'] == variant)} | "
                     f"{sum(r['tier_low_u'] != 'different' for r in g)} |")
    return "\n".join(lines)


def sweep_main(args):
    """--xy-sweep: XY_RATIO_MIN 을 정하는 표. 이 프로세스에서만 screen_tile.XY_RATIO_MIN 을 SWEEP_RATIOS 의 최솟값으로 풀어 놓고 (a) 를 비율 1.0–0.5 로 잰다.
    출력의 후보 = (a) 가 모든 U·ΔU·위상·경로에서 통과하는 가장 작은 비율(그 위 비율도 모두 통과해야 한다)."""
    start = time.time()
    done = []
    progress = lambda kind, u, d, a, p: done.append(1) or (len(done) % 20 == 0 and print(f"  ... {len(done)} B products, {time.time() - start:.0f}s", file=sys.stderr))
    saved, screen_tile.XY_RATIO_MIN = screen_tile.XY_RATIO_MIN, min(SWEEP_RATIOS)
    try:
        same = tile_measure(deltas=XY_DELTAS, axes=(), ratios=SWEEP_RATIOS, progress=progress)
        print(f"# xy ratio sweep (U={TILE_UNITS}, ΔU={XY_DELTAS}, ratios={SWEEP_RATIOS}, phases={TILE_PHASES}; 지원 비율을 {min(SWEEP_RATIOS)} 로 풀어 놓고 잰 (a))\n")
        print(ratio_table(same))
        bad = {id(r) for r in same_failures(same)}
        passing = []
        for ratio in SWEEP_RATIOS:
            group = [r for r in same if r["delta"] and r["axis"] == ratio_axis(ratio)]
            passing.append((ratio, all(id(r) not in bad for r in group)))
        candidate = None
        for ratio, ok in passing:  # SWEEP_RATIOS 는 내림차순: 처음 실패하기 전까지가 연속 통과 구간
            if not ok:
                break
            candidate = ratio
        print(f"\nper ratio (a) all pass: {passing}")
        print(f"XY_RATIO_MIN candidate (smallest ratio with every larger ratio also passing): {candidate}; screen_tile.XY_RATIO_MIN is {saved}; elapsed {time.time() - start:.0f}s")
    finally:
        screen_tile.XY_RATIO_MIN = saved
    return 0


def tile_main(args):
    if args.xy_sweep:
        return sweep_main(args)
    start = time.time()
    done = []
    progress = lambda kind, u, d, a, p: done.append(1) or (len(done) % 20 == 0 and print(f"  ... {len(done)} B products, {time.time() - start:.0f}s", file=sys.stderr))
    same = tile_measure(progress=progress)
    false = false_measure(progress=progress)
    support = support_measure()
    repeat = repeat_measure()
    print(f"# tile gate (U={TILE_UNITS}, ΔU={TILE_DELTAS}, phases={TILE_PHASES}, factor={FACTOR}, tolerance={TOLERANCE}, "
          f"judge p_floor={JUDGE_P_FLOOR} margin={MARGIN_P} u_floor={JUDGE_U_FLOOR} size_u_floor={JUDGE_SIZE_U_FLOOR})\n")
    fail_same, fail_false, fail_support, fail_repeat = same_failures(same), false_failures(false), support_failures(support), repeat_failures(repeat)
    print(f"## ΔU 별 요약 (지원 범위 max|ΔU| ≤ {screen_tile.MAX_DELTA_U_PX} px, {screen_tile.MAX_DELTA_U_PX} 위 {screen_tile.XY_SUPPORT_PX} 이하는 같은 부호로 함께 늘어난 쌍(min/max ≥ {screen_tile.XY_RATIO_MIN})만, "
          f"그 밖은 out_of_support 여야 한다; 표의 ΔU 는 큰 축)\n")
    print(delta_table(same, false))
    print(f"\n## 40 위 xy 구간 축 비율별 (y 축이 ΔU, x 축이 ceil(비율·ΔU); 지원 비율 {screen_tile.XY_RATIO_MIN}, tile 최소 {screen_tile.TILE_MIN_CELLS_XY} 칸)\n")
    print(ratio_table(same, false))
    print("\n## (a) 같은 설계: tile_p ≥ P0 − 0.05, tile ok, tier ≠ different, 2회 동일 (P0 = 같은 tile 경로의 ΔU=0 tile_p; ΔU=0 은 tier 단정에서 뺀다)\n")
    print(tile_table(same))
    print(f"\n## (b) 거짓 일치: 다른 패턴 B′ (P_var0 < {FALSE_P} 이면 모든 ΔU 에서 tile_p None 또는 tier == different 이고 tile_p ≤ P_var0 + {FALSE_SLACK})\n")
    print(tile_table(false))
    print("\n## (b) 변형별 요약\n")
    print(variant_table(false))
    print(f"\n## 지원 범위 확인 (ΔU {SUPPORT_DELTA} 최소 {screen_tile.TILE_MIN_CELLS} 칸, xy ΔU {XY_SUPPORT_DELTA} 최소 {screen_tile.TILE_MIN_CELLS_XY} 칸: U 552 → 폭 5 칸 ok)\n")
    print(tile_table(support))
    print("\n## 반복 unit 오인 반례: 같은 설계에 입력 이동을 한 unit(y)+반 unit(x) 밀어 넣어도 (0,0) 시작점이 골라진다 (real = 실자료 M13·WF3 layer_shift)\n")
    print(repeat_table(repeat))
    ratios = [r["tile_p"] / r["p0"] for r in same if r["delta"] and r["tile_p"] is not None and r["p0"]]
    diffs = [r["tile_p"] - r["p0"] for r in same if r["delta"] and r["tile_p"] is not None and r["p0"] is not None]
    tiled_false = [r["tile_p"] for r in false if r["delta"] and r["tile_p"] is not None]
    low = [r for r in false if r["delta"] and r["p_var0"] is not None and r["p_var0"] < FALSE_P]
    record = [r for r in false if r["delta"] and r["p_var0"] is not None and FALSE_P <= r["p_var0"] < RECORD_P]
    times = [r["seconds"] for r in same + false + support + repeat]
    print(f"\nsummary: same combos={len(same)} (min tile_p/P0={min(ratios):.4f}, min tile_p−P0={min(diffs):.4f}), "
          f"false combos={len(false)} (P_var0<{FALSE_P}: {len(low)}, P_var0 {FALSE_P}–{RECORD_P} record only: {len(record)}), "
          f"different-pattern tile_p present={len(tiled_false)} max={fmt(max(tiled_false, default=None))}, "
          f"tile seconds mean={sum(times) / len(times):.2f} max={max(times):.2f}")
    bad = fail_same + fail_false + fail_support + fail_repeat
    print(f"failing: (a) same design {len(fail_same)} of {len(same)} (tile not ok: {sum(r['tile_status'] != 'ok' for r in fail_same)}), "
          f"(b) false match {len(fail_false)} of {len(false)}, support {len(fail_support)} of {len(support)}, repeat-unit {len(fail_repeat)} of {len(repeat)}")
    print(f"gate: {'PASS' if not bad else 'FAIL'} ({len(bad)} failing of {len(same) + len(false) + len(support) + len(repeat)}); elapsed {time.time() - start:.0f}s")
    for r in fail_false + fail_support + fail_repeat + fail_same[:20]:
        print(f"  FAIL variant={r['variant']} U={r['unit']} ΔU={r['delta']} axis={r['axis']} phase={r['phase']} path={r['path']} status={r['tile_status']} "
              f"tile_p={r['tile_p']} P0={r['p0']} axis_p={r['axis_p']} single_p={r['single_p']} tier={r['tier']}/{r['reason']} repeat_equal={r['repeat_equal']}")
    return 0 if not bad else 1


def parse_phases(text):
    if "-" in text:
        lo, hi = text.split("-")
        return range(int(lo), int(hi) + 1)
    return [int(v) for v in text.split(",")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--unit", type=int, default=UNIT)
    parser.add_argument("--phases", default="0-7", help="ΔU 0–7 에 쓸 위상(예 0-7 또는 0,4)")
    parser.add_argument("--record-phases", default="0,4", help="ΔU 8·12·20·40(기록만)에 쓸 위상")
    parser.add_argument("--tile", action="store_true", help="dl1-size/T4 tile 합성 게이트(U 560·1000, ΔU 8–60, 거짓 일치 B′, 지원 범위)")
    parser.add_argument("--xy-sweep", action="store_true", help="dl1-size/T9: --tile 과 함께. XY_RATIO_MIN 을 정하는 비율 표(비율 1.0–0.5 의 (a))만 낸다")
    args = parser.parse_args(argv)
    if args.xy_sweep and not args.tile:
        parser.error("--xy-sweep needs --tile")
    if args.tile:
        return tile_main(args)
    start = time.time()
    done = []
    progress = lambda d, a, p: done.append(1) or (len(done) % 20 == 0 and print(f"  ... {len(done)} B products, {time.time() - start:.0f}s", file=sys.stderr))
    gate = measure(args.unit, parse_phases(args.phases), GATE_DELTAS, progress=progress)
    record = measure(args.unit, parse_phases(args.record_phases), RECORD_DELTAS, progress=progress)
    print(f"# unit stretch (U={args.unit} px, frame={2 * args.unit + 2 * MARGIN}², CD={CD} px, factor={FACTOR}, tolerance={TOLERANCE}, margin={MARGIN_P})\n")
    print("## 게이트 ΔU 0–7 (P0 = 같은 경로·위상의 ΔU=0 쌍)\n")
    print(table(gate))
    print("\n## 기록만 ΔU 8·12·20·40 (P0 는 위 ΔU=0 같은 위상)\n")
    base = {(r["path"], r["phase"]): r["axis_p"] for r in gate if r["delta"] == 0}
    classify(record, base)
    print(table(record))
    gate_fail = gate_failures(gate) + gate_failures(record)
    print(f"\ngate: {'PASS' if not gate_fail else 'FAIL'} ({len(gate_fail)} failing of {len(gate) + len(record)}); elapsed {time.time() - start:.0f}s")
    for r in gate_fail[:20]:
        print(f"  FAIL ΔU={r['delta']} axis={r['axis']} phase={r['phase']} path={r['path']} P={r['axis_p']} P0={r['p0']}")
    return 0 if not gate_fail else 1


if __name__ == "__main__":
    sys.exit(main())
