"""unit 크기가 다른 제품 쌍을 tile 단위로 맞춰 패턴 점수 tile_p 를 낸다(스크리닝 전용, numpy 만 쓴다. screen() 에는 아직 연결하지 않는다).

좌표·이동 약속은 screen_align 과 같다: 래스터 이동 (dy, dx) 에서 B[y, x] 가 A[y + dy, x + dx] 에 놓인다. 칸 = factor×factor 원본 px.
이 모듈이 받는 shift 는 원점 이동(두 프레임의 원점 = unit 0 시작에서의 이동)이다. compare·layer 정렬의 이동은 프레임 중심을 맞춘 기준이라
크기가 다른 쌍에서는 (A 래스터 shape − B 래스터 shape)//2 만큼 다르므로 origin_shift 로 바꿔서 넘긴다(screen_relations.judge_tile_stage).

구조
  unit_geometry   축마다 U_px = (길이 − 1000)/2. 원본 px 경계 500, 500+U, 500+2U 를 한 번만 floor(px/factor) 로 양자화해 공유 경계 배열을 만든다.
  tile_plan       ΔU(B−A)로 tile 폭 T = U_min/(2·ΔU)(최소 7 칸), 탐색 반경(unit·tile 모두 3 고정), tile 목록을 정한다. ΔU 가 MAX_DELTA_U_PX(40) 를 넘거나 범위를 벗어나면 이유 상태를 돌려준다.
  match_tile      compare(layer_align) 규칙의 tile 매처: 고정 Ω, Dice 우선 층 순서, 모호함·낮은 점수·탐색 경계는 거부(→ 기하 예측으로 대체).
  align_tiles     unit 이동 → tile 이동 → G1(이웃 tile 단차 상자) 반복.
  warp_raster     tile 마다 B 를 그 이동으로 잘라 A 좌표에 놓는다.
  tile_pattern_axis  unit 영역 Ω 의 binding 별 경계 F1 평균 = tile_p.

기하 예측: 축마다 unit k 의 A 칸 c 에서 d(c) = shift + (bA[k] − bB[k]) − round(ΔU·(c − bA[k])/U_A). ΔU = U_B − U_A(원본 px), shift 는 unit 0 시작(여백이 같으므로 두 프레임의 원점)에서의 이동이다.
B 가 더 길면(ΔU > 0) A 의 한 칸이 B 의 더 많은 칸에 해당하므로 unit 안쪽으로 갈수록 이동이 음수 쪽으로 간다(계약 문구의 부호와 반대, 보고서에 적음).
"""
from __future__ import annotations

import math
import re
from fractions import Fraction

import numpy as np

from .screen_relations import boundary_f1

FACTOR = 8
TOLERANCE = 1
MARGIN_PX = 500
MAX_DELTA_U_PX = 40  # 지원 범위(합성 게이트 근거: scripts/measure_unit_stretch.py --tile). 이 위는 out_of_support
S_MAX_CELLS = 2
TILE_MIN_CELLS = 7  # 게이트에서 확인한 가장 좁은 tile 폭(U 560, ΔU 40)
TILE_DIVISOR = 2  # tile 폭 = U_min / (2·ΔU): tile 안 늘어남 차가 0.5 칸 이하
UNIT_RADIUS_CELLS = 3  # 기하 예측이 정확해서 unit·tile 탐색 반경은 ΔU 와 무관한 작은 고정값(경계에 닿은 후보는 search_bound 로 거부하므로 실효 ±2)
TILE_RADIUS_CELLS = 3
MAX_TILES = 1024
MIN_DICE = 0.3
EPSILON = Fraction(1, 100)
COVERAGE_MIN = 0.9
PRESENCE_FRACTION = 0.01
MIN_SUPPORT_DIVISOR = 16
MIN_SUPPORT_FLOOR = 4
LAYER_SUPPORT_CAP = 16
TILE_STATUSES = ("ok", "geometry_unsupported", "tile_too_small", "out_of_support", "too_many_tiles", "unit_align_failed", "low_coverage")


def _natural_key(text):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(text))]


def _rnd_div(n, den):
    """n/den 을 가장 가까운 정수로(반은 올림). 정수 산술이라 결정적이다."""
    return (2 * n + den) // (2 * den)


def _cheb(p, q):
    return max(abs(p[0] - q[0]), abs(p[1] - q[1]))


def _tie_key(d):
    return (d[0] * d[0] + d[1] * d[1], d[0], d[1])


def _split_axis(length, tile):
    """길이를 tile 칸씩 자른 [시작, 끝) 목록. 끝 나머지가 tile/2 미만이면 앞 tile 에 합친다(compare 와 같다). tile 보다 짧으면 하나."""
    n = length // tile
    rem = length - n * tile
    if n == 0:
        return [(0, length)]
    edges = [(k * tile, (k + 1) * tile) for k in range(n)]
    if rem == 0:
        return edges
    if 2 * rem < tile:
        edges[-1] = (edges[-1][0], length)
    else:
        edges.append((n * tile, length))
    return edges


def _crop(arr, y0, y1, x0, x1):
    """arr[y0:y1, x0:x1] 이고 배열 밖은 False."""
    arr = np.asarray(arr, dtype=bool)
    out = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    cy0, cy1, cx0, cx1 = max(y0, 0), min(y1, arr.shape[0]), max(x0, 0), min(x1, arr.shape[1])
    if cy0 < cy1 and cx0 < cx1:
        out[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = arr[cy0:cy1, cx0:cx1]
    return out


# --------------------------------------------------------------------------- 기하와 계획

def unit_geometry(full_shape, factor, margin_px=MARGIN_PX):
    """축별 unit 크기와 공유 경계. 반환 dict: status(ok|geometry_unsupported), u_px (y, x), bounds ([b0,b1,b2] y, x 칸 경계)."""
    factor = int(factor)
    out = {"status": "ok", "reason": None, "full_shape": (int(full_shape[0]), int(full_shape[1])), "factor": factor, "margin_px": int(margin_px),
           "u_px": None, "bounds": None}
    u_px, bounds = [], []
    for length in out["full_shape"]:
        diff = length - 2 * int(margin_px)
        if diff <= 0 or diff % 2:
            out.update(status="geometry_unsupported", reason="axis length minus 2*margin must be a positive even number")
            return out
        u = diff // 2
        u_px.append(u)
        bounds.append([(int(margin_px) + k * u) // factor for k in range(3)])
    out["u_px"], out["bounds"] = tuple(u_px), tuple(bounds)
    return out


def unit_regions(geometry):
    """{(iy, ix): (y0, y1, x0, x1)} A 칸 좌표, 끝 미포함. 반개구간이라 겹치지도 비지도 않는다."""
    by, bx = geometry["bounds"]
    return {(iy, ix): (by[iy], by[iy + 1], bx[ix], bx[ix + 1]) for iy in (0, 1) for ix in (0, 1)}


def tile_plan(geometry_a, geometry_b, factor, tolerance):
    """tile 계획. factor ≠ 8 이거나 tolerance ≠ 1 이면 ValueError. 그 밖의 지원 밖은 status 로 돌려준다(tile_status 값).

    키: status, delta_u_px (y, x; B−A), tile_cells (Ty, Tx 명목 폭), s_max_cells, radius_cells (tile 탐색 반경 Ry, Rx), unit_radius_cells (unit 탐색 반경), units, tiles
    (dict: unit_index, tile_index, region), geometry_a, geometry_b."""
    if factor != FACTOR or tolerance != TOLERANCE:
        raise ValueError(f"screen_tile supports only factor {FACTOR} and tolerance {TOLERANCE}, got factor={factor!r} tolerance={tolerance!r}")
    plan = {"status": "ok", "delta_u_px": None, "tile_cells": None, "s_max_cells": S_MAX_CELLS, "radius_cells": None, "units": None, "tiles": [],
            "geometry_a": geometry_a, "geometry_b": geometry_b}
    if geometry_a["status"] != "ok" or geometry_b["status"] != "ok":
        plan["status"] = "geometry_unsupported"
        return plan
    delta = tuple(geometry_b["u_px"][ax] - geometry_a["u_px"][ax] for ax in (0, 1))
    plan["delta_u_px"] = delta
    plan["units"] = unit_regions(geometry_a)
    plan["unit_radius_cells"] = (UNIT_RADIUS_CELLS, UNIT_RADIUS_CELLS)
    plan["radius_cells"] = (TILE_RADIUS_CELLS, TILE_RADIUS_CELLS)
    if max(abs(delta[0]), abs(delta[1])) > MAX_DELTA_U_PX:
        plan["status"] = "out_of_support"
        return plan
    tile = []
    for ax in (0, 1):
        u_min_px = min(geometry_a["u_px"][ax], geometry_b["u_px"][ax])
        u_min_cells = min(g["bounds"][ax][k + 1] - g["bounds"][ax][k] for g in (geometry_a, geometry_b) for k in (0, 1))
        cap = u_min_cells // 2
        t = cap if delta[ax] == 0 else min(u_min_px // (TILE_DIVISOR * abs(delta[ax])), cap)
        tile.append(t)
    plan["tile_cells"] = tuple(tile)
    if min(tile) < TILE_MIN_CELLS:
        plan["status"] = "tile_too_small"
        return plan
    tiles = []
    for q in sorted(plan["units"]):
        y0, y1, x0, x1 = plan["units"][q]
        for ty, (ra, rb) in enumerate(_split_axis(y1 - y0, tile[0])):
            for tx, (ca, cb) in enumerate(_split_axis(x1 - x0, tile[1])):
                tiles.append({"unit_index": q, "tile_index": (ty, tx), "region": (y0 + ra, y0 + rb, x0 + ca, x0 + cb)})
    if len(tiles) > MAX_TILES:
        plan["status"] = "too_many_tiles"
        plan["tile_count"] = len(tiles)
        return plan
    plan["tiles"] = tiles
    return plan


def _geo_term(plan, axis, k, cell):
    """unit k 의 A 칸 cell 에서 unit 시작 대비 늘어남에 따른 이동 보정 −round(ΔU·(cell − bA[k])/U_A)."""
    delta, u = plan["delta_u_px"][axis], plan["geometry_a"]["u_px"][axis]
    offset = cell - plan["geometry_a"]["bounds"][axis][k]
    return _rnd_div(-delta * offset, u)


def _unit_prediction(plan, shift, q, axis_cells):
    """unit q 중심의 기하 예측 이동 (dy, dx)."""
    ga, gb = plan["geometry_a"], plan["geometry_b"]
    out = []
    for ax in (0, 1):
        k = q[ax]
        out.append(int(shift[ax]) + (ga["bounds"][ax][k] - gb["bounds"][ax][k]) + _geo_term(plan, ax, k, axis_cells[ax]))
    return tuple(out)


# --------------------------------------------------------------------------- 매처

def min_support_cells(region):
    """고정 Ω 의 최소 칸 수(영역 면적 기반)."""
    y0, y1, x0, x1 = region
    return max(MIN_SUPPORT_FLOOR, ((y1 - y0) * (x1 - x0)) // MIN_SUPPORT_DIVISOR)


def min_layer_support_cells(region):
    """층 전경의 최소 칸 수. 면적 기반 하한이되 screen_align 기본(ALIGN_MIN_SUPPORT 16)을 넘지 않는다: 큰 unit 의 가는 배선도 층으로 센다."""
    return min(LAYER_SUPPORT_CAP, min_support_cells(region))


def origin_shift(sa, sb, shift):
    """screen() 의 이동(compare·layer 정렬: 두 프레임 중심을 맞춘 기준)을 tile 모듈이 쓰는 원점 이동(unit 0 시작에서의 이동)으로 바꾼다.
    중심 정렬은 compare 의 center = (A 래스터 shape − B 래스터 shape)//2 이다."""
    return tuple(int(shift[ax]) - (int(sa["shape"][ax]) - int(sb["shape"][ax])) // 2 for ax in (0, 1))


def match_tile(layers, valid_a, valid_b, region, pred, radius, *, min_support, min_layer_support=None, min_dice=MIN_DICE, epsilon=EPSILON):
    """영역 하나의 이동을 기하 예측 pred ± radius 안에서 고른다(compare 규칙).

    layers: {이름: (A bool, B bool)} 래스터(같은 격자). 고정 Ω = 영역의 A valid ∧ 모든 후보 이동에서 B valid. Dice = 2I/(a+b)(Fraction),
    층 순서는 최고 Dice 내림차순·이름 자연 순서, ε 안쪽 후보로 좁혀 간다. 거부(applied False)하면 호출자가 pred 를 쓴다.
    반환: applied, delta (선택 이동; 거부면 None), reason (ok|insufficient_omega|no_layers|low_score|ambiguous|search_bound), omega_cells, dice, layer."""
    y0, y1, x0, x1 = (int(v) for v in region)
    py, px = int(pred[0]), int(pred[1])
    ry, rx = int(radius[0]), int(radius[1])
    h, w = y1 - y0, x1 - x0
    out = {"applied": False, "delta": None, "reason": None, "omega_cells": 0, "dice": None, "layer": None}
    layer_support = min_support if min_layer_support is None else min_layer_support
    omega = _crop(valid_a, y0, y1, x0, x1)
    wide = _crop(valid_b, y0 - py - ry, y1 - py + ry, x0 - px - rx, x1 - px + rx)
    cands = [(ey, ex) for ey in range(-ry, ry + 1) for ex in range(-rx, rx + 1)]
    for ey, ex in cands:
        omega &= wide[ry - ey:ry - ey + h, rx - ex:rx - ex + w]
    out["omega_cells"] = int(np.count_nonzero(omega))
    if out["omega_cells"] < min_support:
        out["reason"] = "insufficient_omega"
        return out
    tables = {}
    for key in sorted(layers, key=_natural_key):
        la, lb = layers[key]
        a_arr = _crop(la, y0, y1, x0, x1) & omega
        a = int(np.count_nonzero(a_arr))
        wb = _crop(lb, y0 - py - ry, y1 - py + ry, x0 - px - rx, x1 - px + rx)
        if a < layer_support or int(np.count_nonzero(wb[ry:ry + h, rx:rx + w] & omega)) < layer_support:
            continue
        table = {}
        for ey, ex in cands:
            moved = wb[ry - ey:ry - ey + h, rx - ex:rx - ex + w] & omega
            den = a + int(np.count_nonzero(moved))
            if den:
                table[(ey, ex)] = Fraction(2 * int(np.count_nonzero(a_arr & moved)), den)
        if table:
            tables[key] = table
    if not tables:
        out["reason"] = "no_layers"
        return out
    best = {k: max(t.values()) for k, t in tables.items()}
    order = sorted(tables, key=lambda k: (-best[k], _natural_key(k)))
    l1 = order[0]
    if float(best[l1]) < min_dice:
        out["reason"] = "low_score"
        return out
    survivors = [d for d, f in tables[l1].items() if f + epsilon >= best[l1]]
    decided = l1
    for key in order[1:]:
        if len(survivors) == 1:
            break
        floor = Fraction(-1)
        top = max(tables[key].get(d, floor) for d in survivors)
        survivors = [d for d in survivors if tables[key].get(d, floor) + epsilon >= top]
        decided = key
    if len(survivors) > 1:
        if max(_cheb(p, q) for p in survivors for q in survivors) >= 2:
            out["reason"] = "ambiguous"
            return out
        floor = Fraction(-1)
        chosen = min(survivors, key=lambda d: (tuple(-tables[k].get(d, floor) for k in order),) + _tie_key(d))
    else:
        chosen = survivors[0]
    out["dice"] = float(tables[l1][chosen])
    out["layer"] = decided
    if out["dice"] < min_dice:
        out["reason"] = "low_score"
    elif abs(chosen[0]) == ry or abs(chosen[1]) == rx:
        out["reason"] = "search_bound"
    else:
        out["reason"] = "ok"
        out["applied"] = True
        out["delta"] = chosen
    return out


def _neighbor_pairs(tiles):
    """4-이웃 tile 쌍 (i, j): j 가 i 의 오른쪽 또는 아래쪽이고 경계를 공유한다(끝 = 시작). unit 경계를 넘는 쌍도 포함한다."""
    by_x0, by_y0 = {}, {}
    for i, t in enumerate(tiles):
        y0, y1, x0, x1 = t["region"]
        by_x0.setdefault(x0, []).append(i)
        by_y0.setdefault(y0, []).append(i)
    pairs = []
    for i, t in enumerate(tiles):
        y0, y1, x0, x1 = t["region"]
        for j in by_x0.get(x1, ()):
            r = tiles[j]["region"]
            if r[0] < y1 and y0 < r[1]:
                pairs.append((i, j))
        for j in by_y0.get(y1, ()):
            r = tiles[j]["region"]
            if r[2] < x1 and x0 < r[3]:
                pairs.append((i, j))
    return pairs


def align_tiles(plan, layers, valid_a, valid_b, shift, min_dice=MIN_DICE):
    """unit 이동 → tile 이동 → G1. plan 은 status ok 여야 한다.

    반환: offsets (tile 목록 순서의 (dy, dx)), predictions (tile 기하 예측), applied (매처 결과를 쓴 tile), reasons, step_replaced (G1 이 되돌린 수),
    unit_shifts {unit: (dy, dx) 중심 이동}, unit_failed {unit: 이유} (기하 예측으로 대체한 unit), origin ("shift"|"zero": 고른 시작점 후보),
    unit_corrected (잔차 보정으로 성공한 unit 수).

    시작점: 입력 shift 와 기하 원점 (0, 0) 두 후보로 unit 단계를 돌려 성공 unit 수가 많은 쪽, 같으면 성공 unit Dice 평균이 높은 쪽, 같으면 입력 shift 를 쓴다.
    부분 실패 보정: 일부 unit 만 성공하면 성공 unit 의 (실제 − 예측) 잔차 평균(축별, round(Fraction) 반올림 half-even)을 실패 unit 예측에 더해 한 번 다시 맞춘다."""
    if plan["status"] != "ok":
        raise ValueError(f"plan status is {plan['status']}")
    ry, rx = plan["radius_cells"]
    uy_r, ux_r = plan["unit_radius_cells"]

    def run_units(origin, pred_add=None):
        shifts, failed, dices, results = {}, {}, {}, {}
        for q in sorted(plan["units"]):
            y0, y1, x0, x1 = plan["units"][q]
            center = ((y0 + y1) // 2, (x0 + x1) // 2)
            pred = _unit_prediction(plan, origin, q, center)
            if pred_add is not None and q in pred_add:
                pred = (pred[0] + pred_add[q][0], pred[1] + pred_add[q][1])
            res = match_tile(layers, valid_a, valid_b, plan["units"][q], pred, (uy_r, ux_r), min_support=min_support_cells(plan["units"][q]),
                             min_layer_support=min_layer_support_cells(plan["units"][q]), min_dice=min_dice)
            results[q] = (pred, res)
            if res["applied"]:
                shifts[q] = (pred[0] + res["delta"][0], pred[1] + res["delta"][1])
                dices[q] = res["dice"]
            else:
                shifts[q] = pred
                failed[q] = res["reason"]
        return shifts, failed, dices, results

    origins = [("shift", (int(shift[0]), int(shift[1])))]
    if origins[0][1] != (0, 0):
        origins.append(("zero", (0, 0)))
    best = None
    for name, origin in origins:
        run = run_units(origin)
        score = (len(run[2]), (sum(run[2].values()) / len(run[2])) if run[2] else -1.0)
        if best is None or score > best[0]:  # 엄밀히 클 때만 바꾼다: 같으면 입력 shift
            best = (score, name, origin, run)
    _, origin_name, origin, (unit_shift, unit_failed, dices, results) = best
    unit_center = {q: ((plan["units"][q][0] + plan["units"][q][1]) // 2, (plan["units"][q][2] + plan["units"][q][3]) // 2) for q in plan["units"]}
    corrected = 0
    if unit_failed and dices:
        residuals = [(unit_shift[q][0] - results[q][0][0], unit_shift[q][1] - results[q][0][1]) for q in dices]
        mean = tuple(int(round(Fraction(sum(r[ax] for r in residuals), len(residuals)))) for ax in (0, 1))
        retry = run_units(origin, {q: mean for q in unit_failed})
        for q in sorted(unit_failed):
            unit_shift[q] = retry[0][q]
            if q in retry[1]:
                unit_failed[q] = retry[1][q]
            else:
                del unit_failed[q]
                corrected += 1
    tiles = plan["tiles"]
    preds, offsets, applied, reasons = [], [], [], []
    for t in tiles:
        q = t["unit_index"]
        y0, y1, x0, x1 = t["region"]
        center = ((y0 + y1) // 2, (x0 + x1) // 2)
        pred = tuple(unit_shift[q][ax] + _geo_term(plan, ax, q[ax], center[ax]) - _geo_term(plan, ax, q[ax], unit_center[q][ax]) for ax in (0, 1))
        preds.append(pred)
        res = match_tile(layers, valid_a, valid_b, t["region"], pred, (ry, rx), min_support=min_support_cells(t["region"]),
                         min_layer_support=min_layer_support_cells(t["region"]), min_dice=min_dice)
        reasons.append(res["reason"])
        applied.append(bool(res["applied"]))
        offsets.append((pred[0] + res["delta"][0], pred[1] + res["delta"][1]) if res["applied"] else pred)
    pairs = _neighbor_pairs(tiles)
    s_max = plan["s_max_cells"]
    replaced = 0
    while True:
        bad = set()
        for i, j in pairs:
            for ax in (0, 1):
                du = preds[j][ax] - preds[i][ax]
                do = offsets[j][ax] - offsets[i][ax]
                if not min(0, du) - s_max <= do <= max(0, du) + s_max:
                    bad.update(k for k in (i, j) if applied[k])
                    break
        if not bad:
            break
        for k in sorted(bad):
            applied[k] = False
            offsets[k] = preds[k]
            reasons[k] = "step_exceeded"
            replaced += 1
    return {"offsets": offsets, "predictions": preds, "applied": applied, "reasons": reasons, "step_replaced": replaced,
            "unit_shifts": unit_shift, "unit_failed": unit_failed, "origin": origin_name, "unit_corrected": corrected}


def warp_raster(mask_b, plan, offsets):
    """tile 마다 B 를 그 이동으로 잘라 A 좌표의 tile 영역에 복사한다(프레임 밖 False). 반환 모양은 (unit 영역 끝, unit 영역 끝) 이고 영역 밖은 False."""
    mask_b = np.asarray(mask_b, dtype=bool)
    tiles = plan["tiles"]
    if len(offsets) != len(tiles):
        raise ValueError("one offset per tile is required")
    bounds = plan["geometry_a"]["bounds"]
    out = np.zeros((bounds[0][2], bounds[1][2]), dtype=bool)
    for t, (dy, dx) in zip(tiles, offsets):
        y0, y1, x0, x1 = t["region"]
        out[y0:y1, x0:x1] = _crop(mask_b, y0 - dy, y1 - dy, x0 - dx, x1 - dx)
    return out


# --------------------------------------------------------------------------- 점수

def _score(sa, sb, plan, offsets, binding_keys, tolerance):
    """unit 영역 Ω 에서 binding 별 경계 F1 평균. 반환 (coverage, tile_p 또는 None)."""
    bounds = plan["geometry_a"]["bounds"]
    box = (slice(bounds[0][0], bounds[0][2]), slice(bounds[1][0], bounds[1][2]))
    warped_valid = warp_raster(sb["valid"], plan, offsets)[box]
    omega = np.asarray(sa["valid"], dtype=bool)[box] & warped_valid
    area = omega.size
    coverage = int(np.count_nonzero(omega)) / area if area else 0.0
    if coverage < COVERAGE_MIN:
        return coverage, None
    values = []
    for key in binding_keys:
        ra, rb = sa["rasters"].get(key), sb["rasters"].get(key)
        if ra is None or rb is None:
            continue
        a = np.asarray(ra["mask"], dtype=bool)[box] & omega
        b = warp_raster(rb["mask"], plan, offsets)[box] & omega
        area_a, area_b = int(np.count_nonzero(a)), int(np.count_nonzero(b))
        top = max(area_a, area_b)
        if top == 0 or area_a < PRESENCE_FRACTION * top or area_b < PRESENCE_FRACTION * top or area_a == 0 or area_b == 0:
            continue
        edges = boundary_f1(a, b, omega, tolerance)
        if edges is not None and math.isfinite(edges[2]):
            values.append(float(edges[2]))
    return coverage, round(float(np.mean(values)), 6) if values else None


def _pseudo_plan(geometry):
    """unit 영역 4개를 tile 4개로 본 계획(단일 이동 점수용)."""
    regions = unit_regions(geometry)
    return {"geometry_a": geometry, "tiles": [{"unit_index": q, "tile_index": (0, 0), "region": regions[q]} for q in sorted(regions)]}


def single_shift_unit_p(sa, sb, shift, binding_keys, tolerance=TOLERANCE):
    """같은 헬퍼로 잰 단일 이동 shift 의 unit 영역 P(기준 비교용). 기하가 지원되지 않거나 coverage < 0.9 이면 None."""
    geometry = unit_geometry(sa["full_shape"], sa["factor"])
    if geometry["status"] != "ok":
        return None
    plan = _pseudo_plan(geometry)
    return _score(sa, sb, plan, [(int(shift[0]), int(shift[1]))] * len(plan["tiles"]), binding_keys, tolerance)[1]


def tile_pattern_axis(sa, sb, shift, sources, binding_keys, factor=FACTOR, tolerance=TOLERANCE, min_dice=MIN_DICE):
    """tile 정렬 후 unit 영역 P. sa, sb: product_signature 결과. shift: unit 0 시작(원점)에서의 래스터 이동 (dy, dx).
    sources: {이름: (A bool, B bool)} 정렬 소스 층(비면 양쪽에 있는 binding 래스터를 쓴다). binding_keys: P 를 낼 binding 이름.

    반환 dict: tile_status, tile_p, tile_size_y_cells, tile_size_x_cells, tile_count, tile_applied, tile_step_replaced, tile_radius_y_cells,
    tile_radius_x_cells, tile_shift_min_y/x, tile_shift_max_y/x (예측 대비 칸). 소요 시간은 넣지 않는다."""
    out = {"tile_status": None, "tile_p": None, "tile_size_y_cells": None, "tile_size_x_cells": None, "tile_count": None, "tile_applied": None,
           "tile_step_replaced": None, "tile_radius_y_cells": None, "tile_radius_x_cells": None, "tile_shift_min_y": None, "tile_shift_max_y": None,
           "tile_shift_min_x": None, "tile_shift_max_x": None}
    plan = tile_plan(unit_geometry(sa["full_shape"], factor), unit_geometry(sb["full_shape"], factor), factor, tolerance)
    if plan["tile_cells"] is not None:
        out["tile_size_y_cells"], out["tile_size_x_cells"] = plan["tile_cells"]
    if plan["radius_cells"] is not None:
        out["tile_radius_y_cells"], out["tile_radius_x_cells"] = plan["radius_cells"]
    if plan["status"] != "ok":
        out["tile_status"] = plan["status"]
        out["tile_count"] = plan.get("tile_count")
        return out
    out["tile_count"] = len(plan["tiles"])
    layers = {k: (np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)) for k, (a, b) in sorted(dict(sources or {}).items())}
    if not layers:
        layers = {k: (np.asarray(sa["rasters"][k]["mask"], dtype=bool), np.asarray(sb["rasters"][k]["mask"], dtype=bool))
                  for k in sorted(binding_keys) if k in sa["rasters"] and k in sb["rasters"]}
    aligned = align_tiles(plan, layers, np.asarray(sa["valid"], dtype=bool), np.asarray(sb["valid"], dtype=bool), shift, min_dice=min_dice)
    out["tile_applied"], out["tile_step_replaced"] = int(sum(aligned["applied"])), aligned["step_replaced"]
    moves = [(o[0] - p[0], o[1] - p[1]) for o, p in zip(aligned["offsets"], aligned["predictions"])]
    out["tile_shift_min_y"], out["tile_shift_max_y"] = min(m[0] for m in moves), max(m[0] for m in moves)
    out["tile_shift_min_x"], out["tile_shift_max_x"] = min(m[1] for m in moves), max(m[1] for m in moves)
    if len(aligned["unit_failed"]) == len(plan["units"]):
        out["tile_status"] = "unit_align_failed"
        return out
    coverage, p = _score(sa, sb, plan, aligned["offsets"], binding_keys, tolerance)
    if coverage < COVERAGE_MIN:
        out["tile_status"] = "low_coverage"
        return out
    out["tile_status"], out["tile_p"] = "ok", p
    return out
