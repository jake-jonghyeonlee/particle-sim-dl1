"""dl1-size/T1: tile 정렬 모듈(screen_tile) 합성 raster 검증. 구조는 2×2 unit + 사방 margin 500, factor 8.

합성 제품은 unit 설계(분수 좌표의 사각형 목록)를 unit 크기에 맞게 배치한다: 사각형 크기(CD)는 고정이고 간격만 U 에 따라 늘어난다.
"""
import numpy as np
import pytest

from psdl1 import screen_tile as st

KEYS = ("ELEC/M1", "ELEC/M2")


def design(seed, n=30):
    rng = np.random.RandomState(seed)
    return [(rng.uniform(0.04, 0.9), rng.uniform(0.04, 0.9), int(rng.randint(2, 7)), int(rng.randint(2, 7))) for _ in range(n)]


def geometry(u_px):
    return st.unit_geometry((2 * u_px + 1000, 2 * u_px + 1000), 8)


def product(u_px, designs):
    """designs: {binding: 사각형 목록}. 같은 설계를 4 unit 에 모두 놓는다."""
    length = 2 * u_px + 1000
    cells = -(-length // 8)
    g = geometry(u_px)
    rasters = {}
    for key, rects in designs.items():
        m = np.zeros((cells, cells), dtype=bool)
        for iy in (0, 1):
            for ix in (0, 1):
                y0, x0 = g["bounds"][0][iy], g["bounds"][1][ix]
                ly, lx = g["bounds"][0][iy + 1] - y0, g["bounds"][1][ix + 1] - x0
                for fy, fx, h, w in rects:
                    y, x = y0 + int(round(fy * ly)), x0 + int(round(fx * lx))
                    m[y:y + h, x:x + w] = True
        rasters[key] = {"mask": m}
    return {"rasters": rasters, "valid": np.ones((cells, cells), dtype=bool), "shape": (cells, cells), "full_shape": (length, length), "factor": 8}


def sources(sa, sb):
    return {k: (sa["rasters"][k]["mask"], sb["rasters"][k]["mask"]) for k in KEYS}


DESIGN = {KEYS[0]: design(1), KEYS[1]: design(2)}
OTHER = {KEYS[0]: design(11), KEYS[1]: design(12)}


# --------------------------------------------------------------------------- 기하

@pytest.mark.parametrize("u_px", [552, 556, 561])
def test_unit_boundaries_are_shared_without_overlap_or_gap(u_px):
    g = geometry(u_px)
    assert g["status"] == "ok" and g["u_px"] == (u_px, u_px)
    cover = np.zeros((300, 300), dtype=int)
    for y0, y1, x0, x1 in st.unit_regions(g).values():
        cover[y0:y1, x0:x1] += 1
    by, bx = g["bounds"]
    assert by == [500 // 8, (500 + u_px) // 8, (500 + 2 * u_px) // 8] == bx
    assert (cover[by[0]:by[2], bx[0]:bx[2]] == 1).all() and cover.sum() == (by[2] - by[0]) * (bx[2] - bx[0])


@pytest.mark.parametrize("shape", [(2105, 2104), (2104, 2105), (1000, 1000), (900, 2104)])
def test_odd_difference_or_non_positive_unit_is_unsupported(shape):
    assert st.unit_geometry(shape, 8)["status"] == "geometry_unsupported"


# --------------------------------------------------------------------------- 계획

def plan_of(ua, ub):
    return st.tile_plan(geometry(ua), geometry(ub), 8, 1)


def test_tile_width_follows_the_delta_u_and_limits():
    p = plan_of(552, 590)  # M13·WF3 형: tile 폭 = U_min // (2·ΔU) = 552 // 76
    assert p["status"] == "ok" and p["tile_cells"] == (7, 7) and p["s_max_cells"] == 2
    assert p["radius_cells"] == (3, 3) and p["unit_radius_cells"] == (3, 3)  # 반경은 ΔU 와 무관한 고정값
    assert plan_of(560, 600)["status"] == "ok" and plan_of(560, 600)["tile_cells"] == (7, 7)  # 지원 상한 ΔU 40
    assert plan_of(552, 592)["status"] == "tile_too_small"  # 552 // 80 = 6 칸 < 7
    assert plan_of(560, 601)["status"] == "out_of_support"
    assert plan_of(480, 541)["status"] == "out_of_support"
    same = plan_of(552, 552)
    assert same["status"] == "ok" and same["tile_cells"] == (34, 34) and same["radius_cells"] == (3, 3)


def test_layer_support_is_capped_but_omega_support_follows_the_area():
    big, small = (0, 125, 0, 125), (0, 20, 0, 20)
    assert st.min_support_cells(big) == 976 and st.min_layer_support_cells(big) == 16  # 큰 unit 의 가는 배선도 층 전경으로 센다
    assert st.min_support_cells(small) == 25 and st.min_layer_support_cells(small) == 16
    assert st.min_support_cells((0, 6, 0, 6)) == 4 == st.min_layer_support_cells((0, 6, 0, 6))


def test_origin_shift_removes_the_frame_center_alignment():
    sa, sb = {"shape": (375, 375)}, {"shape": (385, 383)}
    assert st.origin_shift(sa, sb, (-5, -4)) == (0, 0)  # 중심 정렬 이동 = (A − B)//2 = (−5, −4)
    assert st.origin_shift(sa, sb, (-3, 1)) == (2, 5)
    assert st.origin_shift(sa, sa, (3, -2)) == (3, -2)  # 같은 크기면 그대로


def test_tile_plan_rejects_other_factor_or_tolerance():
    ga = geometry(552)
    with pytest.raises(ValueError):
        st.tile_plan(ga, ga, 4, 1)
    with pytest.raises(ValueError):
        st.tile_plan(ga, ga, 8, 0)


def test_too_many_tiles(monkeypatch):
    monkeypatch.setattr(st, "MAX_TILES", 100)
    p = plan_of(560, 600)
    assert p["status"] == "too_many_tiles" and p["tile_count"] > 100
    sa, sb = product(560, DESIGN), product(600, DESIGN)
    assert st.tile_pattern_axis(sa, sb, (0, 0), sources(sa, sb), KEYS)["tile_status"] == "too_many_tiles"


def test_tiles_use_actual_widths_and_cover_each_unit_once():
    p = plan_of(556, 556)
    cover = np.zeros((300, 300), dtype=int)
    for t in p["tiles"]:
        y0, y1, x0, x1 = t["region"]
        cover[y0:y1, x0:x1] += 1
    by, bx = p["geometry_a"]["bounds"]
    assert (cover[by[0]:by[2], bx[0]:bx[2]] == 1).all() and cover.sum() == (by[2] - by[0]) * (bx[2] - bx[0])


# --------------------------------------------------------------------------- X3, 늘어난 간격

def test_x3_same_product_tile_p_equals_single_shift_unit_p():
    sa = product(480, DESIGN)
    sb = product(480, DESIGN)
    out = st.tile_pattern_axis(sa, sb, (0, 0), sources(sa, sb), KEYS)
    single = st.single_shift_unit_p(sa, sb, (0, 0), KEYS)
    assert out["tile_status"] == "ok" and np.isfinite(out["tile_p"]) and out["tile_p"] == single == 1.0
    assert out["tile_shift_min_y"] == out["tile_shift_max_y"] == out["tile_shift_min_x"] == out["tile_shift_max_x"] == 0


def test_stretched_spacing_tile_p_is_high_and_beats_single_shift():
    sa, sb = product(560, DESIGN), product(598, DESIGN)
    out = st.tile_pattern_axis(sa, sb, (0, 0), sources(sa, sb), KEYS)
    single = st.single_shift_unit_p(sa, sb, (0, 0), KEYS)
    assert out["tile_status"] == "ok" and out["tile_size_y_cells"] == 7 and out["tile_count"] > 4
    assert out["tile_p"] >= 0.95 and out["tile_p"] > single


@pytest.mark.parametrize("seed", [11, 21, 31, 41, 51, 61])
def test_x4_different_pattern_stays_in_envelope_and_does_not_beat_single_p(seed):
    other = {KEYS[0]: design(seed), KEYS[1]: design(seed + 1)}
    sa, sb = product(560, DESIGN), product(598, other)
    out = st.tile_pattern_axis(sa, sb, (0, 0), sources(sa, sb), KEYS)
    single = st.single_shift_unit_p(sa, sb, (0, 0), KEYS)
    ry, rx = out["tile_radius_y_cells"], out["tile_radius_x_cells"]
    assert out["tile_status"] in {"ok", "unit_align_failed"}      # 다른 패턴은 unit 단계에서 Dice 가 낮아 정렬을 거부할 수 있다
    assert -ry <= out["tile_shift_min_y"] and out["tile_shift_max_y"] <= ry and -rx <= out["tile_shift_min_x"] and out["tile_shift_max_x"] <= rx
    assert out["tile_p"] is None or out["tile_p"] <= single + 0.05


# --------------------------------------------------------------------------- X2 매처

def blank(n=40):
    return np.zeros((n, n), dtype=bool)


def test_x2_repeating_pattern_is_ambiguous_and_falls_back():
    a, b = blank(), blank()
    a[:, ::2] = True
    b[:, ::2] = True
    valid = np.ones_like(a)
    res = st.match_tile({"s": (a, b)}, valid, valid, (10, 24, 10, 24), (0, 0), (2, 2), min_support=4)
    assert not res["applied"] and res["reason"] == "ambiguous" and res["delta"] is None


def test_x2_clear_match_is_chosen_inside_the_window():
    a, b = blank(), blank()
    a[12:22, 14:17] = True
    b[12:22, 15:18] = True      # B[x] 는 A[x + d] 에 놓인다 → d = −1
    valid = np.ones_like(a)
    res = st.match_tile({"s": (a, b)}, valid, valid, (10, 24, 10, 24), (0, 0), (3, 3), min_support=4)
    assert res["applied"] and res["delta"] == (0, -1) and res["dice"] == 1.0


def test_x2_valid_hole_is_excluded_for_every_candidate_and_does_not_raise_dice():
    a, b = blank(), blank()
    a[12:22, 14:17] = True
    b[12:22, 15:18] = True
    b[12:22, 21:23] = True      # A 에 없는 방해 줄
    valid = np.ones_like(a)
    hole = valid.copy()
    hole[:, 21:24] = False
    with_hole = st.match_tile({"s": (a, b)}, valid, hole, (10, 24, 10, 24), (0, 0), (3, 3), min_support=4)
    without = st.match_tile({"s": (a, b)}, valid, valid, (10, 24, 10, 24), (0, 0), (3, 3), min_support=4)
    assert with_hole["applied"] and with_hole["delta"] == (0, -1) and with_hole["dice"] == 1.0
    assert without["dice"] < 1.0
    # 구멍(열 21–23)은 후보 이동 ±3 의 어느 것에서든 가려지는 칸 열 18–26 을 Ω 에서 뺀다. 영역은 열 10–23 이라 6 열이 빠진다
    assert with_hole["omega_cells"] == without["omega_cells"] - 14 * 6
    # 참 B 줄이 구멍에 가려지면 Ω 가 A 줄을 빼므로(층 지지 부족) 구멍을 이용해 다른 후보를 고르지 않는다
    covering = valid.copy()
    covering[:, 13:19] = False
    res = st.match_tile({"s": (a, b)}, valid, covering, (10, 24, 10, 24), (0, 0), (3, 3), min_support=4)
    assert not res["applied"] and res["reason"] in {"no_layers", "insufficient_omega"}


def test_x2_reference_outside_the_frame_is_not_valid_in_the_warped_raster():
    sb = product(480, DESIGN)
    plan = st.tile_plan(geometry(480), geometry(480), 8, 1)
    shifted = st.warp_raster(sb["valid"], plan, [(500, 0)] + [(0, 0)] * (len(plan["tiles"]) - 1))
    y0, y1, x0, x1 = plan["tiles"][0]["region"]
    assert not shifted[y0:y1, x0:x1].any() and shifted[plan["tiles"][1]["region"][0]:plan["tiles"][1]["region"][1], plan["tiles"][1]["region"][2]:].any()
    ok = st.warp_raster(sb["valid"], plan, [(0, 0)] * len(plan["tiles"]))
    assert ok[62:200, 62:200].all()


# --------------------------------------------------------------------------- X1/G1

def test_g1_catches_a_violation_across_the_unit_boundary_and_terminates(monkeypatch):
    plan = st.tile_plan(geometry(480), geometry(480), 8, 1)
    target = next(t for t in plan["tiles"] if t["unit_index"] == (0, 0) and t["region"][3] == plan["units"][(0, 0)][3] and t["tile_index"][0] == 0)
    assert any(t["unit_index"] == (0, 1) and t["region"][2] == target["region"][3] and t["region"][0] < target["region"][1] and target["region"][0] < t["region"][1]
               for t in plan["tiles"])

    def fake(layers, va, vb, region, pred, radius, **kw):
        delta = (0, 3) if tuple(region) == tuple(target["region"]) else (0, 0)
        return {"applied": True, "delta": delta, "reason": "ok", "omega_cells": 1, "dice": 1.0, "layer": "s"}
    monkeypatch.setattr(st, "match_tile", fake)
    valid = np.ones((300, 300), dtype=bool)
    out = st.align_tiles(plan, {"s": (valid, valid)}, valid, valid, (0, 0))
    idx = plan["tiles"].index(target)
    assert out["step_replaced"] >= 1 and not out["applied"][idx] and out["offsets"][idx] == out["predictions"][idx]
    for i, j in st._neighbor_pairs(plan["tiles"]):
        for ax in (0, 1):
            du = out["predictions"][j][ax] - out["predictions"][i][ax]
            do = out["offsets"][j][ax] - out["offsets"][i][ax]
            assert min(0, du) - 2 <= do <= max(0, du) + 2


# --------------------------------------------------------------------------- 상태와 결정성

def test_low_coverage_when_the_warped_valid_is_missing():
    sa, sb = product(480, DESIGN), product(480, DESIGN)
    sb["valid"][62:92, :] = False
    out = st.tile_pattern_axis(sa, sb, (0, 0), sources(sa, sb), KEYS)
    assert out["tile_status"] == "low_coverage" and out["tile_p"] is None


def test_unsupported_statuses_carry_no_p():
    sa = product(480, DESIGN)
    assert st.tile_pattern_axis(sa, product(560, DESIGN), (0, 0), sources(sa, sa), KEYS)["tile_status"] == "out_of_support"
    odd = dict(sa, full_shape=(2105, 2104))
    out = st.tile_pattern_axis(sa, odd, (0, 0), sources(sa, sa), KEYS)
    assert out["tile_status"] == "geometry_unsupported" and out["tile_p"] is None


def test_unit_align_failed_when_no_unit_can_be_matched():
    sa, sb = product(480, DESIGN), product(480, DESIGN)
    empty = {k: (np.zeros_like(sa["valid"]), np.zeros_like(sa["valid"])) for k in KEYS}
    out = st.tile_pattern_axis(sa, sb, (0, 0), empty, KEYS)
    assert out["tile_status"] == "unit_align_failed" and out["tile_p"] is None


# --------------------------------------------------------------------------- R1 adv:3 시작점 후보, adv:2 부분 실패 보정

def align_of(sa, sb, shift):
    plan = st.tile_plan(st.unit_geometry(sa["full_shape"], 8), st.unit_geometry(sb["full_shape"], 8), 8, 1)
    return plan, st.align_tiles(plan, sources(sa, sb), sa["valid"], sb["valid"], shift)


@pytest.mark.parametrize("shift", [(3, 0), (0, -4), (4, -4), (5, 5), (-6, 3)])
def test_origin_candidate_rescues_a_center_shift_that_is_off_by_three_or_more_cells(shift):
    """같은 설계 쌍의 중심 정렬 shift 가 원점에서 3칸 이상 어긋나도 (0, 0) 후보로 정렬된다(R1 탐침의 실패 영역).
    기하 예측의 늘어남 항 때문에 한 축 3칸은 입력 shift 로도 정렬되는 경우가 있어 한 축만 어긋난 경우는 4칸부터 단정한다."""
    sa, sb = product(552, DESIGN), product(590, DESIGN)
    plan, out = align_of(sa, sb, shift)
    assert out["origin"] == "zero" and not out["unit_failed"]
    result = st.tile_pattern_axis(sa, sb, shift, sources(sa, sb), KEYS)
    assert result["tile_status"] == "ok" and result["tile_p"] >= 0.95
    assert result["tile_p"] == st.tile_pattern_axis(sa, sb, (0, 0), sources(sa, sb), KEYS)["tile_p"]


@pytest.mark.parametrize("shift", [(0, 0), (1, 0), (0, -2), (2, 2)])
def test_origin_zero_is_kept_when_the_input_shift_works_equally_well(shift):
    sa, sb = product(560, DESIGN), product(560, DESIGN)
    _, out = align_of(sa, sb, shift)
    assert out["origin"] == "zero" and not out["unit_failed"] and out["coverage"] >= st.COVERAGE_MIN  # 둘 다 coverage·성공 수·Dice 가 같으면 (0, 0)


def test_input_shift_wins_only_when_strictly_better():
    sa = product(560, DESIGN)
    sb = displaced_units(sa, {q: (4, 4) for q in ((0, 0), (0, 1), (1, 0), (1, 1))})  # (0, 0) 에서는 탐색 반경 밖이라 모든 unit 이 실패한다
    _, out = align_of(sa, sb, (4, 4))
    assert out["origin"] == "shift" and not out["unit_failed"]
    assert st.tile_pattern_axis(sa, sb, (4, 4), sources(sa, sb), KEYS)["tile_status"] == "ok"


def fake_run(coverage, failed, dice):
    units = ((0, 0), (0, 1), (1, 0), (1, 1))
    return {"offsets": [], "predictions": [], "applied": [], "reasons": [], "step_replaced": 0, "unit_shifts": {}, "unit_failed": {q: "low_score" for q in units[:failed]},
            "unit_corrected": 0, "unit_dices": {q: dice for q in units[failed:]}, "coverage": coverage}


@pytest.mark.parametrize("zero, shift, want", [
    ((0.5, 0, 0.0), (0.95, 3, 0.9), "shift"),   # coverage 0.9 미만인 (0, 0) 은 성공 unit 수·Dice 가 더 좋아도 진다
    ((0.95, 3, 0.9), (0.5, 0, 0.0), "zero"),
    ((0.95, 1, 0.9), (0.95, 2, 0.5), "shift"),  # coverage 가 둘 다 충분하면 성공 unit 수
    ((0.95, 2, 0.5), (0.95, 2, 0.8), "shift"),  # 성공 unit 수가 같으면 Dice 평균
    ((0.95, 2, 0.8), (0.95, 2, 0.8), "zero"),   # 모두 같으면 (0, 0)
    ((0.95, 0, 0.0), (0.95, 0, 0.0), "zero"),   # 둘 다 모든 unit 이 실패해도 (0, 0)
    ((0.91, 4, 0.5), (0.99, 4, 0.5), "zero"),   # coverage 는 임계(0.9) 통과 여부만 본다: 값이 더 높다고 이기지 않는다
])
def test_origin_selection_order_is_coverage_then_unit_count_then_dice_then_zero(monkeypatch, zero, shift, want):
    plan = st.tile_plan(geometry(480), geometry(480), 8, 1)
    results = {(0, 0): fake_run(zero[0], 4 - zero[1], zero[2]), (7, 7): fake_run(shift[0], 4 - shift[1], shift[2])}
    calls = []
    monkeypatch.setattr(st, "_align_from", lambda plan, layers, va, vb, origin, min_dice: calls.append(origin) or results[origin])
    out = st.align_tiles(plan, {}, None, None, (7, 7))
    assert out["origin"] == want and calls == [(0, 0), (7, 7)] and "unit_dices" not in out


def test_a_zero_input_shift_runs_the_candidate_once(monkeypatch):
    plan = st.tile_plan(geometry(480), geometry(480), 8, 1)
    calls = []
    monkeypatch.setattr(st, "_align_from", lambda plan, layers, va, vb, origin, min_dice: calls.append(origin) or fake_run(1.0, 0, 1.0))
    assert st.align_tiles(plan, {}, None, None, (0, 0))["origin"] == "zero" and calls == [(0, 0)]


def test_origin_candidate_is_deterministic():
    sa, sb = product(552, DESIGN), product(590, DESIGN)
    first, second = (st.tile_pattern_axis(sa, sb, (4, -4), sources(sa, sb), KEYS) for _ in range(2))
    assert first == second


def displaced_units(sa, offsets):
    """sa 와 같은 크기의 B: unit 마다 B[y, x] = A[y + dy, x + dx] 가 되도록 놓는다(offsets: {unit: (dy, dx)})."""
    plan = st.tile_plan(st.unit_geometry(sa["full_shape"], 8), st.unit_geometry(sa["full_shape"], 8), 8, 1)
    sb = {**sa, "rasters": {}}
    for key, r in sa["rasters"].items():
        m = np.zeros_like(r["mask"])
        for q, (y0, y1, x0, x1) in plan["units"].items():
            dy, dx = offsets[q]
            m[y0:y1, x0:x1] = st._crop(r["mask"], y0 + dy, y1 + dy, x0 + dx, x1 + dx)
        sb["rasters"][key] = {"mask": m}
    return sb


def test_partial_unit_failure_is_corrected_with_the_mean_residual_of_the_successful_units():
    sa = product(560, DESIGN)
    offsets = {(0, 0): (2, 2), (0, 1): (2, 2), (1, 0): (2, 2), (1, 1): (4, 4)}  # 한 unit 만 입력 예측(0, 0)에서 4칸 어긋나 탐색 반경(실효 ±2) 밖이다
    sb = displaced_units(sa, offsets)
    plan, out = align_of(sa, sb, (0, 0))
    assert out["unit_corrected"] == 1 and out["unit_failed"] == {}
    assert out["unit_shifts"][(1, 1)] == (4, 4) and out["unit_shifts"][(0, 0)] == (2, 2)
    # 보정 없이는 이 unit 이 실패한다: 같은 unit 만 따로 맞춰 본다
    layers = {k: (np.asarray(a), np.asarray(b)) for k, (a, b) in sources(sa, sb).items()}
    alone = st.match_tile(layers, sa["valid"], sb["valid"], plan["units"][(1, 1)], st._unit_prediction(plan, (0, 0), (1, 1), (0, 0)),
                          (3, 3), min_support=4)
    assert not alone["applied"]
    result = st.tile_pattern_axis(sa, sb, (0, 0), sources(sa, sb), KEYS)
    assert result["tile_status"] == "ok"


def test_partial_failure_that_stays_failed_uses_the_corrected_prediction():
    sa = product(560, DESIGN)
    offsets = {(0, 0): (2, 2), (0, 1): (2, 2), (1, 0): (2, 2), (1, 1): (9, 9)}  # 보정 후 예측(2, 2)에서도 7칸 어긋남
    _, out = align_of(sa, displaced_units(sa, offsets), (0, 0))
    assert set(out["unit_failed"]) == {(1, 1)} and out["unit_corrected"] == 0
    assert out["unit_shifts"][(1, 1)] == (2, 2) and out["unit_shifts"][(0, 0)] == (2, 2)


def test_residual_mean_rounds_half_even():
    sa = product(560, DESIGN)
    offsets = {(0, 0): (1, 2), (0, 1): (2, 2), (1, 0): (1, 1), (1, 1): (7, 7)}  # 성공 3 unit 잔차 평균 (4/3, 5/3) → (1, 2)
    _, out = align_of(sa, displaced_units(sa, offsets), (0, 0))
    assert out["unit_shifts"][(0, 0)] == (1, 2) and set(out["unit_failed"]) == {(1, 1)}
    assert out["unit_shifts"][(1, 1)] == (1, 2)  # 보정된 예측


def test_all_units_failing_is_still_unit_align_failed():
    sa = product(560, DESIGN)
    sb = displaced_units(sa, {q: (8, 8) for q in ((0, 0), (0, 1), (1, 0), (1, 1))})
    assert st.tile_pattern_axis(sa, sb, (0, 0), sources(sa, sb), KEYS)["tile_status"] == "unit_align_failed"


def test_same_input_gives_the_same_output():
    sa, sb = product(560, DESIGN), product(598, DESIGN)
    first = st.tile_pattern_axis(sa, sb, (0, 0), sources(sa, sb), KEYS)
    second = st.tile_pattern_axis(sa, sb, (0, 0), sources(sa, sb), KEYS)
    assert first == second and "tile_seconds" not in first
