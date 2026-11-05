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
    return {"rasters": rasters, "valid": np.ones((cells, cells), dtype=bool), "full_shape": (length, length), "factor": 8}


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
    p = plan_of(552, 590)
    assert p["status"] == "ok" and p["tile_cells"] == (14, 14) and p["radius_cells"] == (7, 7) and p["s_max_cells"] == 2
    assert plan_of(480, 540)["status"] == "ok" and plan_of(480, 540)["tile_cells"] == (8, 8)
    assert plan_of(472, 532)["status"] == "tile_too_small"
    assert plan_of(480, 541)["status"] == "out_of_support"
    same = plan_of(552, 552)
    assert same["status"] == "ok" and same["tile_cells"] == (34, 34) and same["radius_cells"] == (2, 2)


def test_tile_plan_rejects_other_factor_or_tolerance():
    ga = geometry(552)
    with pytest.raises(ValueError):
        st.tile_plan(ga, ga, 4, 1)
    with pytest.raises(ValueError):
        st.tile_plan(ga, ga, 8, 0)


def test_too_many_tiles(monkeypatch):
    monkeypatch.setattr(st, "MAX_TILES", 100)
    p = plan_of(480, 540)
    assert p["status"] == "too_many_tiles" and p["tile_count"] > 100
    sa, sb = product(480, DESIGN), product(540, DESIGN)
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
    sa, sb = product(480, DESIGN), product(518, DESIGN)
    out = st.tile_pattern_axis(sa, sb, (0, 0), sources(sa, sb), KEYS)
    single = st.single_shift_unit_p(sa, sb, (0, 0), KEYS)
    assert out["tile_status"] == "ok" and out["tile_size_y_cells"] == 12 and out["tile_count"] > 4
    assert out["tile_p"] >= 0.95 and out["tile_p"] > single


@pytest.mark.parametrize("seed", [11, 21, 31, 41, 51, 61])
def test_x4_different_pattern_stays_in_envelope_and_does_not_beat_single_p(seed):
    other = {KEYS[0]: design(seed), KEYS[1]: design(seed + 1)}
    sa, sb = product(480, DESIGN), product(518, other)
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


def test_same_input_gives_the_same_output():
    sa, sb = product(480, DESIGN), product(518, DESIGN)
    first = st.tile_pattern_axis(sa, sb, (0, 0), sources(sa, sb), KEYS)
    second = st.tile_pattern_axis(sa, sb, (0, 0), sources(sa, sb), KEYS)
    assert first == second and "tile_seconds" not in first
