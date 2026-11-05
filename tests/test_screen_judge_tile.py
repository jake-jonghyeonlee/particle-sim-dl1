"""dl1-size/T3: B 통과 쌍의 tile 정렬 연결(judge_tile_stage) — 정렬 소스·cap·shift 약속·재판정·통계 일치·결정성.

tile 단계는 합성 signature(test_screen_tile.product: 2×2 unit, margin 500 px, factor 8)와 손으로 만든 쌍 dict 로 직접 시험하고,
screen() 연결은 test_screen_judge 의 world 픽스처에서 tile_pattern_axis 를 고정값으로 바꿔 확인한다.
"""
import copy

import numpy as np
import pytest

import psdl1.screen_relations as sr
import psdl1.screen_tile as st
from test_screen_judge import by_pair, verdict
from test_screen_judge_size import SIZE_AXES, U_FLOOR, factor_eight, size_run, world  # noqa: F401
from test_screen_tile import DESIGN, KEYS, design, product, sources

P_FLOOR, MARGIN_V = 0.5, 0.125
SIZE = {"u_floor": U_FLOOR, "enabled": True, "max_pairs": 2000, **sr.JUDGE_SIZE_FIXED}


def with_layers(sig):
    """binding 래스터를 같은 모양의 Layer 항목으로도 붙인다(사이드카가 있는 제품)."""
    sig = dict(sig)
    sig["layers"] = {f"L{i}": {"mask": sig["rasters"][k]["mask"]} for i, k in enumerate(KEYS)}
    return sig


def pair(a, b, u=0.95, anchor="LAYER", delta=38):
    axes = {"axis_p": 0.2, "axis_u": u, "size_tag": "size_differs", "size_delta_unit_px_y": delta, "size_delta_unit_px_x": delta}
    return {"product_a": a, "product_b": b, "same_lineage": False, "metrics_anchor": anchor, "judge_axes": axes, "axis_p": 0.2, "axis_u": u,
            "axis_p_eff": 0.2, "judge_p_floor": P_FLOOR, "judge_u_floor": 0.9, "judge_case": 3, "tier": "unknown", "reason": "size_differs_out_of_support",
            "tile_status": "not_run", "judge_size_u_floor": U_FLOOR}


def stage(pairs, signatures, shifts=None, **size):
    shifts = {(p["product_a"], p["product_b"]): (0, 0) for p in pairs} if shifts is None else shifts
    sr.judge_tile_stage(pairs, signatures, shifts, {**SIZE, **size}, MARGIN_V)
    return pairs


def verdict_of(p):
    return p["tier"], p["reason"], p["judge_case"]


# ---- 같은 설계에서 간격만 늘린 쌍 ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("layered", [True, False])
def test_stretched_same_design_is_tiled_and_not_different(layered):
    sa, sb = product(480, DESIGN), product(518, DESIGN)  # ΔU 38 px, CD 고정
    if layered:
        sa, sb = with_layers(sa), with_layers(sb)
    p = stage([pair("A", "B", anchor="LAYER" if layered else "CNT")], {"A": sa, "B": sb})[0]
    single = st.single_shift_unit_p(sa, sb, (0, 0), KEYS)
    assert p["tile_status"] == "ok" and p["tile_source"] == ("layer" if layered else "binding")
    assert p["tile_p"] >= 0.95 and p["tile_p"] > single
    assert p["tier"] != "different" and p["reason"] in {"close_to_same_lineage", "near_boundary_tiled"}
    assert p["axis_p_eff"] == p["tile_p"] and p["axis_p"] == 0.2  # 판정은 tile_p, scores 의 axis_p 는 단일 P 그대로
    assert p["tile_count"] > 4 and p["tile_size_y_cells"] == 12


def test_layer_anchor_without_common_layers_falls_back_to_bindings():
    sa, sb = product(480, DESIGN), product(518, DESIGN)
    sa["layers"], sb["layers"] = {"X": {"mask": sa["rasters"][KEYS[0]]["mask"]}}, {"Y": {"mask": sb["rasters"][KEYS[0]]["mask"]}}
    p = stage([pair("A", "B")], {"A": sa, "B": sb})[0]
    assert p["tile_source"] == "binding" and p["tile_status"] == "ok"
    sources_, source, keys = sr.tile_inputs(sa, sb, "LAYER")
    assert source == "binding" and sorted(sources_) == sorted(KEYS) and keys == sorted(KEYS)


def test_binding_source_includes_cnt_but_p_keys_do_not():
    sa, sb = product(480, DESIGN), product(518, DESIGN)
    for sig in (sa, sb):
        sig["rasters"][sr.CNT] = {"mask": np.zeros_like(sig["valid"])}
    sources_, source, keys = sr.tile_inputs(sa, sb, "CNT")
    assert source == "binding" and sr.CNT in sources_ and sr.CNT not in keys


# ---- 다른 패턴 ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [11, 21, 31])
def test_different_pattern_never_becomes_close(seed):
    other = {KEYS[0]: design(seed), KEYS[1]: design(seed + 1)}
    sa, sb = with_layers(product(480, DESIGN)), with_layers(product(518, other))
    p = stage([pair("A", "B")], {"A": sa, "B": sb})[0]
    assert p["tile_p"] is None or p["tile_p"] < 0.40
    assert p["reason"] in {"size_differs_out_of_support", "size_differs_below_pattern_floor_tiled"} and p["tier"] in {"different", "unknown"}
    if p["tile_p"] is None:
        assert p["tile_status"] in {"unit_align_failed", "low_coverage"} and p["reason"] == "size_differs_out_of_support"


def test_tile_failure_statuses_keep_unknown_out_of_support_and_tell_the_cause():
    sa, sb = product(480, DESIGN), product(518, DESIGN)
    far = product(560, DESIGN)  # ΔU 80 px > 60
    p = stage([pair("A", "B", delta=80)], {"A": sa, "B": far})[0]
    assert p["tile_status"] == "out_of_support" and p["tile_p"] is None and verdict_of(p) == ("unknown", "size_differs_out_of_support", 3)
    assert p["axis_p_eff"] == 0.2
    tiny = product(480, DESIGN)
    tiny["full_shape"] = (900, 900)
    q = stage([pair("A", "B")], {"A": tiny, "B": sb})[0]
    assert q["tile_status"] == "geometry_unsupported" and verdict_of(q) == ("unknown", "size_differs_out_of_support", 3)


# ---- 대상이 아닌 쌍·cap ---------------------------------------------------------------------------------------------
def test_pairs_that_are_not_not_run_are_untouched():
    sigs = {"A": product(480, DESIGN), "B": product(518, DESIGN)}
    below, other = pair("A", "B", u=0.5), pair("A", "B")
    below.update(tile_status="usage_below_floor", tier="different", reason="size_differs_usage_below_floor", judge_case=2)
    other["tile_status"] = "not_candidate"
    before = copy.deepcopy([below, other])
    stage([below, other], sigs)
    assert [below, other] == before


def test_cap_tiles_the_highest_u_first_and_marks_the_rest_capped():
    sigs = {n: product(u, DESIGN) for n, u in (("A", 480), ("B", 518), ("C", 520))}
    low, high = pair("A", "B", u=0.8), pair("A", "C", u=0.99)
    stage([low, high], sigs, max_pairs=1)
    assert high["tile_status"] == "ok" and low["tile_status"] == "capped"
    assert low.get("tile_p") is None and low.get("tile_source") is None and verdict_of(low) == ("unknown", "size_differs_out_of_support", 3)


def test_cap_ties_break_on_product_names():
    sigs = {n: product(u, DESIGN) for n, u in (("A", 480), ("B", 518), ("C", 520))}
    first, second = pair("A", "B", u=0.9), pair("A", "C", u=0.9)
    stage([second, first], sigs, max_pairs=1)  # 입력 순서와 상관없다
    assert first["tile_status"] == "ok" and second["tile_status"] == "capped"


def test_max_pairs_zero_leaves_everything_not_run():
    sigs = {n: product(u, DESIGN) for n, u in (("A", 480), ("B", 518))}
    p = stage([pair("A", "B")], sigs, max_pairs=0)[0]
    assert p["tile_status"] == "not_run" and p.get("tile_p") is None and "tile_source" not in p and verdict_of(p) == ("unknown", "size_differs_out_of_support", 3)


# ---- shift 약속 ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("shift", [(0, 0), (3, -2), (-4, 5)])
def test_shift_promise_is_b_origin_to_a_origin(shift):
    """B[y, x] == A[y + dy, x + dx] (roi_box·compare·layer 정렬과 같은 약속)인 쌍에서 같은 shift 를 주면 tile_p 가 1 이다."""
    sa = product(480, DESIGN)
    sb = {**sa, "rasters": {k: {"mask": np.roll(r["mask"], (-shift[0], -shift[1]), axis=(0, 1))} for k, r in sa["rasters"].items()}}
    out = st.tile_pattern_axis(sa, sb, shift, sources(sa, sb), KEYS)
    assert out["tile_status"] == "ok" and out["tile_p"] >= 0.99
    p = stage([pair("A", "B", delta=0)], {"A": sa, "B": sb}, shifts={("A", "B"): shift})[0]
    assert p["tile_p"] >= 0.99
    # 부호를 뒤집으면 탐색 반경 밖이라 맞지 않는다: 약속이 방향까지 의미가 있다
    flipped = st.tile_pattern_axis(sa, sb, (-shift[0], -shift[1]), sources(sa, sb), KEYS)
    if max(abs(shift[0]), abs(shift[1])) > 2:
        assert flipped["tile_p"] is None or flipped["tile_p"] < 0.99


def test_screen_uses_the_same_shift_as_the_selected_shift_of_the_report():
    from psdl1.screen_report import selected_shift
    sa = {"um_per_px": 8.0}
    layer = {"metrics_anchor": "LAYER", "layer_shift_y_um": 24.0, "layer_shift_x_um": -16.0, "result": {"shift": (9, 9)}}
    assert selected_shift(layer, sa) == (3, -2)  # screen() 이 쓰는 align["shift"] (정수 이동) 과 같다
    assert selected_shift({"metrics_anchor": "CNT", "result": {"shift": (9, 9)}}, sa) == (9, 9)


# ---- 결정성·통계 ---------------------------------------------------------------------------------------------------
def test_same_input_gives_the_same_pairs():
    sigs = {n: with_layers(product(u, DESIGN)) for n, u in (("A", 480), ("B", 518), ("C", 520))}
    runs = [stage([pair("A", "B", u=0.8), pair("A", "C", u=0.99)], sigs, max_pairs=1) for _ in range(2)]
    assert runs[0] == runs[1]


def test_stats_and_cases_match_the_scores_columns():
    sigs = {n: product(u, DESIGN) for n, u in (("A", 480), ("B", 518), ("C", 520), ("D", 521))}
    pairs = [pair("A", "B", u=0.9), pair("A", "C", u=0.99), pair("A", "D", u=0.8)]
    below = pair("B", "C", u=0.5)
    below.update(tile_status="usage_below_floor", tier="different", reason="size_differs_usage_below_floor", judge_case=2)
    pairs.append(below)
    stage(pairs, sigs, max_pairs=2)
    summary = sr.judge_summary(pairs, {"p": {}, "u": {}}, MARGIN_V, None, None, {**SIZE, "max_pairs": 2})
    stats, cases = summary["size"]["stats"], summary["cases"]
    statuses = [p["tile_status"] for p in pairs]
    assert stats["pairs_capped"] == statuses.count("capped") == 1
    assert stats["pairs_tiled"] == sum(s not in {"capped", "usage_below_floor", "not_run", "not_candidate"} for s in statuses) == 2
    assert stats["pairs_usage_below_floor"] == 1
    assert stats["tile_status"] == {s: statuses.count(s) for s in sorted(set(statuses))}
    assert cases["size_differs_tiled"] == statuses.count("ok")
    assert cases["size_differs_tile_unsupported"] == sum(s in sr.TILE_FAILURE_STATUSES for s in statuses)
    assert cases["size_differs_tiled_different"] == sum(p["reason"] == "size_differs_below_pattern_floor_tiled" for p in pairs)
    assert cases["size_differs_out_of_support"] == sum(p["reason"] == "size_differs_out_of_support" for p in pairs)


# ---- screen() 연결 -------------------------------------------------------------------------------------------------
def fake_tile(p_by_pair):
    """tile_pattern_axis 를 고정값으로 바꾼다: 쌍(sa 의 full_shape 로는 못 가르므로 호출 순서)별 tile_p."""
    calls = []

    def run(sa, sb, shift, sources_, binding_keys, factor=8, tolerance=1, min_dice=0.3):
        calls.append({"shift": shift, "sources": sorted(sources_), "keys": list(binding_keys)})
        tile_p = p_by_pair[len(calls) - 1]
        out = {key: None for key in ("tile_status", "tile_p", "tile_size_y_cells", "tile_size_x_cells", "tile_count", "tile_applied", "tile_step_replaced",
                                     "tile_radius_y_cells", "tile_radius_x_cells", "tile_shift_min_y", "tile_shift_max_y", "tile_shift_min_x", "tile_shift_max_x")}
        out.update(tile_status="ok" if tile_p is not None else "unit_align_failed", tile_p=tile_p, tile_count=7)
        return out
    run.calls = calls
    return run


def test_screen_reapplies_the_table_with_tile_p(world, monkeypatch):
    fake = fake_tile([0.2, 0.9])  # 후보 (A1,B2) U .75 → 첫 번째, (A3,B2) U .95 → 정렬상 U 가 높은 이쪽이 먼저
    monkeypatch.setattr(st, "tile_pattern_axis", fake)
    result = size_run(world)
    pairs = by_pair(result)
    # U 내림차순: (A3,B2)=0.95 가 첫 호출(tile_p 0.2 → different), (A1,B2)=0.75 가 두 번째(tile_p 0.9 → close)
    assert verdict(pairs[("A3", "B2")]) == ("different", "size_differs_below_pattern_floor_tiled", 2)
    assert verdict(pairs[("A1", "B2")]) == ("unknown", "close_to_same_lineage", 3)
    assert pairs[("A3", "B2")]["tile_p"] == 0.2 and pairs[("A3", "B2")]["axis_p_eff"] == 0.2 and pairs[("A3", "B2")]["axis_p"] == 0.2
    assert pairs[("A1", "B2")]["axis_p_eff"] == 0.9 and pairs[("A1", "B2")]["axis_p"] == 0.2 and pairs[("A1", "B2")]["tile_count"] == 7
    assert all(pairs[k]["axis_p_eff"] == pairs[k]["axis_p"] for k in pairs if k not in {("A3", "B2"), ("A1", "B2")})
    assert [c["keys"] for c in fake.calls] == [sorted(c["keys"]) for c in fake.calls] and all(c["sources"] for c in fake.calls)
    judge = result["summary"]["judge"]
    assert judge["size"]["stats"]["pairs_tiled"] == 2 and judge["size"]["stats"]["tile_status"]["ok"] == 2
    assert judge["cases"]["size_differs_tiled"] == 2 and judge["cases"]["size_differs_tiled_different"] == 1
    assert judge["cases"]["size_differs_out_of_support"] == 1  # U 없음 쌍 (A2,B1) 만 남는다


def test_screen_max_pairs_zero_and_cap(world, monkeypatch):
    fake = fake_tile([0.9])
    monkeypatch.setattr(st, "tile_pattern_axis", fake)
    zero = size_run(world, judge_tile_max_pairs=0)
    assert not fake.calls and zero["summary"]["judge"]["size"]["stats"]["tile_status"]["not_run"] == 2
    capped = size_run(world, judge_tile_max_pairs=1)
    pairs = by_pair(capped)
    assert pairs[("A3", "B2")]["tile_status"] == "ok" and pairs[("A1", "B2")]["tile_status"] == "capped"
    stats = capped["summary"]["judge"]["size"]["stats"]
    assert stats["pairs_capped"] == 1 and stats["pairs_tiled"] == 1
    assert verdict(pairs[("A1", "B2")]) == ("unknown", "size_differs_out_of_support", 3)


def test_axis_p_eff_orders_the_cross_group_cap(world, monkeypatch):
    """tile 로 different 가 된 교차 그룹 쌍도 cap 에서 P 순서(tile_p 가 있으면 tile_p)로 가려진다."""
    monkeypatch.setattr(st, "tile_pattern_axis", fake_tile([0.1, 0.1]))
    groups = {"A1": "g1", "A2": "g1", "A3": "g1", "A4": "g1", "B1": "g2", "B2": "g2"}
    result = size_run(world, groups=groups, cross_group_fraction=0.0)
    pairs = by_pair(result)
    assert pairs[("A3", "B2")]["reason"] == "cross_group_capped" and pairs[("A3", "B2")]["axis_p_eff"] == 0.1
