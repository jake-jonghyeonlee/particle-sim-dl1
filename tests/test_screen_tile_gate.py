"""dl1-size/T4·T4b: tile 합성 게이트와 screen() 끝단 검증. 전체 게이트(U 560·1000, ΔU 8–60)는 scripts/measure_unit_stretch.py --tile 로 따로 돌리고,
여기서는 U 560 의 대표 집합(ΔU 8·38·40, 위상 0·4, 축 x·xy, 경로 2)과 지원 범위 밖(ΔU 41·60), B′ 4종(ΔU 38, P_var0 기준), 지원 범위 경계, 게이트 판정 함수, screen() 끝단을 본다.
"""
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

import psdl1.screen_tile as st
from psdl1.screen_relations import screen
from psdl1.split import SPLIT_VERSION
from test_pretrain import write_manifest
from test_screen_layers import write_sidecar

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "measure_unit_stretch.py"
spec = importlib.util.spec_from_file_location("measure_unit_stretch", SCRIPT)
mus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mus)


def describe(r):
    return (f"variant={r['variant']} U={r['unit']} ΔU={r['delta']} axis={r['axis']} phase={r['phase']} path={r['path']} status={r['tile_status']} "
            f"tile_p={r['tile_p']} P0={r['p0']} P_var0={r.get('p_var0')} axis_p={r['axis_p']} single_p={r['single_p']} tier={r['tier']}/{r['reason']}")


@pytest.fixture(scope="module")
def same():
    return mus.tile_measure(units=(560,), deltas=(8, 38, 40, 41, 60), phases=(0, 4), axes=("x", "xy"))


@pytest.fixture(scope="module")
def false():
    return mus.false_measure(units=(560,), deltas=(8, 38), axes=("xy",))


def test_same_design_keeps_tile_p_and_is_never_different(same):
    assert len(same) == 2 * 2 + 5 * 2 * 2 * 2  # ΔU=0 (위상 2 × 경로 2) + ΔU 5 × 축 2 × 위상 2 × 경로 2
    assert all(r["layer_applied"] for r in same if r["path"] == "layer")
    bad = [describe(r) for r in mus.same_failures(same)]
    assert not bad, f"{len(bad)} failing: " + "; ".join(bad)
    inside = [r for r in same if r["delta"] <= st.MAX_DELTA_U_PX]
    assert len(inside) == 2 * 2 + 3 * 2 * 2 * 2
    assert all(r["tile_status"] == "ok" and r["tile_count"] > 4 for r in inside)
    assert all(r["tile_source"] == ("layer" if r["path"] == "layer" else "binding") for r in same)
    # 같은 설계의 늘어난 쌍은 ΔU=0 과 같은 P(1.0 근처)를 지킨다. 중심 정렬 이동 변환이 없으면 크기가 다른 쌍은 모두 이 값을 못 낸다
    assert min(r["tile_p"] - r["p0"] for r in inside if r["delta"]) >= -mus.MARGIN_P


def test_same_design_beyond_the_support_range_is_out_of_support_without_p(same):
    outside = [r for r in same if r["delta"] > st.MAX_DELTA_U_PX]
    assert {r["delta"] for r in outside} == {41, 60}
    assert all(r["tile_status"] == "out_of_support" and r["tile_p"] is None for r in outside)
    assert all(r["tier"] == "unknown" and r["reason"] == "size_differs_out_of_support" for r in outside)


def test_same_design_tile_runs_are_deterministic_and_timed(same):
    assert all(r["repeat_equal"] for r in same)
    assert all(r["seconds"] >= 0 for r in same) and any(r["seconds"] > 0 for r in same)


def test_different_patterns_are_not_made_close_by_tile_alignment(false):
    n = len(mus.VARIANTS)
    assert len(false) == n * 2 + 2 * n * 2  # ΔU=0 기준(B′ 변형 × 경로 2) + ΔU 2 × B′ 변형 × 경로 2
    assert {r["variant"] for r in false} == set(mus.VARIANTS)
    bad = [describe(r) for r in mus.false_failures(false)]
    assert not bad, f"{len(bad)} failing: " + "; ".join(bad)
    # P_var0 가 낮은 변형이 실제로 있어야 이 검증이 의미가 있다. ΔU=0 쌍은 p_var0 == single_p, 늘어난 쌍의 p_var0 는 그 변형의 ΔU=0 값이다
    low = [r for r in false if r["delta"] and r["p_var0"] < mus.FALSE_P]
    assert {"mixed", "cutdrift", "half_pixel"} <= {r["variant"] for r in low}  # R2 adv:2: half_pixel 은 P_var0 0.25–0.30 쪽. R1 adv:1: P_var0 가 낮은 쪽에 퍼진 변형(전체 게이트는 U 1000 의 dense 도 0.34 로 포함한다)
    assert {"dense", "keep_pixel"} <= {r["variant"] for r in false if r["delta"] and r["p_var0"] < mus.RECORD_P}  # 기록 구간(0.35–0.40)까지 퍼진 변형
    assert all(r["tier_low_u"] in {"different", "unknown"} for r in false)  # U 낮음 판정을 같이 잰다
    assert all(r["p_var0"] == r["single_p"] for r in false if r["delta"] == 0)
    assert all(r["tile_p"] is None or r["tile_p"] <= max(r["p_var0"], r["p_var0_best"]) + mus.FALSE_SLACK for r in low)  # 기준은 두 경로 중 높은 단일 P(gate 와 같다)


def test_support_range_boundary_needs_the_minimum_tile_width():
    records = mus.support_measure()
    by_unit = {r["unit"]: r for r in records}
    ok_unit, small_unit = mus.SUPPORT_UNITS
    assert by_unit[ok_unit]["tile_status"] == "ok" and by_unit[ok_unit]["tile_size_y_cells"] == st.TILE_MIN_CELLS and by_unit[ok_unit]["tile_p"] is not None
    assert by_unit[small_unit]["tile_status"] == "tile_too_small" and by_unit[small_unit]["tile_p"] is None
    assert not mus.support_failures(records)


def test_repeat_unit_input_shift_does_not_move_the_start_point_away_from_the_origin():
    """같은 설계의 늘어난 쌍(U 552, ΔU 38)에 입력 이동을 한 unit(y)+반 unit(x) 밀어 넣거나 실자료 M13·WF3 의 layer_shift 를 넣어도 (0, 0) 이 골라지고 tile 이 맞는다."""
    records = mus.repeat_measure(cases=((552, 38),))
    assert [r["shift_name"] for r in records] == ["none", "minus", "plus", "real"]
    assert next(r for r in records if r["shift_name"] == "real")["input_shift"] == mus.REAL_SHIFT
    assert {r["origin"] for r in records} == {"zero"}
    assert all(r["tile_status"] == "ok" and r["coverage"] >= st.COVERAGE_MIN and r["tile_p"] >= r["p0"] - mus.MARGIN_P for r in records)
    assert abs(next(r for r in records if r["shift_name"] == "minus")["origin_shift"][0]) > 60  # 입력 이동이 실제로 한 unit 가까이 어긋나 있다
    assert not mus.repeat_failures(records)


def test_repeat_failures_rules():
    base = {"origin": "zero", "tile_status": "ok", "tile_p": 1.0, "p0": 1.0, "tier": "unknown", "repeat_equal": True}
    assert not mus.repeat_failures([base])
    assert mus.repeat_failures([{**base, "origin": "shift"}]) and mus.repeat_failures([{**base, "tile_p": 0.94}]) and mus.repeat_failures([{**base, "tier": "different"}])
    assert mus.repeat_failures([{**base, "tile_status": "low_coverage", "tile_p": None}])


def test_in_range_pair_with_a_too_narrow_tile_must_report_tile_too_small():
    narrow = {"unit": 552, "delta": 40}  # 552 // 80 = 6 칸 < 7
    assert not mus.in_tile_support(narrow) and mus.expected_out_status(narrow) == "tile_too_small"
    assert mus.in_tile_support({"unit": 560, "delta": 40}) and mus.in_tile_support({"unit": 552, "delta": 38})
    assert mus.expected_out_status({"unit": 560, "delta": 41}) == "out_of_support"


def rec(**kw):
    base = {"delta": 38, "tile_status": "ok", "tile_p": 0.95, "p0": 1.0, "repeat_equal": True, "tier": "unknown", "tier_low_u": "different", "shift_error": 0.0, "p_var0": None}
    return {**base, **kw}


def test_same_failures_rules():
    assert not mus.same_failures([rec(), rec(delta=0, tile_p=1.0, tier="different")])  # ΔU=0 은 tier 를 단정하지 않는다
    assert mus.same_failures([rec(tile_p=0.94)]) and mus.same_failures([rec(tier="different")]) and mus.same_failures([rec(tile_status="low_coverage", tile_p=None)])
    assert mus.same_failures([rec(repeat_equal=False)])
    # 입력 이동이 크게 어긋난 쌍은 tile 성공을 요구하지 않지만 different 로 떨어지면 실패다
    assert not mus.same_failures([rec(shift_error=128.0, tile_status="low_coverage", tile_p=None)])
    assert mus.same_failures([rec(shift_error=128.0, tile_status="low_coverage", tile_p=None, tier="different")])
    # 지원 범위 밖은 out_of_support 여야 한다
    assert not mus.same_failures([rec(delta=41, tile_status="out_of_support", tile_p=None)])
    assert mus.same_failures([rec(delta=41, tile_status="ok")])


def test_false_failures_use_the_unstretched_variant_p_not_the_stretched_single_p():
    low = dict(p_var0=0.2)
    assert not mus.false_failures([rec(tile_p=0.25, tier="different", **low), rec(tile_p=None, **low)])
    assert mus.false_failures([rec(tile_p=0.30, tier="different", **low)])  # tile_p > P_var0 + slack
    assert mus.false_failures([rec(tile_p=0.25, tier="unknown", **low)])  # tile_p 가 있는데 different 가 아니다
    assert mus.false_failures([rec(tile_p=0.25, tier="different", tier_low_u="unknown", **low)])  # U 가 낮은 판정도 different 여야 한다(혼합 판정)
    assert mus.false_failures([rec(tile_p=None, tier_low_u="unknown", **low)])  # tile 실패 + U 낮음 → tile_p 없음 행은 different
    assert not mus.false_failures([rec(tile_p=0.26, tier="different", p_var0=0.12, p_var0_best=0.27)])  # anchor 경로 P_var0 가 낮아도 경로 중 높은 값이 기준
    assert mus.false_failures([rec(tile_p=0.36, tier="different", p_var0=0.12, p_var0_best=0.27)])
    assert not mus.false_failures([rec(tile_p=0.95, tier="unknown", p_var0=0.8)])  # 원래 비슷한 변형은 기준 밖
    assert not mus.false_failures([rec(delta=0, tile_p=0.95, tier="unknown", p_var0=0.2)])  # ΔU=0 은 기준 측정용
    assert mus.false_failures([rec(repeat_equal=False, p_var0=0.8)])


# ---- screen() 끝단 ---------------------------------------------------------------------------------------------------
def build_world(root, spec_list):
    """spec_list: [(이름, 계보, uy, ux, variant)] → manifest, split 경로와 Layer 사이드카를 root 에 쓴다."""
    products, layers = [], {}
    for name, lineage, uy, ux, variant in spec_list:
        product, layer_masks = mus.build(name, uy, ux, 0, variant)
        size = next(iter(product.elements)).mask.shape[0]  # write_manifest 의 frame 은 0.05 µm/px 고정이라 unit_size_um 도 맞춘다(세 제품이 같은 배율이면 판정은 같다)
        products.append(replace(product, lineage_id=lineage, unit_size_um=(size * 0.05, size * 0.05)))
        layers[name] = layer_masks
    manifest = write_manifest(root, products, 64)
    write_sidecar(root, layers)
    split = {"schema_version": SPLIT_VERSION, "train": [s[0] for s in spec_list], "validation": [], "test": []}
    return manifest, split


def test_screen_end_to_end_stretched_copy_is_tiled_and_different_pattern_is_not_close(tmp_path, monkeypatch):
    U = 560
    manifest, split = build_world(tmp_path, [("PA", "LA", U, U, None), ("PS", "LS", U + 38, U + 38, None), ("PB", "LB", U + 38, U + 38, "mixed")])
    calls = []
    original = st.origin_shift
    monkeypatch.setattr(st, "origin_shift", lambda sa, sb, shift: calls.append((tuple(sa["shape"]), tuple(sb["shape"]), tuple(shift), original(sa, sb, shift))) or original(sa, sb, shift))
    result = screen(manifest, split, factor=8, tolerance=1, margin=0.05, judge="three-axis", judge_floor_p=0.45, judge_floor_u=0.9, judge_size_u_floor=0.75)
    pairs = {(p["product_a"], p["product_b"]): p for p in result["pairs"]}
    stretched, other, between = pairs[("PA", "PS")], pairs[("PA", "PB")], pairs[("PB", "PS")]
    assert stretched["tile_status"] == "ok" and stretched["tile_p"] is not None and stretched["tile_p"] >= 0.45
    assert stretched["tier"] != "different" and stretched["reason"] != "size_differs_below_pattern_floor_tiled"
    assert stretched["axis_p_eff"] == stretched["tile_p"] and stretched["metrics_anchor"] == "LAYER"
    assert other["reason"] != "close_to_same_lineage"
    assert other["tile_p"] is None or other["tile_p"] < 0.45
    assert between["reason"] != "close_to_same_lineage"  # 같은 크기의 다른 패턴: 단일 P 로 판정한다
    stats = result["summary"]["judge"]["size"]["stats"]
    assert stats["pairs_tiled"] >= 1 and stats["tile_status"]["ok"] >= 1
    assert stretched["tile_p"] >= 0.9  # 같은 설계의 늘어난 쌍은 ΔU=0 에 가까운 P 를 낸다(중심 정렬 이동을 원점 이동으로 바꾼 뒤)
    # 크기가 다른 쌍(PA 560 vs PS 598)에서 tile 단계 호출마다 원점 이동 변환이 쓰였다: 원점 이동 = 넘어온 이동 − (A 래스터 shape − B 래스터 shape)//2
    assert calls and any(a != b for a, b, _, _ in calls)
    for shape_a, shape_b, shift, origin in calls:
        center = tuple((shape_a[ax] - shape_b[ax]) // 2 for ax in (0, 1))
        assert origin == tuple(shift[ax] - center[ax] for ax in (0, 1))
    assert result["pairs"] == screen(manifest, split, factor=8, tolerance=1, margin=0.05, judge="three-axis", judge_floor_p=0.45, judge_floor_u=0.9,
                                     judge_size_u_floor=0.75)["pairs"]  # 같은 입력은 같은 쌍 목록
