"""dl1-judge/T2: 경우 3 측정 게이트(PLAN v1.2). unit 크기가 다른 합성 쌍이 three-axis 판정에서 different 로 밀리지 않는다.

논리: 판정표는 다른 계보 쌍의 초기 tier 를 P 만으로 정한다. P < p_floor − margin 이면 different, 아니면 unknown 이다.
p_floor 는 같은 계보 쌍(크기가 같다)의 P 분포에서 보정하므로 p_floor ≤ P(ΔU=0) 이다(여기서는 p_floor = P(ΔU=0) 로 둔다).
(a) 지원 범위 안(LAYER 정렬, |ΔU| ≤ 7): P(ΔU) ≥ P(0) − 0.05 이면 P ≥ p_floor − margin 이라 different 가 아니다. 이 부등식과 tier 를 함께 단정한다.
(b) 지원 범위 밖(anchor 정렬 전부, 기록용 ΔU 8·12·20·40): P 는 떨어질 수 있다. v1.2 G-size 안전장치(size_differs 이고 범위 밖이면 unknown,
    size_differs_out_of_support)를 넣은 판정식의 tier 가 different 가 아님을 단정한다. G-size 는 T3 에서 src 에 들어가므로 여기서는 helper 판정식에만 있다.
(a)·(b) 모두 판정식 밖으로 느슨하게 하지 않는다. 합성 제품은 scripts/measure_unit_stretch.py 가 만든다.
ΔU 0–7 전 조합(축 x·y·xy × 위상 0–7 × 경로 2)을 쓴다(약 1 분). ΔU 8–40 은 위상 0·4 만 쓴다.
"""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "measure_unit_stretch.py"
spec = importlib.util.spec_from_file_location("measure_unit_stretch", SCRIPT)
mus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mus)


@pytest.fixture(scope="module")
def gate():
    return mus.measure(phases=range(8), deltas=mus.GATE_DELTAS)


@pytest.fixture(scope="module")
def record(gate):
    out = mus.measure(phases=(0, 4), deltas=mus.RECORD_DELTAS)
    mus.classify(out, {(r["path"], r["phase"]): r["axis_p"] for r in gate if r["delta"] == 0})
    return out


def describe(r):
    return f"ΔU={r['delta']} axis={r['axis']} phase={r['phase']} path={r['path']} P={r['axis_p']} P0={r['p0']} tier={r['tier']}/{r['reason']}"


def test_every_combination_was_measured_and_the_layer_path_used_the_layer_shift(gate, record):
    assert len(gate) == 8 * 2 + 7 * 3 * 8 * 2 and len(record) == 4 * 3 * 2 * 2
    assert all(r["axis_p"] is not None and r["p0"] is not None for r in gate + record)
    assert all(r["layer_applied"] and r["anchor"] == "LAYER" for r in gate + record if r["path"] == "layer")
    assert all(r["size_delta_unit_px_y"] == (r["delta"] if "y" in r["axis"] else 0) and r["size_delta_unit_px_x"] == (r["delta"] if "x" in r["axis"] else 0)
               for r in gate + record if r["delta"])


def test_supported_range_is_layer_with_delta_up_to_7(gate, record):
    assert all(r["in_support"] for r in gate if r["path"] == "layer")
    assert not any(r["in_support"] for r in gate + record if r["path"] == "anchor" and r["delta"])
    assert not any(r["in_support"] for r in record)


def test_a_inside_support_axis_p_drops_at_most_margin_and_tier_is_not_different(gate):
    inside = [r for r in gate if r["in_support"]]
    assert len(inside) == 8 + 7 * 3 * 8
    failing = [describe(r) for r in inside if r["axis_p"] < r["p0"] - mus.MARGIN_P or r["tier"] == "different"]
    assert not failing, f"{len(failing)} failing: " + "; ".join(failing)


def test_b_outside_support_tier_with_g_size_is_not_different(gate, record):
    outside = [r for r in gate + record if not r["in_support"]]
    assert len(outside) == 7 * 3 * 8 + 8 + len(record)  # anchor 경로 전부(ΔU 0 포함) + 기록용 전부
    different = [describe(r) for r in outside if r["tier"] == "different"]
    assert not different, f"{len(different)} different: " + "; ".join(different)
    # G-size 가 실제로 일한 조합이 있다(없으면 이 단정은 아무것도 보호하지 않는다)
    assert any(r["reason"] == "size_differs_out_of_support" for r in outside)
    assert not mus.gate_failures(gate + record)


def test_tier_formula_boundaries():
    tier = mus.tier
    assert tier(None, 1.0) == ("unknown", "no_pattern_value")
    assert tier(0.95, 1.0)[0] == "unknown" and tier(0.949, 1.0)[0] == "different" and tier(1.0, 1.0) == ("unknown", "close_to_same_lineage")
    assert tier(0.5, 1.0, size_tag="size_differs", in_support=True)[0] == "different"
    assert tier(0.5, 1.0, size_tag="size_differs", in_support=False) == ("unknown", "size_differs_out_of_support")
    assert tier(0.5, 1.0, size_tag="same", in_support=False)[0] == "different"  # 같은 크기 쌍은 G-size 의 영향이 없다
    assert tier(0.3, 1.0)[1] == "below_pattern_floor"
    from psdl1.screen_relations import judge_verdict
    for p, floor in ((0.47, 0.52), (0.4699, 0.52), (0.52, 0.52), (0.5199, 0.52)):  # 0.52 - 0.05 = 0.47000000000000003 경계
        for size_tag in ("same", "size_differs"):
            axes = {"axis_p": p, "axis_u": 1.0, "size_tag": size_tag, "size_delta_unit_px_y": 0, "size_delta_unit_px_x": 0}
            expected = judge_verdict(axes, "LAYER", False, floor, 0.0, 0.05)
            assert mus.tier(p, floor, margin=0.05, size_tag=size_tag, in_support=True) == expected[:2], (p, floor, size_tag)
    assert mus.tier(0.47, 0.52)[0] == "unknown"
    assert mus.supported("LAYER", 7, -7) and not mus.supported("LAYER", 8, 0) and not mus.supported("ELECTRODES", 0, 0)
