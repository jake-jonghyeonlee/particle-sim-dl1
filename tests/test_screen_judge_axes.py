"""dl1-judge/T1: 판정 세 축 입력값(axis_p·axis_p_layer·axis_u·size_*) 순수 함수. 합성 행·signature 로 손계산 반례를 단정한다."""
import numpy as np
import pytest

from psdl1.screen_relations import pair_axes, pair_metric_columns, pair_metric_rows, product_signature
from test_screen_metrics import TOL, product, rect, sig


def row(kind, key, status="both", f1=1.0):
    return {"kind": kind, "key": key, "status": status, "boundary_f1": f1}


def shapes(a, b):
    return {"full_shape": a}, {"full_shape": b}


SAME = shapes((64, 64), (64, 64))


def test_only_a_bindings_give_no_axis_p_and_zero_axis_u():
    rows = [row("binding", "gate/l1", "only_a", 0.0), row("role", "gate", "only_a", 0.0)]
    axes = pair_axes(rows, "electrodes", *SAME)
    assert axes["axis_p"] is None and axes["axis_u"] == 0.0


def test_both_row_with_missing_boundary_f1_is_left_out_and_no_valid_row_gives_none():
    rows = [row("binding", "a/l1", f1=0.5), row("binding", "b/l1", f1=None), row("binding", "c/l1", "only_b", 0.0)]
    assert pair_axes(rows, "electrodes", *SAME)["axis_p"] == 0.5
    assert pair_axes([row("binding", "a/l1", f1=None)], "electrodes", *SAME)["axis_p"] is None
    assert pair_axes([], "electrodes", *SAME)["axis_p"] is None and pair_axes([], "electrodes", *SAME)["axis_u"] is None


def test_axis_p_is_a_plain_mean_rounded_to_six_places():
    rows = [row("binding", "a/l1", f1=1.0), row("binding", "b/l1", f1=0.0), row("binding", "c/l1", f1=0.0), row("role", "a", f1=0.9)]
    assert pair_axes(rows, "electrodes", *SAME)["axis_p"] == round(1 / 3, 6)


def test_axis_u_equals_the_existing_usage_jaccard_electrodes():
    rows = [row("binding", "a/l1"), row("binding", "b/l1", "only_a", 0.0), row("binding", "c/l1", "only_b", 0.0), row("binding", "d/l1")]
    assert pair_axes(rows, "electrodes", *SAME)["axis_u"] == pair_metric_columns(rows, "electrodes")["usage_jaccard_electrodes"] == 0.5


def test_axis_p_layer_only_for_layer_anchor():
    rows = [row("binding", "a/l1", f1=1.0), row("layer", "m1", f1=0.25), row("layer", "m2", f1=0.75), row("layer", "m3", "only_a", 0.0), row("layer", "m4", f1=None)]
    on = pair_axes(rows, "LAYER", *SAME)
    assert on["axis_p_layer"] == 0.5 and on["axis_p"] == 1.0  # layer 행은 axis_p 에 섞이지 않는다
    assert pair_axes(rows, "electrodes", *SAME)["axis_p_layer"] is None
    assert pair_axes([row("binding", "a/l1")], "LAYER", *SAME)["axis_p_layer"] is None


def test_a1_a_missing_binding_in_a_shared_role_does_not_lower_axis_p():
    """같은 role 에 binding 둘. B 에는 l2 가 없다. l1 형상은 같다 → binding 단위 P 는 1.0, role 행 평균이었다면 낮다."""
    shape = (128, 128)
    left, right = rect(shape, 16, 112, 8, 40), rect(shape, 16, 112, 80, 120)
    a = product("A", shape, {("gate", "l1"): left, ("gate", "l2"): right})
    b = product("B", shape, {("gate", "l1"): left})
    sa, sb = sig(a), sig(b)
    rows = pair_metric_rows(sa, sb, (0, 0), tolerance=TOL)
    by = {(r["kind"], r["key"]): r for r in rows}
    assert by[("binding", "gate/l1")]["boundary_f1"] == 1.0 and by[("binding", "gate/l2")]["status"] == "only_a"
    role_mean = by[("role", "gate")]["boundary_f1"]
    assert role_mean < 0.9  # role 합집합 형상이 달라져 사용 차이가 섞인다
    axes = pair_axes(rows, "electrodes", sa, sb)
    assert axes["axis_p"] == 1.0 and axes["axis_p"] - role_mean > 0.1
    assert axes["axis_u"] == 0.5


def test_size_differs_with_signed_delta_and_ratio():
    axes = pair_axes([], "electrodes", *shapes((100, 200), (110, 180)))
    assert axes["size_tag"] == "size_differs"
    assert axes["size_ratio_y"] == round(100 / 110, 6) and axes["size_ratio_x"] == round(180 / 200, 6)
    assert axes["size_delta_unit_px_y"] == 5 and axes["size_delta_unit_px_x"] == -10
    assert isinstance(axes["size_delta_unit_px_y"], int)
    swapped = pair_axes([], "electrodes", *shapes((110, 180), (100, 200)))
    assert swapped["size_delta_unit_px_y"] == -5 and swapped["size_delta_unit_px_x"] == 10 and swapped["size_ratio_y"] == axes["size_ratio_y"]


def test_same_shape_is_same_with_unit_ratio_and_zero_delta():
    axes = pair_axes([], "electrodes", *SAME)
    assert axes["size_tag"] == "same" and axes["size_ratio_y"] == axes["size_ratio_x"] == 1.0
    assert axes["size_delta_unit_px_y"] == axes["size_delta_unit_px_x"] == 0


def test_only_one_axis_differs():
    axes = pair_axes([], "electrodes", *shapes((64, 64), (64, 72)))
    assert axes["size_tag"] == "size_differs" and axes["size_ratio_y"] == 1.0 and axes["size_ratio_x"] == round(64 / 72, 6)
    assert axes["size_delta_unit_px_y"] == 0 and axes["size_delta_unit_px_x"] == 4


def test_odd_difference_keeps_a_half_pixel_float():
    axes = pair_axes([], "electrodes", *shapes((64, 64), (65, 64)))
    assert axes["size_delta_unit_px_y"] == 0.5 and axes["size_delta_unit_px_x"] == 0


def test_signature_full_shape_is_used_for_real_signatures():
    shape = (64, 64)
    m = rect(shape, 8, 40, 8, 40)
    sa, sb = sig(product("A", shape, {("gate", "l1"): m})), sig(product("B", (64, 80), {("gate", "l1"): rect((64, 80), 8, 40, 8, 40)}))
    assert pair_axes([], "electrodes", sa, sb)["size_delta_unit_px_x"] == 8
