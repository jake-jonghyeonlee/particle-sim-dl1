"""dl1-judge/T5: three-axis 일 때만 review.html 에 P·U·경우 표시와 새 reason 이 들어가고, legacy review.html 은 바이트 그대로다."""
import hashlib
import re

import pytest

import psdl1.screen_report as report
from psdl1.screen_relations import render_review
from test_screen_judge import MARGIN, SAME, SMALL, by_pair, world  # noqa: F401  (world is a fixture)

NEW_REASONS = ("pattern_same_usage_differs", "size_differs_out_of_support", "pattern_same_usage_unknown", "no_pattern_value")
AXES = {**SAME,
        ("A1", "B1"): {"axis_p": None, "axis_u": 0.9},                 # no_pattern_value
        ("A1", "B2"): {"axis_p": 0.2, "axis_u": 0.9, **SMALL},         # size_differs_out_of_support (LAYER 정렬이 아님)
        ("A2", "B1"): {"axis_p": 0.45, "axis_u": 0.9},                 # near_boundary
        ("A3", "B1"): {"axis_p": 0.7, "axis_u": None},                 # pattern_same_usage_unknown
        ("A3", "B2"): {"axis_p": 0.7, "axis_u": 0.3},                  # pattern_same_usage_differs
        ("A4", "B1"): {"axis_p": 0.3, "axis_u": 0.9}}                  # different (경우 2), 경계에 가까워 검토 목록에 든다
LEGACY_SHA256 = "865af9a634fa0f39b76b87cfa2985d5405cadb34fa8e136e878c2352e69065c2"  # 기준 2cd695f 코드로 같은 fixture 에서 구한 legacy review.html 의 sha256


def cards_of(page):
    return page.split('<section class="card">')[1:]


def test_three_axis_second_line_has_p_u_size_case_and_verdict(world):
    result = world(AXES, margin=MARGIN)
    page = render_review(result, top_pairs=60)
    p = by_pair(result)[("A4", "B1")]
    line = f'P {p["axis_p"]:.2f} (기준 {p["judge_p_floor"]:.2f}) · U {p["axis_u"]:.2f} (기준 {p["judge_u_floor"]:.2f}) · S '
    assert line in page and "크기 같음 · 경우 2 → different" in page
    assert "크기 다름 · 경우 3 → 모름" in page  # A1 ↔ B2
    assert "P — (기준 " in page and "경우 — → 모름" in page  # 패턴 값이 없는 쌍 A1 ↔ B1


def test_three_axis_figure_has_p_and_u_rows_with_floor_lines(world):
    result = world(AXES, margin=MARGIN)
    page = render_review(result, top_pairs=60)
    svg = page[page.index("<svg"):page.index("</svg>")]
    assert re.findall(r'data-metric="(\w+)"', svg) == [*report.DIST_METRICS, "axis_p", "axis_u"]
    judge = result["summary"]["judge"]
    for axis, key in (("axis_p", "p"), ("axis_u", "u")):
        group = svg[svg.index(f'data-metric="{axis}"'):]
        group = group[:group.index("</g>")]
        assert f"floor(global) {judge[key]['global']:.3f}" in group and '<line class="floor"' in group


def test_three_axis_group_floor_is_a_dashed_line_in_the_axis_rows(world):
    groups = {"A1": "g1", "A2": "g1", "A3": "g1", "A4": "g1", "B1": "g1", "B2": "g1"}
    result = world(AXES, groups=groups, margin=MARGIN)
    assert result["summary"]["judge"]["p"]["per_group"]
    svg = report.distribution_svg(result)
    group = svg[svg.index('data-metric="axis_p"'):]
    assert 'class="floor-group"' in group[:group.index("</g>")]


def test_three_axis_table_has_p_u_case_columns_and_header_notes(world):
    result = world(AXES, margin=MARGIN)
    page = render_review(result, top_pairs=60)
    assert "<th>기준</th><th>P</th><th>U</th><th>경우</th><th>이유</th>" in page
    assert "판정 모드: three-axis" in page
    assert "다른 계보 쌍의 초기 tier 는 P 로 정하고 그 뒤 그룹 cap 을 적용한다" in page
    assert "G-size:" in page
    rows = page[page.index("<h2>검토 대상 전체</h2>"):page.index("<h2>카드</h2>")].count("<tr><td>")
    assert rows == sum(bool(p["review"]) for p in result["pairs"])


def test_new_reasons_enter_the_review_list_and_have_text(world):
    result = world(AXES, margin=MARGIN)
    pairs = by_pair(result)
    expected = {("A1", "B1"): "no_pattern_value", ("A1", "B2"): "size_differs_out_of_support", ("A3", "B1"): "pattern_same_usage_unknown", ("A3", "B2"): "pattern_same_usage_differs"}
    for key, reason in expected.items():
        assert pairs[key]["reason"] == reason and reason in pairs[key]["review"]
    assert {k for k, p in pairs.items() if set(p["review"]) & set(NEW_REASONS)} == set(expected)
    by_reason = result["summary"]["review"]["by_reason"]
    assert all(by_reason[r] == 1 for r in NEW_REASONS)
    page = render_review(result, top_pairs=60)
    for reason in NEW_REASONS:
        assert report.REASON_TEXT[reason] in page


def legacy_page(world):
    return render_review(world(AXES, judge="legacy"), top_pairs=60)


def test_legacy_page_has_no_three_axis_output(world):
    page = legacy_page(world)
    assert "axis_p" not in page and "three-axis" not in page and "G-size" not in page and "경우" not in page
    assert "<th>P</th>" not in page and "<th>기준</th><th>이유</th>" in page
    for reason in NEW_REASONS:
        assert reason not in page and report.REASON_TEXT[reason] not in page


def test_legacy_page_is_byte_identical_to_the_baseline(world):
    assert hashlib.sha256(legacy_page(world).encode()).hexdigest() == LEGACY_SHA256
