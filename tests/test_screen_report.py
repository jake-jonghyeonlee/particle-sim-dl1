"""screen-metrics/T8: review.html 의 판단용 이미지(분포 그림, 두 정렬 근거, 썸네일, binding 표, 층별 A | B | 차이). 값은 바꾸지 않는다."""
import json
import os
import re

import pytest

import psdl1.screen_report as report
from conftest import FAMILIES, make_product
from psdl1.screen_relations import main, render_review, screen
from psdl1.split import SPLIT_VERSION
from test_pretrain import write_manifest
from test_screen_align import blobs, put
from test_screen_layers import NAMES, layer_masks, write_sidecar
from test_screen_relations import FACTOR, case, embed  # noqa: F401  (case is a fixture)

BYTE_BUDGET = 12 * 1024 * 1024


def all_review(result):
    for p in result["pairs"]:
        p["review"] = ["near_boundary"]
    return result


def cards_of(page):
    return page.split('<section class="card">')[1:]


def layered_case(root):
    write_sidecar(root, {"BUS_R00": layer_masks((30, 30)), "BUS_T00": layer_masks((10, 20))})


def test_page_is_self_contained_and_has_the_distribution_figure(case):
    manifest, split, root, _ = case
    layered_case(root)
    result = all_review(screen(manifest, split, factor=FACTOR, margin=0.0, floor=0.5))
    page = render_review(result, top_pairs=60)
    assert "http://" not in page and "https://" not in page and 'src="//' not in page and "url(" not in page
    svg = page[page.index("<svg"):page.index("</svg>")]
    assert re.findall(r'data-metric="(\w+)"', svg) == list(report.DIST_METRICS) and len(report.DIST_METRICS) == 7
    assert 'class="floor"' in svg and "AUC" in svg and "n=" in svg
    assert page.index("<svg") < page.index("<h2>검토 대상 전체</h2>") < page.index("<h2>카드</h2>")  # 분포 그림 → 전체 표 → 카드


def test_cards_are_capped_and_have_two_thumbnails_each(case):
    manifest, split, root, _ = case
    layered_case(root)
    result = all_review(screen(manifest, split, factor=FACTOR, margin=0.0))
    for cap in (60, 2):
        cards = cards_of(render_review(result, top_pairs=cap))
        assert len(cards) == min(cap, len(result["pairs"])) and all(card.count('class="thumb"') == 2 for card in cards)


def test_layer_pair_shows_both_alignments_and_folded_layer_tiles(case):
    manifest, split, root, _ = case
    layered_case(root)
    result = all_review(screen(manifest, split, factor=FACTOR, margin=0.0))
    cards = {p["product_a"] + p["product_b"]: c for p, c in zip(sorted(result["pairs"], key=lambda p: (abs(p["similarity"] - p["floor"]), p["product_a"], p["product_b"])),
                                                                 cards_of(render_review(result, top_layers=2)))}
    layered = cards["BUS_R00BUS_T00"]
    assert "L1 Dice(첫 층)" in layered and "결정 층 Dice" in layered and "metrics_anchor LAYER" in layered and "정렬 기준 CNT" in layered and "alignment_score" in layered
    assert "Layer 색 겹침" in layered and layered.count('class="group"') == len(NAMES) and "나머지 1층" in layered and layered.count("<details>") == 1
    assert layered.index("<details>") > layered.index("차이가 큰 층 순")
    wide = cards_of(render_review(result, top_layers=4))
    assert all("<details>" not in c for c in wide if "Layer 색 겹침" in c)
    plain = cards["COMB_R00COMB_T00"]
    assert "Layer 이동 없음" in plain and "layer_missing" in plain and "전극 합집합 + CNT" in plain and "Layer 색 겹침" not in plain


def test_pair_without_layers_gets_binding_tiles_and_a_table_with_the_cnt_row(case):
    manifest, split, *_ = case
    result = all_review(screen(manifest, split, factor=FACTOR, margin=0.0))
    for card in cards_of(render_review(result, top_layers=2)):
        assert 1 <= card.count('class="group"') <= 2 and "binding별 A | B | 차이" in card and "<details>" not in card
        assert "<td>CNT</td>" in card and "경계 F1" in card and 'class="bar"' in card
        assert "gate/layer1" in card or "layer1" in card  # 모든 binding 이 표에 있다
    pair = result["pairs"][0]
    table = report.binding_table(pair)
    assert table.count("<tr>") == 1 + sum(r["kind"] in {"binding", "cnt"} for r in pair["metric_rows"])


def test_dense_distribution_switches_to_a_histogram(case, monkeypatch):
    manifest, split, *_ = case
    result = all_review(screen(manifest, split, factor=FACTOR, margin=0.0))
    points = report.distribution_svg(result)
    assert "<circle" in points and "<rect" not in points and "히스토그램" not in points
    monkeypatch.setattr(report, "DENSE_POINTS", 3)
    bars = report.distribution_svg(result)
    assert "<rect" in bars and "<circle" not in bars and "히스토그램" in bars and bars.count("<rect") % report.HIST_BINS == 0


def test_blank_metric_values_are_not_drawn_and_n_counts_only_values(case):
    manifest, split, *_ = case
    result = all_review(screen(manifest, split, factor=FACTOR, margin=0.0))  # Layer 가 없어 pattern_similarity 는 모두 빈칸
    svg = report.distribution_svg(result)
    group = svg[svg.index('data-metric="pattern_similarity"'):]
    group = group[:group.index("</g>")]
    assert "n=0" in group and "<circle" not in group and "AUC –" in group


def test_cli_values_stay_the_same_and_summary_records_the_html_size(case, tmp_path, capsys):
    manifest, split, root, _ = case
    layered_case(root)
    split_path = root / "split.json"
    split_path.write_text(json.dumps(split))
    runs = []
    for name, extra in (("a", []), ("b", ["--review-layers", "1", "--review-pairs", "2"])):
        main(["--manifest", str(manifest), "--split", str(split_path), "--out", str(tmp_path / f"{name}.csv"), "--factor", str(FACTOR), "--margin", "0.0", *extra])
        capsys.readouterr()
        summary = json.loads((tmp_path / f"{name}.csv.summary.json").read_text(encoding="utf-8"))
        assert summary["review"]["html_bytes"] == (tmp_path / f"{name}.csv.review.html").stat().st_size
        del summary["review"]["html_bytes"]
        runs.append([(tmp_path / f"{name}.csv{suffix}").read_bytes() for suffix in ("", ".scores.csv", ".bindings.csv")] + [json.dumps(summary, sort_keys=True)])
    assert runs[0] == runs[1]  # 렌더링 인수는 어떤 값도 바꾸지 않는다


def synthetic(root, count=12, layers=10, frame=96):
    """계열 6개 × 2제품, 제품마다 Layer `layers`장."""
    names = [f"layer{k:02d}" for k in range(layers)]
    families = list(FAMILIES)
    products, per_product = [], {}
    for i in range(count):
        family = families[i % len(families)]
        offset = (4 + 7 * (i % 5), 6 + 5 * (i % 4))
        product = embed(make_product(family, 48), f"P{i:02d}", f"L{i // 2}", frame, offset)
        products.append(product)
        per_product[product.product_id] = {n: put((frame, frame), blobs((frame * 5 // 12,) * 2, 300 + k + (i // 2) * 17, count=7 * (frame // 96) ** 2, low=4 * (frame // 96), high=12 * (frame // 96)), (offset[0] // 2 + 8, offset[1] // 2 + 8)) for k, n in enumerate(names)}
    manifest = write_manifest(root, products, 64)
    write_sidecar(root, per_product)
    split = {"schema_version": SPLIT_VERSION, "train": [p.product_id for p in products], "validation": [], "test": []}
    return manifest, split


@pytest.mark.parametrize("frame", [96, 384])
def test_html_stays_under_the_size_budget_with_ten_layers_per_card(tmp_path, frame):
    manifest, split = synthetic(tmp_path, frame=frame)
    result = all_review(screen(manifest, split, factor=FACTOR, margin=0.0))
    assert len(result["pairs"]) == 66
    page = render_review(result, top_pairs=60, top_layers=4)
    cards = cards_of(page)
    size = len(page.encode("utf-8"))
    print(f"review.html: {len(cards)} cards, {size} bytes")
    assert len(cards) == 60 and size <= BYTE_BUDGET
    assert all(card.count('class="group"') == 10 and "나머지 6층" in card for card in cards)


@pytest.mark.skipif(not os.environ.get("SCREEN_REPORT_EXAMPLE"), reason="set SCREEN_REPORT_EXAMPLE=<path> to write an example page")
def test_write_example_page(tmp_path):
    manifest, split = synthetic(tmp_path, count=6, layers=6)
    result = screen(manifest, split, factor=FACTOR, margin=0.0, review_band=1.0)
    for p in result["pairs"]:
        p["review"] = p["review"] or ["near_boundary"]
    with open(os.environ["SCREEN_REPORT_EXAMPLE"], "w", encoding="utf-8") as stream:
        stream.write(render_review(result, top_pairs=4, top_layers=3))
