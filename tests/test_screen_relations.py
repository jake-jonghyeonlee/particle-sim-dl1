import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from conftest import FAMILIES, make_product
from psdl1.screen_relations import auc, compare, main, product_signature, render_review, screen
from psdl1.split import SPLIT_VERSION
from psdl1.train import load_relations
from test_pretrain import write_manifest

FACTOR, TOL = 2, 1


def embed(product, name, lineage, size, offset):
    """제품을 size×size 프레임의 offset 위치에 놓는다(같은 물리 크기의 설계가 프레임 안에서 다른 위치에 있는 상황)."""
    oy, ox = offset

    def place(mask):
        out = np.zeros((size, size), dtype=mask.dtype)
        out[oy:oy + mask.shape[0], ox:ox + mask.shape[1]] = mask
        return out
    return replace(product, product_id=name, lineage_id=lineage, unit_size_um=(size * 0.05, size * 0.05),
                   elements=tuple(replace(e, mask=place(e.mask)) for e in product.elements), cnt=place(product.cnt))


def sig(product):
    return product_signature(product, FACTOR, TOL)


OFFSETS = {"bus": ((14, 10), (30, 22)), "comb": ((10, 30), (26, 6)), "ring": ((20, 20), (6, 34))}


@pytest.fixture
def case(tmp_path):
    products = [embed(make_product(f, 48), f"{f.upper()}_{v}", f.upper(), 96, OFFSETS[f][i]) for f in FAMILIES for i, v in enumerate(("R00", "T00"))]
    manifest = write_manifest(tmp_path, products, 64)
    split = {"schema_version": SPLIT_VERSION, "train": [p.product_id for p in products if not p.product_id.startswith("RING")], "validation": [],
             "test": ["RING_R00", "RING_T00"]}
    return manifest, split, tmp_path, products


def test_shifted_copy_is_aligned_and_the_same_shift_applies_to_every_layer(case):
    *_, products = case
    a, b = products[0], products[1]  # BUS_R00 at (14,10), BUS_T00 at (30,22)
    result = compare(sig(a), sig(b), tolerance=TOL)
    assert result["shift"] == ((14 - 30) // FACTOR, (10 - 22) // FACTOR) and result["anchor"] == "CNT"
    assert result["similarity"] > 0.97 and result["usage_jaccard"] == 1.0 and result["flags"] == []
    assert all(distance < 0.05 for _, status, distance in result["bindings"] if status == "both")
    assert result["shift"][0] * FACTOR * 0.05 == pytest.approx(-0.8)  # µm: 16 px at 0.05 µm/px


def test_different_frame_sizes_compare_in_the_same_physical_scale(case):
    *_, products = case
    base = make_product("bus", 48)
    a = embed(base, "A", "L", 96, (14, 10))
    b = embed(base, "B", "L", 128, (40, 50))
    result = compare(sig(a), sig(b), tolerance=TOL)
    assert result["similarity"] > 0.97 and result["roi_overlap"] == pytest.approx(44 / 48) and "low_overlap" not in result["flags"]  # B's frame ends 4 raster px before A's right edge
    assert result["shift"] == ((14 - 40) // FACTOR, (10 - 50) // FACTOR)


def test_identical_products_and_a_missing_electrode(case):
    *_, products = case
    a = products[0]
    same = compare(sig(a), sig(a), tolerance=TOL)
    assert same["similarity"] == 1.0 and same["shift"] == (0, 0) and same["flags"] == []
    emptied = replace(a, elements=(replace(a.elements[0], mask=np.zeros_like(a.elements[0].mask)), *a.elements[1:]))
    result = compare(sig(a), sig(emptied), tolerance=TOL)
    key = f"{a.elements[0].role}/{a.elements[0].layer}"
    assert (key, "only_a", 1.0) in result["bindings"] and result["usage_jaccard"] < 1 and result["similarity"] < 1
    assert result["shift"] == (0, 0)  # the CNT anchor still aligns the pair


def test_cnt_covering_the_whole_frame_falls_back_to_the_electrode_union(case):
    *_, products = case
    a, b = products[0], products[1]
    whole = lambda p: replace(p, cnt=np.ones_like(p.cnt))
    result = compare(sig(whole(a)), sig(whole(b)), tolerance=TOL)
    assert result["anchor"] == "ELECTRODES" and "anchor_fallback_electrodes" in result["flags"]
    assert result["shift"] == ((14 - 30) // FACTOR, (10 - 22) // FACTOR)
    bare = lambda p: replace(p, cnt=np.zeros_like(p.cnt))
    assert compare(sig(bare(a)), sig(bare(b)), tolerance=TOL)["anchor"] == "ELECTRODES"


def test_periodic_pattern_is_flagged_as_ambiguous_and_anchor_mismatch_as_unreliable():
    def striped(name, period=16, width=4, size=96):
        m = np.zeros((size, size), dtype=np.float32)
        for x in range(2, size, period):
            m[:, x:x + width] = 1
        base = make_product("bus", size)
        return replace(base, product_id=name, lineage_id="L", unit_size_um=(size * 0.05,) * 2,
                       elements=(replace(base.elements[0], mask=m),), cnt=m.copy())
    result = compare(sig(striped("A")), sig(striped("B")), tolerance=TOL)
    assert "ambiguous_alignment" in result["flags"]
    base = make_product("bus", 96)
    square, bar = np.zeros((96, 96), np.float32), np.zeros((96, 96), np.float32)
    square[40:50, 40:50] = 1
    bar[10:12, 5:90] = 1
    a = replace(base, elements=(replace(base.elements[0], mask=np.ones((96, 96), np.float32)),), cnt=square)
    b = replace(a, cnt=bar)
    assert "alignment_unreliable" in compare(sig(a), sig(b), tolerance=TOL)["flags"]


def test_same_lineage_is_similar_and_other_lineages_are_different_or_unknown(case):
    manifest, split, *_ = case
    result = screen(manifest, split, factor=FACTOR, margin=0.0)
    by = {(p["product_a"], p["product_b"]): p for p in result["pairs"]}
    assert by[("BUS_R00", "BUS_T00")]["tier"] == "similar" and by[("COMB_R00", "COMB_T00")]["tier"] == "similar"
    assert by[("BUS_R00", "BUS_T00")]["similarity"] > 0.97 and by[("BUS_R00", "BUS_T00")]["shift_y_um"] != 0
    assert result["floor"]["global"] == min(p["similarity"] for p in result["pairs"] if p["same_lineage"])
    for p in result["pairs"]:
        if not p["same_lineage"]:
            assert p["tier"] == ("different" if p["similarity"] < result["floor"]["global"] else "unknown")
    assert len(result["pairs"]) == 6 and result["summary"]["alignment"]["anchors"] == {"CNT": 6}
    assert 0 <= result["summary"]["separation_auc_same_vs_other_lineage"]["similarity"] <= 1


def test_groups_cap_cross_group_pairs_and_must_cover_every_screened_product(case, tmp_path):
    manifest, split, *_ = case
    groups = tmp_path / "groups.json"
    groups.write_text(json.dumps({"BUS_R00": "mobile", "BUS_T00": "mobile", "COMB_R00": "it", "COMB_T00": "it", "RING_R00": "auto", "RING_T00": "auto"}))
    result = screen(manifest, split, groups=groups, factor=FACTOR, margin=0.0, floor=0.99, cross_group_fraction=0.0)
    cross = [p for p in result["pairs"] if p["group_a"] != p["group_b"]]
    assert cross and all(p["tier"] == "unknown" and p["reason"] == "cross_group_capped" for p in cross)
    groups.write_text(json.dumps({"BUS_R00": "mobile"}))
    with pytest.raises(ValueError, match="missing for screened products"):
        screen(manifest, split, groups=groups, factor=FACTOR)
    groups.write_text(json.dumps({**{p: "x" for p in split["train"] + split["test"]}, "NOPE": "x"}))
    with pytest.raises(ValueError, match="unknown IDs"):
        screen(manifest, split, groups=groups, factor=FACTOR)


def test_invalid_options_lineage_leakage_and_missing_calibration_rejected(case):
    manifest, split, *_ = case
    with pytest.raises(ValueError, match="lineage leakage"):
        screen(manifest, dict(split, train=["BUS_R00", "COMB_R00", "COMB_T00"], test=["BUS_T00", "RING_R00", "RING_T00"]), factor=FACTOR)
    for bad in ({"factor": 0}, {"anchor": "pcnt"}, {"max_shift": 0}, {"candidates": 0}):
        with pytest.raises(ValueError):
            screen(manifest, split, **{"factor": FACTOR, **bad})


def test_cli_outputs_relations_scores_review_and_summary(case, capsys):
    manifest, split, root, _ = case
    split_path = root / "split.json"
    split_path.write_text(json.dumps(split))
    out = root / "relations.csv"
    main(["--manifest", str(manifest), "--split", str(split_path), "--out", str(out), "--factor", str(FACTOR), "--margin", "0.0"])
    summary = json.loads(capsys.readouterr().out)
    similar, different = load_relations(out)  # exactly what psdl1.train reads
    assert frozenset(("BUS_R00", "BUS_T00")) in similar and len(different) == summary["pairs_by_tier"]["different"]
    header = next(csv.reader(open(str(out) + ".scores.csv", encoding="utf-8")))
    assert {"similarity", "anchor", "shift_y_um", "shift_x_um", "alignment_score", "roi_overlap", "alignment_flag"} <= set(header)
    text = open(str(out) + ".review.html", encoding="utf-8").read()
    assert "data:image/png;base64" in text and "정렬 기준 CNT" in text and "http" not in text.replace("http-equiv", "")
    assert json.loads(open(str(out) + ".summary.json", encoding="utf-8").read())["alignment"]["anchors"] == {"CNT": 6}
    with pytest.raises(FileExistsError):
        main(["--manifest", str(manifest), "--split", str(split_path), "--out", str(out), "--factor", str(FACTOR)])


def test_review_lists_every_pair_that_needs_a_look_even_when_cards_are_capped(case):
    manifest, split, *_ = case
    narrow = screen(manifest, split, factor=FACTOR, margin=0.0, review_band=0.0)
    wide = screen(manifest, split, factor=FACTOR, margin=0.0, review_band=1.0)
    nothing_extra = [p for p in narrow["pairs"] if "different_near_boundary" in p["review"]]
    near = [p for p in wide["pairs"] if "different_near_boundary" in p["review"]]
    assert len(near) == sum(p["tier"] == "different" for p in wide["pairs"]) > len(nothing_extra)  # a wide band lists every auto-different pair
    assert wide["summary"]["review"]["pairs_to_review"] == sum(bool(p["review"]) for p in wide["pairs"])
    assert wide["summary"]["review"]["by_reason"]["weakest_same_lineage"] >= 1
    page = render_review(wide, top_pairs=1)
    listed = sum(f"{p['product_a']} ↔ {p['product_b']}" in page for p in wide["pairs"] if p["review"])
    assert listed == wide["summary"]["review"]["pairs_to_review"]  # the table has them all
    assert page.count('<section class="card">') == 1 and "나머지" in page


def test_alignment_problems_put_a_pair_on_the_review_list(case, monkeypatch):
    import psdl1.screen_relations as module
    manifest, split, *_ = case
    original = module.compare
    monkeypatch.setattr(module, "compare", lambda *a, **k: {**original(*a, **k), "flags": ["ambiguous_alignment", "low_overlap", "anchor_fallback_electrodes"]})
    result = screen(manifest, split, factor=FACTOR, margin=0.0, review_band=0.0)
    for p in result["pairs"]:
        assert {"ambiguous_alignment", "low_overlap"} <= set(p["review"]) and "anchor_fallback_electrodes" not in p["review"]  # routine fallback is not a review reason
    assert result["summary"]["review"]["pairs_to_review"] == len(result["pairs"])
    assert result["summary"]["review"]["by_reason"]["low_overlap"] == len(result["pairs"])


def incremental_inputs(root, split, *, rows=("BUS_R00,BUS_T00,similar\n", "BUS_R00,RING_R00,different\n"), floor=0.9):
    """train: COMB_T00 is new (3 old); test: RING_T00 is new (1 old)."""
    base_split = {**split, "train": ["BUS_R00", "BUS_T00", "COMB_R00"], "validation": [], "test": ["RING_R00"]}
    base_split_path, base_relations, floor_json = root / "base-split.json", root / "base-relations.csv", root / "floor.json"
    base_split_path.write_text(json.dumps(base_split))
    base_relations.write_text("".join(rows))
    floor_json.write_text(json.dumps({"floor": {"global": floor, "per_group": {"x": 0.8}}}))
    return base_split, base_split_path, base_relations, floor_json


def test_incremental_screens_only_pairs_with_a_new_product(case):
    manifest, split, root, _ = case
    base_split, _, base_relations, floor_json = incremental_inputs(root, split)
    from psdl1.screen_relations import load_floor_spec
    result = screen(manifest, split, partitions=("train", "test"), factor=FACTOR, base_split=base_split, base_relations=base_relations, floor_spec=load_floor_spec(floor_json))
    keys = {(p["product_a"], p["product_b"]) for p in result["pairs"]}
    assert keys == {("BUS_R00", "COMB_T00"), ("BUS_T00", "COMB_T00"), ("COMB_R00", "COMB_T00"), ("RING_R00", "RING_T00")}  # no old x old pair
    summary = result["summary"]
    assert summary["mode"] == "incremental" and summary["new_products"] == {"train": ["COMB_T00"], "test": ["RING_T00"]}
    assert summary["delta_pairs"] == summary["expected_delta_pairs"] == {"train": 3, "test": 1}
    assert summary["base"]["pairs"] == 1 and summary["base"]["relations"] == str(base_relations)  # the BUS_R00/RING_R00 row crosses partitions and is ignored
    assert len(summary["base"]["sha256"]) == 64


def test_incremental_uses_the_fixed_floor_from_summary_and_outputs_only_delta_rows(case, capsys):
    manifest, split, root, _ = case
    _, base_split_path, base_relations, floor_json = incremental_inputs(root, split)
    split_path, out = root / "split.json", root / "relations.csv"
    split_path.write_text(json.dumps(split))
    args = ["--manifest", str(manifest), "--split", str(split_path), "--out", str(out), "--factor", str(FACTOR), "--partitions", "train", "test",
            "--base-split", str(base_split_path), "--base-relations", str(base_relations), "--floor-json", str(floor_json)]
    main(args)
    printed = json.loads(capsys.readouterr().out)
    summary = json.loads(open(str(out) + ".summary.json", encoding="utf-8").read())
    assert summary["floor"]["global"] == 0.9 and summary["floor"]["per_group"] == {} and summary["floor"]["source"] == "base_summary" == printed["floor"]["source"]
    delta_rows = [tuple(r) for r in csv.reader(open(out, encoding="utf-8"))]
    assert all("COMB_T00" in r[:2] or "RING_T00" in r[:2] for r in delta_rows) and ("COMB_R00", "COMB_T00", "similar") in delta_rows
    assert not (root / "relations.csv.delta.csv").exists()
    assert len(list(csv.DictReader(open(str(out) + ".scores.csv", encoding="utf-8")))) == 4
    assert open(base_relations, encoding="utf-8").read() == "BUS_R00,BUS_T00,similar\nBUS_R00,RING_R00,different\n"  # the base file is never touched
    with pytest.raises(SystemExit):
        main([*args, "--floor", "0.5"])
    with pytest.raises(SystemExit):
        main(["--manifest", str(manifest), "--split", str(split_path), "--out", str(root / "o2.csv"), "--base-split", str(base_split_path)])
    with pytest.raises(ValueError, match="floor and floor_spec"):
        screen(manifest, split, factor=FACTOR, floor=0.5, floor_spec={"global": 0.9, "groups": {}})


@pytest.mark.parametrize("rows,mismatch,message", [
    (("BUS_R00,NOPE,similar\n",), None, "not in the manifest"),
    (("BUS_R00,BUS_T00,similar\n", "BUS_T00,BUS_R00,different\n"), None, "conflicting"),
    (("COMB_R00,COMB_T00,different\n",), None, "contradicts"),
    (("COMB_T00,COMB_R00,similar\n",), None, "duplicates"),
    ((), {"train": ["BUS_R00", "BUS_T00", "COMB_R00", "RING_R00"]}, "left the partition"),
    ((), {"train": ["BUS_R00", "GHOST"]}, "not in the manifest"),
])
def test_incremental_rejects_conflicts(case, rows, mismatch, message):
    manifest, split, root, _ = case
    base_split, _, base_relations, _ = incremental_inputs(root, split, rows=rows)
    with pytest.raises(ValueError, match=message):
        screen(manifest, split, partitions=("train", "test"), factor=FACTOR, floor=0.9, base_relations=base_relations, base_split={**base_split, **(mismatch or {})})


def test_incremental_ignores_rows_of_other_partitions_and_non_incremental_output_is_unchanged(case, tmp_path):
    manifest, split, root, _ = case
    base_split, _, with_noise, _ = incremental_inputs(root, split, rows=("RING_R00,RING_T00,similar\n", "BUS_R00,RING_R00,different\n", "BUS_R00,BUS_T00,similar\n"))
    only_train = screen(manifest, split, partitions=("train",), factor=FACTOR, floor=0.9, base_split=base_split, base_relations=with_noise)
    assert only_train["summary"]["base"]["pairs"] == 1 and len(only_train["pairs"]) == 3  # the test-partition row is not used when only train is screened
    split_path = root / "split.json"
    split_path.write_text(json.dumps(split))
    runs = []
    for name in ("a", "b"):
        main(["--manifest", str(manifest), "--split", str(split_path), "--out", str(tmp_path / f"{name}.csv"), "--factor", str(FACTOR), "--margin", "0.0"])
        runs.append([(tmp_path / f"{name}.csv{suffix}").read_bytes() for suffix in ("", ".scores.csv", ".summary.json")])
    assert runs[0] == runs[1] and "mode" not in json.loads(runs[0][2])


def test_auc_basics():
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0 and auc([0.1], [0.9]) == 0.0 and auc([0.5], [0.5]) == 0.5 and auc([], [1]) is None
