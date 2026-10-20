import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from conftest import FAMILIES, make_product
from psdl1.screen_relations import auc, compare, main, product_signature, screen, tolerant_distance
from psdl1.split import SPLIT_VERSION
from psdl1.train import load_relations
from test_pretrain import write_manifest


def variant(product, name, lineage, size):
    return replace(make_product(product, size), product_id=name, lineage_id=lineage)


@pytest.fixture
def case(tmp_path):
    products = [variant(f, f"{f.upper()}_{v}", f.upper(), s) for f in FAMILIES for s, v in ((40, "R00"), (64, "T00"))]
    manifest = write_manifest(tmp_path, products, 64)
    split = {"schema_version": SPLIT_VERSION, "train": [p.product_id for p in products if not p.product_id.startswith("RING")], "validation": [],
             "test": ["RING_R00", "RING_T00"]}
    return manifest, split, tmp_path


def test_tolerant_distance_forgives_small_shifts_but_not_different_shapes():
    base = np.zeros((32, 32), dtype=bool)
    base[10:12, 4:28] = True
    shifted = np.roll(base, 1, axis=0)
    other = np.zeros_like(base)
    other[4:28, 10:12] = True

    def sig(mask):
        from psdl1.screen_relations import F, torch
        dilated = F.max_pool2d(torch.from_numpy(mask.astype(np.float32))[None, None], 5, 1, 2)[0, 0].numpy() > 0
        return {"mask": mask, "dilated": dilated}
    assert tolerant_distance(sig(base), sig(base)) == 0
    assert tolerant_distance(sig(base), sig(shifted)) < 0.05
    assert tolerant_distance(sig(base), sig(other)) > 0.5


def test_electrode_used_on_one_side_only_counts_as_fully_different(case):
    manifest, _, _ = case
    from psdl1.data import load_manifest
    p = next(iter(load_manifest(manifest)))
    emptied = replace(p, elements=(replace(p.elements[0], mask=np.zeros_like(p.elements[0].mask)), *p.elements[1:]))
    a, b = product_signature(p, 64, 2), product_signature(emptied, 64, 2)
    result = compare(a, b)
    key = f"{p.elements[0].role}/{p.elements[0].layer}"
    assert (key, "only_a", 1.0) in result["bindings"] and result["usage_jaccard"] < 1 and result["similarity"] < 1
    assert compare(a, a)["similarity"] == 1.0 and compare(a, a)["usage_jaccard"] == 1.0


def test_same_lineage_is_similar_and_clearly_other_lineages_are_different_or_unknown(case):
    manifest, split, _ = case
    result = screen(manifest, split, grid=64, margin=0.0)
    by = {(p["product_a"], p["product_b"]): p for p in result["pairs"]}
    assert by[("BUS_R00", "BUS_T00")]["tier"] == "similar" and by[("COMB_R00", "COMB_T00")]["tier"] == "similar"
    same = [p["similarity"] for p in result["pairs"] if p["same_lineage"]]
    assert result["floor"]["global"] == min(same)
    for p in result["pairs"]:
        if not p["same_lineage"]:
            assert p["tier"] == ("different" if p["similarity"] < result["floor"]["global"] else "unknown")
    assert all(p["partition"] == "train" for p in result["pairs"]) and len(result["pairs"]) == 6  # four train products
    assert 0 <= result["summary"]["separation_auc_same_vs_other_lineage"]["similarity"] <= 1


def test_groups_cap_cross_group_pairs_and_must_cover_every_screened_product(case, tmp_path):
    manifest, split, _ = case
    groups = tmp_path / "groups.json"
    groups.write_text(json.dumps({"BUS_R00": "mobile", "BUS_T00": "mobile", "COMB_R00": "it", "COMB_T00": "it", "RING_R00": "auto", "RING_T00": "auto"}))
    result = screen(manifest, split, groups=groups, grid=64, margin=0.0, floor=0.99, cross_group_fraction=0.0)
    cross = [p for p in result["pairs"] if p["group_a"] != p["group_b"]]
    assert cross and all(p["tier"] == "unknown" and p["reason"] == "cross_group_capped" for p in cross)  # no within-group different pair exists here
    groups.write_text(json.dumps({"BUS_R00": "mobile"}))
    with pytest.raises(ValueError, match="missing for screened products"):
        screen(manifest, split, groups=groups, grid=64)
    groups.write_text(json.dumps({**{p: "x" for p in split["train"] + split["test"]}, "NOPE": "x"}))
    with pytest.raises(ValueError, match="unknown IDs"):
        screen(manifest, split, groups=groups, grid=64)


def test_lineage_leakage_and_missing_calibration_rejected(case):
    manifest, split, _ = case
    with pytest.raises(ValueError, match="lineage leakage"):
        screen(manifest, dict(split, train=["BUS_R00", "COMB_R00", "COMB_T00"], test=["BUS_T00", "RING_R00", "RING_T00"]), grid=64)
    solo = dict(split, train=["BUS_R00", "COMB_R00"], validation=["BUS_T00", "COMB_T00"])  # no same-lineage pair inside the partition
    with pytest.raises(ValueError, match="lineage leakage"):
        screen(manifest, solo, grid=64)


def test_cli_outputs_relations_scores_review_and_summary(case, capsys):
    manifest, split, root = case
    split_path = root / "split.json"
    split_path.write_text(json.dumps(split))
    out = root / "relations.csv"
    main(["--manifest", str(manifest), "--split", str(split_path), "--out", str(out), "--grid", "64", "--margin", "0.0"])
    summary = json.loads(capsys.readouterr().out)
    similar, different = load_relations(out)  # exactly what psdl1.train reads
    assert frozenset(("BUS_R00", "BUS_T00")) in similar and len(different) == summary["pairs_by_tier"]["different"]
    header = next(csv.reader(open(str(out) + ".scores.csv", encoding="utf-8")))
    assert header[:3] == ["product_a", "product_b", "partition"] and "similarity" in header
    text = open(str(out) + ".review.html", encoding="utf-8").read()
    assert "data:image/png;base64" in text and "http" not in text.replace("http-equiv", "")
    assert json.loads(open(str(out) + ".summary.json", encoding="utf-8").read())["floor"]["global"] == summary["floor"]["global"]
    with pytest.raises(FileExistsError):
        main(["--manifest", str(manifest), "--split", str(split_path), "--out", str(out), "--grid", "64"])


def test_auc_basics():
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0 and auc([0.1], [0.9]) == 0.0 and auc([0.5], [0.5]) == 0.5 and auc([], [1]) is None
