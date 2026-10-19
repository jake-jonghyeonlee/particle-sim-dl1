import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from conftest import FAMILIES, make_product
from psdl1.embed_plot import build_report, cosine_distances, mds_2d, pca_2d, separation_stats
from psdl1.relations_draft import draft_relations, main as draft_main, summarize
from psdl1.split import SPLIT_VERSION
from psdl1.train import load_relations, train
from test_pretrain import write_manifest


@pytest.fixture
def case(tmp_path):
    products = [replace(make_product(f, s), product_id=f"{f.upper()}_{v}", lineage_id=f.upper())
                for f in FAMILIES for s, v in ((40, "R00"), (64, "T00"))]
    manifest = write_manifest(tmp_path, products, 64)
    split = {"schema_version": SPLIT_VERSION, "train": ["BUS_R00", "BUS_T00", "COMB_R00", "COMB_T00"], "validation": [], "test": ["RING_R00", "RING_T00"]}
    return manifest, split, tmp_path


def test_draft_marks_same_lineage_similar_and_other_lineages_different(case):
    manifest, split, _ = case
    rows = draft_relations(manifest, split, partitions=("train",), grid=16)
    relation = {(r["product_a"], r["product_b"]): r["relation"] for r in rows}
    assert relation[("BUS_R00", "BUS_T00")] == "similar" and relation[("COMB_R00", "COMB_T00")] == "similar"
    assert relation[("BUS_R00", "COMB_R00")] == relation[("BUS_T00", "COMB_T00")] == "different"
    assert len(rows) == 6 and all(r["partition"] == "train" for r in rows)  # pairs stay inside the partition


def test_hold_out_nearest_leaves_the_closest_cross_lineage_pairs_unknown(case):
    manifest, split, _ = case
    rows = draft_relations(manifest, split, partitions=("train",), grid=16, hold_out_nearest=0.5)
    cross = sorted((r for r in rows if r["relation"] != "similar"), key=lambda r: r["image_distance"])
    assert [r["relation"] for r in cross] == ["unknown", "unknown", "different", "different"]
    assert all(cross[0]["image_distance"] <= r["image_distance"] for r in cross)
    assert summarize(rows, split, ("train",))["pairs"] == {"similar": 2, "different": 2, "unknown": 2}


def test_cli_writes_loadable_header_free_csv_and_refuses_overwrite(case, capsys):
    manifest, split, root = case
    split_path = root / "split.json"
    split_path.write_text(json.dumps(split))
    out = root / "relations.csv"
    draft_main(["--manifest", str(manifest), "--split", str(split_path), "--out", str(out), "--grid", "16", "--hold-out-nearest", "0.25"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["pairs"]["similar"] == 2 and summary["products_without_different_pair"]["train"] == []
    similar, different = load_relations(out)  # the exact reader psdl1.train uses
    assert frozenset(("BUS_R00", "BUS_T00")) in similar and different
    header = next(csv.reader(open(str(out) + ".review.csv", encoding="utf-8")))
    assert header[:4] == ["product_a", "product_b", "partition", "relation"]
    with pytest.raises(FileExistsError):
        draft_main(["--manifest", str(manifest), "--split", str(split_path), "--out", str(out), "--grid", "16"])


def test_lineage_leakage_and_id_mismatch_rejected(case):
    manifest, split, _ = case
    with pytest.raises(ValueError, match="lineage leakage"):
        draft_relations(manifest, dict(split, train=["BUS_R00", "COMB_R00", "COMB_T00"], test=["BUS_T00", "RING_R00", "RING_T00"]), grid=16)
    with pytest.raises(ValueError, match="not in manifest"):
        draft_relations(manifest, dict(split, test=["RING_R00", "RING_T"]), grid=16)


def test_projection_and_separation_helpers():
    rng = np.random.default_rng(0)
    z = np.vstack([rng.normal([1, 0, 0, 0], 0.05, (3, 4)), rng.normal([0, 1, 0, 0], 0.05, (3, 4))])
    rows = [{"lineage": "A" if i < 3 else "B", "product_id": f"P{i}"} for i in range(6)]
    d = cosine_distances(z)
    assert np.allclose(np.diag(d), 0, atol=1e-9) and np.allclose(d, d.T)
    coords, explained = pca_2d(z)
    assert coords.shape == (6, 2) and 0 < explained[0] <= 1
    mds = mds_2d(d)
    assert mds.shape == (6, 2) and abs(np.linalg.norm(mds[0] - mds[5]) - d[0, 5]) < 0.5 * d[0, 5] + 0.2
    stats, nearest = separation_stats(rows, d)
    assert stats["mean_distance_same_lineage"] < stats["mean_distance_other_lineage"]
    assert stats["nearest_neighbour_same_lineage_rate"] == 1.0 and nearest[0]["product_id"] in {"P1", "P2"}


def test_embedding_report_from_a_trained_checkpoint(case):
    manifest, split, root = case
    from psdl1.data import load_manifest
    products = load_manifest(manifest)
    similar = {frozenset(("BUS_R00", "BUS_T00")), frozenset(("COMB_R00", "COMB_T00"))}
    different = {frozenset((a, b)) for a in ("BUS_R00", "BUS_T00") for b in ("COMB_R00", "COMB_T00")}
    train(products, root / "run", epochs=2, grid=64, batch_size=4, relations=(similar, different), split_manifest=split)
    summary = build_report(root / "run" / "dl1.pt", manifest, root / "report")
    assert summary["products"] == 6 and summary["epoch"] == 2 and summary["inputs_differing_from_checkpoint"] == []
    text = (root / "report" / "embedding_map.html").read_text(encoding="utf-8")
    assert "BUS_R00" in text and "RING_T00" in text and "http" not in text.replace("http-equiv", "")  # self-contained, offline
    assert text.count("data-tip") >= 6 + 36
    rows = list(csv.reader(open(root / "report" / "distances.csv", encoding="utf-8")))
    assert len(rows) == 7 and rows[0][0] == "product_id"
    with pytest.raises(FileExistsError):
        build_report(root / "run" / "dl1.pt", manifest, root / "report")
