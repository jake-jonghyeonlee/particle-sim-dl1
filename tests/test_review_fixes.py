"""R3 finding counterexamples; driver tests use a direct optimizer fixture, no CNN training."""
from dataclasses import replace
import json
import random
import tomllib

import numpy as np
import pytest
import torch
from torch import nn

from conftest import make_product
from psdl1.checkpoint import json_hash, read_checkpoint, save_checkpoint
from psdl1.data import (augment_with_check, build_vocab, grid_check, input_identity, load_manifest)
from psdl1.losses import relation_matrix, validate_matrix
from psdl1.model import ProductEncoder
from psdl1.search import search
from psdl1.train import main, model_config, structured_batch, train

APPROVAL = {"input_method": "user_approved_2026_10_09", "real_grid_linewidth": "pending",
            "endpoint_source_format": "pending", "G6": "pending"}


@pytest.fixture
def driver_fixture(monkeypatch):
    # Exercise the actual trainer/checkpoint/sampler control path with two parameters,
    # direct differentiable loss vectors and AdamW state, never a dataset CNN run.
    class OptimizerFixture(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.2, 0.7]))
    def encode(model, products, grid, roles, layers, generator=None, **kwargs):
        random_term = (torch.rand((), generator=generator) + torch.rand(()) + random.random() + np.random.rand()) * 0.01
        vectors = [model.weight + torch.tensor([i * 0.3, -i * 0.2]) + random_term for i, _ in enumerate(products)]
        return torch.nn.functional.normalize(torch.stack(vectors), dim=1)
    monkeypatch.setattr("psdl1.train.ProductEncoder", OptimizerFixture)
    monkeypatch.setattr("psdl1.train.encode_products", encode)


def cli_files(manifest_case, tmp_path):
    from psdl1.split import SPLIT_VERSION
    doc, write, _ = manifest_case
    other = json.loads(json.dumps(doc["products"][0]))
    other.update(product_id="Q", lineage_id="other-family")
    doc["products"].append(other)
    path = write()
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"schema_version": SPLIT_VERSION, "train": ["P", "Q"], "validation": [], "test": []}))
    relations = tmp_path / "relations.csv"
    relations.write_text("P,Q,different\n")
    return ["--manifest", str(path), "--split", str(split), "--relations", str(relations), "--out", str(tmp_path / "out"), "--epochs", "1"]


def test_b1_cli_without_grid_uses_manifest(manifest_case, tmp_path, monkeypatch):
    args = cli_files(manifest_case, tmp_path)
    received = {}
    def capture(*args, **kwargs):
        received["grid"] = args[3]
        received["loaded_grids"] = [p.grid_check["grid"] for p in args[0]]
        return [0.1]
    monkeypatch.setattr("psdl1.train.train", capture)
    main(args)
    assert received == {"grid": 4, "loaded_grids": [4, 4]}


def test_b1_conflicting_cli_grid_is_rejected_before_train(manifest_case, tmp_path, monkeypatch):
    args = cli_files(manifest_case, tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("conflicting grid reached training")
    monkeypatch.setattr("psdl1.train.train", forbidden)
    with pytest.raises(ValueError, match="conflicts.*candidate_grid.*allow-grid-override"):
        main([*args, "--grid", "2"])


def test_b1_override_recorded_in_checkpoint(manifest_case, tmp_path, driver_fixture):
    args = cli_files(manifest_case, tmp_path)
    main([*args, "--grid", "2", "--allow-grid-override"])
    saved = read_checkpoint(tmp_path / "out" / "dl1.pt")
    assert saved["config"]["grid"] == 2
    assert saved["metadata"]["grid_selection"] == {"manifest_candidate_grid": 4, "selected_grid": 2,
        "source": "explicit_override", "override": True, "override_requested": True}
    assert saved["metadata"]["approval"] == APPROVAL
    assert all(r["approval"] == APPROVAL for r in saved["metadata"]["grid_checks"])


def test_b1_no_grid_stops_api_and_cli(products, manifest_case, tmp_path):
    with pytest.raises(ValueError, match="No grid supplied"):
        train(products, tmp_path)
    doc, write, _ = manifest_case
    del doc["candidate_grid"]
    with pytest.raises(ValueError, match="candidate_grid"):
        load_manifest(write())


def test_n1_same_lineage_variants_never_experimental_negatives():
    ids = ["comb-40", "comb-64", "comb-96", "bus-40"]
    rel = relation_matrix(ids, unlabelled_negatives=True, lineage_ids=["comb", "comb", "comb", "bus"])
    assert rel.tolist() == [[1, -1, -1, 0], [-1, 1, -1, 0], [-1, -1, 1, 0], [0, 0, 0, 1]]
    with pytest.raises(ValueError, match="same-lineage negative"):
        relation_matrix(ids, different={frozenset((ids[0], ids[1]))}, unlabelled_negatives=True, lineage_ids=["comb", "comb", "comb", "bus"])


def test_n2_real_encoder_results_invariant_to_order_and_batch_size(products):
    torch.manual_seed(31)
    two = replace(products[0], elements=products[0].elements[:2], product_id="query")
    roles, layers = build_vocab(products)
    cfg = model_config(16, roles, layers)
    model = ProductEncoder(len(roles), len(layers)).eval()
    a, z = replace(two, product_id="A"), replace(two, product_id="Z")
    m = replace(products[2], product_id="M")
    expected = search(model, cfg, two, [z, m, a], topk=3, batch_size=1)["rankings"]
    for order in ([z, m, a], [a, m, z], [m, z, a]):
        for batch_size in (1, 2, 3, 8):
            assert search(model, cfg, two, order, topk=3, batch_size=batch_size)["rankings"] == expected
    tied = [r["product_id"] for r in expected if r["product_id"] in {"A", "Z"}]
    assert tied == ["A", "Z"]
    assert next(r["score"] for r in expected if r["product_id"] == "A") == next(r["score"] for r in expected if r["product_id"] == "Z")


def test_n5_rectangular_endpoint_matches_padded_image_cell():
    from psdl1.schema import Element, Product, Frame, Connection, Endpoint
    mask = np.zeros((4, 8), dtype=np.float32); mask[3, 7] = 1
    frame = Frame("rect", (4, 8), 0.05, (1.0, -2.0))
    endpoints = (Endpoint("A", (1.375, -1.825)), Endpoint("B", (1.375, -1.825)))
    p = Product("rect", (0.4, 0.2), (Element("A", "L1", mask, element_id="A"), Element("B", "L2", mask, element_id="B")),
                frame=frame, connections=(Connection("C", "CNT", endpoints, mask),), connectivity_observation="complete")
    _, extra, prepared = structured_batch([p], 8, *build_vocab([p]))
    # Width 8, height 4, top pad 2: centre (7.5,3.5) -> (7.5,5.5)/8.
    assert extra["connection_features"][0, 0, 2:6].tolist() == pytest.approx([0.9375, 0.6875, 0.9375, 0.6875], abs=1e-7)
    assert prepared[0].connection_masks[0, 5, 7] == 1
    assert prepared[0].connection_masks[0].sum() == 1


@pytest.mark.parametrize("kind", ["thin", "bridge"])
def test_n7_bad_grid_blocks_before_model_construction(kind, bridge_mask, tmp_path, monkeypatch):
    from psdl1.schema import Element, Product
    mask = np.zeros((8, 8), dtype=np.float32); mask[:, 3] = 1
    if kind == "bridge":
        mask = bridge_mask
    h, w = mask.shape
    p = Product("P", (w * 0.05, h * 0.05), (Element("A", "L1", mask),), lineage_id="P")
    q = replace(p, product_id="Q", lineage_id="Q")
    def forbidden(*args, **kwargs):
        pytest.fail("failing grid constructed model")
    monkeypatch.setattr("psdl1.train.ProductEncoder", forbidden)
    with pytest.raises(ValueError, match="grid check failed.*allow-grid-check-failure"):
        train([p, q], tmp_path, grid=6 if kind == "bridge" else 8, relations=(set(), {frozenset(("P", "Q"))}))


def test_n7_grid_gate_bypass_recorded(products, tmp_path, driver_fixture):
    sample = products[:2]
    train(sample, tmp_path, epochs=1, grid=16, relations=(set(), {frozenset(p.product_id for p in sample)}), allow_grid_check_failure=True)
    saved = read_checkpoint(tmp_path / "dl1.pt")
    assert saved["metadata"]["grid_gate"]["policy"] == "strict_default"
    assert saved["metadata"]["grid_gate"]["bypass_requested"] is True
    assert saved["metadata"]["grid_gate"]["status"] == "bypassed"
    assert saved["metadata"]["grid_gate"]["failures"]


def test_n8_full_resume_driver_and_n9_preparation_cached(products, tmp_path, driver_fixture, monkeypatch):
    import psdl1.train as trainer
    sample = products[:2]
    relation = (set(), {frozenset(p.product_id for p in sample)})
    original = trainer.prepare_product
    calls = []
    def counted(p, *args, **kwargs):
        calls.append(p.product_id)
        return original(p, *args, **kwargs)
    monkeypatch.setattr(trainer, "prepare_product", counted)
    kwargs = {"grid": 16, "relations": relation, "augmentation": "intensity", "allow_grid_check_failure": True}
    continuous = train(sample, tmp_path / "continuous", epochs=3, **kwargs)
    assert calls == [p.product_id for p in sample]  # once per product, not per view/step
    train(sample, tmp_path / "resumed", epochs=1, **kwargs)
    torch.rand(11); np.random.rand(11); random.random()
    resumed = train(sample, tmp_path / "resumed", epochs=3, resume=tmp_path / "resumed" / "dl1.pt", **kwargs)
    left, right = read_checkpoint(tmp_path / "continuous" / "dl1.pt"), read_checkpoint(tmp_path / "resumed" / "dl1.pt")
    assert continuous == resumed
    assert left["weights_hash"] == right["weights_hash"]
    assert left["metadata"]["epochs"] == right["metadata"]["epochs"]
    assert len(calls) == 6


def test_n8_families_have_distinct_cnt_geometry():
    products = [make_product(f, 80) for f in ("bus", "comb", "ring")]
    assert not np.array_equal(products[0].cnt, products[1].cnt)
    assert not np.array_equal(products[1].cnt, products[2].cnt)


def test_n9_missing_checkpoint_config_raises_valueerror(products, tmp_path):
    from psdl1.checkpoint import save_checkpoint
    roles, layers = build_vocab(products)
    model = ProductEncoder(len(roles), len(layers))
    config = model_config(16, roles, layers)
    saved = save_checkpoint(tmp_path / "model.pt", model, config)
    del saved["config"]
    torch.save(saved, tmp_path / "model.pt")
    with pytest.raises(ValueError, match="incomplete checkpoint"):
        read_checkpoint(tmp_path / "model.pt")


def test_n9_diagonal_must_be_positive():
    with pytest.raises(ValueError, match="diagonal"):
        validate_matrix(torch.tensor([[0, 1], [1, 0]]), 2)


def test_n9_integer_float_serialization_same_input_identity(manifest_case):
    doc, write, _ = manifest_case
    p = doc["products"][0]
    p["unit_size_um"] = [2, 2]
    p["frame"]["pixel_pitch_um"] = 0.5
    for ep in p["connections"][0]["endpoints"]:
        ep["xy_um"] = [1.25, -1.75]
    first, = load_manifest(write())
    p["unit_size_um"] = [2.0, 2.0]
    p["frame"]["origin_xy_um"] = [1, -2]
    second, = load_manifest(write())
    assert input_identity(first) == input_identity(second)


def test_n9_cnt_crop_is_reported_even_if_self_is_unchanged():
    # self channel is empty, but its CNT image is lost by translation.
    images = torch.zeros(1, 3, 8, 8); images[0, 1, 1, 1] = 1
    _, report = augment_with_check(images, torch.Generator(), theta=[[1, 0, 2], [0, 1, 0]])
    assert report["elements"][0]["channels"][0]["source_components"] == 0
    assert report["elements"][0]["channels"][1]["source_components"] == 1
    assert report["elements"][0]["channels"][1]["target_components"] == 0
    assert report["elements"][0]["bridge_or_component_loss"] is True
    assert not report["safe_for_positive_label"]


def test_n6_every_report_uses_current_approval(manifest_case):
    _, write, _ = manifest_case
    p, = load_manifest(write())
    cfg = model_config(4, *build_vocab([p]))
    assert p.grid_check["approval"] == APPROVAL
    assert search(object(), cfg, p, [])["approval"] == APPROVAL
    _, report = augment_with_check(torch.zeros(1, 3, 4, 4), torch.Generator())
    assert report["approval"] == APPROVAL


def test_n9_safetensors_is_declared_as_optional_dependency():
    from pathlib import Path
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert project["project"]["optional-dependencies"]["local-weights"] == ["safetensors>=0.8"]
