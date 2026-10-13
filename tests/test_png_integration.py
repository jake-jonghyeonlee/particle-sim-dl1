"""S19/S21/S22/S24/S27: actual importer-to-model boundaries, synthetic only."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from png_helpers import write_stack_order, write_wide_import_fixture, write_model
from psdl1.data import build_vocab, load_manifest, grid_check, input_identity
from psdl1.import_png import run_import
from psdl1.png_contract import ImportOptions
from psdl1.model import ProductEncoder
from psdl1.train import structured_batch, encode_products, model_config, resolve_training_grid, train
from psdl1.losses import masked_contrastive_loss, relation_matrix, UNKNOWN
from psdl1.search import search, embed
from psdl1.schema import validate_product
from psdl1.split import SPLIT_VERSION, validate_split


@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("shared-png-import")
    root = tmp_path / "source"
    write_wide_import_fixture(root)
    write_wide_import_fixture(root, model="SYNTHETIC-WIDE-Z")
    mapping = tmp_path / "metadata.json"
    mapping.write_text(json.dumps({"metadata_schema_version": "dl1_png_metadata_v1", "models": {
        "SYNTHETIC-WIDE": {"lineage_id": "SYNTHETIC-FAMILY", "design_version": "SYNTHETIC-V1"},
        "SYNTHETIC-WIDE-Z": {"lineage_id": "SYNTHETIC-OTHER-FAMILY", "design_version": "SYNTHETIC-OTHER-V1"}}}))
    out = tmp_path / "bundle"
    command = [sys.executable, "-m", "psdl1.import_png", "--root", str(root), "--out", str(out),
               "--export", "--metadata", str(mapping), "--candidate-grid", "128", "--stack-order", str(write_stack_order(tmp_path / "order.json", ["U", "N", "C4", "U5", "C6", "L"]))]
    result = subprocess.run(command, capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert result.returncode == 0, result.stderr
    p, other = load_manifest(out / "manifest.json")
    assert other.product_id == "SYNTHETIC-WIDE-Z"
    return p, out


def setup(p):
    roles, layers = build_vocab([p])
    cfg = model_config(128, roles, layers, model_kwargs={"dim": 16, "out_dim": 8, "heads": 4, "blocks": 0})
    torch.manual_seed(173)
    return ProductEncoder(len(roles), len(layers), **cfg["model_kwargs"]), cfg


def test_imported_validator_structured_train_and_search(imported, tmp_path):
    p, out = imported
    validate_product(p)
    report = tmp_path / "validated.json"
    result = subprocess.run([sys.executable, "-m", "psdl1.data", str(out / "manifest.json"), "--report", str(report)],
                            capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert result.returncode == 0, result.stderr
    assert json.loads(report.read_text())["products"][0]["product_id"] == "SYNTHETIC-WIDE"
    assert len(p.connections) == 3 and p.connectivity_observation == "partial"
    # Explicit synthetic metadata, split and a hand-labelled negative for the
    # optimizer boundary. This proves integration, not retrieval quality.
    other = load_manifest(out / "manifest.json")[1]
    q = replace(other, connections=(replace(other.connections[0], kind="VIA"), *other.connections[1:]))
    held = replace(p, product_id="SYNTHETIC-HELD", lineage_id="SYNTHETIC-HELD-FAMILY")
    split = {"schema_version": SPLIT_VERSION, "train": [p.product_id, q.product_id], "validation": [], "test": [held.product_id]}
    partitions = validate_split([p, q, held], split)
    assert [x.product_id for x in partitions["test"]] == ["SYNTHETIC-HELD"]
    relation = relation_matrix([p.product_id, q.product_id], different={frozenset((p.product_id, q.product_id))},
                               lineage_ids=[p.lineage_id, q.lineage_id])
    assert torch.equal(relation, torch.tensor([[1, 0], [0, 1]]))
    unknown = relation_matrix([p.product_id, q.product_id])
    assert unknown[0, 1] == UNKNOWN  # PNG extraction invents no labels
    model, cfg = setup(p)
    batch, extra, prepared = structured_batch([p, q], 128, cfg["roles"], cfg["layers"])
    assert extra["connection_features"].shape == (2, 3, 9)
    assert extra["connection_valid"].tolist() == [[True] * 3] * 2
    assert extra["connectivity_status"].tolist() == [1, 1]
    assert extra["connection_features"][0, :, 6:8].tolist() == [[0., 1.]] * 3
    before = model.head[-1].weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    z = encode_products(model, [p, q], 128, cfg["roles"], cfg["layers"], prepared=prepared)
    loss, usable = masked_contrastive_loss(z, z, relation)
    assert usable == 2 and torch.isfinite(loss)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert not torch.equal(before, model.head[-1].weight)
    model.eval()
    a, b = replace(p, product_id="CANDIDATE-A"), replace(q, product_id="CANDIDATE-B")
    first = search(model, cfg, held, [b, a], batch_size=1)
    second = search(model, cfg, held, [a, b], batch_size=2)
    assert first["status"] == "available" and first["eligible_count"] == 2
    assert first["rankings"] == second["rankings"]
    assert all(row["gate_status"] == "passed" for row in first["grid_gate"]["products"])
    assert cfg["use_unit_size"] is False
    training_out = tmp_path / "synthetic-one-step"
    history = train([p, q, held], training_out, epochs=1, grid=128, batch_size=2, seed=173,
                    relations=(set(), {frozenset((p.product_id, q.product_id))}), split_manifest=split,
                    model_kwargs=cfg["model_kwargs"])
    assert len(history) == 1 and np.isfinite(history).all()
    from psdl1.checkpoint import read_checkpoint
    saved = read_checkpoint(training_out / "dl1.pt")
    assert saved["metadata"]["split"] == split
    assert saved["metadata"]["grid_gate"]["status"] == "passed"
    assert saved["metadata"]["epochs"][0]["steps"] == 1
    cli_output = tmp_path / "search.json"
    searched = subprocess.run([sys.executable, "-m", "psdl1.search", "--ckpt", str(training_out / "dl1.pt"),
        "--query-manifest", str(out / "manifest.json"), "--query", p.product_id,
        "--candidates-manifest", str(out / "manifest.json"), "--out", str(cli_output)],
        capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert searched.returncode == 0, searched.stderr
    cli_result = json.loads(cli_output.read_text())
    assert cli_result["status"] == "available" and cli_result["eligible_count"] == 1
    assert cli_result["rankings"][0]["product_id"] == "SYNTHETIC-WIDE-Z"


@pytest.mark.parametrize("field,name", [("role", "SYNTHETIC-UNSEEN-ELECTRODE"), ("layer", "SYNTHETIC-UNSEEN-LAYER")])
def test_vocabulary_inventory_and_oov_search_diagnostics(imported, field, name):
    p, _ = imported
    model, cfg = setup(p)
    oov = replace(p, product_id="SYNTHETIC-OOV", elements=(replace(p.elements[0], **{field: name}), *p.elements[1:]))
    query = search(model, cfg, oov, [p])
    assert query["status"] == "unavailable"
    assert query["reason"] == {"reason": "unknown_training_vocabulary", "roles": [name] if field == "role" else [], "layers": [name] if field == "layer" else []}
    candidate = search(model, cfg, p, [oov])
    assert candidate["reason"] == "no_eligible_candidates"
    assert candidate["excluded"] == [{"product_id": "SYNTHETIC-OOV", **query["reason"]}]
    with pytest.raises(ValueError, match="unknown_training_vocabulary"):
        embed(model, cfg, [oov])
    assert all(row["gate_status"] == "passed" for row in query["grid_gate"]["products"])


def test_extraction_coverage_status_reaches_model(imported):
    p, out = imported
    report = json.loads((out / "import-report.json").read_text())
    assert report["electrical_verification"] == "not_performed"
    assert report["status_summary"][0]["extraction_coverage"] == "partial"
    # Controlled coverage variants isolate the existing embedding feature;
    # complete here is explicitly synthetic coverage, never physical proof.
    variants = [replace(p, product_id="SYNTHETIC-UNAVAILABLE", connectivity_observation="unavailable", connections=()),
                p, replace(p, product_id="SYNTHETIC-COMPLETE", connectivity_observation="complete")]
    model, cfg = setup(p)
    seen = []
    hook = model.connectivity.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    vectors = encode_products(model, variants, 128, cfg["roles"], cfg["layers"])
    hook.remove()
    assert len(seen) == 1 and seen[0].tolist() == [0, 1, 2]
    assert not torch.equal(vectors[1], vectors[2])
    assert input_identity(variants[1]) != input_identity(variants[2])


def test_grid_loss_gates_and_original_relations_survive(tmp_path):
    root = tmp_path / "source"
    contact = tuple((y, x) for y in range(600, 603) for x in range(600, 603))
    write_model(root, [("A", "1-U.png", contact), ("B", "7-L.png", contact),
                       ("cno", "6_C.png", ((601, 601),))], model="SYNTHETIC-M")
    mapping = tmp_path / "metadata.json"
    mapping.write_text(json.dumps({"metadata_schema_version": "dl1_png_metadata_v1", "models": {
        "SYNTHETIC-M": {"lineage_id": "SYNTHETIC-FAMILY", "design_version": "SYNTHETIC-V1"}}}))
    out = tmp_path / "bundle"
    assert run_import(ImportOptions(root, out, True, mapping, 16, stack_order=write_stack_order(tmp_path / "stack-order.json", ['U', 'C', 'L'])))[1] == 0
    p, = load_manifest(out / "manifest.json")
    assert len(p.connections) == 1 and p.connections[0].mask.sum() == 1
    assert p.connections[0].endpoints[0].xy_um == pytest.approx((60.15, 60.15))
    assert p.connections[0].mask[601, 601] == 1 and p.cnt[601, 601] == 1
    check = grid_check(p, 16)
    assert check["status"] == "warning" and check["transform"]["resized_shape_yx"] == [13, 16]
    assert any(e["thin_line_warning"] for e in check["elements"])
    roles, layers = build_vocab([p]); cfg = model_config(16, roles, layers, model_kwargs={"dim": 16, "out_dim": 8, "heads": 4, "blocks": 0})
    model = ProductEncoder(len(roles), len(layers), **cfg["model_kwargs"])
    q = replace(p, product_id="SYNTHETIC-QUERY")
    blocked = search(model, cfg, q, [p])
    assert blocked["reason"] == {"reason": "grid_check_failed"}
    permitted = search(model, cfg, q, [p], allow_grid_check_failure=True)
    assert permitted["status"] == "available" and all(x["gate_status"] == "bypassed" for x in permitted["grid_gate"]["products"])
    with pytest.raises(ValueError, match="conflicts"):
        resolve_training_grid([p], 32)
    assert resolve_training_grid([p], 32, allow_grid_override=True)[1]["override"] is True
    split = {"schema_version": SPLIT_VERSION, "train": [p.product_id, q.product_id], "validation": [], "test": []}
    with pytest.raises(ValueError, match="grid check failed before training"):
        train([p, q], tmp_path / "must-not-train", epochs=1, grid=16, split_manifest=split,
              relations=(set(), {frozenset((p.product_id, q.product_id))}))


def test_offline_docs_and_pending_quality_contract(tmp_path, manifest_case):
    _, write, _ = manifest_case
    manifest = write()
    script = '''
import importlib.abc, sys
class NoPillow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "PIL" or fullname.startswith("PIL."):
            raise ImportError("Pillow deliberately unavailable")
sys.meta_path.insert(0, NoPillow())
from psdl1.data import load_manifest, build_vocab
from psdl1.train import model_config
from psdl1.model import ProductEncoder
from psdl1.checkpoint import save_checkpoint
from psdl1.search import load_model, embed
from psdl1.png_source import discover_source
p = load_manifest(sys.argv[1]); r,l = build_vocab(p)
c = model_config(4,r,l,model_kwargs={"dim":16,"out_dim":8,"heads":4,"blocks":0})
m = ProductEncoder(len(r),len(l),**c["model_kwargs"])
save_checkpoint(sys.argv[2],m,c)
m,c = load_model(sys.argv[2]); assert embed(m,c,p).shape == (1,8)
try: discover_source(sys.argv[3])
except ImportError as e: assert "png" in str(e).lower() or "Pillow" in str(e)
else: raise AssertionError("PNG must diagnose its missing optional dependency")
assert not any(k == "PIL" or k.startswith("PIL.") for k in sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script, str(manifest), str(tmp_path / "offline.pt"), str(tmp_path)],
                            capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert result.returncode == 0, result.stderr


def test_png_offline_documents_preserve_pending_quality():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    usage = (root / "docs/USAGE.md").read_text()
    contract = (root / "docs/data_contract.md").read_text()
    training = (root / "docs/training.md").read_text()
    for text in (readme, usage, training):
        assert "0.1" in text and "Pillow" in text
        assert "complete means extraction coverage only; electrical verification not performed" in text
        assert "G6" in text and "5–10" in text
    assert "hatchling" in readme and "hatchling" in usage
    assert "full frame extent" in readme and "full frame extent" in training
    assert "No model-input crop/tiling" in readme and "모델 입력은 그대로 전체 frame" in usage
    assert "retrieval-quality acceptance" in readme
    assert "03bf3e33c7107af75297aa298fa323e24018f28d" in contract
    assert "No labels, split, aliases or grid" in readme
    assert "intensity" in training and "off by default" in training
