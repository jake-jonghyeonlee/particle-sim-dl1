from dataclasses import replace

import pytest
import torch

from conftest import make_product
from psdl1.data import build_vocab, load_manifest
from psdl1.evaluate import evaluate, normalized_image_distance
from psdl1.model import ProductEncoder
from psdl1.search import search
from psdl1.split import SPLIT_VERSION
from psdl1.train import model_config, structured_batch


def setup_model(products):
    roles, layers = build_vocab(products)
    cfg = model_config(16, roles, layers)
    return ProductEncoder(len(roles), len(layers)).eval(), cfg


def test_stable_ties_and_candidate_batch_bound(products, monkeypatch):
    model, cfg = setup_model(products)
    query = replace(products[0], product_id="query")
    candidates = [replace(products[0], product_id=pid) for pid in ("Z", "B", "A", "C", "D")]
    sizes = []
    def constant_embed(model, cfg, items, *, batch_size):
        sizes.append(len(items))
        return torch.tensor([[1., 0.]] * len(items))
    monkeypatch.setattr("psdl1.search.embed", constant_embed)
    result = search(model, cfg, query, candidates, topk=3, batch_size=2)
    assert [r["product_id"] for r in result["rankings"]] == ["A", "B", "C"]
    assert sizes == [1, 2, 2, 1]
    reverse = search(model, cfg, query, candidates[::-1], topk=3, batch_size=2)
    assert reverse["rankings"] == result["rankings"]


@pytest.mark.parametrize("topk", [0, -1, True, 1.5])
def test_invalid_topk(products, topk):
    model, cfg = setup_model(products)
    with pytest.raises(ValueError, match="topk"):
        search(model, cfg, products[0], products[1:], topk=topk)


def test_explicit_no_candidate_reason(products):
    model, cfg = setup_model(products)
    result = search(model, cfg, products[0], [])
    assert result["status"] == "unavailable" and result["reason"] == "no_candidates"


@pytest.mark.parametrize("field", ["role", "layer"])
def test_role_or_layer_absent_from_training_is_rejected(products, field):
    model, cfg = setup_model(products)
    unknown = replace(products[0], product_id="unknown", elements=(replace(products[0].elements[0], **{field: "never-trained"}), *products[0].elements[1:]))
    result = search(model, cfg, unknown, products)
    assert result["reason"]["reason"] == "unknown_training_vocabulary"
    assert result["reason"]["roles" if field == "role" else "layers"] == ["never-trained"]
    result = search(model, cfg, products[0], [unknown])
    assert result["reason"] == "no_eligible_candidates"
    assert result["excluded"][0]["reason"] == "unknown_training_vocabulary"


def test_checkpoint_config_fingerprint_checked_before_encoding(products):
    model, cfg = setup_model(products)
    model.config_fingerprint = "wrong"
    with pytest.raises(ValueError, match="fingerprint"):
        search(model, cfg, products[0], products[1:])


def endpoint_heldout(manifest_case):
    _, write, _ = manifest_case
    product, = load_manifest(write())
    product = replace(product, connections=(replace(product.connections[0], kind="CNT"),))
    train = replace(product, product_id="train", lineage_id="training-family")
    query = replace(product, product_id="query", lineage_id="heldout-family")
    c = product.connections[0]
    other = replace(product, product_id="endpoint-negative", lineage_id="heldout-family",
                    connections=(replace(c, endpoints=(replace(c.endpoints[0], xy_um=(1.075, -1.975)), c.endpoints[1])),))
    split = {"schema_version": SPLIT_VERSION, "train": ["train"], "validation": [], "test": ["query", "endpoint-negative"]}
    return [train, query, other], split


def test_endpoint_only_hard_negative_is_heldout_not_untrained_ranking_claim(manifest_case):
    products, split = endpoint_heldout(manifest_case)
    train, query, other = products
    roles, layers = build_vocab([train])
    _, extra, prepared = structured_batch([query, other], 4, roles, layers)
    assert torch.equal(prepared[0].images, prepared[1].images)
    assert not torch.equal(extra["connection_features"][0], extra["connection_features"][1])
    assert normalized_image_distance(query, other, 4) == 0  # shape baseline is blind to endpoints
    model, cfg = setup_model([train])
    model.training_metadata = {"split": {"train": ["train"]}}
    with pytest.raises(ValueError, match="untrained"):
        evaluate(model, cfg, products, split, (set(), {frozenset(("query", "endpoint-negative"))}), hard_negative_pairs=[("query", "endpoint-negative")])


def test_evaluation_metrics_and_unknown_coverage_using_fixed_ranking_fixture(manifest_case, monkeypatch):
    products, split = endpoint_heldout(manifest_case)
    products.append(replace(products[1], product_id="positive", lineage_id="other-heldout"))
    split["test"].append("positive")
    model, cfg = setup_model([products[0]])
    model.training_metadata = {"split": {"train": ["train"]}}
    from psdl1.checkpoint import json_hash
    from psdl1.data import input_identity
    model.training_metadata = {"split": split, "input_hash": json_hash({p.product_id: {"identity": input_identity(p), "lineage": p.lineage_id, "design": p.design_version} for p in products})}
    model.trained_epochs = 1  # direct harness fixture, not an encoder performance test
    def fixed_search(model, cfg, query, candidates, **kwargs):
        return {"status": "available", "reason": None, "rankings": [{"product_id": p.product_id, "score": 0} for p in sorted(candidates, key=lambda p: p.product_id)], "excluded": []}
    monkeypatch.setattr("psdl1.evaluate.search", fixed_search)
    positive = {frozenset(("query", "positive"))}
    negative = {frozenset(("query", "endpoint-negative"))}
    report = evaluate(model, cfg, products, split, (positive, negative), topk=2, hard_negative_pairs=[("query", "endpoint-negative")])
    assert report["evaluated_queries"] == 2
    assert report["recall_at_k"] == 1 and report["baseline_recall_at_k"] == 1
    assert report["unavailable_queries"] == 1
    assert report["hard_negative_pairs"] == [["endpoint-negative", "query"]]
    assert report["performance_acceptance"] == "not_established"
    assert report["queries"][1]["unknown_candidate_pairs"] == 1


def test_search_cli_uses_independent_query_and_candidate_manifests(manifest_case, tmp_path, monkeypatch):
    import json
    from psdl1.search import main
    _, write, _ = manifest_case
    path = write()
    query, = load_manifest(path)
    candidate = replace(query, product_id="candidate")
    query = replace(query, product_id="independent-query")
    model, cfg = setup_model([query])
    monkeypatch.setattr("psdl1.search.load_model", lambda path: (model, cfg))
    loaded_paths = []
    def loader(path, **kwargs):
        loaded_paths.append(path)
        return [query] if path == "query.json" else [candidate]
    monkeypatch.setattr("psdl1.search.load_manifest", loader)
    out = tmp_path / "search.json"
    main(["--ckpt", "fixture.pt", "--query-manifest", "query.json", "--candidates-manifest", "candidates.json",
          "--query", "independent-query", "--out", str(out)])
    result = json.loads(out.read_text())
    assert loaded_paths == ["query.json", "candidates.json"]
    assert result["query_id"] == "independent-query"
    assert result["rankings"][0]["product_id"] == "candidate"  # sole candidate, no performance assertion


def test_image_baseline_accepts_binary_255_valid_and_shape_encodings(products):
    original = products[0]
    encoded = replace(original, elements=tuple(replace(e, mask=e.mask * 255) for e in original.elements),
                      valid=torch.ones_like(torch.from_numpy(original.elements[0].mask)).numpy() * 255)
    assert normalized_image_distance(original, encoded, 16) == 0
