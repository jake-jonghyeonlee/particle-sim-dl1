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
    cfg = model_config(products[0].grid_check["manifest_candidate_grid"] if products[0].grid_check else 64, roles, layers)
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
        ordered = sorted(candidates, key=lambda p: (p.product_id != "query", p.product_id)) if query.product_id == "positive" else sorted(candidates, key=lambda p: p.product_id)
        return {"status": "available", "reason": None, "rankings": [{"product_id": p.product_id, "score": 0.9 - i * 0.3} for i, p in enumerate(ordered[:kwargs["topk"]])], "excluded": []}
    monkeypatch.setattr("psdl1.evaluate.search", fixed_search)
    positive = {frozenset(("query", "positive"))}
    negative = {frozenset(("query", "endpoint-negative"))}
    report = evaluate(model, cfg, products, split, (positive, negative), topk=1, hard_negative_pairs=[("query", "endpoint-negative")])
    assert report["evaluated_queries"] == 2
    assert report["recall_at_k"] == 0.5 and report["baseline_recall_at_k"] == 0
    per_query = {row["query_id"]: row for row in report["queries"]}
    assert per_query["query"]["recall_at_k"] == 0 and per_query["positive"]["recall_at_k"] == 1
    item = report["hard_negative_evaluation"]["items"][0]
    direction = next(d for d in item["directions"] if d["query_id"] == "query")
    assert direction["score"] == pytest.approx(0.9) and direction["rank"] == 1
    assert direction["known_positives"] == [{"product_id": "positive", "score": pytest.approx(0.6), "rank": 2}]
    assert direction["below_all_known_positives"] is False
    assert direction["margin_to_lowest_positive"] == pytest.approx(-0.3)
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


def test_n3_evaluation_cli_scores_hard_negative_outside_topk_and_missing_list(manifest_case, tmp_path, monkeypatch):
    import json
    from psdl1.checkpoint import json_hash
    from psdl1.data import input_identity
    from psdl1.evaluate import main
    products, split = endpoint_heldout(manifest_case)
    products.append(replace(products[1], product_id="positive", lineage_id="positive-family"))
    split["test"].append("positive")
    model, cfg = setup_model([products[0]])
    model.trained_epochs = 1  # metric fixture, not performance evidence
    model.training_metadata = {"split": split, "input_hash": json_hash({p.product_id: {"identity": input_identity(p), "lineage": p.lineage_id, "design": p.design_version} for p in products})}
    monkeypatch.setattr("psdl1.evaluate.load_model", lambda path: (model, cfg))
    monkeypatch.setattr("psdl1.evaluate.load_manifest", lambda *args, **kwargs: products)
    def scored_fixture(model, cfg, query, candidates, **kwargs):
        ordered = sorted(candidates, key=lambda p: (p.product_id != "positive", p.product_id))
        return {"status": "available", "reason": None, "excluded": [],
                "rankings": [{"product_id": p.product_id, "score": 0.8 - i * 0.3} for i, p in enumerate(ordered[:kwargs["topk"]])]}
    monkeypatch.setattr("psdl1.evaluate.search", scored_fixture)
    split_file, labels, hard, out = (tmp_path / name for name in ("split.json", "relations.csv", "hard.json", "result.json"))
    split_file.write_text(json.dumps(split))
    labels.write_text("query,positive,similar\nquery,endpoint-negative,different\n")
    hard.write_text(json.dumps([["query", "endpoint-negative"]]))
    args = ["--ckpt", "fixture.pt", "--manifest", "fixture.json", "--split", str(split_file), "--relations", str(labels), "--topk", "1", "--out", str(out)]
    main([*args, "--hard-negatives", str(hard)])
    report = json.loads(out.read_text())
    direction = next(d for d in report["hard_negative_evaluation"]["items"][0]["directions"] if d["query_id"] == "query")
    assert direction["score"] == pytest.approx(0.5)
    assert direction["rank"] == 2  # outside the requested topk=1, still scored
    assert direction["known_positives"] == [{"product_id": "positive", "score": pytest.approx(0.8), "rank": 1}]
    assert direction["below_all_known_positives"] is True
    assert direction["margin_to_lowest_positive"] == pytest.approx(0.3)
    main(args)
    assert json.loads(out.read_text())["hard_negative_evaluation"] == {"status": "unavailable", "reason": "none_supplied", "items": []}


def test_n3_unknown_hard_negative_has_unavailable_scores_not_available_claim(manifest_case):
    from psdl1.checkpoint import json_hash
    from psdl1.data import input_identity
    products, split = endpoint_heldout(manifest_case)
    negative = products[2]
    products[2] = replace(negative, elements=(replace(negative.elements[0], role="unseen"), *negative.elements[1:]))
    model, cfg = setup_model([products[0]])
    model.trained_epochs = 1  # provenance fixture; both searches stop before any forward
    model.training_metadata = {"split": split, "input_hash": json_hash({p.product_id: {"identity": input_identity(p), "lineage": p.lineage_id, "design": p.design_version} for p in products})}
    report = evaluate(model, cfg, products, split, (set(), {frozenset(("query", "endpoint-negative"))}), hard_negative_pairs=[("query", "endpoint-negative")])
    hard = report["hard_negative_evaluation"]
    assert hard["status"] == "unavailable" and hard["reason"] == "scores_unavailable"
    assert all(d["score"] is None and d["status"] == "unavailable" and d["reason"] for d in hard["items"][0]["directions"])
