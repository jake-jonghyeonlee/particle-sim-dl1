"""Held-out evaluation harness; no performance claim from an untrained encoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .checkpoint import json_hash
from .data import input_identity, load_manifest, normalize_unit
from .losses import validate_relations
from .search import load_model, search, vocabulary_reason
from .schema import binary_mask, approval_status
from .split import validate_split
from .train import load_relations


def normalized_image_distance(a, b, grid):
    """Role/layer-aligned mean absolute coverage difference; no learnt parameters."""
    def images(p):
        valid = np.ones_like(p.elements[0].mask) if p.valid is None else binary_mask(p.valid)
        return {(e.role, e.layer): normalize_unit(binary_mask(e.mask) * valid, grid) for e in p.elements}
    aa, bb = images(a), images(b)
    keys = sorted(aa.keys() | bb.keys())
    zero = np.zeros((grid, grid), dtype=np.float32)
    return float(np.mean([np.abs(aa.get(k, zero) - bb.get(k, zero)).mean() for k in keys]))


def evaluate(model, cfg, products, split_manifest, relations, *, topk=5, batch_size=32, hard_negative_pairs=()):
    if type(topk) is not int or topk <= 0:
        raise ValueError("topk must be positive")
    parts = validate_split(products, split_manifest)
    similar, different = validate_relations(*relations, product_ids=[p.product_id for p in products])
    heldout = parts["test"]
    if not heldout:
        raise ValueError("held-out test partition required")
    # Reject overlap with the checkpoint's actual training partition, including lineage.
    trained_ids = set(getattr(model, "training_metadata", {}).get("split", {}).get("train", []))
    if not trained_ids or set(split_manifest["train"]) != trained_ids:
        raise ValueError("evaluation split must match checkpoint training products")
    trained_epochs = getattr(model, "trained_epochs", 0)
    if trained_epochs <= 0:
        raise ValueError("untrained encoder ranking cannot validate held-out performance")
    expected_hash = getattr(model, "training_metadata", {}).get("input_hash")
    records = {p.product_id: {"identity": input_identity(p), "lineage": p.lineage_id, "design": p.design_version} for p in products}
    if expected_hash is None or expected_hash != json_hash(records):
        raise ValueError("evaluation inputs/lineages differ from checkpoint split provenance")
    if getattr(model, "training_metadata", {}).get("split") != split_manifest:
        raise ValueError("evaluation split differs from checkpoint split provenance")
    hard = {frozenset(pair) for pair in hard_negative_pairs}
    validate_relations(set(), hard, product_ids=[p.product_id for p in products])
    heldout_ids = {p.product_id for p in heldout}
    if not hard <= different or any(not pair <= heldout_ids for pair in hard):
        raise ValueError("hard negatives must be labelled different and wholly held out")
    rows, recalls, baseline_recalls = [], [], []
    full_results = {}
    for query in heldout:
        candidates = [p for p in heldout if p.product_id != query.product_id]
        full = search(model, cfg, query, candidates, topk=max(1, len(candidates)), batch_size=batch_size)
        full_results[query.product_id] = full
        result = {**full, "rankings": full["rankings"][:topk]}
        eligible = [p for p in candidates if vocabulary_reason(p, cfg) is None]
        baseline = sorted(((p.product_id, normalized_image_distance(query, p, cfg["grid"])) for p in eligible), key=lambda row: (row[1], row[0]))[:topk] if result["status"] == "available" else []
        positive = {p.product_id for p in eligible if frozenset((query.product_id, p.product_id)) in similar}
        known = sum(frozenset((query.product_id, p.product_id)) in similar | different for p in eligible)
        status = "available" if positive and result["status"] == "available" else "unavailable"
        recall = len(positive & {r["product_id"] for r in result["rankings"]}) / len(positive) if status == "available" else None
        base_recall = len(positive & {pid for pid, _ in baseline}) / len(positive) if status == "available" else None
        if recall is not None:
            recalls.append(recall)
            baseline_recalls.append(base_recall)
        rows.append({"query_id": query.product_id, "lineage_id": query.lineage_id,
                     "metric_status": status, "metric_reason": None if status == "available" else (result["reason"] or "no_known_positive_candidate"),
                     "known_candidate_pairs": known, "unknown_candidate_pairs": len(eligible) - known,
                     "known_positive_candidates": len(positive), "recall_at_k": recall, "baseline_recall_at_k": base_recall,
                     "model": result, "baseline": [{"product_id": pid, "distance": score} for pid, score in baseline]})
    hard_items = []
    for pair in sorted(map(sorted, hard)):
        directions = []
        for qid, nid in (pair, pair[::-1]):
            full = full_results[qid]
            ranked = full["rankings"]
            negative = next((r for r in ranked if r["product_id"] == nid), None)
            positives = [r for r in ranked if frozenset((qid, r["product_id"])) in similar]
            comparison_available = negative is not None and bool(positives)
            directions.append({"query_id": qid, "negative_id": nid,
                               "status": "available" if negative else "unavailable",
                               "reason": None if negative else (full["reason"] or "hard_negative_candidate_excluded"),
                               "score": negative["score"] if negative else None,
                               "rank": next((i + 1 for i, r in enumerate(ranked) if r["product_id"] == nid), None),
                               "known_positives": [{"product_id": r["product_id"], "score": r["score"],
                                                    "rank": ranked.index(r) + 1} for r in positives],
                               "comparison_status": "available" if comparison_available else "unavailable",
                               "comparison_reason": None if comparison_available else "no_known_positive_candidate" if negative else "hard_negative_score_unavailable",
                               "below_all_known_positives": all(negative["score"] < r["score"] for r in positives) if comparison_available else None,
                               "margin_to_lowest_positive": min(r["score"] for r in positives) - negative["score"] if comparison_available else None})
        scored = sum(d["status"] == "available" for d in directions)
        hard_items.append({"pair": pair, "status": "available" if scored == 2 else "partial" if scored else "unavailable",
                           "reason": None if scored == 2 else "some_scores_unavailable" if scored else "scores_unavailable",
                           "directions": directions, "acceptance_threshold": "G6_pending"})
    all_scored = bool(hard_items) and all(item["status"] == "available" for item in hard_items)
    any_scored = any(item["status"] != "unavailable" for item in hard_items)
    hard_evaluation = {"status": "available" if all_scored else "partial" if any_scored else "unavailable",
                       "reason": None if all_scored else "none_supplied" if not hard_items else "some_scores_unavailable" if any_scored else "scores_unavailable",
                       "items": hard_items}
    return {"schema_version": "dl1_evaluation_v1", "approval": approval_status(), "performance_acceptance": "not_established",
            "topk": topk, "model_config": cfg, "checkpoint_weights_hash": getattr(model, "weights_hash", None),
            "input_hash": expected_hash, "code_hash": model.training_metadata.get("code_hash"), "split_hash": json_hash(split_manifest), "evaluated_queries": len(recalls),
            "unavailable_queries": len(rows) - len(recalls), "recall_at_k": float(np.mean(recalls)) if recalls else None,
            "baseline_recall_at_k": float(np.mean(baseline_recalls)) if baseline_recalls else None,
            "hard_negative_pairs": sorted(map(sorted, hard)), "hard_negative_evaluation": hard_evaluation, "queries": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare held-out candidate and normalized-image baseline")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--relations", required=True)
    ap.add_argument("--hard-negatives", help="optional JSON list of held-out product-ID pairs")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    model, cfg = load_model(a.ckpt)
    result = evaluate(model, cfg, load_manifest(a.manifest, grid=cfg["grid"]), json.loads(Path(a.split).read_text()),
                      load_relations(a.relations), topk=a.topk, batch_size=a.batch_size,
                      hard_negative_pairs=json.loads(Path(a.hard_negatives).read_text()) if a.hard_negatives else ())
    Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
