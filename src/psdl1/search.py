"""Independent query/candidate search with explicit availability and stable ties."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .checkpoint import read_checkpoint, validate_config
from .data import load_manifest, input_identity, grid_check
from .model import ProductEncoder
from .schema import approval_status
from .train import encode_products


def load_model(ckpt):
    saved = read_checkpoint(ckpt)
    cfg = saved["config"]
    model = ProductEncoder(len(cfg["roles"]), len(cfg["layers"]), cfg["backbone"], None,
                           use_unit_size=cfg["use_unit_size"], **cfg["model_kwargs"])
    model.load_state_dict(saved["state_dict"], strict=True)
    model.config_fingerprint = saved["fingerprint"]
    model.training_metadata = saved["metadata"]
    model.trained_epochs = saved["epoch"]
    model.weights_hash = saved["weights_hash"]
    return model.eval(), cfg


def vocabulary_reason(product, cfg):
    roles = sorted({e.role for e in product.elements} - cfg["roles"].keys())
    layers = sorted({e.layer for e in product.elements} - cfg["layers"].keys())
    if roles or layers:
        return {"reason": "unknown_training_vocabulary", "roles": roles, "layers": layers}
    if any(e.mask is None for e in product.elements):
        return {"reason": "missing_shape_observation"}
    return None


def inference_grid_selection(products, cfg, *, allow_grid_override=False):
    if type(allow_grid_override) is not bool:
        raise ValueError("allow_grid_override must be boolean")
    candidates = sorted({p.grid_check["manifest_candidate_grid"] for p in products
                         if p.grid_check is not None and "manifest_candidate_grid" in p.grid_check})
    conflicts = [value for value in candidates if value != cfg["grid"]]
    if conflicts and not allow_grid_override:
        raise ValueError(f"checkpoint grid {cfg['grid']} conflicts with manifest candidate_grid {conflicts}; --allow-grid-override required")
    return {"selected_grid": cfg["grid"], "manifest_candidate_grids": candidates,
            "source": "checkpoint", "override": bool(conflicts), "override_requested": allow_grid_override}


def inference_grid_checks(products, grid, *, allow_grid_check_failure=False):
    if type(allow_grid_check_failure) is not bool:
        raise ValueError("allow_grid_check_failure must be boolean")
    checks = []
    for p in products:
        report = grid_check(p, grid)
        failed = any(e.get("thin_line_warning") or e.get("bridge_or_component_loss") for e in report["elements"])
        checks.append({**report, "gate_status": "bypassed" if failed and allow_grid_check_failure else "blocked" if failed else "passed"})
    return {"policy": "strict_default", "bypass_requested": allow_grid_check_failure, "products": checks}


@torch.no_grad()
def embed(model, cfg, products, *, batch_size=32, cache=None):
    fingerprint = validate_config(cfg)
    if getattr(model, "config_fingerprint", fingerprint) != fingerprint:
        raise ValueError("model/config fingerprint mismatch")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for p in products:
        reason = vocabulary_reason(p, cfg)
        if reason:
            raise ValueError(f"{p.product_id}: {reason}")
    model.eval()
    if not products:
        return torch.empty(0, model.head[-1].out_features)
    # Canonical inference shape: neither padding neighbours nor requested chunk size
    # changes the model's arithmetic. Chunks bound memory; forwards are per product.
    cache = {} if cache is None else cache
    vectors = []
    for start in range(0, len(products), batch_size):
        for p in products[start:start + batch_size]:
            key = (id(model), fingerprint, p.product_id, input_identity(p))
            if key not in cache:
                cache[key] = encode_products(model, [p], cfg["grid"], cfg["roles"], cfg["layers"]).cpu()
            vectors.append(cache[key])
    z = torch.cat(vectors)
    if not torch.isfinite(z).all():
        raise ValueError("nonfinite search embeddings")
    return z


@torch.no_grad()
def search(model, cfg, query, candidates, *, topk=5, batch_size=32, allow_grid_override=False, allow_grid_check_failure=False, embedding_cache=None):
    if type(topk) is not int or topk <= 0:
        raise ValueError("topk must be positive")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    fingerprint = validate_config(cfg)
    if getattr(model, "config_fingerprint", fingerprint) != fingerprint:
        raise ValueError("model/config fingerprint mismatch")
    if len({p.product_id for p in candidates}) != len(candidates):
        raise ValueError("candidate product IDs must be unique")
    selection = inference_grid_selection([query, *candidates], cfg, allow_grid_override=allow_grid_override)
    gate = inference_grid_checks([query, *sorted(candidates, key=lambda p: p.product_id)], cfg["grid"], allow_grid_check_failure=allow_grid_check_failure)
    blocked = {r["product_id"] for r in gate["products"] if r["gate_status"] == "blocked"}
    result = {"grid": cfg["grid"], "grid_selection": selection, "grid_gate": gate,
              "inference_policy": "one_forward_per_product_cached_within_evaluation; batch_size_is_chunk_bound",
              "query_id": query.product_id, "approval": approval_status(), "status": "unavailable", "reason": None,
              "rankings": [], "excluded": [], "candidate_count": len(candidates), "eligible_count": 0,
              "score_policy": "canonical_single_product_float64_cosine_round6_then_id",
              "fingerprint": validate_config(cfg), "query_connectivity": query.connectivity_status}
    reason = vocabulary_reason(query, cfg)
    if reason or query.product_id in blocked:
        result["reason"] = reason or {"reason": "grid_check_failed"}
        return result
    eligible = []
    for p in candidates:
        reason = {"reason": "query_itself"} if p.product_id == query.product_id else vocabulary_reason(p, cfg)
        reason = reason or ({"reason": "grid_check_failed"} if p.product_id in blocked else None)
        if reason:
            result["excluded"].append({"product_id": p.product_id, **reason})
        else:
            eligible.append(p)
    result["excluded"].sort(key=lambda row: row["product_id"])
    result["eligible_count"] = len(eligible)
    if not eligible:
        result["reason"] = "no_candidates" if not candidates else "no_eligible_candidates"
        return result
    cache_kwargs = {} if embedding_cache is None else {"cache": embedding_cache}
    qz = embed(model, cfg, [query], batch_size=batch_size, **cache_kwargs)[0]
    # Single-query search retains scores only; evaluation supplies a per-command embedding cache.
    scores = []
    for start in range(0, len(eligible), batch_size):
        chunk = eligible[start:start + batch_size]
        vectors = embed(model, cfg, chunk, batch_size=batch_size, **cache_kwargs).to(torch.float64)
        scores.extend(round(float((vector * qz.to(torch.float64)).sum()), 6) for vector in vectors)
    ranked = sorted(zip(eligible, scores), key=lambda pair: (-pair[1], pair[0].product_id))[:topk]
    result.update(status="available", rankings=[{"product_id": p.product_id, "score": score,
                    "connectivity_status": p.connectivity_status, "input_identity": input_identity(p)} for p, score in ranked])
    return result


def rank(model, cfg, query, candidates, *, topk=None, batch_size=32):
    result = search(model, cfg, query, candidates, topk=max(1, len(candidates)) if topk is None else topk, batch_size=batch_size)
    return [(row["product_id"], row["score"]) for row in result["rankings"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description="DL1 candidate search JSON")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--query-manifest", required=True)
    ap.add_argument("--candidates-manifest", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--allow-grid-override", action="store_true")
    ap.add_argument("--allow-grid-check-failure", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    model, cfg = load_model(a.ckpt)
    queries = {p.product_id: p for p in load_manifest(a.query_manifest, grid=cfg["grid"])}
    if a.query not in queries:
        raise ValueError("query ID absent from independent query manifest")
    candidates = load_manifest(a.candidates_manifest, grid=cfg["grid"])
    result = search(model, cfg, queries[a.query], candidates, topk=a.topk, batch_size=a.batch_size, allow_grid_override=a.allow_grid_override, allow_grid_check_failure=a.allow_grid_check_failure)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if a.out:
        Path(a.out).write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
