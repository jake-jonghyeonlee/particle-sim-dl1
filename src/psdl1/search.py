"""Independent query/candidate search with explicit availability and stable ties."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .checkpoint import read_checkpoint, validate_config
from .data import load_manifest, input_identity
from .model import ProductEncoder
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


@torch.no_grad()
def embed(model, cfg, products, *, batch_size=32):
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
    chunks = [encode_products(model, products[i:i + batch_size], cfg["grid"], cfg["roles"], cfg["layers"]).cpu()
              for i in range(0, len(products), batch_size)]
    z = torch.cat(chunks)
    if not torch.isfinite(z).all():
        raise ValueError("nonfinite search embeddings")
    return z


@torch.no_grad()
def search(model, cfg, query, candidates, *, topk=5, batch_size=32):
    if type(topk) is not int or topk <= 0:
        raise ValueError("topk must be positive")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    validate_config(cfg)
    if len({p.product_id for p in candidates}) != len(candidates):
        raise ValueError("candidate product IDs must be unique")
    result = {"query_id": query.product_id, "approval": {"input_method": "user_approved_2026_10_09", "real_grid_linewidth": "pending", "endpoint_source_format": "pending", "G6": "pending"}, "status": "unavailable", "reason": None,
              "rankings": [], "excluded": [], "candidate_count": len(candidates), "eligible_count": 0,
              "fingerprint": validate_config(cfg), "query_connectivity": query.connectivity_status}
    reason = vocabulary_reason(query, cfg)
    if reason:
        result["reason"] = reason
        return result
    eligible = []
    for p in candidates:
        reason = {"reason": "query_itself"} if p.product_id == query.product_id else vocabulary_reason(p, cfg)
        if reason:
            result["excluded"].append({"product_id": p.product_id, **reason})
        else:
            eligible.append(p)
    result["eligible_count"] = len(eligible)
    if not eligible:
        result["reason"] = "no_candidates" if not candidates else "no_eligible_candidates"
        return result
    qz = embed(model, cfg, [query], batch_size=batch_size)[0]
    # Encode candidate chunks and retain only scores; no whole-corpus embedding tensor.
    scores = []
    for start in range(0, len(eligible), batch_size):
        chunk = eligible[start:start + batch_size]
        scores.extend((embed(model, cfg, chunk, batch_size=batch_size) @ qz).tolist())
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
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    model, cfg = load_model(a.ckpt)
    queries = {p.product_id: p for p in load_manifest(a.query_manifest, grid=cfg["grid"])}
    if a.query not in queries:
        raise ValueError("query ID absent from independent query manifest")
    candidates = load_manifest(a.candidates_manifest, grid=cfg["grid"])
    result = search(model, cfg, queries[a.query], candidates, topk=a.topk, batch_size=a.batch_size)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if a.out:
        Path(a.out).write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
