"""검색: 신규 제품 하나를 넣어 닮은 순서로 기존 제품을 낸다.

    python -m psdl1.search --ckpt runs/dl1/dl1.pt --manifest data/manifest.json --query B --topk 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import Product, load_manifest
from .model import ProductEncoder
from .train import encode_products


def load_model(ckpt: str | Path) -> tuple[ProductEncoder, dict]:
    saved = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg = saved["config"]
    model = ProductEncoder(len(cfg["roles"]), len(cfg["layers"]), cfg["backbone"], None, use_unit_size=cfg["use_unit_size"])
    model.load_state_dict(saved["state_dict"])
    return model.eval(), cfg


@torch.no_grad()
def embed(model: ProductEncoder, cfg: dict, products: list[Product]) -> torch.Tensor:
    return encode_products(model, products, cfg["grid"], cfg["roles"], cfg["layers"])


@torch.no_grad()
def rank(model: ProductEncoder, cfg: dict, query: Product, candidates: list[Product]) -> list[tuple[str, float]]:
    """(product_id, 코사인 유사도)를 닮은 순서로. 비교 가능한 불량 조합은 여기서 판단하지 않는다."""
    z = embed(model, cfg, [query] + candidates)
    scores = (z[1:] @ z[0]).tolist()
    return sorted(zip([c.product_id for c in candidates], scores), key=lambda t: -t[1])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="DL1 검색")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--query", required=True, help="신규 제품의 product_id")
    ap.add_argument("--topk", type=int, default=5)
    a = ap.parse_args(argv)
    products = load_manifest(a.manifest)
    by_id = {p.product_id: p for p in products}
    if a.query not in by_id:
        raise SystemExit(f"manifest에 없는 제품: {a.query}")
    model, cfg = load_model(a.ckpt)
    for pid, score in rank(model, cfg, by_id[a.query], [p for p in products if p.product_id != a.query])[: a.topk]:
        print(f"{pid}\t{score:.4f}")


if __name__ == "__main__":
    main()
