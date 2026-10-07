"""학습: 같은 제품을 다르게 변형한 두 장은 가깝게, 다른 제품은 구분.

    python -m psdl1.train --manifest data/manifest.json --out runs/dl1 --epochs 200
    python -m psdl1.train ... --backbone convnextv2_tiny --weights weights/convnextv2_tiny.safetensors
    python -m psdl1.train ... --relations data/relations.csv   # product_a,product_b,similar|different
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from .data import Product, augment, build_vocab, collate, load_manifest, product_tensors, unit_feature
from .losses import masked_contrastive_loss, relation_matrix
from .model import ProductEncoder


def load_relations(path: str | Path) -> tuple[set[frozenset], set[frozenset]]:
    similar, different = set(), set()
    with open(path, newline="", encoding="utf-8") as f:
        for a, b, kind in csv.reader(f):
            if kind.strip() not in ("similar", "different"):
                raise ValueError(f"관계는 similar 또는 different: {kind!r}")
            (similar if kind.strip() == "similar" else different).add(frozenset((a.strip(), b.strip())))
    return similar, different


def encode_products(model: ProductEncoder, products: list[Product], grid: int, roles: dict, layers: dict, generator: torch.Generator | None = None):
    """generator를 주면 변형한 view, 안 주면 원본."""
    items = []
    for p in products:
        images, role_ids, layer_ids = product_tensors(p, grid, roles, layers)
        if generator is not None:
            images = augment(images, generator)
        items.append((images, role_ids, layer_ids, unit_feature(p.unit_size_um)))
    images, role_ids, layer_ids, valid, unit = collate(items)
    return model(images, role_ids, layer_ids, valid, unit)


def train(products: list[Product], out_dir: str | Path, epochs: int = 100, grid: int = 256, batch_size: int = 8, lr: float = 3e-4,
          backbone: str = "tiny", weights_file: str | None = None, use_unit_size: bool = False,
          relations: tuple[set[frozenset], set[frozenset]] | None = None, seed: int = 0) -> list[float]:
    if len(products) < 2:
        raise ValueError("제품이 둘 이상 있어야 학습할 수 있다")
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    roles, layers = build_vocab(products)
    model = ProductEncoder(len(roles), len(layers), backbone, weights_file, use_unit_size=use_unit_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    warmup = max(1, epochs // 10)
    history = []
    for epoch in range(epochs):
        for group in opt.param_groups:
            group["lr"] = lr * min(1.0, (epoch + 1) / warmup)
        order = torch.randperm(len(products), generator=gen).tolist()
        total, steps = 0.0, 0
        for start in range(0, len(order), batch_size):
            batch = [products[i] for i in order[start:start + batch_size]]
            if len(batch) < 2:
                continue
            ids = [p.product_id for p in batch]
            rel = relation_matrix(ids, *(relations or (None, None)))
            z1 = encode_products(model, batch, grid, roles, layers, gen)
            z2 = encode_products(model, batch, grid, roles, layers, gen)
            l1, n1 = masked_contrastive_loss(z1, z2, rel)
            l2, n2 = masked_contrastive_loss(z2, z1, rel.T)
            if n1 + n2 == 0:
                continue
            loss = 0.5 * (l1 + l2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, steps = total + loss.item(), steps + 1
        if steps == 0:
            raise ValueError("구분할 쌍이 한 번도 없었다. 정답 파일에 different 쌍을 넣거나 정답 파일 없이 학습한다")
        history.append(total / steps)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = {"grid": grid, "backbone": backbone, "use_unit_size": use_unit_size, "roles": roles, "layers": layers}
    torch.save({"state_dict": model.state_dict(), "config": config}, out / "dl1.pt")
    (out / "history.json").write_text(json.dumps(history), encoding="utf-8")
    return history


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="DL1 학습")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--grid", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone", default="tiny", help="tiny 또는 timm 모델 이름")
    ap.add_argument("--weights", default=None, help="사전학습 가중치 파일 (네트워크로 받지 않음)")
    ap.add_argument("--use-unit-size", action="store_true")
    ap.add_argument("--relations", default=None, help="product_a,product_b,similar|different")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    history = train(load_manifest(a.manifest), a.out, a.epochs, a.grid, a.batch_size, a.lr, a.backbone, a.weights, a.use_unit_size,
                    load_relations(a.relations) if a.relations else None, a.seed)
    print(f"마지막 손실 {history[-1]:.4f} · 저장 {a.out}/dl1.pt")


if __name__ == "__main__":
    main()
