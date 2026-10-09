"""Candidate supervised training; labels and held-out acceptance remain G6 pending."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from .checkpoint import code_hash, file_hash, json_hash, resume_checkpoint, save_checkpoint, validate_config
from .data import Product, build_vocab, collate, grid_check, input_identity, load_manifest, prepare_product, unit_feature
from .losses import NEGATIVE, masked_contrastive_loss, relation_coverage, relation_matrix, validate_relations
from .model import ProductEncoder
from .split import training_only_split, validate_split
from .schema import approval_status


def load_relations(path):
    similar, different = set(), set()
    with open(path, newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if len(row) != 3:
                raise ValueError("relations CSV requires product_a,product_b,similar|different")
            a, b, kind = (v.strip() for v in row)
            if kind not in {"similar", "different"}:
                raise ValueError("relation must be similar or different")
            (similar if kind == "similar" else different).add(frozenset((a, b)))
    return validate_relations(similar, different)


class RelationBatchSampler:
    """Every labelled negative edge is scheduled each epoch, including sparse edges."""
    def __init__(self, product_ids, batch_size, similar=None, different=None, *, unlabelled_negatives=False, lineage_ids=None):
        if type(batch_size) is not int or batch_size < 2:
            raise ValueError("batch_size must be at least two")
        self.ids = list(product_ids)
        self.batch_size = batch_size
        self.similar, self.different = validate_relations(similar, different, product_ids=self.ids)
        self.relation = relation_matrix(self.ids, self.similar, self.different, unlabelled_negatives=unlabelled_negatives, lineage_ids=lineage_ids)
        self.edges = [(i, j) for i in range(len(self.ids)) for j in range(i + 1, len(self.ids)) if self.relation[i, j] == NEGATIVE]
        if not self.edges:
            raise ValueError("구분할 쌍이 없다: provide different labels or explicitly opt into unlabelled negatives")

    def batches(self, generator):
        edges = [self.edges[i] for i in torch.randperm(len(self.edges), generator=generator).tolist()]
        remaining = set(edges)
        result = []
        for edge in edges:
            if edge not in remaining:
                continue
            batch = list(edge)
            # Fill with source positives first, then other products; never duplicate a product.
            order = torch.randperm(len(self.ids), generator=generator).tolist()
            positives = [i for i in order if i not in batch and any(frozenset((self.ids[i], self.ids[j])) in self.similar for j in batch)]
            batch.extend((positives + [i for i in order if i not in batch and i not in positives])[:self.batch_size - len(batch)])
            members = set(batch)
            remaining = {e for e in remaining if not set(e) <= members}
            result.append(batch)
        return result


def structured_batch(products, grid, roles, layers, *, augmentation="none", generator=None, prepared=None):
    if not products:
        raise ValueError("cannot encode an empty product batch")
    if augmentation not in {"none", "intensity"}:
        raise ValueError("crop/geometry augmentation is disabled: connection preservation is unconfirmed")
    if prepared is None:
        prepared = [prepare_product(p, grid, roles, layers) for p in products]
    if len(prepared) != len(products):
        raise ValueError("prepared batch must match products")
    items = []
    for p, item in zip(products, prepared):
        images = item.images
        if augmentation == "intensity":
            if generator is None:
                raise ValueError("intensity ablation requires a generator")
            # Same scalar for all elements/channels; original endpoint/valid geometry is untouched.
            images = images * (0.9 + 0.1 * torch.rand((), generator=generator))
        items.append((images, item.role_ids, item.layer_ids, unit_feature(p.unit_size_um)))
    batch = collate(items)
    k = max(len(item.connections) for item in prepared)
    cf = torch.zeros(len(products), k, 9)
    cv = torch.zeros(len(products), k, dtype=torch.bool)
    for b, (p, item) in enumerate(zip(products, prepared)):
        by_id = {e.key: i for i, e in enumerate(p.elements)}
        for j, c in enumerate(item.connections):
            a, z = c.endpoints
            # Exactly the image resize/pad recipe, in normalized candidate-grid XY.
            h, w = p.frame.shape_yx
            nh, nw = max(1, round(h * grid / max(h, w))), max(1, round(w * grid / max(h, w)))
            left, top = (grid - nw) // 2, (grid - nh) // 2
            def position(endpoint):
                x = (endpoint.xy_um[0] - p.frame.origin_xy_um[0]) / p.frame.pixel_pitch_um
                y = (endpoint.xy_um[1] - p.frame.origin_xy_um[1]) / p.frame.pixel_pitch_um
                return ((x * nw / w + left) / grid, (y * nh / h + top) / grid)
            ax, ay = position(a)
            zx, zy = position(z)
            cf[b, j] = torch.tensor([by_id[a.element_id], by_id[z.element_id], ax, ay, zx, zy,
                                      c.kind == "VIA", c.observation == "observed", float(item.connection_masks[j].mean())])
            cv[b, j] = True
    extra = {"spatial_valid": torch.stack([item.valid for item in prepared]), "connection_features": cf,
             "connection_valid": cv, "connectivity_status": torch.tensor([{"unavailable": 0, "partial": 1, "available": 2}[item.connectivity_status] for item in prepared])}
    return batch, extra, prepared


def encode_products(model, products, grid, roles, layers, generator=None, *, augmentation="none", prepared=None):
    batch, extra, _ = structured_batch(products, grid, roles, layers, augmentation=augmentation, generator=generator, prepared=prepared)
    device = next(model.parameters()).device
    batch = tuple(t.to(device) for t in batch)
    extra = {key: t.to(device) for key, t in extra.items()}
    return model(*batch, **extra)


def model_config(grid, roles, layers, backbone="tiny", use_unit_size=False, model_kwargs=None):
    return {"grid": grid, "backbone": backbone, "use_unit_size": use_unit_size,
            "roles": roles, "layers": layers, "unknown_vocabulary": "reject", "model_kwargs": model_kwargs or {}}



def resolve_training_grid(products, grid=None, *, allow_grid_override=False):
    if type(allow_grid_override) is not bool:
        raise ValueError("allow_grid_override must be boolean")
    reports = [p.grid_check for p in products]
    candidates = {r["manifest_candidate_grid"] for r in reports if r is not None and "manifest_candidate_grid" in r}
    if len(candidates) > 1:
        raise ValueError("inconsistent manifest candidate_grid values")
    candidate = next(iter(candidates), None)
    if grid is None:
        if candidate is None or any(r is None or "manifest_candidate_grid" not in r for r in reports):
            raise ValueError("No grid supplied: load a manifest with candidate_grid or specify an explicit API grid")
        grid = candidate
    if type(grid) is not int or grid <= 0:
        raise ValueError("grid must be a positive integer")
    override = candidate is not None and grid != candidate
    if override and not allow_grid_override:
        raise ValueError(f"grid {grid} conflicts with manifest candidate_grid {candidate}; explicit --allow-grid-override is required")
    return grid, {"manifest_candidate_grid": candidate, "selected_grid": grid,
                  "source": "manifest_candidate" if candidate is not None and not override else "explicit_override" if override else "api_explicit",
                  "override": override, "override_requested": allow_grid_override}

def train(products, out_dir, epochs=100, grid=None, batch_size=8, lr=3e-4, backbone="tiny", weights_file=None,
          use_unit_size=False, relations=None, seed=0, *, split_manifest=None, unlabelled_negatives=False,
          augmentation="none", resume=None, device="cpu", temperature=0.1, model_kwargs=None,
          allow_grid_override=False, allow_grid_check_failure=False):
    if len(products) < 2:
        raise ValueError("제품이 둘 이상 있어야 학습할 수 있다")
    grid, grid_selection = resolve_training_grid(products, grid, allow_grid_override=allow_grid_override)
    if type(allow_grid_check_failure) is not bool:
        raise ValueError("allow_grid_check_failure must be boolean")
    if type(epochs) is not int or epochs <= 0 or type(grid) is not int or grid <= 0 or isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0 or isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("positive epochs/grid/lr/temperature and integer seed required")
    if augmentation not in {"none", "intensity"}:
        raise ValueError("only none/intensity augmentation is allowed")
    if type(unlabelled_negatives) is not bool:
        raise ValueError("unlabelled_negatives must be an explicit boolean")
    if torch.device(device).type not in {"cpu", "cuda"}:
        raise ValueError("training resume supports CPU/CUDA RNG only")
    if split_manifest is None:
        split_manifest = training_only_split(products)
    partitions = validate_split(products, split_manifest)
    training = partitions["train"]
    if len(training) < 2:
        raise ValueError("training split requires at least two products")
    all_similar, all_different = validate_relations(*(relations or (None, None)), product_ids=[p.product_id for p in products])
    train_ids = {p.product_id for p in training}
    similar = {pair for pair in all_similar if pair <= train_ids}
    different = {pair for pair in all_different if pair <= train_ids}
    sampler = RelationBatchSampler([p.product_id for p in training], batch_size, similar, different, unlabelled_negatives=unlabelled_negatives, lineage_ids=[p.lineage_id for p in training])
    roles, layers = build_vocab(training)
    prepared_training = [prepare_product(p, grid, roles, layers) for p in training]
    prepared_cache = {p.product_id: item for p, item in zip(training, prepared_training)}
    failures = [{"product_id": p.product_id, "reasons": item.grid_check["reasons"]} for p, item in zip(training, prepared_training)
                if any(e.get("thin_line_warning") or e.get("bridge_or_component_loss") for e in item.grid_check["elements"])]
    if failures and not allow_grid_check_failure:
        raise ValueError(f"grid check failed before training: {failures}; select a reviewed grid or explicitly pass --allow-grid-check-failure")
    grid_gate = {"policy": "strict_default", "bypass_requested": allow_grid_check_failure,
                 "status": "bypassed" if failures else "passed", "failures": failures}
    config = model_config(grid, roles, layers, backbone, use_unit_size, model_kwargs)
    validate_config(config)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = ProductEncoder(len(roles), len(layers), backbone, weights_file, use_unit_size=use_unit_size, **config["model_kwargs"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    records = {p.product_id: {"identity": input_identity(p), "lineage": p.lineage_id, "design": p.design_version} for p in products}
    input_hash = json_hash(records)
    contract = {"split": split_manifest, "similar": sorted(map(sorted, all_similar)), "different": sorted(map(sorted, all_different)),
                "augmentation": augmentation, "unlabelled_negatives": unlabelled_negatives, "batch_size": batch_size,
                "lr": lr, "temperature": temperature, "seed": seed, "grid_selection": grid_selection,
                "allow_grid_check_failure": allow_grid_check_failure}
    metadata = {"input_hash": input_hash, "code_hash": code_hash(), "initial_weights_hash": file_hash(weights_file) if weights_file else None,
                "seed": seed, "split": split_manifest, "input_records": records,
                "grid_selection": grid_selection, "grid_gate": grid_gate,
                "grid_checks": [item.grid_check for item in prepared_training], "contract": contract, "contract_hash": json_hash(contract),
                "training_statistics": {"product_count": len(training), "lineage_count": len({p.lineage_id for p in training}),
                                        "mean_unit_size_um": np.mean([p.unit_size_um for p in training], axis=0).tolist()},
                "relation_coverage": relation_coverage(sampler.relation), "epochs": [], "approval": approval_status()}
    start, history = 0, []
    if resume:
        saved = resume_checkpoint(resume, model, optimizer, gen, expected_config=config, expected_input_hash=input_hash,
                                  expected_contract_hash=metadata["contract_hash"], expected_code_hash=metadata["code_hash"])
        metadata, start, history = saved["metadata"], saved["epoch"], saved["history"]
        if start >= epochs:
            raise ValueError("resume epochs must extend saved epoch")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(json.dumps(split_manifest, indent=2) + "\n")
    model.train()
    for epoch in range(start, epochs):
        total, steps, effective, excluded = 0.0, 0, 0, 0
        seen_negative, seen_positive, seen_products = set(), set(), set()
        for indices in sampler.batches(gen):
            batch = [training[i] for i in indices]
            rel = sampler.relation[indices][:, indices].to(device)
            coverage = relation_coverage(rel)
            z1 = encode_products(model, batch, grid, roles, layers, gen, augmentation=augmentation, prepared=[prepared_cache[p.product_id] for p in batch])
            z2 = encode_products(model, batch, grid, roles, layers, gen, augmentation=augmentation, prepared=[prepared_cache[p.product_id] for p in batch])
            l1, n1 = masked_contrastive_loss(z1, z2, rel, temperature)
            l2, n2 = masked_contrastive_loss(z2, z1, rel.T, temperature)
            if n1 + n2 == 0:
                continue
            optimizer.zero_grad()
            loss = (l1 + l2) * 0.5
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
            effective += coverage["effective_pairs"]
            excluded += coverage["excluded_positive_only_rows"]
            seen_products.update(p.product_id for p in batch)
            seen_positive.update(tuple(sorted((batch[i].product_id, batch[j].product_id))) for i in range(len(batch)) for j in range(i + 1, len(batch)) if rel[i, j] == 1)
            seen_negative.update(tuple(sorted((batch[i].product_id, batch[j].product_id))) for i in range(len(batch)) for j in range(i + 1, len(batch)) if rel[i, j] == NEGATIVE)
        if not steps:
            raise ValueError("no usable contrastive rows")
        history.append(total / steps)
        metadata["epochs"].append({"epoch": epoch + 1, "steps": steps, "effective_pairs": effective,
                                    "excluded_positive_only_rows": excluded, "covered_negative_pairs": len(seen_negative),
                                    "required_negative_pairs": len(sampler.edges), "covered_positive_pairs": len(seen_positive),
                                    "unsampled_products": sorted(train_ids - seen_products)})
        save_checkpoint(out / "dl1.pt", model, config, optimizer=optimizer, generator=gen, epoch=epoch + 1, history=history, metadata=metadata)
        (out / "history.json").write_text(json.dumps(history) + "\n")
    return history


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    preliminary, _ = pre.parse_known_args(argv)
    ap = argparse.ArgumentParser(description="DL1 candidate training; input method approved, real grid/endpoint format and G6 pending")
    ap.add_argument("--config", help="JSON defaults; explicit CLI arguments override")
    ap.add_argument("--manifest")
    ap.add_argument("--split", help="explicit lineage-safe split JSON")
    ap.add_argument("--out")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--grid", type=int, default=None, help="defaults to manifest candidate_grid; never guessed")
    ap.add_argument("--allow-grid-override", action="store_true")
    ap.add_argument("--allow-grid-check-failure", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--backbone", default="tiny")
    ap.add_argument("--weights")
    ap.add_argument("--use-unit-size", action="store_true")
    ap.add_argument("--relations")
    ap.add_argument("--unlabelled-negatives", action="store_true")
    ap.add_argument("--augmentation", choices=("none", "intensity"), default="none")
    ap.add_argument("--resume")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    if preliminary.config:
        defaults = json.loads(Path(preliminary.config).read_text())
        allowed = {action.dest for action in ap._actions} - {"help", "config"}
        if not isinstance(defaults, dict) or defaults.keys() - allowed:
            raise ValueError("unknown training config fields")
        ap.set_defaults(**defaults)
    a = ap.parse_args(argv)
    if not a.manifest or not a.split or not a.out:
        ap.error("manifest, split, and out are required through config or CLI")
    products = load_manifest(a.manifest)
    grid, _ = resolve_training_grid(products, a.grid, allow_grid_override=a.allow_grid_override)
    history = train(products, a.out, a.epochs, grid, a.batch_size, a.lr, a.backbone, a.weights,
                    a.use_unit_size, load_relations(a.relations) if a.relations else None, a.seed,
                    split_manifest=json.loads(Path(a.split).read_text()), unlabelled_negatives=a.unlabelled_negatives,
                    augmentation=a.augmentation, resume=a.resume, device=a.device, temperature=a.temperature, allow_grid_override=a.allow_grid_override,
                    allow_grid_check_failure=a.allow_grid_check_failure)
    print(json.dumps({"last_loss": history[-1], "checkpoint": str(Path(a.out) / "dl1.pt"), "approval": approval_status()}))


if __name__ == "__main__":
    main()
