"""Candidate supervised training; labels and held-out acceptance remain G6 pending."""
from __future__ import annotations

import argparse
import csv
import time
import json
import math
from pathlib import Path
import random
from collections import Counter

import numpy as np
import torch

from .checkpoint import (check_parent_compatible, code_hash, extend_vocab, file_hash, init_from_checkpoint, json_hash, read_checkpoint,
                         resume_checkpoint, save_checkpoint, validate_config)
from .data import Product, build_vocab, collate, frame_transform, grid_check, input_identity, load_manifest, prepare_product, unit_feature
from .losses import NEGATIVE, masked_contrastive_loss, relation_coverage, relation_matrix, validate_relations
from .model import ProductEncoder
from .split import training_only_split, validate_split
from .progress import Progress, stderr_sink
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
    def __init__(self, product_ids, batch_size, similar=None, different=None, *, unlabelled_negatives=False, lineage_ids=None, repeat=None):
        if type(batch_size) is not int or batch_size < 2:
            raise ValueError("batch_size must be at least two")
        self.ids = list(product_ids)
        self.batch_size = batch_size
        self.similar, self.different = validate_relations(similar, different, product_ids=self.ids)
        self.relation = relation_matrix(self.ids, self.similar, self.different, unlabelled_negatives=unlabelled_negatives, lineage_ids=lineage_ids)
        self.edges = [(i, j) for i in range(len(self.ids)) for j in range(i + 1, len(self.ids)) if self.relation[i, j] == NEGATIVE]
        if not self.edges:
            raise ValueError("구분할 쌍이 없다: provide different labels or explicitly opt into unlabelled negatives")
        self.repeat = repeat

    @property
    def repeat(self):
        return self._repeat

    @repeat.setter
    def repeat(self, repeat):
        """edge -> epoch 당 최소 포함 횟수(기본 1). edges 집합은 바뀌지 않는다."""
        repeat = dict(repeat or {})
        known = set(self.edges)
        if any(e not in known or type(k) is not int or k < 1 for e, k in repeat.items()):
            raise ValueError("repeat requires positive integers for existing negative edges")
        self._repeat = repeat

    def batches(self, generator):
        schedule = [e for e in self.edges for _ in range(self._repeat.get(e, 1))]
        order = torch.randperm(len(schedule), generator=generator).tolist()
        remaining = Counter(schedule[i] for i in order)  # 다중집합: 0 이 된 edge 는 지운다
        result = []
        for i in order:
            edge = schedule[i]
            if edge not in remaining:
                continue
            batch = list(edge)
            # Fill with source positives first, then other products; never duplicate a product.
            order = torch.randperm(len(self.ids), generator=generator).tolist()
            positives = [i for i in order if i not in batch and any(frozenset((self.ids[i], self.ids[j])) in self.similar for j in batch)]
            batch.extend((positives + [i for i in order if i not in batch and i not in positives])[:self.batch_size - len(batch)])
            members = set(batch)
            for e in [e for e in remaining if e[0] in members and e[1] in members]:
                remaining[e] -= 1
                if not remaining[e]:
                    del remaining[e]
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
        transform = frame_transform(p.frame.shape_yx, grid, p.frame.pixel_pitch_um) if item.connections else None
        for j, c in enumerate(item.connections):
            a, z = c.endpoints
            # Exactly the image resize/pad recipe, in normalized candidate-grid XY.
            def position(endpoint):
                x = (endpoint.xy_um[0] - p.frame.origin_xy_um[0]) / p.frame.pixel_pitch_um
                y = (endpoint.xy_um[1] - p.frame.origin_xy_um[1]) / p.frame.pixel_pitch_um
                return transform.source_edge_to_normalized(x, y)
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

def parent_delta_pairs(sampler, parent_metadata):
    """현재 negative edge 중 부모가 학습하지 않은 쌍 → 정렬된 제품 ID 쌍 목록. 부모 metadata 로 부모 negative 를 재구성한다."""
    try:
        contract, train_ids = parent_metadata["contract"], parent_metadata["split"]["train"]
        lineage = [parent_metadata["input_records"][pid]["lineage"] for pid in train_ids]
        keep = set(train_ids)
        similar = {frozenset(x) for x in contract["similar"] if set(x) <= keep}
        different = {frozenset(x) for x in contract["different"] if set(x) <= keep}
        relation = relation_matrix(train_ids, similar, different, unlabelled_negatives=contract["unlabelled_negatives"], lineage_ids=lineage)
    except (KeyError, TypeError) as error:
        raise ValueError(f"init-from parent metadata incomplete: {error!r}") from error
    old = {frozenset((train_ids[i], train_ids[j])) for i, j in (relation == NEGATIVE).nonzero().tolist() if i < j}
    return sorted(sorted(pair) for pair in ({sampler.ids[i], sampler.ids[j]} for i, j in sampler.edges) if frozenset(pair) not in old)


def delta_edges(sampler, pairs):
    """저장된 delta 쌍(제품 ID) → 현재 sampler 인덱스 edge 집합. 형식·중복·미등록 ID·negative 아닌 쌍은 거부한다."""
    index, edges, result = {pid: i for i, pid in enumerate(sampler.ids)}, set(sampler.edges), set()
    if not isinstance(pairs, list):
        raise ValueError("delta_negative_pairs must be a list")
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(not isinstance(v, str) for v in pair):
            raise ValueError("delta_negative_pairs requires product ID pairs")
        if any(v not in index for v in pair):
            raise ValueError(f"delta_negative_pairs references unknown product: {list(pair)}")
        edge = tuple(sorted((index[pair[0]], index[pair[1]])))
        if edge[0] == edge[1] or edge not in edges:
            raise ValueError(f"delta_negative_pairs is not a current negative edge: {list(pair)}")
        if edge in result:
            raise ValueError(f"delta_negative_pairs has duplicates: {list(pair)}")
        result.add(edge)
    return result


PARENT_CONTRACT_KEYS = ("weights_hash", "contract_hash", "input_hash", "fingerprint")


def train(products, out_dir, epochs=100, grid=None, batch_size=8, lr=3e-4, backbone="tiny", weights_file=None,
          use_unit_size=False, relations=None, seed=0, *, split_manifest=None, unlabelled_negatives=False,
          augmentation="none", resume=None, device="cpu", temperature=0.1, model_kwargs=None,
          allow_grid_override=False, allow_grid_check_failure=False, progress_sink=None, progress_interval=10.0,
          amp=False, grad_checkpoint=False, init_from=None, new_edge_repeat=1, round_index=None):
    progress = Progress(progress_sink, progress_interval)
    if init_from and resume:
        raise ValueError("--init-from and --resume are exclusive")
    if init_from and weights_file:
        raise ValueError("--init-from and --weights are exclusive")
    if type(new_edge_repeat) is not int or new_edge_repeat < 1:
        raise ValueError("new_edge_repeat must be a positive integer")
    if round_index is not None and (type(round_index) is not int or round_index < 1):
        raise ValueError("round must be a positive integer")
    if new_edge_repeat != 1 and not init_from and not resume:
        raise ValueError("new-edge-repeat needs a parent")
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
    if type(amp) is not bool or type(grad_checkpoint) is not bool:
        raise ValueError("amp and grad_checkpoint must be boolean")
    if amp and torch.device(device).type != "cuda":
        raise ValueError("amp requires a CUDA device")
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
    parent = saved_resume = None  # parent: --init-from 체크포인트, saved_resume: --resume 체크포인트
    if init_from:
        parent = read_checkpoint(init_from)
    elif resume:
        saved_resume = read_checkpoint(resume)
        if not saved_resume["metadata"].get("parent") and new_edge_repeat != 1:
            raise ValueError("new-edge-repeat needs a parent")
        if round_index is not None and round_index != saved_resume["metadata"].get("round"):
            raise ValueError("resume round differs")
    reference = parent["config"] if parent else saved_resume["config"] if saved_resume and saved_resume["metadata"].get("parent") else None
    roles, layers = build_vocab(training)
    if reference:  # warm start 어휘: 부모 번호 유지 + 새 이름 정렬 추가
        roles, layers = extend_vocab(reference["roles"], roles), extend_vocab(reference["layers"], layers)
    prepared_training, prepare_started = [], time.monotonic()
    progress.emit("prepare", state="started", products=len(training), grid=grid)
    for index, p in enumerate(training, 1):
        prepared_training.append(prepare_product(p, grid, roles, layers))
        progress.emit("prepare", state="product_prepared", index=index, products=len(training), product_id=p.product_id, elapsed_s=round(time.monotonic() - prepare_started, 1))
    prepared_cache = {p.product_id: item for p, item in zip(training, prepared_training)}
    failures = [{"product_id": p.product_id, "reasons": item.grid_check["reasons"]} for p, item in zip(training, prepared_training)
                if any(e.get("thin_line_warning") or e.get("bridge_or_component_loss") for e in item.grid_check["elements"])]
    if failures and not allow_grid_check_failure:
        raise ValueError(f"grid check failed before training: {failures}; select a reviewed grid or explicitly pass --allow-grid-check-failure")
    grid_gate = {"policy": "strict_default", "bypass_requested": allow_grid_check_failure,
                 "status": "bypassed" if failures else "passed", "failures": failures}
    config = model_config(grid, roles, layers, backbone, use_unit_size, model_kwargs)
    validate_config(config)
    if parent:
        check_parent_compatible(parent["config"], config)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = ProductEncoder(len(roles), len(layers), backbone, weights_file, use_unit_size=use_unit_size, **config["model_kwargs"]).to(device)
    model.grad_checkpoint = grad_checkpoint
    warm = parent_record = delta_pairs = None
    if parent:  # optimizer 를 만들기 전에 가중치만 싣는다. optimizer·scaler·RNG·epoch 는 새로 시작한다.
        warm = init_from_checkpoint(init_from, model, expected_config=config, saved=parent)["warm_start"]
        names = build_vocab(training)
        for field, used in zip(("roles", "layers"), names):
            warm[field]["unused_parent_names"] = sorted(parent["config"][field].keys() - used.keys())
        pm = parent["metadata"]
        parent_record = {"path": str(init_from), **{k: parent.get(k) if k in ("weights_hash", "fingerprint") else pm.get(k) for k in PARENT_CONTRACT_KEYS},
                         "code_hash": pm.get("code_hash"), "code_hash_matches": pm.get("code_hash") == code_hash(),
                         "epoch": parent["epoch"], "round": pm.get("round")}
        delta_pairs = parent_delta_pairs(sampler, pm)
    elif saved_resume and saved_resume["metadata"].get("parent"):
        parent_record = saved_resume["metadata"]["parent"]
        delta_pairs = saved_resume["metadata"].get("delta_negative_pairs")
        if not isinstance(parent_record, dict) or any(k not in parent_record for k in PARENT_CONTRACT_KEYS):
            raise ValueError("resume parent record incomplete")
    delta = None
    if delta_pairs is not None:
        delta = delta_edges(sampler, delta_pairs)
        delta_pairs = sorted(sorted(sampler_pair) for sampler_pair in delta_pairs)
        sampler.repeat = {e: new_edge_repeat for e in delta}
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)  # 재개 시 스케일은 기본값에서 다시 적응한다
    records = {p.product_id: {"identity": input_identity(p), "lineage": p.lineage_id, "design": p.design_version} for p in products}
    input_hash = json_hash(records)
    contract = {"split": split_manifest, "similar": sorted(map(sorted, all_similar)), "different": sorted(map(sorted, all_different)),
                "augmentation": augmentation, "unlabelled_negatives": unlabelled_negatives, "batch_size": batch_size,
                "lr": lr, "temperature": temperature, "seed": seed, "grid_selection": grid_selection,
                "allow_grid_check_failure": allow_grid_check_failure, "new_edge_repeat": new_edge_repeat,
                "init_from": None if parent_record is None else {k: parent_record[k] for k in PARENT_CONTRACT_KEYS},
                "delta_negative_pairs_hash": None if delta_pairs is None else json_hash(delta_pairs)}
    metadata = {"input_hash": input_hash, "code_hash": code_hash(), "initial_weights_hash": file_hash(weights_file) if weights_file else None,
                "round": round_index, "parent": parent_record, "warm_start": warm, "delta_negative_pairs": delta_pairs,
                "seed": seed, "split": split_manifest, "input_records": records,
                "grid_selection": grid_selection, "grid_gate": grid_gate,
                "grid_checks": [item.grid_check for item in prepared_training], "contract": contract, "contract_hash": json_hash(contract),
                "training_statistics": {"product_count": len(training), "lineage_count": len({p.lineage_id for p in training}),
                                        "mean_unit_size_um": np.mean([p.unit_size_um for p in training], axis=0).tolist()},
                "relation_coverage": relation_coverage(sampler.relation), "epochs": [], "approval": approval_status()}
    start, history = 0, []
    if resume:
        # 저장된 metadata.parent·delta 쌍으로 만든 contract 가 저장 contract_hash 와 같아야 한다(부모 파일은 다시 읽지 않는다).
        saved = resume_checkpoint(resume, model, optimizer, gen, expected_config=config, expected_input_hash=input_hash,
                                  expected_contract_hash=metadata["contract_hash"], expected_code_hash=metadata["code_hash"])
        metadata, start, history = saved["metadata"], saved["epoch"], saved["history"]
        if start >= epochs:
            raise ValueError("resume epochs must extend saved epoch")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(json.dumps(split_manifest, indent=2) + "\n")
    model.train()
    progress.emit("train", state="started", device=str(device), cuda_device=torch.cuda.get_device_name(0) if torch.device(device).type == "cuda" else None,
                  grid=grid, backbone=backbone, batch_size=batch_size, train_products=len(training), epochs=epochs, start_epoch=start)
    for epoch in range(start, epochs):
        total, steps, effective, excluded = 0.0, 0, 0, 0
        seen_negative, seen_positive, seen_products = set(), set(), set()
        delta_count = Counter()
        epoch_started = time.monotonic()
        epoch_batches = sampler.batches(gen)
        for position, indices in enumerate(epoch_batches, 1):
            batch = [training[i] for i in indices]
            rel = sampler.relation[indices][:, indices].to(device)
            coverage = relation_coverage(rel)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                z1 = encode_products(model, batch, grid, roles, layers, gen, augmentation=augmentation, prepared=[prepared_cache[p.product_id] for p in batch])
                z2 = encode_products(model, batch, grid, roles, layers, gen, augmentation=augmentation, prepared=[prepared_cache[p.product_id] for p in batch])
            z1, z2 = z1.float(), z2.float()
            l1, n1 = masked_contrastive_loss(z1, z2, rel, temperature)
            l2, n2 = masked_contrastive_loss(z2, z1, rel.T, temperature)
            if n1 + n2 == 0:
                continue
            optimizer.zero_grad()
            loss = (l1 + l2) * 0.5
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            steps += 1
            effective += coverage["effective_pairs"]
            excluded += coverage["excluded_positive_only_rows"]
            seen_products.update(p.product_id for p in batch)
            if position == len(epoch_batches) or progress.due():
                elapsed = time.monotonic() - epoch_started
                progress.emit("train", state="running", epoch=epoch + 1, epochs=epochs, step=position, steps=len(epoch_batches), loss=round(total / steps, 6),
                              elapsed_s=round(elapsed, 1), epoch_eta_s=round(elapsed / position * (len(epoch_batches) - position), 1))
            seen_positive.update(tuple(sorted((batch[i].product_id, batch[j].product_id))) for i in range(len(batch)) for j in range(i + 1, len(batch)) if rel[i, j] == 1)
            seen_negative.update(tuple(sorted((batch[i].product_id, batch[j].product_id))) for i in range(len(batch)) for j in range(i + 1, len(batch)) if rel[i, j] == NEGATIVE)
            if delta:
                delta_count.update(e for e in (tuple(sorted((a, b))) for a in indices for b in indices if a < b) if e in delta)
        if not steps:
            raise ValueError("no usable contrastive rows")
        history.append(total / steps)
        progress.emit("train", state="epoch_completed", epoch=epoch + 1, epochs=epochs, loss=round(history[-1], 6), steps=steps, elapsed_s=round(time.monotonic() - epoch_started, 1))
        metadata["epochs"].append({"epoch": epoch + 1, "steps": steps, "effective_pairs": effective,
                                    "excluded_positive_only_rows": excluded, "covered_negative_pairs": len(seen_negative),
                                    "required_negative_pairs": len(sampler.edges), "covered_positive_pairs": len(seen_positive),
                                    "unsampled_products": sorted(train_ids - seen_products)}
                                   | ({} if delta is None else {"delta_negative_pairs": len(delta), "new_edge_repeat": new_edge_repeat,
                                                                "delta_coverage_min": min(delta_count[e] for e in delta) if delta else None}))
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
    ap.add_argument("--init-from", help="warm start: load a parent dl1.pt's weights and extend its vocabulary (exclusive with --resume/--weights)")
    ap.add_argument("--new-edge-repeat", type=int, default=1, help="schedule negative edges the parent never trained this many times per epoch")
    ap.add_argument("--round", type=int, default=None, help="round number recorded in metadata")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="CUDA fp16 autocast + GradScaler to cut activation memory")
    ap.add_argument("--grad-checkpoint", action="store_true", help="recompute backbone activations in backward to cut memory (slower)")
    ap.add_argument("--no-progress", action="store_true", help="suppress JSON progress lines on stderr")
    ap.add_argument("--progress-interval", type=float, default=10.0, help="seconds between in-epoch progress lines")
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
                    allow_grid_check_failure=a.allow_grid_check_failure,
                    progress_sink=None if a.no_progress else stderr_sink, progress_interval=a.progress_interval,
                    amp=a.amp, grad_checkpoint=a.grad_checkpoint, init_from=a.init_from, new_edge_repeat=a.new_edge_repeat, round_index=a.round)
    print(json.dumps({"last_loss": history[-1], "checkpoint": str(Path(a.out) / "dl1.pt"), "approval": approval_status()}))


if __name__ == "__main__":
    main()
