"""FCMAE 방식(dense 근사) 자기지도 사전학습: 가린 패치를 복원하며 백본만 학습한다.

sparse convolution 없이, 가린 위치를 0으로 두고 같은 합성곱 백본에 통과시킨다. 라벨은 쓰지 않는다.
train 구획만 학습에 쓰고 validation은 복원 손실 확인에만 쓴다. test는 학습·통계·해시에 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .checkpoint import capture_rng, code_hash, json_hash, restore_rng, state_hash
from .data import CHANNELS, _iter_manifest_products, binary_mask, input_identity, normalize_unit
from .model import build_backbone
from .schema import approval_status, validate_product
from .split import validate_split

PRETRAIN_VERSION = "dl1_pretrain_v1"
IN_CHANS = CHANNELS + 1  # ProductEncoder가 백본에 넣는 채널 수(이미지 3 + 관측 범위 1)


def element_images(product, grid):
    """prepare_product와 같은 이미지 구성(grid 검사·어휘 없이). 요소 [N,3,G,G]와 관측 범위 [G,G]."""
    _, valid = validate_product(product)
    if any(e.mask is None for e in product.elements):
        raise ValueError("missing element observation cannot be represented as zero")
    norm = np.stack([normalize_unit(binary_mask(e.mask) * valid, grid) for e in product.elements])
    sources = [binary_mask(c.mask) for c in product.connections if c.mask is not None]
    if product.cnt is not None:
        sources.append(binary_mask(product.cnt))
    cnt = normalize_unit(np.maximum.reduce(sources) * valid, grid) if sources else np.zeros((grid, grid), dtype=np.float32)
    rest = np.clip(norm.sum(0)[None] - norm, 0, 1) if len(norm) > 1 else np.zeros_like(norm)
    images = np.stack([norm, np.broadcast_to(cnt, norm.shape), rest], axis=1).astype(np.float32)
    return images, normalize_unit(valid, grid)


def patch_mask(batch, grid, patch, ratio, generator=None):
    """[B,1,G,G], 1이 가려진 픽셀. 패치마다 같은 비율로 가리고 최소 한 패치는 남긴다."""
    n = grid // patch
    if not 0 < ratio < 1 or n * n < 2:
        raise ValueError("mask ratio must be in (0,1) and the grid needs at least two patches")
    k = min(max(1, round(ratio * n * n)), n * n - 1)
    chosen = torch.rand(batch, n * n, generator=generator).argsort(1)[:, :k]
    mask = torch.zeros(batch, n * n).scatter_(1, chosen, 1.0).view(batch, 1, n, n)
    return mask.repeat_interleave(patch, 2).repeat_interleave(patch, 3)


def feature_map(backbone, x):
    return backbone.body(x) if hasattr(backbone, "body") else backbone.forward_features(x)


class MaskedPretrainer(nn.Module):
    """백본 + 버릴 디코더. 특징 해상도에서 패치 픽셀을 한 번에 예측한다(pixel shuffle)."""

    def __init__(self, backbone, decoder_dim=128):
        super().__init__()
        self.backbone = backbone
        was_training = backbone.training
        backbone.eval()
        with torch.no_grad():
            probe = feature_map(backbone, torch.zeros(1, IN_CHANS, 64, 64))
        backbone.train(was_training)
        self.stride = 64 // probe.shape[-1]
        if probe.ndim != 4 or probe.shape[-1] * self.stride != 64:
            raise ValueError("backbone must return a [B,C,h,w] feature map with a power-of-two stride")
        d = decoder_dim
        self.decoder = nn.Sequential(nn.Conv2d(probe.shape[1], d, 3, padding=1), nn.GroupNorm(1, d), nn.GELU(),
                                     nn.Conv2d(d, d, 3, padding=1), nn.GroupNorm(1, d), nn.GELU(),
                                     nn.Conv2d(d, CHANNELS * self.stride ** 2, 1))

    def forward(self, images, valid, mask):
        """images [B,3,G,G], valid [B,1,G,G], mask [B,1,G,G] → 복원 logit [B,3,G,G]."""
        features = feature_map(self.backbone, torch.cat([images * (1 - mask), valid], dim=1))
        return F.pixel_shuffle(self.decoder(features), self.stride)


def reconstruction_loss(logits, target, mask, valid, fg_weight):
    """가려진 관측 픽셀에서만 계산한 가중 BCE. 전경 픽셀에 fg_weight를 곱해 빈 칸 예측으로 쏠림을 줄인다."""
    weight = mask * valid * (1 + (fg_weight - 1) * (target > 0).float())
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (bce * weight).sum() / weight.sum().clamp_min(1.0)


@torch.no_grad()
def reconstruction_stats(logits, target, mask, valid, fg_weight):
    """손실, 상수 예측 기준선 손실, 가려진 영역의 전경 교집합/합집합 픽셀 수."""
    loss = reconstruction_loss(logits, target, mask, valid, fg_weight)
    seen = (mask * valid).expand_as(target)
    p = ((target * seen).sum() / seen.sum().clamp_min(1.0)).clamp(1e-6, 1 - 1e-6)
    baseline = reconstruction_loss(torch.full_like(target, 0.0) + torch.log(p / (1 - p)), target, mask, valid, fg_weight)
    pred, true = (logits > 0) & (seen > 0), (target > 0.5) & (seen > 0)
    return float(loss), float(baseline), int((pred & true).sum()), int((pred | true).sum())


class CacheReader:
    """요소 이미지 캐시를 mmap으로 읽는다. 배치마다 필요한 요소만 디스크에서 가져온다."""

    def __init__(self, cache_dir, entries):
        self.arrays = [np.load(Path(cache_dir) / e["file"], mmap_mode="r") for e in entries]
        self.valids = [np.load(Path(cache_dir) / e["valid_file"]) for e in entries]
        self.pairs = [(i, j) for i, a in enumerate(self.arrays) for j in range(a.shape[0])]

    def __len__(self):
        return len(self.pairs)

    def batches(self, batch_size, generator=None):
        order = torch.randperm(len(self.pairs), generator=generator).tolist() if generator is not None else list(range(len(self.pairs)))
        for start in range(0, len(order), batch_size):
            chosen = [self.pairs[k] for k in order[start:start + batch_size]]
            images = torch.from_numpy(np.stack([self.arrays[i][j] for i, j in chosen]).astype(np.float32))
            valid = torch.from_numpy(np.stack([self.valids[i] for i, _ in chosen]).astype(np.float32))[:, None]
            yield images, valid


def _atomic_save(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save_npy(path, array):
    def write(temporary):
        with open(temporary, "wb") as stream:
            np.save(stream, array, allow_pickle=False)
    _atomic_save(path, write)


def scan_and_cache(manifest, split, cache_dir, grid):
    """manifest를 한 번 훑어 계보를 검증하고 train/validation 요소 이미지를 캐시한다.

    test 제품은 manifest 검증을 위해 읽히지만 캐시·해시·학습에는 들어가지 않는다."""
    wanted = {pid for key in ("train", "validation") if isinstance(split.get(key), list) for pid in split[key]}
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.json"
    old = json.loads(index_path.read_text()) if index_path.is_file() else {}
    old = old.get("products", {}) if old.get("grid") == grid else {}
    stubs, entries = [], {}
    for product in _iter_manifest_products(manifest, grid=grid):
        stubs.append(SimpleNamespace(product_id=product.product_id, lineage_id=product.lineage_id))
        if product.product_id not in wanted:
            continue
        identity = input_identity(product)
        name = hashlib.sha256(product.product_id.encode()).hexdigest()[:20]
        entry = {"file": f"{name}.npy", "valid_file": f"{name}.valid.npy", "identity": identity}
        reusable = old.get(product.product_id, {}).get("identity") == identity and all((cache_dir / entry[k]).is_file() for k in ("file", "valid_file"))
        if not reusable:
            images, valid = element_images(product, grid)
            _save_npy(cache_dir / entry["file"], images.astype(np.float16))
            _save_npy(cache_dir / entry["valid_file"], valid.astype(np.float16))
        entries[product.product_id] = entry
    partitions = validate_split(stubs, split)
    _atomic_save(index_path, lambda t: Path(t).write_text(json.dumps({"grid": grid, "products": entries}, indent=1)))
    return {key: [entries[p.product_id] for p in partitions[key]] for key in ("train", "validation")}, entries


@torch.no_grad()
def evaluate_reconstruction(model, reader, config, device, seed):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    total = baseline = 0.0
    count = inter = union = 0
    for images, valid in reader.batches(config["batch_size"]):
        mask = patch_mask(len(images), config["grid"], config["patch"], config["mask_ratio"], gen).to(device)
        images, valid = images.to(device), valid.to(device)
        loss, base, i, u = reconstruction_stats(model(images, valid, mask), images, mask, valid, config["fg_weight"])
        total, baseline, count, inter, union = total + loss * len(images), baseline + base * len(images), count + len(images), inter + i, union + u
    return {"loss": total / count, "baseline_loss": baseline / count, "masked_foreground_iou": inter / union if union else None, "elements": count}


def pretrain(manifest, split, out_dir, *, epochs=100, batch_size=4, lr=3e-4, backbone="tiny", mask_ratio=0.6, patch=None,
             fg_weight=5.0, decoder_dim=128, seed=0, device="cpu", grid=None, allow_grid_override=False, resume=None):
    if any(type(v) is not int or v <= 0 for v in (epochs, batch_size, decoder_dim)) or type(seed) is not int or not 0 <= seed < 2**32 or lr <= 0 or fg_weight < 1:
        raise ValueError("positive epochs/batch_size/decoder_dim/lr, fg_weight >= 1 and integer seed required")
    if torch.device(device).type not in {"cpu", "cuda"}:
        raise ValueError("pretraining supports CPU/CUDA only")
    manifest = Path(manifest)
    candidate = json.loads(manifest.read_text(encoding="utf-8"))["candidate_grid"]
    grid = candidate if grid is None else grid
    if type(grid) is not int or grid <= 0 or (grid != candidate and not allow_grid_override):
        raise ValueError(f"grid {grid} conflicts with manifest candidate_grid {candidate}; explicit --allow-grid-override is required")
    out = Path(out_dir)
    cache_dir = out / "cache"
    partitions, entries = scan_and_cache(manifest, split, cache_dir, grid)
    if not partitions["train"]:
        raise ValueError("training split is empty")
    torch.manual_seed(seed)
    net = MaskedPretrainer(build_backbone(backbone, IN_CHANS), decoder_dim)
    patch = net.stride if patch is None else patch
    if patch % net.stride or grid % patch:
        raise ValueError(f"patch {patch} must be a multiple of the backbone stride {net.stride} and divide grid {grid}")
    net.to(device)
    config = {"backbone": backbone, "grid": grid, "patch": patch, "stride": net.stride, "mask_ratio": mask_ratio, "fg_weight": fg_weight,
              "decoder_dim": decoder_dim, "batch_size": batch_size, "lr": lr, "seed": seed}
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
    gen = torch.Generator().manual_seed(seed)
    input_hash = json_hash({pid: e["identity"] for pid, e in sorted(entries.items())})
    metadata = {"input_hash": input_hash, "code_hash": code_hash(), "config": config, "config_hash": json_hash(config), "split": split,
                "approval": approval_status(), "epochs": []}
    start, history = 0, []
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        meta = saved.get("metadata", {})
        if saved.get("format_version") != PRETRAIN_VERSION or meta.get("config_hash") != metadata["config_hash"] or meta.get("input_hash") != input_hash or meta.get("code_hash") != metadata["code_hash"] or meta.get("split") != split:
            raise ValueError("resume config/input/split/code differs from the saved pretraining")
        net.backbone.load_state_dict(saved["backbone"], strict=True)
        net.decoder.load_state_dict(saved["decoder"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"], gen)
        metadata, start, history = meta, saved["epoch"], saved["history"]
        if start >= epochs:
            raise ValueError("resume epochs must extend saved epoch")
    train_reader = CacheReader(cache_dir, partitions["train"])
    valid_reader = CacheReader(cache_dir, partitions["validation"]) if partitions["validation"] else None
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(start, epochs):
        net.train()
        total, steps = 0.0, 0
        for images, valid in train_reader.batches(batch_size, gen):
            mask = patch_mask(len(images), grid, patch, mask_ratio, gen).to(device)
            images, valid = images.to(device), valid.to(device)
            loss = reconstruction_loss(net(images, valid, mask), images, mask, valid, fg_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, steps = total + float(loss.detach()), steps + 1
        history.append(total / steps)
        record = {"epoch": epoch + 1, "steps": steps, "train_loss": history[-1],
                  "validation": None if valid_reader is None else evaluate_reconstruction(net, valid_reader, config, device, seed + epoch + 1)}
        metadata["epochs"].append(record)
        state = {key: value.detach().cpu() for key, value in net.backbone.state_dict().items()}
        metadata["backbone_weights_hash"] = state_hash(state)
        saved = {"format_version": PRETRAIN_VERSION, "backbone": state, "decoder": {k: v.detach().cpu() for k, v in net.decoder.state_dict().items()},
                 "optimizer": optimizer.state_dict(), "rng": capture_rng(gen), "epoch": epoch + 1, "history": history, "metadata": metadata}
        _atomic_save(out / "pretrain.pt", lambda t: torch.save(saved, t))
        _atomic_save(out / "backbone.pt", lambda t: torch.save(state, t))  # psdl1.train --weights 에 그대로 쓴다
        (out / "history.json").write_text(json.dumps(metadata["epochs"], indent=1) + "\n")
    try:
        from safetensors.torch import save_file
        _atomic_save(out / "backbone.safetensors", lambda t: save_file({k: v.contiguous() for k, v in state.items()}, t))
    except ImportError:
        pass
    (out / "pretrain.json").write_text(json.dumps({k: metadata[k] for k in ("input_hash", "code_hash", "config", "config_hash", "split", "backbone_weights_hash", "approval")}, indent=1) + "\n")
    return history


def main(argv=None):
    ap = argparse.ArgumentParser(description="DL1 FCMAE-style (dense approximation) backbone pretraining; produces backbone-only weights for psdl1.train --weights")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True, help="explicit lineage-safe split JSON; train/validation only are read for learning")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone", default="tiny")
    ap.add_argument("--mask-ratio", type=float, default=0.6)
    ap.add_argument("--patch", type=int, default=None, help="mask patch size in pixels; default = backbone stride")
    ap.add_argument("--fg-weight", type=float, default=5.0)
    ap.add_argument("--decoder-dim", type=int, default=128)
    ap.add_argument("--grid", type=int, default=None, help="defaults to manifest candidate_grid")
    ap.add_argument("--allow-grid-override", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume")
    a = ap.parse_args(argv)
    history = pretrain(a.manifest, json.loads(Path(a.split).read_text()), a.out, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                       backbone=a.backbone, mask_ratio=a.mask_ratio, patch=a.patch, fg_weight=a.fg_weight, decoder_dim=a.decoder_dim,
                       seed=a.seed, device=a.device, grid=a.grid, allow_grid_override=a.allow_grid_override, resume=a.resume)
    print(json.dumps({"last_loss": history[-1], "backbone": str(Path(a.out) / "backbone.pt"), "approval": approval_status()}))


if __name__ == "__main__":
    main()
