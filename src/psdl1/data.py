"""제품 하나 = 전극 x layer 이미지들의 묶음.

unit 전체를 같은 픽셀 격자로 맞춘다(비율 유지, 남는 쪽은 빈칸).
unit 크기가 달라도 배치와 연결이 그대로 겹치게 하려는 것이다.
절대 치수는 여기서 사라진다. 필요하면 unit 크기를 조건으로 따로 준다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CHANNELS = 3  # 이 전극·layer / CNT / 제품의 나머지 배선


@dataclass(frozen=True)
class Element:
    role: str   # source, drain, gate ... 제품 간 같은 역할
    layer: str
    mask: np.ndarray  # 2-D, 원본 격자


@dataclass(frozen=True)
class Product:
    product_id: str
    unit_size_um: tuple[float, float]  # (가로, 세로)
    elements: tuple[Element, ...]
    cnt: np.ndarray | None = None


def normalize_unit(mask: np.ndarray, grid: int) -> np.ndarray:
    """2-D mask를 grid x grid로. 면적 평균으로 줄이고 비율을 유지하며 가운데에 둔다."""
    m = torch.as_tensor(np.asarray(mask), dtype=torch.float32)
    if m.ndim != 2:
        raise ValueError("mask는 2-D")
    h, w = m.shape
    scale = grid / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = F.interpolate(m[None, None], size=(nh, nw), mode="area")[0, 0]
    out = torch.zeros(grid, grid)
    top, left = (grid - nh) // 2, (grid - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out.numpy()


def narrowest_run_px(mask: np.ndarray) -> int:
    """가로·세로로 이어진 True 구간 중 가장 짧은 길이. 줄였을 때 가는 선이 남는지 볼 때 쓴다."""
    m = np.asarray(mask).astype(bool)
    best = max(m.shape)
    for arr in (m, m.T):
        padded = np.pad(arr, ((0, 0), (1, 1))).astype(np.int8)
        edges = np.diff(padded, axis=1)
        for row in edges:
            starts, ends = np.flatnonzero(row == 1), np.flatnonzero(row == -1)
            if len(starts):
                best = min(best, int((ends - starts).min()))
    return best


def product_tensors(product: Product, grid: int, roles: dict[str, int], layers: dict[str, int]):
    """원소별 이미지 [N, 3, grid, grid]와 역할·layer 번호 [N]."""
    if not product.elements:
        raise ValueError(f"원소가 없는 제품: {product.product_id}")
    norm = [normalize_unit(e.mask, grid) for e in product.elements]
    total = np.clip(np.sum(norm, axis=0), 0.0, 1.0)
    cnt = normalize_unit(product.cnt, grid) if product.cnt is not None else np.zeros((grid, grid), dtype=np.float32)
    images = np.stack([np.stack([n, cnt, np.clip(total - n, 0.0, 1.0)]) for n in norm]).astype(np.float32)
    try:
        role_ids = [roles[e.role] for e in product.elements]
        layer_ids = [layers[e.layer] for e in product.elements]
    except KeyError as err:
        raise ValueError(f"어휘에 없는 역할 또는 layer: {err.args[0]!r} ({product.product_id})") from None
    return torch.from_numpy(images), torch.tensor(role_ids), torch.tensor(layer_ids)


def build_vocab(products: list[Product]) -> tuple[dict[str, int], dict[str, int]]:
    """역할·layer 이름 → 번호. 0은 빈칸(padding)용으로 남긴다."""
    roles = sorted({e.role for p in products for e in p.elements})
    layers = sorted({e.layer for p in products for e in p.elements})
    return {r: i + 1 for i, r in enumerate(roles)}, {l: i + 1 for i, l in enumerate(layers)}


def collate(items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """원소 수가 다른 제품들을 한 묶음으로. valid가 False인 자리는 빈칸."""
    n = max(x[0].shape[0] for x in items)
    b = len(items)
    c, g = items[0][0].shape[1], items[0][0].shape[2]
    images = torch.zeros(b, n, c, g, g)
    role_ids = torch.zeros(b, n, dtype=torch.long)
    layer_ids = torch.zeros(b, n, dtype=torch.long)
    valid = torch.zeros(b, n, dtype=torch.bool)
    unit = torch.stack([x[3] for x in items])
    for i, (img, r, l, _) in enumerate(items):
        k = img.shape[0]
        images[i, :k], role_ids[i, :k], layer_ids[i, :k], valid[i, :k] = img, r, l, True
    return images, role_ids, layer_ids, valid, unit


def unit_feature(unit_size_um: tuple[float, float]) -> torch.Tensor:
    """unit 크기 조건. 자릿수 차이를 줄이려고 log를 쓴다."""
    return torch.log(torch.tensor(unit_size_um, dtype=torch.float32))


def augment(images: torch.Tensor, generator: torch.Generator, scale=(0.6, 1.0), shift=0.08) -> torch.Tensor:
    """한 제품의 모든 원소에 같은 변형을 준다: 무작위로 일부를 잘라 원래 크기로 키우고 조금 옮긴다.

    크기가 달라도 같은 구조로 보도록 배우게 하려는 것이다. 원소마다 다르게 변형하면
    전극 사이의 배치 관계가 깨지므로 제품 단위로 한 번만 뽑는다.
    """
    n, c, g, _ = images.shape
    s = float(torch.empty(1).uniform_(scale[0], scale[1], generator=generator))
    tx, ty = (torch.empty(2).uniform_(-shift, shift, generator=generator) + (torch.rand(2, generator=generator) - 0.5) * (1 - s)).tolist()
    theta = torch.tensor([[s, 0.0, tx * 2], [0.0, s, ty * 2]]).expand(n, 2, 3)
    grid = F.affine_grid(theta, (n, c, g, g), align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def load_manifest(path: str | Path) -> list[Product]:
    """manifest.json을 읽는다. mask는 .npy(2-D). 경로는 manifest 폴더 기준 상대 경로.

    {"products": [{"product_id": "A", "unit_size_um": [60, 60], "cnt": "A/cnt.npy",
                   "elements": [{"role": "gate", "layer": "layer1", "mask": "A/gate_layer1.npy"}]}]}
    """
    path = Path(path)
    root = path.parent.resolve()

    def read(rel: str) -> np.ndarray:
        target = (root / rel).resolve()
        if root not in target.parents:
            raise ValueError(f"manifest 폴더 밖의 경로: {rel}")
        arr = np.load(target, allow_pickle=False)
        return (arr > 0).astype(np.float32)  # 0/255로 저장된 mask도 0/1로

    products = []
    for raw in json.loads(path.read_text(encoding="utf-8"))["products"]:
        elements = tuple(Element(e["role"], e["layer"], read(e["mask"])) for e in raw["elements"])
        shapes = {e.mask.shape for e in elements}
        if len(shapes) != 1:
            raise ValueError(f"한 제품의 mask 격자가 서로 다르다: {raw['product_id']}")
        products.append(Product(raw["product_id"], tuple(raw["unit_size_um"]), elements, read(raw["cnt"]) if raw.get("cnt") else None))
    if len({p.product_id for p in products}) != len(products):
        raise ValueError("product_id가 겹친다")
    return products
