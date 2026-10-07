"""원소 인코더 + 조건 결합 + Set Transformer → 제품 특징 하나."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import CHANNELS


class TinyBackbone(nn.Module):
    """timm 없이 도는 작은 합성곱 인코더. 테스트와 가벼운 시험용."""

    def __init__(self, in_chans: int = CHANNELS, width: int = 32):
        super().__init__()
        dims = [width, width * 2, width * 4]
        layers, prev = [], in_chans
        for d in dims:
            layers += [nn.Conv2d(prev, d, 3, stride=2, padding=1), nn.GroupNorm(1, d), nn.GELU(), nn.Conv2d(d, d, 3, padding=1), nn.GroupNorm(1, d), nn.GELU()]
            prev = d
        self.body = nn.Sequential(*layers)
        self.num_features = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x).mean(dim=(2, 3))


def build_backbone(name: str = "tiny", in_chans: int = CHANNELS, weights_file: str | None = None) -> nn.Module:
    """name이 'tiny'면 내장 인코더, 아니면 timm 모델 이름(예: convnextv2_tiny).

    weights_file을 주면 그 파일에서만 가중치를 읽는다. 네트워크로 받지 않는다.
    """
    if name == "tiny":
        if weights_file:
            raise ValueError("tiny 인코더에는 사전학습 가중치가 없다")
        return TinyBackbone(in_chans)
    try:
        import timm
    except ImportError as err:
        raise ImportError("timm 모델을 쓰려면 timm을 설치해야 한다 (pip install timm)") from err
    if weights_file:
        return timm.create_model(name, pretrained=True, pretrained_cfg_overlay={"file": weights_file}, in_chans=in_chans, num_classes=0)
    return timm.create_model(name, pretrained=False, in_chans=in_chans, num_classes=0)


class SetBlock(nn.Module):
    """원소들끼리 서로 보는 자기 주의 블록."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        h = self.n1(x)
        x = x + self.attn(h, h, h, key_padding_mask=pad, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class ProductEncoder(nn.Module):
    """images [B,N,C,H,W], role_ids [B,N], layer_ids [B,N], valid [B,N], unit [B,2] → [B,D] (길이 1)."""

    def __init__(self, n_roles: int, n_layers: int, backbone: str = "tiny", weights_file: str | None = None,
                 dim: int = 128, out_dim: int = 128, heads: int = 4, blocks: int = 2, use_unit_size: bool = False):
        super().__init__()
        self.backbone = build_backbone(backbone, CHANNELS, weights_file)
        self.feat_norm = nn.LayerNorm(self.backbone.num_features)  # 이미지 특징과 조건의 크기를 맞춘다
        self.role = nn.Embedding(n_roles + 1, 32, padding_idx=0)
        self.layer = nn.Embedding(n_layers + 1, 32, padding_idx=0)
        self.use_unit_size = use_unit_size
        self.unit = nn.Sequential(nn.Linear(2, 32), nn.GELU()) if use_unit_size else None
        cond = 64 + (32 if use_unit_size else 0)
        # 이어 붙임 + MLP: 조건에 따라 이미지 특징의 뜻이 달라지게 하려면 비선형이 필요하다
        self.fuse = nn.Sequential(nn.Linear(self.backbone.num_features + cond, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(SetBlock(dim, heads) for _ in range(blocks))
        self.seed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pool = nn.MultiheadAttention(dim, heads, batch_first=True)
        # 묶음 안에서 특징을 가운데로 모으는 정규화를 둬서, 모든 제품이 같은 특징으로 쏠리는 것을 막는다
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Linear(dim, out_dim))

    def forward(self, images, role_ids, layer_ids, valid, unit=None):
        b, n = valid.shape
        feats = self.feat_norm(self.backbone(images.flatten(0, 1))).view(b, n, -1)
        cond = [self.role(role_ids), self.layer(layer_ids)]
        if self.use_unit_size:
            if unit is None:
                raise ValueError("use_unit_size=True면 unit 크기를 줘야 한다")
            cond.append(self.unit(unit)[:, None].expand(b, n, -1))
        x = self.fuse(torch.cat([feats, *cond], dim=-1))
        pad = ~valid
        for block in self.blocks:
            x = block(x, pad)
        pooled = self.pool(self.seed.expand(b, -1, -1), x, x, key_padding_mask=pad, need_weights=False)[0][:, 0]
        return F.normalize(self.head(pooled), dim=-1)
