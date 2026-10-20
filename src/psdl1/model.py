"""원소 인코더 + 조건 결합 + Set Transformer → 제품 특징 하나."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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


MODEL_INPUT_VERSION = "dl1_structured_v2_padded_endpoints"


def build_backbone(name="tiny", in_chans=CHANNELS, weights_file=None):
    """Construct without pretrained downloads; optional local state dict only."""
    if weights_file is not None and not Path(weights_file).is_file():
        raise ValueError(f"local weights file does not exist: {weights_file}")
    if name == "tiny":
        net = TinyBackbone(in_chans)
    else:
        try:
            import timm
        except ImportError as err:
            raise ImportError("optional timm dependency is required") from err
        net = timm.create_model(name, pretrained=False, in_chans=in_chans, num_classes=0)
    if weights_file:
        path = Path(weights_file)
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(path), device="cpu")
        elif path.suffix in {".pt", ".pth"}:
            state = torch.load(path, map_location="cpu", weights_only=True)
        else:
            raise ValueError("local weights format must be .pt/.pth/.safetensors")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in state.values()):
            raise ValueError("local weights require a finite state dict")
        net.load_state_dict(state, strict=True)
    return net


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
        if any(type(v) is not int or v <= 0 for v in (n_roles, n_layers, dim, out_dim, heads)) or type(blocks) is not int or blocks < 0 or dim % heads:
            raise ValueError("positive dimensions and divisible attention heads required")
        self.backbone = build_backbone(backbone, CHANNELS + 1, weights_file)
        # 학습 전용 메모리 절약 스위치. 가중치·연산 결과는 바뀌지 않으므로 config/체크포인트 계약에 넣지 않는다.
        self.grad_checkpoint = False
        self.checkpoint_chunk = 16
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
        self.endpoint = nn.Sequential(nn.Linear(dim + 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.connection = nn.Sequential(nn.Linear(dim + 3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.connectivity = nn.Embedding(3, dim)  # unavailable, partial, source-complete
        # LayerNorm supports singleton batches; collapse prevention requires evaluation.
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, out_dim))

    def _run_backbone(self, x):
        if not (self.grad_checkpoint and self.training and torch.is_grad_enabled()):
            return self.backbone(x)
        # TinyBackbone은 GroupNorm이라 샘플끼리 독립이므로 조각내도 결과가 같다. 다른 백본(BatchNorm 가능)은 통째로 한 번에.
        chunks = x.split(self.checkpoint_chunk) if isinstance(self.backbone, TinyBackbone) else (x,)
        return torch.cat([checkpoint(self.backbone, c, use_reentrant=False) for c in chunks])

    def forward(self, images, role_ids, layer_ids, valid, unit=None, *, spatial_valid=None,
                connection_features=None, connection_valid=None, connectivity_status=None):
        if valid.ndim != 2 or valid.dtype != torch.bool or not valid.numel() or not valid.any(dim=1).all():
            raise ValueError("each product requires at least one valid element")
        b, n = valid.shape
        if images.ndim != 5 or images.shape[:3] != (b, n, CHANNELS) or images.shape[-1] != images.shape[-2] or images.shape[-1] <= 0 or not torch.isfinite(images).all() or (images < 0).any() or (images > 1).any():
            raise ValueError("finite [B,N,3,G,G] coverage images required")
        for ids, embedding in ((role_ids, self.role), (layer_ids, self.layer)):
            if ids.shape != (b, n) or ids.dtype != torch.long or (ids < 0).any() or (ids >= embedding.num_embeddings).any() or (ids[valid] == 0).any() or (ids[~valid] != 0).any():
                raise ValueError("role/layer IDs must be known, bound, and zero only for padding")
        g = images.shape[-1]
        if spatial_valid is None:
            spatial_valid = torch.ones(b, g, g, device=images.device, dtype=images.dtype)
        if spatial_valid.shape != (b, g, g) or not torch.isfinite(spatial_valid).all() or (spatial_valid < 0).any() or (spatial_valid > 1).any() or not spatial_valid.flatten(1).any(dim=1).all():
            raise ValueError("spatial validity must have observed coverage per product")
        coverage = spatial_valid[:, None, None].expand(b, n, 1, g, g)
        backbone_input = torch.cat([images, coverage], dim=2)
        feats = self.feat_norm(self._run_backbone(backbone_input.flatten(0, 1))).view(b, n, -1)
        cond = [self.role(role_ids), self.layer(layer_ids)]
        if self.use_unit_size:
            if unit is None or unit.shape != (b, 2) or not torch.isfinite(unit).all():
                raise ValueError("use_unit_size=True면 unit 크기를 줘야 한다")
            cond.append(self.unit(unit)[:, None].expand(b, n, -1))
        x = self.fuse(torch.cat([feats, *cond], dim=-1))
        pad = ~valid
        for block in self.blocks:
            x = block(x, pad)
        pooled = self.pool(self.seed.expand(b, -1, -1), x, x, key_padding_mask=pad, need_weights=False)[0][:, 0]
        if connectivity_status is None:
            connectivity_status = torch.zeros(b, dtype=torch.long, device=images.device)
        if connectivity_status.shape != (b,) or connectivity_status.dtype != torch.long or (connectivity_status < 0).any() or (connectivity_status > 2).any():
            raise ValueError("invalid connectivity availability")
        pooled = pooled + self.connectivity(connectivity_status)
        if connection_features is not None:
            if connection_features.ndim != 3 or connection_features.shape[0] != b or connection_features.shape[2] != 9 or not torch.isfinite(connection_features).all():
                raise ValueError("connection features require finite [B,K,9]")
            k = connection_features.shape[1]
            if connection_valid is None or connection_valid.shape != (b, k) or connection_valid.dtype != torch.bool:
                raise ValueError("connection padding mask required")
            cf = connection_features
            if k:
                indices = cf[:, :, :2]
                if (indices != indices.floor()).any() or (indices < 0).any() or (indices >= n).any() or (cf[:, :, 2:] < 0).any() or (cf[:, :, 2:] > 1).any():
                    raise ValueError("invalid bound endpoint features")
                batch = torch.arange(b, device=images.device)[:, None].expand(b, k)
                ia, ib = indices[:, :, 0].long(), indices[:, :, 1].long()
                if ((~valid[batch, ia] | ~valid[batch, ib]) & connection_valid).any():
                    raise ValueError("endpoint bound to padding")
                # Undirected endpoint pairing: swapping ends leaves the relation feature unchanged.
                ea = self.endpoint(torch.cat([x[batch, ia], cf[:, :, 2:4]], dim=-1))
                eb = self.endpoint(torch.cat([x[batch, ib], cf[:, :, 4:6]], dim=-1))
                edge = self.connection(torch.cat([ea + eb, cf[:, :, 6:]], dim=-1))
                pooled = pooled + (edge * connection_valid[:, :, None]).sum(dim=1) / connection_valid.sum(dim=1, keepdim=True).clamp_min(1)
        elif connection_valid is not None:
            raise ValueError("connection features required with mask")
        return F.normalize(self.head(pooled), dim=-1)
