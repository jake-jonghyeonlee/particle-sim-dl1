"""Original-shape preparation; whole-unit/set input approved; real grid and endpoint format pending."""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .schema import (SCHEMA_VERSION, Connection, Element, Endpoint, Frame, Product,
                     asset_path, binary_mask, fields, identifier, reject_forbidden,
                     validate_product, numbers, approval_status)

CHANNELS = 3  # image channels; structured encoder also consumes spatial valid and bindings


def _grid(grid):
    if type(grid) is not int or grid <= 0:
        raise ValueError("grid must be a positive integer")
    return grid


@dataclass(frozen=True)
class FrameTransform:
    """Shared rounded full-frame geometry; raster loss is not reversible."""
    shape_yx: tuple[int, int]
    grid: int
    resized_shape_yx: tuple[int, int]
    padding_ltrb: tuple[int, int, int, int]
    pixel_pitch_um: float | None = None

    @property
    def scale_xy(self):
        h, w = self.shape_yx
        nh, nw = self.resized_shape_yx
        return nw / w, nh / h

    def pixel_to_grid(self, x, y):
        """Source pixel-centre indices to continuous grid-edge coordinates."""
        sx, sy = self.scale_xy
        left, top, _, _ = self.padding_ltrb
        return (x + 0.5) * sx + left, (y + 0.5) * sy + top

    def grid_to_pixel(self, gx, gy):
        sx, sy = self.scale_xy
        left, top, _, _ = self.padding_ltrb
        return (gx - left) / sx - 0.5, (gy - top) / sy - 0.5

    def source_edge_to_normalized(self, x, y):
        """Keep legacy multiply/divide order exactly for model feature bytes."""
        h, w = self.shape_yx
        nh, nw = self.resized_shape_yx
        left, top, _, _ = self.padding_ltrb
        return ((x * nw / w + left) / self.grid, (y * nh / h + top) / self.grid)

    def to_record(self):
        sx, sy = self.scale_xy
        return {"source_shape_yx": list(self.shape_yx), "grid": self.grid,
                "resized_shape_yx": list(self.resized_shape_yx), "padding_ltrb": list(self.padding_ltrb),
                "scale_xy": [sx, sy], "original_pixel_pitch_um": self.pixel_pitch_um,
                "effective_content_pitch_xy_um": None if self.pixel_pitch_um is None else [self.pixel_pitch_um / sx, self.pixel_pitch_um / sy],
                "comparison_scope": "full_frame", "padding_physical_extent": "none",
                "mapping": "grid_edge_xy=((x+0.5)*sx+left,(y+0.5)*sy+top)"}


def frame_transform(shape_yx, grid, pixel_pitch_um=None):
    """One unchanged area-resize/central-pad recipe; never guess a grid."""
    _grid(grid)
    if len(shape_yx) != 2 or any(type(v) is not int or v <= 0 for v in shape_yx):
        raise ValueError("transform requires positive integer shape_yx")
    if pixel_pitch_um is not None:
        numbers([pixel_pitch_um], 1, "pixel_pitch_um", positive=True)
    h, w = shape_yx
    nh, nw = max(1, round(h * grid / max(h, w))), max(1, round(w * grid / max(h, w)))
    left, top = (grid - nw) // 2, (grid - nh) // 2
    return FrameTransform(tuple(shape_yx), grid, (nh, nw),
                          (left, top, grid - nw - left, grid - nh - top), pixel_pitch_um)


def normalize_unit(mask: np.ndarray, grid: int) -> np.ndarray:
    """Common full-frame area resize + central pad, never per-element centring."""
    _grid(grid)
    m = torch.from_numpy(binary_mask(mask))
    transform = frame_transform(tuple(m.shape), grid)
    nh, nw = transform.resized_shape_yx
    resized = F.interpolate(m[None, None], size=(nh, nw), mode="area")[0, 0]
    out = torch.zeros(grid, grid)
    left, top, _, _ = transform.padding_ltrb
    out[top:top + nh, left:left + nw] = resized
    return out.numpy()


def narrowest_run_px(mask: np.ndarray) -> int:
    """Auxiliary axis-run estimate; zero for empty, not a physical linewidth proof."""
    m = binary_mask(mask).astype(bool)
    lengths = []
    for arr in (m, m.T):
        edges = np.diff(np.pad(arr, ((0, 0), (1, 1))).astype(np.int8), axis=1)
        for row in edges:
            starts, ends = np.flatnonzero(row == 1), np.flatnonzero(row == -1)
            if len(starts):
                lengths.append(int((ends - starts).min()))
    return min(lengths, default=0)


def _components(mask):
    """4-neighbour components via horizontal runs (no optional dependency)."""
    parent = []
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    previous = []
    for row in np.asarray(mask, dtype=bool):
        edges = np.diff(np.pad(row, (1, 1)).astype(np.int8))
        current = []
        j = 0
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            idx = len(parent)
            parent.append(idx)
            while j < len(previous) and previous[j][1] <= start:
                j += 1
            k = j
            while k < len(previous) and previous[k][0] < end:
                parent[find(idx)] = find(previous[k][2])
                k += 1
            current.append((start, end, idx))
        previous = current
    return len({find(i) for i in range(len(parent))})


def check_resampling(source, transformed, *, threshold=0.5):
    """Conservative bridge/component diagnostic, also applicable after common crop."""
    src = binary_mask(source)
    out = np.asarray(transformed)
    if out.ndim != 2 or not out.size or not np.isfinite(out).all() or (out < 0).any() or (out > 1).any():
        raise ValueError("transformed mask must be finite 2-D coverage in [0,1]")
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0,1]")
    before, after = _components(src), _components(out >= threshold)
    return {"threshold": threshold, "neighbours": 4,
            "source_components": before, "target_components": after,
            "source_active_pixels": int(src.sum()), "target_active_pixels": int((out >= threshold).sum()),
            "bridge_or_component_loss": before != after,
            "connectivity_preserving": False,
            "certification": "not_certified: component equality and area similarity do not prove connectivity"}


def grid_check(product: Product, grid: int):
    shape, valid = validate_product(product)
    _grid(grid)
    scale = grid / max(shape)
    evidence = []
    for name, mask in [(e.key, e.mask) for e in product.elements] + [(c.connection_id, c.mask) for c in product.connections] + ([("global_CNT", product.cnt)] if product.cnt is not None else []):
        if mask is None:
            evidence.append({"id": name, "observation": "missing"})
            continue
        source = binary_mask(mask) * valid
        width = narrowest_run_px(source)
        evidence.append({"id": name, "observation": "observed", "axis_run_px": width,
                         "estimated_run_at_grid_px": width * scale,
                         "thin_line_warning": bool(width and width * scale < 2),
                         **check_resampling(source, normalize_unit(source, grid))})
    reasons = []
    if any(e.get("thin_line_warning") for e in evidence):
        reasons.append("axis_run_below_2px; real_minimum_linewidth_unconfirmed")
    if any(e.get("bridge_or_component_loss") for e in evidence):
        reasons.append("bridge_or_component_loss_at_grid")
    if any(e["observation"] == "missing" for e in evidence) or not valid.all():
        reasons.append("missing_observations_or_valid_holes")
    return {"product_id": product.product_id, "grid": grid, "source_shape_yx": list(shape),
            "transform": frame_transform(shape, grid, product.frame.pixel_pitch_um if product.frame else product.unit_size_um[0] / shape[1]).to_record(),
            "scale": scale, "policy": "whole_unit_isotropic_area_pad_candidate",
            "approval": approval_status(), "connectivity_status": product.connectivity_status,
            "connectivity_preserving": False, "status": "warning" if reasons else "candidate_only",
            "reasons": reasons, "elements": evidence}


def input_identity(product: Product):
    """Order-independent fingerprint; preserve bound role/layer and endpoint geometry."""
    validate_product(product)
    def digest(m):
        return None if m is None else hashlib.sha256(binary_mask(m).tobytes()).hexdigest()
    payload = {"frame": None if product.frame is None else product.frame.__dict__,
               "unit": product.unit_size_um, "valid": digest(product.valid), "cnt": digest(product.cnt),
               "connectivity_observation": product.connectivity_observation,
               "elements": sorted([(e.key, e.electrode_id or e.role, e.role, e.layer, e.observation, digest(e.mask)) for e in product.elements]),
               "connections": sorted([(c.connection_id, c.kind, c.observation, sorted((ep.element_id, ep.xy_um) for ep in c.endpoints), digest(c.mask)) for c in product.connections])}
    def canonical(value):
        if isinstance(value, dict):
            return {key: canonical(v) for key, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [canonical(v) for v in value]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value) if value != 0 else 0.0
        return value
    return hashlib.sha256(json.dumps(canonical(payload), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class PreparedInput:
    images: torch.Tensor
    role_ids: torch.Tensor
    layer_ids: torch.Tensor
    valid: torch.Tensor  # spatial observed coverage, distinct from collate element padding
    connections: tuple[Connection, ...]  # original-frame endpoint bindings
    connection_masks: torch.Tensor
    identity: str
    grid_check: dict
    connectivity_status: str


def prepare_product(product: Product, grid: int, roles, layers):
    shape, valid = validate_product(product)
    if any(e.mask is None for e in product.elements):
        raise ValueError("missing element observation cannot be represented as zero")
    report = grid_check(product, grid)
    if report["reasons"]:
        warnings.warn(f"{product.product_id}: grid check: {', '.join(report['reasons'])}", UserWarning, stacklevel=2)
    norm = [normalize_unit(binary_mask(e.mask) * valid, grid) for e in product.elements]
    cnt_sources = [binary_mask(c.mask) for c in product.connections if c.mask is not None]
    if product.cnt is not None:
        cnt_sources.append(binary_mask(product.cnt))
    cnt = normalize_unit(np.maximum.reduce(cnt_sources) * valid, grid) if cnt_sources else np.zeros((grid, grid), dtype=np.float32)
    # Exclude self before composing: overlapping other wiring must survive.
    rest = [np.clip(np.sum([other for j, other in enumerate(norm) if j != i], axis=0), 0, 1)
            if len(norm) > 1 else np.zeros((grid, grid), dtype=np.float32) for i in range(len(norm))]
    images = np.stack([np.stack([n, cnt, r]) for n, r in zip(norm, rest)]).astype(np.float32)
    try:
        rids = [roles[e.role] for e in product.elements]
        lids = [layers[e.layer] for e in product.elements]
    except KeyError as err:
        raise ValueError(f"어휘에 없는 역할 또는 layer: {err.args[0]!r} ({product.product_id})") from None
    cm = [normalize_unit(binary_mask(c.mask) * valid, grid) if c.mask is not None else np.zeros((grid, grid), dtype=np.float32) for c in product.connections]
    return PreparedInput(torch.from_numpy(images), torch.tensor(rids), torch.tensor(lids),
                         torch.from_numpy(normalize_unit(valid, grid)), product.connections,
                         torch.from_numpy(np.stack(cm)) if cm else torch.empty(0, grid, grid),
                         input_identity(product), report, product.connectivity_status)


def product_tensors(product: Product, grid: int, roles, layers):
    """Compatibility image-only interface; the trainer uses the structured consumer."""
    item = prepare_product(product, grid, roles, layers)
    return item.images, item.role_ids, item.layer_ids


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


def augment_with_check(images: torch.Tensor, generator: torch.Generator, scale=(0.6, 1.0), shift=0.08, *, theta=None):
    """Shared candidate transform with evidence; no automatic positive-label approval."""
    if images.ndim != 4 or not images.shape[0] or images.shape[-1] != images.shape[-2] or not torch.isfinite(images).all() or (images < 0).any() or (images > 1).any():
        raise ValueError("images must be nonempty finite [N,C,G,G] coverage")
    if len(scale) != 2 or not all(np.isfinite(v) for v in scale) or not 0 < scale[0] <= scale[1] or not np.isfinite(shift) or shift < 0:
        raise ValueError("invalid augmentation scale or shift")
    n, c, g, _ = images.shape
    if theta is None:
        s = float(torch.empty(1).uniform_(scale[0], scale[1], generator=generator))
        tx, ty = (torch.empty(2).uniform_(-shift, shift, generator=generator) + (torch.rand(2, generator=generator) - 0.5) * (1 - s)).tolist()
        theta = [[s, 0.0, tx * 2], [0.0, s, ty * 2]]
    theta = torch.as_tensor(theta, device=images.device, dtype=images.dtype)
    if theta.shape != (2, 3) or not torch.isfinite(theta).all():
        raise ValueError("one finite 2x3 common transform required")
    sample_grid = F.affine_grid(theta.expand(n, 2, 3), (n, c, g, g), align_corners=False)
    out = F.grid_sample(images, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    channels = [[check_resampling((src[ch].detach().cpu().numpy() >= 0.5).astype(np.float32), dst[ch].detach().cpu().numpy())
                 for ch in range(c)] for src, dst in zip(images, out)]
    evidence = [{**checks[0], "channels": checks,
                 "bridge_or_component_loss": any(check["bridge_or_component_loss"] for check in checks)} for checks in channels]
    report = {"elements": evidence, "connectivity_preserving": False, "approval": approval_status(),
              "safe_for_positive_label": False, "common_theta": theta.detach().cpu().tolist()}
    return out, report


def augment(images: torch.Tensor, generator: torch.Generator, scale=(0.6, 1.0), shift=0.08) -> torch.Tensor:
    """Legacy common-transform candidate; T9 must inspect evidence before labelling."""
    out, report = augment_with_check(images, generator, scale, shift)
    if any(e["bridge_or_component_loss"] for e in report["elements"]):
        warnings.warn("common transform loses a bridge/component; connectivity not preserved", UserWarning, stacklevel=2)
    return out


def load_manifest(path: str | Path, *, grid: int | None = None) -> list[Product]:
    """Strict versioned draft manifest; store per-product grid evidence on load."""
    path = Path(path)
    root = path.parent.resolve()
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result
    doc = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    reject_forbidden(doc)
    fields(doc, ("schema_version", "candidate_grid", "products"))
    if doc["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    _grid(doc["candidate_grid"])
    chosen_grid = doc["candidate_grid"] if grid is None else _grid(grid)
    if not isinstance(doc["products"], list) or not doc["products"]:
        raise ValueError("products must be nonempty list")
    def read(rel):
        return binary_mask(np.load(asset_path(root, rel), allow_pickle=False))
    products = []
    for raw in doc["products"]:
        fields(raw, ("product_id", "lineage_id", "design_version", "unit_size_um", "frame", "valid", "elements", "connections", "connectivity_observation"), ("cnt",))
        for key in ("product_id", "lineage_id", "design_version"):
            identifier(raw[key], key)
        fr = raw["frame"]
        fields(fr, ("frame_id", "length_unit", "origin_xy_um", "pixel_pitch_um", "shape_yx", "axis_order"))
        if not isinstance(fr["shape_yx"], list):
            raise ValueError("shape_yx must be a list")
        frame = Frame(fr["frame_id"], tuple(fr["shape_yx"]), fr["pixel_pitch_um"], fr["origin_xy_um"], fr["length_unit"], fr["axis_order"])
        if not isinstance(raw["elements"], list) or not isinstance(raw["connections"], list):
            raise ValueError("elements and connections must be lists")
        elements = []
        for e in raw["elements"]:
            fields(e, ("element_id", "electrode_id", "role", "actual_layer", "observation", "mask"))
            identifier(e["element_id"], "element_id")
            identifier(e["electrode_id"], "electrode_id")
            elements.append(Element(e["role"], e["actual_layer"], None if e["mask"] is None else read(e["mask"]), e["electrode_id"], e["element_id"], e["observation"]))
        connections = []
        for c in raw["connections"]:
            fields(c, ("connection_id", "kind", "endpoints", "mask", "observation"))
            if not isinstance(c["endpoints"], list):
                raise ValueError("endpoints must be list")
            endpoints = []
            for ep in c["endpoints"]:
                fields(ep, ("element_id", "xy_um"))
                endpoints.append(Endpoint(ep["element_id"], numbers(ep["xy_um"], 2, "endpoint xy_um")))
            connections.append(Connection(c["connection_id"], c["kind"], tuple(endpoints), None if c["mask"] is None else read(c["mask"]), c["observation"]))
        product = Product(raw["product_id"], raw["unit_size_um"], tuple(elements), None if raw.get("cnt") is None else read(raw["cnt"]),
                          raw["lineage_id"], raw["design_version"], frame, read(raw["valid"]), tuple(connections), raw["connectivity_observation"])
        report = grid_check(product, chosen_grid)
        report["manifest_candidate_grid"] = doc["candidate_grid"]
        products.append(replace(product, grid_check=report))
    if len({p.product_id for p in products}) != len(products):
        raise ValueError("duplicate product_id")
    if not all(np.isclose(p.frame.pixel_pitch_um, products[0].frame.pixel_pitch_um, atol=1e-9, rtol=1e-9) for p in products):
        raise ValueError("products must share the same pixel pitch")
    return products


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate draft original-shape inputs; never certifies connectivity")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--grid", type=int)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    products = load_manifest(args.manifest, grid=args.grid)
    report = {"schema_version": SCHEMA_VERSION, "approval": approval_status(), "products": [p.grid_check for p in products]}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
