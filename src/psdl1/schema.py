"""DL1 original-shape draft; input method approved, real grid/endpoint format pending."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import numpy as np

SCHEMA_VERSION = "dl1_shape_v1_draft"
FORBIDDEN_FIELDS = {"hotspot", "ca", "fab"}


def approval_status():
    return {"input_method": "user_approved_2026_10_09", "real_grid_linewidth": "pending",
            "endpoint_source_format": "pending", "G6": "pending"}


def identifier(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name}: expected nonempty trimmed ID")
    return value


def numbers(value, count, name, *, positive=False):
    if not isinstance(value, (tuple, list)) or len(value) != count:
        raise ValueError(f"{name}: expected {count} numbers")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        raise ValueError(f"{name}: expected numeric values")
    a = np.asarray(value, dtype=float)
    if not np.isfinite(a).all() or (positive and (a <= 0).any()):
        raise ValueError(f"{name}: finite {'positive ' if positive else ''}values required")
    return tuple(float(v) for v in a)


def binary_mask(value, name="mask"):
    a = np.asarray(value)
    if a.ndim != 2 or not all(a.shape):
        raise ValueError(f"{name}: nonempty 2-D mask required")
    if a.dtype.kind not in "buif" or not np.isfinite(a).all():
        raise ValueError(f"{name}: finite numeric binary mask required")
    if not (np.isin(a, [0, 1]).all() or np.isin(a, [0, 255]).all()):
        raise ValueError(f"{name}: binary 0/1 or 0/255 mask required")
    return (a != 0).astype(np.float32)


def asset_path(root: Path, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError("asset path must be relative; absolute paths are forbidden")
    if "\\" in value or ".." in Path(value).parts:
        raise ValueError("asset path escapes manifest 폴더 밖 or uses unsupported separators")
    target = (root / value).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError("asset path points manifest 폴더 밖")
    if target.suffix != ".npy" or not target.is_file():
        raise ValueError(f"missing .npy asset: {value}")
    return target


def reject_forbidden(value):
    if isinstance(value, dict):
        for k, v in value.items():
            if k.casefold() in FORBIDDEN_FIELDS:
                raise ValueError(f"forbidden model input field: {k}")
            reject_forbidden(v)
    elif isinstance(value, list):
        for item in value:
            reject_forbidden(item)


def fields(raw, required, optional=()):
    if not isinstance(raw, dict):
        raise ValueError("expected object")
    missing, extra = set(required) - raw.keys(), raw.keys() - set(required) - set(optional)
    if missing or extra:
        raise ValueError(f"contract fields: missing={sorted(missing)}, unknown={sorted(extra)}")


@dataclass(frozen=True)
class Frame:
    frame_id: str
    shape_yx: tuple[int, int]
    pixel_pitch_um: float
    origin_xy_um: tuple[float, float] = (0.0, 0.0)
    length_unit: str = "um"
    axis_order: str = "yx"


@dataclass(frozen=True)
class Element:
    role: str
    layer: str  # actual layer, never independently centred
    mask: np.ndarray | None
    electrode_id: str = ""  # direct synthetic callers may use role as ID
    element_id: str = ""
    observation: str = "observed"

    @property
    def key(self):
        return self.element_id or f"{self.electrode_id or self.role}@{self.layer}"


@dataclass(frozen=True)
class Endpoint:
    element_id: str
    xy_um: tuple[float, float]


@dataclass(frozen=True)
class Connection:
    connection_id: str
    kind: str  # CNT or VIA, source-supplied relation only
    endpoints: tuple[Endpoint, Endpoint]
    mask: np.ndarray | None
    observation: str = "observed"


@dataclass(frozen=True)
class Product:
    product_id: str
    unit_size_um: tuple[float, float]
    elements: tuple[Element, ...]
    cnt: np.ndarray | None = None
    lineage_id: str | None = None
    design_version: str | None = None
    frame: Frame | None = None
    valid: np.ndarray | None = None
    connections: tuple[Connection, ...] = ()
    connectivity_observation: str = "unavailable"
    grid_check: dict | None = None

    @property
    def connectivity_status(self):
        if self.connectivity_observation != "complete":
            return self.connectivity_observation
        if any(c.observation != "observed" for c in self.connections) or any(e.observation != "observed" for e in self.elements) or (self.valid is not None and not np.asarray(self.valid).all()):
            return "partial"
        return "available"  # supplied metadata, never CNN-verified


def validate_product(p: Product):
    identifier(p.product_id, "product_id")
    for name in ("lineage_id", "design_version"):
        if getattr(p, name) is not None:
            identifier(getattr(p, name), name)
    unit = numbers(p.unit_size_um, 2, "unit_size_um", positive=True)
    if not p.elements:
        raise ValueError("product requires elements")
    if p.connectivity_observation not in {"complete", "partial", "unavailable"}:
        raise ValueError("invalid connectivity observation")
    if p.connections and p.connectivity_observation == "unavailable":
        raise ValueError("relations require partial or complete connectivity observation")
    ids, bindings, shapes = set(), set(), set()
    for e in p.elements:
        identifier(e.role, "role")
        identifier(e.layer, "actual_layer")
        identifier(e.key, "element_id")
        identifier(e.electrode_id or e.role, "electrode_id")
        binding = (e.role, e.layer)
        if e.key in ids or binding in bindings:
            raise ValueError("duplicate element ID or role-layer binding")
        ids.add(e.key)
        bindings.add(binding)
        if e.observation not in {"observed", "missing"} or (e.mask is None) != (e.observation == "missing"):
            raise ValueError("element observation and mask disagree")
        if e.mask is not None:
            shapes.add(binary_mask(e.mask).shape)
    f = p.frame
    if f is not None:
        identifier(f.frame_id, "frame_id")
        if f.length_unit != "um" or f.axis_order != "yx":
            raise ValueError("frame requires um and yx")
        if not isinstance(f.shape_yx, (tuple, list)) or len(f.shape_yx) != 2 or any(type(v) is not int or v <= 0 for v in f.shape_yx):
            raise ValueError("invalid frame shape_yx")
        pitch = numbers([f.pixel_pitch_um], 1, "pixel_pitch_um", positive=True)[0]
        numbers(f.origin_xy_um, 2, "origin_xy_um")
        if not np.allclose(unit, (f.shape_yx[1] * pitch, f.shape_yx[0] * pitch), atol=1e-9, rtol=1e-9):
            raise ValueError("unit_size_um disagrees with frame extent")
        shapes.add(tuple(f.shape_yx))
    if not shapes or len(shapes) != 1:
        raise ValueError("one product must share the same 2-D frame grid")
    shape = next(iter(shapes))
    if f is None and not np.isclose(unit[0] / shape[1], unit[1] / shape[0], atol=1e-9, rtol=1e-9):
        raise ValueError("unit requires isotropic pixel pitch")
    valid = np.ones(shape, dtype=np.float32) if p.valid is None else binary_mask(p.valid, "valid")
    if valid.shape != shape:
        raise ValueError("valid grid differs from frame")
    if p.cnt is not None and binary_mask(p.cnt, "CNT").shape != shape:
        raise ValueError("CNT grid differs from frame")
    connection_ids = set()
    by_id = {e.key: e for e in p.elements}
    for c in p.connections:
        identifier(c.connection_id, "connection_id")
        if c.connection_id in connection_ids or c.kind not in {"CNT", "VIA"}:
            raise ValueError("duplicate connection ID or invalid kind")
        connection_ids.add(c.connection_id)
        if c.observation not in {"observed", "missing"} or (c.mask is None) != (c.observation == "missing"):
            raise ValueError("connection observation and mask disagree")
        if len(c.endpoints) != 2 or c.endpoints[0].element_id == c.endpoints[1].element_id:
            raise ValueError("connection requires two distinct bound endpoints")
        if f is None:
            raise ValueError("endpoint relations require explicit frame")
        cm = None if c.mask is None else binary_mask(c.mask, c.kind)
        if cm is not None and cm.shape != shape:
            raise ValueError("connection mask grid differs from frame")
        for ep in c.endpoints:
            identifier(ep.element_id, "endpoint element_id")
            if ep.element_id not in by_id:
                raise ValueError("endpoint references unknown element")
            xy = numbers(ep.xy_um, 2, "endpoint xy_um")
            pos = (np.asarray(xy) - f.origin_xy_um) / f.pixel_pitch_um
            if (np.asarray(xy) < f.origin_xy_um).any() or (np.asarray(xy) >= np.asarray(f.origin_xy_um) + np.asarray((shape[1], shape[0])) * f.pixel_pitch_um).any():
                raise ValueError("endpoint outside half-open frame")
            x, y = np.minimum(np.floor(pos).astype(int), (shape[1] - 1, shape[0] - 1))
            e = by_id[ep.element_id]
            if c.observation == "observed" and (e.mask is None or not valid[y, x] or not binary_mask(e.mask)[y, x] or not cm[y, x]):
                raise ValueError("observed endpoint must occupy valid electrode and connection masks")
    return shape, valid
