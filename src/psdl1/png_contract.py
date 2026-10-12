"""Versioned PNG sidecar contracts; never additions to the strict v1 manifest.

T1 freezes these public records for extraction (T2) and publication (T3).
None resource values mean unknown/not computed, never an estimated zero.
"""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
from typing import Any

import numpy as np

REPORT_SCHEMA_VERSION = "dl1_png_import_v1"
METADATA_SCHEMA_VERSION = "dl1_png_metadata_v1"
ALGORITHM_ID = "nearest_overlap_common_intersection"
ALGORITHM_VERSION = "1"
PIXEL_PITCH_UM = 0.1
VALID_ASSUMPTION = "full_frame_observed_no_coverage_metadata"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def canonical_id(kind: str, value: Any) -> str:
    """Filename-safe SHA-256 of a namespaced canonical JSON value."""
    return kind + "_" + hashlib.sha256(canonical_json([kind, value])).hexdigest()


def mask_hash(mask: np.ndarray) -> str:
    """Binary geometry identity, independent of PNG mode/compression/filename."""
    a = np.asarray(mask)
    if a.ndim != 2 or not np.isin(a, [0, 1]).all():
        raise ValueError("mask_hash requires a binary 2-D mask")
    digest = hashlib.sha256(canonical_json({"shape_yx": list(a.shape)}))
    digest.update(np.ascontiguousarray(a, dtype=np.bool_).tobytes())
    return digest.hexdigest()


def component_id(model: str, stack: int, layer: str, binding: tuple,
                 shape_yx: tuple[int, int], bbox_xyxy_px: tuple[int, int, int, int],
                 local_mask: np.ndarray) -> str:
    """Hash bbox-local geometry without allocating a full-frame component mask."""
    return canonical_id("component", [model, stack, layer, binding, shape_yx,
                                      bbox_xyxy_px, mask_hash(local_mask)])


@dataclass(frozen=True)
class DecoderPolicy:
    pillow_version: str
    default_max_image_pixels: int | None
    requested_max_image_pixels: int | None
    effective_max_image_pixels: int
    explicit_override: bool


@dataclass(frozen=True)
class SourceIssue:
    path: str
    reason: str
    exception_class: str | None = None
    shape_yx: tuple[int, int] | None = None
    pixels: int | None = None
    effective_max_image_pixels: int | None = None


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    shape_yx: tuple[int, int]
    frame_extent_um: tuple[float, float]
    core_bbox_xyxy_px: tuple[int, int, int, int]
    pixel_pitch_um: float = PIXEL_PITCH_UM
    origin_xy_um: tuple[float, float] = (0.0, 0.0)
    comparison_scope: str = "full_frame"
    core_layout: tuple[int, int] = (2, 2)
    single_unit_extent_um: None = None
    valid_assumption: str = VALID_ASSUMPTION
    alignment_evidence: str = "source_assertion_not_measured_registration"

    def pixel_centre_um(self, x: int, y: int) -> tuple[float, float]:
        return ((x + 0.5) * self.pixel_pitch_um, (y + 0.5) * self.pixel_pitch_um)


@dataclass(frozen=True)
class SourceAsset:
    asset_id: str
    model: str
    kind: str  # electrode or cno
    electrode: str | None
    role: str | None
    stack: int
    layer: str
    relative_path: str
    original_filename: str
    file_sha256: str
    mask_sha256: str
    decoded_mode: str
    shape_yx: tuple[int, int]
    mask: np.ndarray = field(repr=False, compare=False)
    polarity: str = "black=1,white=0"

    @property
    def binding(self) -> tuple:
        return (self.model, self.kind, self.electrode, self.stack, self.layer)


@dataclass(frozen=True)
class ResourceScenario:
    resolved_relation_count: int
    generated_mask_asset_count: int
    bool_payload_bytes: int
    npy_header_bytes: int
    projected_bool_npy_export_bytes: int
    projected_load_manifest_retained_float32_mask_bytes: int


@dataclass(frozen=True)
class ResourceEstimate:
    shape_yx: tuple[int, int] | None = None
    frame_pixels: int | None = None
    electrode_asset_count: int | None = None
    cnt_file_count: int | None = None
    cnt_component_count: int | None = None
    resolved_relation_count: int | None = None
    global_cnt_indicator: int | None = None
    generated_mask_asset_count: int | None = None
    bool_payload_bytes: int | None = None
    npy_header_bytes: int | None = None
    projected_bool_npy_export_bytes: int | None = None
    projected_load_manifest_retained_float32_mask_bytes: int | None = None
    all_components_resolved: ResourceScenario | None = None
    unknown_reasons: tuple[str, ...] = ("extraction_not_run",)
    excludes: tuple[str, ...] = ("manifest", "report", "preview")
    overhead_note: str = ("Not peak RAM or runtime prediction; temporary arrays, labels, decoder "
                          "buffers, Python objects, grid tensors and validation add overhead.")


@dataclass(frozen=True)
class ModelSource:
    model_id: str
    assets: tuple[SourceAsset, ...]
    stack_table: tuple[tuple[int, str], ...]
    roles: tuple[str, ...]
    layers: tuple[str, ...]
    frame: FrameRecord | None
    cno_present: bool
    repeated_layers: tuple[str, ...] = ()
    errors: tuple[SourceIssue, ...] = ()
    warnings: tuple[SourceIssue, ...] = ()
    resource_estimate: ResourceEstimate = field(default_factory=ResourceEstimate)

    @property
    def valid(self) -> bool:
        return not self.errors and self.frame is not None

    @property
    def electrodes(self) -> tuple[SourceAsset, ...]:
        return tuple(a for a in self.assets if a.kind == "electrode")

    @property
    def cnt_assets(self) -> tuple[SourceAsset, ...]:
        return tuple(a for a in self.assets if a.kind == "cno")


@dataclass(frozen=True)
class SourceInventory:
    models: tuple[ModelSource, ...]
    decoder: DecoderPolicy
    role_union: tuple[str, ...]
    layer_union: tuple[str, ...]
    unique_roles: tuple[str, ...]
    unique_layers: tuple[str, ...]
    errors: tuple[SourceIssue, ...] = ()
    warnings: tuple[SourceIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return bool(self.models) and not self.errors and all(m.valid for m in self.models)


def model_identity(model: ModelSource) -> str:
    return canonical_id("model", [model.model_id, model.frame.shape_yx if model.frame else None,
                                  sorted((a.binding, a.mask_sha256) for a in model.assets)])


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    lineage_id: str
    design_version: str


@dataclass(frozen=True)
class MetadataRecord:
    models: tuple[ModelMetadata, ...] = ()
    pending_models: tuple[str, ...] = ()
    metadata_schema_version: str = METADATA_SCHEMA_VERSION


@dataclass(frozen=True)
class ImportOptions:
    root: Path | None = None
    out: Path | None = None
    export: bool = False
    metadata: Path | None = None
    candidate_grid: int | None = None
    inspect_grids: tuple[int, ...] = ()
    png_max_image_pixels: int | None = None
    verify_bundle: Path | None = None

    def __post_init__(self):
        for name, values in (("candidate_grid", (self.candidate_grid,)),
                             ("png_max_image_pixels", (self.png_max_image_pixels,)),
                             ("inspect_grids", self.inspect_grids)):
            for value in values:
                if value is not None and (type(value) is not int or value <= 0):
                    raise ValueError(f"{name}: positive integer required")


@dataclass(frozen=True)
class ImportReport:
    inventory: SourceInventory
    metadata: MetadataRecord
    report_schema_version: str = REPORT_SCHEMA_VERSION
    algorithm_id: str = ALGORITHM_ID
    algorithm_version: str = ALGORITHM_VERSION
    component_neighbourhood: int = 4
    pending_policy: str = "unresolved_emit_no_relation_skipped_layers_need_confirmation"
    electrical_verification: str = "not_performed"
    real_data_validation: str = "not_performed"
    extraction_status: str = "not_run"
    export_status: str = "not_requested"
    grid_readiness: str = "pending_explicit_grid"
    # T2/T3 supply JSON-serializable extraction/grid/publication evidence.
    extraction: tuple[dict, ...] = ()
    grid_evidence: tuple[dict, ...] = ()
    generated_manifest_sha256: str | None = None
    generated_asset_hashes: tuple[tuple[str, str], ...] = ()
    product_input_identities: tuple[tuple[str, str], ...] = ()


def require_report_version(report: dict) -> None:
    if report.get("report_schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported PNG report_schema_version")
