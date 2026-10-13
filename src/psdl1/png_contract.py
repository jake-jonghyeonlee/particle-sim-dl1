"""Versioned PNG sidecar contracts; never additions to the strict v1 manifest.

These public records describe source inventory, extraction and publication.
None resource values mean unknown/not computed, never an estimated zero.
"""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
from typing import Any

import numpy as np

REPORT_SCHEMA_VERSION = "dl1_png_import_v3"
METADATA_SCHEMA_VERSION = "dl1_png_metadata_v1"
ALGORITHM_ID = "nearest_overlap_common_intersection"
ALGORITHM_VERSION = "2"
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


def component_id(model: str, layer: str, binding: tuple,
                 shape_yx: tuple[int, int], bbox_xyxy_px: tuple[int, int, int, int],
                 local_mask: np.ndarray, *, local_mask_sha256: str | None = None) -> str:
    """Hash bbox-local geometry without allocating a full-frame component mask."""
    return canonical_id("component", [model, layer, binding, shape_yx,
                                      bbox_xyxy_px, local_mask_sha256 if local_mask_sha256 is not None else mask_hash(local_mask)])


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
    layer: str
    filename_number: int
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
        if self.kind == "cno":
            return (self.model, "cno", None, self.layer, self.mask_sha256)
        return (self.model, self.kind, self.electrode, self.layer)


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
                          "buffers, Python objects, grid tensors and validation add overhead; "
                          "PNG decoding can transiently use about 17–18 bytes/pixel plus its compressed "
                          "snapshot, and inventory retains about one bool byte/pixel/asset across all models.")


STACK_ORDER_SCHEMA_VERSION = "dl1_png_stack_order_v1"


@dataclass(frozen=True)
class StackOrderInput:
    stack_order_schema_version: str | None = None
    source: str | None = None
    file_sha256: str | None = None
    document: dict | None = None
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class OrderEntry:
    order_index: int
    kind: str
    name: str
    asset_ids: tuple[str, ...]


@dataclass(frozen=True)
class StackOrderResolution:
    status: str = "not_required"
    source: str | None = None
    selected_list: str | None = None
    resolved_order: tuple[OrderEntry, ...] = ()
    ignored_names: tuple[str, ...] = ()
    resolved_order_sha256: str | None = None
    errors: tuple[str, ...] = ()
    matched_pairs: tuple[tuple[str, str], ...] = ()


def order_digest(entries):
    return hashlib.sha256(canonical_json([[e.kind, e.name] for e in entries])).hexdigest()


@dataclass(frozen=True)
class ModelSource:
    model_id: str
    assets: tuple[SourceAsset, ...]
    roles: tuple[str, ...]
    layers: tuple[str, ...]
    frame: FrameRecord | None
    cno_present: bool
    errors: tuple[SourceIssue, ...] = ()
    warnings: tuple[SourceIssue, ...] = ()
    resource_estimate: ResourceEstimate = field(default_factory=ResourceEstimate)

    stack_order: StackOrderResolution = field(default_factory=StackOrderResolution)

    def order_index(self, asset):
        return next((e.order_index for e in self.stack_order.resolved_order if asset.asset_id in e.asset_ids), None)

    @property
    def electrode_layers(self) -> tuple[str, ...]:
        return tuple(sorted({a.layer for a in self.electrodes}))

    @property
    def cno_layers(self) -> tuple[str, ...]:
        return tuple(sorted({a.layer for a in self.cnt_assets}))

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
                                  [(a.binding, a.mask_sha256) for a in sorted(model.assets,
                                      key=lambda a: (a.kind, a.electrode or "", a.layer, a.mask_sha256))]])


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
    stack_order: Path | None = None

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
    stack_order: StackOrderInput = field(default_factory=StackOrderInput)
    report_schema_version: str = REPORT_SCHEMA_VERSION
    algorithm_id: str = ALGORITHM_ID
    algorithm_version: str = ALGORITHM_VERSION
    component_neighbourhood: int = 4
    same_electrode_required: bool = False
    pending_policy: str = "unresolved_emit_no_relation_skipped_layers_need_confirmation"
    electrical_verification: str = "not_performed"
    real_data_validation: str = "not_performed"
    extraction_status: str = "not_run"
    export_status: str = "not_requested"
    grid_readiness: str = "pending_explicit_grid"
    publication_method: str | None = None
    # T2/T3 supply JSON-serializable extraction/grid/publication evidence.
    extraction: tuple["ModelExtraction", ...] = ()
    grid_evidence: tuple[dict, ...] = ()
    generated_manifest_sha256: str | None = None
    generated_asset_hashes: tuple[tuple[str, str], ...] = ()
    product_input_identities: tuple[tuple[str, str], ...] = ()
    candidate_grid: int | None = None
    pending_prerequisites: tuple[str, ...] = ()
    diagnostics: tuple[SourceIssue, ...] = ()
    duplicate_geometry: tuple[dict, ...] = ()
    preview: tuple[dict, ...] = ()
    status_summary: tuple[dict, ...] = ()
    extraction_run_meaning: str = "extractor_ran; distinct_from_extraction_coverage_and_model_status"


def require_report_version(report: dict) -> None:
    if not isinstance(report, dict) or report.get("report_schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported PNG report_schema_version")
    if report.get("algorithm_id") != ALGORITHM_ID or report.get("algorithm_version") != ALGORITHM_VERSION:
        raise ValueError("unsupported PNG algorithm_id/algorithm_version")
    if not isinstance(report.get("extraction"), list) or any(not isinstance(e, dict) for e in report["extraction"]):
        raise ValueError("invalid PNG stack order evidence: extraction list")
    for extraction in report["extraction"]:
        if extraction.get("algorithm_id") != ALGORITHM_ID or extraction.get("algorithm_version") != ALGORITHM_VERSION:
            raise ValueError("unsupported PNG algorithm_id/algorithm_version")
    try:
        _validate_order_evidence(report)
    except (ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("invalid PNG stack order evidence: " + str(exc)) from exc


@dataclass(frozen=True)
class LayerEvidence:
    order_index: int
    layer: str
    asset_ids: tuple[str, ...]
    source_paths: tuple[str, ...]


@dataclass(frozen=True)
class SkippedLayers:
    electrode_layers: tuple[LayerEvidence, ...] = ()
    cno_layers: tuple[LayerEvidence, ...] = ()

    @property
    def present(self) -> bool:
        return bool(self.electrode_layers or self.cno_layers)


@dataclass(frozen=True)
class OverlapCandidate:
    element_id: str
    asset_id: str
    electrode: str
    order_index: int
    layer: str
    source_path: str
    overlap_pixels: int


@dataclass(frozen=True)
class IgnoredCandidate:
    candidate: OverlapCandidate
    reason: str = "deeper_than_nearest_overlapping_lower"


@dataclass(frozen=True)
class ExtractedEndpoint:
    element_id: str
    xy_um: tuple[float, float]


@dataclass(frozen=True)
class ExtractedRelation:
    connection_id: str
    component_id: str
    endpoints: tuple[ExtractedEndpoint, ExtractedEndpoint]
    representative_xy_px: tuple[int, int]
    common_intersection_pixels: int
    common_intersection_sha256: str
    skipped_upper_layers: SkippedLayers
    skipped_lower_layers: SkippedLayers
    needs_confirmation: bool
    selection_reason: str = "unique_nearest_overlapping_upper_and_lower_with_connected_common_intersection"
    kind: str = "CNT"
    observation: str = "observed"


@dataclass(frozen=True)
class ExtractedComponent:
    component_id: str
    source_asset_id: str
    source_path: str
    cnt_order_index: int | None
    cnt_filename_number: int
    cnt_layer: str
    shape_yx: tuple[int, int]
    bbox_xyxy_px: tuple[int, int, int, int]
    pixel_count: int
    mask_sha256: str  # canonical bbox-local mask; bbox/shape bound by component_id
    mask: np.ndarray = field(repr=False, compare=False)  # bbox-local only
    above_candidates: tuple[OverlapCandidate, ...] = ()
    below_candidates: tuple[OverlapCandidate, ...] = ()
    selected_upper_candidates: tuple[OverlapCandidate, ...] = ()
    selected_lower_candidates: tuple[OverlapCandidate, ...] = ()
    ignored_lower_candidates: tuple[IgnoredCandidate, ...] = ()
    unresolved_reasons: tuple[str, ...] = ()
    common_intersection_pixels: int | None = None
    common_intersection_regions: int | None = None
    relation: ExtractedRelation | None = None


@dataclass(frozen=True)
class ExtractionSummary:
    component_count: int
    resolved_relation_count: int
    unresolved_component_count: int
    needs_confirmation_relations: int
    skipped_upper_relations: int
    skipped_lower_relations: int
    skipped_upper_electrode_layers: int
    skipped_upper_cno_layers: int
    skipped_lower_electrode_layers: int
    skipped_lower_cno_layers: int


@dataclass(frozen=True)
class ModelExtraction:
    model_id: str
    components: tuple[ExtractedComponent, ...]
    connectivity_observation: str
    summary: ExtractionSummary
    resource_estimate: ResourceEstimate
    source_reasons: tuple[str, ...] = ()
    extraction_status: str = "complete"
    algorithm_id: str = ALGORITHM_ID
    algorithm_version: str = ALGORITHM_VERSION
    component_neighbourhood: int = 4
    same_electrode_required: bool = False
    position_source: str | None = None
    component_neighbourhood_evidence: str = "user_accepted_4_neighbour"
    rule_evidence: str = "implementation_interpretation_not_electrical_verification"
    residual_limitations: str = "connected_common_intersection_cannot_prove_single_physical_contact"
    electrical_verification: str = "not_performed"
    real_data_validation: str = "not_performed"
    coverage_meaning: str = "complete means extraction coverage only; electrical verification not performed"

    @property
    def relations(self) -> tuple[ExtractedRelation, ...]:
        return tuple(c.relation for c in self.components if c.relation is not None)


def _validate_order_evidence(report):
    from .png_source import _order_document
    def require(condition, reason):
        if not condition:
            raise ValueError(reason)
    def digest(value):
        return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
    top = report['stack_order']
    require(set(top) == {'stack_order_schema_version', 'source', 'file_sha256', 'document', 'errors'}, 'input fields')
    require(isinstance(top['errors'], list) and all(isinstance(e, str) for e in top['errors']), 'input errors')
    require(top['source'] in {None, 'user_supplied_order_file'}, 'input source')
    require(top['file_sha256'] is None or digest(top['file_sha256']), 'input hash')
    if top['document'] is not None:
        _order_document(top['document'])
        require(top['stack_order_schema_version'] == STACK_ORDER_SCHEMA_VERSION and
                top['source'] == 'user_supplied_order_file' and digest(top['file_sha256']) and not top['errors'], 'validated input')
    else:
        require(top['stack_order_schema_version'] is None, 'unparsed version')
        require(top['source'] is not None or (top['file_sha256'] is None and not top['errors']), 'absent input')
        require(top['source'] is None or bool(top['errors']), 'invalid input errors')
    require(report['extractor_run_status'] in {'ran', 'partial', 'blocked'}, 'run status')
    require(report['same_electrode_required'] is False and report['component_neighbourhood'] == 4, 'algorithm parameters')
    extractions = {e['model_id']: e for e in report['extraction']}
    require(len(extractions) == len(report['extraction']), 'duplicate extraction')
    states = []
    for model in report['inventory']['models']:
        order = model['stack_order']
        require(set(order) == {'status', 'source', 'selected_list', 'resolved_order', 'ignored_names', 'resolved_order_sha256', 'errors', 'matched_pairs'}, 'resolution fields')
        status = order['status']
        require(status in {'resolved', 'required', 'invalid', 'not_required'}, 'resolution status')
        require(isinstance(order['errors'], list) and all(isinstance(e, str) for e in order['errors']), 'resolution errors')
        require(isinstance(order['ignored_names'], list) and isinstance(order['resolved_order'], list), 'resolution lists')
        require('electrode_stack_table' not in model and 'stack_table' not in model, 'obsolete number table')
        assets = model['assets']
        groups = {}
        for a in assets:
            require('stack' not in a and type(a['filename_number']) is int and a['filename_number'] > 0, 'asset number')
            require(a['kind'] in {'electrode', 'cno'}, 'asset kind')
            require('electrode_stack' not in a, 'obsolete asset number position')
            groups.setdefault((a['kind'], a['layer']), []).append(a)
        if status == 'resolved':
            require(top['document'] is not None and order['source'] == 'user_supplied_order_file' and order['selected_list'] == 'order' and not order['errors'], 'resolved input')
            present = {a['layer'] for a in assets}
            by_folded_name = {n.casefold(): n for n in present}
            require(len(by_folded_name) == len(present), 'source case collision')
            names = top['document']['order']
            require(set(by_folded_name) <= {n.casefold() for n in names}, 'reference coverage')
            require(order['ignored_names'] == [n for n in names if n.casefold() not in by_folded_name], 'ignored names')
            pairs = [[n, by_folded_name[n.casefold()]] for n in names if n.casefold() in by_folded_name]
            require(order['matched_pairs'] == pairs, 'matched pairs')
            entries = order['resolved_order']
            require(len(entries) == len(pairs), 'resolved coverage')
            for i, (entry, (_, name)) in enumerate(zip(entries, pairs)):
                require(set(entry) == {'order_index', 'kind', 'name', 'asset_ids'}, 'entry fields')
                require(type(entry['order_index']) is int and entry['order_index'] == i and entry['name'] == name, 'entry order')
                matching = [(k, aa) for (k, n), aa in groups.items() if n == name]
                require(len(matching) == 1, 'name kind collision')
                kind, aa = matching[0]
                require(entry['kind'] == kind, 'entry kind')
                require(entry['asset_ids'] == sorted(a['asset_id'] for a in aa) and len(set(entry['asset_ids'])) == len(aa), 'asset coverage')
                require(all(type(a['order_index']) is int and a['order_index'] == i for a in aa), 'asset position')
            value = [[e['kind'], e['name']] for e in entries]
            require(order['resolved_order_sha256'] == hashlib.sha256(canonical_json(value)).hexdigest(), 'semantic digest')
        else:
            require(order['source'] is None and order['selected_list'] is None and order['resolved_order'] == [] and
                    order['ignored_names'] == [] and order['matched_pairs'] == [] and order['resolved_order_sha256'] is None, 'unresolved evidence')
            require(all(a['order_index'] is None for a in assets), 'unresolved asset position')
            if status == 'required':
                require(model['cno_present'] and top['source'] is None and bool(order['errors']), 'required input')
            if status == 'not_required':
                require(not model['cno_present'] and top['source'] is None and not order['errors'], 'not required input')
            if status == 'invalid':
                require(top['source'] is not None and bool(order['errors']), 'invalid input')
        extraction = extractions[model['model_id']]
        require(extraction['algorithm_id'] == ALGORITHM_ID and extraction['algorithm_version'] == ALGORITHM_VERSION,
                'algorithm version')
        require(extraction['same_electrode_required'] is False and extraction['component_neighbourhood'] == 4, 'extraction parameters')
        require(extraction['position_source'] == order['source'], 'position source')
        require(extraction['extractor_run_status'] in {'ran', 'blocked'}, 'extraction status')
        states.append(extraction['extractor_run_status'])
        if status in {'required', 'invalid'}:
            require(extraction['extractor_run_status'] == 'blocked' and extraction['connectivity_observation'] == 'unavailable', 'blocked extraction')
            require(all(c['relation'] is None and c['cnt_order_index'] is None and not c['above_candidates'] and not c['below_candidates'] and
                        not c['selected_upper_candidates'] and not c['selected_lower_candidates'] for c in extraction['components']), 'blocked relations')
        assets_by_id = {a['asset_id']: a for a in assets}
        for component in extraction['components']:
            cnt = assets_by_id[component['source_asset_id']]
            require(cnt['kind'] == 'cno' and component['cnt_layer'] == cnt['layer'] and
                    component['cnt_filename_number'] == cnt['filename_number'], 'component source')
            require(component['cnt_order_index'] == cnt['order_index'], 'component position')
            if status in {'required', 'invalid'}:
                reason = 'stack_order_required' if status == 'required' else 'invalid_stack_order'
                require(component['unresolved_reasons'] == [reason] and not component['ignored_lower_candidates'], 'blocked component')
            for field in ['above_candidates', 'below_candidates', 'selected_upper_candidates', 'selected_lower_candidates']:
                for candidate in component[field]:
                    electrode = assets_by_id[candidate['asset_id']]
                    require(electrode['kind'] == 'electrode' and candidate['order_index'] == electrode['order_index'] and
                            'electrode_stack' not in candidate and candidate['layer'] == electrode['layer'], 'candidate position')
            if component['relation'] is not None:
                require(status == 'resolved', 'relation requires resolved order')
                upper, lower = component['selected_upper_candidates'], component['selected_lower_candidates']
                require(len(upper) == len(lower) == 1, 'relation selected pair')
                for field, low, high in [('skipped_upper_layers', upper[0]['order_index'], cnt['order_index']),
                                         ('skipped_lower_layers', cnt['order_index'], lower[0]['order_index'])]:
                    skipped = component['relation'][field]
                    require(set(skipped) == {'electrode_layers', 'cno_layers'}, 'skipped fields')
                    for kind in ['electrode', 'cno']:
                        expected = []
                        for entry in order['resolved_order']:
                            if entry['kind'] == kind and low < entry['order_index'] < high:
                                expected.append({'order_index': entry['order_index'],
                                                 'layer': entry['name'], 'asset_ids': entry['asset_ids'],
                                                 'source_paths': sorted(a['relative_path'] for a in assets if a['asset_id'] in entry['asset_ids'])})
                        require(skipped[kind + '_layers'] == expected, 'skipped order evidence')
        require(all('cnt_stack' not in c and 'equal_stack_candidates' not in c for c in extraction['components']), 'obsolete component fields')
    require(set(extractions) == {m['model_id'] for m in report['inventory']['models']}, 'extraction coverage')
    expected_status = 'ran' if all(s == 'ran' for s in states) else 'blocked' if all(s == 'blocked' for s in states) else 'partial'
    require(report['extractor_run_status'] == expected_status, 'aggregate run status')
