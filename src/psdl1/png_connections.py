"""Conservative original-resolution extraction; no electrical certification.

4-neighbour components are user accepted; skipped-layer physical semantics
remain an implementation interpretation. Bbox-local extraction does not remove the v1 consumer's O(R*N)
full-frame connection masks, float32 loading or validation overhead.
"""
from dataclasses import dataclass, replace
from io import BytesIO

import numpy as np

from .png_contract import (ExtractedComponent, ExtractedEndpoint, ExtractedRelation,
                           ExtractionSummary, IgnoredCandidate, LayerEvidence, ModelExtraction,
                           OverlapCandidate, ResourceEstimate, ResourceScenario, SkippedLayers,
                           canonical_id, component_id, mask_hash)
from .schema import Connection, Element, Endpoint, Frame, Product, validate_product


@dataclass(frozen=True)
class ComponentBox:
    label: int
    bbox_xyxy_px: tuple[int, int, int, int]
    pixel_count: int
    first_yx: tuple[int, int]


def label_components(mask):
    """One label array; horizontal runs with union/find using edge overlap only.

    Bboxes/counts derive from the runs, without scanning labels once per component.
    Labels are deterministic by first foreground pixel in y/x order.
    """
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("label_components requires a 2-D bool mask")
    h, w = mask.shape
    parent, sizes, runs, previous = [], [], [], []

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            if sizes[a] < sizes[b]:
                a, b = b, a
            parent[b] = a
            sizes[a] += sizes[b]

    for y in range(h):
        edges = np.flatnonzero(np.diff(np.pad(mask[y], (1, 1)).astype(np.int8)))
        current = []
        for x0, x1 in zip(edges[::2].tolist(), edges[1::2].tolist()):
            run_id = len(parent)
            parent.append(run_id); sizes.append(1)
            current.append((x0, x1, run_id))
            runs.append((y, x0, x1, run_id))
        # Half-open runs must share an x pixel: diagonal touching is not overlap.
        j = 0
        for x0, x1, run_id in current:
            while j < len(previous) and previous[j][1] <= x0:
                j += 1
            k = j
            while k < len(previous) and previous[k][0] < x1:
                union(run_id, previous[k][2])
                k += 1
        previous = current

    aggregates = {}
    for y, x0, x1, run_id in runs:
        root = find(run_id)
        if root not in aggregates:
            aggregates[root] = [x0, y, x1, y + 1, x1 - x0, (y, x0)]
        else:
            a = aggregates[root]
            a[0] = min(a[0], x0); a[2] = max(a[2], x1); a[3] = y + 1
            a[4] += x1 - x0
    ordered = sorted(aggregates, key=lambda r: aggregates[r][5])
    mapping = {root: i + 1 for i, root in enumerate(ordered)}
    labels = np.zeros((h, w), dtype=np.int32)
    for y, x0, x1, run_id in runs:
        labels[y, x0:x1] = mapping[find(run_id)]
    boxes = tuple(ComponentBox(mapping[r], tuple(aggregates[r][:4]),
                               aggregates[r][4], aggregates[r][5]) for r in ordered)
    return labels, boxes


def element_id(asset):
    if asset.kind != "electrode":
        raise ValueError("CNT source cannot be an electrode endpoint")
    return canonical_id("element", asset.binding)


def overlap_pixels(component_mask, electrode_slice):
    """Instrumentation boundary: operands are identically sized bbox slices."""
    if component_mask.shape != electrode_slice.shape:
        raise ValueError("candidate slices must share component bbox")
    return int(np.count_nonzero(component_mask & electrode_slice))


def common_intersection(component_mask, upper_slice, lower_slice):
    if component_mask.shape != upper_slice.shape or component_mask.shape != lower_slice.shape:
        raise ValueError("common-intersection slices must share component bbox")
    return component_mask & upper_slice & lower_slice


def representative_pixel(intersection):
    """Exact centroid distance with y/x ties; vectorize only with proven int64 bounds."""
    ys, xs = np.nonzero(intersection)
    count = len(xs)
    if not count:
        raise ValueError("representative pixel requires nonempty intersection")
    maximum = np.iinfo(np.int64).max
    max_y, max_x = int(ys.max()), int(xs.max())
    py_y = None if count * max_y <= maximum else ys.tolist()
    py_x = None if count * max_x <= maximum else xs.tolist()
    sum_y = int(ys.sum(dtype=np.int64)) if py_y is None else sum(py_y)
    sum_x = int(xs.sum(dtype=np.int64)) if py_x is None else sum(py_x)
    mean_y, rem_y = divmod(sum_y, count)
    mean_x, rem_x = divmod(sum_x, count)
    bound_y = max(abs(int(ys.min()) - mean_y), abs(max_y - mean_y))
    bound_x = max(abs(int(xs.min()) - mean_x), abs(max_x - mean_x))
    # Drop the constant centroid remainder squares from all distances and cancel
    # one factor of count. Each intermediate and absolute score is <= bound.
    bound = count * (bound_y ** 2 + bound_x ** 2) + 2 * rem_y * bound_y + 2 * rem_x * bound_x
    if bound <= maximum:
        dy = ys.astype(np.int64) - mean_y
        dx = xs.astype(np.int64) - mean_x
        scores = count * (dy * dy + dx * dx) - 2 * rem_y * dy - 2 * rem_x * dx
        index = int(np.argmin(scores))  # nonzero order is y then x
        return int(xs[index]), int(ys[index])
    # Convert each coordinate once; no tuple list and no repeated NumPy scalar
    # conversion inside the exact key. This is also used beyond the int64 proof.
    best_score = None
    best_x = best_y = 0
    py_y = ys.tolist() if py_y is None else py_y
    py_x = xs.tolist() if py_x is None else py_x
    for y, x in zip(py_y, py_x):
        dy, dx = y - mean_y, x - mean_x
        score = count * (dy * dy + dx * dx) - 2 * rem_y * dy - 2 * rem_x * dx
        if best_score is None or score < best_score:
            best_score, best_x, best_y = score, x, y
    return best_x, best_y


def component_slice(label_window, label):
    """Compare only an already sliced label window; instrument this boundary."""
    return label_window == label


def _skipped(model, low, high):
    def group(kind):
        return tuple(LayerEvidence(e.order_index, e.name, e.asset_ids,
                                   tuple(sorted(a.relative_path for a in model.assets if a.asset_id in e.asset_ids)))
                     for e in model.stack_order.resolved_order if e.kind == kind and low < e.order_index < high)
    return SkippedLayers(group("electrode"), group("cno"))


def _selected_pair_reasons(upper, lower):
    reasons = []
    if not upper:
        reasons.append("missing_upper_overlap")
    elif len(upper) > 1:
        reasons.append("multiple_upper_electrodes_on_selected_layer")
    if not lower:
        reasons.append("missing_lower_overlap")
    elif len(lower) > 1:
        reasons.append("multiple_lower_electrodes_on_selected_layer")
    if len(upper) == len(lower) == 1 and upper[0].element_id == lower[0].element_id:
        reasons.append("same_bound_element")
    return tuple(reasons)


def _extract_component(model, cnt, labels, box, electrode_by_id):
    x0, y0, x1, y1 = box.bbox_xyxy_px
    window = np.s_[y0:y1, x0:x1]
    local = component_slice(labels[window], box.label)
    local.flags.writeable = False
    local_hash = mask_hash(local)
    cid = component_id(model.model_id, cnt.layer, cnt.binding,
                       cnt.shape_yx, box.bbox_xyxy_px, local, local_mask_sha256=local_hash)
    position = model.order_index(cnt)
    if model.cnt_mode == "image_only" or model.stack_order.status != "resolved":
        reason = "cnt_image_only_no_relations" if model.cnt_mode == "image_only" else "invalid_stack_order"
        return ExtractedComponent(cid, cnt.asset_id, cnt.relative_path, None, cnt.filename_number, cnt.layer,
                                  cnt.shape_yx, box.bbox_xyxy_px, box.pixel_count, local_hash, local,
                                  unresolved_reasons=(reason,))
    candidates = []
    for asset in sorted(model.electrodes, key=lambda a: (model.order_index(a) if model.order_index(a) is not None else -1, a.layer, a.electrode)):
        count = overlap_pixels(local, asset.mask[window])
        if count:
            candidates.append(OverlapCandidate(element_id(asset), asset.asset_id, asset.electrode,
                                                model.order_index(asset), asset.layer, asset.relative_path, count))
    above = tuple(c for c in candidates if c.order_index < position)
    below = tuple(c for c in candidates if c.order_index > position)
    upper_position = max((a.order_index for a in above), default=None)
    lower_position = min((a.order_index for a in below), default=None)
    upper = tuple(c for c in above if c.order_index == upper_position)
    lower = tuple(c for c in below if c.order_index == lower_position)
    ignored = tuple(IgnoredCandidate(c) for c in below if lower and c.order_index > lower[0].order_index)
    reasons = _selected_pair_reasons(upper, lower)
    intersection_count = regions = None
    relation = None
    if not reasons:
        joint = common_intersection(local, electrode_by_id[upper[0].element_id].mask[window],
                                    electrode_by_id[lower[0].element_id].mask[window])
        intersection_count = int(np.count_nonzero(joint))
        if not intersection_count:
            regions = 0
            reasons = ("empty_common_intersection",)
        else:
            joint_labels, joint_boxes = label_components(joint)
            regions = len(joint_boxes)
            del joint_labels
            if regions != 1:
                reasons = ("disconnected_common_intersection_possible_multicontact",)
            else:
                x, y = representative_pixel(joint)
                original_xy = (x + x0, y + y0)
                xy_um = model.frame.pixel_centre_um(*original_xy)
                skipped_upper = _skipped(model, upper[0].order_index, position)
                skipped_lower = _skipped(model, position, lower[0].order_index)
                connection_id = canonical_id("connection", [cid, upper[0].element_id,
                                                           lower[0].element_id, original_xy])
                relation = ExtractedRelation(
                    connection_id, cid, (ExtractedEndpoint(upper[0].element_id, xy_um),
                                         ExtractedEndpoint(lower[0].element_id, xy_um)),
                    original_xy, intersection_count, mask_hash(joint), skipped_upper, skipped_lower,
                    skipped_upper.present or skipped_lower.present)
    return ExtractedComponent(cid, cnt.asset_id, cnt.relative_path, position, cnt.filename_number, cnt.layer,
                              cnt.shape_yx, box.bbox_xyxy_px, box.pixel_count, local_hash, local,
                              above, below, upper, lower, ignored, reasons,
                              intersection_count, regions, relation)


def _npy_header_bytes(shape):
    header = BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": np.lib.format.dtype_to_descr(np.dtype(bool)),
                                                "fortran_order": False, "shape": tuple(shape)})
    return header.tell()


def estimate_resources(model, component_count, resolved_count):
    """Actual bool NPY header encoding; estimates exclude other bundle files."""
    if not model.valid:
        return ResourceEstimate(unknown_reasons=("invalid_or_undecodable_source",),
                                all_components_resolved_applicability="not_applicable_image_only" if model.cnt_mode == "image_only" else "relations_only")
    if (type(component_count) is not int or type(resolved_count) is not int or
            not 0 <= resolved_count <= component_count):
        raise ValueError("resource counts require 0 <= resolved <= components")
    h, w = model.frame.shape_yx
    pixels, electrodes, cnt_files = h * w, len(model.electrodes), sum(len(a.provenance_files) or 1 for a in model.cnt_assets)
    # Export a global CNT mask when at least one observed CNT asset exists,
    # including all-white assets; a missing cno has no fabricated observation.
    indicator = int(cnt_files > 0)
    header = _npy_header_bytes((h, w))
    def scenario(relations):
        masks = electrodes + 1 + indicator + relations
        payload = pixels * masks * np.dtype(bool).itemsize
        headers = header * masks
        return ResourceScenario(relations, masks, payload, headers, payload + headers,
                                pixels * masks * np.dtype(np.float32).itemsize)
    if model.cnt_mode == "image_only" and resolved_count != 0:
        raise ValueError("image-only resources cannot contain relations")
    actual, upper = scenario(resolved_count), None if model.cnt_mode == "image_only" else scenario(component_count)
    return ResourceEstimate((h, w), pixels, electrodes, cnt_files, component_count, resolved_count,
                            indicator, actual.generated_mask_asset_count, actual.bool_payload_bytes,
                            actual.npy_header_bytes, actual.projected_bool_npy_export_bytes,
                            actual.projected_load_manifest_retained_float32_mask_bytes, upper, (),
                            all_components_resolved_applicability="not_applicable_image_only" if model.cnt_mode == "image_only" else "relations_only")


def extract_model(model):
    """Inspect without metadata, grids or full-frame per-relation allocations."""
    if not model.valid:
        return ModelExtraction(model.model_id, (), "unavailable", ExtractionSummary(*(0,) * 10, geometry_only_component_count=0 if model.cnt_mode == 'image_only' else None),
                               estimate_resources(model, 0, 0), ("invalid_source",), "blocked",
                               position_source=model.stack_order.source, cnt_mode=model.cnt_mode)
    by_id = {}
    for a in model.electrodes:
        eid = element_id(a)
        if eid in by_id:
            raise ValueError("duplicate/colliding emitted element identity")
        by_id[eid] = a
    components, seen = [], set(by_id)
    for cnt in sorted(model.cnt_assets, key=lambda a: (model.order_index(a) if model.order_index(a) is not None else -1, a.layer, a.mask_sha256)):
        labels, boxes = label_components(cnt.mask)
        for box in boxes:
            component = _extract_component(model, cnt, labels, box, by_id)
            for identity in (component.component_id, component.relation.connection_id if component.relation else None):
                if identity is not None:
                    if identity in seen:
                        raise ValueError("duplicate/colliding emitted component/connection identity")
                    seen.add(identity)
            components.append(component)
        del labels
    components.sort(key=lambda c: (c.cnt_order_index if c.cnt_order_index is not None else -1, c.cnt_layer, c.source_asset_id,
                                   c.bbox_xyxy_px[1], c.bbox_xyxy_px[0], c.component_id))
    relations = [c.relation for c in components if c.relation is not None]
    count, resolved = len(components), len(relations)
    status = "unavailable" if not resolved else "complete" if resolved == count else "partial"
    source_reasons = ()
    if not model.cno_present:
        source_reasons = ("missing_cno_directory",)
    elif not model.cnt_assets:
        source_reasons = ("present_cno_without_png_assets",)
    elif not count:
        source_reasons = ("observed_cnt_empty_foreground",)
    blocked = model.stack_order.status == "invalid"
    if blocked:
        source_reasons += ("invalid_stack_order",)
    elif model.cnt_mode == "image_only":
        source_reasons += ("cnt_image_only_no_relations",)
    summary = ExtractionSummary(
        count, resolved, 0 if model.cnt_mode == 'image_only' else count - resolved, sum(r.needs_confirmation for r in relations),
        sum(r.skipped_upper_layers.present for r in relations),
        sum(r.skipped_lower_layers.present for r in relations),
        sum(len(r.skipped_upper_layers.electrode_layers) for r in relations),
        sum(len(r.skipped_upper_layers.cno_layers) for r in relations),
        sum(len(r.skipped_lower_layers.electrode_layers) for r in relations),
        sum(len(r.skipped_lower_layers.cno_layers) for r in relations),
        geometry_only_component_count=count if model.cnt_mode == 'image_only' else None)
    return ModelExtraction(model.model_id, tuple(components), status, summary,
                           estimate_resources(model, count, resolved), source_reasons,
                           extraction_status="blocked" if blocked else "not_run_image_only" if model.cnt_mode == "image_only" else "complete",
                           position_source=model.stack_order.source, cnt_mode=model.cnt_mode)


def extract_inventory(inventory):
    results = tuple(extract_model(m) for m in inventory.models)
    models = tuple(replace(m, resource_estimate=r.resource_estimate) for m, r in zip(inventory.models, results))
    return replace(inventory, models=models), results


def component_mask(component):
    """Compatibility boundary ONLY: materialize one original-frame bool mask."""
    mask = np.zeros(component.shape_yx, dtype=np.bool_)
    x0, y0, x1, y1 = component.bbox_xyxy_px
    mask[y0:y1, x0:x1] = component.mask
    return mask


def iter_connections(extraction):
    """Export can consume/release one full-frame mask at a time."""
    if extraction.cnt_mode == "image_only" and extraction.relations:
        raise ValueError("image-only products cannot contain relations")
    for component in extraction.components:
        r = component.relation
        if r is not None:
            yield Connection(r.connection_id, "CNT", tuple(Endpoint(e.element_id, e.xy_um) for e in r.endpoints),
                             component_mask(component), "observed")


def to_product(model, extraction, *, metadata=None):
    """Legacy compatibility consumer; retains O(R*N) full-frame masks.

    Callers must surface resource_estimate BEFORE this potentially heavy boundary.
    Metadata stays absent for inspection unless explicitly supplied by the caller.
    """
    if not model.valid or extraction.extraction_status not in {"complete", "not_run_image_only"} or model.model_id != extraction.model_id:
        raise ValueError("valid source and matching completed extraction required")
    if metadata is not None and metadata.model_id != model.model_id:
        raise ValueError("metadata must name the exact source model")
    f = model.frame
    cnt = None
    if model.cnt_assets:
        cnt = np.zeros(f.shape_yx, dtype=np.bool_)
        for a in model.cnt_assets:
            cnt |= a.mask  # preserve unresolved foreground too
    elements = tuple(Element(a.role, a.layer, a.mask, a.electrode, element_id(a), "observed")
                     for a in sorted(model.electrodes, key=lambda a: (model.order_index(a) if model.order_index(a) is not None else -1, a.layer, a.electrode)))
    product = Product(model.model_id, f.frame_extent_um, elements, cnt,
                      metadata.lineage_id if metadata else None, metadata.design_version if metadata else None,
                      Frame(f.frame_id, f.shape_yx, f.pixel_pitch_um, f.origin_xy_um),
                      np.ones(f.shape_yx, dtype=np.bool_), tuple(iter_connections(extraction)),
                      extraction.connectivity_observation)
    validate_product(product)
    return product
