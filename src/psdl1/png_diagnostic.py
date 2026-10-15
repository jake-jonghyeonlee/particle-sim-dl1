"""Inspection-only, mask-major component correspondence and bounded diagnostics.

No geometry/electrical certification is implied. Source labels live for one
observation only. The public iterator yields scalar outcomes in requested grid
order, stopping after its first failed outcome; KeyboardInterrupt propagates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import data
from .schema import binary_mask, validate_product

VERSION = 'dl1_grid_diagnostic_v1'


@dataclass(frozen=True)
class Observation:
    observation_id: str
    scope: str
    mask: np.ndarray | None = field(repr=False, compare=False)
    legacy_id: str | None = None
    layer: str | None = None
    role: str | None = None
    source_asset_ids: tuple[str, ...] = ()
    source_paths: tuple[str, ...] = ()
    component_id: str | None = None

    def identity(self):
        return {k: getattr(self, k) for k in ('observation_id', 'scope', 'legacy_id',
                'layer', 'role', 'source_asset_ids', 'source_paths', 'component_id')}


def observation_descriptors(product, model=None, extraction=None):
    """Bind Product order to live source provenance without materializing masks."""
    from .png_connections import element_id
    assets = {} if model is None else {element_id(a): a for a in model.electrodes}
    components = {} if extraction is None else {c.relation.connection_id: c
                  for c in extraction.components if c.relation is not None}
    cnt = () if model is None else model.cnt_assets
    def provenance(aa):
        return tuple(p['relative_path'] for a in aa for p in
                     (a.provenance_files or ({'relative_path': a.relative_path},)))
    for index, e in enumerate(product.elements):
        a = assets.get(e.key)
        yield Observation(f'o{index + 1}', 'electrode', e.mask, e.key, e.layer, e.role,
                          () if a is None else (a.asset_id,), () if a is None else provenance((a,)))
    offset = len(product.elements)
    for index, c in enumerate(product.connections):
        comp = components.get(c.connection_id)
        yield Observation(f'o{offset + index + 1}', 'connection', c.mask, c.connection_id,
                          None if comp is None else comp.cnt_layer, None,
                          () if comp is None else (comp.source_asset_id,),
                          () if comp is None else (comp.source_path,),
                          None if comp is None else comp.component_id)
    offset += len(product.connections)
    if product.cnt is not None:
        offset += 1
        yield Observation(f'o{offset}', 'global_cnt_union', product.cnt, 'global_CNT',
                          source_asset_ids=tuple(a.asset_id for a in cnt), source_paths=provenance(cnt))
    for index, a in enumerate(cnt):
        yield Observation(f'o{offset + index + 1}', 'cnt_source', a.mask, layer=a.layer,
                          source_asset_ids=(a.asset_id,), source_paths=provenance((a,)))


def _runs(mask):
    """Numeric (row, start, length) arrays in lexicographic order."""
    edges = np.diff(np.pad(mask, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    row, start = np.nonzero(edges == 1)
    _, end = np.nonzero(edges == -1)
    return np.column_stack((row, start, end - start)).astype(np.int32)


@dataclass
class Labels:
    image: np.ndarray
    count: int
    bounds: np.ndarray  # component index 1..count, half-open xyxy
    first_yx: np.ndarray


def label_components(mask, *, runs=None):
    """Vectorized horizontal-run graph with minimum-root hooking/compression.

    Python iterates convergence rounds, never pixels or components. IDs follow
    first foreground y/x order. No scipy or extraction-labeler changes.
    """
    m = np.asarray(mask, dtype=bool)
    starts = m & ~np.pad(m[:, :-1], ((0, 0), (1, 0)))
    ids = np.cumsum(starts.ravel(), dtype=np.int32).reshape(m.shape)
    ids[~m] = 0
    n = int(starts.sum())
    parent = np.arange(n + 1, dtype=np.int32)
    a, b = ids[:-1], ids[1:]
    active = (a != 0) & (b != 0) & (a != b)
    keys = np.unique(a[active].astype(np.int64) * (n + 1) + b[active])
    pairs = np.column_stack((keys // (n + 1), keys % (n + 1)))
    while pairs.size:
        a, b = parent[pairs[:, 0]], parent[pairs[:, 1]]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        if np.array_equal(lo, hi):
            break
        np.minimum.at(parent, hi, lo)
        while not np.array_equal(parent, parent[parent]):
            parent = parent[parent]
    roots = np.unique(parent[1:])
    mapping = np.zeros(n + 1, dtype=np.int32)
    mapping[roots] = np.arange(1, len(roots) + 1)
    labels = mapping[parent][ids]
    count = len(roots)
    runs = _runs(m) if runs is None else runs
    yy, xx, lengths = runs.T
    cc = labels[yy, xx]
    bounds = np.zeros((count + 1, 4), dtype=np.int32)
    bounds[:, :2] = np.iinfo(np.int32).max
    np.minimum.at(bounds[:, 0], cc, xx)
    np.minimum.at(bounds[:, 1], cc, yy)
    np.maximum.at(bounds[:, 2], cc, xx + lengths)
    np.maximum.at(bounds[:, 3], cc, yy + 1)
    first = np.full(count + 1, m.size, dtype=np.int64)
    np.minimum.at(first, cc, yy * m.shape[1] + xx)
    first[0] = 0
    return Labels(labels, count, bounds, np.column_stack(np.unravel_index(first, m.shape)))


@dataclass
class SourceMeasurement:
    source: np.ndarray
    labels: Labels
    horizontal: np.ndarray
    vertical: np.ndarray
    narrowest: int
    active: int

    @property
    def working_bytes(self):
        return sum(a.nbytes for a in (self.source, self.labels.image, self.labels.bounds,
                   self.labels.first_yx, self.horizontal, self.vertical))


def measure_source(mask, valid):
    source = binary_mask(mask) * valid
    horizontal, vertical = _runs(source.astype(bool)), _runs(source.T.astype(bool))
    width = min(int(horizontal[:, 2].min(initial=max(source.shape) + 1)),
                int(vertical[:, 2].min(initial=max(source.shape) + 1))) if source.any() else 0
    return SourceMeasurement(source, label_components(source, runs=horizontal), horizontal, vertical,
                             width, int(source.sum()))


def yellow_cells(measured, transform):
    """Lossless target-cell totals of all qualifying oriented maximal runs."""
    g = transform.grid
    counts = np.zeros((g * g, 2), dtype=np.int64)
    minimum = np.full(g * g, np.iinfo(np.int32).max, dtype=np.int32)
    for orientation, runs in enumerate((measured.horizontal, measured.vertical)):
        rows, starts, lengths = runs.T
        keep = lengths * (g / max(transform.shape_yx)) < 2
        rows, starts, lengths = rows[keep], starts[keep], lengths[keep]
        middle = starts + (lengths - 1) / 2
        x, y = (middle, rows) if orientation == 0 else (rows, middle)
        gx, gy = transform.pixel_to_grid(x, y)
        cells = np.floor(gy).astype(np.int64) * g + np.floor(gx).astype(np.int64)
        np.add.at(counts[:, orientation], cells, 1)
        np.minimum.at(minimum, cells, lengths)
    cells = np.flatnonzero(counts.sum(axis=1))
    for cell in cells:
        yield {'cell_xy': [int(cell % g), int(cell // g)],
               'run_count': int(counts[cell].sum()), 'min_length_px': int(minimum[cell]),
               'horizontal_runs': int(counts[cell, 0]), 'vertical_runs': int(counts[cell, 1])}


def correspondence(source_labels, target_labels, transform):
    """Unique edges and real witnesses through exact integer adaptive bins.

    Each source pixel visits <=2 bins per axis for downsampling; upsampling
    enumerates all actual support bins, including overlapping adaptive bins.
    """
    h, w = transform.shape_yx
    nh, nw = transform.resized_shape_yx
    left, top, _, _ = transform.padding_ltrb
    y, x = (coordinate.astype(np.int32) for coordinate in np.nonzero(source_labels.image))
    sl = source_labels.image[y, x]
    x0=(x.astype(np.int64)*nw//w).astype(np.int32)
    x1=(((x.astype(np.int64)+1)*nw+w-1)//w).astype(np.int32)
    y0=(y.astype(np.int64)*nh//h).astype(np.int32)
    y1=(((y.astype(np.int64)+1)*nh+h-1)//h).astype(np.int32)
    base = target_labels.count + 1
    def supports():
        for dy in range(int((y1-y0).max(initial=0))):
            for dx in range(int((x1-x0).max(initial=0))):
                ii=np.flatnonzero((x0+dx<x1)&(y0+dy<y1))
                tx,ty=x0[ii]+dx+left,y0[ii]+dy+top
                yield ii,tx,ty,target_labels.image[ty,tx]
    supported=np.zeros(len(x),dtype=bool)
    for ii,tx,ty,tt in supports():supported[ii[tt!=0]]=True
    lost=~supported
    lost_counts=np.bincount(sl[lost],minlength=source_labels.count+1)
    # Prefer a genuinely lost region far from retained material, rather than
    # ordinary edge erosion. Exact separable city-block distance on target cells.
    distance=np.where(target_labels.image!=0,0,2*transform.grid).astype(np.int32)
    for axis in (1,0):
        index=np.arange(transform.grid,dtype=np.int32)
        if axis==0:index=index[:,None]
        distance=np.minimum.accumulate(distance-index,axis=axis)+index
        distance=np.flip(np.minimum.accumulate(np.flip(distance+index,axis=axis),axis=axis),axis=axis)-index
    mx=(((2*x.astype(np.int64)+1)*nw//(2*w)+left).clip(0,transform.grid-1)).astype(np.int32)
    my=(((2*y.astype(np.int64)+1)*nh//(2*h)+top).clip(0,transform.grid-1)).astype(np.int32)
    score=distance[my,mx]
    maximum=np.full(source_labels.count+1,-1,dtype=np.int32)
    np.maximum.at(maximum,sl[lost],score[lost])
    representative=np.full(source_labels.count+1,len(x),dtype=np.int64)
    eligible=lost&(score==maximum[sl])
    np.minimum.at(representative,sl[eligible],np.flatnonzero(eligible))
    lost_xy=np.full((source_labels.count+1,2),-1,dtype=np.int32)
    cc=np.flatnonzero(lost_counts)
    lost_xy[cc]=np.column_stack((x[representative[cc]],y[representative[cc]]))
    chunks=[]
    for ii,tx,ty,tt in supports():
        keep=tt!=0;ii,tx,ty,tt=ii[keep],tx[keep],ty[keep],tt[keep]
        ss=sl[ii];key=ss.astype(np.int64)*base+tt
        delta_x=x[ii].astype(np.int64)-lost_xy[ss,0]
        delta_y=y[ii].astype(np.int64)-lost_xy[ss,1]
        proximity=np.where(lost_counts[ss]>0,delta_x*delta_x+delta_y*delta_y,0)
        # ii is row-major, so stable key grouping already supplies the y/x
        # tie-break; tx/ty are fixed for each pixel in this support pass.
        order=np.argsort(key,kind='stable')
        if len(order):
            sorted_key=key[order]
            starts=np.r_[0,np.flatnonzero(sorted_key[1:]!=sorted_key[:-1])+1]
            minimum=np.minimum.reduceat(proximity[order],starts)
            sizes=np.diff(np.r_[starts,len(order)])
            candidates=np.where(proximity[order]==np.repeat(minimum,sizes),order,len(order))
            chosen=np.minimum.reduceat(candidates,starts)
        else:
            chosen=order
        chunks.append(np.column_stack((ss[chosen],tt[chosen],x[ii[chosen]],y[ii[chosen]],
                                       tx[chosen],ty[chosen],proximity[chosen])))
    if not chunks or not any(len(chunk) for chunk in chunks):
        return np.empty((0,6),dtype=np.int32),lost_counts,lost_xy
    edges=np.concatenate(chunks)
    order=np.lexsort((edges[:,4],edges[:,5],edges[:,2],edges[:,3],edges[:,6],edges[:,1],edges[:,0]))
    edges=edges[order];keys=edges[:,0]*base+edges[:,1]
    return edges[np.r_[True,keys[1:]!=keys[:-1]],:6].astype(np.int32),lost_counts,lost_xy


@dataclass
class GridMeasurement:
    coverage: np.ndarray
    target: Labels
    edges: np.ndarray
    transform: data.FrameTransform
    lost_counts: np.ndarray
    lost_xy: np.ndarray


def measure_grid(measured, grid):
    transform = data.frame_transform(measured.source.shape, grid)
    coverage = data.normalize_unit(measured.source, grid)
    target = label_components(coverage >= .5)
    edges, lost_counts, lost_xy = correspondence(measured.labels, target, transform)
    if len(np.unique(edges[:, 1])) != target.count:
        raise RuntimeError('target component has no source support')
    return GridMeasurement(coverage, target, edges, transform, lost_counts, lost_xy)


def iter_events(observation, measured, result):
    """Yield event + its sites; output-sensitive, no per-component frame scans."""
    edges, target, transform = result.edges, result.target, result.transform
    source = measured.labels
    sd = np.bincount(edges[:, 0], minlength=source.count + 1)
    td = np.bincount(edges[:, 1], minlength=target.count + 1)
    # Group edge indices once; do not scan all edges for each event.
    so = np.argsort(edges[:, 0], kind='stable')
    to = np.argsort(edges[:, 1], kind='stable')
    sp, tp = np.r_[0, np.cumsum(sd)], np.r_[0, np.cumsum(td)]
    for kind, ids in (('disappearance', np.flatnonzero(sd[1:] == 0) + 1),
                      ('split', np.flatnonzero(sd[1:] >= 2) + 1),
                      ('merge_suspected', np.flatnonzero(td[1:] >= 2) + 1)):
        for component in ids:
            if kind == 'disappearance':
                y, x = source.first_yx[component]
                selected = []
                sites = [{'source_xy': [int(x), int(y)], 'target_xy': None,
                          'grid_xy': list(transform.pixel_to_grid(int(x), int(y))), 'lost_source_xy': [int(x),int(y)]}]
                sources, targets = [int(component)], []
            else:
                order, offsets = (so, sp) if kind == 'split' else (to, tp)
                selected = edges[order[offsets[component]:offsets[component + 1]]]
                sources, targets = np.unique(selected[:, 0]).tolist(), np.unique(selected[:, 1]).tolist()
                sites = [{'source_xy': [int(e[2]), int(e[3])], 'target_xy': [int(e[4]), int(e[5])],
                          'grid_xy': [float(e[4] + .5), float(e[5] + .5)],
                          'lost_source_xy': result.lost_xy[e[0]].tolist() if result.lost_counts[e[0]] else None} for e in selected]
            event_id = f'{observation.observation_id}:{kind}:{component}'
            yield ({'event_id': event_id, 'kind': kind, 'source_components': sources,
                    'target_components': targets, 'source_bounds': source.bounds[sources].tolist(),
                    'target_bounds': target.bounds[targets].tolist(),
                    'composite_ambiguity_possible': True,
                    'lost_source_counts': result.lost_counts[sources].tolist()}, sites)


class NullSink:
    """Kernel-only scalar accumulator, with the same complete analysis."""
    def __init__(self):
        self.counts = {k: 0 for k in ('observations', 'thin_runs', 'thin_cells',
                      'disappearance', 'split', 'merge_suspected', 'sites', 'clusters')}
        self.clusters = np.zeros((12, 12), dtype=bool)
        self.incomplete = False
        self.eligible = 0
        self.inset_observations = 0

    def observation(self, observation, measured, result):
        self.counts['observations'] += 1
        if measured is None:
            self.incomplete = True
            return
        for record in yellow_cells(measured, result.transform):
            self.counts['thin_runs'] += record['run_count']
            self.counts['thin_cells'] += 1
        eligible = False
        for event, sites in iter_events(observation, measured, result):
            eligible = True
            self.eligible += 1
            self.counts[event['kind']] += 1
            self.counts['sites'] += len(sites)
            for site in sites:
                x, y = np.asarray(site['grid_xy']) * 12 / result.transform.grid
                self.clusters[min(11, int(y)), min(11, int(x))] = True
        self.inset_observations += eligible

    def finish(self, check):
        self.counts['clusters'] = int(self.clusters.sum())
        incomplete = self.incomplete or 'missing_observations_or_valid_holes' in check['reasons']
        state = ('observation_incomplete' if incomplete else
                 ('change_detected' if self.counts['sites'] else 'no_change_detected'))
        shown = min(3, self.eligible)
        return {'diagnostic_schema_version': VERSION, 'state': 'complete',
                'shape_state': state, 'counts': dict(self.counts),
                'eligible_insets': self.eligible, 'insets_shown': shown,
                'unshown_insets': self.eligible - shown, 'all_sites_included': True}

    def close(self):
        self.clusters = None


def inspect_grid_diagnostics(product, source_descriptors=None, grids=(), sink_factory: Callable | None = None, *,
                             propagate_exceptions=()):
    """Mask-major computation; ordered scalar success/failure outcomes (A4).

    source_descriptors, when supplied, must include Product observations in their
    legacy order followed by diagnostic-only supplements. sink_factory(grid,
    transform, valid) returns an observation/finish/close sink. Errors propagate
    from source validation/measurement. Per-grid Exception failures become one
    failed outcome and stop emission; BaseException always propagates. Callers
    requiring special progress errors to propagate should pass them via
    propagate_exceptions. Sink lifetime is tied to this generator's close.
    """
    shape, valid = validate_product(product)
    descriptors = observation_descriptors(product) if source_descriptors is None else source_descriptors
    grids = tuple(dict.fromkeys(grids))
    if not grids:
        return
    states = []
    pitch = product.frame.pixel_pitch_um if product.frame else product.unit_size_um[0] / shape[1]
    try:
        for grid in grids:
            try:
                transform = data.frame_transform(shape, grid, pitch)
                sink = NullSink() if sink_factory is None else sink_factory(grid, transform, valid)
                states.append({'grid': grid, 'sink': sink, 'evidence': [], 'error': None})
            except Exception as exc:
                if isinstance(exc, propagate_exceptions):
                    raise
                states.append({'grid': grid, 'sink': None, 'evidence': [],
                               'error': (type(exc).__name__, str(exc))})
        for observation in descriptors:
            if observation.mask is not None and np.shape(observation.mask) != shape:
                raise ValueError('observation mask shape differs from Product frame')
            measured = None if observation.mask is None else measure_source(observation.mask, valid)
            for state in states:
                result = None
                if state['error'] is not None:
                    continue
                try:
                    grid = state['grid']
                    result = None if measured is None else measure_grid(measured, grid)
                    if observation.legacy_id is not None:
                        evidence = {'id': observation.legacy_id, 'observation': 'missing'}
                        if measured is not None:
                            scale = grid / max(shape)
                            evidence = {'id': observation.legacy_id, 'observation': 'observed',
                                'axis_run_px': measured.narrowest,
                                'estimated_run_at_grid_px': measured.narrowest * scale,
                                'thin_line_warning': bool(measured.narrowest and measured.narrowest * scale < 2),
                                **data._component_evidence(measured.labels.count, measured.active,
                                  result.target.count, int((result.coverage >= .5).sum()))}
                        state['evidence'].append(evidence)
                    state['sink'].observation(observation, measured, result)
                except Exception as exc:
                    if isinstance(exc, propagate_exceptions):
                        raise
                    state['error'] = (type(exc).__name__, str(exc))
                finally:
                    result = None
            del measured
        for state in states:
            grid = state['grid']
            if state['error'] is None:
                try:
                    check = data._assemble_grid_check(product, shape, valid, grid, state['evidence'])
                    diagnostic = state['sink'].finish(check)
                except Exception as exc:
                    if isinstance(exc, propagate_exceptions):
                        raise
                    state['error'] = (type(exc).__name__, str(exc))
            if state['error'] is not None:
                error_type, reason = state['error']
                yield {'grid': grid, 'state': 'failed', 'error_type': error_type, 'reason': reason}
                break
            yield {'grid': grid, 'state': 'complete', 'grid_check': check, 'diagnostic': diagnostic}
    finally:
        for state in states:
            if state['sink'] is not None:
                state['sink'].close()
        # The first sink may own a directory still occupied by a later sink.
        for state in states:
            cleanup = getattr(state['sink'], 'cleanup_directories', None)
            if cleanup is not None:
                cleanup()


COUNT_KEYS = ('observations', 'thin_runs', 'thin_cells', 'disappearance', 'split',
              'merge_suspected', 'sites', 'clusters')
META_KEYS = {'diagnostic_schema_version', 'state', 'shape_state', 'counts', 'insets_shown',
             'eligible_insets', 'unshown_insets', 'all_sites_included'}
ARTIFACT_KEYS = {'png', 'evidence', 'png_sha256', 'evidence_sha256'}
RECORD_FIELDS = {
    'header': {'type', 'diagnostic_schema_version', 'model_id', 'grid', 'transform', 'threshold',
               'connectivity', 'scope', 'cnt_mode', 'valid_holes', 'display'},
    'observation': {'type', 'observation_id', 'scope', 'legacy_id', 'layer', 'role', 'source_asset_ids',
                    'source_paths', 'component_id', 'state', 'source_components', 'target_components',
                    'active_pixels', 'axis_run_px'},
    'yellow_cell': {'type', 'observation_id', 'cell_xy', 'run_count', 'min_length_px',
                    'horizontal_runs', 'vertical_runs'},
    'event': {'type', 'observation_id', 'event_id', 'kind', 'site_count', 'source_components',
              'target_components', 'source_bounds', 'target_bounds', 'composite_ambiguity_possible', 'lost_source_counts'},
    'site': {'type', 'observation_id', 'event_id', 'site_index', 'cluster_id', 'source_xy',
             'target_xy', 'grid_xy', 'lost_source_xy'},
    'cluster': {'type', 'cluster_id', 'site_count', 'display_bounds_xyxy'},
    'summary': {'type', 'legacy_status'} | META_KEYS,
}


def _require(condition, message='invalid diagnostic evidence'):
    if not condition:
        raise ValueError(message)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def validate_diagnostic_metadata(value, row=None):
    """Exact nested v1 contract; old v3 rows with no diagnostic remain valid."""
    _require(isinstance(value, dict) and value.get('diagnostic_schema_version') == VERSION,
             'unsupported diagnostic version')
    if value.get('state') == 'not_evaluated':
        _require(set(value) == {'diagnostic_schema_version', 'state', 'reason', 'requested_grids'})
        _require(isinstance(value['reason'], str) and bool(value['reason']))
        gg = value['requested_grids']
        _require(isinstance(gg, list) and bool(gg) and all(_integer(g, 1) for g in gg)
                 and len(set(gg)) == len(gg))
        if row is not None:
            _require(row.get('purpose') in {'inspect_not_evaluated', 'inspect_failed_not_approved'})
        return
    _require(value.get('state') == 'complete' and set(value) == META_KEYS | ARTIFACT_KEYS)
    _require(value['shape_state'] in {'change_detected', 'no_change_detected', 'observation_incomplete'})
    cc = value['counts']
    _require(isinstance(cc, dict) and set(cc) == set(COUNT_KEYS) and all(_integer(v) for v in cc.values()))
    _require(value['all_sites_included'] is True)
    _require(all(_integer(value[k]) for k in ('insets_shown', 'eligible_insets', 'unshown_insets')))
    eligible = sum(cc[k] for k in ('disappearance', 'split', 'merge_suspected'))
    _require(value['eligible_insets'] == eligible and value['insets_shown'] == min(3, eligible)
             and value['unshown_insets'] == eligible - value['insets_shown'])
    _require(cc['clusters'] <= min(144, cc['sites']) and cc['thin_cells'] <= cc['thin_runs'])
    for key in ('png', 'evidence'):
        _require(isinstance(value[key], str) and bool(value[key]))
    for key in ('png_sha256', 'evidence_sha256'):
        _require(isinstance(value[key], str) and len(value[key]) == 64
                 and all(ch in '0123456789abcdef' for ch in value[key]))
    if row is not None:
        _require(row.get('purpose') == 'inspect_only_not_approved')


def _json_line(line):
    import json
    def pairs(items):
        result = {}
        for k, v in items:
            _require(k not in result, 'duplicate diagnostic JSON field')
            result[k] = v
        return result
    def invalid(value):
        raise ValueError('nonfinite diagnostic JSON')
    return json.loads(line, object_pairs_hook=pairs, parse_constant=invalid)


def _coordinate(value, width, height, integer=False):
    _require(isinstance(value, list) and len(value) == 2)
    _require(all((type(x) in (int, float) and np.isfinite(x)) for x in value))
    if integer:
        _require(all(type(x) is int for x in value))
    _require(0 <= value[0] < width and 0 <= value[1] < height, 'diagnostic coordinate out of bounds')


def _content_coordinate(value,transform):
    left,top,_,_=transform['padding_ltrb'];nh,nw=transform['resized_shape_yx']
    _require(left<=value[0]<left+nw and top<=value[1]<top+nh,'diagnostic coordinate in padding')


def _bounds(values, ids, width, height):
    _require(isinstance(values, list) and len(values) == len(ids))
    for box in values:
        _require(isinstance(box, list) and len(box) == 4 and all(_integer(v) for v in box))
        _require(0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)


def verify_diagnostic_ledger(path, metadata, row, cnt_mode):
    """Stream validation with one current observation/event and 144 bins only.

    No global observation/event/site sets or source mask decoding. Per-event
    contributor/bounds arrays are released on the next event; records are not
    accumulated in the import report.
    """
    counts = dict.fromkeys(COUNT_KEYS, 0)
    clusters = np.zeros(144, dtype=np.int64)
    bounds = np.zeros((144, 4), dtype=float)
    bounds[:, :2] = np.inf; bounds[:, 2:] = -np.inf
    observation = event = None
    remaining = 0
    yellow_cell = -1
    event_order = (-1, -1)
    cluster_id = -1
    phase = 'observations'
    summary = None
    transform = row['transform']
    grid = transform['grid']; h, w = transform['source_shape_yx']
    with path.open(encoding='utf-8') as handle:
        header = _json_line(next(handle, 'null'))
        _require(isinstance(header, dict) and set(header) == RECORD_FIELDS['header'])
        _require(header['type'] == 'header' and header['diagnostic_schema_version'] == VERSION)
        _require(header['model_id'] == row['model_id'] and _integer(header['grid'],1) and header['grid'] == grid
                 and header['transform'] == transform, 'diagnostic header identity/transform mismatch')
        _require(header['threshold'] == .5 and header['connectivity'] == 4
                 and header['scope'] == 'logical_observations' and header['cnt_mode'] == cnt_mode
                 and type(header['valid_holes']) is bool)
        _require(header['display'] == {'size_xy': [1100,1040], 'origin_xy': [20,84], 'viewport': 768,
                 'cluster_bin_px': 64, 'base_reduction': 'nearest', 'yellow_reduction': 'OR'})
        incomplete = header['valid_holes']
        for line in handle:
            record = _json_line(line)
            _require(isinstance(record, dict) and record.get('type') in RECORD_FIELDS)
            kind = record['type']
            _require(set(record) == RECORD_FIELDS[kind], 'unknown/missing diagnostic record fields')
            _require(summary is None and kind != 'header', 'diagnostic record after summary or duplicate header')
            if remaining:
                _require(kind == 'site', 'missing event sites')
            if kind == 'observation':
                _require(phase == 'observations')
                counts['observations'] += 1
                _require(record['observation_id'] == f"o{counts['observations']}")
                _require(record['scope'] in {'electrode','connection','global_cnt_union','cnt_source'})
                for key in ('legacy_id','layer','role','component_id'):
                    _require(record[key] is None or isinstance(record[key], str))
                for key in ('source_asset_ids','source_paths'):
                    _require(isinstance(record[key], list) and all(isinstance(s, str) for s in record[key]))
                _require(record['state'] in {'observed','observed_empty','missing'})
                scalars = [record[k] for k in ('source_components','target_components','active_pixels','axis_run_px')]
                if record['state'] == 'missing':
                    _require(all(x is None for x in scalars)); incomplete = True
                else:
                    _require(all(_integer(x) for x in scalars))
                    _require(record['active_pixels'] <= h*w and record['source_components'] <= record['active_pixels']
                             and record['target_components'] <= grid*grid and record['axis_run_px'] <= max(h,w))
                    _require((record['state'] == 'observed_empty') == (record['active_pixels'] == 0))
                    if record['state'] == 'observed_empty': _require(all(x == 0 for x in scalars))
                observation = record; event = None; yellow_cell = -1; event_order = (-1,-1)
            elif kind in {'yellow_cell','event','site'}:
                _require(phase == 'observations' and observation is not None
                         and record['observation_id'] == observation['observation_id']
                         and observation['state'] == 'observed')
                if kind == 'yellow_cell':
                    _require(event is None)
                    _coordinate(record['cell_xy'], grid, grid, True)
                    _content_coordinate(record['cell_xy'],transform)
                    _require(record['min_length_px']>=observation['axis_run_px'])
                    cell = record['cell_xy'][1]*grid + record['cell_xy'][0]
                    _require(cell > yellow_cell); yellow_cell = cell
                    _require(_integer(record['run_count'],1) and _integer(record['min_length_px'],1)
                             and record['min_length_px']*grid/max(h,w) < 2)
                    _require(_integer(record['horizontal_runs']) and _integer(record['vertical_runs'])
                             and record['horizontal_runs']+record['vertical_runs'] == record['run_count'])
                    counts['thin_cells'] += 1; counts['thin_runs'] += record['run_count']
                elif kind == 'event':
                    event = record
                    _require(event['kind'] in ('disappearance','split','merge_suspected'))
                    _require(event['composite_ambiguity_possible'] is True and _integer(event['site_count'],1))
                    sources, targets = event['source_components'], event['target_components']
                    for values, limit in ((sources,observation['source_components']), (targets,observation['target_components'])):
                        _require(isinstance(values,list) and all(_integer(v,1) and v <= limit for v in values)
                                 and values == sorted(set(values)))
                    _require(isinstance(event['lost_source_counts'],list) and len(event['lost_source_counts'])==len(sources)
                             and all(_integer(n) and n<=observation['active_pixels'] for n in event['lost_source_counts']))
                    _bounds(event['source_bounds'],sources,w,h); _bounds(event['target_bounds'],targets,grid,grid)
                    if event['kind'] == 'disappearance':
                        _require(len(sources)==1 and targets==[] and event['site_count']==1)
                        rank, component = 0,sources[0]
                    elif event['kind'] == 'split':
                        _require(len(sources)==1 and len(targets)>=2 and event['site_count']==len(targets))
                        rank, component = 1,sources[0]
                    else:
                        _require(len(targets)==1 and len(sources)>=2 and event['site_count']==len(sources))
                        rank, component = 2,targets[0]
                    _require((rank,component)>event_order); event_order = (rank,component)
                    _require(event['event_id'] == f"{observation['observation_id']}:{event['kind']}:{component}")
                    counts[event['kind']] += 1; remaining = event['site_count']
                else:
                    _require(event is not None and remaining>0 and _integer(record['site_index']) and record['event_id']==event['event_id']
                             and record['site_index']==event['site_count']-remaining)
                    _coordinate(record['source_xy'],w,h,True); _coordinate(record['grid_xy'],grid,grid)
                    sx,sy = record['source_xy']
                    source_box=event['source_bounds'][record['site_index'] if event['kind']=='merge_suspected' else 0]
                    _require(source_box[0]<=sx<source_box[2] and source_box[1]<=sy<source_box[3], 'diagnostic source witness outside component bounds')
                    if record['lost_source_xy'] is not None:
                        _coordinate(record['lost_source_xy'],w,h,True)
                        lx,ly=record['lost_source_xy']
                        _require(source_box[0]<=lx<source_box[2] and source_box[1]<=ly<source_box[3])
                    if event['kind']=='disappearance':
                        _require(record['target_xy'] is None)
                        nh,nw = transform['resized_shape_yx']; left,top,_,_ = transform['padding_ltrb']
                        _require(np.allclose(record['grid_xy'],[(sx+.5)*nw/w+left,(sy+.5)*nh/h+top],rtol=0,atol=1e-10))
                    else:
                        _coordinate(record['target_xy'],grid,grid,True)
                        _content_coordinate(record['target_xy'],transform)
                        tx,ty = record['target_xy']
                        _require(record['grid_xy']==[tx+.5,ty+.5], 'diagnostic target witness centre mismatch')
                        target_box=event['target_bounds'][record['site_index'] if event['kind']=='split' else 0]
                        _require(target_box[0]<=tx<target_box[2] and target_box[1]<=ty<target_box[3])
                    px,py = np.asarray(record['grid_xy'])*768/grid
                    cid = min(11,int(py//64))*12+min(11,int(px//64))
                    _require(_integer(record['cluster_id']) and record['cluster_id']==cid)
                    clusters[cid] += 1
                    bounds[cid,:2] = np.minimum(bounds[cid,:2],(px,py))
                    bounds[cid,2:] = np.maximum(bounds[cid,2:],(px,py))
                    counts['sites'] += 1; remaining -= 1
            elif kind == 'cluster':
                phase = 'clusters'; cid = record['cluster_id']
                _require(_integer(cid) and cluster_id < cid < 144 and clusters[cid]>0)
                _require(_integer(record['site_count'],1) and record['site_count']==int(clusters[cid]) and
                         isinstance(record['display_bounds_xyxy'],list) and len(record['display_bounds_xyxy'])==4)
                _require(np.isfinite(record['display_bounds_xyxy']).all() and
                         np.allclose(record['display_bounds_xyxy'],bounds[cid],rtol=0,atol=1e-9))
                cluster_id = cid; counts['clusters'] += 1
            elif kind == 'summary':
                _require(remaining==0 and counts['clusters']==int(np.count_nonzero(clusters)))
                validate_diagnostic_metadata({**{k:record[k] for k in META_KEYS},
                                              **{k:metadata[k] for k in ARTIFACT_KEYS}})
                _require(record['legacy_status']==row['grid_check']['status'])
                _require({k:record[k] for k in META_KEYS} == {k:metadata[k] for k in META_KEYS})
                _require(record['counts']==counts, 'diagnostic summary counts mismatch')
                state = 'observation_incomplete' if incomplete else ('change_detected' if counts['sites'] else 'no_change_detected')
                _require(record['shape_state']==state)
                summary = record
        _require(summary is not None, 'missing diagnostic summary')


def verify_diagnostic_artifacts(root, report, files, resolve):
    """Bind exact namespace/hashes, bounded PNG header and streamed ledger."""
    import struct
    import zlib
    from pathlib import PurePosixPath
    from .png_contract import canonical_id
    inspect_rows=[row for row in report.get('grid_evidence',[]) if row.get('purpose')=='inspect_only_not_approved']
    _require(not any('diagnostic' in row for row in inspect_rows) or
             all('diagnostic' in row for row in inspect_rows),'mixed diagnostic generations')
    models={model['model_id']:model for model in report['inventory']['models']}
    references = set()  # two scalar paths per grid/report row, never site IDs
    for row in report.get('grid_evidence', []):
        if 'diagnostic' not in row:
            continue
        metadata = row['diagnostic']
        validate_diagnostic_metadata(metadata,row)
        if metadata['state'] != 'complete':
            continue
        model=models.get(row['model_id'])
        _require(model is not None and model['frame'] is not None, 'unknown diagnostic model')
        _require(metadata['counts']['observations']==len(row['grid_check']['elements'])+
                 sum(a['kind']=='cno' for a in model['assets']), 'diagnostic observation count mismatch')
        _require(row['transform']['source_shape_yx']==model['frame']['shape_yx'] and
                 row['transform']['original_pixel_pitch_um']==model['frame']['pixel_pitch_um'])
        grid = row['grid_check']['grid']
        _require(row['grid_check']['product_id']==row['model_id'] and row['transform']==row['grid_check']['transform'])
        transform = row['transform']
        expected_transform = data.frame_transform(tuple(transform['source_shape_yx']),grid,
                                                   transform['original_pixel_pitch_um']).to_record()
        _require(transform==expected_transform, 'invalid diagnostic transform')
        prefix = f"diagnostics/{canonical_id('model',[row['model_id']])}/grid-{grid}"
        for field,suffix in (('png','.png'),('evidence','.jsonl')):
            relative = metadata[field]
            _require(relative==prefix+suffix and relative not in references, 'invalid/duplicate diagnostic namespace')
            path = resolve(root,relative)
            _require(path.is_file() and files.get(relative)==metadata[field+'_sha256'], 'diagnostic hash/reference mismatch')
            references.add(relative)
        with resolve(root,metadata['png']).open('rb') as handle:
            header = handle.read(33)
        _require(len(header)==33 and header[:8]==b'\x89PNG\r\n\x1a\n' and header[8:16]==b'\0\0\0\rIHDR', 'invalid diagnostic PNG header')
        _require(struct.unpack('>IIBBBBB',header[16:29])==(1100,1040,8,2,0,0,0), 'invalid diagnostic PNG dimensions/format')
        _require(struct.unpack('>I',header[29:33])[0]==zlib.crc32(header[12:29]), 'invalid diagnostic PNG IHDR checksum')
        verify_diagnostic_ledger(resolve(root,metadata['evidence']),metadata,row,report['cnt_mode'])
    _require(references=={p for p in files if p=='diagnostics' or p.startswith('diagnostics/')},
             'unreferenced/missing diagnostic artifacts')

    required_directories = {'diagnostics'} if references else set()
    required_directories.update(str(PurePosixPath(p).parent) for p in references)
    actual_directories = {p.relative_to(root).as_posix() for p in root.rglob('*')
                          if p.is_dir() and (p.relative_to(root).as_posix()=='diagnostics'
                                             or p.relative_to(root).as_posix().startswith('diagnostics/'))}
    _require(actual_directories==required_directories, 'invalid diagnostic directory namespace')
