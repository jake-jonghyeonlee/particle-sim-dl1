"""Bounded raster/stream sinks for png_diagnostic; no source remeasurement."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .png_diagnostic import VERSION, iter_events, yellow_cells

SIZE = (1100, 1040)
VIEW = 768
ORIGIN = (20, 84)
COLORS = {'electrode': (74, 120, 200), 'cnt': (46, 139, 87),
          'overlap': (123, 94, 167), 'padding': (236, 236, 236),
          'yellow': (242, 194, 0), 'red': (224, 32, 27)}
# Deterministic per-layer palette (no green/yellow/red: reserved for CNT and
# markers). Layers get colors in first-seen Product order, which is identical
# for every grid of one product; extra layers use golden-ratio hues.
LAYER_PALETTE = ((31, 119, 180), (255, 127, 14), (148, 103, 189), (140, 86, 75),
                 (227, 119, 194), (0, 128, 128), (60, 60, 160), (190, 140, 60),
                 (100, 170, 230), (170, 0, 170), (90, 110, 140), (200, 80, 0))
LAYER_ALPHA, CNT_ALPHA = .5, .6
DIFF = {'both': (95, 95, 95), 'source_only': (230, 0, 160), 'target_only': (0, 160, 230)}
CROP = 90
YELLOW_DETAILS = 2


def layer_color(index):
    if index < len(LAYER_PALETTE):
        return LAYER_PALETTE[index]
    import colorsys
    hue = (index * .618034) % 1
    return tuple(int(c * 255) for c in colorsys.hsv_to_rgb(hue, .75, .7))


def tint(color, alpha=LAYER_ALPHA):
    """Swatch colour of one translucent layer over white, as in the overview."""
    return tuple(int(round(255 * (1 - alpha) + c * alpha)) for c in color)


def _judgment(metadata, has_thin):
    """Display-only judgment; legacy status and diagnostic v1 stay unchanged."""
    state = metadata['shape_state']
    if state == 'observation_incomplete':
        return 'INSPECTION INCOMPLETE', COLORS['red']
    if state == 'change_detected':
        return 'SHAPE CHANGE DETECTED', COLORS['red']
    if has_thin:
        return 'THIN AREAS: REFERENCE ONLY', (140, 113, 0)
    return 'NO SHAPE CHANGE DETECTED', (35, 75, 70)


def display_xy(grid_xy, grid):
    return [ORIGIN[i] + float(grid_xy[i]) * VIEW / grid for i in (0, 1)]


def _project_mask(mask):
    """Categorical display only. Max/OR prevents lost cells when reducing."""
    g = mask.shape[0]
    if g <= VIEW:
        index = np.arange(VIEW) * g // VIEW
        return mask[index[:, None], index]
    y, x = np.nonzero(mask)
    out = np.zeros((VIEW, VIEW), dtype=bool)
    out[y * VIEW // g, x * VIEW // g] = True
    return out


def _base_mask(mask):
    # Plan section 4 specifies nearest reduction of target geometry, with an
    # explicit caption. Yellow uses OR independently, including subpixel cells.
    index = np.minimum(np.arange(VIEW) * mask.shape[0] // VIEW, mask.shape[0] - 1)
    return mask[index[:, None], index]


def _holes(valid, transform):
    h, w = valid.shape
    nh, nw = transform.resized_shape_yx
    left, top, _, _ = transform.padding_ltrb
    # Exact adaptive-support any-hole map, independent from foreground coverage.
    integral = np.pad(np.cumsum(np.cumsum(~valid.astype(bool), axis=0), axis=1), ((1, 0), (1, 0)))
    x = np.arange(nw); y = np.arange(nh)
    x0, x1 = x * w // nw, ((x + 1) * w + nw - 1) // nw
    y0, y1 = y * h // nh, ((y + 1) * h + nh - 1) // nh
    count = integral[y1[:, None], x1] - integral[y0[:, None], x1] - integral[y1[:, None], x0] + integral[y0[:, None], x0]
    out = np.zeros((transform.grid, transform.grid), dtype=bool)
    out[top:top + nh, left:left + nw] = count != 0
    return _project_mask(out)


BAND_PIXELS = 1 << 16  # max original pixels scanned per row band


def _tiles(y0, y1, x0, x1):
    """Row bands (x-tiled if very wide) of <= BAND_PIXELS elements each."""
    cols = min(x1 - x0, BAND_PIXELS)
    rows = max(1, BAND_PIXELS // cols)
    for b0 in range(y0, y1, rows):
        for c0 in range(x0, x1, cols):
            yield b0, min(y1, b0 + rows), c0, min(x1, c0 + cols)


def _context_window(transform, event, minimum=4, pad=1):
    """Integer square grid ROI covering an event's source+target bounds.

    Display only: uses the bounds already recorded in the event, so both
    pieces and the joining region of a MERGE/SPLIT are in view when they fit.
    """
    g = transform.grid
    boxes = [list(b) for b in event['target_bounds']]
    for x0, y0, x1, y1 in event['source_bounds']:
        a, b = transform.pixel_to_grid(x0, y0)
        c, d = transform.pixel_to_grid(x1, y1)
        boxes.append([a, b, c, d])
    x0 = min(b[0] for b in boxes) - pad; y0 = min(b[1] for b in boxes) - pad
    x1 = max(b[2] for b in boxes) + pad; y1 = max(b[3] for b in boxes) + pad
    span = int(min(g, max(minimum, np.ceil(x1 - x0), np.ceil(y1 - y0))))
    return (int(np.floor((x0 + x1 - span) / 2)), int(np.floor((y0 + y1 - span) / 2))), span


def _join_seam(event, limit=8):
    """Closest gap centre between two disjoint source boxes (display only).

    Considers at most the first ``limit`` recorded source bounds (fan-in may be
    huge) and only box pairs separated on one axis while overlapping on the
    other. Returns the seam centre in source pixels, or None when no simple
    seam exists. Not a proof of a physical bridge.
    """
    boxes = event['source_bounds'][:limit]
    best = None
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            for lo, hi in ((0, 2), (1, 3)):
                ol, oh = 1 - lo, 3 - lo
                first, second = (a, b) if a[lo] <= b[lo] else (b, a)
                gap = second[lo] - first[hi]
                o0, o1 = max(a[ol], b[ol]), min(a[oh], b[oh])
                if gap >= 0 and o1 > o0 and (best is None or gap < best[0]):
                    mid = (first[hi] + second[lo]) / 2
                    cross = (o0 + o1) / 2
                    best = gap, (mid, cross) if lo == 0 else (cross, mid)
    return None if best is None else best[1]


def _crop_pair(source, target, transform, grid_xy, span=8, n=132, corner=None):
    """Capture at most two n-square buffers using the same target window.

    Source sampling happens directly from that window's full-frame footprint;
    no giant source crop or image is allocated and no analysis is performed.
    Display pixels are nearest samples of original pixels plus an OR scatter of
    every original foreground pixel: no dilation/MaxFilter, so an original
    pixel never covers more than its own footprint (or one display pixel).
    Source footprint and reduced target are scanned in tiles of <= BAND_PIXELS.
    """
    g = transform.grid
    span = min(span, g)
    if corner is None:
        corner = int(grid_xy[0]) - span // 2, int(grid_xy[1]) - span // 2
    x0 = max(0, min(g - span, corner[0]))
    y0 = max(0, min(g - span, corner[1]))
    x1, y1 = x0 + span, y0 + span
    gx = x0 + (np.arange(n) + .5) * span / n
    gy = y0 + (np.arange(n) + .5) * span / n
    sx, _ = transform.grid_to_pixel(gx, 0)
    _, sy = transform.grid_to_pixel(0, gy)
    sx, sy = np.floor(sx + .5).astype(int), np.floor(sy + .5).astype(int)
    good = ((sy[:, None] >= 0) & (sy[:, None] < source.shape[0]) &
            (sx >= 0) & (sx < source.shape[1]))
    src = source[np.clip(sy, 0, source.shape[0] - 1)[:, None], np.clip(sx, 0, source.shape[1] - 1)] != 0
    src &= good
    left, top, _, _ = transform.padding_ltrb
    nh, nw = transform.resized_shape_yx
    h, w = source.shape
    ax0, ax1 = max(0, (x0 - left) * w // nw), min(w, ((x1 - left) * w + nw - 1) // nw)
    ay0, ay1 = max(0, (y0 - top) * h // nh), min(h, ((y1 - top) * h + nh - 1) // nh)
    if ax0 < ax1 and ay0 < ay1:
        for b0, b1, c0, c1 in _tiles(ay0, ay1, ax0, ax1):
            py, px = np.nonzero(source[b0:b1, c0:c1])
            qx = np.floor(((px + c0 + .5) * nw / w + left - x0) * n / span).astype(int)
            qy = np.floor(((py + b0 + .5) * nh / h + top - y0) * n / span).astype(int)
            keep = (qx >= 0) & (qx < n) & (qy >= 0) & (qy < n)
            src[qy[keep], qx[keep]] = True
            del py, px, qx, qy, keep
    dst = target[np.floor(gy).astype(int)[:, None], np.floor(gx).astype(int)]
    if span > n:
        # Reduction: OR every target cell so no grid cell disappears.
        for b0, b1, c0, c1 in _tiles(y0, y1, x0, x1):
            cy, cx = np.nonzero(target[b0:b1, c0:c1])
            dst[(cy + b0 - y0) * n // span, (cx + c0 - x0) * n // span] = True
            del cy, cx
    return src, dst, [x0, y0, x1, y1]


def _source_px_per_display(transform, window, n):
    """Original pixels per crop display pixel (<1 means pure enlargement)."""
    h, w = transform.shape_yx
    nh, nw = transform.resized_shape_yx
    return (window[2] - window[0]) * max(w / nw, h / nh) / n


def _scale_caption(ratio):
    if ratio <= 1:
        return f'src x{1 / ratio:.1f} nearest'
    return f'src OR 1:{ratio:.1f}'


def _detail_span(transform, widest, narrowest):
    """Grid cells per crop: largest span whose source footprint fits CROP px."""
    h, w = transform.shape_yx
    nh, nw = transform.resized_shape_yx
    per_cell = max(w / nw, h / nh)
    return int(max(narrowest, min(widest, CROP // per_cell)))


def _thin_run(measured, transform, cell, length):
    """Locate one minimal qualifying axis run of a yellow cell (display only).

    Uses the same oriented maximal runs and midpoint->cell rule as
    yellow_cells(); returns (orientation, row, start, length) or None.
    """
    g = transform.grid
    for orientation, runs in enumerate((measured.horizontal, measured.vertical)):
        rows, starts, lengths = runs.T
        hit = np.flatnonzero(lengths == length)
        if not hit.size:
            continue
        middle = starts[hit] + (length - 1) / 2
        x, y = (middle, rows[hit]) if orientation == 0 else (rows[hit], middle)
        gx, gy = transform.pixel_to_grid(x, y)
        match = np.flatnonzero((np.floor(gx).astype(np.int64) == cell[0]) &
                               (np.floor(gy).astype(np.int64) == cell[1]))
        if match.size:
            i = hit[match[0]]
            return orientation, int(rows[i]), int(starts[i]), int(lengths[i])
    return None


def _select_insets(candidates):
    ordered=sorted(candidates,key=lambda item:item['key'])
    selected=[];kinds=set();observations=set()
    for item in ordered:
        if item['key'][0] not in kinds and len(selected)<3:
            selected.append(item);kinds.add(item['key'][0]);observations.add(item['observation_id'])
    for item in ordered:
        if len(selected)<3 and item['observation_id'] not in observations:
            selected.append(item);observations.add(item['observation_id'])
    for item in ordered:
        if len(selected)<3 and not any(item is other for other in selected):selected.append(item)
    return selected


class DiagnosticSink:
    """One fixed display working set and <=3 paired buffers per grid.

    Files are owned by the caller's staging tree. Failure closes handles but
    removes only its own unfinished files; publication is the importer's job.
    """
    def __init__(self, png_path, ledger_path, *, model_id, transform, valid, cnt_mode='image_only',
                 progress=None, observation_count=None):
        self.png_path, self.ledger_path = Path(png_path), Path(ledger_path)
        self.model_id, self.transform, self.cnt_mode = model_id, transform, cnt_mode
        self.progress, self.observation_count = progress, observation_count
        self._ledger_owned = self._png_owned = False
        self._created_directories = set()
        # close() must be safe from the first mkdir through every allocation.
        self._finished = False
        self._file = None
        self.rings = None
        self.electrode = self.cnt = self.yellow = self.holes = None
        self.cluster_counts = self.cluster_bounds = None
        self.insets = []
        self.yellow_examples = []
        self.layers = {}  # layer name -> palette index, first-seen order
        self.layer_rgb = None
        self.event_observations = {k: 0 for k in ('disappearance', 'split', 'merge_suspected')}
        try:
            for directory in (self.png_path.parent, self.ledger_path.parent):
                missing = []
                while not directory.exists():
                    missing.append(directory)
                    directory = directory.parent
                for directory in reversed(missing):
                    try:
                        directory.mkdir()
                    except FileExistsError:
                        pass
                    else:
                        self._created_directories.add(directory)
            self.electrode = np.zeros((VIEW, VIEW), dtype=bool)
            # One fixed translucent RGB accumulator; no per-layer planes.
            self.layer_rgb = np.full((VIEW, VIEW, 3), 255, dtype=np.float32)
            self.cnt = np.zeros_like(self.electrode)
            self.yellow = np.zeros_like(self.electrode)
            self.holes = _holes(valid, transform) if not valid.all() else np.zeros_like(self.electrode)
            self.incomplete = not bool(valid.all())
            self.rings = Image.new('RGBA', (VIEW, VIEW), (0, 0, 0, 0))
            self.cluster_counts = np.zeros((12, 12), dtype=np.int64)
            self.cluster_bounds = np.zeros((12, 12, 4), dtype=float)
            self.cluster_bounds[:, :, :2] = np.inf
            self.cluster_bounds[:, :, 2:] = -np.inf
            self.counts = {k: 0 for k in ('observations', 'thin_runs', 'thin_cells', 'disappearance',
                                        'split', 'merge_suspected', 'sites', 'clusters')}
            self.eligible = 0
            self._observation_index = 0
            self._file = self.ledger_path.open('x', encoding='utf-8')
            self._ledger_owned = True
            self._write('header', diagnostic_schema_version=VERSION, model_id=model_id,
                    grid=transform.grid, transform=transform.to_record(), threshold=.5,
                    connectivity=4, scope='logical_observations', cnt_mode=cnt_mode,
                    valid_holes=not bool(valid.all()),
                    display={'size_xy': list(SIZE), 'origin_xy': list(ORIGIN), 'viewport': VIEW,
                             'cluster_bin_px': 64, 'base_reduction': 'nearest', 'yellow_reduction': 'OR'})
        except BaseException:
            self.close()
            raise

    def _write(self, record_type, **fields):
        self._file.write(json.dumps({'type': record_type, **fields}, ensure_ascii=True,
                                    allow_nan=False, separators=(',', ':')) + '\n')

    def observation(self, observation, measured, result):
        self._observation_index += 1
        self.counts['observations'] += 1
        if self.progress and (self._observation_index == 1 or self._observation_index % 100 == 0
                              or self._observation_index == self.observation_count):
            self.progress(state='advanced', grid=self.transform.grid,
                          observations_done=self._observation_index, observations_total=self.observation_count)
        state = 'missing' if measured is None else ('observed_empty' if measured.active == 0 else 'observed')
        self.incomplete |= measured is None
        self._write('observation', **observation.identity(), state=state,
                    source_components=None if measured is None else measured.labels.count,
                    target_components=None if result is None else result.target.count,
                    active_pixels=None if measured is None else measured.active,
                    axis_run_px=None if measured is None else measured.narrowest)
        if measured is None:
            return
        target = result.coverage >= .5
        if observation.scope in ('electrode', 'global_cnt_union'):
            plane = self.electrode if observation.scope == 'electrode' else self.cnt
            shown = _base_mask(target)
            plane |= shown
            if observation.scope == 'electrode':
                name = observation.layer or observation.legacy_id or '-'
                index = self.layers.setdefault(name, len(self.layers))
                color = np.asarray(layer_color(index), dtype=np.float32)
                self.layer_rgb[shown] += (color - self.layer_rgb[shown]) * LAYER_ALPHA
                del shown
        # Keep at most a grid-size temporary for yellow; no per-grid large mask
        # survives this observation. Every qualifying cell contributes to OR.
        ym = np.zeros(target.shape, dtype=bool)
        thinnest = None
        for record in yellow_cells(measured, result.transform):
            self._write('yellow_cell', observation_id=observation.observation_id, **record)
            self.counts['thin_runs'] += record['run_count']
            self.counts['thin_cells'] += 1
            x, y = record['cell_xy']; ym[y, x] = True
            rank = (not target[y, x], record['min_length_px'])
            if thinnest is None or rank < thinnest[0]:
                thinnest = rank, record
        self.yellow |= _project_mask(ym)
        del ym
        if thinnest is not None:
            self._yellow_example(observation, measured, result, target, thinnest[1])
        draw = ImageDraw.Draw(self.rings)
        observed_kinds = set()
        for event, sites in iter_events(observation, measured, result):
            observed_kinds.add(event['kind'])
            self.counts[event['kind']] += 1
            self.eligible += 1
            self._write('event', observation_id=observation.observation_id, site_count=len(sites), **event)
            for index, site in enumerate(sites):
                x, y = np.asarray(site['grid_xy']) * VIEW / self.transform.grid
                bx, by = min(11, int(x // 64)), min(11, int(y // 64))
                self.cluster_counts[by, bx] += 1
                bounds = self.cluster_bounds[by, bx]
                bounds[:2] = np.minimum(bounds[:2], (x, y))
                bounds[2:] = np.maximum(bounds[2:], (x, y))
                self.counts['sites'] += 1
                self._write('site', observation_id=observation.observation_id,
                            event_id=event['event_id'], site_index=index, cluster_id=by * 12 + bx,
                            **site)
                draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline='white', width=5)
                draw.ellipse((x - 10, y - 10, x + 10, y + 10), outline=COLORS['red'], width=3)
            priority = {'disappearance': 0, 'split': 1, 'merge_suspected': 2}[event['kind']]
            key = (priority, self._observation_index, event['event_id'])
            # Prefer distinct observations, then fill unused detail slots with
            # further events. Selection keeps exactly <=3 paired buffers live.
            candidate = {'key': key, 'event_id': event['event_id'], 'scope': observation.scope,
                         'observation_id': observation.observation_id, 'layer': observation.layer}
            selected = _select_insets(self.insets + [candidate])
            if any(item is candidate for item in selected):
                self.insets = [item for item in self.insets if any(item is chosen for chosen in selected)]
                del selected
                center=sites[0]['grid_xy']
                if sites[0]['lost_source_xy'] is not None:
                    center=list(result.transform.pixel_to_grid(*sites[0]['lost_source_xy']))
                # Same window for source/target; source is never dilated.
                # LOSS stays local; SPLIT/MERGE show the whole event context.
                seam = _join_seam(event) if event['kind'] == 'merge_suspected' else None
                if event['kind'] == 'disappearance':
                    corner, span = None, _detail_span(result.transform, 8, 2)
                elif seam is not None:
                    # MERGE: local window on the closest source gap so the
                    # original gap stays visible at enlargement (half the
                    # usual detail span, >= 2 cells, so a 2px gap is ~5 px).
                    center = list(result.transform.pixel_to_grid(*seam))
                    span = max(2, _detail_span(result.transform, 8, 2) // 2)
                    corner = int(np.floor(center[0])) - span // 2, int(np.floor(center[1])) - span // 2
                else:
                    corner, span = _context_window(result.transform, event)
                src, dst, window = _crop_pair(measured.source, target, result.transform, center,
                                              span, CROP, corner)
                candidate.update(source=src, target=dst, window=window, witness=center,
                                 context=('join' if seam is not None else
                                          event['kind'] != 'disappearance'),
                                 color=self._color(observation),
                                 kind=event['kind'], scale=_source_px_per_display(result.transform, window, CROP))
                self.insets.append(candidate)
                self.insets.sort(key=lambda item: item['key'])
                del src, dst

        for kind in observed_kinds:
            self.event_observations[kind] += 1

    def _color(self, observation):
        """Detail mask colour equals the observation's overview layer colour."""
        if observation.scope != 'electrode':
            return COLORS['cnt']
        name = observation.layer or observation.legacy_id or '-'
        return layer_color(self.layers[name]) if name in self.layers else COLORS['electrode']

    def _yellow_example(self, observation, measured, result, target, record):
        """Keep <=YELLOW_DETAILS thinnest-cell crops (one per observation)."""
        # Prefer thin cells that survive in the grid result (pure reference,
        # not already a red loss), then the shortest original run.
        cx, cy = record['cell_xy']
        key = (0 if target[cy, cx] else 1, record['min_length_px'], self._observation_index)
        if len(self.yellow_examples) >= YELLOW_DETAILS and key >= self.yellow_examples[-1]['key']:
            return
        transform = result.transform
        run = _thin_run(measured, transform, (cx, cy), record['min_length_px'])
        src, dst, window = _crop_pair(measured.source, target, transform, (cx + .5, cy + .5),
                                      _detail_span(transform, 6, 3), CROP)
        self.yellow_examples.append({
            'key': key, 'observation_id': observation.observation_id, 'scope': observation.scope,
            'layer': observation.layer, 'color': self._color(observation), 'cell': [cx, cy], 'record': dict(record), 'run': run,
            'run_grid': None if run is None else self._run_grid(run, transform),
            'source': src, 'target': dst, 'window': window,
            'scale': _source_px_per_display(transform, window, CROP),
            'projected': record['min_length_px'] * transform.grid / max(transform.shape_yx)})
        self.yellow_examples.sort(key=lambda item: item['key'])
        del self.yellow_examples[YELLOW_DETAILS:]

    @staticmethod
    def _run_grid(run, transform):
        orientation, row, start, length = run
        if orientation == 0:
            x0, y0 = transform.pixel_to_grid(start - .5, row - .5)
            x1, y1 = transform.pixel_to_grid(start + length - .5, row + .5)
        else:
            x0, y0 = transform.pixel_to_grid(row - .5, start - .5)
            x1, y1 = transform.pixel_to_grid(row + .5, start + length - .5)
        return [float(x0), float(y0), float(x1), float(y1)]

    def finish(self, check):
        self.counts['clusters'] = int(np.count_nonzero(self.cluster_counts))
        for by, bx in zip(*np.nonzero(self.cluster_counts)):
            self._write('cluster', cluster_id=int(by * 12 + bx),
                        site_count=int(self.cluster_counts[by, bx]),
                        display_bounds_xyxy=self.cluster_bounds[by, bx].tolist())
        state = ('observation_incomplete' if self.incomplete else
                 ('change_detected' if self.counts['sites'] else 'no_change_detected'))
        metadata = {'diagnostic_schema_version': VERSION, 'state': 'complete',
                    'shape_state': state, 'counts': dict(self.counts),
                    'insets_shown': len(self.insets), 'eligible_insets': self.eligible,
                    'unshown_insets': self.eligible - len(self.insets),
                    'all_sites_included': True}
        self._write('summary', **metadata, legacy_status=check['status'])
        self._file.close()
        self._render(check, metadata)
        artifact = {**metadata, 'png': str(self.png_path), 'evidence': str(self.ledger_path),
                'png_sha256': _sha256(self.png_path), 'evidence_sha256': _sha256(self.ledger_path)}
        self._finished = True
        return artifact

    def _crop_image(self, item, color, grid, witness=None, run_grid=None):
        """Return three registered RGB crops: source, target, difference."""
        x0, y0, x1, y1 = item['window']
        span = x1 - x0
        src, dst = item['source'], item['target']
        crops = []
        for kind in ('source', 'target', 'difference'):
            crop = np.full((CROP, CROP, 3), 255, dtype=np.uint8)
            if kind == 'difference':
                crop[src & dst] = DIFF['both']
                crop[src & ~dst] = DIFF['source_only']
                crop[~src & dst] = DIFF['target_only']
            else:
                crop[item[kind]] = color
            image = Image.fromarray(crop)
            marker = ImageDraw.Draw(image)
            # Target grid-cell boundaries; annotation only, masks unchanged.
            if span <= 16:
                for k in range(1, span):
                    p = int(round(k * CROP / span))
                    marker.line((p, 0, p, CROP), fill=(200, 200, 200))
                    marker.line((0, p, CROP, p), fill=(200, 200, 200))
            if witness is not None:
                wx = (witness[0] - x0) * CROP / span
                wy = (witness[1] - y0) * CROP / span
                marker.ellipse((wx - 8, wy - 8, wx + 8, wy + 8), outline=COLORS['red'], width=2)
            if run_grid is not None:
                a, b, c, d = [(v - o) * CROP / span for v, o in zip(run_grid, (x0, y0, x0, y0))]
                marker.rectangle((a - 2, b - 2, c + 1, d + 1), outline=(140, 113, 0), width=1)
            crops.append(image)
        return crops

    @staticmethod
    def _layer_line(draw, xy, item, font, chars=44):
        """Swatch in the overview layer colour, then scope and layer name."""
        x, y = xy
        draw.rectangle((x, y + 1, x + 9, y + 10), fill=item['color'], outline=(90, 90, 90))
        label = f"layer {item['layer'] or '-'}" if item['scope'] == 'electrode' else item['scope']
        draw.text((x + 13, y), label[:chars], fill='black', font=font)

    def _render(self, check, metadata):
        g = self.transform.grid
        image = Image.new('RGB', SIZE, 'white')
        base_image = None
        try:
            base = np.rint(self.layer_rgb).astype(np.uint8)
            left, top, right, bottom = self.transform.padding_ltrb
            index = np.arange(VIEW) * g // VIEW
            pad = ((index[:, None] < top) | (index[:, None] >= g - bottom) |
                   (index < left) | (index >= g - right))
            base[pad] = COLORS['padding']
            yy, xx = np.indices((VIEW, VIEW))
            # CNT is composited after all electrode layers so it stays visible;
            # overlap gets a sparse dark hatch that keeps layer colours readable.
            cnt = self.cnt
            base[cnt] = (base[cnt] * (1 - CNT_ALPHA) + np.array(COLORS['cnt']) * CNT_ALPHA).astype(np.uint8)
            base[self.electrode & cnt & ((xx - yy) % 6 == 0)] = (40, 20, 70)
            base[self.holes] = (204, 204, 204)
            base[self.holes & ((xx + yy) % 10 < 2)] = (150, 150, 150)
            base[self.yellow] = ((base[self.yellow].astype(np.uint16) + np.array(COLORS['yellow'])) // 2).astype(np.uint8)
            # Dark cell-outline pixels remain bounded in display resolution.
            surrounded=np.pad(self.yellow,1,constant_values=False)
            edge=self.yellow & (~surrounded[:-2,1:-1] | ~surrounded[2:,1:-1] |
                                ~surrounded[1:-1,:-2] | ~surrounded[1:-1,2:])
            base[edge] = (140, 113, 0)
            del yy, xx
            base_image = Image.fromarray(base)
            del base
            image.paste(base_image, ORIGIN)
            image.paste(self.rings, ORIGIN, self.rings)
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default(size=15)
            small = ImageFont.load_default(size=13)
            tiny = ImageFont.load_default(size=11)
            full_title = f'{self.model_id} | grid {g}'
            draw.text((20, 8), full_title[:130], fill='black', font=font)
            verdict, color = _judgment(metadata, bool(self.yellow.any()))
            draw.text((20, 30), verdict, fill=color, font=ImageFont.load_default(size=28))
            draw.text((20, 64), f'Actual grid result | {g} -> {VIEW}px nearest display | layers translucent, '
                      'composition order is not physical stack', fill='black', font=small)
            draw.rectangle((19, 83, 788, 852), outline=(90, 90, 90))
            # Overview locator labels for detail panels.
            for i, inset in enumerate(self.insets):
                x, y = display_xy(inset['witness'], g)
                draw.text((x + 13, y - 18), f'R{i + 1}', fill=COLORS['red'], font=small,
                          stroke_width=2, stroke_fill='white')
            for i, item in enumerate(self.yellow_examples):
                (ax, ay), (bx, by) = display_xy(item['cell'], g), display_xy([c + 1 for c in item['cell']], g)
                draw.rectangle((ax - 3, ay - 3, max(bx, ax + 1) + 3, max(by, ay + 1) + 3), outline=(140, 113, 0), width=2)
                draw.text((max(bx, ax + 1) + 5, ay - 14), f'Y{i + 1}', fill=(110, 90, 0), font=small,
                          stroke_width=2, stroke_fill='white')
            px = 800
            draw.text((px, 84), f'Red shape-change details: {len(self.insets)} of {self.eligible} events',
                      fill='black', font=small)
            if not self.insets:
                draw.text((px, 104), 'No red events at this grid.', fill=(80, 80, 80), font=small)
            for row, inset in enumerate(self.insets):
                y = 104 + row * 170
                kind = {'disappearance': 'LOSS', 'split': 'SPLIT', 'merge_suspected': 'MERGE suspected'}[inset['kind']]
                draw.text((px, y), f"R{row + 1} {kind} | obs {inset['observation_id']}"[:48],
                          fill=COLORS['red'], font=small)
                self._layer_line(draw, (px, y + 15), inset, tiny)
                w = inset['window']
                where = {'join': 'local join context'}.get(inset.get('context'),
                                                           'event bounds+1' if inset.get('context') else 'local')
                draw.text((px, y + 28), f'ROI grid x{w[0]}-{w[2]} y{w[1]}-{w[3]} ({where}; all 3)',
                          fill='black', font=tiny)
                for col, (label, crop) in enumerate(zip(('original', f'grid {g}', 'difference'),
                                                        self._crop_image(inset, inset['color'], g, inset['witness']))):
                    x = px + col * 96
                    draw.text((x, y + 41), label, fill='black', font=tiny)
                    with crop:
                        image.paste(crop, (x, y + 54))
                    draw.rectangle((x - 1, y + 53, x + CROP, y + 54 + CROP), outline=(100, 100, 100))
                draw.text((px, y + 146), f"{_scale_caption(inset['scale'])}; no dilation", fill='black', font=tiny)
                draw.text((px, y + 158), 'diff: gray both / magenta orig-only / cyan grid-only', fill='black', font=tiny)
            yt = 614
            draw.text((px, yt), f'Yellow thin references: {len(self.yellow_examples)} examples, '
                      f"{self.counts['thin_cells']} cells", fill=(110, 90, 0), font=small)
            if not self.yellow_examples:
                draw.text((px, yt + 18), 'No thin-run references at this grid.', fill=(80, 80, 80), font=small)
            for row, item in enumerate(self.yellow_examples):
                y = yt + 17 + row * 116
                draw.text((px, y), f"Y{row + 1} obs {item['observation_id']}"[:30], fill=(110, 90, 0), font=tiny)
                self._layer_line(draw, (px + 194, y), item, tiny, 14)
                tone = item['color']
                crops = self._crop_image(item, tone, g, run_grid=item['run_grid'])
                for col, label in enumerate(('original', f'grid {g}')):
                    x = px + col * 96
                    draw.text((x, y + 12), label, fill='black', font=tiny)
                    with crops[col]:
                        image.paste(crops[col], (x, y + 25))
                    draw.rectangle((x - 1, y + 24, x + CROP, y + 25 + CROP), outline=(100, 100, 100))
                crops[2].close()
                rec = item['record']
                axis = '' if item['run'] is None else ('horizontal ', 'vertical ')[item['run'][0]]
                h, w = self.transform.shape_yx
                lines = [f"{axis}run {rec['min_length_px']}px", f"x {g}/{max(h, w)}",
                         f"-> {item['projected']:.2f} cells", '(< 2: reference)', f"cell {item['cell']}",
                         _scale_caption(item['scale'])]
                for i, line in enumerate(lines):
                    draw.text((px + 194, y + 14 + i * 13), line, fill='black', font=tiny)
            draw.rectangle((20, 872, 32, 884), fill=COLORS['yellow'], outline=(140, 113, 0))
            draw.text((40, 869), 'Thin-area reference only (axis run x grid/max(H,W) < 2; diagonals not measured)',
                      fill='black', font=small)
            draw.ellipse((20, 899, 33, 912), outline=COLORS['red'], width=3)
            draw.text((40, 896), 'Check shape changes; all detected locations marked (locators, not break pixels)',
                      fill='black', font=small)
            # Layer legend: identical colours on every grid of this product.
            names = list(self.layers)
            shown = names[:14]
            entries = [(f'layer {n}', tint(layer_color(self.layers[n])), False) for n in shown]
            if len(names) > len(shown):
                entries.append((f'+{len(names) - len(shown)} more layers (JSONL)', (255, 255, 255), False))
            entries += [('CNT (drawn last)', tint(COLORS['cnt'], CNT_ALPHA), False), ('electrode+CNT', None, True)]
            for i, (label, swatch, hatch) in enumerate(entries):
                x, y = 800 + (i % 2) * 148, 864 + (i // 2) * 16
                if hatch:
                    draw.rectangle((x, y + 2, x + 10, y + 12), outline=(40, 20, 70))
                    draw.line((x, y + 12, x + 10, y + 2), fill=(40, 20, 70))
                else:
                    draw.rectangle((x, y + 2, x + 10, y + 12), fill=swatch, outline=(120, 120, 120))
                draw.text((x + 14, y), label[:20], fill='black', font=tiny)
            c = self.event_observations
            lines = [f"Observations: loss {c['disappearance']} | split {c['split']} | merge? {c['merge_suspected']}",
                     'Overlapping observations may refer to the same change.',
                     f"Red details: {len(self.insets)} shown | {self.eligible - len(self.insets)} more",
                     'Yellow is a heuristic reference, not measured linewidth or proven damage.',
                     'Gray: padding / hatch: unobserved | Full evidence: paired JSONL']
            for row, line in enumerate(lines):
                draw.text((20, 926 + row * 21), line, fill='black', font=font if row == 0 else small)
            # Exclusive output avoids accidental replacement outside staging.
            with self.png_path.open('xb') as output:
                self._png_owned = True
                image.save(output, format='PNG')
        finally:
            if base_image is not None:
                base_image.close()
            image.close()

    def close(self):
        if self._file is not None and not self._file.closed:
            self._file.close()
        if not self._finished:
            for path, owned in ((self.ledger_path, self._ledger_owned), (self.png_path, self._png_owned)):
                if owned:
                    path.unlink(missing_ok=True)
            self.cleanup_directories()
        if self.rings is not None:
            self.rings.close()
        self.insets.clear()
        self.yellow_examples.clear()
        self.progress = None
        # Release numeric planes even if the caller retains a settled sink.
        self.electrode = self.cnt = self.yellow = self.holes = self.layer_rgb = None
        self.cluster_counts = self.cluster_bounds = None


    def cleanup_directories(self):
        """Retry owned empty directories after sibling sinks have closed."""
        if not self._finished:
            for directory in sorted(self._created_directories, key=lambda p: len(p.parts), reverse=True):
                try:
                    directory.rmdir()
                except OSError:
                    pass

def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

