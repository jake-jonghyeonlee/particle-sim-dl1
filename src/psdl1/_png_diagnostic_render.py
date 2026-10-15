"""Bounded raster/stream sinks for png_diagnostic; no source remeasurement."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from .png_diagnostic import VERSION, iter_events, yellow_cells

SIZE = (1100, 1040)
VIEW = 768
ORIGIN = (20, 84)
COLORS = {'electrode': (74, 120, 200), 'cnt': (46, 139, 87),
          'overlap': (123, 94, 167), 'padding': (236, 236, 236),
          'yellow': (242, 194, 0), 'red': (224, 32, 27)}


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


def _crop_pair(source, target, transform, grid_xy):
    """Capture at most two 132-square buffers using the same target window.

    Source sampling happens directly from that window's full-frame footprint;
    no giant source crop or image is allocated and no analysis is performed.
    """
    g = transform.grid
    span = min(8, g)
    x0 = max(0, min(g - span, int(grid_xy[0]) - span // 2))
    y0 = max(0, min(g - span, int(grid_xy[1]) - span // 2))
    x1, y1 = x0 + span, y0 + span
    n = 132
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
        py, px = np.nonzero(source[ay0:ay1, ax0:ax1])
        qx = np.floor(((px + ax0 + .5) * nw / w + left - x0) * n / span).astype(int)
        qy = np.floor(((py + ay0 + .5) * nh / h + top - y0) * n / span).astype(int)
        keep = (qx >= 0) & (qx < n) & (qy >= 0) & (qy < n)
        src[qy[keep], qx[keep]] = True
    dst = target[np.floor(gy).astype(int)[:, None], np.floor(gx).astype(int)]
    return src, dst, [x0, y0, x1, y1]


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
        self.insets = []
        self.eligible = 0
        self._observation_index = 0
        self._finished = False
        self._file = None
        try:
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
            plane |= _base_mask(target)
        # Keep at most a grid-size temporary for yellow; no per-grid large mask
        # survives this observation. Every qualifying cell contributes to OR.
        ym = np.zeros(target.shape, dtype=bool)
        for record in yellow_cells(measured, result.transform):
            self._write('yellow_cell', observation_id=observation.observation_id, **record)
            self.counts['thin_runs'] += record['run_count']
            self.counts['thin_cells'] += 1
            x, y = record['cell_xy']; ym[y, x] = True
        self.yellow |= _project_mask(ym)
        draw = ImageDraw.Draw(self.rings)
        for event, sites in iter_events(observation, measured, result):
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
                src, dst, window = _crop_pair(measured.source, target, result.transform, center)
                enlargement=7 if 0 < int(src.sum()) < 36 else 1
                if enlargement>1:
                    with Image.fromarray(src.astype(np.uint8)*255) as original:
                        with original.filter(ImageFilter.MaxFilter(enlargement)) as enlarged:
                            src=np.asarray(enlarged)!=0
                candidate.update(source=src, target=dst, window=window, witness=center,
                                 source_enlargement=enlargement)
                self.insets.append(candidate)
                self.insets.sort(key=lambda item: item['key'])
                del src, dst

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

    def _render(self, check, metadata):
        g = self.transform.grid
        image = Image.new('RGB', SIZE, 'white')
        base_image = None
        try:
            base = np.full((VIEW, VIEW, 3), 255, dtype=np.uint8)
            left, top, right, bottom = self.transform.padding_ltrb
            index = np.arange(VIEW) * g // VIEW
            pad = ((index[:, None] < top) | (index[:, None] >= g - bottom) |
                   (index < left) | (index >= g - right))
            base[pad] = COLORS['padding']
            base[self.electrode] = COLORS['electrode']
            base[self.cnt] = COLORS['cnt']
            base[self.electrode & self.cnt] = COLORS['overlap']
            yy, xx = np.indices((VIEW, VIEW))
            base[self.holes] = (204, 204, 204)
            base[self.holes & ((xx + yy) % 10 < 2)] = (150, 150, 150)
            base[self.yellow] = ((base[self.yellow].astype(np.uint16) + np.array(COLORS['yellow'])) // 2).astype(np.uint8)
            # Dark cell-outline pixels remain bounded in display resolution.
            surrounded=np.pad(self.yellow,1,constant_values=False)
            edge=self.yellow & (~surrounded[:-2,1:-1] | ~surrounded[2:,1:-1] |
                                ~surrounded[1:-1,:-2] | ~surrounded[1:-1,2:])
            base[edge] = (140, 113, 0)
            base_image = Image.fromarray(base)
            image.paste(base_image, ORIGIN)
            image.paste(self.rings, ORIGIN, self.rings)
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default(size=15)
            small = ImageFont.load_default(size=13)
            full_title = f'{self.model_id} | grid {g} | CNT mode {self.cnt_mode}'
            # Bounded title with full original identity preserved in header.
            for row in range(2):
                text = full_title[row * 118:(row + 1) * 118]
                draw.text((20, 12 + row * 22), text, fill='black', font=font)
            draw.text((20, 60), f'Threshold >= 0.5 | full-frame area + central pad | display {VIEW}/{g} (nearest)', fill='black', font=small)
            draw.rectangle((19, 83, 788, 852), outline=(90, 90, 90))
            for by, bx in zip(*np.nonzero(self.cluster_counts > 1)):
                count = str(int(self.cluster_counts[by, bx]))
                # Reserve each bin's top strip for its count, never truncate.
                x, y = ORIGIN[0] + bx * 64 + 3, ORIGIN[1] + by * 64 + 2
                badge_font = small
                while draw.textbbox((0, 0), count, font=badge_font)[2] > 58:
                    badge_font = ImageFont.load_default(size=max(1, badge_font.size - 1))
                box = draw.textbbox((x, y), count, font=badge_font)
                draw.rectangle((box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1), fill='white', outline=COLORS['red'])
                draw.text((x, y), count, fill=COLORS['red'], font=badge_font)
            draw.text((810, 84), 'Affected-region detail (capped)', fill='black', font=small)
            for row, inset in enumerate(self.insets):
                y = 116 + row * 238
                title = f"{inset['observation_id']} {inset['scope']} layer {inset['layer'] or '-'}"
                draw.text((810, y), title[:38], fill='black', font=small)
                draw.text((810, y + 18), inset['key'][2].split(':')[1] + ' | affected region', fill='black', font=small)
                draw.text((810,y+201),f"source stamp {inset['source_enlargement']}px (display only)",fill='black',font=small)
                caption=f"window {inset['window']}"
                caption_font=ImageFont.load_default(size=11)
                while draw.textbbox((0,0),caption,font=caption_font)[2]>280 and caption_font.size>5:
                    caption_font=ImageFont.load_default(size=caption_font.size-1)
                draw.text((810,y+216),caption,fill='black',font=caption_font)
                for col, kind in enumerate(('source', 'target')):
                    draw.text((810 + col * 140, y + 39), kind, fill='black', font=small)
                    crop = np.full((132, 132, 3), 255, dtype=np.uint8)
                    crop[inset[kind]] = COLORS['electrode'] if inset['scope'] == 'electrode' else COLORS['cnt']
                    with Image.fromarray(crop) as crop_image:
                        marker=ImageDraw.Draw(crop_image)
                        wx=(inset['witness'][0]-inset['window'][0])*132/(inset['window'][2]-inset['window'][0])
                        wy=(inset['witness'][1]-inset['window'][1])*132/(inset['window'][3]-inset['window'][1])
                        marker.ellipse((wx-9,wy-9,wx+9,wy+9),outline='white',width=4)
                        marker.ellipse((wx-7,wy-7,wx+7,wy+7),outline=COLORS['red'],width=2)
                        image.paste(crop_image, (810 + col * 140, y + 59))
                    draw.rectangle((810 + col * 140, y + 59, 942 + col * 140, y + 191), outline=(100, 100, 100))
            draw.rectangle((20, 872, 32, 884), fill=COLORS['yellow'], outline=(140, 113, 0))
            draw.text((40, 869), 'thin-run candidates; gray = padding / hatch = unobserved', fill='black', font=small)
            draw.ellipse((20, 895, 33, 908), outline=COLORS['red'], width=3)
            draw.text((40, 891), 'shape-change evidence (D / S / M?) at real component witnesses', fill='black', font=small)
            for i, name in enumerate(('electrode', 'cnt', 'overlap')):
                x = 810 + i * 90
                draw.rectangle((x, 872, x + 10, 882), fill=COLORS[name])
                draw.text((x, 890), name, fill='black', font=small)
            c = self.counts
            lines = [f"Y thin-run candidates: {c['thin_runs']} runs in {c['thin_cells']} cells (observation totals)",
                     f"R shape change: D {c['disappearance']} S {c['split']} M? {c['merge_suspected']} ({c['sites']} sites; observation totals)",
                     {'change_detected': 'shape change detected', 'no_change_detected': 'no shape change detected by this check', 'observation_incomplete': 'observation incomplete'}[metadata['shape_state']],
                     f"{c['sites']} sites / {c['clusters']} clusters; all sites included; insets {len(self.insets)} of {self.eligible} eligible; {self.eligible - len(self.insets)} detail views not shown",
                     f"legacy grid status: {check['status']} (approval unchanged); layer overlap may hide shape; markers retained",
                     'Component correspondence is not proof of geometry, linewidth, topology or electrical connectivity.']
            for row, line in enumerate(lines):
                draw.text((20, 920 + row * 19), line, fill='black', font=small)
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
        self.rings.close()
        self.insets.clear()
        self.progress = None
        # Release numeric planes even if the caller retains a settled sink.
        self.electrode = self.cnt = self.yellow = self.holes = None
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

