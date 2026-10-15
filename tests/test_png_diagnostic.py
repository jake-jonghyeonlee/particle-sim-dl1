"""Independent numeric oracles for inspection-only diagnostics."""
from dataclasses import replace
import gc
import json
import weakref

import numpy as np
from PIL import Image
import pytest

from psdl1 import data, png_diagnostic as diag
from psdl1._png_diagnostic_render import DiagnosticSink, COLORS, ORIGIN, VIEW, display_xy, _project_mask
from psdl1.schema import Connection, Element, Endpoint, Frame, Product


def product(mask, *, other=(), cnt=None, valid=None, relations=False):
    elements = (Element('E', 'L', mask),) + tuple(other)
    connections = ()
    if relations:
        elements += (Element('B', 'M', np.ones_like(mask)),)
        connections = (Connection('c1', 'CNT', (Endpoint('E@L', (5, 5)), Endpoint('B@M', (5, 5))), np.ones_like(mask)),)
    return Product('synthetic', (mask.shape[1], mask.shape[0]), elements, cnt,
                   frame=Frame("frame", mask.shape, 1), valid=valid, connections=connections,
                   connectivity_observation='partial' if relations else 'unavailable')


def bridge(extra=False):
    m = np.zeros((16, 16), dtype=bool)
    m[4:8, 0:4] = m[4:8, 12:16] = True
    m[5, 4:12] = True
    if extra:
        m[12, 4:8] = m[14, 4:8] = True
    return m


def analyze(mask, grid):
    measured = diag.measure_source(mask, np.ones_like(mask))
    result = diag.measure_grid(measured, grid)
    events = list(diag.iter_events(diag.Observation('o1', 'electrode', mask), measured, result))
    return measured, result, events


def render(tmp_path, p, grids=(4,), descriptors=None):
    def factory(g, transform, valid):
        return DiagnosticSink(tmp_path / f'g{g}.png', tmp_path / f'g{g}.jsonl',
                              model_id=p.product_id, transform=transform, valid=valid)
    outcomes = list(diag.inspect_grid_diagnostics(p, descriptors, grids, factory))
    records = [json.loads(s) for s in (tmp_path / f'g{grids[0]}.jsonl').read_text().splitlines()]
    return outcomes, records


def test_equal_count_split_merge_and_witness_oracle():
    source, result, events = analyze(bridge(True), 4)
    np.testing.assert_array_equal(result.coverage[1], [1, .25, .25, 1])
    np.testing.assert_array_equal(result.coverage[3], [0, .5, 0, 0])
    assert source.labels.count == result.target.count == 3
    assert not data.grid_check(product(bridge(True)), 4)['elements'][0]['bridge_or_component_loss']
    assert [e['kind'] for e, _ in events] == ['split', 'merge_suspected']
    assert [s['target_xy'] for s in events[0][1]] == [[0, 1], [3, 1]]
    assert [s['source_xy'] for s in events[1][1]] == [[4, 12], [4, 14]]
    assert [s['target_xy'] for s in events[1][1]] == [[1, 3], [1, 3]]


def test_real_bridge_and_isolated_pixel():
    source, result, events = analyze(bridge(), 4)
    assert source.labels.count == 1 and result.target.count == 2
    assert [s['grid_xy'] for s in events[0][1]] == [[.5, 1.5], [3.5, 1.5]]
    _, _, finer = analyze(bridge(), 16)
    assert finer == []
    m = np.zeros((16, 16), dtype=bool); m[2, 3] = True
    source, result, events = analyze(m, 4)
    assert result.coverage[0, 0] == .0625
    assert events[0][0]['kind'] == 'disappearance'
    assert events[0][1][0]['grid_xy'] == [.875, .625]
    assert list(diag.yellow_cells(source, result.transform)) == [
        {'cell_xy': [0, 0], 'run_count': 2, 'min_length_px': 1,
         'horizontal_runs': 1, 'vertical_runs': 1}]


def test_diagonal_exact_half_and_below():
    m = np.eye(2, dtype=bool)
    source, result, events = analyze(m, 1)
    assert source.labels.count == 2 and result.coverage[0, 0] == .5
    assert events[0][0]['kind'] == 'merge_suspected'
    assert len(events[0][1]) == 2
    _, _, events = analyze(m, 2)
    assert events == []
    m[1, 1] = False
    _, result, events = analyze(m, 1)
    assert result.coverage[0, 0] == .25
    assert events[0][0]['kind'] == 'disappearance'


@pytest.mark.parametrize('shape,grid,content,pad', [((7, 11), 4, (3, 4), (0, 0, 0, 1)),
    ((11, 7), 4, (4, 3), (0, 0, 1, 0)), ((3, 5), 8, (5, 8), (0, 1, 0, 2)),
    ((1001, 1203), 32, (27, 32), (0, 2, 0, 3))])
def test_registration_independent_bin_oracle(shape, grid, content, pad):
    # Source pattern is deterministic; oracle uses literal accepted content/pad.
    yy, xx = np.indices(shape)
    m = (xx % 4 == 0) | (yy % 5 == 0)
    measured, result, _ = analyze(m, grid)
    assert result.transform.resized_shape_yx == content
    assert result.transform.padding_ltrb == pad
    nh, nw = content; h, w = shape
    expected = np.zeros((grid, grid), dtype=np.float32)
    for y in range(nh):
        for x in range(nw):
            x0, x1 = x * w // nw, ((x + 1) * w + nw - 1) // nw
            y0, y1 = y * h // nh, ((y + 1) * h + nh - 1) // nh
            expected[y + pad[1], x + pad[0]] = m[y0:y1, x0:x1].mean()
    np.testing.assert_allclose(result.coverage, expected, rtol=1e-6, atol=1e-7)
    for edge in result.edges:
        _, _, sx, sy, tx, ty = edge
        x, y = tx - pad[0], ty - pad[1]
        assert x * w // nw <= sx < ((x + 1) * w + nw - 1) // nw
        assert y * h // nh <= sy < ((y + 1) * h + nh - 1) // nh
        assert m[sy, sx] and result.coverage[ty, tx] >= .5
    # Independent endpoint mapping, including odd padding, to display.
    gx = (.5 * nw / w + pad[0]); gy = (.5 * nh / h + pad[1])
    np.testing.assert_allclose(result.transform.pixel_to_grid(0, 0), (gx, gy))
    np.testing.assert_allclose(display_xy((gx, gy), grid), (20 + gx * 768 / grid, 84 + gy * 768 / grid))


def test_many_to_many_and_union_support():
    # Literal 8x8 many-to-many fixture, independent expected adjacency below.
    m = np.array([[1,0,0,0,1,1,1,0], [1,0,0,1,0,1,0,1],
                  [0,1,0,0,1,1,0,1], [0,1,1,0,1,1,1,0],
                  [0,0,1,0,1,1,0,1], [0,0,0,1,0,1,0,0],
                  [0,1,0,1,0,0,0,1], [0,1,1,1,1,0,0,1]], dtype=bool)
    source, result, events = analyze(m, 4)
    assert result.edges[:, :2].tolist() == [[1,1], [2,2], [4,2], [5,1], [5,2], [7,2], [8,3]]
    kinds = [e['kind'] for e, _ in events]
    assert kinds.count('split') == 1 and kinds.count('merge_suspected') == 2
    assert len(np.unique(result.edges[:, :2], axis=0)) == len(result.edges)
    # Tiny diagonal contributor would disappear alone but is adjacent via support
    # to the union's retained cell. Correspondence deliberately retains its edge.
    m = np.array([[1, 0], [0, 1]], dtype=bool)
    _, result, events = analyze(m, 1)
    assert len(result.edges) == 2
    assert not any(e['kind'] == 'disappearance' for e, _ in events)


@pytest.mark.parametrize('seed', range(10))
def test_vector_labeler_matches_legacy_random(seed):
    rng = np.random.default_rng(seed)
    for shape in ((1, 1), (1, 40), (40, 1), (35, 47)):
        for density in (0, .1, .5, .9, 1):
            m = rng.random(shape) < density
            labels = diag.label_components(m)
            assert labels.count == data._components(m)
            assert np.array_equal(labels.image > 0, m)
            first = labels.first_yx[1:]
            assert list(map(tuple, first)) == sorted(map(tuple, first))


@pytest.mark.parametrize('relations', [False, True])
@pytest.mark.parametrize('state', ['full', 'hole', 'missing', 'empty'])
def test_legacy_exact_equality(relations, state):
    m = bridge(True)
    valid = np.ones_like(m); valid[6, 1] = state != 'hole'
    p = product(m, cnt=np.eye(16, dtype=bool), valid=valid, relations=relations)
    if state == 'missing':
        p = replace(p, elements=(replace(p.elements[0], mask=None, observation='missing'),) + p.elements[1:])
    if state == 'empty':
        p = replace(p, elements=(replace(p.elements[0], mask=np.zeros_like(m)),) + p.elements[1:])
    if relations and state in ('missing', 'empty'):
        p = replace(p, connections=(replace(p.connections[0], mask=None, observation='missing'),))
    expected = list(data.iter_grid_checks(p, (4, 8, 16)))
    outcomes = list(diag.inspect_grid_diagnostics(p, grids=(4, 8, 16)))
    assert [o['grid_check'] for o in outcomes] == expected


def test_once_per_observation_and_grid(monkeypatch):
    p = product(bridge(), cnt=np.eye(16, dtype=bool))
    calls = {'source': 0, 'normalize': 0}
    original_measure, original_norm = diag.measure_source, data.normalize_unit
    def measure(*args):
        calls['source'] += 1; return original_measure(*args)
    def normalize(*args):
        calls['normalize'] += 1; return original_norm(*args)
    monkeypatch.setattr(diag, 'measure_source', measure)
    monkeypatch.setattr(data, 'normalize_unit', normalize)
    monkeypatch.setattr(data, '_components', lambda *_: pytest.fail('legacy labeling repeated'))
    outcomes = list(diag.inspect_grid_diagnostics(p, grids=(4, 8, 4, 16)))
    assert len(outcomes) == 3
    assert calls == {'source': 2, 'normalize': 6}


def test_renderer_ledger_conservation_and_colors(tmp_path):
    m = bridge(True)
    outcomes, records = render(tmp_path, product(m, cnt=m))
    metadata = outcomes[0]['diagnostic']; counts = metadata['counts']
    assert outcomes[0]['state'] == 'complete'
    assert metadata['shape_state'] == 'change_detected'
    sites = [r for r in records if r['type'] == 'site']
    clusters = [r for r in records if r['type'] == 'cluster']
    yellow = [r for r in records if r['type'] == 'yellow_cell']
    assert sum(c['site_count'] for c in clusters) == len(sites) == counts['sites'] == 8
    assert counts['thin_runs'] == sum(y['run_count'] for y in yellow)
    assert counts['thin_cells'] == len(yellow)
    assert counts['split'] == counts['merge_suspected'] == 2
    assert sum(r['site_count'] for r in records if r['type'] == 'event') == len(sites)
    for cluster in clusters:
        assert cluster['site_count'] == sum(s['cluster_id'] == cluster['cluster_id'] for s in sites)
    assert records[0]['diagnostic_schema_version'] == diag.VERSION
    assert records[-1]['type'] == 'summary'
    with Image.open(tmp_path / 'g4.png') as image:
        assert image.size == (1100, 1040) and image.mode == 'RGB'
        colors = np.asarray(image)
        assert (colors == COLORS['red']).all(axis=2).any()
        assert (colors == COLORS['overlap']).all(axis=2).any()
        # First target row is empty; no source thumbnail was painted there.
        assert image.getpixel((110, 110)) == (255, 255, 255)


def test_more_than_three_insets_explicit_omission(tmp_path):
    m = np.zeros((16, 16), dtype=bool); m[1, 1] = True
    p = product(m, other=tuple(Element(f'E{i}', f'L{i}', m) for i in range(5)))
    outcomes, records = render(tmp_path, p)
    summary = records[-1]
    assert summary['eligible_insets'] == 6
    assert summary['insets_shown'] == 3
    assert outcomes[0]['diagnostic']['insets_shown'] == 3
    assert outcomes[0]['diagnostic']['unshown_insets'] == 3
    assert summary['counts']['disappearance'] == 6


def test_hidden_layer_and_cnt_source_evidence_survives(tmp_path):
    tiny = np.zeros((16, 16), bool); tiny[1, 1] = True
    broad = np.ones_like(tiny)
    p = product(tiny, other=(Element('broad', 'B', broad),), cnt=broad)
    observations = list(diag.observation_descriptors(p))
    observations.append(diag.Observation('o4', 'cnt_source', tiny, layer='C',
                        source_asset_ids=('asset',), source_paths=('a.png', 'duplicate.png')))
    outcomes, records = render(tmp_path, p, descriptors=observations)
    assert outcomes[0]['diagnostic']['counts']['disappearance'] == 2
    assert [r['observation_id'] for r in records if r['type'] == 'event'] == ['o1', 'o4']
    assert all(r['composite_ambiguity_possible'] for r in records if r['type'] == 'event')
    with Image.open(tmp_path / 'g4.png') as image:
        assert image.getpixel((500, 500)) == COLORS['overlap']
    assert next(r for r in records if r['type'] == 'observation' and r['scope'] == 'cnt_source')['source_paths'] == ['a.png', 'duplicate.png']


def test_empty_missing_and_holes(tmp_path):
    m = np.zeros((7, 11), bool); m[3, :] = True
    valid = np.ones_like(m); valid[:, 5] = False
    p = product(m, other=(Element('empty', 'B', np.zeros_like(m)),
                          Element('missing', 'C', None, observation='missing')), valid=valid)
    outcomes, records = render(tmp_path, p, (11,))
    assert outcomes[0]['diagnostic']['shape_state'] == 'observation_incomplete'
    obs = [r for r in records if r['type'] == 'observation']
    assert [r['state'] for r in obs] == ['observed', 'observed_empty', 'missing']
    assert obs[0]['source_components'] == 2
    assert not any(r['type'] == 'event' for r in records)
    with Image.open(tmp_path / 'g11.png') as im:
        assert im.getpixel((30, 90)) == COLORS['padding']


def test_yellow_or_display_preserves_subpixel_cells():
    m = np.zeros((1000, 1000), bool); m[999, 999] = m[1, 1] = True
    projected = _project_mask(m)
    assert projected.sum() == 2 and projected[767, 767] and projected[0, 0]


def test_failure_order_cleanup_and_interrupt(tmp_path, monkeypatch):
    p = product(bridge())
    closed = []; references = []
    original = diag.measure_source
    def measure(*args):
        out = original(*args); references.append(weakref.ref(out.source)); return out
    monkeypatch.setattr(diag, 'measure_source', measure)
    class Sink(diag.NullSink):
        def __init__(self, grid):
            super().__init__(); self.grid = grid
        def finish(self, check):
            if self.grid == 8: raise OSError('render failed')
        def close(self): closed.append(self.grid)
    outcomes = list(diag.inspect_grid_diagnostics(p, grids=(4, 8, 16), sink_factory=lambda g, *_: Sink(g)))
    assert [o['grid'] for o in outcomes] == [4, 8]
    assert outcomes[1]['state'] == 'failed' and outcomes[1]['reason'] == 'render failed'
    assert closed == [4, 8, 16]
    gc.collect(); assert all(r() is None for r in references)
    def fail(*args): raise ValueError('source measurement failure')
    monkeypatch.setattr(diag, 'measure_source', fail)
    with pytest.raises(ValueError, match='source measurement'):
        list(diag.inspect_grid_diagnostics(p, grids=(4,), sink_factory=lambda g, *_: Sink(g)))
    assert closed[-1] == 4
    def interrupt(*args): raise KeyboardInterrupt
    monkeypatch.setattr(diag, 'measure_source', interrupt)
    with pytest.raises(KeyboardInterrupt):
        list(diag.inspect_grid_diagnostics(p, grids=(4,), sink_factory=lambda g, *_: Sink(g)))


def test_actual_second_render_failure(tmp_path, monkeypatch):
    original = DiagnosticSink._render
    def fail(self, *args):
        if self.transform.grid == 8: raise OSError('injected renderer failure')
        return original(self, *args)
    monkeypatch.setattr(DiagnosticSink, '_render', fail)
    outcomes, _ = render(tmp_path, product(bridge()), (4, 8, 16))
    assert [o['state'] for o in outcomes] == ['complete', 'failed']
    assert (tmp_path / 'g4.png').is_file()
    assert not (tmp_path / 'g8.png').exists() and not (tmp_path / 'g16.png').exists()


def test_single_observation_fills_three_inset_slots(tmp_path):
    m = np.zeros((16, 16), dtype=bool)
    m[1::4, 1] = True
    outcomes, records = render(tmp_path, product(m))
    assert records[-1]['eligible_insets'] == 4
    assert records[-1]['insets_shown'] == 3
    assert records[-1]['unshown_insets'] == 1
    assert outcomes[0]['diagnostic']['counts']['disappearance'] == 4


@pytest.mark.parametrize('shape,grid', [((7, 11), 4), ((3, 5), 8), ((2, 2), 1)])
def test_every_support_edge_matches_independent_bin_enumeration(shape, grid):
    rng = np.random.default_rng(819)
    for _ in range(8):
        m = rng.random(shape) < .55
        source, result, _ = analyze(m, grid)
        nh, nw = result.transform.resized_shape_yx
        left, top, _, _ = result.transform.padding_ltrb
        h, w = shape
        expected = set()
        for y in range(nh):
            for x in range(nw):
                target_id = result.target.image[y + top, x + left]
                if not target_id:
                    continue
                x0, x1 = x * w // nw, ((x + 1) * w + nw - 1) // nw
                y0, y1 = y * h // nh, ((y + 1) * h + nh - 1) // nh
                for source_id in np.unique(source.labels.image[y0:y1, x0:x1]):
                    if source_id:
                        expected.add((int(source_id), int(target_id)))
        assert set(map(tuple, result.edges[:, :2])) == expected


def test_failed_grid_releases_observation_arrays_before_next_mask(monkeypatch):
    original = diag.measure_source
    references = []
    def measure(*args):
        assert all(reference() is None for reference in references)
        out = original(*args)
        references.extend((weakref.ref(out.source), weakref.ref(out.labels.image)))
        return out
    monkeypatch.setattr(diag, 'measure_source', measure)
    class Failing(diag.NullSink):
        def observation(self, *args): raise OSError('stream write failed')
    m = bridge()
    p = product(m, other=(Element('other', 'B', m),))
    result = list(diag.inspect_grid_diagnostics(p, grids=(4, 8), sink_factory=lambda *_: Failing()))
    assert result == [{'grid': 4, 'state': 'failed', 'error_type': 'OSError', 'reason': 'stream write failed'}]
    assert all(reference() is None for reference in references)


def test_early_close_and_progress_exceptions_close_all_sinks(tmp_path):
    p = product(bridge())
    sinks = []
    def factory(g, transform, valid):
        sink = DiagnosticSink(tmp_path / f'{g}.png', tmp_path / f'{g}.jsonl',
                              model_id=p.product_id, transform=transform, valid=valid)
        sinks.append(sink)
        return sink
    iterator = diag.inspect_grid_diagnostics(p, grids=(4, 8), sink_factory=factory)
    assert next(iterator)['state'] == 'complete'
    iterator.close()
    assert all(sink._file.closed and sink.electrode is None and not sink.insets for sink in sinks)
    class ProgressError(Exception): pass
    class Failing(diag.NullSink):
        def observation(self, *args): raise ProgressError('progress callback failed')
    with pytest.raises(ProgressError, match='progress callback failed'):
        list(diag.inspect_grid_diagnostics(p, grids=(4,), sink_factory=lambda *_: Failing(),
                                          propagate_exceptions=(ProgressError,)))


def test_no_requested_grids_does_no_source_work(monkeypatch):
    monkeypatch.setattr(diag, 'measure_source', lambda *_: pytest.fail('unrequested source work'))
    assert list(diag.inspect_grid_diagnostics(product(bridge()))) == []


def test_sink_constructor_header_failure_closes_resources(tmp_path, monkeypatch):
    sinks = []
    original = DiagnosticSink._write
    def fail(self, record_type, **fields):
        if record_type == 'header':
            sinks.append(self)
            raise OSError('header write failed')
        return original(self, record_type, **fields)
    monkeypatch.setattr(DiagnosticSink, '_write', fail)
    p = product(bridge())
    def factory(g, transform, valid):
        return DiagnosticSink(tmp_path / 'failure.png', tmp_path / 'failure.jsonl',
                              model_id=p.product_id, transform=transform, valid=valid)
    outcomes = list(diag.inspect_grid_diagnostics(p, grids=(4,), sink_factory=factory))
    assert outcomes[0]['state'] == 'failed' and outcomes[0]['reason'] == 'header write failed'
    assert len(sinks) == 1 and sinks[0]._file.closed and sinks[0].electrode is None
    with pytest.raises(ValueError):
        sinks[0].rings.getpixel((0, 0))


@pytest.fixture(scope='module')
def sealed_diagnostic_bundle(tmp_path_factory):
    from psdl1 import import_png as importer
    from test_png_streaming import import_options
    directory=tmp_path_factory.mktemp('sealed-diagnostics')
    options=import_options(directory,no_preview=True)
    _,code=importer.run_import(options)
    assert code==0
    return options.out


@pytest.mark.parametrize('mutation', ['png_bytes','missing_ledger','version','header_model','header_grid',
    'header_transform','traversal','duplicate_reference','extra_metadata','counts','unknown_type',
    'extra_record_field','nonfinite','negative_coordinate','orphan','dimensions','summary_counts',
    'missing_cluster','duplicate_site','unknown_report_field','extra_directory'])
def test_strict_diagnostic_bundle_tampering(sealed_diagnostic_bundle,tmp_path,mutation):
    import shutil,struct,zlib
    from psdl1 import import_png as importer
    root=tmp_path/'copy';shutil.copytree(sealed_diagnostic_bundle,root)
    report=json.loads((root/'import-report.json').read_text())
    row=report['grid_evidence'][0];artifact=row['diagnostic']
    ledger=root/artifact['evidence'];png=root/artifact['png']
    records=[json.loads(line) for line in ledger.read_text().splitlines()]
    if mutation=='png_bytes':
        png.write_bytes(png.read_bytes()[:-10]+b'broken-png')
        with pytest.raises(ValueError):importer.verify_bundle(root)
        return
    elif mutation=='missing_ledger':ledger.unlink()
    elif mutation=='version':artifact['diagnostic_schema_version']='future'
    elif mutation=='header_model':records[0]['model_id']='another-model'
    elif mutation=='header_grid':records[0]['grid']+=1
    elif mutation=='header_transform':records[0]['transform']['padding_ltrb'][0]+=1
    elif mutation=='traversal':artifact['png']='../outside.png'
    elif mutation=='duplicate_reference':report['grid_evidence'].append(row.copy())
    elif mutation=='extra_metadata':artifact['unrecognized']=1
    elif mutation=='counts':artifact['counts']['thin_runs']+=1
    elif mutation=='unknown_type':records[1]['type']='invented'
    elif mutation=='extra_record_field':records[1]['invented']=0
    elif mutation in ('nonfinite','negative_coordinate'):
        site=next(r for r in records if r['type']=='site')
        site['grid_xy'][0]=float('inf') if mutation=='nonfinite' else -1
    elif mutation=='orphan':shutil.copyfile(png,root/'diagnostics'/'orphan.png')
    elif mutation=='dimensions':
        raw=bytearray(png.read_bytes());raw[16:20]=struct.pack('>I',999)
        raw[29:33]=struct.pack('>I',zlib.crc32(raw[12:29]));png.write_bytes(raw)
    elif mutation=='summary_counts':records[-1]['counts']['thin_runs']+=1
    elif mutation=='missing_cluster':records=[r for r in records if r['type']!='cluster']
    elif mutation=='duplicate_site':
        index=next(i for i,r in enumerate(records) if r['type']=='site');records.insert(index,records[index].copy())
    elif mutation=='unknown_report_field':report['invented']=0
    elif mutation=='extra_directory':(root/'diagnostics'/'unused').mkdir()
    if mutation!='missing_ledger':
        ledger.write_text(''.join(json.dumps(r)+'\n' for r in records))
        artifact['evidence_sha256']=importer.file_sha256(ledger)
    if mutation!='traversal':artifact['png_sha256']=importer.file_sha256(png)
    importer._write_json(root/'import-report.json',report);importer._seal_bundle(root)
    with pytest.raises(ValueError):importer.verify_bundle(root)


def test_old_v3_without_diagnostics_remains_readable(sealed_diagnostic_bundle,tmp_path):
    import shutil
    from psdl1 import import_png as importer
    root=tmp_path/'old';shutil.copytree(sealed_diagnostic_bundle,root)
    report=json.loads((root/'import-report.json').read_text())
    for row in report['grid_evidence']:row.pop('diagnostic',None)
    shutil.rmtree(root/'diagnostics')
    importer._write_json(root/'import-report.json',report);importer._seal_bundle(root)
    assert importer.verify_bundle(root)['status']=='verified'


@pytest.mark.parametrize('failure',['interrupt','diagnostic_progress'])
def test_integrated_diagnostic_failure_never_publishes(tmp_path,monkeypatch,failure):
    from psdl1 import import_png as importer
    from test_png_streaming import import_options
    options=import_options(tmp_path,no_preview=True,export=True)
    progress=None
    if failure=='second_render':
        original=DiagnosticSink._render
        def fail(self,*args):
            if self.transform.grid==32:raise OSError('second render failed')
            return original(self,*args)
        monkeypatch.setattr(DiagnosticSink,'_render',fail)
    elif failure=='source_measurement':
        def fail(*args):raise OSError('source measurement failed')
        monkeypatch.setattr(diag,'measure_source',fail)
    elif failure=='interrupt':
        def fail(*args):raise KeyboardInterrupt
        monkeypatch.setattr(diag,'measure_source',fail)
    else:
        def progress(event):
            if event['stage']=='diagnostic':raise ValueError('diagnostic transport failed')
    with pytest.raises((RuntimeError,KeyboardInterrupt)):
        importer.run_import(options,progress_sink=progress)
    assert not options.out.exists()
    stage,=tmp_path.glob('.out.staging-*')
    assert not (stage/'manifest.json').exists()
    if failure=='second_render':
        report=json.loads((stage/'import-report.json').read_text())
        assert [r['purpose'] for r in report['grid_evidence']]==['inspect_only_not_approved','inspect_failed_not_approved']
        assert report['grid_evidence'][0]['grid_check']['grid']==16
        assert report['grid_evidence'][1]['diagnostic']['state']=='not_evaluated'


def test_candidate_grid_alone_emits_no_diagnostics(tmp_path):
    from psdl1 import import_png as importer
    from test_png_streaming import import_options
    options=replace(import_options(tmp_path,export=True,no_preview=True),inspect_grids=())
    _,code=importer.run_import(options)
    assert code==0 and not (options.out/'diagnostics').exists()
    report=json.loads((options.out/'import-report.json').read_text())
    assert all('diagnostic' not in row for row in report['grid_evidence'])


@pytest.mark.parametrize('relations',[False,True])
@pytest.mark.parametrize('no_preview',[False,True])
@pytest.mark.parametrize('export',[False,True])
def test_automatic_unique_diagnostics_all_import_paths(tmp_path,relations,no_preview,export):
    from psdl1 import import_png as importer
    from test_png_streaming import import_options
    options=import_options(tmp_path,relations=relations,no_preview=no_preview,export=export)
    assert importer.run_import(options)[1]==0
    report=json.loads((options.out/'import-report.json').read_text())
    rows=[r for r in report['grid_evidence'] if r['purpose']=='inspect_only_not_approved']
    assert [r['grid_check']['grid'] for r in rows]==[16,32]
    assert len(list((options.out/'diagnostics').rglob('*.png')))==2
    assert len(list((options.out/'diagnostics').rglob('*.jsonl')))==2
    assert (options.out/'preview.html').exists()==(not no_preview)
    assert (options.out/'manifest.json').exists()==export
    assert all('diagnostic' not in r for r in report['grid_evidence'] if r['purpose']=='export_candidate_not_approved')
    assert importer.verify_bundle(options.out)['status']=='verified'


@pytest.mark.parametrize('cnt',['absent','empty'])
def test_missing_cnt_is_not_observed_empty(tmp_path,cnt):
    from png_helpers import write_model
    from psdl1 import import_png as importer
    from psdl1.png_contract import ImportOptions
    specs=[('E','1-L.png',((501,501),))]
    if cnt=='empty':specs.append(('cno','2-C.png',()))
    write_model(tmp_path/'source',specs,model='M',shape=(1001,1003))
    options=ImportOptions(root=tmp_path/'source',out=tmp_path/'out',inspect_grids=(16,),no_preview=True)
    assert importer.run_import(options)[1]==0
    report=json.loads((options.out/'import-report.json').read_text())
    artifact=report['grid_evidence'][0]['diagnostic']
    records=[json.loads(line) for line in (options.out/artifact['evidence']).read_text().splitlines()]
    cnt_observations=[r for r in records if r['type']=='observation' and r['scope']!='electrode']
    assert [r['state'] for r in cnt_observations]==(['observed_empty','observed_empty'] if cnt=='empty' else [])


@pytest.mark.parametrize('export',[False,True])
@pytest.mark.parametrize('failure',['normalize','render'])
def test_in_evaluator_grid_failure_publishes_verified_failure_bundle(tmp_path,monkeypatch,export,failure):
    from psdl1 import import_png as importer
    from test_png_streaming import import_options
    options=replace(import_options(tmp_path,no_preview=True,export=export),inspect_grids=(16,32,64))
    if failure=='normalize':
        original=data.normalize_unit
        def fail(mask,grid):
            if grid==32:raise MemoryError('grid 32 failed')
            return original(mask,grid)
        monkeypatch.setattr(data,'normalize_unit',fail)
    else:
        original=DiagnosticSink._render
        def fail(self,*args):
            if self.transform.grid==32:raise OSError('grid 32 failed')
            return original(self,*args)
        monkeypatch.setattr(DiagnosticSink,'_render',fail)
    progress=[]
    _,code=importer.run_import(options,progress_sink=progress.append)
    assert [(r['grid'],r['state']) for r in progress if r['stage']=='diagnostic' and r['state']!='advanced']==[
        (16,'started'),(32,'started'),(64,'started'),(16,'completed'),(32,'failed'),(64,'skipped')]
    assert code==2 and importer.verify_bundle(options.out)['status']=='verified'
    report=json.loads((options.out/'import-report.json').read_text())
    assert [r['purpose'] for r in report['grid_evidence']]==['inspect_only_not_approved','inspect_failed_not_approved']
    assert report['grid_evidence'][0]['grid_check']['grid']==16
    assert not (options.out/'manifest.json').exists()
    assert sorted(p.name for p in (options.out/'diagnostics').rglob('*') if p.is_file())==['grid-16.jsonl','grid-16.png']


def test_inset_selection_kind_coverage_then_observation_diversity():
    from psdl1._png_diagnostic_render import _select_insets
    def item(kind,obs,index):return {'key':(kind,obs,str(index)),'observation_id':f'o{obs}'}
    candidates=[item(0,1,1),item(0,1,2),item(0,2,3),item(1,1,4),item(2,1,5)]
    assert [x['key'][0] for x in _select_insets(candidates)]==[0,1,2]
    selected=_select_insets(candidates[:3]+[item(0,3,6)])
    assert {x['observation_id'] for x in selected}=={'o1','o2','o3'}


def test_split_witnesses_follow_lost_bridge_material():
    measured,result,events=analyze(bridge(),4)
    event,sites=events[0]
    assert event['lost_source_counts']==[8]
    assert [s['source_xy'] for s in sites]==[[3,5],[12,5]]
    assert all(s['lost_source_xy']==[4,5] for s in sites)
    assert result.lost_counts.sum()==8


def test_asymmetric_render_registration_and_exact_two_pixel_boundary(tmp_path):
    m=np.zeros((16,16),bool);m[:8,8:]=True
    outcomes,records=render(tmp_path,product(m),(4,))
    assert outcomes[0]['grid_check']['elements'][0]['estimated_run_at_grid_px']==2
    assert not outcomes[0]['grid_check']['elements'][0]['thin_line_warning']
    assert not any(r['type']=='yellow_cell' for r in records)
    with Image.open(tmp_path/'g4.png') as image:
        assert image.getpixel((20+672,84+96))==COLORS['electrode']
        assert image.getpixel((20+96,84+672))==(255,255,255)


@pytest.mark.parametrize('mutation',['drop_yellow','source_box','grid_xy','shape_state'])
def test_verifier_direct_conservation_and_witness_invariants(sealed_diagnostic_bundle,tmp_path,mutation):
    import shutil
    from psdl1 import import_png as importer
    root=tmp_path/'copy';shutil.copytree(sealed_diagnostic_bundle,root)
    report=json.loads((root/'import-report.json').read_text());row=report['grid_evidence'][0]
    path=root/row['diagnostic']['evidence'];records=[json.loads(line) for line in path.read_text().splitlines()]
    if mutation=='drop_yellow':
        index=next(i for i,r in enumerate(records) if r['type']=='yellow_cell');records.pop(index)
    elif mutation=='source_box':
        event=next(r for r in records if r['type']=='event');event['source_bounds'][0]=[0,0,1,1]
    elif mutation=='grid_xy':
        site=next(r for r in records if r['type']=='site');site['grid_xy'][0]+=.01
        for cluster in (r for r in records if r['type']=='cluster'):
            positions=np.array([r['grid_xy'] for r in records if r['type']=='site' and r['cluster_id']==cluster['cluster_id']])*768/row['transform']['grid']
            cluster['display_bounds_xyxy']=[*positions.min(axis=0),*positions.max(axis=0)]
    else:
        row['diagnostic']['shape_state']='no_change_detected'
        records[-1]['shape_state']='no_change_detected'
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    row['diagnostic']['evidence_sha256']=importer.file_sha256(path)
    importer._write_json(root/'import-report.json',report);importer._seal_bundle(root)
    with pytest.raises(ValueError,match='summary counts' if mutation=='drop_yellow' else None):importer.verify_bundle(root)


def test_verifier_rejects_mixed_diagnostic_generations(sealed_diagnostic_bundle,tmp_path):
    import shutil
    from psdl1 import import_png as importer
    root=tmp_path/'copy';shutil.copytree(sealed_diagnostic_bundle,root)
    report=json.loads((root/'import-report.json').read_text())
    artifact=report['grid_evidence'][1].pop('diagnostic')
    for field in ('png','evidence'):(root/artifact[field]).unlink()
    importer._write_json(root/'import-report.json',report);importer._seal_bundle(root)
    with pytest.raises(ValueError,match='mixed diagnostic'):importer.verify_bundle(root)


@pytest.mark.parametrize('bad',['yellow_pad','target_pad','min_length'])
def test_ledger_content_bounds_and_minimum_run(tmp_path,bad):
    m=np.zeros((16,32),bool)
    if bad=='target_pad':m[:,:16]=bridge(True)
    else:m[:4,:4]=True
    outcomes,records=render(tmp_path,product(m),(8,))
    row={'model_id':'synthetic','transform':outcomes[0]['grid_check']['transform'],
         'grid_check':outcomes[0]['grid_check']}
    if bad=='target_pad':
        record=next(r for r in records if r['type']=='site' and r['target_xy'] is not None)
        record['target_xy'][1]=0;record['grid_xy'][1]=.5
    else:
        record=next(r for r in records if r['type']=='yellow_cell')
        if bad=='yellow_pad':record['cell_xy'][1]=0
        else:record['min_length_px']=1
    path=tmp_path/'altered.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises(ValueError,match='padding' if bad!='min_length' else None):
        diag.verify_diagnostic_ledger(path,outcomes[0]['diagnostic'],row,'image_only')


def _lexsort_correspondence_oracle(source_labels, target_labels, transform):
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
        order=np.lexsort((tx,ty,x[ii],y[ii],proximity,key))
        sorted_key=key[order]
        chosen=order[np.r_[True,sorted_key[1:]!=sorted_key[:-1]]] if len(order) else order
        chunks.append(np.column_stack((ss[chosen],tt[chosen],x[ii[chosen]],y[ii[chosen]],
                                       tx[chosen],ty[chosen],proximity[chosen])))
    if not chunks or not any(len(chunk) for chunk in chunks):
        return np.empty((0,6),dtype=np.int32),lost_counts,lost_xy
    edges=np.concatenate(chunks)
    order=np.lexsort((edges[:,4],edges[:,5],edges[:,2],edges[:,3],edges[:,6],edges[:,1],edges[:,0]))
    edges=edges[order];keys=edges[:,0]*base+edges[:,1]
    return edges[np.r_[True,keys[1:]!=keys[:-1]],:6].astype(np.int32),lost_counts,lost_xy


@pytest.mark.parametrize('grid',[4,8,17])
def test_grouped_witnesses_match_previous_lexsort(grid):
    rng=np.random.default_rng(42)
    masks=[np.ones((31,43),bool),bridge(True)]
    masks.extend(rng.random((23,31))<density for density in (.1,.5,.9))
    thin=np.zeros((29,37),bool);thin[::4,:]=True;thin[:,::5]=True
    masks.append(thin)
    for mask in masks:
        source=diag.measure_source(mask,np.ones_like(mask))
        result=diag.measure_grid(source,grid)
        expected=_lexsort_correspondence_oracle(source.labels,result.target,result.transform)
        for actual,want in zip((result.edges,result.lost_counts,result.lost_xy),expected):
            np.testing.assert_array_equal(actual,want)


@pytest.mark.parametrize('existing',['ledger','png'])
def test_unfinished_sink_preserves_unowned_file(tmp_path,existing):
    png,ledger=tmp_path/'g4.png',tmp_path/'g4.jsonl'
    path=ledger if existing=='ledger' else png
    path.write_bytes(b'unowned sentinel')
    if existing=='ledger':
        transform=data.frame_transform((16,16),4)
        with pytest.raises(FileExistsError):
            DiagnosticSink(png,ledger,model_id='synthetic',transform=transform,valid=np.ones((16,16),bool))
        assert not png.exists()
    else:
        outcomes=list(diag.inspect_grid_diagnostics(product(bridge()),grids=(4,),
            sink_factory=lambda g,t,v:DiagnosticSink(png,ledger,model_id='synthetic',transform=t,valid=v)))
        assert outcomes[0]['state']=='failed' and outcomes[0]['error_type']=='FileExistsError'
        assert not ledger.exists()
    assert path.read_bytes()==b'unowned sentinel'


def test_hash_failure_after_render_removes_unfinished_pair(tmp_path,monkeypatch):
    from psdl1 import _png_diagnostic_render as renderer
    sinks=[]
    def factory(g,t,v):
        sink=DiagnosticSink(tmp_path/'g4.png',tmp_path/'g4.jsonl',model_id='synthetic',transform=t,valid=v)
        sinks.append(sink);return sink
    def fail_hash(path):
        assert (tmp_path/'g4.png').is_file() and (tmp_path/'g4.jsonl').is_file()
        assert not sinks[0]._finished
        raise OSError('hash failed after render')
    monkeypatch.setattr(renderer,'_sha256',fail_hash)
    outcomes=list(diag.inspect_grid_diagnostics(product(bridge()),grids=(4,),sink_factory=factory))
    assert outcomes[0]['state']=='failed' and outcomes[0]['reason']=='hash failed after render'
    assert not (tmp_path/'g4.png').exists() and not (tmp_path/'g4.jsonl').exists()
    assert tmp_path.is_dir()


def test_sink_preserves_preexisting_empty_directories(tmp_path):
    parent=tmp_path/'diagnostics'/'empty_model';parent.mkdir(parents=True)
    sink=DiagnosticSink(parent/'g4.png',parent/'g4.jsonl',model_id='synthetic',
        transform=data.frame_transform((16,16),4),valid=np.ones((16,16),bool))
    sink.close()
    assert parent.is_dir() and parent.parent.is_dir()
    assert list(parent.iterdir())==[]


def test_sibling_sinks_remove_only_owned_empty_directories(tmp_path,monkeypatch):
    parent=tmp_path/'diagnostics'/'new_model'
    def fail(*args):raise OSError('normalization failed')
    monkeypatch.setattr(data,'normalize_unit',fail)
    outcomes=list(diag.inspect_grid_diagnostics(product(bridge()),grids=(4,8),
        sink_factory=lambda g,t,v:DiagnosticSink(parent/f'g{g}.png',parent/f'g{g}.jsonl',
            model_id='synthetic',transform=t,valid=v)))
    assert outcomes[0]['state']=='failed'
    assert not parent.parent.exists() and tmp_path.is_dir()


def test_lost_representative_is_farthest_instead_of_first():
    mask=np.zeros((32,32),bool)
    mask[8:16,:8]=mask[8:16,24:]=True;mask[11,8:24]=True
    source,result,events=analyze(mask,8)
    assert result.lost_counts[1]==16
    # First lost pixel is (8,11); midpoint cell distance selects (12,11).
    np.testing.assert_array_equal(result.lost_xy[1],[12,11])
    assert events[0][0]['kind']=='split'
    assert all(site['lost_source_xy']==[12,11] for site in events[0][1])
    assert [site['source_xy'] for site in events[0][1]]==[[7,11],[24,11]]


@pytest.mark.parametrize('seed,component,expected,ties',[
    (4,24,[11,12],[[11,12],[8,13],[10,13]]),
    (91,15,[9,8],[[9,8],[11,8]]),
])
def test_nearest_witness_ties_use_y_then_x(seed,component,expected,ties):
    mask=np.random.default_rng(seed).random((16,16))<.55
    source,result,_=analyze(mask,4)
    edge=next(edge for edge in result.edges if edge[0]==component)
    candidates=np.array(ties)
    squared=((candidates-result.lost_xy[component])**2).sum(axis=1)
    assert np.all(squared==squared[0])
    assert np.all(source.labels.image[candidates[:,1],candidates[:,0]]==component)
    assert np.all(result.target.image[candidates[:,1]//4,candidates[:,0]//4]==edge[1])
    np.testing.assert_array_equal(edge[2:4],expected)


@pytest.mark.parametrize('mutation',['source_xy','grid_xy'])
def test_verifier_checks_split_witness_source_box_and_target_centre(tmp_path,mutation):
    outcomes,records=render(tmp_path,product(bridge()),(4,))
    event=next(record for record in records if record['type']=='event' and record['kind']=='split')
    site=next(record for record in records if record['type']=='site' and record['event_id']==event['event_id'])
    if mutation=='source_xy':
        site['source_xy']=[0,0]  # valid frame coordinate, outside this component only
        message='source witness outside component bounds'
    else:
        site['grid_xy'][0]+=.01  # split branch, valid frame and unchanged cluster
        message='target witness centre mismatch'
    path=tmp_path/'tampered.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    row={'model_id':'synthetic','transform':outcomes[0]['grid_check']['transform'],
        'grid_check':outcomes[0]['grid_check']}
    with pytest.raises(ValueError,match=message):
        diag.verify_diagnostic_ledger(path,outcomes[0]['diagnostic'],row,'image_only')
