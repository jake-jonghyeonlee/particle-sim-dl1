"""Streaming foundations: snapshot consistency, scoped measurements and strict reads."""
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
import gc
import weakref

import numpy as np
import pytest

from psdl1 import data, png_source as source, png_connections as connections
from psdl1.schema import Element, Product
from png_helpers import write_model, write_png, write_stack_order


def assert_no_arrays(value):
    assert not isinstance(value, np.ndarray)
    if is_dataclass(value):
        for field in fields(value):
            assert_no_arrays(getattr(value, field.name))
    elif isinstance(value, dict):
        for item in value.values():
            assert_no_arrays(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            assert_no_arrays(item)


def snapshots(root):
    scanner = source._SourceScanner(root)
    models, extractions = [], []
    for model in scanner:
        extraction = connections.extract_model(model)
        models.append(source._detach_model(replace(model, resource_estimate=extraction.resource_estimate)))
        extractions.append(connections._detach_extraction(extraction))
        del model, extraction
    return scanner.finish(models), tuple(extractions)


def simple_tree(root):
    for model in ('A', 'B'):
        write_model(root, [('E', '1-L.png', ((501, 501),)),
                           ('cno', '2-C.png', ((501, 501),)),
                           ('cno', '002_C.png', ((501, 501),))], model=model)


@pytest.mark.parametrize('relations', [False, True])
def test_stream_scan_matches_eager_inventory(tmp_path, relations):
    simple_tree(tmp_path / 'src')
    order = write_stack_order(tmp_path / 'order.json', ['L', 'C']) if relations else None
    eager = source.discover_source(tmp_path / 'src', stack_order=order)
    scanner = source._SourceScanner(tmp_path / 'src', stack_order=order)
    models = tuple(scanner)
    assert scanner.finish(models) == eager
    for got, expected in zip(models, eager.models):
        for a, b in zip(got.assets, expected.assets):
            np.testing.assert_array_equal(a.mask, b.mask)
            assert a.mask is not None
    with pytest.raises(ValueError, match='single-use'):
        tuple(scanner)


def test_detached_records_own_no_arrays(tmp_path):
    simple_tree(tmp_path)
    inventory, extractions = snapshots(tmp_path)
    assert_no_arrays(inventory)
    assert_no_arrays(extractions)
    for model, extraction in zip(inventory.models, extractions):
        with pytest.raises(ValueError, match='detached'):
            connections.extract_model(model)
        with pytest.raises(ValueError, match='detached'):
            connections.to_product(model, extraction)
        with pytest.raises(ValueError, match='detached'):
            connections.component_mask(extraction.components[0])
        with pytest.raises(ValueError, match='detached'):
            tuple(connections.iter_connections(extraction))
        with source.decoder_session() as policy:
            live, replay = connections._reload_extraction(tmp_path, model, extraction, policy)
        assert source._detach_model(live) == model
        assert connections._detach_extraction(replay) == extraction


def test_reload_checks_all_merged_cnt_provenance(tmp_path, monkeypatch):
    simple_tree(tmp_path)
    inventory, extractions = snapshots(tmp_path)
    model = inventory.models[0]
    merged, = model.cnt_assets
    assert len(merged.provenance_files) == 2
    decoded = []
    original = source._decode
    def decode(path, relative, policy):
        decoded.append(relative)
        return original(path, relative, policy)
    monkeypatch.setattr(source, '_decode', decode)
    with source.decoder_session() as policy:
        connections._reload_extraction(tmp_path, model, extractions[0], policy)
    assert decoded == [f['relative_path'] for a in model.assets for f in (a.provenance_files or ({'relative_path': a.relative_path},))]
    other = next(f for f in merged.provenance_files if f['relative_path'] != merged.relative_path)
    # Same geometry, changed compressed bytes must still invalidate the snapshot.
    write_png(tmp_path, other['relative_path'], pixels=((501, 501),), compress_level=0)
    with source.decoder_session() as policy, pytest.raises(ValueError, match='source changed'):
        connections._reload_extraction(tmp_path, model, extractions[0], policy)


@pytest.mark.parametrize('change', ['geometry', 'bytes', 'delete', 'escape', 'mode'])
def test_reload_rejects_changed_or_escaping_source(tmp_path, change):
    root = tmp_path / 'src'; simple_tree(root)
    inventory, extractions = snapshots(root)
    model = inventory.models[0]; asset = model.electrodes[0]
    path = root / asset.relative_path
    if change == 'geometry':
        write_png(root, asset.relative_path, pixels=((502, 501),))
    elif change == 'bytes':
        write_png(root, asset.relative_path, pixels=((501, 501),), compress_level=0)
    elif change == 'mode':
        write_png(root, asset.relative_path, pixels=((501, 501),), mode='RGB')
    elif change == 'delete':
        path.unlink()
    else:
        outside = tmp_path / 'outside.png'; outside.write_bytes(path.read_bytes())
        path.unlink(); path.symlink_to(outside)
    with source.decoder_session() as policy, pytest.raises(ValueError, match='source changed'):
        connections._reload_extraction(root, model, extractions[0], policy)


def test_reload_rejects_reextraction_evidence_change(tmp_path):
    simple_tree(tmp_path)
    inventory, extractions = snapshots(tmp_path)
    model, extraction = inventory.models[0], extractions[0]
    damaged = replace(extraction, source_reasons=('incorrect-pass-one-evidence',))
    with source.decoder_session() as policy, pytest.raises(ValueError, match='extraction evidence'):
        connections._reload_extraction(tmp_path, model, damaged, policy)


def test_streaming_global_id_collision_and_case_notices(tmp_path, monkeypatch):
    write_model(tmp_path, [('E', '1-L.png', ((501, 501),))], model='A')
    write_model(tmp_path, [('F', '1-l.png', ((501, 501),))], model='B')
    real = source.canonical_id
    monkeypatch.setattr(source, 'canonical_id', lambda kind, value: 'asset-collision' if kind == 'asset' else real(kind, value))
    inventory, _ = snapshots(tmp_path)
    assert inventory.models[0].valid
    assert not inventory.models[1].valid
    assert [e.reason for e in inventory.models[1].errors] == ['duplicate/colliding emitted identity; also A/E/1-L.png']
    assert 'names ["L", "l"], models ["A", "B"]' in inventory.warnings[0].reason


def test_scanner_releases_previous_model_before_next_decode(tmp_path, monkeypatch):
    simple_tree(tmp_path)
    old = source._decode; references = []
    def decode(path, relative, policy):
        if relative.startswith('B/'):
            gc.collect()
            assert all(r() is None for r in references)
        result = old(path, relative, policy)
        if relative.startswith('A/'):
            references.append(weakref.ref(result[0]))
        return result
    monkeypatch.setattr(source, '_decode', decode)
    inventory, _ = snapshots(tmp_path)
    assert len(inventory.models) == 2


def test_merged_cnt_releases_redundant_decode_before_next_file(tmp_path, monkeypatch):
    write_model(tmp_path, [('E', '1-L.png', ((501, 501),)),
                           ('cno', '001_C.png', ((501, 501),)),
                           ('cno', '002_C.png', ((501, 501),)),
                           ('cno', '003_C.png', ((501, 501),))])
    original = source._decode; redundant = []
    def decode(path, relative, policy):
        if relative.endswith('003_C.png'):
            assert redundant[0]() is None
        result = original(path, relative, policy)
        if relative.endswith('002_C.png'):
            redundant.append(weakref.ref(result[0]))
        return result
    monkeypatch.setattr(source, '_decode', decode)
    model, = source.discover_source(tmp_path).models
    assert len(model.cnt_assets[0].provenance_files) == 3


def effective_product():
    a = np.zeros((7, 11), bool); a[2:5, 1:4] = True; a[2:5, 7:10] = True; a[3, 4:7] = True
    b = np.zeros_like(a); b[1:6, 3:8] = True
    valid = np.ones_like(a); valid[:, 5] = False
    return Product('holes', (1.1, .7), (Element('A', 'L', a), Element('B', 'K', b),
                   Element('empty', 'Q', np.zeros_like(a)), Element('missing', 'R', None, observation='missing')),
                   cnt=b.copy(), valid=valid)


def test_grid_stats_once_per_effective_mask(monkeypatch, manifest_case):
    product = effective_product(); widths, components = [], []
    old_width, old_components = data.narrowest_run_px, data._components
    def width(mask):
        widths.append(mask.copy()); return old_width(mask)
    def count(mask):
        components.append(np.asarray(mask).shape); return old_components(mask)
    monkeypatch.setattr(data, 'narrowest_run_px', width)
    monkeypatch.setattr(data, '_components', count)
    checks = list(data.iter_grid_checks(product, [4, 8, 12]))
    assert len(checks) == 3 and len(widths) == 4
    assert len(components) == 4 + 4 * 3
    for got, mask in zip(widths, [product.elements[0].mask, product.elements[1].mask, product.elements[2].mask, product.cnt]):
        np.testing.assert_array_equal(got, mask * product.valid)
    assert not np.array_equal(widths[0], widths[1])
    assert checks[0]['elements'][0]['source_components'] == 2
    assert checks[0]['elements'][2]['source_active_pixels'] == 0
    assert checks[0]['elements'][3] == {'id': 'missing@R', 'observation': 'missing'}
    # Relations are separate logical observations even when their pixels match
    # an electrode or the global CNT. They must also measure once per call.
    _, write, _ = manifest_case
    relation_product, = data.load_manifest(write())
    widths.clear(); components.clear()
    relation_checks = list(data.iter_grid_checks(relation_product, [4, 8, 12]))
    assert len(widths) == 4 and len(components) == 16
    assert [e['id'] for e in relation_checks[0]['elements']] == ['A-L1', 'B-L2', 'V1', 'global_CNT']


def test_grid_stats_do_not_survive_product_mutation():
    p = effective_product()
    before = next(data.iter_grid_checks(p, [8]))['elements'][0]
    p.elements[0].mask[0, 0] = True
    changed = next(data.iter_grid_checks(p, [8]))['elements'][0]
    assert changed['source_active_pixels'] == before['source_active_pixels'] + 1
    assert changed['source_components'] == before['source_components'] + 1
    p.valid[0, 0] = False
    assert next(data.iter_grid_checks(p, [8]))['elements'][0] == before


def test_late_grid_failure_keeps_prior_evidence(monkeypatch):
    original = data.normalize_unit
    def normalize(mask, grid):
        if grid == 8:
            raise ValueError('late grid failed')
        return original(mask, grid)
    monkeypatch.setattr(data, 'normalize_unit', normalize)
    checks = data.iter_grid_checks(effective_product(), [4, 8, 12])
    first = next(checks)
    with pytest.raises(ValueError, match='late grid failed'):
        next(checks)
    assert first['grid'] == 4 and len(first['elements']) == 5


def two_products(manifest_case):
    doc, write, root = manifest_case
    doc['products'].append(deepcopy(doc['products'][0]))
    doc['products'][1]['product_id'] = 'Q'
    return doc, write, root


def test_manifest_iterator_matches_eager_products(manifest_case):
    _, write, _ = two_products(manifest_case)
    eager = data.load_manifest(write(), grid=8)
    streamed = list(data._iter_manifest_products(write(), grid=8))
    assert [data.input_identity(p) for p in streamed] == [data.input_identity(p) for p in eager]
    assert [p.grid_check for p in streamed] == [p.grid_check for p in eager]
    assert [p.product_id for p in streamed] == ['P', 'Q']
    assert streamed[0].grid_check['manifest_candidate_grid'] == 4


@pytest.mark.parametrize('error', ['duplicate', 'pitch', 'corrupt', 'duplicate-and-corrupt'])
def test_manifest_iterator_late_global_error(manifest_case, error):
    doc, write, root = two_products(manifest_case)
    last = doc['products'][-1]
    if error in ('duplicate', 'duplicate-and-corrupt'):
        last['product_id'] = 'P'
    elif error == 'pitch':
        last['frame']['pixel_pitch_um'] = .1
        last['unit_size_um'] = [.4, .4]
    if 'corrupt' in error:
        (root / 'corrupt.npy').write_bytes(b'not-npy')
        last['elements'][-1]['mask'] = 'corrupt.npy'
    iterator = data._iter_manifest_products(write())
    assert next(iterator).product_id == 'P'
    pattern = 'duplicate product_id' if error == 'duplicate' else 'same pixel pitch' if error == 'pitch' else 'allow_pickle'
    if 'corrupt' not in error:
        assert next(iterator).product_id == last['product_id']
    with pytest.raises(ValueError, match=pattern) as streamed_error:
        next(iterator)
    with pytest.raises(ValueError) as eager_error:
        data.load_manifest(write())
    assert str(streamed_error.value) == str(eager_error.value)


def test_manifest_iterator_releases_previous_product(manifest_case, monkeypatch):
    _, write, _ = two_products(manifest_case)
    references = []
    original = data._manifest_product
    def construct(*args):
        gc.collect()
        assert all(r() is None for r in references)
        product = original(*args)
        references.extend(weakref.ref(m) for m in [product.valid, product.cnt, *[e.mask for e in product.elements], *[c.mask for c in product.connections]])
        return product
    monkeypatch.setattr(data, '_manifest_product', construct)
    iterator = data._iter_manifest_products(write())
    first = next(iterator); del first
    second = next(iterator); del second
    with pytest.raises(StopIteration):
        next(iterator)
    assert all(r() is None for r in references)


# Importer-level streaming lifecycle cases (T3).
from dataclasses import replace as dc_replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import os

from psdl1 import import_png as importer, png_preview as preview
from psdl1.png_contract import ImportOptions


def import_options(tmp_path, *, relations=False, export=False, invalid=False, no_preview=False, model_count=1):
    root = tmp_path / 'source'
    for index in range(model_count):
        write_model(root, [('A','1-U.png',((501,501),)), ('B','3-L.png',((501,501),)),
                           ('cno','2-C.png',((501,501),(200,200)))], model=f'M{index}', shape=(1001,1003))
    if invalid:
        write_png(root, 'Z/A/1-U.png', shape=(9,11))
    metadata = tmp_path/'metadata.json'
    metadata.write_text(json.dumps({'metadata_schema_version':'dl1_png_metadata_v1','models': {
        name:{'lineage_id':'family-'+name,'design_version':'v1'} for name in [f'M{i}' for i in range(model_count)]}}))
    order = write_stack_order(tmp_path/'order.json',['U','C','L']) if relations else None
    return ImportOptions(root=root,out=tmp_path/'out',export=export,metadata=metadata,
        candidate_grid=16 if export else None,inspect_grids=(16,32,16),stack_order=order,no_preview=no_preview)


@pytest.mark.parametrize('relations', [False,True])
@pytest.mark.parametrize('operation', ['inspect','export','blocked'])
def test_no_preview_inspect_and_export_verify(tmp_path, monkeypatch, relations, operation):
    options=import_options(tmp_path,relations=relations,export=operation!='inspect',invalid=operation=='blocked',no_preview=True)
    def forbidden(*args, **kwargs):
        raise AssertionError('renderer must never be called')
    monkeypatch.setattr(preview,'render_model_preview',forbidden)
    monkeypatch.setattr(preview,'write_preview_index',forbidden)
    _,status=importer.run_import(options)
    assert status == (2 if operation=='blocked' else 0)
    report=json.loads((options.out/'import-report.json').read_text())
    assert report['preview']==[]
    assert [x['reason'] for x in report['diagnostics']].count('preview_skipped_by_request: --no-preview')==1
    assert not (options.out/'preview').exists() and not (options.out/'preview.html').exists()
    shutil.rmtree(options.root)
    assert importer.verify_bundle(options.out)['status']=='verified'
    assert (options.out/'manifest.json').exists() == (operation=='export')


def test_progress_stages_ordered_and_stdout_unchanged(tmp_path):
    options=import_options(tmp_path,relations=True,export=True,model_count=2)
    run=subprocess.run([sys.executable,'-m','psdl1.import_png','--root',str(options.root),'--out',str(options.out),
        '--export','--candidate-grid','16','--inspect-grids','32','16','32','--metadata',str(options.metadata),
        '--stack-order',str(options.stack_order)],capture_output=True,text=True,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'))
    assert run.returncode==0,run.stderr
    stdout=[json.loads(line) for line in run.stdout.splitlines()]
    assert [event['event'] for event in stdout]==['resource_inventory','completed']
    events=[json.loads(line) for line in run.stderr.splitlines()]
    assert events and all(e['event']=='progress' for e in events)
    assert events[0]=={'event':'progress','phase':'inventory','stage':'inventory','state':'started'}
    for model in ['M0','M1']:
        grids=[(e.get('grid'),e['state']) for e in events if e['phase']=='inspect_preview' and e['stage']=='grid' and e.get('model_id')==model]
        assert grids==[(16,'started'),(16,'completed'),(32,'started'),(32,'completed')]
        for phase in ['inventory','inspect_preview','export','validation']:
            marks=[e['state'] for e in events if e['stage']=='model' and e['phase']==phase and e.get('model_id')==model]
            assert marks==['started','completed']
    assert [e for e in events if e['stage']=='preview_index'][-1]['state']=='completed'
    assert events[-1]=={'event':'progress','phase':'publication','stage':'complete','state':'completed'}
    assert next(i for i,e in enumerate(events) if e['stage']=='reading') < next(i for i,e in enumerate(events) if e['stage']=='grid')
    reads=[e for e in events if e['stage']=='reading' and e['state']=='completed']
    assert len(reads)==6 and all(e['files_done']==e['files_total']==3 for e in reads)


def test_progress_reading_precedes_decode_and_file_batches(tmp_path,monkeypatch):
    root=tmp_path/'source'
    for n in range(102):
        write_png(root,f'M/A/{n+1}-L{n}.png',shape=(2,3))
    events=[];original=source._decode
    def decode(*args):
        assert any(e['stage']=='reading' and e['state']=='started' for e in events)
        return original(*args)
    monkeypatch.setattr(source,'_decode',decode)
    importer.run_import(ImportOptions(root=root,out=tmp_path/'out',no_preview=True),progress_sink=events.append)
    marks=[e for e in events if e['phase']=='inventory' and e['stage']=='reading']
    assert [(e['state'],e.get('files_done')) for e in marks]==[('started',None),('advanced',1),('advanced',100),('completed',102)]
    assert marks[-1]['files_total']==102


def test_no_preview_verify_exclusivity_and_strict_report_fields(tmp_path):
    options=import_options(tmp_path,no_preview=True)
    importer.run_import(options)
    with pytest.raises(ValueError,match='alone'):
        importer.run_import(ImportOptions(verify_bundle=options.out,no_preview=True))
    with pytest.raises(SystemExit):
        importer.main(['--verify-bundle',str(options.out),'--no-preview'])
    for value in [0,1,None,'false']:
        with pytest.raises(ValueError,match='bool'):
            ImportOptions(no_preview=value)
    report_path=options.out/'import-report.json';original=json.loads(report_path.read_text())
    for mutation in ['field','preview','duplicate','directory']:
        report=deepcopy(original)
        if mutation=='field':report['preview_status']='skipped'
        elif mutation=='preview':report['preview']=[{'fake':True}]
        elif mutation=='duplicate':report['diagnostics'].append(report['diagnostics'][-1])
        else:(options.out/'preview').mkdir()
        importer._write_json(report_path,report);importer._seal_bundle(options.out)
        with pytest.raises(ValueError,match='fields|skipped-preview'):
            importer.verify_bundle(options.out)
        if mutation=='directory':(options.out/'preview').rmdir()


@pytest.mark.parametrize('change',['bytes','delete','merged','escape'])
def test_reload_source_change_never_publishes(tmp_path,change):
    options=import_options(tmp_path,export=True,no_preview=True)
    if change=='merged':
        write_png(options.root,'M0/cno/002_C.png',shape=(1001,1003),pixels=((501,501),(200,200)))
    def mutate(event):
        if event['event']!='resource_inventory':return
        relative='M0/cno/2-C.png' if change=='merged' else 'M0/A/1-U.png'
        path=options.root/relative
        if change in ('bytes','merged'):
            write_png(options.root,relative,shape=(1001,1003),pixels=((501,501),(200,200)) if change=='merged' else ((501,501),),compress_level=0)
        elif change=='delete':path.unlink()
        else:
            outside=tmp_path/'outside.png';outside.write_bytes(path.read_bytes());path.unlink();path.symlink_to(outside)
    with pytest.raises(RuntimeError,match='source changed'):
        importer.run_import(options,event_sink=mutate)
    stage,=tmp_path.glob('.out.staging-*')
    assert not options.out.exists() and not (stage/'manifest.json').exists()
    report=json.loads((stage/'import-report.json').read_text())
    assert report['export_status']=='failed'
    assert all(a['mask_sha256'] for m in report['inventory']['models'] for a in m['assets'])


def test_preview_final_warnings_include_later_models(tmp_path,monkeypatch):
    options=import_options(tmp_path,export=True,model_count=2)
    original=importer.inspect_grid_diagnostics
    def checks(product,descriptors,grids,sink_factory,**kwargs):
        descriptors=tuple(descriptors)
        for grid in grids:
            if product.product_id=='M1' and grid==32:raise ValueError('late <grid> failure')
            yield from original(product,descriptors,[grid],sink_factory,**kwargs)
    monkeypatch.setattr(importer,'inspect_grid_diagnostics',checks)
    _,code=importer.run_import(options)
    assert code==2
    report=json.loads((options.out/'import-report.json').read_text())
    assert [(e['model_id'],e['purpose']) for e in report['grid_evidence']]==[
        ('M0','inspect_only_not_approved'),('M0','inspect_only_not_approved'),
        ('M1','inspect_only_not_approved'),('M1','inspect_failed_not_approved')]
    html=(options.out/'preview.html').read_text()
    assert html.index('late &lt;grid&gt; failure') < html.index('<h2>M0</h2>')
    assert 'Possible alias/duplicate geometry' in html
    for row in report['preview']:
        assert (options.out/row['overview']).is_file()
    assert not (options.out/'assets').exists() and not (options.out/'manifest.json').exists()


@pytest.mark.parametrize('error',['duplicate','pitch','asset'])
def test_manifest_late_error_never_promotes(tmp_path,monkeypatch,error):
    options=import_options(tmp_path,export=True,no_preview=True,model_count=2)
    original=importer._write_json
    def write(path,payload):
        if path.name=='manifest.pending.json':
            last=payload['products'][-1]
            if error=='duplicate':last['product_id']=payload['products'][0]['product_id']
            elif error=='pitch':
                last['frame']['pixel_pitch_um']=.2
                last['unit_size_um']=[v*2 for v in last['unit_size_um']]
            else:(path.parent/last['elements'][-1]['mask']).write_bytes(b'corrupt')
        return original(path,payload)
    monkeypatch.setattr(importer,'_write_json',write)
    with pytest.raises(RuntimeError):importer.run_import(options)
    stage,=tmp_path.glob('.out.staging-*')
    assert (stage/'manifest.pending.json').is_file()
    assert not options.out.exists() and not (stage/'manifest.json').exists()


@pytest.mark.parametrize('phase',['grid','preview'])
def test_streaming_release_on_grid_and_preview_failure(tmp_path,monkeypatch,phase):
    options=import_options(tmp_path,no_preview=phase=='grid',model_count=2)
    refs=[];decode=source._decode
    def track(*args):
        result=decode(*args);refs.append(weakref.ref(result[0]));return result
    monkeypatch.setattr(source,'_decode',track)
    if phase=='grid':
        def fail(*args,**kwargs):raise ValueError('grid error')
        monkeypatch.setattr(importer,'inspect_grid_diagnostics',fail)
        importer.run_import(options)
    else:
        def fail(*args):raise ValueError('preview error')
        monkeypatch.setattr(preview,'render_model_preview',fail)
        # Do not retain exception info/tracebacks in the test's reference audit.
        def run():
            try:importer.run_import(options)
            except RuntimeError as exc:assert 'preview error' in str(exc)
            else:pytest.fail('expected preview failure')
        run()
    gc.collect()
    assert all(ref() is None for ref in refs)
    path=options.out if phase=='grid' else next(tmp_path.glob('.out.staging-*'))
    report=json.loads((path/'import-report.json').read_text())
    assert_no_arrays(report)


def test_progress_sink_failure_aborts_instead_of_becoming_grid_failure(tmp_path):
    options=import_options(tmp_path,no_preview=True)
    def sink(event):
        if event['stage']=='grid' and event['state']=='started':raise ValueError('transport failed')
    with pytest.raises(RuntimeError,match='transport failed'):
        importer.run_import(options,progress_sink=sink)
    assert not options.out.exists()


@pytest.mark.parametrize('export',[False,True])
def test_no_preview_skips_unused_reload_passes(tmp_path,monkeypatch,export):
    options=dc_replace(import_options(tmp_path,export=export,no_preview=True),inspect_grids=())
    calls=[];original=source._decode
    def decode(path,relative,policy):
        calls.append(relative);return original(path,relative,policy)
    monkeypatch.setattr(source,'_decode',decode)
    importer.run_import(options)
    assert len(calls)==(6 if export else 3)


@pytest.mark.parametrize('transport',['progress','event'])
def test_post_publication_transport_failure_preserves_verified_bundle(tmp_path,monkeypatch,transport):
    options=import_options(tmp_path,export=True,no_preview=True)
    def failure_evidence(*args):
        pytest.fail('published bundle must not enter staging failure recovery')
    monkeypatch.setattr(importer,'_failure_evidence',failure_evidence)
    def sink(event):
        if (transport=='progress' and event.get('phase')=='publication') or (transport=='event' and event['event']=='completed'):
            raise BrokenPipeError('completion pipe closed')
    with pytest.raises(RuntimeError) as error:
        importer.run_import(options,progress_sink=sink if transport=='progress' else None,
                            event_sink=sink if transport=='event' else None)
    message=str(error.value)
    assert 'was published at '+str(options.out) in message
    assert 'verified' in message and 'transport failed' in message
    assert 'completion pipe closed' in message and 'unpublished' not in message
    assert importer.verify_bundle(options.out)['status']=='verified'
    assert not list(tmp_path.glob('.out.staging-*'))


@pytest.mark.parametrize('transport',['progress','event'])
def test_cli_post_publication_transport_failure_reports_accurate_status(tmp_path,monkeypatch,capsys,transport):
    import builtins
    options=import_options(tmp_path,no_preview=True)
    original=builtins.print
    def output(*args,**kwargs):
        if args and isinstance(args[0],str) and args[0].startswith('{'):
            record=json.loads(args[0])
            if (transport=='progress' and record.get('phase')=='publication') or (transport=='event' and record.get('event')=='completed'):
                raise BrokenPipeError('completion pipe closed')
        return original(*args,**kwargs)
    monkeypatch.setattr(builtins,'print',output)
    code=importer.main(['--root',str(options.root),'--out',str(options.out),'--no-preview'])
    assert code==2  # Completion transport failed, even though publication succeeded.
    stderr=capsys.readouterr().err
    assert 'was published at '+str(options.out) in stderr
    assert 'verified' in stderr and 'unpublished' not in stderr
    assert importer.verify_bundle(options.out)['status']=='verified'
    assert not list(tmp_path.glob('.out.staging-*'))


def test_progress_empty_electrode_model_has_balanced_reading_events(tmp_path,monkeypatch):
    root=tmp_path/'source'
    for name in ('empty','malformed','rejected'):
        (root/name/'E').mkdir(parents=True)
    write_png(root,'malformed/E/bad-name.png',shape=(2,3))
    (root/'rejected/E/1-L.png').write_bytes(b'not a PNG')
    events=[];original=source._decode
    def decode(path,relative,policy):
        model=relative.split('/')[0]
        assert any(e['stage']=='reading' and e['state']=='started' and e['model_id']==model for e in events)
        return original(path,relative,policy)
    monkeypatch.setattr(source,'_decode',decode)
    importer.run_import(ImportOptions(root=root,out=tmp_path/'out',no_preview=True),progress_sink=events.append)
    for name,total in [('empty',0),('malformed',0),('rejected',1)]:
        marks=[e for e in events if e['stage']=='reading' and e.get('model_id')==name]
        assert marks[0]['state']=='started' and marks[-1]['state']=='completed'
        assert 'files_total' not in marks[0]
        assert marks[-1]['files_done']==marks[-1]['files_total']==total
        assert [e['state'] for e in marks]==(['started','advanced','completed'] if total else ['started','completed'])


@pytest.mark.parametrize('relations', [False, True])
def test_diagnostic_descriptors_preserve_live_provenance_and_legacy(tmp_path, relations):
    from psdl1 import png_diagnostic as diagnostic
    if relations:
        for name in ('A', 'B'):
            write_model(tmp_path / 'src', [('E', '1-U.png', ((501, 501),)),
                        ('B', '3-L.png', ((501, 501),)),
                        ('cno', '2-C.png', ((501, 501),))], model=name)
    else:
        simple_tree(tmp_path / 'src')
    order = write_stack_order(tmp_path / 'order.json', ['U', 'C', 'L']) if relations else None
    inventory = source.discover_source(tmp_path / 'src', stack_order=order)
    for model in inventory.models:
        extraction = connections.extract_model(model)
        product = connections.to_product(model, extraction)
        descriptors = tuple(diagnostic.observation_descriptors(product, model, extraction))
        assert len([d for d in descriptors if d.scope == 'electrode']) == len(model.electrodes)
        assert len([d for d in descriptors if d.scope == 'connection']) == len(product.connections) == int(relations)
        assert len([d for d in descriptors if d.scope == 'global_cnt_union']) == 1
        cnt_source, = [d for d in descriptors if d.scope == 'cnt_source']
        asset, = model.cnt_assets
        assert cnt_source.source_asset_ids == (asset.asset_id,)
        assert cnt_source.layer == asset.layer
        assert set(cnt_source.source_paths) == {p['relative_path'] for p in
                (asset.provenance_files or ({'relative_path': asset.relative_path},))}
        assert cnt_source.mask is asset.mask
        results = list(diagnostic.inspect_grid_diagnostics(product, descriptors, (16, 32)))
        assert [r['grid_check'] for r in results] == list(data.iter_grid_checks(product, (16, 32)))
        assert_no_arrays(results)


def test_diagnostic_measurement_lifetime_is_one_observation(monkeypatch):
    from psdl1 import png_diagnostic as diagnostic
    references = []
    original = diagnostic.measure_source
    def measure(*args):
        assert all(reference() is None for reference in references)
        result = original(*args)
        references.extend(weakref.ref(a) for a in (result.source, result.labels.image,
                                                  result.horizontal, result.vertical))
        return result
    monkeypatch.setattr(diagnostic, 'measure_source', measure)
    mask = np.eye(16, dtype=bool)
    product = Product('lifetime', (16, 16), tuple(Element(f'E{i}', f'L{i}', mask) for i in range(4)))
    results = list(diagnostic.inspect_grid_diagnostics(product, grids=(4, 8, 16)))
    assert all(reference() is None for reference in references)
    assert_no_arrays(results)
