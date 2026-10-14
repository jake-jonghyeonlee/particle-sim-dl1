"""Actual d4dd896 oracle, isolated from candidate imports and helper algorithms."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from png_helpers import write_import_fixture, write_model, write_png, write_stack_order

BASELINE = 'd4dd896d315d99c7e72e68570a95f659b24aa6ea'
REPO = Path(__file__).resolve().parents[1]

DRIVER = r'''
import json, pathlib, sys, torch
import numpy as np
import psdl1.data as data
import psdl1.import_png as importer
from psdl1.png_contract import ImportOptions
from psdl1.schema import Product, Element

torch.set_num_threads(1)
base, dest, package = map(pathlib.Path, sys.argv[1:])
assert pathlib.Path(data.__file__).resolve() == (package/'psdl1/data.py').resolve()
result = {'module': str(pathlib.Path(data.__file__).resolve()), 'runs': {}}
if 'no_preview' in ImportOptions.__dataclass_fields__:
    from dataclasses import replace
    original_run_import = importer.run_import
    def run_and_no_preview(options, *, event_sink):
        result_default = original_run_import(options, event_sink=event_sink)
        skip_out = options.out.with_name(options.out.name+'-no-preview')
        events = []
        _, code = original_run_import(replace(options, out=skip_out, no_preview=True), event_sink=events.append)
        assert importer.verify_bundle(skip_out)['status'] == 'verified'
        result['runs'][skip_out.name] = {'status': code, 'events': events}
        return result_default
    importer.run_import = run_and_no_preview
for mode in ['image_only', 'relations']:
    for export in [False, True]:
        name = mode + ('-export' if export else '-inspect')
        events = []
        summary, status = importer.run_import(ImportOptions(
            root=base/'source', out=dest/name, export=export,
            metadata=base/'metadata.json', candidate_grid=32 if export else None,
            inspect_grids=(32,64,32), stack_order=base/'order.json' if mode == 'relations' else None),
            event_sink=events.append)
        assert importer.verify_bundle(dest/name)['status'] == 'verified'
        result['runs'][name] = {'status': status, 'events': events}
for case in ['invalid', 'duplicate', 'duplicate-image-only', 'partial']:
    name = case
    events = []
    source_case = 'duplicate' if case == 'duplicate-image-only' else 'invalid' if case == 'partial' else case
    summary, status = importer.run_import(ImportOptions(
        root=base/source_case, out=dest/name, export=True, candidate_grid=32,
        metadata=base/(case+'-metadata.json'), inspect_grids=(32,64,32),
        stack_order=None if case == 'duplicate-image-only' else base/('invalid-order.json' if case in ('invalid','partial') else 'order.json')),
        event_sink=events.append)
    assert importer.verify_bundle(dest/name)['status'] == 'verified'
    result['runs'][name] = {'status': status, 'events': events}
# Handcrafted valid holes, empty/missing observations, equal-shape unequal masks.
a=np.zeros((7,11),bool); a[2:5,1:4]=1; a[2:5,7:10]=1; a[3,4:7]=1
b=np.zeros_like(a); b[1:6,3:8]=1
valid=np.ones_like(a); valid[:,5]=0
p=Product('holes',(1.1,.7),(Element('A','L',a),Element('B','K',b),
    Element('empty','Q',np.zeros_like(a)), Element('missing','R',None,observation='missing')),
    cnt=b.copy(),valid=valid)
result['grids'] = [data.grid_check(p,g) for g in (4,8,12)] if not hasattr(data,'iter_grid_checks') else list(data.iter_grid_checks(p,(4,8,12)))
(dest/'result.json').write_text(json.dumps(result))
'''


@pytest.fixture(scope='module')
def archived_oracle(tmp_path_factory):
    base = tmp_path_factory.mktemp('stream-oracle')
    archived = base / 'archive'; archived.mkdir()
    archive = subprocess.Popen(['git', 'archive', BASELINE], cwd=REPO, stdout=subprocess.PIPE)
    extracted = subprocess.run(['tar', '-x', '-C', str(archived)], stdin=archive.stdout, capture_output=True)
    archive.stdout.close()
    assert archive.wait() == 0 and extracted.returncode == 0, 'required baseline archive unavailable'
    # Explicit synthetic bytes. Two model shapes at the same pitch; shared and
    # unique vocabulary, resolved/skipped/ambiguous/margin/empty CNT components.
    write_import_fixture(base / 'source', model='A', shape=(1001, 1003))
    write_import_fixture(base / 'source', model='B', shape=(1003, 1005))
    write_model(base / 'source', [('UNIQUE', '1-ONLY.png', ((900, 900),))], model='C', shape=(1001, 1003))
    # C has no cno directory, so no observation is fabricated.
    write_stack_order(base / 'order.json', ['U','N','C4','U5','C6','L','ONLY'])
    (base / 'metadata.json').write_text(json.dumps({'metadata_schema_version':'dl1_png_metadata_v1',
        'models': {name:{'lineage_id':'family-'+name,'design_version':'v1'} for name in 'ABC'}}))
    # A late invalid source, root notice, invalid order and malformed metadata.
    write_model(base / 'invalid', [('E', '1-L.png', ((501,501),))], model='A')
    (base / 'invalid' / 'A' / 'cno').mkdir()  # present, no PNG assets
    write_png(base / 'invalid', 'Z/E/1-L.png', shape=(9, 11))
    (base / 'invalid' / 'notes.txt').write_text('root notice')
    write_stack_order(base / 'invalid-order.json', ['missing'])
    (base / 'invalid-metadata.json').write_text('{malformed')
    (base / 'partial-metadata.json').write_text(json.dumps({'metadata_schema_version':'dl1_png_metadata_v1',
        'models': {'A':{'lineage_id':'family-A','design_version':'v1'}}}))
    # Same-name duplicate CNT is independently blocking in relations mode;
    # image_only merge/provenance behavior is covered by the scanner tests.
    write_model(base / 'duplicate', [('E','1-U.png',((501,501),)),
        ('cno','2-C6.png',((501,501),)),('cno','002_C6.png',((501,501),))],model='A')
    (base / 'duplicate-metadata.json').write_text(json.dumps({'metadata_schema_version':'dl1_png_metadata_v1',
        'models': {'A':{'lineage_id':'family-A','design_version':'v1'}}}))
    (base / 'duplicate-image-only-metadata.json').write_bytes((base / 'duplicate-metadata.json').read_bytes())
    outputs = []
    for label, package in [('baseline', archived/'src'), ('candidate', REPO/'src')]:
        dest = base / label; dest.mkdir()
        run = subprocess.run([sys.executable, '-c', DRIVER, str(base), str(dest), str(package)],
            cwd=base, env=dict(os.environ, PYTHONPATH=str(package), PYTHONDONTWRITEBYTECODE='1',
                             OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'), capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        result = json.loads((dest/'result.json').read_text())
        assert Path(result['module']) == (package/'psdl1/data.py').resolve()
        outputs.append((dest,result))
    return outputs


def normalize_notes(value):
    """The only allowed semantic difference is the documented resource prose."""
    if isinstance(value, dict):
        return {k: '<allowed resource overhead prose>' if k == 'overhead_note' else normalize_notes(v) for k,v in value.items()}
    if isinstance(value, list):
        return [normalize_notes(v) for v in value]
    return value


def compare_bundle(archived_oracle, name):
    (before, old), (after, new) = archived_oracle
    before, after = before/name, after/name
    old_run, new_run = old['runs'][name], new['runs'][name]
    assert old_run['status'] == new_run['status']
    def events(run):
        result = normalize_notes(run['events'])
        assert [e['event'] for e in result] == ['resource_inventory','completed']
        result[-1]['output'] = '<destination>'
        return result
    assert events(old_run) == events(new_run)
    old_files = {p.relative_to(before).as_posix():p for p in before.rglob('*') if p.is_file()}
    new_files = {p.relative_to(after).as_posix():p for p in after.rglob('*') if p.is_file()}
    assert old_files.keys() == new_files.keys()
    for relative in old_files:
        a,b = old_files[relative].read_bytes(),new_files[relative].read_bytes()
        if relative == 'import-report.json':
            assert normalize_notes(json.loads(a)) == normalize_notes(json.loads(b))
            assert a == (json.dumps(json.loads(a),indent=2,sort_keys=True,ensure_ascii=False)+'\n').encode()
            assert b == (json.dumps(json.loads(b),indent=2,sort_keys=True,ensure_ascii=False)+'\n').encode()
        elif relative == 'bundle-integrity.json':
            # Both bundles were verified by their own package above. Every
            # linked hash is also checked directly; only the report hash differs.
            x,y=json.loads(a),json.loads(b)
            for record, files in [(x,old_files),(y,new_files)]:
                for path, digest in record['files'].items():
                    assert digest == hashlib.sha256(files[path].read_bytes()).hexdigest()
                record['files']['import-report.json'] = '<report hash>'
            assert x == y
        else:
            assert a == b, relative


@pytest.mark.parametrize('mode', ['image_only','relations'])
@pytest.mark.parametrize('operation', ['inspect','export'])
def test_stream_import_matches_d4dd896(archived_oracle, mode, operation):
    compare_bundle(archived_oracle, mode+'-'+operation)


@pytest.mark.parametrize('case', ['invalid','duplicate','duplicate-image-only','partial'])
def test_invalid_multimodel_matches_d4dd896(archived_oracle, case):
    compare_bundle(archived_oracle, case)


def test_multigrid_matches_baseline_with_valid_holes_and_missing_masks(archived_oracle):
    assert archived_oracle[0][1]['grids'] == archived_oracle[1][1]['grids']
    first = archived_oracle[1][1]['grids'][0]['elements']
    assert first[0]['source_components'] == 2
    assert first[2]['source_active_pixels'] == 0
    assert first[3]['observation'] == 'missing'


@pytest.mark.parametrize('name', ['image_only-inspect','relations-inspect','image_only-export','relations-export',
                                 'invalid','duplicate','duplicate-image-only','partial'])
def test_no_preview_matches_d4dd896_except_preview(archived_oracle,name):
    (before, old), (after, new) = archived_oracle
    old_report = normalize_notes(json.loads((before/name/'import-report.json').read_text()))
    new_report = normalize_notes(json.loads((after/(name+'-no-preview')/'import-report.json').read_text()))
    assert new['runs'][name+'-no-preview']['status'] == old['runs'][name]['status']
    assert new_report['preview'] == []
    old_report['preview'] = []
    assert new_report['diagnostics'].pop() == {'path':'.','reason':'preview_skipped_by_request: --no-preview',
        'exception_class':None,'shape_yx':None,'pixels':None,'effective_max_image_pixels':None}
    assert old_report == new_report
    old_files = {p.relative_to(before/name).as_posix():p for p in (before/name).rglob('*') if p.is_file()
                 and not p.relative_to(before/name).as_posix().startswith('preview/') and p.name not in ('preview.html','import-report.json','bundle-integrity.json')}
    target = after/(name+'-no-preview')
    assert not (target/'preview').exists() and not (target/'preview.html').exists()
    new_files = {p.relative_to(target).as_posix():p for p in target.rglob('*') if p.is_file() and p.name not in ('import-report.json','bundle-integrity.json')}
    assert old_files.keys() == new_files.keys()
    for path in old_files: assert old_files[path].read_bytes() == new_files[path].read_bytes()
