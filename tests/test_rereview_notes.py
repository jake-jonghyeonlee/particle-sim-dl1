"""R1–R6 counterexamples: inference fixtures, never real training."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from psdl1.checkpoint import json_hash
from psdl1.data import build_vocab, grid_check, input_identity, load_manifest
from psdl1.evaluate import evaluate
from psdl1.model import ProductEncoder
from psdl1.search import search
from psdl1.split import SPLIT_VERSION
from psdl1.train import model_config, resolve_training_grid


def evaluation_fixture(products, grid=4):
    products = [replace(p, product_id=pid, lineage_id=pid) for p, pid in zip(products, ('train', 'query', 'candidate'))]
    roles, layers = build_vocab(products[:1])
    cfg = model_config(grid, roles, layers)
    model = ProductEncoder(len(roles), len(layers)).eval()
    split = {'schema_version': SPLIT_VERSION, 'train': ['train'], 'validation': [], 'test': ['query', 'candidate']}
    model.trained_epochs = 1  # provenance fixture, no optimizer or performance assertion
    model.training_metadata = {'split': split, 'input_hash': json_hash({p.product_id: {'identity': input_identity(p), 'lineage': p.lineage_id, 'design': p.design_version} for p in products})}
    return model, cfg, products, split


@pytest.mark.parametrize('command', ['search', 'evaluate'])
def test_r1_cli_grid_conflict_and_recorded_override(command, manifest_case, tmp_path, monkeypatch):
    doc, write, _ = manifest_case
    doc['candidate_grid'] = 8
    original, = load_manifest(write())
    model, cfg, products, split = evaluation_fixture([original] * 3)
    # Real manifests through the CLI, local model fixture only.
    doc['products'] = [dict(doc['products'][0], product_id=p.product_id, lineage_id=p.lineage_id) for p in products]
    path = write()
    out = tmp_path / 'result.json'
    if command == 'search':
        from psdl1.search import main
        monkeypatch.setattr('psdl1.search.load_model', lambda _: (model, cfg))
        args = ['--ckpt', 'fixture.pt', '--query-manifest', str(path), '--candidates-manifest', str(path), '--query', 'query', '--out', str(out)]
    else:
        from psdl1.evaluate import main
        monkeypatch.setattr('psdl1.evaluate.load_model', lambda _: (model, cfg))
        sf, rf = tmp_path / 'split.json', tmp_path / 'relations.csv'
        sf.write_text(json.dumps(split)); rf.write_text('query,candidate,similar\n')
        args = ['--ckpt', 'fixture.pt', '--manifest', str(path), '--split', str(sf), '--relations', str(rf), '--out', str(out)]
    with pytest.raises(ValueError, match='checkpoint grid 4 conflicts.*candidate_grid.*8'):
        main(args)
    assert not out.exists()
    main([*args, '--allow-grid-override'])
    report = json.loads(out.read_text())
    assert report['grid'] == 4
    assert report['grid_selection'] == {'selected_grid': 4, 'manifest_candidate_grids': [8], 'source': 'checkpoint', 'override': True, 'override_requested': True}


@pytest.mark.parametrize('failure', ['thin', 'bridge'])
def test_r2_query_and_candidate_gate_search_and_evaluation(failure, manifest_case, bridge_mask):
    _, write, _ = manifest_case
    original, = load_manifest(write())
    original = replace(original, grid_check=None, frame=None, connections=(), cnt=None, connectivity_observation='unavailable')
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[3, :] = 1
    if failure == 'bridge':
        mask = bridge_mask
    h, w = mask.shape
    bad = replace(original, unit_size_um=(w * .05, h * .05), valid=None,
                  elements=tuple(replace(e, mask=mask) for e in original.elements))
    model, cfg, products, split = evaluation_fixture([original, original, bad])
    query, candidate = products[1:]
    result = search(model, cfg, query, [candidate])
    assert result['status'] == 'unavailable' and result['reason'] == 'no_eligible_candidates'
    assert result['excluded'] == [{'product_id': 'candidate', 'reason': 'grid_check_failed'}]
    evidence = result['grid_gate']['products'][1]
    assert evidence['gate_status'] == 'blocked'
    assert evidence['elements'][0]['thin_line_warning'] is True
    if failure == 'thin':
        assert evidence['elements'][0]['axis_run_px'] == 1
        assert evidence['elements'][0]['estimated_run_at_grid_px'] == .5  # 1 * grid4 / unit8
    else:
        assert evidence['elements'][0]['bridge_or_component_loss'] is True
    blocked_query = search(model, cfg, candidate, [query])
    assert blocked_query['reason'] == {'reason': 'grid_check_failed'}
    bypass = search(model, cfg, candidate, [query], allow_grid_check_failure=True)
    assert bypass['status'] == 'available'
    assert bypass['grid_gate']['bypass_requested'] is True
    assert bypass['grid_gate']['products'][0]['gate_status'] == 'bypassed'
    relations = ({frozenset(('query', 'candidate'))}, set())
    report = evaluate(model, cfg, products, split, relations)
    assert report['evaluated_queries'] == 0 and report['recall_at_k'] is None
    assert all(not row['baseline'] for row in report['queries'])
    assert report['grid_gate']['products'][1]['gate_status'] == 'blocked'
    assert report['queries'][1]['model']['reason'] == {'reason': 'grid_check_failed'}
    report = evaluate(model, cfg, products, split, relations, allow_grid_check_failure=True)
    assert report['evaluated_queries'] == 2
    assert report['grid_gate']['bypass_requested'] is True
    assert report['grid_gate']['products'][1]['gate_status'] == 'bypassed'


def test_r3_handwritten_grid_report_is_not_manifest_provenance(products):
    p = products[0]
    handmade = replace(p, grid_check=grid_check(p, 64))
    with pytest.raises(ValueError, match='No grid supplied'):
        resolve_training_grid([handmade])
    grid, selection = resolve_training_grid([handmade], 64)
    assert grid == 64
    assert selection == {'manifest_candidate_grid': None, 'selected_grid': 64, 'source': 'api_explicit', 'override': False, 'override_requested': False}


def test_r4_current_docs_have_no_guessed_validator_grid_or_stale_commit_status():
    root = Path(__file__).resolve().parents[1]
    assert '--grid 256' not in (root / 'docs/data_contract.md').read_text()
    assert 'Changes below remain uncommitted' not in (root / 'docs/validation.md').read_text()


def test_r5_evaluation_encodes_each_product_once_real_encoder(manifest_case, monkeypatch):
    _, write, _ = manifest_case
    p, = load_manifest(write())
    model, cfg, products, split = evaluation_fixture([p] * 3)
    import psdl1.search as module
    original_encode = module.encode_products
    calls = []
    def count(model, items, *args, **kwargs):
        calls.extend(p.product_id for p in items)
        return original_encode(model, items, *args, **kwargs)
    monkeypatch.setattr(module, 'encode_products', count)
    report = evaluate(model, cfg, products, split, ({frozenset(('query', 'candidate'))}, set()))
    assert sorted(calls) == ['candidate', 'query']  # two forwards, not four
    assert report['recall_at_k'] == 1
    assert all(row['model']['grid'] == 4 for row in report['queries'])


@pytest.mark.parametrize('flag', ['allow_grid_override', 'allow_grid_check_failure'])
def test_r1_r2_flags_require_boolean(manifest_case, flag):
    _, write, _ = manifest_case
    p, = load_manifest(write())
    model, cfg, products, _ = evaluation_fixture([p] * 3)
    with pytest.raises(ValueError, match='boolean'):
        search(model, cfg, products[1], products[2:], **{flag: 'false'})


@pytest.mark.parametrize('command', ['search', 'evaluate'])
def test_r2_cli_thin_products_blocked_and_bypass_recorded(command, manifest_case, tmp_path, monkeypatch):
    doc, write, root = manifest_case
    base = doc['products'][0]
    base.update(connections=[], connectivity_observation='unavailable', cnt=None)
    doc['products'] = [json.loads(json.dumps(base)) for _ in range(3)]
    thin = np.zeros((4, 4), dtype=np.uint8)
    thin[1, :] = 1
    np.save(root / 'thin.npy', thin)
    for raw, pid in zip(doc['products'], ('train', 'query', 'candidate')):
        raw.update(product_id=pid, lineage_id=pid)
        if pid != 'train':
            for element in raw['elements']:
                element['mask'] = 'thin.npy'
    path = write()
    model, cfg, products, split = evaluation_fixture(load_manifest(path))
    out = tmp_path / 'out.json'
    if command == 'search':
        from psdl1.search import main
        monkeypatch.setattr('psdl1.search.load_model', lambda _: (model, cfg))
        args = ['--ckpt', 'fixture.pt', '--query-manifest', str(path), '--candidates-manifest', str(path), '--query', 'query', '--out', str(out)]
    else:
        from psdl1.evaluate import main
        monkeypatch.setattr('psdl1.evaluate.load_model', lambda _: (model, cfg))
        sf, rf = tmp_path / 'split.json', tmp_path / 'relations.csv'
        sf.write_text(json.dumps(split))
        rf.write_text('query,candidate,similar\n')
        args = ['--ckpt', 'fixture.pt', '--manifest', str(path), '--split', str(sf), '--relations', str(rf), '--out', str(out)]
    main(args)
    report = json.loads(out.read_text())
    checks = {p['product_id']: p for p in report['grid_gate']['products']}
    assert checks['query']['gate_status'] == checks['candidate']['gate_status'] == 'blocked'
    assert checks['query']['elements'][0]['estimated_run_at_grid_px'] == 1
    main([*args, '--allow-grid-check-failure'])
    report = json.loads(out.read_text())
    checks = {p['product_id']: p for p in report['grid_gate']['products']}
    assert report['grid_gate']['bypass_requested'] is True
    assert checks['query']['gate_status'] == checks['candidate']['gate_status'] == 'bypassed'
    assert report['status'] == 'available' if command == 'search' else report['evaluated_queries'] == 2
