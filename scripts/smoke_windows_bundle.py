#!/usr/bin/env python3
"""Installed-wheel-only synthetic PNG + CPU train/search smoke. No source path injection."""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def deny_network(*args, **kwargs):
    raise RuntimeError('network forbidden in offline smoke')


# subprocess module entry points receive the same Python-level network guard.
GUARD = """import runpy,socket,sys
def deny(*a,**k): raise RuntimeError('network forbidden in offline smoke')
socket.socket.connect=deny;socket.socket.connect_ex=deny
socket.create_connection=deny;socket.getaddrinfo=deny
module=sys.argv.pop(1);runpy.run_module(module,run_name='__main__')
"""


def run_module(work, label, module, *args):
    result = subprocess.run([sys.executable, '-I', '-c', GUARD, module, *map(str, args)],
                            cwd=work, capture_output=True, text=True, encoding='utf-8')
    (work / (label + '.stdout.log')).write_text(result.stdout, encoding='utf-8')
    (work / (label + '.stderr.log')).write_text(result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'{label} failed ({result.returncode}): {result.stderr[-4000:]}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path, help='new directory, outside the verified bundle')
    args = parser.parse_args()
    work = args.out.resolve()
    work.mkdir(parents=True, exist_ok=False)
    for name in ('connect', 'connect_ex'):
        setattr(socket.socket, name, deny_network)
    socket.create_connection = socket.getaddrinfo = deny_network
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                      PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1', PYTHONIOENCODING='utf-8')
    import numpy as np
    from PIL import Image
    import PIL
    import torch
    import psdl1
    from psdl1.data import load_manifest
    from psdl1.import_png import verify_bundle
    from psdl1.train import train
    from psdl1.checkpoint import read_checkpoint
    torch.set_num_threads(1)
    location = Path(psdl1.__file__).resolve()
    dist = importlib.metadata.distribution('psdl1')
    if dist.read_text('direct_url.json') or not location.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f'smoke requires installed non-editable wheel: {location}')
    if PIL.__version__ != '12.3.0':
        raise RuntimeError('Pillow 12.3.0 required')
    source = work / 'source PNG 한글'
    models = ['SYNTHETIC-OFFLINE-A', 'SYNTHETIC-OFFLINE-B']
    for index, model in enumerate(models):
        for role, name, box in [('top', '1-upper.png', (200, 200, 500, 500)),
                                ('bottom', '3-lower.png', (650, 300, 900, 900)),
                                ('cno', '2-CNT.png', (700, 550, 820, 650))]:
            mask = np.full((1001, 1203), 255, dtype=np.uint8)
            y0, x0, y1, x1 = box
            mask[y0:y1, x0 + index * 20:x1 + index * 20] = 0
            path = source / model / role / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(path)
    inspected = work / 'inspect 결과'
    run_module(work, 'inspect', 'psdl1.import_png', '--root', source, '--out', inspected,
               '--inspect-grids', 64, 128, '--no-preview')
    run_module(work, 'verify-inspect', 'psdl1.import_png', '--verify-bundle', inspected)
    report = json.loads((inspected / 'import-report.json').read_text(encoding='utf-8'))
    diagnostic_rows = [r for r in report['grid_evidence'] if 'diagnostic' in r]
    if len(diagnostic_rows) != 4:
        raise RuntimeError('missing model/grid diagnostics')
    for row in diagnostic_rows:
        diagnostic = row['diagnostic']
        if diagnostic['state'] != 'complete':
            raise RuntimeError('incomplete diagnostic')
        for field in ('png', 'evidence'):
            if not (inspected / diagnostic[field]).is_file():
                raise RuntimeError('missing PNG/JSONL diagnostic')
    metadata = work / 'synthetic metadata.json'
    metadata.write_text(json.dumps({'metadata_schema_version': 'dl1_png_metadata_v1', 'models': {
        model: {'lineage_id': f'SYNTHETIC-FAMILY-{i}', 'design_version': 'SYNTHETIC-V1'}
        for i, model in enumerate(models)}}), encoding='utf-8')
    exported = work / 'export 결과'
    run_module(work, 'export', 'psdl1.import_png', '--root', source, '--out', exported,
               '--export', '--metadata', metadata, '--candidate-grid', 128, '--no-preview')
    run_module(work, 'verify-export', 'psdl1.import_png', '--verify-bundle', exported)
    run_module(work, 'validate', 'psdl1.data', exported / 'manifest.json', '--report', work / 'validation.json')
    verified = verify_bundle(exported)
    products = load_manifest(exported / 'manifest.json')
    if len(products) != 2 or verified['export_status'] != 'exported':
        raise RuntimeError('invalid synthetic export')
    held = replace(products[0], product_id='SYNTHETIC-HELD', lineage_id='SYNTHETIC-HELD-FAMILY')
    from psdl1.split import SPLIT_VERSION
    split = {'schema_version': SPLIT_VERSION, 'train': models, 'validation': [], 'test': [held.product_id]}
    trained = work / 'synthetic CPU train'
    history = train([*products, held], trained, epochs=1, grid=128, batch_size=2, seed=173,
                    relations=(set(), {frozenset(models)}), split_manifest=split,
                    model_kwargs={'dim': 16, 'out_dim': 8, 'heads': 4, 'blocks': 0})
    checkpoint = read_checkpoint(trained / 'dl1.pt')
    if len(history) != 1 or not np.isfinite(history).all() or checkpoint['metadata']['epochs'][0]['steps'] != 1:
        raise RuntimeError('CPU one-step training failed')
    run_module(work, 'search', 'psdl1.search', '--ckpt', trained / 'dl1.pt',
               '--query-manifest', exported / 'manifest.json', '--query', models[0],
               '--candidates-manifest', exported / 'manifest.json', '--out', work / 'search.json')
    result = json.loads((work / 'search.json').read_text(encoding='utf-8'))
    if result['status'] != 'available' or result['eligible_count'] != 1 or result['rankings'][0]['product_id'] != models[1]:
        raise RuntimeError('CPU search failed')
    evidence = {'status': 'passed', 'platform': sys.platform, 'python': sys.version,
                'installed_psdl1': str(location), 'torch': torch.__version__, 'numpy': np.__version__,
                'pillow': PIL.__version__, 'cuda_available': torch.cuda.is_available(),
                'synthetic_only': True, 'diagnostic_rows': len(diagnostic_rows),
                'export': verified, 'cpu_training_steps': 1, 'search_eligible': result['eligible_count']}
    (work / 'smoke-result.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
