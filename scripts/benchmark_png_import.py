#!/usr/bin/env python3
"""Opt-in synthetic PNG importer benchmark; no production-data success claim.

Fixture creation and warm-up are outside the timed/tracemalloc region. Peak RSS
is process-lifetime RSS (including imports/warm-up), not retained-array payload.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time
import tracemalloc


def fixtures(base, count, shape, assets):
    import numpy as np
    from PIL import Image
    root = base/'source'; root.mkdir(parents=True)
    for index in range(count):
        specs = [('A','1-U.png',[(501,501)]), ('B','3-L.png',[(501,501)]),
                 ('cno','2-C.png',[(501,501),(200,200)])]
        specs += [(f'EXTRA{k}', '1-U.png', [(700+k % 100,700+k % 100)]) for k in range(assets-3)]
        for role,name,pixels in specs:
            mask=np.full(shape,255,dtype=np.uint8)
            for y,x in pixels:mask[y,x]=0
            path=root/f'M{index:04d}'/role/name;path.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(mask).save(path,compress_level=6)
    (base/'order.json').write_text(json.dumps({'stack_order_schema_version':'dl1_png_stack_order_v1','order':['U','C','L']}))
    (base/'metadata.json').write_text(json.dumps({'metadata_schema_version':'dl1_png_metadata_v1','models':{
        f'M{i:04d}':{'lineage_id':f'synthetic-family-{i}','design_version':'synthetic-v1'} for i in range(count)}}))
    return root


def measured(args):
    import torch
    import psdl1.data as data
    from psdl1.import_png import run_import, verify_bundle
    from psdl1.png_contract import ImportOptions
    torch.set_num_threads(1)
    selected=Path(args.package).resolve()
    assert Path(data.__file__).resolve()==selected/'psdl1/data.py'
    base=Path(args.fixture); output=Path(args.output)
    supports_skip='no_preview' in ImportOptions.__dataclass_fields__
    kwargs={'no_preview':True} if supports_skip else {}
    # One importer warm-up, fixed one-model fixture, no grid work.
    run_import(ImportOptions(root=Path(args.warm)/'source',out=output.with_name(output.name+'-warm'),**kwargs))
    kwargs={'no_preview':not args.show_preview} if supports_skip else {}
    if not supports_skip and not args.show_preview:
        raise ValueError('baseline has no --no-preview; compare preview-enabled scenarios')
    options=ImportOptions(root=base/'source',out=output,export=args.export,
        metadata=base/'metadata.json',candidate_grid=args.grids[0] if args.export else None,
        inspect_grids=tuple(args.grids),stack_order=base/'order.json' if args.mode=='relations' else None,**kwargs)
    tracemalloc.start(); start=time.perf_counter()
    _,status=run_import(options)
    elapsed=time.perf_counter()-start
    _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    verified=verify_bundle(output)['status']
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes=int(rss if sys.platform=='darwin' else rss*1024)
    report=json.loads((output/'import-report.json').read_text())
    files=list((base/'source').rglob('*.png'))
    print(json.dumps({'package':str(selected),'label':args.label,'python':platform.python_version(),
        'versions':{name:importlib.metadata.version(name) for name in ['Pillow','numpy','torch']},
        'commit':args.commit,'models':len(report['inventory']['models']),'files':len(files),
        'shape_yx':args.frame,'assets_per_model':args.assets_per_model,'grids':args.grids,
        'mode':args.mode,'preview':args.show_preview,'operation':'export' if args.export else 'inspect',
        'compressed_bytes':sum(p.stat().st_size for p in files),'elapsed_seconds':elapsed,
        'timing':'tracemalloc_instrumented','tracemalloc_peak_bytes':peak,'process_peak_rss_bytes':rss_bytes,
        'rss_scope':'process_lifetime_including_imports_and_warmup',
        'output_bytes':sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
        'exit_status':status,'verified':verified,'synthetic':True},sort_keys=True),flush=True)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',type=int,default=[2,8])
    parser.add_argument('--frame',nargs=2,type=int,default=[1001,1003],metavar=('H','W'))
    parser.add_argument('--assets-per-model',type=int,default=6)
    parser.add_argument('--grids',nargs='+',type=int,default=[32,64])
    parser.add_argument('--modes',nargs='+',choices=['image_only','relations'],default=['image_only','relations'])
    parser.add_argument('--preview',choices=['on','off','both'],default='both')
    parser.add_argument('--operation',choices=['inspect','export','both'],default='both')
    parser.add_argument('--baseline-src',type=Path,help='archived baseline src directory; preview-enabled comparisons only')
    parser.add_argument('--keep-output',type=Path,help='new benchmark-owned directory to retain fixtures and bundles')
    # Internal fresh-process invocation; parent never imports measured package.
    parser.add_argument('--measure',action='store_true',help=argparse.SUPPRESS)
    for flag in ['fixture','output','package','warm','label','commit','mode']:
        parser.add_argument('--'+flag,help=argparse.SUPPRESS)
    parser.add_argument('--show-preview',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--export',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if min(args.models)<1 or min(args.frame)<=1000 or args.assets_per_model<3 or min(args.grids)<1:
        parser.error('positive model/grid counts, H/W>1000 and assets-per-model>=3 required')
    if args.measure:return measured(args)
    repo=Path(__file__).resolve().parents[1]
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()
    preview=[True,False] if args.preview=='both' else [args.preview=='on']
    export=[False,True] if args.operation=='both' else [args.operation=='export']
    packages=[('candidate',repo/'src',commit+'+working-tree')]
    if args.baseline_src:packages.append(('baseline',args.baseline_src.resolve(),'d4dd896d315d99c7e72e68570a95f659b24aa6ea'))
    def scenarios(base):
        warm=base/'warm'; fixtures(warm,1,args.frame,args.assets_per_model)
        for count in args.models:
            fixture=base/f'fixture-{count}';fixtures(fixture,count,args.frame,args.assets_per_model)
            for (label,package,revision),mode,show,do_export in itertools.product(packages,args.modes,preview,export):
                if label=='baseline' and not show:continue
                name=f'{label}-{count}-{mode}-{show}-{do_export}'
                command=[sys.executable,str(Path(__file__).resolve()),'--measure','--fixture',str(fixture),
                    '--output',str(base/name),'--warm',str(warm),'--package',str(package),'--label',label,
                    '--commit',revision,'--mode',mode,'--frame',*map(str,args.frame),
                    '--assets-per-model',str(args.assets_per_model),'--grids',*map(str,args.grids)]
                if show:command.append('--show-preview')
                if do_export:command.append('--export')
                run=subprocess.run(command,cwd=base,env=dict(os.environ,PYTHONPATH=str(package),
                    PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),capture_output=True,text=True)
                if run.returncode:
                    print(run.stderr,file=sys.stderr);raise RuntimeError(f'benchmark scenario failed: {name}')
                print(run.stdout.strip(),flush=True)
    if args.keep_output:
        args.keep_output.resolve().mkdir(parents=True,exist_ok=False)
        scenarios(args.keep_output.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix='psdl1-png-benchmark-') as temporary:scenarios(Path(temporary))
    return 0


if __name__=='__main__':raise SystemExit(main())
