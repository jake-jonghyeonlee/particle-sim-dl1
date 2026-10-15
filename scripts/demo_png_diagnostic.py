#!/usr/bin/env python3
"""Generate only synthetic PNG sources and a verified two-grid diagnostic bundle."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True

# Permit direct execution from a source checkout, without installation changes.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    import numpy as np
    import torch
    from PIL import Image
    from psdl1.import_png import run_import,verify_bundle
    from psdl1.png_contract import ImportOptions
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path,help='fresh directory for synthetic source and bundle')
    args=parser.parse_args()
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    shape=(1001,1203)
    a=np.zeros(shape,bool)
    a[200:500,120:420]=a[200:500,780:1080]=True
    a[350,420:780]=True;a[90,80]=True                      # split bridge + loss
    b=np.zeros(shape,bool);b[650:900,300:900]=True
    hidden=np.zeros(shape,bool);hidden[760,600]=True
    merge=np.zeros(shape,bool)                             # 2px gap -> merge?
    merge[560:620,100:300]=merge[560:620,302:500]=True
    gate=np.zeros(shape,bool);gate[150:950,560:700]=True    # overlaps several layers
    via=np.zeros(shape,bool);via[100:600,900:1000]=True
    cnt=np.zeros(shape,bool);cnt[700:820,550:650]=True;cnt[850,1050]=True
    thin=np.zeros(shape,bool)                              # thin-only product
    thin[100:900,300:324]=thin[100:900,700:730]=True;thin[400:600,900:1100]=True
    source=out/'source'
    for model,role,name,mask in [('SYNTHETIC-DIAGNOSTIC','bridge','1-top.png',a),
                                 ('SYNTHETIC-DIAGNOSTIC','broad','3-bottom.png',b),
                                 ('SYNTHETIC-DIAGNOSTIC','hidden','4-buried.png',hidden),
                                 ('SYNTHETIC-DIAGNOSTIC','merge','5-metal2.png',merge),
                                 ('SYNTHETIC-DIAGNOSTIC','gate','6-gate.png',gate),
                                 ('SYNTHETIC-DIAGNOSTIC','via','7-via.png',via),
                                 ('SYNTHETIC-DIAGNOSTIC','cno','2-CNT.png',cnt),
                                 ('SYNTHETIC-THIN-ONLY','line','1-top.png',thin)]:
        path=source/model/role/name;path.parent.mkdir(parents=True,exist_ok=True)
        with Image.fromarray(np.where(mask,0,255).astype(np.uint8)) as image:
            image.save(path)
    bundle=out/'bundle'
    _,code=run_import(ImportOptions(root=source,out=bundle,inspect_grids=(64,128),no_preview=True))
    if code:raise RuntimeError(f'synthetic import failed: {code}')
    assert verify_bundle(bundle)['status']=='verified'
    report=json.loads((bundle/'import-report.json').read_text())
    for row in report['grid_evidence']:
        if 'diagnostic' in row and 'png' in row['diagnostic']:
            print(row['model_id'],row['diagnostic']['shape_state'],bundle/row['diagnostic']['png'])
    return 0


if __name__=='__main__':raise SystemExit(main())
