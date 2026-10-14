"""Deterministic retained-array bound, not a total RSS/allocator/decoder bound."""
import gc
import json
import weakref

import numpy as np
import pytest

from psdl1 import png_source as source, png_connections as connections, data, import_png as importer
from test_png_streaming import import_options, assert_no_arrays


class RetainedBytes:
    def __init__(self):
        self.live = {}
        self.peak = 0

    def watch(self, array):
        if array is None:
            return
        while isinstance(array.base, np.ndarray):
            array = array.base
        key = id(array)
        if key not in self.live:
            self.live[key] = (weakref.ref(array, lambda ref, key=key: self.live.pop(key, None)), array.nbytes)
            self.peak = max(self.peak, sum(n for _,n in self.live.values()))

    def product(self,p):
        for array in [p.valid,p.cnt,*[e.mask for e in p.elements],*[c.mask for c in p.connections]]:
            self.watch(array)

    def empty(self):
        gc.collect()
        assert not self.live, 'previous model/Product allocations survived into next model'


@pytest.mark.parametrize('relations',[False,True])
@pytest.mark.parametrize('export',[False,True])
def test_peak_live_mask_bytes_independent_of_model_count(tmp_path,monkeypatch,relations,export):
    peaks=[]
    # At least one scenario exercises real preview images and its per-model frame.
    show_preview = not export
    tracker=RetainedBytes()
    decode,component,product,loaded,full = source._decode,connections._extract_component,importer.to_product,data._manifest_product,connections.component_mask
    def track_decode(*args):
        result=decode(*args);tracker.watch(result[0]);return result
    def track_component(*args):
        result=component(*args);tracker.watch(result.mask);return result
    def track_product(*args,**kwargs):
        result=product(*args,**kwargs);tracker.product(result);return result
    def track_loaded(*args):
        tracker.empty()  # catches caller/generator locals and list(iterator).
        result=loaded(*args);tracker.product(result);return result
    def track_full(*args):
        result=full(*args);tracker.watch(result);return result
    original_payload = importer.report_payload
    def audit_payload(report):
        assert_no_arrays(report)  # Audit actual dataclasses, before JSON conversion.
        return original_payload(report)
    monkeypatch.setattr(importer,'report_payload',audit_payload)
    monkeypatch.setattr(source,'_decode',track_decode)
    monkeypatch.setattr(connections,'_extract_component',track_component)
    monkeypatch.setattr(importer,'to_product',track_product)
    monkeypatch.setattr(data,'_manifest_product',track_loaded)
    monkeypatch.setattr(connections,'component_mask',track_full)
    for count in (2,8):
        case=tmp_path/str(count);case.mkdir()
        options=import_options(case,relations=relations,export=export,no_preview=not show_preview,model_count=count)
        def progress(event):
            if event['stage']=='model' and event['state']=='started' and event['phase'] in ('inventory','inspect_preview','export'):
                tracker.empty()
        _,code=importer.run_import(options,progress_sink=progress)
        assert code==0
        tracker.empty()
        report=json.loads((options.out/'import-report.json').read_text())
        assert_no_arrays(report)
        assert len(report['inventory']['models'])==len(report['extraction'])==count
        assert all(len(m['assets'])==3 for m in report['inventory']['models'])
        assert all(len(r['components'])==2 for r in report['extraction'])
        assert all(r['summary']['resolved_relation_count']==int(relations) for r in report['extraction'])
        assert len(report['grid_evidence'])==count*(3 if export else 2)
        assert len(report['preview'])==(count if show_preview else 0)
        if export:
            assert len(report['product_input_identities'])==count
            assert len(report['generated_asset_hashes'])==count*(4+int(relations))
        pixels=1001*1003
        # Actual tracked retained allocations: A=3 source masks, two 1px bbox
        # components; Product adds valid/global_CNT and at most one relation.
        # Staged loading holds float32 for E2+valid+CNT+(R1 in relations).
        inspection_bound=(5+int(relations))*pixels+2
        staged_bound=4*(4+int(relations))*pixels if export else 0
        allowance=max(inspection_bound,staged_bound)
        assert tracker.peak<=allowance
        peaks.append(tracker.peak)
        tracker.peak=0
    # Fixed one-model footprint for identical-complexity models, no N tolerance.
    assert peaks[0]==peaks[1]
