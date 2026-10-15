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


@pytest.mark.parametrize('relations',[False,True])
def test_diagnostic_working_arrays_and_images_release_between_models(tmp_path,monkeypatch,relations):
    from psdl1 import png_diagnostic as diagnostic, _png_diagnostic_render as render
    tracker=RetainedBytes();peaks=[];images=[]
    class NumericProbe:
        # Observe diagnostic numeric queues/graphs, runs, labels, coverages,
        # display/crop planes independently from the existing source-mask bound.
        def __getattr__(self,name):
            function=getattr(np,name)
            if name not in {'zeros','zeros_like','full','full_like','arange','cumsum','column_stack',
                            'concatenate','pad','diff','unique','nonzero','indices','array','asarray'}:
                return function
            def observed(*args,**kwargs):
                out=function(*args,**kwargs)
                values=out if isinstance(out,tuple) else (out,)
                for value in values:
                    if isinstance(value,np.ndarray):tracker.watch(value)
                return out
            return observed
    monkeypatch.setattr(diagnostic,'np',NumericProbe())
    monkeypatch.setattr(render,'np',NumericProbe())
    measure_source,measure_grid=diagnostic.measure_source,diagnostic.measure_grid
    def measured_source(*args):
        out=measure_source(*args)
        for a in (out.source,out.labels.image,out.labels.bounds,out.labels.first_yx,out.horizontal,out.vertical):tracker.watch(a)
        return out
    def measured_grid(*args):
        out=measure_grid(*args)
        for a in (out.coverage,out.target.image,out.edges,out.lost_counts,out.lost_xy):tracker.watch(a)
        return out
    monkeypatch.setattr(diagnostic,'measure_source',measured_source)
    monkeypatch.setattr(diagnostic,'measure_grid',measured_grid)
    crop_pair=render._crop_pair
    def captured_crop(*args):
        out=crop_pair(*args)
        tracker.watch(out[0]);tracker.watch(out[1])
        return out
    monkeypatch.setattr(render,'_crop_pair',captured_crop)
    new=render.Image.new
    def image_new(mode,size,*args,**kwargs):
        out=new(mode,size,*args,**kwargs)
        if size in ((768,768),(1100,1040)):
            images.append(weakref.ref(out))
        return out
    monkeypatch.setattr(render.Image,'new',image_new)
    for count in (2,8):
        case=tmp_path/str(count);case.mkdir()
        options=import_options(case,relations=relations,no_preview=True,model_count=count)
        def progress(event):
            if event['phase']=='inspect_preview' and event['stage']=='model' and event['state']=='started':
                tracker.empty()
                assert all(ref() is None for ref in images)
        _,code=importer.run_import(options,progress_sink=progress)
        assert code==0
        tracker.empty();assert all(ref() is None for ref in images)
        peaks.append(tracker.peak);tracker.peak=0
    assert peaks[0]==peaks[1] and peaks[0]>1001*1003*8


def test_grid_count_does_not_multiply_source_measurements_or_reload(tmp_path,monkeypatch):
    from dataclasses import replace
    from psdl1 import png_diagnostic as diagnostic
    counts={'source':0,'normalize':0,'decode':0,'reload':0}
    measured,normalize,decode,reload=diagnostic.measure_source,data.normalize_unit,source._decode,importer._reload_extraction
    def source_measure(*args):counts['source']+=1;return measured(*args)
    def norm(*args):counts['normalize']+=1;return normalize(*args)
    def decoded(*args):counts['decode']+=1;return decode(*args)
    def reloaded(*args,**kwargs):counts['reload']+=1;return reload(*args,**kwargs)
    monkeypatch.setattr(diagnostic,'measure_source',source_measure)
    monkeypatch.setattr(data,'normalize_unit',norm)
    monkeypatch.setattr(source,'_decode',decoded)
    monkeypatch.setattr(importer,'_reload_extraction',reloaded)
    results=[]
    for grids in ((16,),(16,32,64)):
        case=tmp_path/str(len(grids));case.mkdir()
        options=replace(import_options(case,no_preview=True),inspect_grids=grids)
        assert importer.run_import(options)[1]==0
        results.append(counts.copy())
        for k in counts:counts[k]=0
    assert results[0]['source']==results[1]['source']==4 # E2 + global CNT + canonical CNT
    assert results[1]['normalize']==3*results[0]['normalize']==12
    # Same as b8c8a22: one scan and one verified reload; no extra decoder pass.
    assert results[0]['decode']==results[1]['decode']==6
    assert results[0]['reload']==results[1]['reload']==1
