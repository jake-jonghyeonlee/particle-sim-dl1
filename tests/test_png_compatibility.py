"""S20: the actual 03bf3e3 package and candidate run in isolated subprocesses."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

BASELINE = "03bf3e33c7107af75297aa298fa323e24018f28d"
REPO = Path(__file__).resolve().parents[1]

# Both subprocesses consume exactly these independently specified NPY bytes.
DRIVER = '''
import json, pathlib, sys, warnings
import numpy as np, torch
import psdl1.data as data
from psdl1.train import structured_batch, model_config, encode_products
from psdl1.model import ProductEncoder
from psdl1.checkpoint import code_hash, save_checkpoint, read_checkpoint, resume_checkpoint
from psdl1.search import load_model
warnings.simplefilter("ignore"); torch.set_num_threads(1); torch.manual_seed(47)
manifest, output, checkpoint, mode = sys.argv[1:]
p = data.load_manifest(manifest); roles,layers = data.build_vocab(p)
cfg = model_config(16,roles,layers,model_kwargs={"dim":16,"out_dim":8,"heads":4,"blocks":0})
m = ProductEncoder(len(roles),len(layers),**cfg["model_kwargs"])
opt = torch.optim.AdamW(m.parameters())
if mode == "baseline":
    save_checkpoint(checkpoint,m,cfg,optimizer=opt,metadata={"code_hash":code_hash(),"input_hash":"SYNTHETIC-FIXTURE"})
else:
    m,loaded = load_model(checkpoint); assert loaded == cfg
    before = torch.get_rng_state().clone()
    try:
        resume_checkpoint(checkpoint,m,opt,torch.Generator(),expected_config=cfg,
                          expected_code_hash=code_hash(),expected_input_hash="SYNTHETIC-FIXTURE")
    except ValueError as e: assert "code hash differs" in str(e), str(e)
    else: raise AssertionError("changed-code resume must be rejected")
    assert torch.equal(before,torch.get_rng_state())
m.eval(); batch,extra,prepared = structured_batch(p,16,roles,layers)
arrays = {"batch"+str(i):v.numpy() for i,v in enumerate(batch)}
arrays.update({k:v.numpy() for k,v in extra.items()})
arrays["normalized"] = data.normalize_unit(p[0].elements[0].mask,16)
arrays["embedding"] = encode_products(m,p,16,roles,layers,prepared=prepared).detach().numpy()
np.savez(output,**arrays)
pathlib.Path(output+".json").write_text(json.dumps({"module":data.__file__,"identity":data.input_identity(p[0]),
  "cfg":cfg,"code_hash":code_hash(),"model_input_version":read_checkpoint(checkpoint)["model_input_version"]}))
'''


@pytest.fixture(scope="module")
def golden(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("archived-npy-baseline")
    archived = tmp / "baseline"; archived.mkdir()
    # Contract-required archive, not a worktree/checkout or alternate branch.
    archive = subprocess.Popen(["git", "archive", BASELINE], cwd=REPO, stdout=subprocess.PIPE)
    extracted = subprocess.run(["tar", "-x", "-C", str(archived)], stdin=archive.stdout, capture_output=True)
    archive.stdout.close()
    assert archive.wait() == 0 and extracted.returncode == 0
    mask = np.zeros((7, 11), dtype=np.bool_); mask[1:6, 2:9] = True
    cnt = np.zeros((7, 11), dtype=np.bool_); cnt[2:5, 4:7] = True
    np.save(tmp / "element.npy", mask); np.save(tmp / "cnt.npy", cnt); np.save(tmp / "valid.npy", np.ones((7, 11), bool))
    manifest = {"schema_version": "dl1_shape_v1_draft", "candidate_grid": 16, "products": [{
        "product_id": "SYNTHETIC-LEGACY", "lineage_id": "SYNTHETIC-LEGACY-FAMILY", "design_version": "SYNTHETIC-V1",
        "unit_size_um": [0.55, 0.35], "frame": {"frame_id": "SYNTHETIC-FRAME", "shape_yx": [7, 11],
        "pixel_pitch_um": 0.05, "origin_xy_um": [1.0, -2.0], "length_unit": "um", "axis_order": "yx"},
        "valid": "valid.npy", "cnt": "cnt.npy", "elements": [
        {"element_id": "A", "electrode_id": "A", "role": "A", "actual_layer": "L1", "mask": "element.npy", "observation": "observed"},
        {"element_id": "B", "electrode_id": "B", "role": "B", "actual_layer": "L2", "mask": "element.npy", "observation": "observed"}],
        "connectivity_observation": "complete", "connections": [{"connection_id": "SYNTHETIC-CNT", "kind": "CNT",
        "observation": "observed", "mask": "cnt.npy", "endpoints": [
        {"element_id": "A", "xy_um": [1.275, -1.825]}, {"element_id": "B", "xy_um": [1.275, -1.825]}]}]}]}
    path = tmp / "manifest.json"; path.write_text(json.dumps(manifest))
    checkpoint = tmp / "baseline.pt"
    outputs = []
    for label, package in [("baseline", archived / "src"), ("candidate", REPO / "src")]:
        output = tmp / (label + ".npz")
        env = dict(os.environ, PYTHONPATH=str(package), PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
        run = subprocess.run([sys.executable, "-c", DRIVER, str(path), str(output), str(checkpoint), label],
                             cwd=tmp, env=env, capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        record = json.loads(Path(str(output) + ".json").read_text())
        assert Path(record["module"]).resolve() == (package / "psdl1/data.py").resolve()
        outputs.append((output, record))
    return outputs, checkpoint, tmp


def test_legacy_npy_checkpoint_and_resume_policy(golden):
    outputs, _, _ = golden
    before, after = [np.load(x[0]) for x in outputs]
    assert set(before.files) == set(after.files)
    for key in before.files:
        assert before[key].dtype == after[key].dtype
        assert before[key].shape == after[key].shape
        assert before[key].tobytes() == after[key].tobytes(), key
    # Exact original frame features, normalized continuous endpoint, 9 fields.
    row = after["connection_features"][0, 0]
    np.testing.assert_array_equal(row[:8], np.array([0, 1, 0.5, (3.5 * 10 / 7 + 3) / 16,
                                                   0.5, (3.5 * 10 / 7 + 3) / 16, 0, 1], np.float32))
    assert outputs[0][1]["identity"] == outputs[1][1]["identity"]
    assert outputs[0][1]["cfg"] == outputs[1][1]["cfg"]
    assert outputs[0][1]["code_hash"] != outputs[1][1]["code_hash"]
    assert outputs[1][1]["model_input_version"] == "dl1_structured_v2_padded_endpoints"


@pytest.mark.parametrize("field,value", [("schema_version", "old"), ("model_input_version", "old"), ("fingerprint", "bad")])
def test_S20_archived_checkpoint_schema_and_fingerprint_rejections(golden, field, value):
    import torch
    from psdl1.checkpoint import read_checkpoint
    _, checkpoint, tmp = golden
    saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
    saved[field] = value
    bad = tmp / (field + ".pt"); torch.save(saved, bad)
    with pytest.raises(ValueError, match="incompatible|hash"):
        read_checkpoint(bad)
