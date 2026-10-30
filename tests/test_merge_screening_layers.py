"""screen-metrics/T3: merge_bundles carries the screening-only Layer sidecar into screening_layers/rNN/."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from png_helpers import write_png
import psdl1.import_png as importer
from psdl1 import merge_bundles as mb
from psdl1.import_png import file_sha256
from psdl1.png_contract import ImportOptions

BASELINE = "7db1472e2a1c92ca88382ecf84e245f57c53788d"
LAYERS = {"1-alpha.png": ((7, 7), (7, 8)), "2-beta.png": ((5, 5),)}


def _export(tmp_path, model, *, layers, shift=0):
    root = tmp_path / f"src-{model}-{shift}"
    write_png(root, f"{model}/Merged_Layer/A/1-X.png", pixels=((10, 10), (10, 11)))
    write_png(root, f"{model}/Merged_Layer/B/2-Y.png", pixels=((11, 11),))
    write_png(root, f"{model}/CNO/3-OCNT.png", pixels=((12, 12),))
    if layers:
        for name, pixels in LAYERS.items():
            write_png(root, f"{model}/Layer/{name}", pixels=tuple((y + shift, x) for y, x in pixels))
    meta = tmp_path / f"meta-{model}-{shift}.json"
    meta.write_text(json.dumps({"metadata_schema_version": "dl1_png_metadata_v1", "models": {
        model: {"lineage_id": f"LIN-{model}", "design_version": "V1"}}}))
    out = tmp_path / f"bundle-{model}-{shift}"
    summary, code = importer.run_import(ImportOptions(root, out, True, meta, 16))
    assert code == 0 and summary["export_status"] == "exported"
    return out


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("merge-layers")
    return {"plain1": _export(tmp, "P1", layers=False), "plain2": _export(tmp, "P2", layers=False),
            "plain3": _export(tmp, "P3", layers=False),
            "lay1": _export(tmp, "L1", layers=True), "lay2": _export(tmp, "L2", layers=True),
            "lay3": _export(tmp, "L3", layers=True), "lay2x": _export(tmp, "L2", layers=True, shift=1)}


def _files(root):
    return sorted(p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file())


def _report(root):
    return json.loads((Path(root) / "merge-report.json").read_text())


def _merge(tmp_path, base, add, name, round_index=2, base_round=1):
    out = tmp_path / name
    mb.merge_bundles(base, add, out, round_index=round_index, base_round=base_round)
    return out


def _sidecar(root):
    return json.loads((Path(root) / "screening-layers.json").read_text())


def _layer_files(root):
    return [f for f in _files(root) if f.startswith("screening_layers/")]


def test_neither_side_has_a_sidecar(bundles, tmp_path):
    out = _merge(tmp_path, bundles["plain1"], bundles["plain2"], "r02")
    assert not (out / "screening-layers.json").exists() and not (out / "screening_layers").exists()
    assert "screening_layers" not in _report(out)
    assert mb.verify_merged_bundle(out)["status"] == "verified"


def test_no_sidecar_merge_is_byte_identical_to_the_baseline_code(bundles, tmp_path):
    repo = Path(__file__).resolve().parents[1]
    try:
        source = subprocess.run(["git", "show", f"{BASELINE}:src/psdl1/merge_bundles.py"], cwd=repo,
                                capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("baseline commit is not available")
    path = tmp_path / "baseline_merge_bundles.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("psdl1.baseline_merge_bundles", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old_out, new_out = tmp_path / "old", tmp_path / "new"
    old.merge_bundles(bundles["plain1"], bundles["plain2"], old_out, round_index=2, base_round=1)
    mb.merge_bundles(bundles["plain1"], bundles["plain2"], new_out, round_index=2, base_round=1)
    assert _files(old_out) == _files(new_out)
    for name in _files(old_out):
        if name.startswith("assets/"):
            continue
        assert (old_out / name).read_bytes() == (new_out / name).read_bytes(), name
    # An old-format merge-report (no new field) is accepted by the new verifier and input check.
    assert mb.verify_merged_bundle(old_out)["status"] == "verified"
    assert mb.merge_inputs_match(old_out, bundles["plain1"], bundles["plain2"], 2, base_round=1) is None


def test_only_base_has_a_sidecar(bundles, tmp_path):
    out = _merge(tmp_path, bundles["lay1"], bundles["plain2"], "r02")
    assert mb.verify_merged_bundle(out)["status"] == "verified"
    assert mb.main(["--verify", str(out)]) == 0
    assert list(_sidecar(out)["products"]) == ["L1"]
    paths = _layer_files(out)
    assert len(paths) == 2 and all(p.startswith("screening_layers/r01/") for p in paths)
    assert [i["path"] for i in _report(out)["screening_layers"]] == paths
    assert {i["source_bundle"] for i in _report(out)["screening_layers"]} == {"base"}


def test_only_add_has_a_sidecar(bundles, tmp_path):
    out = _merge(tmp_path, bundles["plain1"], bundles["lay2"], "r02")
    assert mb.verify_merged_bundle(out)["status"] == "verified"
    assert list(_sidecar(out)["products"]) == ["L2"]
    assert all(p.startswith("screening_layers/r02/") for p in _layer_files(out))
    assert {i["source_bundle"] for i in _report(out)["screening_layers"]} == {"add"}


def test_both_sides_have_a_sidecar_and_masks_keep_their_bytes(bundles, tmp_path):
    out = _merge(tmp_path, bundles["lay1"], bundles["lay2"], "r02")
    assert mb.verify_merged_bundle(out)["status"] == "verified"
    assert sorted(_sidecar(out)["products"]) == ["L1", "L2"]
    for item in _report(out)["screening_layers"]:
        origin = bundles["lay1" if item["source_bundle"] == "base" else "lay2"]
        assert file_sha256(out / item["path"]) == file_sha256(origin / item["source_path"]) == item["sha256"]
        assert item["path"].startswith("screening_layers/r01/" if item["source_bundle"] == "base" else "screening_layers/r02/")
    layer = _sidecar(out)["products"]["L2"]["layers"][0]
    assert np.load(out / layer["mask"]).dtype == np.bool_
    assert mb.merge_inputs_match(out, bundles["lay1"], bundles["lay2"], 2, base_round=1) is None


def test_duplicate_product_with_sidecars_is_refused(bundles, tmp_path):
    with pytest.raises(ValueError, match="duplicate product IDs"):
        _merge(tmp_path, bundles["lay2"], bundles["lay2x"], "dup")


def test_remerge_of_a_merged_base_keeps_old_paths(bundles, tmp_path):
    second = _merge(tmp_path, bundles["lay1"], bundles["lay2"], "r02")
    third = _merge(tmp_path, second, bundles["lay3"], "r03", round_index=3, base_round=None)
    assert mb.verify_merged_bundle(third)["status"] == "verified"
    old = {i["path"] for i in _report(second)["screening_layers"]}
    new = _report(third)["screening_layers"]
    kept = [i for i in new if i["source_bundle"] == "base"]
    assert {i["path"] for i in kept} == old and all(i["path"] == i["source_path"] for i in kept)
    assert all(i["path"].startswith("screening_layers/r03/") for i in new if i["source_bundle"] == "add")
    assert sorted(_sidecar(third)["products"]) == ["L1", "L2", "L3"]
    plain_tail = _merge(tmp_path, second, bundles["plain3"], "r03b", round_index=3, base_round=None)
    assert mb.verify_merged_bundle(plain_tail)["status"] == "verified"
    assert {i["path"] for i in _report(plain_tail)["screening_layers"]} == old
    from_plain = _merge(tmp_path, _merge(tmp_path, bundles["plain1"], bundles["plain2"], "p02"), bundles["lay3"], "p03",
                        round_index=3, base_round=None)
    assert mb.verify_merged_bundle(from_plain)["status"] == "verified"
    assert all(p.startswith("screening_layers/r03/") for p in _layer_files(from_plain))


def _tampered(tmp_path, bundles, name, mutate, *, reseal=True):
    out = _merge(tmp_path, bundles["lay1"], bundles["lay2"], name + "-src")
    copy = tmp_path / name
    shutil.copytree(out, copy)
    mutate(copy)
    if reseal:
        mb._seal_bundle(copy)
    return copy


def _flip_npy(copy):
    target = copy / _report(copy)["screening_layers"][0]["path"]
    data = bytearray(target.read_bytes())
    data[-1] ^= 1
    target.unlink()
    target.write_bytes(bytes(data))


def _json_path(copy):
    doc = _sidecar(copy)
    layer = doc["products"]["L1"]["layers"][0]
    flat = layer["mask"].replace("screening_layers/r01/", "screening_layers/")
    (copy / layer["mask"]).rename(copy / flat)  # the file moves with its sidecar path, outside rNN
    layer["mask"] = flat
    (copy / "screening-layers.json").write_text(json.dumps(doc, sort_keys=True, indent=2) + "\n")


def _extra_file(copy):
    (copy / "screening_layers" / "r02" / "extra.npy").write_bytes(b"x")


def _report_list(copy):
    doc = _report(copy)
    doc["screening_layers"].pop()
    (copy / "merge-report.json").write_text(json.dumps(doc))


def _report_dropped(copy):
    doc = _report(copy)
    del doc["screening_layers"]
    (copy / "merge-report.json").write_text(json.dumps(doc))


@pytest.mark.parametrize("mutate, match", [
    (_flip_npy, "mask hash mismatch"),
    (_json_path, "not under screening_layers/rNN"),
    (_extra_file, "unreferenced or missing screening layer"),
    (_report_list, "merge report screening layers differ"),
    (_report_dropped, "does not list the screening layers"),
])
def test_resealed_sidecar_tampering_is_rejected(bundles, tmp_path, mutate, match):
    copy = _tampered(tmp_path, bundles, mutate.__name__, mutate)
    with pytest.raises(ValueError, match=match):
        mb.verify_merged_bundle(copy)


def test_unsealed_sidecar_change_is_rejected(bundles, tmp_path):
    copy = _tampered(tmp_path, bundles, "unsealed", _flip_npy, reseal=False)
    with pytest.raises(ValueError, match="hash mismatch"):
        mb.verify_merged_bundle(copy)


def test_merge_inputs_match_notices_sidecar_differences(bundles, tmp_path):
    out = _merge(tmp_path, bundles["lay1"], bundles["lay2"], "r02")
    assert mb.merge_inputs_match(out, bundles["lay1"], bundles["lay2"], 2, base_round=1) is None
    assert "add integrity_sha256" in mb.merge_inputs_match(out, bundles["lay1"], bundles["lay2x"], 2, base_round=1)
    plain = _merge(tmp_path, bundles["plain1"], bundles["plain2"], "plain")
    assert mb.merge_inputs_match(plain, bundles["lay1"], bundles["plain2"], 2, base_round=1) is not None
    assert mb.merge_inputs_match(out, bundles["plain1"], bundles["plain2"], 2, base_round=1) is not None
    # Same recorded hashes but a sidecar that disappeared from the merged folder.
    stripped = tmp_path / "stripped"
    shutil.copytree(out, stripped)
    (stripped / "screening-layers.json").unlink()
    assert "sidecar" in mb.merge_inputs_match(stripped, bundles["lay1"], bundles["lay2"], 2, base_round=1)
