"""screen-metrics/T2: v2 `Layer/` is a screening-only sidecar; training inputs are unchanged."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from png_helpers import write_png, write_stack_order
import psdl1.import_png as importer
from psdl1.png_contract import ImportOptions, canonical_id, mask_hash

SHAPE = (1001, 1203)
LAYER_PIXELS = {"1-alpha.png": ((7, 7), (7, 8)), "2-beta.png": ((5, 5),), "3-X.png": ((9, 9), (9, 10), (9, 11))}


def _electrodes(root, model="M"):
    write_png(root, f"{model}/Merged_Layer/A/1-X.png", pixels=((10, 10), (10, 11)))
    write_png(root, f"{model}/Merged_Layer/B/2-Y.png", pixels=((11, 11),))
    write_png(root, f"{model}/CNO/3-OCNT.png", pixels=((12, 12),))


def _v2(root, model="M", layers=True):
    _electrodes(root, model)
    if layers:
        for name, pixels in LAYER_PIXELS.items():
            write_png(root, f"{model}/Layer/{name}", pixels=pixels)


def _metadata(path, model="M"):
    path.write_text(json.dumps({"metadata_schema_version": "dl1_png_metadata_v1", "models": {
        model: {"lineage_id": "LIN", "design_version": "V1"}}}))
    return path


def _export(tmp_path, name, *, layers=True, stack_order=None):
    root = tmp_path / f"src-{name}"
    _v2(root, layers=layers)
    out = tmp_path / f"out-{name}"
    options = ImportOptions(root, out, True, _metadata(tmp_path / f"meta-{name}.json"), 16, stack_order=stack_order)
    summary, code = importer.run_import(options)
    assert code == 0 and summary["export_status"] == "exported"
    return out


def _report(out):
    return json.loads((out / "import-report.json").read_text())


def _integrity_files(out):
    return json.loads((out / "bundle-integrity.json").read_text())["files"]


def _mask(pixels):
    mask = np.zeros(SHAPE, dtype=bool)
    for y, x in pixels:
        mask[y, x] = True
    return mask


def test_export_writes_sidecar_in_deterministic_order(tmp_path):
    out = _export(tmp_path, "a")
    doc = json.loads((out / "screening-layers.json").read_text())
    assert doc["schema_version"] == "dl1_screening_layers_v1" and list(doc["products"]) == ["M"]
    entry = doc["products"]["M"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert entry["frame_id"] == manifest["products"][0]["frame"]["frame_id"]
    assert [(l["layer"], l["filename_number"], l["relative_path"]) for l in entry["layers"]] == [
        ("alpha", 1, "M/Layer/1-alpha.png"), ("beta", 2, "M/Layer/2-beta.png"), ("X", 3, "M/Layer/3-X.png")]
    for layer in entry["layers"]:
        pixels = LAYER_PIXELS[Path(layer["relative_path"]).name]
        assert layer["mask"] == "screening_layers/" + canonical_id("screening_layer", ["M", layer["layer"]]) + ".npy"
        mask = np.load(out / layer["mask"], allow_pickle=False)
        assert mask.dtype == np.bool_ and mask.shape == SHAPE and (mask == _mask(pixels)).all()
        assert layer["mask_sha256"] == mask_hash(mask)
        assert importer.file_sha256(out / layer["mask"]) == _integrity_files(out)[layer["mask"]]
    assert importer.verify_bundle(out)["status"] == "verified"
    assert _report(out)["status_summary"][0]["screening_layers"] == 3


def test_relations_mode_writes_the_same_sidecar(tmp_path):
    image_only = _export(tmp_path, "img")
    order = write_stack_order(tmp_path / "order.json", ["X", "Y", "OCNT"])
    relations = _export(tmp_path, "rel", stack_order=order)
    assert _report(relations)["cnt_mode"] == "relations"
    assert (relations / "screening-layers.json").read_bytes() == (image_only / "screening-layers.json").read_bytes()
    assert importer.verify_bundle(relations)["status"] == "verified"


def test_layer_presence_changes_only_the_sidecar(tmp_path):
    with_layer = _export(tmp_path, "with")
    without = _export(tmp_path, "without", layers=False)
    assert (with_layer / "manifest.json").read_bytes() == (without / "manifest.json").read_bytes()
    a, b = _report(with_layer), _report(without)
    for key in ("generated_asset_hashes", "product_input_identities", "generated_manifest_sha256"):
        assert a[key] == b[key] and a[key]
    files_a, files_b = _integrity_files(with_layer), _integrity_files(without)
    assert {p for p in files_a if p not in files_b} == {
        "screening-layers.json", *[p for p in files_a if p.startswith("screening_layers/")]}
    assert not {p for p in files_b if p not in files_a}
    assert all(files_a[p] == files_b[p] for p in files_b if p not in {"import-report.json"} and not p.startswith("preview"))
    assert not (without / "screening-layers.json").exists() and not (without / "screening_layers").exists()
    assert "layer_files" not in json.dumps(b["inventory"]) and "screening_layers" not in b["status_summary"][0]
    assert [x["asset_id"] for x in a["inventory"]["models"][0]["assets"]] == [
        x["asset_id"] for x in b["inventory"]["models"][0]["assets"]]


def test_sidecar_is_not_part_of_manifest_or_assets(tmp_path):
    out = _export(tmp_path, "a")
    manifest = (out / "manifest.json").read_text()
    assert "Layer" not in manifest and "screening" not in manifest
    assert not any(p.startswith("assets/") and "screening" in p for p in _integrity_files(out))
    assert len(list((out / "assets").iterdir())) == len({
        r[0] for r in _report(out)["generated_asset_hashes"]})


def _tampered(tmp_path, out, name, mutate):
    copy = tmp_path / f"tamper-{name}"
    shutil.copytree(out, copy)
    mutate(copy)
    importer._seal_bundle(copy)
    return copy


def _sidecar(copy):
    return json.loads((copy / "screening-layers.json").read_text())


def _write_sidecar(copy, doc):
    (copy / "screening-layers.json").write_text(json.dumps(doc, sort_keys=True, indent=2) + "\n")


def _first_layer(doc):
    return doc["products"]["M"]["layers"][0]


def _flip_npy(copy):
    path = copy / _first_layer(_sidecar(copy))["mask"]
    mask = np.load(path, allow_pickle=False)
    mask[0, 0] = ~mask[0, 0]
    np.save(path, mask, allow_pickle=False)


def _wrong_dtype(copy):
    path = copy / _first_layer(_sidecar(copy))["mask"]
    np.save(path, np.load(path, allow_pickle=False).astype(np.uint8), allow_pickle=False)


def _wrong_shape(copy):
    path = copy / _first_layer(_sidecar(copy))["mask"]
    np.save(path, np.zeros((3, 3), dtype=np.bool_), allow_pickle=False)


def _bad_hash(copy):
    doc = _sidecar(copy)
    _first_layer(doc)["mask_sha256"] = "0" * 64
    _write_sidecar(copy, doc)


def _extra_file(copy):
    np.save(copy / "screening_layers" / "extra.npy", np.zeros(SHAPE, dtype=np.bool_), allow_pickle=False)


def _unknown_product(copy):
    doc = _sidecar(copy)
    doc["products"]["ZZ"] = doc["products"].pop("M")
    _write_sidecar(copy, doc)


def _wrong_frame(copy):
    doc = _sidecar(copy)
    doc["products"]["M"]["frame_id"] = "frame_other"
    _write_sidecar(copy, doc)


def _escape(copy):
    doc = _sidecar(copy)
    _first_layer(doc)["mask"] = "screening_layers/../assets/" + next((copy / "assets").iterdir()).name
    _write_sidecar(copy, doc)


def _absolute(copy):
    doc = _sidecar(copy)
    _first_layer(doc)["mask"] = str(copy / _first_layer(doc)["mask"])
    _write_sidecar(copy, doc)


def _case_duplicate(copy):
    doc = _sidecar(copy)
    doc["products"]["M"]["layers"][1]["layer"] = doc["products"]["M"]["layers"][0]["layer"].upper()
    _write_sidecar(copy, doc)


def _bad_version(copy):
    doc = _sidecar(copy)
    doc["schema_version"] = "dl1_screening_layers_v2"
    _write_sidecar(copy, doc)


def _drop_sidecar_json(copy):
    (copy / "screening-layers.json").unlink()


def _drop_layer_entry(copy):
    doc = _sidecar(copy)
    doc["products"]["M"]["layers"].pop()
    _write_sidecar(copy, doc)


def _empty_products(copy):
    doc = _sidecar(copy)
    for layer in doc["products"]["M"]["layers"]:
        (copy / layer["mask"]).unlink()
    doc["products"] = {}
    _write_sidecar(copy, doc)


def _nested_path(copy):
    doc = _sidecar(copy)
    layer = _first_layer(doc)
    nested = layer["mask"].replace("screening_layers/", "screening_layers/sub/")
    (copy / "screening_layers" / "sub").mkdir()
    (copy / layer["mask"]).rename(copy / nested)  # the file moves with its sidecar path
    layer["mask"] = nested
    _write_sidecar(copy, doc)


@pytest.mark.parametrize("mutation", [
    _flip_npy, _wrong_dtype, _wrong_shape, _bad_hash, _extra_file, _unknown_product, _wrong_frame, _escape,
    _absolute, _case_duplicate, _bad_version, _drop_sidecar_json, _drop_layer_entry, _empty_products, _nested_path])
def test_verify_rejects_tampered_sidecar_even_when_resealed(tmp_path, mutation):
    out = _export(tmp_path, "a")
    assert importer.verify_bundle(out)["status"] == "verified"
    copy = _tampered(tmp_path, out, mutation.__name__, mutation)
    with pytest.raises(ValueError):
        importer.verify_bundle(copy)


def test_verify_without_reseal_rejects_any_sidecar_change(tmp_path):
    out = _export(tmp_path, "a")
    copy = tmp_path / "raw"
    shutil.copytree(out, copy)
    _flip_npy(copy)
    with pytest.raises(ValueError, match="hash mismatch"):
        importer.verify_bundle(copy)


def test_layerless_bundle_verifies_as_before(tmp_path):
    out = _export(tmp_path, "plain", layers=False)
    assert importer.verify_bundle(out)["status"] == "verified"
    copy = _tampered(tmp_path, out, "stray", lambda c: (c / "screening_layers").mkdir())
    importer.verify_bundle(copy)  # an empty directory carries no file to reference
    copy = _tampered(tmp_path, out, "stray-file", lambda c: ((c / "screening_layers").mkdir(),
                     np.save(c / "screening_layers" / "x.npy", np.zeros(SHAPE, dtype=np.bool_))))
    with pytest.raises(ValueError):
        importer.verify_bundle(copy)


def test_source_change_between_scan_and_export_is_refused(tmp_path, monkeypatch):
    root = tmp_path / "src"
    _v2(root)
    real = importer._write_screening_layers

    def mutate_then_write(stage, root_, model, policy):
        write_png(root_, "M/Layer/2-beta.png", pixels=((6, 6),))
        return real(stage, root_, model, policy)

    monkeypatch.setattr(importer, "_write_screening_layers", mutate_then_write)
    options = ImportOptions(root, tmp_path / "out", True, _metadata(tmp_path / "meta.json"), 16)
    with pytest.raises(RuntimeError, match="source changed"):
        importer.run_import(options)
