"""screen-metrics/T2: v2 `Layer/` scan and strict errors. Torch-free: runs on the Python 3.9 smoke interpreter."""
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from png_helpers import write_png, write_stack_order
from psdl1.png_contract import mask_hash
from psdl1.png_source import discover_source

SHAPE = (1001, 1203)
LAYER_PIXELS = {"1-alpha.png": ((7, 7), (7, 8)), "2-beta.png": ((5, 5),), "3-X.png": ((9, 9), (9, 10), (9, 11))}


def _v2(root, model="M", layers=True):
    write_png(root, f"{model}/Merged_Layer/A/1-X.png", pixels=((10, 10), (10, 11)))
    write_png(root, f"{model}/Merged_Layer/B/2-Y.png", pixels=((11, 11),))
    write_png(root, f"{model}/CNO/3-OCNT.png", pixels=((12, 12),))
    if layers:
        for name, pixels in LAYER_PIXELS.items():
            write_png(root, f"{model}/Layer/{name}", pixels=pixels)


def _mask(pixels):
    mask = np.zeros(SHAPE, dtype=bool)
    for y, x in pixels:
        mask[y, x] = True
    return mask


def _reasons(issues):
    return [i.reason for i in issues]


def test_scan_records_layer_files_without_touching_electrode_inventory(tmp_path):
    _v2(tmp_path)
    model = discover_source(tmp_path).models[0]
    assert [(f.name, f.filename_number, f.relative_path, f.shape_yx) for f in model.layer_files] == [
        ("alpha", 1, "M/Layer/1-alpha.png", SHAPE), ("beta", 2, "M/Layer/2-beta.png", SHAPE),
        ("X", 3, "M/Layer/3-X.png", SHAPE)]
    assert all(len(f.file_sha256) == 64 and f.mask_sha256 == mask_hash(_mask(LAYER_PIXELS[Path(f.relative_path).name]))
               for f in model.layer_files)
    assert len(model.assets) == 3 and model.roles == ("A", "B") and model.layers == ("OCNT", "X", "Y")
    assert not any("/Layer/" in a.relative_path for a in model.assets)


def test_layer_named_like_electrode_layer_keeps_stack_order(tmp_path):
    order = write_stack_order(tmp_path / "order.json", ["X", "Y", "OCNT"])
    _v2(tmp_path / "with")
    _v2(tmp_path / "without", layers=False)
    with_layer = discover_source(tmp_path / "with", stack_order=order).models[0]
    without = discover_source(tmp_path / "without", stack_order=order).models[0]
    assert with_layer.valid and without.valid
    assert with_layer.stack_order.status == "resolved"
    assert with_layer.stack_order == without.stack_order
    assert [a.asset_id for a in with_layer.assets] == [a.asset_id for a in without.assets]
    assert with_layer.layers == without.layers


@pytest.mark.parametrize("case", ["name", "casefold", "shape", "decode", "grey"])
def test_layer_errors_are_model_errors(tmp_path, case):
    _v2(tmp_path, layers=False)
    write_png(tmp_path, "M/Layer/1-ok.png")
    if case == "name":
        write_png(tmp_path, "M/Layer/nonumber.png")
        expected = "expected positive numeric prefix"
    elif case == "casefold":
        write_png(tmp_path, "M/Layer/2-Foo.png")
        write_png(tmp_path, "M/Layer/3-fOO.png")
        expected = "layer_name_duplicate"
    elif case == "shape":
        write_png(tmp_path, "M/Layer/2-wide.png", shape=(1001, 1204))
        expected = "differs from model frame"
    elif case == "decode":
        (tmp_path / "M/Layer/2-broken.png").write_bytes(b"not a png")
        expected = "M/Layer/2-broken.png"
    else:
        Image.fromarray(np.full(SHAPE, 128, dtype=np.uint8)).save(tmp_path / "M/Layer/2-grey.png")
        expected = "exactly black or white"
    result = discover_source(tmp_path)
    model = result.models[0]
    assert not result.valid and not model.valid
    assert any(expected in i.reason or expected in i.path for i in model.errors)
    assert len(model.assets) == 3 and model.frame is not None


def test_empty_layer_folder_warns_and_yields_no_entries(tmp_path):
    _v2(tmp_path, layers=False)
    (tmp_path / "M" / "Layer").mkdir()
    result = discover_source(tmp_path)
    model = result.models[0]
    assert result.valid and model.layer_files == ()
    assert sum("layer_folder_empty" in r for r in _reasons(model.warnings)) == 1
    assert not any("layer_folder_screening_only" in r for r in _reasons(model.warnings))


def test_legacy_layer_folder_stays_an_electrode(tmp_path):
    write_png(tmp_path, "M/Layer/1-X.png")
    write_png(tmp_path, "M/A/2-Y.png")
    model = discover_source(tmp_path).models[0]
    assert model.layer_files == () and model.roles == ("A", "Layer")
    assert not any("layer_folder" in r for r in _reasons(model.warnings))
