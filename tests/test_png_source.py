"""T1 plan S01-S05, S12, S24 and S25: independent synthetic oracles."""
import builtins
from dataclasses import FrozenInstanceError
import json
import shutil
import warnings

import numpy as np
from PIL import Image
import pytest

from png_helpers import write_png
from psdl1.png_contract import (ImportOptions, ImportReport, canonical_id, component_id,
                               model_identity, require_report_version)
from psdl1.png_source import (PNGSourceError, decode_png, discover_source, load_metadata,
                             parse_png_name, report_payload)


@pytest.mark.parametrize("mode", ["1", "L", "RGB", "LA", "RGBA", "P"])
@pytest.mark.parametrize("pixels", [((0, 0), (2, 3)), ()])
def test_png_modes_and_black_foreground(tmp_path, mode, pixels):
    path = write_png(tmp_path, "M/A/1-X.png", shape=(3, 4), pixels=pixels, mode=mode)
    mask, actual_mode, _ = decode_png(path)
    expected = [[False] * 4 for _ in range(3)]
    for y, x in pixels:
        expected[y][x] = True
    assert mask.tolist() == expected
    assert actual_mode == mode
    assert mask.dtype == np.bool_
    assert not mask.flags.writeable


@pytest.mark.parametrize("case", ["grey", "colour", "alpha0", "alpha128", "palette_alpha",
                                  "palette_colour", "L_transparency", "16bit", "corrupt",
                                  "truncated", "animated", "orientation", "not_png"])
def test_png_modes_and_black_foreground_rejections(tmp_path, case):
    path = tmp_path / "bad.png"
    if case == "grey":
        Image.new("L", (3, 4), 127).save(path)
    elif case == "colour":
        Image.new("RGB", (3, 4), (0, 1, 0)).save(path)
    elif case.startswith("alpha"):
        Image.new("RGBA", (3, 4), (0, 0, 0, int(case[5:]))).save(path)
    elif case.startswith("palette"):
        write_png(tmp_path, "bad.png", shape=(3, 4), mode="P", pixels=((0, 0),),
                  **({"transparency": 0} if case == "palette_alpha" else {}))
        if case == "palette_colour":
            with Image.open(path) as image:
                image.putpalette([0, 1, 0, 255, 255, 255] + [0] * (256 * 3 - 6))
                image.save(path)
    elif case == "L_transparency":
        Image.new("L", (3, 4), 0).save(path, transparency=0)
    elif case == "16bit":
        Image.fromarray(np.full((3, 4), 65535, dtype=np.uint16)).save(path)
    elif case == "corrupt":
        path.write_bytes(b"not a png")
    elif case == "truncated":
        write_png(tmp_path, "bad.png", shape=(3, 4))
        path.write_bytes(path.read_bytes()[:45])
    elif case == "animated":
        Image.new("L", (3, 4), 0).save(path, save_all=True,
                                       append_images=[Image.new("L", (3, 4), 255)])
    elif case == "orientation":
        exif = Image.Exif(); exif[274] = 6
        Image.new("L", (3, 4), 0).save(path, exif=exif)
    else:
        Image.new("L", (3, 4), 0).save(path, format="BMP")
    with pytest.raises(PNGSourceError) as err:
        decode_png(path)
    assert str(path) in str(err.value)
    assert err.value.issue.exception_class


def test_numeric_stack_and_first_separator(tmp_path):
    assert parse_png_name("1-pxl.png") == (1, "pxl")
    assert parse_png_name("2-SD2.png") == (2, "SD2")
    assert parse_png_name("006_OCNT.png") == (6, "OCNT")
    assert parse_png_name("002-name_with-more_parts.png") == (2, "name_with-more_parts")
    for name in ["0-X.png", "-1-X.png", "X.png", "2-.png", "2_X.PNG", "1.5-X.png", "1- X.png"]:
        with pytest.raises(ValueError):
            parse_png_name(name)
    for name in ["10-Z.png", "2-SD2.png", "1-pxl.png"]:
        write_png(tmp_path, "M/A/" + name)
    write_png(tmp_path, "M/cno/006_OCNT.png")
    model, = discover_source(tmp_path).models
    assert model.stack_table == ((1, "pxl"), (2, "SD2"), (6, "OCNT"), (10, "Z"))
    assert [a.stack for a in model.assets] == [1, 2, 6, 10]
    assert model.cnt_assets[0].original_filename == "006_OCNT.png"


@pytest.mark.parametrize("case", ["shape", "height", "duplicate", "conflict", "repeated_binding",
                                  "layout", "flat", "badname", "whitespace"])
def test_shape_duplicate_and_stack_conflict_errors(tmp_path, case):
    first = write_png(tmp_path, "M/A/6-X.png")
    relative = {"shape": "M/B/7-Y.png", "height": "M/B/7-Y.png",
                "duplicate": "M/A/006_X.png", "conflict": "M/B/6-Y.png",
                "repeated_binding": "M/A/7-X.png", "layout": "M/A/deeper/7-Y.png",
                "flat": "M/7-Y.png", "badname": "M/B/nonsense.png",
                "whitespace": "M/ B/7-Y.png"}[case]
    path = write_png(tmp_path, relative, shape=(1001, 1204) if case == "shape" else
                     (1002, 1203) if case == "height" else (1001, 1203))
    inventory = discover_source(tmp_path)
    assert not inventory.valid
    errors = inventory.models[0].errors
    assert any(relative in e.path or path.parent.relative_to(tmp_path).as_posix() == e.path for e in errors)
    if case in {"shape", "height", "duplicate", "conflict", "repeated_binding"}:
        assert any(first.relative_to(tmp_path).as_posix() in e.reason for e in errors)
    assert inventory.models[0].resource_estimate.frame_pixels is None


def test_frame_extent_core_bbox_and_source_containment(tmp_path):
    root = tmp_path / "root"
    path = write_png(root, "M/A/1-X.png", pixels=((0, 0), (1000, 1202)))
    original = path.read_bytes()
    model, = discover_source(root).models
    assert model.valid
    frame = model.frame
    assert frame.shape_yx == (1001, 1203)
    assert frame.core_bbox_xyxy_px == (500, 500, 703, 501)
    assert frame.frame_extent_um == pytest.approx((120.3, 100.1))
    assert frame.origin_xy_um == (0, 0) and frame.pixel_pitch_um == 0.1
    assert frame.pixel_centre_um(0, 0) == (0.05, 0.05)
    assert frame.pixel_centre_um(1202, 1000) == pytest.approx((120.25, 100.05))
    assert model.assets[0].mask[0, 0] and model.assets[0].mask[1000, 1202]
    assert path.read_bytes() == original
    for index, shape in enumerate([(1000, 1203), (1001, 1000), (999, 1001)]):
        write_png(root, f"small{index}/A/1-X.png", shape=shape)
    outside = write_png(tmp_path, "outside.png")
    (root / "M/A/2-Y.png").symlink_to(outside)
    (root / "escape").symlink_to(tmp_path)
    inventory = discover_source(root)
    assert not inventory.valid
    assert any("escaping" in e.reason for e in inventory.errors)
    assert any(e.path == "M/A/2-Y.png" and "escaping" in e.reason for e in inventory.models[0].errors)
    assert all(not m.valid for m in inventory.models if m.model_id.startswith("small"))


def metadata_file(tmp_path, document):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(document))
    return path


def test_explicit_metadata_and_exact_names(tmp_path):
    for relative in ["M/A/1-X.png", "M/B/2-x.png", "Q/a/1-X.png"]:
        write_png(tmp_path / "source", relative)
    inv = discover_source(tmp_path / "source")
    assert inv.models[0].roles == ("A", "B")
    assert inv.role_union == ("A", "B", "a")
    assert inv.models[0].layers == ("X", "x")
    assert load_metadata(None, ["M", "Q"]).pending_models == ("M", "Q")
    doc = {"metadata_schema_version": "dl1_png_metadata_v1",
           "models": {"M": {"lineage_id": "family", "design_version": "v2"}}}
    result = load_metadata(metadata_file(tmp_path, doc), ["M", "Q"])
    assert result.pending_models == ("Q",)
    assert result.models[0].lineage_id == "family"
    with pytest.raises(ValueError, match="missing explicit"):
        load_metadata(metadata_file(tmp_path, doc), ["M", "Q"], require_complete=True)
    write_png(tmp_path / "source", "R/A/1-X.png")
    write_png(tmp_path / "source", "R/B/2-X.png")
    repeated = discover_source(tmp_path / "source").models[2]
    assert repeated.repeated_layers == ("X",)
    assert len(repeated.assets) == 2  # no merging distinct bindings
    write_png(tmp_path / "source", "R/A/2-X.png")
    assert not discover_source(tmp_path / "source").models[2].valid


@pytest.mark.parametrize("case", ["version", "unknown_root", "unknown_model", "unknown_field",
                                  "hotspot", "missing_field", "whitespace", "numeric", "null", "models_list", "root_list"])
def test_explicit_metadata_and_exact_names_rejections(tmp_path, case):
    doc = {"metadata_schema_version": "dl1_png_metadata_v1",
           "models": {"M": {"lineage_id": "f", "design_version": "v"}}}
    if case == "version": doc["metadata_schema_version"] = "v2"
    elif case == "unknown_root": doc["grid"] = 100
    elif case == "unknown_model": doc["models"]["Q"] = doc["models"]["M"]
    elif case in {"unknown_field", "hotspot"}: doc["models"]["M"][case] = True
    elif case == "missing_field": del doc["models"]["M"]["lineage_id"]
    elif case == "models_list": doc["models"] = []
    elif case == "root_list": doc = []
    else: doc["models"]["M"]["lineage_id"] = {"whitespace": " f", "numeric": 1, "null": None}[case]
    with pytest.raises(ValueError):
        load_metadata(metadata_file(tmp_path, doc), ["M"])


@pytest.mark.parametrize("text", [
    '{"metadata_schema_version":"dl1_png_metadata_v1","models":{},"models":{}}',
    '{"metadata_schema_version":"dl1_png_metadata_v1","models":{"M":{"lineage_id":"a","lineage_id":"b","design_version":"v"}}}',
    '{"metadata_schema_version":"dl1_png_metadata_v1","models":{"M":{"lineage_id":NaN,"design_version":"v"}}}',
])
def test_metadata_duplicate_keys_and_nonfinite_rejected(tmp_path, text):
    path = tmp_path / "metadata.json"; path.write_text(text)
    with pytest.raises(ValueError): load_metadata(path, ["M"])


def test_enumeration_and_root_relocation_invariance(tmp_path, monkeypatch):
    first, second = tmp_path / "one", tmp_path / "two"
    for relative in ["M/B/10-Z.png", "M/A/2-X.png", "M/cno/006_C.png"]:
        write_png(first, relative)
    before = discover_source(first).models[0]
    shutil.copytree(first, second)
    (second / "M/A/2-X.png").rename(second / "M/A/002_X.png")
    write_png(second, "M/B/10-Z.png", compress_level=0)
    from pathlib import Path
    iterdir = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda p: iter(reversed(list(iterdir(p)))))
    after = discover_source(second).models[0]
    assert model_identity(before) == model_identity(after)
    assert [a.asset_id for a in before.assets] == [a.asset_id for a in after.assets]
    assert [a.mask_sha256 for a in before.assets] == [a.mask_sha256 for a in after.assets]
    assert before.assets[-1].file_sha256 != after.assets[-1].file_sha256
    write_png(second, "M/B/10-Z.png", pixels=((1, 1),))
    assert model_identity(discover_source(second).models[0]) != model_identity(before)
    assert canonical_id("element", ["M", "A", 2, "X"]) != canonical_id("element", ["M", "B", 2, "X"])
    assert component_id("M", 6, "C", ("M", "cno", None, 6, "C"), (1001, 1203),
                        (3, 4, 5, 5), np.array([[True, False]])) != component_id(
                            "M", 6, "C", ("M", "cno", None, 6, "C"), (1001, 1203),
                            (3, 4, 5, 5), np.array([[False, True]]))


def test_vocabulary_inventory_and_oov_search_diagnostics(tmp_path):
    for relative in ["M/A/1-X.png", "M/A/2-Y.png", "M/B/1-X.png", "M/cno/3-C.png",
                     "Q/A/1-X.png", "Q/D/2-Z.png", "Q/cno/3-C.png"]:
        write_png(tmp_path, relative)
    inv = discover_source(tmp_path)
    assert inv.valid
    assert [(m.roles, m.layers) for m in inv.models] == [(("A", "B"), ("C", "X", "Y")),
                                                        (("A", "D"), ("C", "X", "Z"))]
    assert inv.role_union == ("A", "B", "D") and inv.layer_union == ("C", "X", "Y", "Z")
    assert inv.unique_roles == ("B", "D") and inv.unique_layers == ("Y", "Z")
    report = ImportReport(inv, load_metadata(None, ["M", "Q"]))
    payload = report_payload(report)
    assert payload["metadata"]["pending_models"] == ["M", "Q"]
    assert payload["electrical_verification"] == "not_performed"
    assert payload["inventory"]["models"][0]["resource_estimate"]["cnt_component_count"] is None
    assert "mask" not in payload["inventory"]["models"][0]["assets"][0]
    json.dumps(payload, allow_nan=False)
    with pytest.raises(FrozenInstanceError): report.export_status = "exported"
    require_report_version(payload)
    with pytest.raises(ValueError): require_report_version({"report_schema_version": "v2"})


def test_pillow_pixel_limit_error_and_explicit_override(tmp_path, monkeypatch):
    write_png(tmp_path, "warning/A/1-X.png", shape=(1001, 1001))  # N=1,002,001
    write_png(tmp_path, "hard/A/1-X.png", shape=(1001, 2000))  # N=2,002,000
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000_000)
    filters = warnings.filters[:]
    inv = discover_source(tmp_path)
    assert Image.MAX_IMAGE_PIXELS == 1_000_000 and warnings.filters == filters
    assert not inv.valid
    hard, warning = inv.models
    assert hard.errors[0].exception_class == "DecompressionBombError"
    assert hard.errors[0].pixels == 2_002_000
    assert warning.errors[0].exception_class == "DecompressionBombWarning"
    assert warning.errors[0].pixels == 1_002_001
    assert all(m.errors[0].effective_max_image_pixels == 1_000_000 for m in inv.models)
    allowed = discover_source(tmp_path, png_max_image_pixels=3_000_000)
    assert allowed.valid
    assert allowed.decoder.default_max_image_pixels == 1_000_000
    assert allowed.decoder.requested_max_image_pixels == allowed.decoder.effective_max_image_pixels == 3_000_000
    assert allowed.decoder.explicit_override
    assert Image.MAX_IMAGE_PIXELS == 1_000_000 and warnings.filters == filters
    with pytest.raises(PNGSourceError):
        decode_png(tmp_path / "hard/A/1-X.png", png_max_image_pixels=10)
    assert Image.MAX_IMAGE_PIXELS == 1_000_000 and warnings.filters == filters
    for value in [0, -1, False, True, 1.5, float("inf"), "3000000"]:
        with pytest.raises(ValueError): discover_source(tmp_path, png_max_image_pixels=value)
        with pytest.raises(ValueError): ImportOptions(tmp_path, tmp_path / "out", png_max_image_pixels=value)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", None)
    with pytest.raises(ValueError, match="disabled"): discover_source(tmp_path)
    assert discover_source(tmp_path, png_max_image_pixels=3_000_000).valid
    assert Image.MAX_IMAGE_PIXELS is None


def test_npy_path_and_png_contract_import_without_pillow(monkeypatch, manifest_case):
    _, write, _ = manifest_case
    original = builtins.__import__
    def no_pillow(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("synthetic missing optional dependency")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_pillow)
    from psdl1.data import load_manifest
    assert len(load_manifest(write())) == 1
    with pytest.raises(ImportError, match="offline wheels"):
        decode_png("unused.png")


def test_unrelated_files_and_missing_empty_cnt_are_distinct(tmp_path):
    write_png(tmp_path, "absent/A/1-X.png", pixels=())
    write_png(tmp_path, "empty/A/1-X.png", pixels=())
    write_png(tmp_path, "empty/cno/2-C.png", pixels=())
    (tmp_path / "empty/readme.txt").write_text("unrelated")
    inv = discover_source(tmp_path)
    assert inv.valid
    assert not inv.models[0].cno_present and inv.models[0].cnt_assets == ()
    assert inv.models[1].cno_present and len(inv.models[1].cnt_assets) == 1
    assert inv.models[1].cnt_assets[0].mask.sum() == 0
    assert inv.models[1].warnings[0].path == "empty/readme.txt"


def test_true_16bit_rgb_rejected_before_pillow_downconversion(tmp_path):
    import struct
    import zlib
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
    # One true 16-bit RGB black pixel: Pillow exposes RGB but would lose bit depth.
    path = tmp_path / "rgb16.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" +
                     chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 16, 2, 0, 0, 0)) +
                     chunk(b"IDAT", zlib.compress(b"\0" * 7)) + chunk(b"IEND", b""))
    with pytest.raises(PNGSourceError, match="bit depth"):
        decode_png(path)


def test_truncated_loading_global_and_cycles_preserved(tmp_path, monkeypatch):
    from PIL import ImageFile
    monkeypatch.setattr(ImageFile, "LOAD_TRUNCATED_IMAGES", True)
    path = write_png(tmp_path, "M/A/1-X.png")
    path.write_bytes(path.read_bytes()[:45])
    (tmp_path / "M/A/cycle").symlink_to(tmp_path)
    inv = discover_source(tmp_path)
    assert not inv.valid
    assert ImageFile.LOAD_TRUNCATED_IMAGES is True
    assert any(e.path == "M/A/1-X.png" for e in inv.models[0].errors)
    assert any(e.path == "M/A/cycle" for e in inv.models[0].errors)


def test_emitted_identity_collision_fails_and_repeated_cnt_layer_is_preserved(tmp_path, monkeypatch):
    for relative in ["M/A/1-X.png", "M/cno/2-C.png", "M/cno/3-C.png"]:
        write_png(tmp_path, relative)
    inventory = discover_source(tmp_path)
    assert inventory.valid
    assert inventory.models[0].repeated_layers == ("C",)
    assert len(inventory.models[0].cnt_assets) == 2
    monkeypatch.setattr("psdl1.png_source.canonical_id", lambda kind, value: kind + "_collision")
    inventory = discover_source(tmp_path)
    assert not inventory.valid
    assert any("colliding" in e.reason for e in inventory.models[0].errors)
