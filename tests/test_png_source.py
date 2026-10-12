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


@pytest.mark.parametrize("rows", [63, 32, 1, 0])
def test_ra_b1_short_complete_zlib_greyscale_1bit(tmp_path, rows):
    from png_helpers import raw_png
    # 64x64 depth=1 needs exactly 64*(1 filter + 8 packed pixels) = 576 bytes.
    path = tmp_path / f"short-{rows}.png"
    path.write_bytes(raw_png(64, 64, 1, 0, (b"\0" + b"\xff" * 8) * rows))
    with pytest.raises(PNGSourceError, match="scanline|truncated") as error:
        decode_png(path)
    assert error.value.issue.reason.startswith("IDAT scanline byte count is short:")
    assert str(path) in str(error.value)


def test_ra_b1_short_complete_zlib_greyscale_8bit(tmp_path):
    from png_helpers import raw_png
    path = tmp_path / "short-L.png"
    path.write_bytes(raw_png(4, 4, 8, 0, b"\0\xff\xff\0\0" * 3))
    with pytest.raises(PNGSourceError, match="scanline") as error:
        decode_png(path)
    assert error.value.issue.reason.startswith("IDAT scanline byte count is short:")
    assert str(path) in str(error.value)


@pytest.mark.parametrize("depth", [1, 2, 4, 8])
@pytest.mark.parametrize("transparent_sample", ["white", "black"])
def test_ra_b2_raw_greyscale_trns_rejected(tmp_path, depth, transparent_sample):
    import struct
    from png_helpers import raw_png
    # Samples [0,max,max,0], packed independently at each declared bit depth.
    packed = {1: b"\x60", 2: b"\x3c", 4: b"\x0f\xf0", 8: b"\x00\xff\xff\x00"}[depth]
    sample = (1 << depth) - 1 if transparent_sample == "white" else 0
    path = tmp_path / f"trns-{depth}-{transparent_sample}.png"
    path.write_bytes(raw_png(4, 1, depth, 0, b"\0" + packed,
                            before_idat=((b"tRNS", struct.pack(">H", sample)),)))
    with pytest.raises(PNGSourceError, match="tRNS") as error:
        decode_png(path)
    assert error.value.issue.reason == "non-palette tRNS is nonopaque/unsupported"
    assert str(path) in str(error.value)


def test_ra_b2_nonpalette_rgb_trns_rejected_even_if_unused(tmp_path):
    import struct
    from png_helpers import raw_png
    path = tmp_path / "rgb-trns.png"
    path.write_bytes(raw_png(1, 1, 8, 2, b"\0\xff\xff\xff",
                            before_idat=((b"tRNS", struct.pack(">HHH", 1, 2, 3)),)))
    with pytest.raises(PNGSourceError, match="tRNS"):
        decode_png(path)


@pytest.mark.parametrize("depth", [2, 4, 8])
def test_ra_b3_palette_out_of_range_index(tmp_path, depth):
    from png_helpers import raw_png
    # PLTE[0]=white, PLTE[1]=black; raw indices [1,0,3,1]. Index 3 is undefined.
    packed = {2: b"\x4d", 4: b"\x10\x31", 8: b"\x01\x00\x03\x01"}[depth]
    path = tmp_path / f"palette-{depth}.png"
    path.write_bytes(raw_png(4, 1, depth, 3, b"\0" + packed,
                            before_idat=((b"PLTE", b"\xff\xff\xff\0\0\0"),)))
    with pytest.raises(PNGSourceError, match="PLTE entry") as error:
        decode_png(path)
    assert error.value.issue.reason == "palette pixel index has no PLTE entry"
    assert str(path) in str(error.value)


def test_ra_b3_missing_palette(tmp_path):
    from png_helpers import raw_png
    path = tmp_path / "undefined.png"
    path.write_bytes(raw_png(4, 1, 8, 3, b"\0\1\0\0\1"))
    with pytest.raises(PNGSourceError, match="PLTE") as error:
        decode_png(path)
    assert error.value.issue.reason == "IDAT order invalid or palette PNG missing PLTE"
    assert str(path) in str(error.value)


@pytest.mark.parametrize("case", ["after_iend", "concatenated_png", "missing_crc_iend", "wrong_idat_crc",
                                  "interlace2", "compression1", "filter_method1", "overlong_stream"])
def test_ra_n5_lenient_structures_now_rejected(tmp_path, case):
    from png_helpers import raw_png, png_chunk
    import zlib
    valid = raw_png(4, 1, 8, 0, b"\0\0\xff\xff\0")
    if case == "after_iend": data = valid + b"trailing"
    elif case == "concatenated_png": data = valid + valid
    elif case == "missing_crc_iend": data = valid[:-16]
    elif case == "wrong_idat_crc":
        data = valid[:33] + png_chunk(b"IDAT", zlib.compress(b"\0\0\xff\xff\0"), corrupt_crc=True) + valid[-12:]
    elif case == "interlace2": data = raw_png(4, 1, 8, 0, b"\0\0\xff\xff\0", interlace=2)
    elif case == "compression1": data = raw_png(4, 1, 8, 0, b"\0\0\xff\xff\0", compression=1)
    elif case == "filter_method1": data = raw_png(4, 1, 8, 0, b"\0\0\xff\xff\0", filtering=1)
    else: data = raw_png(4, 1, 8, 0, b"\0\0\xff\xff\0EXTRA")
    path = tmp_path / f"{case}.png"; path.write_bytes(data)
    with pytest.raises(PNGSourceError) as error:
        decode_png(path)
    reasons = {
        "after_iend": "trailing bytes after IEND", "concatenated_png": "trailing bytes after IEND",
        "missing_crc_iend": "PNG IDAT CRC mismatch/truncation", "wrong_idat_crc": "PNG IDAT CRC mismatch/truncation",
        "interlace2": "unsupported IHDR compression/filter/interlace method",
        "compression1": "unsupported IHDR compression/filter/interlace method",
        "filter_method1": "unsupported IHDR compression/filter/interlace method",
        "overlong_stream": "IDAT scanline byte count is longer than IHDR requires"}
    assert error.value.issue.reason == reasons[case]
    assert str(path) in str(error.value)


@pytest.mark.parametrize("case", ["ihdr_not_first", "duplicate_ihdr", "missing_iend", "unknown_critical",
                                  "plte_after_idat", "noncontiguous_idat", "duplicate_plte", "iend_payload",
                                  "bad_filter_byte", "invalid_colour_depth", "invalid_plte_length",
                                  "extra_zlib_stream", "incomplete_zlib", "bad_ancillary_crc"])
def test_png_critical_order_and_stream_integrity(tmp_path, case):
    import struct
    import zlib
    from png_helpers import raw_png, png_chunk
    raw = b"\0\0\xff\xff\0"
    valid = raw_png(4, 1, 8, 0, raw)
    if case == "ihdr_not_first": data = valid[:8] + png_chunk(b"tEXt", b"x\0y") + valid[8:]
    elif case == "duplicate_ihdr": data = valid[:33] + valid[8:33] + valid[33:]
    elif case == "missing_iend": data = valid[:-12]
    elif case == "unknown_critical": data = valid[:33] + png_chunk(b"ABCD", b"") + valid[33:]
    elif case == "plte_after_idat": data = raw_png(4, 1, 8, 2, b"\0" + b"\xff" * 12, after_idat=((b"PLTE", b"\0\0\0"),))
    elif case == "noncontiguous_idat":
        stream = zlib.compress(raw)
        data = valid[:33] + png_chunk(b"IDAT", stream[:3]) + png_chunk(b"tEXt", b"x\0y") + png_chunk(b"IDAT", stream[3:]) + valid[-12:]
    elif case == "duplicate_plte": data = raw_png(1, 1, 8, 3, b"\0\0", before_idat=((b"PLTE", b"\0\0\0"),) * 2)
    elif case == "iend_payload": data = valid[:-12] + png_chunk(b"IEND", b"x")
    elif case == "bad_filter_byte": data = raw_png(4, 1, 8, 0, b"\5\0\xff\xff\0")
    elif case == "invalid_colour_depth": data = raw_png(4, 1, 4, 2, raw)
    elif case == "invalid_plte_length": data = raw_png(1, 1, 8, 3, b"\0\0", before_idat=((b"PLTE", b"\0\0"),))
    elif case == "extra_zlib_stream": data = raw_png(4, 1, 8, 0, raw, idat_parts=(zlib.compress(raw) + zlib.compress(raw),))
    elif case == "incomplete_zlib": data = raw_png(4, 1, 8, 0, raw, idat_parts=(zlib.compress(raw)[:-1],))
    else: data = valid[:33] + png_chunk(b"tEXt", b"x\0y", corrupt_crc=True) + valid[33:]
    path = tmp_path / f"{case}.png"; path.write_bytes(data)
    with pytest.raises(PNGSourceError) as error: decode_png(path)
    reasons = {
        "ihdr_not_first": "IHDR must be the first PNG chunk", "duplicate_ihdr": "PNG requires exactly one 13-byte IHDR first",
        "missing_iend": "PNG missing/truncated chunk or IEND", "unknown_critical": "unknown critical PNG chunk",
        "plte_after_idat": "invalid PLTE length/order/colour type", "noncontiguous_idat": "IDAT order invalid or palette PNG missing PLTE",
        "duplicate_plte": "invalid PLTE length/order/colour type", "iend_payload": "invalid IEND length/order or missing IDAT",
        "bad_filter_byte": "invalid PNG scanline filter type", "invalid_colour_depth": "invalid/unsupported IHDR dimensions/bit depth/colour type",
        "invalid_plte_length": "invalid PLTE length/order/colour type", "extra_zlib_stream": "extra compressed bytes after IDAT zlib stream",
        "incomplete_zlib": "truncated/incomplete IDAT zlib stream", "bad_ancillary_crc": "PNG tEXt CRC mismatch/truncation"}
    assert error.value.issue.reason == reasons[case]
    assert str(path) in str(error.value)


@pytest.mark.parametrize("depth", [1, 2, 4, 8])
def test_valid_raw_greyscale_depths_have_unchanged_polarity(tmp_path, depth):
    from png_helpers import raw_png
    packed = {1: b"\x60", 2: b"\x3c", 4: b"\x0f\xf0", 8: b"\x00\xff\xff\x00"}[depth]
    path = tmp_path / f"valid-{depth}.png"
    path.write_bytes(raw_png(4, 1, depth, 0, b"\0" + packed))
    mask, _, _ = decode_png(path)
    assert mask.tolist() == [[True, False, False, True]]


@pytest.mark.parametrize("colour", [0, 3])
@pytest.mark.parametrize("stream", ["exact", "short", "long"])
def test_adam7_per_pass_scanline_counts(tmp_path, colour, stream):
    from png_helpers import raw_png
    white = 255 if colour == 0 else 1
    # Independently drawn 3x3 checkerboard, Adam7 nonempty passes 1,4,5,6,7.
    # Each row begins with filter 0; expected pass byte counts are 2,2,3,4,4.
    raw = bytes([0, 0, 0, 0, 0, 0, 0, 0, white, 0, white, 0, white, 0, white])
    assert len(raw) == 15
    if stream == "short": raw = raw[:-1]
    elif stream == "long": raw += b"x"
    palette = ((b"PLTE", b"\0\0\0\xff\xff\xff"),) if colour == 3 else ()
    path = tmp_path / f"adam7-{colour}-{stream}.png"
    path.write_bytes(raw_png(3, 3, 8, colour, raw, interlace=1, before_idat=palette))
    if stream == "exact":
        mask, _, _ = decode_png(path)
        assert mask.tolist() == [[True, False, True], [False, True, False], [True, False, True]]
    else:
        with pytest.raises(PNGSourceError, match="scanline"): decode_png(path)


@pytest.mark.parametrize("filter_type,first,second", [
    (0, b"\0\1\0\1", b"\1\0\1\0"),
    (1, b"\0\1\xff\1", b"\1\xff\1\xff"),
    (2, b"\0\1\0\1", b"\1\xff\1\xff"),
    (3, b"\0\1\0\1", b"\1\xff\1\xff"),
    (4, b"\0\1\xff\1", b"\1\xff\1\xff"),
])
def test_palette_indices_checked_after_each_png_unfilter(tmp_path, filter_type, first, second):
    from png_helpers import raw_png
    path = tmp_path / f"filter-{filter_type}.png"
    path.write_bytes(raw_png(4, 2, 8, 3, bytes([filter_type]) + first + bytes([filter_type]) + second,
                            before_idat=((b"PLTE", b"\0\0\0\xff\xff\xff"),)))
    mask, _, _ = decode_png(path)
    assert mask.tolist() == [[True, False, True, False], [False, True, False, True]]


@pytest.mark.parametrize("depth,packed", [(1, b"\x90"), (2, b"\x41"), (4, b"\x10\x01"), (8, b"\1\0\0\1")])
def test_palette_exact_indices_and_fully_opaque_trns(tmp_path, depth, packed):
    from png_helpers import raw_png
    path = tmp_path / f"palette-valid-{depth}.png"
    path.write_bytes(raw_png(4, 1, depth, 3, b"\0" + packed,
                            before_idat=((b"PLTE", b"\xff\xff\xff\0\0\0"), (b"tRNS", b"\xff\xff"))))
    mask, _, _ = decode_png(path)
    assert mask.tolist() == [[True, False, False, True]]


def test_unused_transparent_palette_entry_rejected(tmp_path):
    from png_helpers import raw_png
    path = tmp_path / "unused-alpha.png"
    path.write_bytes(raw_png(1, 1, 8, 3, b"\0\0", before_idat=(
        (b"PLTE", b"\xff\xff\xff\0\0\0"), (b"tRNS", b"\xff\0"))))
    with pytest.raises(PNGSourceError, match="tRNS"): decode_png(path)


def test_valid_stream_split_across_many_idat_chunks(tmp_path):
    import zlib
    from png_helpers import raw_png
    raw = b"\0\0\xff\xff\0"
    stream = zlib.compress(raw)
    path = tmp_path / "split-idat.png"
    path.write_bytes(raw_png(4, 1, 8, 0, raw, idat_parts=(b"",) + tuple(bytes([b]) for b in stream) + (b"",)))
    assert decode_png(path)[0].tolist() == [[True, False, False, True]]


def test_inventory_empty_electrode_and_nonmodel_directories(tmp_path):
    root = tmp_path / "source"
    write_png(root, "M/A/1-X.png")
    (root / "M/B").mkdir()
    result = discover_source(root)
    assert result.valid
    assert any(e.path == "M/B" and "no PNG" in e.reason for e in result.models[0].warnings)
    (root / "__MACOSX").mkdir()
    result = discover_source(root)
    assert not result.valid
    bad = next(m for m in result.models if m.model_id == "__MACOSX")
    assert any("no decoded electrode" in e.reason for e in bad.errors)


@pytest.mark.parametrize("reserved", ["CNO", "Cno", "cNo"])
def test_inventory_case_variant_of_reserved_cno_is_error(tmp_path, reserved):
    write_png(tmp_path, "M/A/1-X.png")
    write_png(tmp_path, f"M/{reserved}/6_C.png")
    result = discover_source(tmp_path)
    assert not result.valid
    assert any(e.path == f"M/{reserved}" and "reserved" in e.reason and "'cno'" in e.reason
               for e in result.models[0].errors)
    assert result.models[0].roles == ("A",)


def test_inventory_non_nfc_names_warn_without_normalising(tmp_path):
    import unicodedata
    nfc, nfd = "é", "e\u0301"
    assert unicodedata.normalize("NFC", nfd) == nfc
    write_png(tmp_path, f"M/{nfd}/1-{nfd}.png")
    write_png(tmp_path, f"Q/{nfc}/1-{nfc}.png")
    result = discover_source(tmp_path)
    assert result.valid
    assert result.models[0].roles == result.models[0].layers == (nfd,)
    assert result.models[1].roles == result.models[1].layers == (nfc,)
    assert result.role_union == tuple(sorted([nfc, nfd]))
    assert any("not NFC" in e.reason for e in result.models[0].warnings)
    assert result.models[0].assets[0].asset_id != result.models[1].assets[0].asset_id
    assert canonical_id("element", ["M", nfc]) != canonical_id("element", ["M", nfd])


@pytest.mark.parametrize("case", ["8x4", "8x7_offset", "1x1_offset", "full_single_frame", "4x4_offset",
                                  "zero_frames", "two_actl", "actl_after_idat", "fctl_only", "fdat_only"])
def test_ra2_apng_rejected_with_validator_reason_before_pillow(tmp_path, monkeypatch, case):
    import struct
    import zlib
    from png_helpers import raw_png
    from psdl1.png_source import _validate_png_structure
    size = {"8x4": (8, 4, 0, 0), "8x7_offset": (8, 7, 0, 1), "1x1_offset": (1, 1, 7, 7),
            "4x4_offset": (4, 4, 2, 2)}.get(case, (8, 8, 0, 0))
    control = (b"fcTL", struct.pack(">IIIIIHHBB", 0, *size, 1, 100, 0, 0))
    actl = (b"acTL", struct.pack(">II", 0 if case == "zero_frames" else 1, 0))
    before = (actl, control)
    after = ()
    if case == "zero_frames": after = ((b"fdAT", struct.pack(">I", 1) + zlib.compress(b"\0\xff")),)
    elif case == "two_actl": before = (actl, actl, control)
    elif case == "actl_after_idat": before, after = (), (actl,)
    elif case == "fctl_only": before = (control,)
    elif case == "fdat_only": before, after = (), ((b"fdAT", b"\0\0\0\1"),)
    path = tmp_path / f"{case}.png"
    path.write_bytes(raw_png(8, 8, 8, 0, (b"\0" + b"\xff" * 8) * 8,
                            before_idat=before, after_idat=after))
    with pytest.raises(ValueError) as direct:
        _validate_png_structure(path)
    assert str(direct.value) == "APNG is unsupported"
    def forbidden_pillow_open(*args, **kwargs):
        pytest.fail("Pillow was called before structural rejection")
    monkeypatch.setattr(Image, "open", forbidden_pillow_open)
    with pytest.raises(PNGSourceError) as error:
        decode_png(path)
    assert error.value.issue.reason == "APNG is unsupported"


@pytest.mark.parametrize("colour,depth", [(0, 1), (0, 2), (0, 4), (0, 8), (2, 8),
                                         (3, 1), (3, 2), (3, 4), (3, 8), (4, 8), (6, 8)])
@pytest.mark.parametrize("shape", [(9, 9), (11, 13)])
def test_adam7_every_pass_nonempty_for_supported_types(tmp_path, colour, depth, shape):
    from png_helpers import encode_adam7_binary, raw_png
    # Independently specified repeating geometry; every Adam7 pass is nonempty.
    pattern = np.array([[(x + 2 * y) % 3 == 0 for x in range(shape[1])] for y in range(shape[0])])
    raw = encode_adam7_binary(pattern, depth, colour)
    palette = ((b"PLTE", b"\0\0\0\xff\xff\xff"),) if colour == 3 else ()
    path = tmp_path / f"adam7-{colour}-{depth}-{shape}.png"
    path.write_bytes(raw_png(shape[1], shape[0], depth, colour, raw, interlace=1, before_idat=palette))
    assert decode_png(path)[0].tolist() == pattern.tolist()


@pytest.mark.parametrize("depth,packed,expected,palette", [
    (1, b"\x1f", [True, True, True], b"\0\0\0"),  # set padding bits are not pixels
    (2, b"\x47", [True, False, True], b"\xff\xff\xff\0\0\0"),
    (4, b"\x10\x1f", [True, False, True], b"\xff\xff\xff\0\0\0"),
])
def test_packed_index_shift_order_and_padding_are_not_symmetric(tmp_path, depth, packed, expected, palette):
    from png_helpers import raw_png
    path = tmp_path / f"asymmetric-packed-{depth}.png"
    path.write_bytes(raw_png(3, 1, depth, 3, b"\0" + packed, before_idat=((b"PLTE", palette),)))
    assert decode_png(path)[0].tolist() == [expected]


def test_paeth_tie_prefers_left_over_upper_left(tmp_path):
    from png_helpers import raw_png
    # Second row, column 1: left=3, above=0, upper-left=1. Paeth p=2,
    # distances (1,2,1), so LEFT 3 wins the tie; upper-left 1 is white.
    path = tmp_path / "paeth-tie.png"
    palette = b"\xff\xff\xff" * 3 + b"\0\0\0"
    path.write_bytes(raw_png(2, 2, 8, 3, b"\0\1\0\4\2\0", before_idat=((b"PLTE", palette),)))
    assert decode_png(path)[0].tolist() == [[False, False], [True, True]]
    bad = tmp_path / "paeth-tie-undefined.png"
    # Correct left predictor 3 plus residual 1 gives undefined index 4;
    # wrong upper-left predictor 1 gives valid index 2 and hides the defect.
    bad.write_bytes(raw_png(2, 2, 8, 3, b"\0\1\0\4\2\1", before_idat=((b"PLTE", palette),)))
    with pytest.raises(PNGSourceError) as error:
        decode_png(bad)
    assert error.value.issue.reason == "palette pixel index has no PLTE entry"


@pytest.mark.parametrize("mode,bits", [("1", None), ("L", None), ("RGB", None), ("LA", None), ("RGBA", None),
                                      ("P", 1), ("P", 2), ("P", 4), ("P", 8)])
@pytest.mark.parametrize("compression", [0, 9])
def test_pillow_plain_modes_with_ancillary_chunks_still_accepted(tmp_path, mode, bits, compression):
    from PIL.PngImagePlugin import PngInfo
    info = PngInfo()
    info.add_text("purpose", "opaque synthetic fixture")
    info.add_text("compressed", "black-white", zip=True)
    info.add_itxt("unicode", "검증 fixture")
    exif = Image.Exif(); exif[274] = 1
    options = {"pnginfo": info, "exif": exif, "dpi": (72, 72)}
    if bits is not None: options["bits"] = bits
    path = write_png(tmp_path, f"plain-{mode}-{bits}-{compression}.png", shape=(19, 17),
                     mode=mode, pixels=((1, 2), (18, 16)), compress_level=compression, **options)
    mask, _, _ = decode_png(path)
    assert np.argwhere(mask).tolist() == [[1, 2], [18, 16]]


def test_pillow_large_multiple_idat_and_ancillary_chunks_accepted(tmp_path):
    import struct
    from PIL.PngImagePlugin import PngInfo
    info = PngInfo(); info.add_text("kind", "plain")
    path = write_png(tmp_path, "multiple.png", shape=(1500, 1700), mode="P", bits=1,
                     pixels=((0, 0), (1499, 1699)), compress_level=0, pnginfo=info)
    data = path.read_bytes(); offset = 8; idats = 0
    while offset < len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        idats += data[offset + 4:offset + 8] == b"IDAT"
        offset += length + 12
    assert idats > 1
    assert np.argwhere(decode_png(path)[0]).tolist() == [[0, 0], [1499, 1699]]


def test_source_replacement_after_validation_cannot_change_snapshot(monkeypatch, tmp_path):
    import hashlib
    from png_helpers import raw_png
    import psdl1.png_source as module
    path = write_png(tmp_path, "M/A/1-X.png", pixels=((0, 0),))
    original = path.read_bytes()
    validate = module._validate_png_structure
    def replace_after_validation(snapshot, **kwargs):
        validate(snapshot, **kwargs)
        # The new on-disk PNG is valid and has different geometry; it must not
        # influence the already captured/validated decoder snapshot or byte hash.
        path.write_bytes(raw_png(1203, 1001, 8, 0, (b"\0" + b"\xff" * 1203) * 1001))
    monkeypatch.setattr(module, "_validate_png_structure", replace_after_validation)
    model, = module.discover_source(tmp_path).models
    asset, = model.assets
    assert asset.mask.sum() == 1 and asset.mask[0, 0]
    assert asset.file_sha256 == hashlib.sha256(original).hexdigest()
    assert asset.file_sha256 != hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("depth,entries,packed", [(1, 1, b"\x80"), (2, 2, b"\x80"), (4, 2, b"\x20")])
def test_palette_index_equal_entry_count_rejected(tmp_path, depth, entries, packed):
    from png_helpers import raw_png
    path = tmp_path / "undefined.png"
    path.write_bytes(raw_png(1, 1, depth, 3, b"\0" + packed,
                            before_idat=((b"PLTE", b"\0\0\0" * entries),)))
    with pytest.raises(PNGSourceError) as error:
        decode_png(path)
    assert error.value.issue.reason == "palette pixel index has no PLTE entry"


def test_pixel_guard_precedes_inflate_and_pillow_open(monkeypatch, tmp_path):
    from png_helpers import raw_png
    path = tmp_path / "huge.png"
    path.write_bytes(raw_png(100000, 100000, 8, 0, b"", idat_parts=(b"not-zlib",)))
    def forbidden(*args, **kwargs):
        pytest.fail("over-limit input reached Pillow pixel decoding")
    monkeypatch.setattr(Image, "open", forbidden)
    with pytest.raises(PNGSourceError) as error:
        decode_png(path, png_max_image_pixels=100)
    assert error.value.issue.exception_class == "DecompressionBombError"
    assert error.value.issue.pixels == 10_000_000_000


def test_rb_report_separates_model_and_cnt_vocabulary_and_versions(tmp_path):
    for relative in ["M/A/1-X.png", "M/B/2-Y.png", "M/cno/3-ONLY-M-CNT.png",
                     "Q/A/1-X.png", "Q/D/2-Z.png", "Q/cno/3-ONLY-Q-CNT.png"]:
        write_png(tmp_path, relative)
    inv = discover_source(tmp_path)
    payload = report_payload(ImportReport(inv, load_metadata(None, ["M", "Q"])))
    inventory = payload["inventory"]
    assert inventory["models"][0]["electrode_layers"] == ["X", "Y"]
    assert inventory["models"][0]["cno_layers"] == ["ONLY-M-CNT"]
    assert inventory["models"][1]["electrode_layers"] == ["X", "Z"]
    assert inventory["models"][1]["cno_layers"] == ["ONLY-Q-CNT"]
    assert inventory["electrode_layer_union"] == ["X", "Y", "Z"]
    assert inventory["unique_electrode_layers"] == ["Y", "Z"]
    assert inventory["cno_layer_union"] == ["ONLY-M-CNT", "ONLY-Q-CNT"]
    assert inventory["unique_cno_layers"] == ["ONLY-M-CNT", "ONLY-Q-CNT"]
    assert inventory["all_source_layer_union"] == ["ONLY-M-CNT", "ONLY-Q-CNT", "X", "Y", "Z"]
    assert "layer_union" not in inventory and "unique_layers" not in inventory
    assert payload["report_schema_version"] == "dl1_png_import_v2"
    assert payload["extractor_run_status"] == "not_run" and "extraction_status" not in payload
    require_report_version(payload)
    for version in ["dl1_png_import_v1", "unsupported", None]:
        with pytest.raises(ValueError, match="unsupported"):
            require_report_version({**payload, "report_schema_version": version})
