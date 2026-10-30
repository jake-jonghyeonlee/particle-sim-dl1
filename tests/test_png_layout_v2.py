"""layer-align/T7: Merged_Layer / CNO / Layer folder layout and legacy compatibility."""
from pathlib import Path

import pytest

from png_helpers import write_png
from psdl1.png_source import discover_source


def _v2(root, model="M"):
    write_png(root, f"{model}/Merged_Layer/A/1-X.png", pixels=((0, 0),))
    write_png(root, f"{model}/Merged_Layer/B/2-Y.png", pixels=((1, 1),))
    write_png(root, f"{model}/CNO/3-OCNT.png", pixels=((2, 2),))


def _legacy(root, model="M"):
    write_png(root, f"{model}/A/1-X.png", pixels=((0, 0),))
    write_png(root, f"{model}/B/2-Y.png", pixels=((1, 1),))
    write_png(root, f"{model}/cno/3-OCNT.png", pixels=((2, 2),))


def _reasons(issues):
    return [i.reason for i in issues]


def test_v2_reads_electrodes_and_cnt(tmp_path):
    _v2(tmp_path)
    result = discover_source(tmp_path)
    assert result.valid
    model = result.models[0]
    assert model.roles == ("A", "B")
    assert model.cno_present
    assert {a.relative_path for a in model.assets} == {
        "M/Merged_Layer/A/1-X.png", "M/Merged_Layer/B/2-Y.png", "M/CNO/3-OCNT.png"}
    assert sum("layout_structure: merged_layer_v2" in r for r in _reasons(model.warnings)) == 1


def test_v2_accepts_lowercase_cno(tmp_path):
    write_png(tmp_path, "M/Merged_Layer/A/1-X.png")
    write_png(tmp_path, "M/cno/3-OCNT.png")
    result = discover_source(tmp_path)
    assert result.valid and result.models[0].cno_present


def test_v2_layer_folder_is_screening_only(tmp_path):
    _v2(tmp_path)
    write_png(tmp_path, "M/Layer/1-X.png")
    write_png(tmp_path, "M/Layer/9-NOPE.png")
    result = discover_source(tmp_path)
    model = result.models[0]
    assert result.valid
    assert "NOPE" not in model.layers and "NOPE" not in model.roles
    assert all("/Layer/" not in a.relative_path for a in model.assets)
    assert [(f.name, f.filename_number, f.relative_path) for f in model.layer_files] == [
        ("X", 1, "M/Layer/1-X.png"), ("NOPE", 9, "M/Layer/9-NOPE.png")]
    assert sum("layer_folder_screening_only: 2 files" in r for r in _reasons(model.warnings)) == 1
    assert not any("layer_folder_ignored" in r for r in _reasons(model.warnings))


def test_v2_mixed_png_folder_is_error(tmp_path):
    _v2(tmp_path)
    write_png(tmp_path, "M/Extra/1-X.png")
    result = discover_source(tmp_path)
    assert not result.valid
    assert any("layout_structure_mixed" in r for r in _reasons(result.models[0].errors))


def test_v2_two_merged_layer_variants_is_error(tmp_path, monkeypatch):
    write_png(tmp_path, "M/Merged_Layer/A/1-X.png")
    real = Path.iterdir
    model = tmp_path / "M"

    def fake(self):  # case-insensitive filesystems cannot hold both spellings
        items = list(real(self))
        if self == model.resolve() or self == model:
            items.append(model / "merged_layer")
        return iter(items)

    monkeypatch.setattr(Path, "iterdir", fake)
    result = discover_source(tmp_path)
    assert not result.valid
    assert any("layout_structure_mixed" in r for r in _reasons(result.models[0].errors))


def test_v2_cno_and_CNO_together_is_error(tmp_path, monkeypatch):
    _v2(tmp_path)
    real = Path.iterdir
    model = tmp_path / "M"

    def fake(self):
        items = list(real(self))
        if self == model.resolve() or self == model:
            items.append(model / "cno")
        return iter(items)

    monkeypatch.setattr(Path, "iterdir", fake)
    write_png(tmp_path, "M/cno/3-OCNT.png")  # on case-insensitive FS this is the CNO folder
    result = discover_source(tmp_path)
    assert not result.valid
    assert any("cnt_folder_duplicate" in r for r in _reasons(result.models[0].errors))


def test_legacy_unchanged_uppercase_cno_still_error(tmp_path):
    write_png(tmp_path, "M/A/1-X.png")
    write_png(tmp_path, "M/CNO/3-C.png")
    result = discover_source(tmp_path)
    assert not result.valid
    assert any("reserved" in r for r in _reasons(result.models[0].errors))
    assert not any("layout_structure" in r for r in _reasons(result.models[0].warnings))


def test_legacy_layer_folder_is_electrode(tmp_path):
    write_png(tmp_path, "M/Layer/1-X.png")
    write_png(tmp_path, "M/A/2-Y.png")
    result = discover_source(tmp_path)
    assert result.valid
    assert result.models[0].roles == ("A", "Layer")
    assert not any("layout_structure" in r or "layer_folder_ignored" in r
                   for r in _reasons(result.models[0].warnings))


def test_legacy_and_v2_same_content_same_ids_and_masks(tmp_path):
    _legacy(tmp_path / "old")
    _v2(tmp_path / "new")
    old = discover_source(tmp_path / "old").models[0]
    new = discover_source(tmp_path / "new").models[0]
    assert old.layers == new.layers and old.roles == new.roles
    assert [a.asset_id for a in old.assets] == [a.asset_id for a in new.assets]
    assert [a.mask_sha256 for a in old.assets] == [a.mask_sha256 for a in new.assets]
    assert [a.relative_path for a in old.assets] != [a.relative_path for a in new.assets]
