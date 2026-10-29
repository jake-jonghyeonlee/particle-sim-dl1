"""merge_bundles: two exported bundles become one round bundle with intact asset hashes."""
import json
import os
from pathlib import Path
import shutil

import pytest

from png_helpers import write_image_only_fixture, write_stack_order, write_wide_import_fixture
from psdl1 import merge_bundles as mb
from psdl1.data import load_manifest
from psdl1.import_png import file_sha256, run_import
from psdl1.png_contract import ImportOptions

ORDER = ["U", "N", "C4", "U5", "C6", "L"]


def export(tmp_path, name, model, *, grid=128, image_only=False):
    root, out = tmp_path / f"src-{name}", tmp_path / f"bundle-{name}"
    (write_image_only_fixture if image_only else write_wide_import_fixture)(root, model=model)
    mapping = tmp_path / f"meta-{name}.json"
    mapping.write_text(json.dumps({"metadata_schema_version": "dl1_png_metadata_v1", "models": {
        model: {"lineage_id": f"FAMILY-{model}", "design_version": "SYNTHETIC-V1"}}}))
    order = None if image_only else write_stack_order(tmp_path / f"order-{name}.json", ORDER)
    options = ImportOptions(root=root, out=out, export=True, metadata=mapping, candidate_grid=grid, stack_order=order)
    assert run_import(options)[1] == 0
    return out


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("merge-sources")
    return {"a": export(tmp, "a", "WIDE-A"), "b": export(tmp, "b", "WIDE-B"), "c": export(tmp, "c", "WIDE-C"),
            "dup": export(tmp, "dup", "WIDE-A"), "grid": export(tmp, "grid", "WIDE-G", grid=64),
            "image": export(tmp, "image", "M", image_only=True)}


def report(path):
    return json.loads((Path(path) / "merge-report.json").read_text())


def merged(tmp_path, bundles, name="r02"):
    out = tmp_path / name
    mb.merge_bundles(bundles["a"], bundles["b"], out, round_index=2, base_round=1)
    return out


def test_merge_two_exports_verifies_and_keeps_asset_hashes(bundles, tmp_path):
    out = merged(tmp_path, bundles)
    assert mb.verify_merged_bundle(out)["product_count"] == 2
    assert mb.main(["--verify", str(out)]) == 0
    products = load_manifest(out / "manifest.json")
    assert [p.product_id for p in products] == ["WIDE-A", "WIDE-B"]
    info = report(out)
    assert info["product_count"] == 2 and info["round"] == 2 and info["asset_method"] in {"hardlink", "copy"}
    originals = {"base": bundles["a"], "add": bundles["b"]}
    for item in info["assets"]:
        assert file_sha256(out / item["path"]) == file_sha256(originals[item["source_bundle"]] / item["source_path"]) == item["sha256"]
        assert item["path"].startswith("assets/r01/" if item["source_bundle"] == "base" else "assets/r02/")
    manifest = json.loads((out / "manifest.json").read_text())
    paths = {p for product in manifest["products"] for p in mb._asset_paths(product)}
    assert paths == {item["path"] for item in info["assets"]}
    assert info["base"]["product_ids"] == ["WIDE-A"] and info["add"]["product_ids"] == ["WIDE-B"]
    for side, folder in (("base", bundles["a"]), ("add", bundles["b"])):
        assert info[side]["manifest_sha256"] == file_sha256(folder / "manifest.json")
        assert info[side]["integrity_sha256"] == file_sha256(folder / "bundle-integrity.json")


@pytest.mark.parametrize("case, match", [("dup", "duplicate product IDs"), ("grid", "candidate_grid differs"),
                                         ("image", "cnt_mode differs")])
def test_merge_refuses_duplicate_product_grid_and_cnt_mode_mismatch(bundles, tmp_path, case, match):
    out = tmp_path / "r02"
    with pytest.raises(ValueError, match=match):
        mb.merge_bundles(bundles["a"], bundles[case], out, round_index=2, base_round=1)
    assert not (out / "manifest.json").exists()


def test_merge_of_a_merged_base_keeps_old_paths_and_adds_the_new_round(bundles, tmp_path):
    second = merged(tmp_path, bundles)
    third = tmp_path / "r03"
    mb.merge_bundles(second, bundles["c"], third, round_index=3)
    assert mb.verify_merged_bundle(third)["product_count"] == 3
    old = {item["path"] for item in report(second)["assets"]}
    new = report(third)["assets"]
    kept = [item for item in new if item["source_bundle"] == "base"]
    assert {item["path"] for item in kept} == old and all(item["path"] == item["source_path"] for item in kept)
    assert all(item["path"].startswith("assets/r03/") for item in new if item["source_bundle"] == "add")
    with pytest.raises(ValueError, match="must be greater"):
        mb.merge_bundles(second, bundles["c"], tmp_path / "again", round_index=2)
    with pytest.raises(ValueError, match="--base-round"):
        mb.merge_bundles(bundles["a"], bundles["b"], tmp_path / "nobase", round_index=2)


def test_merge_falls_back_to_copy_when_link_fails(bundles, tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError(18, "cross-device link")
    monkeypatch.setattr(mb.os, "link", refuse)
    out = merged(tmp_path, bundles)
    assert report(out)["asset_method"] == "copy"
    assert {item["asset_method"] for item in report(out)["assets"]} == {"copy"}
    assert mb.verify_merged_bundle(out)["status"] == "verified"


def test_asset_method_is_recorded_per_file_and_mixed_summary(bundles, tmp_path, monkeypatch):
    real = os.link
    def link_only_base(source, destination, **kwargs):
        if "bundle-b" in str(source):
            raise OSError(18, "cross-device link")
        return real(source, destination, **kwargs)
    monkeypatch.setattr(mb.os, "link", link_only_base)
    info = report(merged(tmp_path, bundles))
    assert info["asset_method"] == "mixed"
    assert {item["source_bundle"]: item["asset_method"] for item in info["assets"]} == {"base": "hardlink", "add": "copy"}


def test_verify_rejects_tampered_asset_and_unlisted_file(bundles, tmp_path):
    out = merged(tmp_path, bundles)
    extra = out / "assets" / "r02" / "extra.npy"
    extra.write_bytes(b"x")
    with pytest.raises(ValueError, match="unlisted"):
        mb.verify_merged_bundle(out)
    extra.unlink()
    assert mb.verify_merged_bundle(out)["status"] == "verified"
    target = out / report(out)["assets"][0]["path"]
    os.unlink(target)
    target.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        mb.verify_merged_bundle(out)


def merged_from_copy(tmp_path, bundles):
    """Merge from a private copy of bundle A so the shared fixture stays intact."""
    base = tmp_path / "base-copy"
    shutil.copytree(bundles["a"], base)
    out = tmp_path / "r02"
    mb.merge_bundles(base, bundles["b"], out, round_index=2, base_round=1)
    assert report(out)["asset_method"] == "hardlink"
    return base, out


def test_deleting_an_original_asset_keeps_the_merged_bundle_valid(bundles, tmp_path):
    base, out = merged_from_copy(tmp_path, bundles)
    (base / report(out)["assets"][0]["source_path"]).unlink()
    assert mb.verify_merged_bundle(out)["status"] == "verified"


def test_changing_an_original_asset_in_place_fails_the_merged_verify(bundles, tmp_path):
    base, out = merged_from_copy(tmp_path, bundles)
    item = next(i for i in report(out)["assets"] if i["source_bundle"] == "base")
    original = base / item["source_path"]
    assert original.stat().st_ino == (out / item["path"]).stat().st_ino
    with original.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        last = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes([last[0] ^ 1]))
    with pytest.raises(ValueError, match="hash mismatch"):
        mb.verify_merged_bundle(out)


def test_merge_inputs_match_compares_hashes_and_round(bundles, tmp_path):
    out = merged(tmp_path, bundles)
    assert mb.merge_inputs_match(out, bundles["a"], bundles["b"], 2) is None
    assert "round" in mb.merge_inputs_match(out, bundles["a"], bundles["b"], 3)
    assert "base manifest_sha256" in mb.merge_inputs_match(out, bundles["c"], bundles["b"], 2)
    assert "add manifest_sha256" in mb.merge_inputs_match(out, bundles["a"], bundles["c"], 2)
    assert "incomplete" in mb.merge_inputs_match(tmp_path / "missing", bundles["a"], bundles["b"], 2)
    assert "cannot be compared" in mb.merge_inputs_match(out, tmp_path / "missing", bundles["b"], 2)


def test_merge_inputs_match_checks_base_round_and_rejects_failed_folders(bundles, tmp_path, monkeypatch):
    out = merged(tmp_path, bundles)
    assert mb.merge_inputs_match(out, bundles["a"], bundles["b"], 2, base_round=1) is None
    assert "base round" in mb.merge_inputs_match(out, bundles["a"], bundles["b"], 2, base_round=2)
    monkeypatch.setattr(mb, "_seal_bundle", lambda path: (_ for _ in ()).throw(RuntimeError("injected")))
    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError):
        mb.merge_bundles(bundles["a"], bundles["b"], failed, round_index=2, base_round=1)
    assert (failed / "merge-report.json").exists() and (failed / "manifest.failed.json").exists()
    assert "incomplete" in mb.merge_inputs_match(failed, bundles["a"], bundles["b"], 2, base_round=1)


def reseal(root, manifest=None, report=None, *, relink=True):
    """Rewrite manifest/report with a mutation, then seal again so only the content check can object."""
    root = Path(root)
    docs = {name: json.loads((root / name).read_text()) for name in ("manifest.json", "merge-report.json")}
    if manifest:
        manifest(docs["manifest.json"])
    (root / "manifest.json").write_text(json.dumps(docs["manifest.json"]))
    if relink:
        docs["merge-report.json"]["generated_manifest_sha256"] = file_sha256(root / "manifest.json")
    if report:
        report(docs["merge-report.json"])
    (root / "merge-report.json").write_text(json.dumps(docs["merge-report.json"]))
    mb._seal_bundle(root)


def swap_products(manifest):
    manifest["products"].reverse()


def point_at_unlisted_asset(manifest):
    manifest["products"][0]["valid"] = "assets/r02/not-there.npy"


def set_key(key, value):
    return lambda doc: doc.__setitem__(key, value)


def break_first_asset(key, value):
    return lambda doc: doc["assets"][0].__setitem__(key, value)


@pytest.mark.parametrize("manifest, report, relink, match", [
    (swap_products, None, True, "product list differs"),
    (None, set_key("generated_manifest_sha256", "0" * 64), True, "hash link mismatch"),
    (None, set_key("round", 1), True, "not under assets/rNN"),
    (point_at_unlisted_asset, None, True, "not listed in the bundle"),
    (None, set_key("candidate_grid", 999), True, "grid/schema"),
    (None, set_key("schema_version", "other"), True, "grid/schema"),
    (None, set_key("asset_method", "copy"), True, "asset_method differs"),
    (None, break_first_asset("source_bundle", "elsewhere"), True, "invalid merge report asset entry"),
    (None, break_first_asset("source_path", "../escape.npy"), True, "invalid merge report asset entry"),
    (None, break_first_asset("asset_method", "symlink"), True, "invalid merge report asset entry"),
])
def test_verify_rejects_resealed_tampering(bundles, tmp_path, manifest, report, relink, match):
    out = merged(tmp_path, bundles)
    reseal(out, manifest, report, relink=relink)
    with pytest.raises(ValueError, match=match):
        mb.verify_merged_bundle(out)


def test_reseal_helper_without_mutation_still_verifies(bundles, tmp_path):
    out = merged(tmp_path, bundles)
    reseal(out)
    assert mb.verify_merged_bundle(out)["status"] == "verified"


def test_cli_merge_and_verify_exit_codes(bundles, tmp_path, capsys):
    out = tmp_path / "cli"
    assert mb.main(["--base", str(bundles["a"]), "--add", str(bundles["b"]), "--out", str(out),
                    "--round", "2", "--base-round", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    assert mb.main(["--verify", str(out)]) == 0
    capsys.readouterr()
    assert mb.main(["--base", str(bundles["a"]), "--add", str(bundles["dup"]), "--out", str(tmp_path / "bad"),
                    "--round", "2", "--base-round", "1"]) == 2
    assert "duplicate product IDs" in capsys.readouterr().err
    assert mb.main(["--base", str(bundles["a"]), "--add", str(bundles["b"]), "--out", str(out),
                    "--round", "2", "--base-round", "1"]) == 2  # existing --out is refused
    with pytest.raises(SystemExit) as exc:
        mb.main(["--verify", str(out), "--round", "2"])
    assert exc.value.code == 2


def test_partial_output_is_left_without_manifest(bundles, tmp_path, monkeypatch):
    def fail(path):
        raise RuntimeError("injected failure")
    monkeypatch.setattr(mb, "_seal_bundle", fail)
    out = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="injected"):
        mb.merge_bundles(bundles["a"], bundles["b"], out, round_index=2, base_round=1)
    assert out.is_dir() and any((out / "assets").rglob("*.npy"))
    assert not (out / "manifest.json").exists() and not (out / "bundle-integrity.json").exists()
    assert (out / "manifest.failed.json").exists()
    monkeypatch.undo()
    monkeypatch.setattr(mb, "verify_merged_bundle", lambda path: (_ for _ in ()).throw(ValueError("injected verify")))
    second = tmp_path / "partial2"
    with pytest.raises(ValueError, match="injected verify"):
        mb.merge_bundles(bundles["a"], bundles["b"], second, round_index=2, base_round=1)
    assert not (second / "manifest.json").exists() and not (second / "bundle-integrity.json").exists()
