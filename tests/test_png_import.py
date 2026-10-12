"""Real CLI export/inspect/verify counterexamples S12-S15, S24-S26."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from png_helpers import write_import_fixture, write_png
from psdl1.data import load_manifest, input_identity, frame_transform
import psdl1.import_png as importer
from psdl1.png_contract import ImportOptions


def cli(*args):
    return subprocess.run([sys.executable, "-m", "psdl1.import_png", *map(str, args)],
                          env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)


def metadata(path, model="SYNTHETIC-M"):
    path.write_text(json.dumps({"metadata_schema_version": "dl1_png_metadata_v1", "models": {
        model: {"lineage_id": "SYNTHETIC-TEST-FAMILY", "design_version": "SYNTHETIC-TEST-V1"}}}))
    return path


def report(out):
    return json.loads((out / "import-report.json").read_text())


def export(tmp_path):
    root, out = tmp_path / "source", tmp_path / "out"
    write_import_fixture(root)
    mapping = metadata(tmp_path / "metadata.json")
    run = cli("--root", root, "--out", out, "--export", "--metadata", mapping, "--candidate-grid", 16)
    assert run.returncode == 0, run.stderr
    return root, out, run


def test_auto_manifest_export_and_pending_prerequisites(tmp_path):
    root, out, run = export(tmp_path)
    doc = report(out)
    assert doc["export_status"] == "exported" and doc["metadata"]["pending_models"] == []
    p, = load_manifest(out / "manifest.json")
    assert p.lineage_id == "SYNTHETIC-TEST-FAMILY" and p.design_version == "SYNTHETIC-TEST-V1"
    assert p.lineage_id != p.product_id and p.design_version != p.product_id
    assert p.frame.shape_yx == (1001, 1203) and p.frame.pixel_pitch_um == 0.1
    assert p.unit_size_um == pytest.approx([120.3, 100.1])
    assert len(p.elements) == 5 and len(p.connections) == 3 and p.connectivity_observation == "partial"
    assert p.cnt[801, 801] == 1  # unresolved CNT survives global channel
    np.testing.assert_allclose(sorted(e.xy_um for c in p.connections for e in c.endpoints),
                               sorted([(2.15, 2.15)] * 2 + [(60.15, 60.15)] * 2 + [(70.15, 70.15)] * 2))
    arrays = list((out / "assets").glob("*.npy"))
    assert len(arrays) == 10  # E5 + valid1 + globalCNT1 + R3
    assert all(np.load(path, allow_pickle=False).dtype == np.bool_ for path in arrays)
    assert sum(path.stat().st_size for path in arrays) == 12_043_310  # 10*(1,204,203+128)
    assert doc["inventory"]["models"][0]["resource_estimate"]["projected_bool_npy_export_bytes"] == 12_043_310
    assert [e["event"] for e in map(json.loads, run.stdout.splitlines())] == ["resource_inventory", "completed"]
    validate_report = tmp_path / "validator.json"
    checked = subprocess.run([sys.executable, "-m", "psdl1.data", str(out / "manifest.json"), "--report", str(validate_report)],
                             env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
    assert checked.returncode == 0 and json.loads(validate_report.read_text())["products"][0]["product_id"] == "SYNTHETIC-M"
    for args in [[], ["--candidate-grid", "16"], ["--metadata", str(tmp_path / "metadata.json")]]:
        pending_out = tmp_path / ("pending-" + str(len(args)) + ("grid" if "--candidate-grid" in args else "metadata"))
        blocked = cli("--root", root, "--out", pending_out, "--export", *args)
        assert blocked.returncode == 2
        assert report(pending_out)["export_status"] == "blocked"
        assert (pending_out / "preview.html").exists() and not (pending_out / "manifest.json").exists()
        assert not (pending_out / "assets").exists()


@pytest.mark.parametrize("mutation", ["asset", "manifest", "report_version", "preview"])
def test_provenance_hashes_and_strict_bundle_verification(tmp_path, mutation):
    root, out, _ = export(tmp_path)
    original = report(out)
    assert original["electrical_verification"] == original["real_data_validation"] == "not_performed"
    assert original["product_input_identities"][0][1] == input_identity(load_manifest(out / "manifest.json")[0])
    copied = tmp_path / "copied"; shutil.copytree(out, copied); shutil.rmtree(root)
    assert cli("--verify-bundle", copied).returncode == 0
    if mutation == "asset":
        path = next((copied / "assets").glob("*.npy")); data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    elif mutation == "manifest": (copied / "manifest.json").write_text("{}")
    elif mutation == "preview": (copied / "preview.html").write_text("tampered")
    else:
        doc = report(copied); doc["report_schema_version"] = "unsupported"; (copied / "import-report.json").write_text(json.dumps(doc))
        # Updating the integrity hash must still fail the explicit version check.
        seal = json.loads((copied / "bundle-integrity.json").read_text())
        seal["files"]["import-report.json"] = hashlib.sha256((copied / "import-report.json").read_bytes()).hexdigest()
        (copied / "bundle-integrity.json").write_text(json.dumps(seal))
    assert cli("--verify-bundle", copied).returncode != 0
    with pytest.raises(ValueError): importer.verify_bundle(copied)


def test_sidecar_fields_still_rejected_by_legacy_manifest(tmp_path):
    _, out, _ = export(tmp_path)
    manifest = json.loads((out / "manifest.json").read_text()); manifest["electrical_verification"] = "not_performed"
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unknown|contract"): load_manifest(out / "manifest.json")


def test_publication_interruption_and_no_overwrite(tmp_path, monkeypatch):
    root, out = tmp_path / "source", tmp_path / "out"
    write_import_fixture(root)
    meta = metadata(tmp_path / "metadata.json")
    options = ImportOptions(root, out, True, meta, 16)
    original_write = importer._write_assets
    def interrupted(stage, *args):
        original_write(stage, *args)
        raise RuntimeError("synthetic interruption after asset write")
    monkeypatch.setattr(importer, "_write_assets", interrupted)
    with pytest.raises(RuntimeError, match="staging evidence retained"): importer.run_import(options)
    assert not out.exists()
    stages = list(tmp_path.glob(".out.staging-*")); assert len(stages) == 1
    assert (stages[0] / "import-report.json").exists() and not (stages[0] / "manifest.json").exists()
    source_before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*.png")}
    monkeypatch.setattr(importer, "_write_assets", original_write)
    importer.run_import(replace(options, out=tmp_path / "retry"))
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*.png")} == source_before
    with pytest.raises(FileExistsError): importer.run_import(replace(options, out=tmp_path / "retry"))
    inside = root / "output"
    with pytest.raises(ValueError, match="outside"): importer.run_import(replace(options, out=inside))
    assert not inside.exists()
    stage = tmp_path / "separate-stage"; stage.mkdir(); (stage / "manifest.json").write_text("candidate")
    raced = tmp_path / "raced"; raced.mkdir()
    with pytest.raises(OSError): importer._publish_noreplace(stage, raced)
    assert stage.exists() and list(raced.iterdir()) == []  # even empty destinations are never overwritten
    for path in ["../outside", "/absolute", "assets/../../outside", "C:\\outside"]:
        with pytest.raises(ValueError): importer.bundle_path(stage, path)


def test_invalid_model_blocks_entire_export_pool(tmp_path):
    root = tmp_path / "source"; write_import_fixture(root)
    bad = write_png(root, "BAD/A/1-X.png"); Image.new("L", (1203, 1001), 127).save(bad)
    out = tmp_path / "out"
    result = cli("--root", root, "--out", out, "--export", "--metadata", metadata(tmp_path / "meta.json"), "--candidate-grid", 16)
    assert result.returncode == 2 and not (out / "manifest.json").exists() and not (out / "assets").exists()
    doc = report(out)
    assert "invalid_source_inventory" in doc["pending_prerequisites"]
    assert doc["inventory"]["models"][0]["errors"][0]["path"] == "BAD/A/1-X.png"
    assert (out / "preview.html").exists()


def test_enumeration_and_root_relocation_invariance(tmp_path):
    root, out, _ = export(tmp_path)
    other = tmp_path / "relocated"; shutil.copytree(root, other)
    (other / "SYNTHETIC-M/cno/006_C6.png").rename(other / "SYNTHETIC-M/cno/6-C6.png")
    second = tmp_path / "second"
    assert cli("--root", other, "--out", second, "--export", "--metadata", tmp_path / "metadata.json", "--candidate-grid", 16).returncode == 0
    assert (out / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    assert report(out)["product_input_identities"] == report(second)["product_input_identities"]
    assert report(out)["generated_asset_hashes"] == report(second)["generated_asset_hashes"]


def test_vocabulary_inventory_and_oov_search_diagnostics(tmp_path):
    root = tmp_path / "source"
    for path in ["M/A/1-X.png", "M/A/2-Y.png", "M/B/1-X.png", "Q/A/1-X.png", "Q/D/2-Z.png"]:
        write_png(root, path)
    (root / "M/readme.txt").write_text("ignored")
    out = tmp_path / "inspect"
    assert cli("--root", root, "--out", out).returncode == 0
    inventory = report(out)["inventory"]
    assert inventory["role_union"] == ["A", "B", "D"] and inventory["electrode_layer_union"] == ["X", "Y", "Z"]
    assert inventory["unique_roles"] == ["B", "D"] and inventory["unique_electrode_layers"] == ["Y", "Z"]
    assert inventory["models"][0]["warnings"][0]["path"] == "M/readme.txt"
    assert report(out)["metadata"]["pending_models"] == ["M", "Q"]
    assert not (out / "manifest.json").exists()


def test_pillow_pixel_limit_error_and_explicit_override(tmp_path, monkeypatch):
    root = tmp_path / "source"; write_png(root, "M/A/1-X.png")
    previous = Image.MAX_IMAGE_PIXELS
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1_000_000)
    blocked = tmp_path / "blocked"
    assert importer.main(["--root", str(root), "--out", str(blocked)]) == 2
    assert report(blocked)["inventory"]["models"][0]["errors"][0]["exception_class"] == "DecompressionBombWarning"
    out = tmp_path / "override"
    assert importer.main(["--root", str(root), "--out", str(out), "--png-max-image-pixels", "2000000"]) == 0
    policy = report(out)["inventory"]["decoder"]
    assert policy["default_max_image_pixels"] == 1_000_000 and policy["effective_max_image_pixels"] == 2_000_000
    assert policy["explicit_override"] and Image.MAX_IMAGE_PIXELS == 1_000_000
    for bad in ["0", "-1", "None", "1.5"]:
        result = cli("--root", root, "--out", tmp_path / "invalid", "--png-max-image-pixels", bad)
        assert result.returncode != 0 and not (tmp_path / "invalid").exists()


def test_explicit_inspect_grid_evidence_without_export(tmp_path, monkeypatch):
    root = tmp_path / "source"; write_import_fixture(root)
    no_grid = tmp_path / "no-grid"
    assert importer.run_import(ImportOptions(root, no_grid))[1] == 0
    assert report(no_grid)["grid_evidence"] == [] and report(no_grid)["candidate_grid"] is None
    assert report(no_grid)["metadata"]["pending_models"] == ["SYNTHETIC-M"]
    original = importer.to_product
    def report_precedes_heavy_work(model, extraction, **kwargs):
        stages = list(tmp_path.glob(".grids.staging-*")); assert len(stages) == 1
        stored = report(stages[0])
        assert stored["inventory"]["models"][0]["resource_estimate"]["frame_pixels"] == 1_204_203
        return original(model, extraction, **kwargs)
    monkeypatch.setattr(importer, "to_product", report_precedes_heavy_work)
    out = tmp_path / "grids"
    assert importer.run_import(ImportOptions(root, out, inspect_grids=(16, 32)))[1] == 0
    checks = report(out)["grid_evidence"]
    assert [e["grid_check"]["grid"] for e in checks] == [16, 32]
    assert checks[0]["transform"]["resized_shape_yx"] == [13, 16]
    assert checks[0]["transform"]["padding_ltrb"] == [0, 1, 0, 2]
    assert checks[1]["transform"]["resized_shape_yx"] == [27, 32]
    assert checks[1]["transform"]["padding_ltrb"] == [0, 2, 0, 3]
    for evidence in checks:
        assert evidence["transform"] == frame_transform((1001, 1203), evidence["grid_check"]["grid"], 0.1).to_record()
        assert evidence["grid_check"]["transform"] == evidence["transform"]
    assert all(e["purpose"] == "inspect_only_not_approved" and not e["grid_check"]["connectivity_preserving"] for e in checks)
    assert report(out)["metadata"]["models"] == [] and not (out / "manifest.json").exists()


def test_multi_model_export_requires_all_explicit_metadata(tmp_path):
    root = tmp_path / "source"
    write_import_fixture(root, model="FIRST"); write_import_fixture(root, model="SECOND")
    mapping = metadata(tmp_path / "mapping.json", model="FIRST")
    blocked = tmp_path / "blocked"
    assert cli("--root", root, "--out", blocked, "--export", "--metadata", mapping, "--candidate-grid", 8).returncode == 2
    assert report(blocked)["metadata"]["pending_models"] == ["SECOND"]
    assert not (blocked / "manifest.json").exists() and not (blocked / "assets").exists()
    doc = json.loads(mapping.read_text()); doc["models"]["SECOND"] = {"lineage_id": "SYNTHETIC-SECOND-FAMILY", "design_version": "SYNTHETIC-SECOND-V1"}
    mapping.write_text(json.dumps(doc))
    out = tmp_path / "all"
    assert cli("--root", root, "--out", out, "--export", "--metadata", mapping, "--candidate-grid", 8).returncode == 0
    assert [p.product_id for p in load_manifest(out / "manifest.json")] == ["FIRST", "SECOND"]
    assert len(report(out)["product_input_identities"]) == 2


def test_cli_explicit_inspect_grids_and_invalid_metadata_are_diagnostic(tmp_path):
    root = tmp_path / "source"; write_import_fixture(root)
    out = tmp_path / "grids"
    assert cli("--root", root, "--out", out, "--inspect-grids", 16, 32).returncode == 0
    assert [e["grid_check"]["grid"] for e in report(out)["grid_evidence"]] == [16, 32]
    assert report(out)["metadata"]["models"] == []
    assert report(out)["status_summary"][0] == {
        "model_id": "SYNTHETIC-M", "extractor_run": "ran", "extraction_coverage": "partial",
        "model_connectivity_status": "partial", "model_vocabulary_eligibility": "not_evaluated_without_checkpoint",
        "electrical_verification": "not_performed"}
    bad = tmp_path / "bad.json"; bad.write_text('{"unknown":"bad mapping"}')
    diagnostic = tmp_path / "bad-metadata"
    assert cli("--root", root, "--out", diagnostic, "--export", "--metadata", bad, "--candidate-grid", 16).returncode == 2
    assert "invalid_metadata" in report(diagnostic)["pending_prerequisites"]
    assert not (diagnostic / "manifest.json").exists() and (diagnostic / "preview.html").exists()


def test_grid_failure_preserves_diagnostics_preview_and_blocks_export(tmp_path, monkeypatch):
    root = tmp_path / "source"; write_import_fixture(root)
    out = tmp_path / "failed-grid"
    def failed_grid(*args):
        raise RuntimeError("synthetic grid runner failure")
    monkeypatch.setattr(importer, "grid_check", failed_grid)
    _, code = importer.run_import(ImportOptions(root, out, True, metadata(tmp_path / "metadata.json"),
                                                16, inspect_grids=(8,)))
    assert code == 2 and report(out)["export_status"] == "blocked"
    assert "inspect_grid_evaluation_failed" in report(out)["pending_prerequisites"]
    assert "synthetic grid runner failure" in report(out)["grid_evidence"][0]["error"]
    assert (out / "preview.html").exists() and not (out / "manifest.json").exists()


def _tree_snapshot(root):
    return {p.relative_to(root).as_posix(): ("directory" if p.is_dir() else ("symlink", str(p.readlink()))
            if p.is_symlink() else (p.read_bytes(), p.stat().st_mtime_ns)) for p in root.rglob("*")}


@pytest.mark.parametrize("row", ["case-model", "case-root", "unicode-model", "case-top", "case-electrode"])
def test_rb_B1_review_table_refused_without_any_creation(tmp_path, row):
    import unicodedata
    source = tmp_path / ("입력" if row == "unicode-model" else "Src")
    write_import_fixture(source, model="M1")
    mapping = metadata(tmp_path / "meta.json", model="M1")
    if row == "unicode-model":
        alias = tmp_path / unicodedata.normalize("NFD", "입력")
        if not alias.exists() or not os.path.samefile(source, alias):
            pytest.skip("temp filesystem distinguishes NFC/NFD spellings")
        root, out = source, alias / "M1/o"
    else:
        alias = tmp_path / "SRC"
        if not alias.exists() or not os.path.samefile(source, alias):
            pytest.skip("temp filesystem is case-sensitive")
        root, out = {
            "case-model": (source, alias / "M1/o6"),
            "case-root": (tmp_path / "src", source / "M1/o8"),
            "case-top": (source, tmp_path / "src/o5"),
            "case-electrode": (source, alias / "M1/A/o7"),
        }[row]
    before = _tree_snapshot(tmp_path)
    assert importer.main(["--root", str(root), "--out", str(out), "--export", "--metadata", str(mapping), "--candidate-grid", "64"]) == 2
    assert _tree_snapshot(tmp_path) == before
    assert not out.exists()


def test_rb_B1_exact_relative_symlink_and_reverse_containment(tmp_path, monkeypatch):
    root = tmp_path / "Src"; write_import_fixture(root)
    link = tmp_path / "alias"; link.symlink_to(root, target_is_directory=True)
    before = _tree_snapshot(tmp_path)
    pairs = [(root, root / "o1"), (root, root / "SYNTHETIC-M/A/o"),
             (root, root / "../Src/o2"), (root, link / "inside"), (link, root / "inside"),
             (root, tmp_path), (root, root)]
    monkeypatch.chdir(root)
    pairs.append((Path("."), Path("SYNTHETIC-M/relative")))
    for source, out in pairs:
        assert importer.main(["--root", str(source), "--out", str(out)]) == 2
        assert _tree_snapshot(tmp_path) == before


def test_argument_and_source_failures_do_not_create_parent_or_stage(tmp_path):
    bad = tmp_path / "file"; bad.write_text("not a source directory")
    before = _tree_snapshot(tmp_path)
    for root, extra in [(tmp_path / "missing", []), (bad, []), (tmp_path, ["--candidate-grid", "16"])]:
        assert importer.main(["--root", str(root), "--out", str(tmp_path / "new/parent/output"), *extra]) == 2
        assert _tree_snapshot(tmp_path) == before


@pytest.fixture(scope="module")
def shared_export(tmp_path_factory):
    # One immutable real-CLI export reused by corruption checks, never modified.
    return export(tmp_path_factory.mktemp("rb-export"))[1]


@pytest.mark.parametrize("change", ["extra-file", "missing-file", "empty-directory", "nested-directory", "manifest-report-link"])
def test_rb_S14_unlisted_missing_entries_and_manifest_report_hash_link(tmp_path, shared_export, change):
    copied = tmp_path / "copy"; shutil.copytree(shared_export, copied)
    if change == "extra-file":
        (copied / "unexpected.txt").write_text("not in integrity")
    elif change == "missing-file":
        (copied / "preview.html").unlink()
    elif change == "empty-directory":
        (copied / "empty-extra").mkdir()
    elif change == "nested-directory":
        (copied / "preview/empty-extra").mkdir()
    else:
        doc = report(copied); doc["generated_manifest_sha256"] = "0" * 64
        importer._write_json(copied / "import-report.json", doc)
        integrity = json.loads((copied / "bundle-integrity.json").read_text())
        integrity["files"]["import-report.json"] = importer.file_sha256(copied / "import-report.json")
        importer._write_json(copied / "bundle-integrity.json", integrity)
    with pytest.raises((ValueError, OSError), match="unlisted|missing|link"):
        importer.verify_bundle(copied)


@pytest.mark.parametrize("malformation", ["missing-status", "list-report", "missing-inventory", "missing-publication", "missing-algorithm", "missing-run-status", "bad-products", "bad-assets"])
def test_verify_malformed_report_or_manifest_is_clean_cli_diagnostic(tmp_path, shared_export, malformation, capsys):
    copied = tmp_path / "copy"; shutil.copytree(shared_export, copied)
    doc = report(copied)
    if malformation == "missing-status": del doc["export_status"]
    elif malformation == "list-report": doc = []
    elif malformation == "missing-inventory": del doc["inventory"]
    elif malformation == "missing-publication": del doc["publication_method"]
    elif malformation == "missing-algorithm": del doc["algorithm_version"]
    elif malformation == "missing-run-status": del doc["extractor_run_status"]
    elif malformation == "bad-assets": doc["generated_asset_hashes"] = None
    else:
        manifest = json.loads((copied / "manifest.json").read_text()); manifest["products"] = None
        importer._write_json(copied / "manifest.json", manifest)
        doc["generated_manifest_sha256"] = importer.file_sha256(copied / "manifest.json")
    importer._write_json(copied / "import-report.json", doc)
    importer._seal_bundle(copied)  # bypass file hash so structural report checks run
    assert importer.main(["--verify-bundle", str(copied)]) == 2
    error = capsys.readouterr().err
    assert error and "Traceback" not in error


@pytest.mark.parametrize("capability", ["missing-symbol", "enosys", "einval", "windows"])
def test_publish_fallback_claim_and_manifest_last(tmp_path, shared_export, monkeypatch, capability):
    stage = tmp_path / "stage"; shutil.copytree(shared_export, stage)
    out = tmp_path / "published"
    def unavailable(*args):
        if capability in {"missing-symbol", "windows"}: raise AttributeError("no native symbol")
        import errno
        raise OSError(errno.ENOSYS if capability == "enosys" else errno.EINVAL, "native operation unavailable")
    if capability == "windows":
        monkeypatch.setattr(importer.sys, "platform", "win32")
        def no_libc(*args, **kwargs): raise AssertionError("Windows must not try loading libc")
        monkeypatch.setattr(importer.ctypes, "CDLL", no_libc)
    else:
        monkeypatch.setattr(importer, "_native_publish_noreplace", unavailable)
    original_copy = importer._copy_exclusive
    copied = []
    def ordered(source, target):
        assert not (out / "manifest.json").exists()
        copied.append(target.relative_to(out).as_posix())
        original_copy(source, target)
    monkeypatch.setattr(importer, "_copy_exclusive", ordered)
    assert importer._publish_noreplace(stage, out) == "exclusive_directory_claim_manifest_last"
    assert copied[-1] == "manifest.json"
    assert report(out)["publication_method"] == "exclusive_directory_claim_manifest_last"
    assert importer.verify_bundle(out)["status"] == "verified"
    before = _tree_snapshot(out)
    with pytest.raises(FileExistsError): importer._publish_noreplace(stage, out)
    assert _tree_snapshot(out) == before


@pytest.mark.parametrize("interrupt", [False, True])
def test_publish_fallback_failure_never_leaves_consumable_manifest(tmp_path, shared_export, monkeypatch, interrupt):
    stage = tmp_path / "stage"; shutil.copytree(shared_export, stage)
    out = tmp_path / "failed"
    def unavailable(*args): raise AttributeError("native symbol absent")
    monkeypatch.setattr(importer, "_native_publish_noreplace", unavailable)
    original = importer._copy_exclusive
    def failed_after_final_manifest(source, target):
        original(source, target)
        if target.name == "manifest.json":
            raise KeyboardInterrupt("synthetic stop") if interrupt else OSError("synthetic final copy fault")
    monkeypatch.setattr(importer, "_copy_exclusive", failed_after_final_manifest)
    with pytest.raises(KeyboardInterrupt if interrupt else OSError): importer._publish_noreplace(stage, out)
    assert not (out / "manifest.json").exists()
    assert (out / "manifest.pending.json").exists()
    assert report(out)["export_status"] == ("interrupted" if interrupt else "failed")
    assert not (stage / "manifest.json").exists() and (stage / "manifest.pending.json").exists()
    assert report(stage)["publication_method"] == "exclusive_directory_claim_manifest_last"


@pytest.mark.parametrize("point", ["validation", "seal", "verify", "publish", "interrupt-seal", "writing-leftover"])
def test_rb_S15_multi_stage_failure_and_pending_manifest(tmp_path, monkeypatch, point):
    root = tmp_path / "source"; write_import_fixture(root)
    out = tmp_path / "out"
    mapping = metadata(tmp_path / "metadata.json")
    original_loader = importer.load_manifest
    def guarded_loader(path):
        assert path.name == "manifest.pending.json"
        assert not (path.parent / "manifest.json").exists()
        if point == "validation": raise RuntimeError("synthetic validation fault")
        return original_loader(path)
    monkeypatch.setattr(importer, "load_manifest", guarded_loader)
    target = {"seal": "_seal_bundle", "verify": "verify_bundle", "publish": "_publish_noreplace", "interrupt-seal": "_seal_bundle"}.get(point)
    if target:
        def fail(*args):
            assert (args[0] / "manifest.json").is_file()
            if point == "interrupt-seal": raise KeyboardInterrupt("synthetic interrupt")
            raise RuntimeError("synthetic " + point + " fault")
        monkeypatch.setattr(importer, target, fail)
    if point == "writing-leftover":
        original_write = importer._write_json
        def leftovers(path, payload):
            if path.name == "import-report.json" and payload["export_status"] == "exported":
                path.with_name(path.name + ".writing").write_text("abandoned partial report")
                raise OSError("synthetic original write fault")
            return original_write(path, payload)
        monkeypatch.setattr(importer, "_write_json", leftovers)
    with pytest.raises(KeyboardInterrupt if point == "interrupt-seal" else RuntimeError, match="synthetic"):
        importer.run_import(ImportOptions(root, out, True, mapping, 16))
    stage, = tmp_path.glob(".out.staging-*")
    assert not out.exists() and not (stage / "manifest.json").exists()
    assert (stage / "manifest.pending.json").is_file()
    assert report(stage)["export_status"] == ("interrupted" if point == "interrupt-seal" else "failed")
    if point == "writing-leftover": assert (stage / "import-report.json.writing").exists()


def test_metadata_diagnostic_is_location_independent(tmp_path):
    root = tmp_path / "source"; write_png(root, "M/A/1-X.png")
    outputs = []
    for name in ("first", "second"):
        where = tmp_path / name; where.mkdir()
        bad = where / "metadata.json"; bad.write_text('{"unknown": "not metadata"}')
        out = where / "out"
        assert importer.run_import(ImportOptions(root, out, True, bad, 16))[1] == 2
        payload = report(out)
        diagnostic, = [d for d in payload["diagnostics"] if d["path"] == "metadata"]
        assert str(tmp_path) not in json.dumps(payload, ensure_ascii=False)
        assert diagnostic["reason"] == "metadata: metadata requires exactly metadata_schema_version and models"
        outputs.append((out / "import-report.json").read_bytes())
    assert outputs[0] == outputs[1]


def test_model_run_status_does_not_say_complete_for_partial_coverage(shared_export):
    doc = report(shared_export)
    assert doc["report_schema_version"] == "dl1_png_import_v2"
    assert doc["extractor_run_status"] == "ran"
    model, = doc["extraction"]
    assert model["extractor_run_status"] == "ran" and model["connectivity_observation"] == "partial"
    assert "extraction_status" not in model and "extraction_status" not in doc
    expected = {"darwin_rename_excl"} if sys.platform == "darwin" else {"linux_renameat2_noreplace", "exclusive_directory_claim_manifest_last"}
    assert doc["publication_method"] in expected


def test_fallback_raced_claim_preserves_destination(tmp_path, shared_export, monkeypatch):
    import errno
    stage = tmp_path / "stage"; shutil.copytree(shared_export, stage)
    out = tmp_path / "raced"
    def unavailable_and_raced(*args):
        out.mkdir(); (out / "existing.txt").write_text("existing user evidence")
        raise OSError(errno.ENOSYS, "native unavailable")
    monkeypatch.setattr(importer, "_native_publish_noreplace", unavailable_and_raced)
    with pytest.raises(FileExistsError): importer._publish_noreplace(stage, out)
    assert sorted(p.name for p in out.iterdir()) == ["existing.txt"]
    assert (out / "existing.txt").read_text() == "existing user evidence"
    assert not (out / "manifest.json").exists()
