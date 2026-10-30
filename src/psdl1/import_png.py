"""Offline PNG inspection/export; strict v1 manifest plus versioned provenance."""
from contextlib import closing
import argparse
import ctypes
import errno
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
import tempfile

import numpy as np

from .data import frame_transform, input_identity, _iter_manifest_products
from .png_diagnostic import inspect_grid_diagnostics, observation_descriptors, VERSION
from ._png_diagnostic_render import DiagnosticSink
from .png_connections import (element_id, extract_model, iter_connections, to_product,
                              _detach_extraction, _reload_extraction)
from .png_contract import (ImportOptions, ImportReport, SourceIssue, canonical_id, mask_hash,
                           require_report_version)
from .png_source import (decoder_session, load_metadata, report_payload, load_stack_order,
                         _SourceScanner, _decode, _detach_model, _require_model_masks)
from .schema import SCHEMA_VERSION, validate_product

BUNDLE_SCHEMA_VERSION = "dl1_png_bundle_v3"
SCREENING_LAYERS_SCHEMA_VERSION = "dl1_screening_layers_v1"
SCREENING_LAYERS_FILE = "screening-layers.json"
SCREENING_LAYERS_DIR = "screening_layers"


def _add_note(exc, message):
    """BaseException.add_note is Python 3.11+; older interpreters keep the note in __notes__."""
    if hasattr(exc, "add_note"):
        exc.add_note(message)
    else:
        exc.__notes__ = [*getattr(exc, "__notes__", []), message]


class _ProgressSinkError(RuntimeError):
    """Transport failure must abort, never become recoverable grid evidence."""



def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique, parse_constant=invalid)


def _write_json(path, payload):
    path = Path(path)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    # Replace only this invocation's own staged report; never an existing output.
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
    temporary.replace(path)


def bundle_path(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative or
            PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).is_absolute() or
            any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise ValueError("invalid contained bundle path")
    path = root / relative
    if path.is_symlink():
        raise ValueError("generated bundle files may not be symlinks")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("bundle path escapes root") from exc
    return path


def _seal_bundle(stage):
    files = {p.relative_to(stage).as_posix(): file_sha256(p)
             for p in sorted(stage.rglob("*")) if p.is_file() and p.name != "bundle-integrity.json"}
    _write_json(stage / "bundle-integrity.json", {"bundle_schema_version": BUNDLE_SCHEMA_VERSION,
                                                  "hash_algorithm": "sha256", "files": files,
                                                  "directories": sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_dir())})


def _verify_bundle(bundle):
    """Verify contained file hashes and manifest/report/asset links, without PNG sources.

    This detects corruption; hashes are not signatures or proof of electrical truth.
    Verification reads headers of generated bool NPYs, without a full legacy load.
    """
    root = Path(bundle).resolve(strict=True)
    integrity = _read_json(bundle_path(root, "bundle-integrity.json"))
    if not isinstance(integrity, dict) or set(integrity) != {"bundle_schema_version", "hash_algorithm", "files", "directories"}:
        raise ValueError("invalid bundle integrity fields")
    if integrity["bundle_schema_version"] != BUNDLE_SCHEMA_VERSION or integrity["hash_algorithm"] != "sha256":
        raise ValueError("unsupported bundle integrity version/algorithm")
    files = integrity["files"]
    if not isinstance(files, dict) or "import-report.json" not in files:
        raise ValueError("bundle integrity requires import-report.json")
    for relative, expected in files.items():
        path = bundle_path(root, relative)
        if not path.is_file():
            raise ValueError(f"missing or non-file bundle entry: {relative}")
        if not isinstance(expected, str) or len(expected) != 64 or file_sha256(path) != expected:
            raise ValueError(f"bundle hash mismatch: {relative}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")}
    directories = integrity["directories"]
    if not isinstance(directories, list) or any(not isinstance(d, str) for d in directories) or len(set(directories)) != len(directories):
        raise ValueError("invalid bundle directory listing")
    for relative in directories:
        path = bundle_path(root, relative)
        if not path.is_dir():
            raise ValueError("missing bundle directory")
    if actual != set(files) | set(directories) | {"bundle-integrity.json"}:
        raise ValueError("unlisted or missing bundle files")
    report = _read_json(bundle_path(root, "import-report.json"))
    require_report_version(report)
    expected_fields = (set(ImportReport.__dataclass_fields__) - {"extraction_status"}) | {"extractor_run_status"}
    if set(report) != expected_fields:
        raise ValueError("invalid or incomplete import report fields")
    for key, kind in (("inventory", dict), ("metadata", dict), ("extraction", list), ("preview", list),
                      ("status_summary", list), ("diagnostics", list)):
        if not isinstance(report.get(key), kind):
            raise ValueError("invalid or incomplete import report: " + key)
    skipped = sum(isinstance(issue, dict) and issue.get("reason") == "preview_skipped_by_request: --no-preview"
                  for issue in report["diagnostics"])
    if skipped and (skipped != 1 or report["preview"] or "preview.html" in files or
                    any(p == "preview" or p.startswith("preview/") for p in actual)):
        raise ValueError("inconsistent skipped-preview evidence")
    from .png_diagnostic import verify_diagnostic_artifacts
    verify_diagnostic_artifacts(root, report, files, bundle_path)
    if report.get("export_status") not in {"exported", "not_requested", "blocked"}:
        raise ValueError("invalid or incomplete import report export_status")
    if report["export_status"] == "exported":
        if files.get("manifest.json") != report["generated_manifest_sha256"]:
            raise ValueError("manifest/report hash link mismatch")
        manifest = _read_json(bundle_path(root, "manifest.json"))
        if set(manifest) != {"schema_version", "candidate_grid", "products"} or manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError("invalid exported manifest contract")
        listed_assets = dict(report["generated_asset_hashes"])
        if len(listed_assets) != len(report["generated_asset_hashes"]):
            raise ValueError("duplicate reported generated asset")
        required = set()
        for product in manifest["products"]:
            if report["cnt_mode"] == "image_only" and (product["connections"] != [] or product["connectivity_observation"] != "unavailable"):
                raise ValueError("image-only manifest contains relation evidence")
            for relative in ([product["valid"]] + ([product["cnt"]] if product.get("cnt") else []) +
                             [e["mask"] for e in product["elements"]] + [c["mask"] for c in product["connections"]]):
                required.add(relative)
                if not relative.startswith("assets/") or files.get(relative) != listed_assets.get(relative):
                    raise ValueError("manifest asset/report hash link mismatch")
                mask = np.load(bundle_path(root, relative), mmap_mode="r", allow_pickle=False)
                if mask.dtype != np.bool_ or tuple(mask.shape) != tuple(product["frame"]["shape_yx"]):
                    raise ValueError("generated mask dtype/shape mismatch")
                del mask
        if set(listed_assets) != required or {p for p in files if p.startswith("assets/")} != required:
            raise ValueError("unreferenced or missing generated assets")
        if {p[0] for p in report["product_input_identities"]} != {p["product_id"] for p in manifest["products"]}:
            raise ValueError("missing Product input identities")
        _verify_screening_layers(root, files, manifest)
    elif ("manifest.json" in files or SCREENING_LAYERS_FILE in files or
          any(p.startswith("assets/") or p.startswith(SCREENING_LAYERS_DIR + "/") for p in files)):
        raise ValueError("diagnostic-only bundle contains model-ready files")
    return {"status": "verified", "report_schema_version": report["report_schema_version"],
            "export_status": report["export_status"], "file_count": len(files),
            "verification_meaning": "bundle_hash_integrity_not_electrical_verification"}


def verify_bundle(bundle):
    """Malformed evidence always produces a clean diagnostic, never a traceback."""
    try:
        return _verify_bundle(bundle)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as exc:
        raise ValueError("invalid or incomplete bundle evidence: " + str(exc)) from exc


def _native_publish_noreplace(stage, destination):
    """Native atomic no-replace operation; capability errors trigger a fallback."""
    if sys.platform != "darwin" and not sys.platform.startswith("linux"):
        raise AttributeError("no native atomic no-replace directory publication API")
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(stage), os.fsencode(destination), 0x00000004)
    elif sys.platform.startswith("linux"):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(stage), -100, os.fsencode(destination), 1)
    else:
        raise AttributeError("no native atomic no-replace directory publication API")
    if result:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def _publication_record(stage, method):
    if (stage / "import-report.json").is_file():
        payload = _read_json(stage / "import-report.json")
        payload["publication_method"] = method
        _write_json(stage / "import-report.json", payload)
        _seal_bundle(stage)
        verify_bundle(stage)


def _disable_manifest(directory):
    final = directory / "manifest.json"
    if final.exists():
        pending = directory / "manifest.pending.json"
        if pending.exists():
            pending = directory / "manifest.failed.json"
        final.rename(pending)


def _failure_evidence(directory, report, exc):
    """Best effort, without reusing a possibly abandoned .writing file."""
    _disable_manifest(directory)
    payload = report_payload(report) if report is not None else _read_json(directory / "import-report.json")
    if report is not None:
        try:
            payload["publication_method"] = _read_json(directory / "import-report.json").get("publication_method")
        except (ValueError, OSError, AttributeError):
            pass
    payload["export_status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
    payload.setdefault("diagnostics", []).append({"path": ".", "reason": str(exc), "exception_class": type(exc).__name__})
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".failure-report-", suffix=".writing",
                                     dir=directory, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(handle.name, directory / "import-report.json")


def _copy_exclusive(source, target):
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        for block in iter(lambda: incoming.read(65536), b""):
            outgoing.write(block)


def _publish_claimed(stage, destination):
    """Exclusive mkdir claim; manifest last, no final manifest left on failure.

    This fallback is not atomic directory visibility: readers may see a partial
    diagnostic directory. It never replaces an existing destination.
    """
    destination.mkdir()  # the claim itself is atomic and refuses every existing path
    try:
        for source in sorted(stage.rglob("*")):
            relative = source.relative_to(stage)
            if source.is_dir():
                (destination / relative).mkdir()
            elif relative.as_posix() != "manifest.json":
                _copy_exclusive(source, destination / relative)
        if (stage / "manifest.json").is_file():
            _copy_exclusive(stage / "manifest.json", destination / "manifest.json")
        verify_bundle(destination)
    except BaseException as exc:
        try:
            _failure_evidence(destination, None, exc)
        except Exception:
            # Disabling the consumable manifest has priority over report writing.
            _disable_manifest(destination)
        raise


def _publish_noreplace(stage, destination):
    try:
        if os.path.lexists(destination):
            raise FileExistsError("output destination already exists")
        method = "darwin_rename_excl" if sys.platform == "darwin" else "linux_renameat2_noreplace"
        _publication_record(stage, method)
        try:
            _native_publish_noreplace(stage, destination)
            return method
        except AttributeError:
            pass
        except OSError as exc:
            if exc.errno not in {errno.ENOSYS, errno.EINVAL}:
                raise
        method = "exclusive_directory_claim_manifest_last"
        _publication_record(stage, method)
        _publish_claimed(stage, destination)
        return method
    except BaseException as exc:
        try:
            _failure_evidence(stage, None, exc)
        except Exception as evidence_error:
            _add_note(exc, "failure evidence could not be updated: " + str(evidence_error))
        raise


def _refuse_overlapping_trees(root, out):
    # resolve() does not canonicalise case or Unicode on APFS. Compare existing
    # ancestor identities before making even a parent or staging directory.
    for ancestor in (out, *out.parents):
        if ancestor.exists() and os.path.samefile(ancestor, root):
            raise ValueError("output must be outside the input tree")
    if out.exists():
        for ancestor in (root, *root.parents):
            if os.path.samefile(ancestor, out):
                raise ValueError("input root must be outside the output destination")


def _duplicates(inventory):
    groups = {}
    for model in inventory.models:
        for a in model.assets:
            groups.setdefault(a.mask_sha256, []).append(a.relative_path)
    result = [{"kind": "identical_asset_geometry", "source_paths": sorted(paths),
               "mask_sha256": value, "meaning": "possible_alias_or_duplicate_no_automatic_merge"}
              for value, paths in sorted(groups.items()) if len(paths) > 1]
    model_groups = {}
    for model in inventory.models:
        if model.valid:
            key = tuple(sorted((a.kind, a.electrode or "", a.layer, a.mask_sha256) for a in model.assets))
            model_groups.setdefault(key, []).append(model.model_id)
    result.extend({"kind": "identical_model_geometry", "model_ids": sorted(names),
                   "meaning": "possible_in_root_symlink_alias_or_duplicate_product_geometry_not_order_equivalence"}
                  for _, names in sorted(model_groups.items()) if len(names) > 1)
    return tuple(result)


def _write_assets(stage, model, extraction, metadata):
    """One bool connection mask at a time; return scalar manifest evidence."""
    _require_model_masks(model)
    hashes = []
    def save(identity, mask):
        relative = "assets/" + identity + ".npy"
        path = bundle_path(stage, relative)
        with path.open("xb") as handle:
            np.save(handle, np.asarray(mask, dtype=np.bool_), allow_pickle=False)
        hashes.append((relative, file_sha256(path)))
        return relative
    f = model.frame
    valid = save(canonical_id("valid", [model.model_id, f.frame_id]), np.ones(f.shape_yx, dtype=np.bool_))
    elements = []
    for a in sorted(model.electrodes, key=lambda a: (model.order_index(a) if model.order_index(a) is not None else -1, a.layer, a.electrode)):
        eid = element_id(a)
        elements.append({"element_id": eid, "electrode_id": a.electrode, "role": a.role,
                         "actual_layer": a.layer, "observation": "observed", "mask": save(eid, a.mask)})
    cnt_path = None
    if model.cnt_assets:
        cnt = np.zeros(f.shape_yx, dtype=np.bool_)
        for a in model.cnt_assets:
            cnt |= a.mask
        cnt_path = save(canonical_id("global_cnt", [model.model_id, f.frame_id]), cnt)
        del cnt
    connections = []
    for connection in iter_connections(extraction):
        connections.append({"connection_id": connection.connection_id, "kind": "CNT", "observation": "observed",
                            "endpoints": [{"element_id": e.element_id, "xy_um": list(e.xy_um)} for e in connection.endpoints],
                            "mask": save(connection.connection_id, connection.mask)})
        del connection
    product = {"product_id": model.model_id, "lineage_id": metadata.lineage_id,
               "design_version": metadata.design_version, "unit_size_um": list(f.frame_extent_um),
               "frame": {"frame_id": f.frame_id, "length_unit": "um", "axis_order": "yx",
                         "shape_yx": list(f.shape_yx), "origin_xy_um": list(f.origin_xy_um), "pixel_pitch_um": 0.1},
               "valid": valid, "elements": elements, "connections": connections,
               "connectivity_observation": extraction.connectivity_observation}
    if cnt_path is not None:
        product["cnt"] = cnt_path
    return product, hashes


def _scan_inventory(root, options, reference, progress):
    """Only detached records leave this inventory phase."""
    scanner = _SourceScanner(root, png_max_image_pixels=options.png_max_image_pixels,
                             stack_order=reference, progress=progress)
    models, extractions = [], []
    with closing(iter(scanner)) as stream:
        for model in stream:
            result = extract_model(model)
            models.append(_detach_model(replace(model, resource_estimate=result.resource_estimate)))
            extractions.append(_detach_extraction(result))
            del model, result
    return scanner.finish(models), tuple(extractions)


def _inspect_preview_model(root, model, result, policy, grids, stage, no_preview, progress):
    """All live source/Product/image references end before returning scalar evidence."""
    evidence, diagnostics, rows, section = [], [], (), None
    failed = False
    blocked = result.extraction_status == "blocked"
    if grids and (blocked or not model.valid):
        evidence.append({"model_id": model.model_id, "purpose": "inspect_not_evaluated",
                         "reason": "invalid_source_inventory" if not model.valid else "invalid_stack_order"})
        evidence[-1]["diagnostic"] = {"diagnostic_schema_version": VERSION, "state": "not_evaluated",
                                      "reason": evidence[-1]["reason"], "requested_grids": list(grids)}
        progress(stage="grid", state="skipped", model_id=model.model_id, reason=evidence[-1]["reason"])
        for grid in grids:
            progress(stage="diagnostic", state="skipped", model_id=model.model_id, grid=grid,
                     reason=evidence[-1]["reason"])
    if model.valid and ((grids and not blocked) or not no_preview):
        # Source-change failures abort the import, outside recoverable grid errors.
        model, result = _reload_extraction(root, model, result, policy, progress=progress)
    if grids and model.valid and not blocked:
        product = None
        grid_record = {}
        diagnostic_started = set()
        def record_failure(message, exception_class):
            nonlocal failed
            failed = True
            diagnostics.append(SourceIssue(model.model_id, "inspect grid evaluation failed: " + message, exception_class))
            evidence.append({"model_id": model.model_id, "purpose": "inspect_failed_not_approved", "error": message,
                             "diagnostic": {"diagnostic_schema_version": VERSION, "state": "not_evaluated",
                                            "reason": message, "requested_grids": [grid_record.get("grid", grids[0])]}})
            progress(stage="diagnostic", state="failed", model_id=model.model_id, error=message, **grid_record)
            progress(stage="grid", state="failed", model_id=model.model_id, error=message, **grid_record)
            failed_index=grids.index(grid_record.get("grid",grids[0]))
            for later in grids[failed_index+1:]:
                if later in diagnostic_started:
                    progress(stage="diagnostic",state="skipped",model_id=model.model_id,grid=later,reason="earlier_grid_failed")
        try:
            product = to_product(model, result)
            def sink_factory(grid, transform, valid):
                progress(stage="diagnostic", state="started", model_id=model.model_id, grid=grid)
                diagnostic_started.add(grid)
                directory = stage / "diagnostics" / canonical_id("model", [model.model_id])
                return DiagnosticSink(directory / f"grid-{grid}.png", directory / f"grid-{grid}.jsonl",
                                      model_id=model.model_id, transform=transform, valid=valid,
                                      cnt_mode=model.cnt_mode,
                                      observation_count=len(product.elements)+len(product.connections)+
                                                        int(product.cnt is not None)+len(model.cnt_assets),
                                      progress=lambda **record: progress(stage="diagnostic", model_id=model.model_id, **record))
            descriptors = observation_descriptors(product, model, result)
            with closing(inspect_grid_diagnostics(product, descriptors, grids, sink_factory,
                         propagate_exceptions=(_ProgressSinkError,))) as checks:
                for index, grid in enumerate(grids, 1):
                    grid_record = {"grid": grid, "grid_index": index, "grid_count": len(grids)}
                    progress(stage="grid", state="started", model_id=model.model_id, **grid_record)
                    outcome = next(checks)
                    if outcome["state"] != "complete":
                        record_failure(outcome["reason"], outcome["error_type"])
                        break
                    artifact = outcome["diagnostic"]
                    for field in ("png", "evidence"):
                        artifact[field] = Path(artifact[field]).relative_to(stage).as_posix()
                    evidence.append({"model_id": model.model_id, "purpose": "inspect_only_not_approved",
                                     "transform": frame_transform(model.frame.shape_yx, grid, model.frame.pixel_pitch_um).to_record(),
                                     "grid_check": outcome["grid_check"], "diagnostic": artifact})
                    progress(stage="diagnostic", state="completed", model_id=model.model_id, grid=grid)
                    progress(stage="grid", state="completed", model_id=model.model_id,
                             grid=grid, grid_index=index, grid_count=len(grids))
        except Exception as exc:
            if isinstance(exc, _ProgressSinkError):
                raise
            record_failure(str(exc), type(exc).__name__)
        finally:
            del product
    if not no_preview:
        from .png_preview import render_model_preview
        progress(stage="preview", state="started", model_id=model.model_id)
        try:
            rows, section = render_model_preview(stage, model, result)
        except BaseException:
            progress(stage="preview", state="failed", model_id=model.model_id)
            raise
        progress(stage="preview", state="completed", model_id=model.model_id)
    return evidence, diagnostics, failed, rows, section


def _export_model(stage, root, model, result, metadata, policy, progress):
    model, result = _reload_extraction(root, model, result, policy, progress=progress)
    return _write_assets(stage, model, result, metadata)


def _write_screening_layers(stage, root, model, policy):
    """Screening-only sidecar masks, one decode at a time; never part of manifest or Product assets.

    Returns the product's sidecar record, or None when the model has no `Layer/` PNG.
    """
    if not model.layer_files:
        return None
    (stage / SCREENING_LAYERS_DIR).mkdir(exist_ok=True)
    layers = []
    for record in sorted(model.layer_files, key=lambda r: (r.filename_number, r.name)):
        try:
            mask, _, digest = _decode(root / record.relative_path, record.relative_path, policy)
            if (digest != record.file_sha256 or tuple(mask.shape) != tuple(record.shape_yx) or
                    mask_hash(mask) != record.mask_sha256):
                raise ValueError("snapshot differs from inventory")
        except (OSError, ValueError, RuntimeError) as exc:
            raise ValueError(f"source changed: {record.relative_path}: {exc}") from exc
        relative = SCREENING_LAYERS_DIR + "/" + canonical_id("screening_layer", [model.model_id, record.name]) + ".npy"
        with bundle_path(stage, relative).open("xb") as handle:
            np.save(handle, np.asarray(mask, dtype=np.bool_), allow_pickle=False)
        del mask
        layers.append({"layer": record.name, "filename_number": record.filename_number,
                       "relative_path": record.relative_path, "mask": relative,
                       "mask_sha256": record.mask_sha256})
    return {"frame_id": model.frame.frame_id, "layers": layers}


def _verify_screening_layers(root, files, manifest):
    """Sidecar links: schema, containment, product/frame, case-unique names, bool/shape, geometry hash."""
    frames = {p["product_id"]: p["frame"] for p in manifest["products"]}
    referenced = set()
    if SCREENING_LAYERS_FILE in files:
        doc = _read_json(bundle_path(root, SCREENING_LAYERS_FILE))
        if not isinstance(doc, dict) or set(doc) != {"schema_version", "products"}:
            raise ValueError("invalid screening layers fields")
        if doc["schema_version"] != SCREENING_LAYERS_SCHEMA_VERSION or not isinstance(doc["products"], dict):
            raise ValueError("unsupported screening layers schema")
        for product_id, entry in doc["products"].items():
            if product_id not in frames:
                raise ValueError("screening layers product is not in manifest")
            frame = frames[product_id]
            if (not isinstance(entry, dict) or set(entry) != {"frame_id", "layers"} or
                    entry["frame_id"] != frame["frame_id"] or not isinstance(entry["layers"], list) or not entry["layers"]):
                raise ValueError("invalid screening layers product entry")
            folded = set()
            previous = None
            for layer in entry["layers"]:
                if not isinstance(layer, dict) or set(layer) != {"layer", "filename_number", "relative_path", "mask", "mask_sha256"}:
                    raise ValueError("invalid screening layer fields")
                name, number, mask_path = layer["layer"], layer["filename_number"], layer["mask"]
                if (not isinstance(name, str) or not name or type(number) is not int or number <= 0 or
                        not isinstance(layer["relative_path"], str) or not isinstance(layer["mask_sha256"], str)):
                    raise ValueError("invalid screening layer values")
                if name.casefold() in folded:
                    raise ValueError("duplicate screening layer name")
                folded.add(name.casefold())
                if previous is not None and previous > (number, name):
                    raise ValueError("screening layers are not in deterministic order")
                previous = (number, name)
                if (not isinstance(mask_path, str) or not mask_path.startswith(SCREENING_LAYERS_DIR + "/") or
                        mask_path in referenced or files.get(mask_path) is None):
                    raise ValueError("invalid or duplicate screening layer mask path")
                referenced.add(mask_path)
                mask = np.load(bundle_path(root, mask_path), mmap_mode="r", allow_pickle=False)
                if mask.dtype != np.bool_ or tuple(mask.shape) != tuple(frame["shape_yx"]):
                    raise ValueError("screening layer mask dtype/shape mismatch")
                if mask_hash(np.array(mask)) != layer["mask_sha256"]:
                    raise ValueError("screening layer mask hash mismatch")
                del mask
    if {p for p in files if p.startswith(SCREENING_LAYERS_DIR + "/")} != referenced:
        raise ValueError("unreferenced or missing screening layer files")


def run_import(options, *, event_sink=None, progress_sink=None):
    """Build in a new sibling stage; leave failed stages for explicit inspection."""
    if options.verify_bundle is not None:
        if options.no_preview:
            raise ValueError("--verify-bundle must be used alone")
        return verify_bundle(options.verify_bundle), 0
    if options.root is None or options.out is None:
        raise ValueError("--root and --out are required for inspection/export")
    root = options.root.resolve(strict=True)
    out = options.out.resolve()
    if not root.is_dir():
        raise ValueError("source root must be a directory")
    _refuse_overlapping_trees(root, out)
    if os.path.lexists(options.out):
        raise FileExistsError("output destination already exists; choose a new --out")
    if out == root or out.is_relative_to(root):
        raise ValueError("output must be outside the input tree")
    if options.candidate_grid is not None and not options.export:
        raise ValueError("--candidate-grid requires --export; inspection uses --inspect-grids")
    stage = None
    report = None
    published = False
    def emit(payload):
        if event_sink is not None:
            event_sink(payload)
    def progress(phase, **record):
        if progress_sink is not None:
            try:
                progress_sink({"event": "progress", "phase": phase, **record})
            except Exception as exc:
                raise _ProgressSinkError(str(exc)) from exc
    try:
        # Nested source session retains the explicit guard for the whole import.
        with decoder_session(options.png_max_image_pixels) as decoder_policy:
            reference = load_stack_order(options.stack_order)
            progress("inventory", stage="inventory", state="started")
            inventory, extraction = _scan_inventory(root, options, reference,
                                                    lambda **record: progress("inventory", **record))
            progress("inventory", stage="inventory", state="completed", model_count=len(inventory.models))
            inventory = replace(inventory, decoder=decoder_policy)
            diagnostics = []
            metadata_error = False
            if inventory.decoder.pillow_version != "12.3.0":
                diagnostics.append(SourceIssue(".", "runtime Pillow differs from tested/pinned 12.3.0: " + inventory.decoder.pillow_version))
            try:
                metadata = load_metadata(options.metadata, [m.model_id for m in inventory.models])
            except (ValueError, OSError) as exc:
                metadata_error = True
                metadata = load_metadata(None, [m.model_id for m in inventory.models])
                diagnostics.append(SourceIssue("metadata", str(exc)))
            diagnostics.extend(SourceIssue("stack_order", e) for e in reference.errors)
            for model in inventory.models:
                diagnostics.extend(SourceIssue("stack_order", e) for e in model.stack_order.errors if e not in reference.errors)
            pending = []
            order_invalid = bool(reference.errors) or any(m.stack_order.status == "invalid" for m in inventory.models)
            if order_invalid:
                pending.append("invalid_stack_order")
            if options.export:
                if metadata.pending_models:
                    pending.append("explicit_metadata_missing")
                if options.candidate_grid is None:
                    pending.append("explicit_candidate_grid_missing")
                if not inventory.valid:
                    pending.append("invalid_source_inventory")
                if metadata_error:
                    pending.append("invalid_metadata")
            layer_counts = {m.model_id: len(m.layer_files) for m in inventory.models}
            duplicates = _duplicates(inventory)
            diagnostics.extend(SourceIssue(".", "possible duplicate/alias geometry: " + str(entry)) for entry in duplicates)
            report = ImportReport(inventory, metadata, stack_order=reference, cnt_mode="image_only" if options.stack_order is None else "relations", extraction=extraction,
                                  extraction_status=("not_run_image_only" if options.stack_order is None and all(r.extraction_status == "not_run_image_only" for r in extraction) else "extractor_ran" if all(r.extraction_status == "complete" for r in extraction) else
                                                     "blocked" if all(r.extraction_status == "blocked" for r in extraction) else "partial"),
                                  candidate_grid=options.candidate_grid, pending_prerequisites=tuple(pending),
                                  diagnostics=tuple(diagnostics), duplicate_geometry=duplicates,
                                  status_summary=tuple({"model_id": r.model_id,
                                      "extractor_run": "ran" if r.extraction_status == "complete" else r.extraction_status,
                                      "extraction_coverage": r.connectivity_observation,
                                      "model_connectivity_status": "available" if r.connectivity_observation == "complete" else r.connectivity_observation,
                                      "model_vocabulary_eligibility": "not_evaluated_without_checkpoint",
                                      "electrical_verification": "not_performed",
                                      **({"screening_layers": layer_counts[r.model_id]} if layer_counts.get(r.model_id) else {})}
                                     for r in extraction),
                                  export_status="blocked" if pending and options.export else "not_requested",
                                  grid_readiness="explicit_candidate_supplied" if options.candidate_grid else "pending_explicit_grid")
            out.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=out.parent))
            initial_payload = report_payload(report)
            _write_json(stage / "import-report.json", initial_payload)
            emit({"event": "resource_inventory", "models": [{"model_id": m.model_id,
                  "resource_estimate": initial_payload["inventory"]["models"][i]["resource_estimate"],
                  "roles": list(m.roles), "electrode_layers": list(m.electrode_layers), "cno_layers": list(m.cno_layers)} for i, m in enumerate(inventory.models)],
                  "role_union": inventory.role_union, "electrode_layer_union": initial_payload["inventory"]["electrode_layer_union"],
                  "cno_layer_union": initial_payload["inventory"]["cno_layer_union"],
                  "unique_roles": inventory.unique_roles, "unique_electrode_layers": initial_payload["inventory"]["unique_electrode_layers"],
                  "unique_cno_layers": initial_payload["inventory"]["unique_cno_layers"]})
            evidence, preview_rows, sections = [], [], []
            grid_error = False
            grids = tuple(sorted(set(options.inspect_grids)))
            if not options.no_preview:
                (stage / "preview").mkdir()
            for index, (model, result) in enumerate(zip(inventory.models, extraction), 1):
                progress("inspect_preview", stage="model", state="started", model_id=model.model_id,
                         model_index=index, model_count=len(inventory.models))
                model_evidence, model_diagnostics, failed, rows, section = _inspect_preview_model(
                    root, model, result, decoder_policy, grids, stage, options.no_preview,
                    lambda **record: progress("inspect_preview", **record))
                evidence.extend(model_evidence)
                diagnostics.extend(model_diagnostics)
                grid_error |= failed
                preview_rows.extend(rows)
                if section is not None:
                    sections.append(section)
                progress("inspect_preview", stage="model", state="completed", model_id=model.model_id,
                         model_index=index, model_count=len(inventory.models))
            if grid_error and options.export:
                pending.append("inspect_grid_evaluation_failed")
            if options.no_preview:
                diagnostics.append(SourceIssue(".", "preview_skipped_by_request: --no-preview"))
                progress("inspect_preview", stage="preview", state="skipped", reason="by_request")
            report = replace(report, grid_evidence=tuple(evidence), diagnostics=tuple(diagnostics),
                             preview=tuple(preview_rows), pending_prerequisites=tuple(pending),
                             export_status="blocked" if pending and options.export else report.export_status)
            if not options.no_preview:
                from .png_preview import write_preview_index
                progress("inspect_preview", stage="preview_index", state="started")
                write_preview_index(stage, inventory, extraction, sections, report=report)
                progress("inspect_preview", stage="preview_index", state="completed")
            if options.export and not pending:
                (stage / "assets").mkdir()
                entries, hashes, screening_layers = [], [], {}
                mapping = {m.model_id: m for m in metadata.models}
                for index, (model, result) in enumerate(zip(inventory.models, extraction), 1):
                    progress("export", stage="model", state="started", model_id=model.model_id,
                             model_index=index, model_count=len(inventory.models))
                    raw, asset_hashes = _export_model(stage, root, model, result, mapping[model.model_id], decoder_policy,
                                                     lambda **record: progress("export", **record))
                    entries.append(raw); hashes.extend(asset_hashes)
                    layer_record = _write_screening_layers(stage, root, model, decoder_policy)
                    if layer_record is not None:
                        screening_layers[model.model_id] = layer_record
                    progress("export", stage="model", state="completed", model_id=model.model_id,
                             model_index=index, model_count=len(inventory.models))
                if screening_layers:
                    _write_json(stage / SCREENING_LAYERS_FILE,
                                {"schema_version": SCREENING_LAYERS_SCHEMA_VERSION, "products": screening_layers})
                manifest = {"schema_version": SCHEMA_VERSION, "candidate_grid": options.candidate_grid, "products": entries}
                pending_manifest = stage / "manifest.pending.json"
                _write_json(pending_manifest, manifest)
                identities, export_evidence = [], []
                progress("validation", stage="validation", state="started")
                with closing(_iter_manifest_products(pending_manifest)) as products:
                    index = 0
                    # enumerate caches its last tuple, retaining the previous
                    # Product during the next loader allocation. Count scalars.
                    for product in products:
                        index += 1
                        progress("validation", stage="model", state="started", model_id=product.product_id,
                                 model_index=index, model_count=len(entries))
                        validate_product(product)
                        identities.append((product.product_id, input_identity(product)))
                        export_evidence.append({"model_id": product.product_id, "purpose": "export_candidate_not_approved",
                                                "transform": frame_transform(product.frame.shape_yx, options.candidate_grid, product.frame.pixel_pitch_um).to_record(),
                                                "grid_check": product.grid_check})
                        model_id = product.product_id
                        del product
                        progress("validation", stage="model", state="completed", model_id=model_id,
                                 model_index=index, model_count=len(entries))
                identities = tuple(sorted(identities))
                export_evidence = tuple(export_evidence)
                progress("validation", stage="validation", state="completed")
                pending_manifest.rename(stage / "manifest.json")
                report = replace(report, export_status="exported", generated_manifest_sha256=file_sha256(stage / "manifest.json"),
                                 generated_asset_hashes=tuple(sorted(hashes)), product_input_identities=identities,
                                 grid_evidence=tuple(evidence) + export_evidence)
            _write_json(stage / "import-report.json", report_payload(report))
            _seal_bundle(stage)
            verify_bundle(stage)
            _publish_noreplace(stage, out)
            published = True
            status = 2 if (options.export and pending) or order_invalid or not inventory.valid or metadata_error or grid_error else 0
            summary = {"event": "completed", "output": str(out), "export_status": report.export_status,
                       "source_valid": inventory.valid, "cnt_mode": report.cnt_mode, "pending_prerequisites": pending}
            progress("publication", stage="complete", state="completed")
            emit(summary)
            return summary, status
    except BaseException as exc:
        if published:
            message = (f"bundle was published at {out} and verified, but progress/event "
                       f"transport failed: {exc}")
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                _add_note(exc, message)
                raise
            raise RuntimeError(message) from exc
        if stage is not None:
            try:
                _failure_evidence(stage, report, exc)
            except Exception as failure_error:
                _add_note(exc, "failure evidence could not be updated: " + str(failure_error))
                _disable_manifest(stage)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            _add_note(exc, f"unpublished staging evidence retained at {stage}")
            raise
        if stage is None:
            raise
        raise RuntimeError(f"import failed; unpublished staging evidence retained at {stage}: {exc}") from exc


def _positive(text):
    try:
        number = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("positive integer required") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("positive integer required")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline PNG inspection; export needs explicit metadata and candidate grid; no grid defaults")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path, help="new destination outside input tree; existing paths are refused")
    parser.add_argument("--export", action="store_true", help="all-model export; missing metadata/grid gives diagnostic report and nonzero exit, no manifest")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--stack-order", type=Path, help="optional reference order enables CNT relations; without it CNT is exported as an image only")
    parser.add_argument("--candidate-grid", type=_positive)
    parser.add_argument("--inspect-grids", type=_positive, nargs="+", default=[])
    parser.add_argument("--png-max-image-pixels", type=_positive, help="explicit finite Pillow pixel guard override, recorded; never disables guard")
    parser.add_argument("--no-preview", action="store_true", help="skip preview rasters and HTML; preserve all source, extraction and grid evidence")
    parser.add_argument("--verify-bundle", type=Path, help="verify published hashes without source PNGs")
    args = parser.parse_args(argv)
    if args.verify_bundle and (args.root or args.out or args.export or args.metadata or args.stack_order or args.candidate_grid or args.inspect_grids or args.png_max_image_pixels or args.no_preview):
        parser.error("--verify-bundle must be used alone")
    options = ImportOptions(root=args.root, out=args.out, export=args.export, metadata=args.metadata,
                            candidate_grid=args.candidate_grid, inspect_grids=tuple(args.inspect_grids),
                            png_max_image_pixels=args.png_max_image_pixels, verify_bundle=args.verify_bundle,
                            stack_order=args.stack_order, no_preview=args.no_preview)
    try:
        summary, code = run_import(options, event_sink=lambda event: print(json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True),
                                   progress_sink=lambda event: print(json.dumps(event, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True))
        if args.verify_bundle:
            print(json.dumps(summary, sort_keys=True))
        return code
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
