"""Offline PNG inspection/export; strict v1 manifest plus versioned provenance."""
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

from .data import frame_transform, grid_check, input_identity, load_manifest
from .png_connections import element_id, extract_inventory, iter_connections, to_product
from .png_contract import (ImportOptions, ImportReport, SourceIssue, canonical_id,
                           require_report_version)
from .png_source import decoder_session, discover_source, load_metadata, report_payload, load_stack_order
from .schema import SCHEMA_VERSION, validate_product

BUNDLE_SCHEMA_VERSION = "dl1_png_bundle_v3"


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
    elif "manifest.json" in files or any(p.startswith("assets/") for p in files):
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
            exc.add_note("failure evidence could not be updated: " + str(evidence_error))
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
            key = tuple(sorted((a.kind, a.electrode or "", a.stack or 0, a.layer, a.mask_sha256) for a in model.assets))
            model_groups.setdefault(key, []).append(model.model_id)
    result.extend({"kind": "identical_model_geometry", "model_ids": sorted(names),
                   "meaning": "possible_in_root_symlink_alias_or_duplicate_product_geometry_not_order_equivalence"}
                  for _, names in sorted(model_groups.items()) if len(names) > 1)
    return tuple(result)


def _write_assets(stage, model, extraction, metadata):
    """One bool connection mask at a time; staged legacy validation remains eager."""
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
    for a in sorted(model.electrodes, key=lambda a: (a.stack, a.electrode, a.layer)):
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


def run_import(options, *, event_sink=None):
    """Build in a new sibling stage; leave failed stages for explicit inspection."""
    if options.verify_bundle is not None:
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
    def emit(payload):
        if event_sink is not None:
            event_sink(payload)
    try:
        # Nested source session retains the explicit guard for the whole import.
        with decoder_session(options.png_max_image_pixels) as decoder_policy:
            reference = load_stack_order(options.stack_order)
            inventory, extraction = extract_inventory(discover_source(root, png_max_image_pixels=options.png_max_image_pixels, stack_order=reference))
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
            if any(m.stack_order.status == "required" for m in inventory.models):
                pending.append("explicit_stack_order_missing")
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
            duplicates = _duplicates(inventory)
            diagnostics.extend(SourceIssue(".", "possible duplicate/alias geometry: " + str(entry)) for entry in duplicates)
            report = ImportReport(inventory, metadata, stack_order=reference, extraction=extraction,
                                  extraction_status=("extractor_ran" if all(r.extraction_status == "complete" for r in extraction) else
                                                     "blocked" if all(r.extraction_status == "blocked" for r in extraction) else "partial"),
                                  candidate_grid=options.candidate_grid, pending_prerequisites=tuple(pending),
                                  diagnostics=tuple(diagnostics), duplicate_geometry=duplicates,
                                  status_summary=tuple({"model_id": r.model_id,
                                      "extractor_run": "ran" if r.extraction_status == "complete" else "blocked",
                                      "extraction_coverage": r.connectivity_observation,
                                      "model_connectivity_status": "available" if r.connectivity_observation == "complete" else r.connectivity_observation,
                                      "model_vocabulary_eligibility": "not_evaluated_without_checkpoint",
                                      "electrical_verification": "not_performed"} for r in extraction),
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
            evidence = []
            grid_error = False
            for model, result in zip(inventory.models, extraction):
                if result.extraction_status == "blocked" and options.inspect_grids:
                    evidence.append({"model_id": model.model_id, "purpose": "inspect_not_evaluated",
                                     "reason": "stack_order_required" if model.stack_order.status == "required" else "invalid_stack_order"})
                elif model.valid and options.inspect_grids:
                    product = None
                    try:
                        product = to_product(model, result)
                        for grid in sorted(set(options.inspect_grids)):
                            evidence.append({"model_id": model.model_id, "purpose": "inspect_only_not_approved",
                                             "transform": frame_transform(model.frame.shape_yx, grid, model.frame.pixel_pitch_um).to_record(), "grid_check": grid_check(product, grid)})
                    except Exception as exc:
                        grid_error = True
                        diagnostics.append(SourceIssue(model.model_id, "inspect grid evaluation failed: " + str(exc), type(exc).__name__))
                        evidence.append({"model_id": model.model_id, "purpose": "inspect_failed_not_approved", "error": str(exc)})
                    finally:
                        del product
            if grid_error and options.export:
                pending.append("inspect_grid_evaluation_failed")
            report = replace(report, grid_evidence=tuple(evidence), diagnostics=tuple(diagnostics),
                             pending_prerequisites=tuple(pending), export_status="blocked" if pending and options.export else report.export_status)
            from .png_preview import write_preview
            preview = write_preview(stage, inventory, extraction, report=report)
            report = replace(report, preview=preview)
            if options.export and not pending:
                (stage / "assets").mkdir()
                entries, hashes = [], []
                mapping = {m.model_id: m for m in metadata.models}
                for model, result in zip(inventory.models, extraction):
                    raw, asset_hashes = _write_assets(stage, model, result, mapping[model.model_id])
                    entries.append(raw); hashes.extend(asset_hashes)
                manifest = {"schema_version": SCHEMA_VERSION, "candidate_grid": options.candidate_grid, "products": entries}
                pending_manifest = stage / "manifest.pending.json"
                _write_json(pending_manifest, manifest)
                loaded = load_manifest(pending_manifest)
                for product in loaded:
                    validate_product(product)
                identities = tuple(sorted((p.product_id, input_identity(p)) for p in loaded))
                export_evidence = tuple({"model_id": p.product_id, "purpose": "export_candidate_not_approved",
                                         "transform": frame_transform(p.frame.shape_yx, options.candidate_grid, p.frame.pixel_pitch_um).to_record(), "grid_check": p.grid_check}
                                        for p in loaded)
                del loaded
                pending_manifest.rename(stage / "manifest.json")
                report = replace(report, export_status="exported", generated_manifest_sha256=file_sha256(stage / "manifest.json"),
                                 generated_asset_hashes=tuple(sorted(hashes)), product_input_identities=identities,
                                 grid_evidence=tuple(evidence) + export_evidence)
            _write_json(stage / "import-report.json", report_payload(report))
            _seal_bundle(stage)
            verify_bundle(stage)
            _publish_noreplace(stage, out)
            status = 2 if (options.export and pending) or order_invalid or not inventory.valid or metadata_error or grid_error else 0
            summary = {"event": "completed", "output": str(out), "export_status": report.export_status,
                       "source_valid": inventory.valid, "pending_prerequisites": pending}
            emit(summary)
            return summary, status
    except BaseException as exc:
        if stage is not None:
            try:
                _failure_evidence(stage, report, exc)
            except Exception as failure_error:
                exc.add_note("failure evidence could not be updated: " + str(failure_error))
                _disable_manifest(stage)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            exc.add_note(f"unpublished staging evidence retained at {stage}")
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
    parser.add_argument("--stack-order", type=Path, help="user-supplied reference order; required for CNT export, optional diagnostic inspection")
    parser.add_argument("--candidate-grid", type=_positive)
    parser.add_argument("--inspect-grids", type=_positive, nargs="+", default=[])
    parser.add_argument("--png-max-image-pixels", type=_positive, help="explicit finite Pillow pixel guard override, recorded; never disables guard")
    parser.add_argument("--verify-bundle", type=Path, help="verify published hashes without source PNGs")
    args = parser.parse_args(argv)
    if args.verify_bundle and (args.root or args.out or args.export or args.metadata or args.stack_order or args.candidate_grid or args.inspect_grids or args.png_max_image_pixels):
        parser.error("--verify-bundle must be used alone")
    options = ImportOptions(root=args.root, out=args.out, export=args.export, metadata=args.metadata,
                            candidate_grid=args.candidate_grid, inspect_grids=tuple(args.inspect_grids),
                            png_max_image_pixels=args.png_max_image_pixels, verify_bundle=args.verify_bundle,
                            stack_order=args.stack_order)
    try:
        summary, code = run_import(options, event_sink=lambda event: print(json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True))
        if args.verify_bundle:
            print(json.dumps(summary, sort_keys=True))
        return code
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
