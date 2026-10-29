"""Merge two exported PNG bundles into one round bundle; assets keep their hashes.

The merged bundle is a new folder: manifest.json (assets under assets/rNN/),
merge-report.json and bundle-integrity.json. Verification is hash integrity plus a
full strict manifest read; it is not electrical verification.
"""
from __future__ import annotations

from contextlib import closing
import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import sys

import numpy as np

from .data import _iter_manifest_products
from .import_png import (BUNDLE_SCHEMA_VERSION, _read_json, _seal_bundle, _write_json,
                         bundle_path, file_sha256, verify_bundle)
from .schema import SCHEMA_VERSION

MERGE_SCHEMA_VERSION = "dl1_merge_bundle_v1"
VERIFICATION_MEANING = "bundle_hash_integrity_not_electrical_verification"
_ASSET_ROUND = re.compile(r"assets/r([0-9]{2,})/[^/]+")
_REPORT_FIELDS = {"merge_schema_version", "round", "candidate_grid", "cnt_mode", "schema_version", "base", "add",
                  "asset_method", "assets", "generated_manifest_sha256", "product_count"}


def _asset_paths(product):
    paths = [product["valid"]] + ([product["cnt"]] if product.get("cnt") else [])
    paths += [e["mask"] for e in product["elements"]] + [c["mask"] for c in product["connections"]]
    return [p for p in paths if p is not None]


def _remap_product(product, remap):
    product = copy.deepcopy(product)
    product["valid"] = remap(product["valid"])
    if product.get("cnt"):
        product["cnt"] = remap(product["cnt"])
    for item in product["elements"] + product["connections"]:
        if item["mask"] is not None:
            item["mask"] = remap(item["mask"])
    return product


def _describe(bundle, role):
    """Verify one input bundle and return the facts the merge needs."""
    root = Path(bundle).resolve(strict=True)
    if (root / "merge-report.json").is_file():
        kind = "merge"
        if role == "add":
            raise ValueError("--add must be an import bundle, not a merged bundle")
        verify_merged_bundle(root)
        report = _read_json(root / "merge-report.json")
        cnt_mode, rounds = report["cnt_mode"], report["round"]
    elif (root / "import-report.json").is_file():
        kind = "import"
        status = verify_bundle(root)
        if status["export_status"] != "exported":
            raise ValueError(f"{role} bundle was not exported: {status['export_status']}")
        report = _read_json(root / "import-report.json")
        cnt_mode, rounds = report["cnt_mode"], None
    else:
        raise ValueError(f"{role} is neither an import bundle nor a merged bundle: {root}")
    integrity = _read_json(root / "bundle-integrity.json")
    manifest = _read_json(root / "manifest.json")
    return {"root": root, "kind": kind, "round": rounds, "cnt_mode": cnt_mode, "manifest": manifest,
            "files": integrity["files"], "manifest_sha256": integrity["files"]["manifest.json"],
            "integrity_sha256": file_sha256(root / "bundle-integrity.json"),
            "product_ids": [p["product_id"] for p in manifest["products"]]}


def _check_compatible(base, add):
    clash = sorted(set(base["product_ids"]) & set(add["product_ids"]))
    if clash:
        raise ValueError(f"duplicate product IDs in base and add: {clash}")
    for label, left, right in (("candidate_grid", base["manifest"]["candidate_grid"], add["manifest"]["candidate_grid"]),
                               ("cnt_mode", base["cnt_mode"], add["cnt_mode"]),
                               ("schema_version", base["manifest"]["schema_version"], add["manifest"]["schema_version"])):
        if left != right:
            raise ValueError(f"{label} differs: base={left!r} add={right!r}")


def _place(source, destination):
    """Hard link when possible, else copy; return the method used."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copyfile(source, destination)
        return "copy"


def _disable(path):
    """Move a final file aside (never delete) so a failed merge has no sealed look."""
    if path.exists():
        target = path.with_name(path.stem + ".failed" + path.suffix)
        counter = 1
        while target.exists():
            counter += 1
            target = path.with_name(f"{path.stem}.failed{counter}{path.suffix}")
        path.rename(target)


def merge_bundles(base_dir, add_dir, out_dir, *, round_index, base_round=None):
    if type(round_index) is not int or round_index < 1:
        raise ValueError("round must be an integer >= 1")
    base, add = _describe(base_dir, "base"), _describe(add_dir, "add")
    if base["kind"] == "import":
        if type(base_round) is not int or base_round < 1:
            raise ValueError("--base-round is required for an import base bundle")
        base["round"] = base_round
    elif base_round is not None:
        raise ValueError("--base-round only applies to an import base bundle")
    if round_index <= base["round"]:
        raise ValueError(f"round {round_index} must be greater than the base round {base['round']}")
    _check_compatible(base, add)
    out = Path(out_dir)
    out.mkdir(exist_ok=False)
    out = out.resolve()

    products, assets = [], []
    methods, placed = set(), set()
    for label, side in (("base", base), ("add", add)):
        folder = None
        if side["kind"] == "import":
            folder = f"assets/r{side['round'] if label == 'base' else round_index:02d}/"
        for product in side["manifest"]["products"]:
            def remap(source, side=side, label=label, folder=folder):
                if not source.startswith("assets/"):
                    raise ValueError(f"unexpected asset path: {source}")
                destination = source if folder is None else folder + source[len("assets/"):]
                if destination in placed:
                    return destination
                method = _place(bundle_path(side["root"], source), bundle_path(out, destination))
                digest = file_sha256(bundle_path(out, destination))
                if digest != side["files"].get(source):
                    raise ValueError(f"merged asset hash differs from its source: {source}")
                methods.add(method)
                placed.add(destination)
                assets.append({"path": destination, "source_bundle": label, "source_path": source,
                               "sha256": digest, "asset_method": method})
                return destination
            products.append(_remap_product(product, remap))
    assets.sort(key=lambda item: item["path"])
    manifest = {"schema_version": base["manifest"]["schema_version"],
                "candidate_grid": base["manifest"]["candidate_grid"], "products": products}
    _write_json(out / "manifest.pending.json", manifest)
    manifest_sha = file_sha256(out / "manifest.pending.json")
    method = methods.pop() if len(methods) == 1 else "mixed"
    side_report = lambda side: {"path": str(side["root"]), "kind": side["kind"], "round": side["round"],
                                "manifest_sha256": side["manifest_sha256"],
                                "integrity_sha256": side["integrity_sha256"], "product_ids": side["product_ids"]}
    report = {"merge_schema_version": MERGE_SCHEMA_VERSION, "round": round_index,
              "candidate_grid": base["manifest"]["candidate_grid"], "cnt_mode": base["cnt_mode"],
              "schema_version": base["manifest"]["schema_version"], "base": side_report(base), "add": side_report(add),
              "asset_method": method, "assets": assets, "generated_manifest_sha256": manifest_sha,
              "product_count": len(products)}
    report["add"].pop("round")
    try:
        _write_json(out / "merge-report.json", report)
        (out / "manifest.pending.json").rename(out / "manifest.json")
        _seal_bundle(out)
        result = verify_merged_bundle(out)
    except BaseException:
        _disable(out / "manifest.json")
        _disable(out / "manifest.pending.json")
        _disable(out / "bundle-integrity.json")
        raise
    return result


def merge_inputs_match(out_dir, base_dir, add_dir, round_index):
    """None when the merged folder was built from exactly these inputs, else the reason."""
    try:
        report = _read_json(Path(out_dir) / "merge-report.json")
        if report["round"] != round_index:
            return f"merge-report round {report['round']} differs from {round_index}"
        for label, folder in (("base", base_dir), ("add", add_dir)):
            folder = Path(folder)
            for key, name in (("manifest_sha256", "manifest.json"), ("integrity_sha256", "bundle-integrity.json")):
                current = file_sha256(folder / name)
                if report[label][key] != current:
                    return f"{label} {key} differs from the current {folder / name}"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"merge-report cannot be compared: {exc}"
    return None


def _verify_merged_bundle(bundle):
    root = Path(bundle).resolve(strict=True)
    integrity = _read_json(bundle_path(root, "bundle-integrity.json"))
    if not isinstance(integrity, dict) or set(integrity) != {"bundle_schema_version", "hash_algorithm", "files", "directories"}:
        raise ValueError("invalid bundle integrity fields")
    if integrity["bundle_schema_version"] != BUNDLE_SCHEMA_VERSION or integrity["hash_algorithm"] != "sha256":
        raise ValueError("unsupported bundle integrity version/algorithm")
    files = integrity["files"]
    if not isinstance(files, dict) or not {"merge-report.json", "manifest.json"} <= set(files):
        raise ValueError("merged bundle integrity requires merge-report.json and manifest.json")
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
        if not bundle_path(root, relative).is_dir():
            raise ValueError("missing bundle directory")
    if actual != set(files) | set(directories) | {"bundle-integrity.json"}:
        raise ValueError("unlisted or missing bundle files")
    report = _read_json(bundle_path(root, "merge-report.json"))
    if not isinstance(report, dict) or set(report) != _REPORT_FIELDS or report["merge_schema_version"] != MERGE_SCHEMA_VERSION:
        raise ValueError("invalid merge report fields")
    if type(report["round"]) is not int or report["round"] < 1 or report["generated_manifest_sha256"] != files["manifest.json"]:
        raise ValueError("manifest/report hash link mismatch")
    manifest = _read_json(bundle_path(root, "manifest.json"))
    if set(manifest) != {"schema_version", "candidate_grid", "products"} or manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("invalid merged manifest contract")
    listed = {}
    for item in report["assets"]:
        if item["path"] in listed:
            raise ValueError("duplicate reported merged asset")
        listed[item["path"]] = item["sha256"]
    required = set()
    for product in manifest["products"]:
        for relative in _asset_paths(product):
            required.add(relative)
            match = _ASSET_ROUND.fullmatch(relative)
            if not match or int(match.group(1)) > report["round"]:
                raise ValueError(f"merged asset is not under assets/rNN/: {relative}")
            if files.get(relative) != listed.get(relative):
                raise ValueError("manifest asset/report hash link mismatch")
            mask = np.load(bundle_path(root, relative), mmap_mode="r", allow_pickle=False)
            if mask.dtype != np.bool_ or tuple(mask.shape) != tuple(product["frame"]["shape_yx"]):
                raise ValueError("generated mask dtype/shape mismatch")
            del mask
    if set(listed) != required or {p for p in files if p.startswith("assets/")} != required:
        raise ValueError("unreferenced or missing merged assets")
    ids = [p["product_id"] for p in manifest["products"]]
    if ids != report["base"]["product_ids"] + report["add"]["product_ids"] or report["product_count"] != len(ids):
        raise ValueError("merge report product list differs from the manifest")
    with closing(_iter_manifest_products(root / "manifest.json")) as iterator:
        for _ in iterator:
            pass
    return {"status": "verified", "round": report["round"], "product_count": len(ids),
            "verification_meaning": VERIFICATION_MEANING}


def verify_merged_bundle(bundle):
    """Malformed evidence always produces a clean diagnostic, never a traceback."""
    try:
        return _verify_merged_bundle(bundle)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as exc:
        raise ValueError("invalid or incomplete merged bundle evidence: " + str(exc)) from exc


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False,
                                     description="Merge two exported PNG bundles into a new round bundle, or verify one")
    parser.add_argument("--base", type=Path)
    parser.add_argument("--add", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--round", type=int)
    parser.add_argument("--base-round", type=int, help="round number of an import base bundle (usually 1)")
    parser.add_argument("--verify", type=Path, help="verify a merged bundle; use alone")
    args = parser.parse_args(argv)
    try:
        if args.verify is not None:
            if any(v is not None for v in (args.base, args.add, args.out, args.round, args.base_round)):
                parser.error("--verify must be used alone")
            result = verify_merged_bundle(args.verify)
        else:
            if args.base is None or args.add is None or args.out is None or args.round is None:
                parser.error("--base, --add, --out and --round are required")
            result = merge_bundles(args.base, args.add, args.out, round_index=args.round, base_round=args.base_round)
    except (ValueError, OSError) as exc:
        print(f"merge_bundles: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
