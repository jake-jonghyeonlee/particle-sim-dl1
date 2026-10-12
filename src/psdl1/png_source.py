"""Read-only, original-resolution PNG inventory. Pillow is imported on demand.

Decoding is serialized because Pillow's pixel guard is process-global. Neither
geometry nor source names are resampled, registered, thresholded or aliased.
"""
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path
import re
import struct
import threading
import warnings

import numpy as np

from .png_contract import (DecoderPolicy, FrameRecord, ImportReport, METADATA_SCHEMA_VERSION,
                           MetadataRecord, ModelMetadata, ModelSource, ResourceEstimate,
                           SourceAsset, SourceInventory, SourceIssue, canonical_id, mask_hash)
from .schema import identifier

_DECODER_LOCK = threading.RLock()


def _pillow():
    try:
        from PIL import Image, __version__
    except ImportError as exc:
        raise ImportError("PNG input requires the optional psdl1[png] dependency (Pillow); "
                          "install from your approved offline wheels. NPY input needs no Pillow.") from exc
    return Image, __version__


@contextmanager
def decoder_session(png_max_image_pixels=None):
    """Retain installed default, or explicitly override for this serial session."""
    if png_max_image_pixels is not None and (type(png_max_image_pixels) is not int
                                             or png_max_image_pixels <= 0):
        raise ValueError("png_max_image_pixels: finite positive integer required")
    Image, version = _pillow()
    from PIL import ImageFile
    with _DECODER_LOCK:
        previous = Image.MAX_IMAGE_PIXELS
        previous_truncated = ImageFile.LOAD_TRUNCATED_IMAGES
        effective = previous if png_max_image_pixels is None else png_max_image_pixels
        if type(effective) is not int or effective <= 0:
            raise ValueError("Pillow pixel guard is disabled/invalid; supply an explicit positive override")
        try:
            Image.MAX_IMAGE_PIXELS = effective
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                yield DecoderPolicy(version, previous, png_max_image_pixels, effective,
                                    png_max_image_pixels is not None)
        finally:
            Image.MAX_IMAGE_PIXELS = previous
            ImageFile.LOAD_TRUNCATED_IMAGES = previous_truncated


class PNGSourceError(ValueError):
    def __init__(self, issue):
        self.issue = issue
        super().__init__(f"{issue.path}: {issue.reason}")


def parse_png_name(filename: str) -> tuple[int, str]:
    match = re.fullmatch(r"([0-9]+)[-_](.+)\.png", filename)
    if match is None or int(match[1]) <= 0:
        raise ValueError(f"{filename}: expected positive stack prefix and first '-'/'_' separator")
    return int(match[1]), identifier(match[2], "layer")


def _header_shape(path):
    # IHDR gives dimensions even when Pillow raises before returning an Image.
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) == 24 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR":
        w, h = struct.unpack(">II", header[16:24])
        return h, w
    return None


def _decode(path, diagnostic_path, policy):
    Image, _ = _pillow()
    shape = None
    try:
        shape = _header_shape(path)
        with path.open("rb") as handle:
            header = handle.read(26)
        if shape is not None and len(header) >= 26 and header[24] > 8:
            raise ValueError(f"unsupported PNG bit depth: {header[24]}")
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError("file is not PNG")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("multi-frame PNG is unsupported")
            if image.getexif().get(274, 1) != 1:
                raise ValueError("nonidentity orientation metadata is unsupported")
            mode = image.mode
            if mode not in {"1", "L", "RGB", "LA", "RGBA", "P"}:
                raise ValueError(f"unsupported PNG mode/bit depth: {mode}")
            image.load()  # fail on corrupt/truncated pixel streams before conversion
            # Exact palette/alpha expansion only; never luminance conversion.
            rgba = np.asarray(image.convert("RGBA"))
            if not np.all(rgba[..., 3] == 255):
                raise ValueError("nonopaque PNG pixels are unsupported")
            rgb = rgba[..., :3]
            black = np.all(rgb == 0, axis=2)
            white = np.all(rgb == 255, axis=2)
            if not np.all(black | white):
                raise ValueError("pixels must be exactly black or white (no grey/colour threshold)")
            mask = np.ascontiguousarray(black, dtype=np.bool_)
            mask.flags.writeable = False
            return mask, mode
    except Exception as exc:
        raise PNGSourceError(SourceIssue(diagnostic_path, str(exc), type(exc).__name__, shape,
                                         shape[0] * shape[1] if shape else None,
                                         policy.effective_max_image_pixels)) from exc


def decode_png(path, *, png_max_image_pixels=None):
    """Standalone strict decoder returning (bool mask, original mode, policy)."""
    path = Path(path)
    with decoder_session(png_max_image_pixels) as policy:
        mask, mode = _decode(path, str(path), policy)
        return mask, mode, policy


def discover_source(root, *, png_max_image_pixels=None) -> SourceInventory:
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"{root}: source root must be a directory")
    models, errors, notices = [], [], []
    emitted_ids = {}

    def rel(path):
        return path.relative_to(root).as_posix()

    def safe(path, issues):
        try:
            path.resolve(strict=True).relative_to(root)
            return True
        except (OSError, ValueError, RuntimeError):
            issues.append(SourceIssue(rel(path), "escaping or broken source symlink/path"))
            return False

    def claim_id(value, source_path, issues):
        if value in emitted_ids:
            issues.append(SourceIssue(source_path, f"duplicate/colliding emitted identity; also {emitted_ids[value]}"))
        else:
            emitted_ids[value] = source_path

    def unexpected(path, issues, warning_list):
        # Classify PNG at any unsupported depth as an error; unrelated files warn.
        if path.is_dir():
            issues.append(SourceIssue(rel(path), "unsupported directory layout"))
            if path.is_symlink():
                return  # do not recurse through a directory alias or symlink cycle
            for nested in sorted(path.iterdir()):
                if safe(nested, issues):
                    unexpected(nested, issues, warning_list)
        elif path.suffix.lower() == ".png":
            issues.append(SourceIssue(rel(path), "PNG at unsupported layout/depth"))
        else:
            warning_list.append(SourceIssue(rel(path), "ignored unrelated non-PNG file"))

    with decoder_session(png_max_image_pixels) as policy:
        for model_dir in sorted(root.iterdir()):
            if not safe(model_dir, errors):
                continue
            if not model_dir.is_dir():
                unexpected(model_dir, errors, notices)
                continue
            model_errors, model_warnings, assets = [], [], []
            try:
                identifier(model_dir.name, "model")
            except ValueError as exc:
                model_errors.append(SourceIssue(rel(model_dir), str(exc)))
            stacks, stack_paths, bindings = {}, {}, {}
            cno_present = False
            for folder in sorted(model_dir.iterdir()):
                if not safe(folder, model_errors):
                    continue
                if not folder.is_dir():
                    unexpected(folder, model_errors, model_warnings)
                    continue
                try:
                    identifier(folder.name, "electrode")
                except ValueError as exc:
                    model_errors.append(SourceIssue(rel(folder), str(exc)))
                is_cnt = folder.name == "cno"
                cno_present |= is_cnt
                for path in sorted(folder.iterdir()):
                    if not safe(path, model_errors):
                        continue
                    if path.is_dir() or path.suffix.lower() != ".png":
                        unexpected(path, model_errors, model_warnings)
                        continue
                    try:
                        stack, layer = parse_png_name(path.name)
                    except ValueError as exc:
                        model_errors.append(SourceIssue(rel(path), str(exc)))
                        continue
                    binding = (folder.name, stack, layer) if is_cnt else (folder.name, layer)
                    if binding in bindings:
                        model_errors.append(SourceIssue(rel(path), f"duplicate normalized binding; also {bindings[binding]}"))
                        model_errors.append(SourceIssue(bindings[binding], f"duplicate normalized binding; also {rel(path)}"))
                    else:
                        bindings[binding] = rel(path)
                    if stack in stacks and stacks[stack] != layer:
                        model_errors.append(SourceIssue(rel(path), f"conflicting stack {stack}; also {stack_paths[stack]} ({stacks[stack]})"))
                    else:
                        stacks[stack], stack_paths[stack] = layer, rel(path)
                    try:
                        mask, mode = _decode(path, rel(path), policy)
                        source_binding = (model_dir.name, "cno" if is_cnt else "electrode",
                                          None if is_cnt else folder.name, stack, layer)
                        asset_id = canonical_id("asset", source_binding)
                        claim_id(asset_id, rel(path), model_errors)
                        assets.append(SourceAsset(
                            asset_id, model_dir.name, source_binding[1],
                            source_binding[2], source_binding[2], stack, layer, rel(path), path.name,
                            hashlib.sha256(path.read_bytes()).hexdigest(), mask_hash(mask), mode,
                            tuple(mask.shape), mask))
                    except PNGSourceError as exc:
                        model_errors.append(exc.issue)
            assets.sort(key=lambda a: (a.stack, a.kind, a.electrode or "", a.layer, a.relative_path))
            frame = None
            if assets:
                h, w = assets[0].shape_yx
                for a in assets:
                    if a.shape_yx != (h, w):
                        model_errors.append(SourceIssue(a.relative_path, f"shape mismatch with {assets[0].relative_path}"))
                    if min(a.shape_yx) <= 1000:
                        model_errors.append(SourceIssue(a.relative_path, "source frame requires W>1000 and H>1000"))
                frame = FrameRecord(canonical_id("frame", [model_dir.name, h, w, 0.1, [0, 0]]),
                                    (h, w), (w * 0.1, h * 0.1), (500, 500, w - 500, h - 500))
                claim_id(frame.frame_id, rel(model_dir), model_errors)
            if not any(a.kind == "electrode" for a in assets):
                model_errors.append(SourceIssue(rel(model_dir), "no decoded electrode assets"))
            repeated = tuple(sorted(layer for layer in set(stacks.values())
                                    if list(stacks.values()).count(layer) > 1))
            if repeated:
                model_warnings.append(SourceIssue(rel(model_dir), "actual layer names repeated at distinct stacks: " + ", ".join(repeated)))
            estimate = ResourceEstimate(unknown_reasons=("invalid_source",) if model_errors else ("extraction_not_run",))
            models.append(ModelSource(model_dir.name, tuple(assets), tuple(sorted(stacks.items())),
                                      tuple(sorted({a.role for a in assets if a.role is not None})),
                                      tuple(sorted({a.layer for a in assets})), frame, cno_present,
                                      repeated, tuple(model_errors), tuple(model_warnings), estimate))
    if not models:
        errors.append(SourceIssue(".", "no source models discovered"))
    def names(field):
        counts = {}
        for model in models:
            for name in getattr(model, field):
                counts[name] = counts.get(name, 0) + 1
        return tuple(sorted(counts)), tuple(sorted(n for n, c in counts.items() if c == 1))
    roles, unique_roles = names("roles")
    layers, unique_layers = names("layers")
    return SourceInventory(tuple(models), policy, roles, layers, unique_roles, unique_layers,
                           tuple(errors), tuple(notices))


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate metadata key: {key}")
        result[key] = value
    return result


def load_metadata(path, model_names, *, require_complete=False) -> MetadataRecord:
    known = set(model_names)
    if path is None:
        result = MetadataRecord(pending_models=tuple(sorted(known)))
    else:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_strict_object,
                             parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"invalid JSON constant: {x}")))
            if not isinstance(doc, dict) or set(doc) != {"metadata_schema_version", "models"}:
                raise ValueError("metadata requires exactly metadata_schema_version and models")
            if doc["metadata_schema_version"] != METADATA_SCHEMA_VERSION:
                raise ValueError("unsupported metadata_schema_version")
            mapping = doc["models"]
            if not isinstance(mapping, dict) or set(mapping) - known:
                raise ValueError("models must be a mapping with only discovered model keys")
            entries = []
            for name, values in sorted(mapping.items()):
                identifier(name, "model")
                if not isinstance(values, dict) or set(values) != {"lineage_id", "design_version"}:
                    raise ValueError(f"{name}: expected exactly lineage_id and design_version")
                entries.append(ModelMetadata(name, identifier(values["lineage_id"], "lineage_id"),
                                             identifier(values["design_version"], "design_version")))
            result = MetadataRecord(tuple(entries), tuple(sorted(known - set(mapping))))
        except (ValueError, OSError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
    if require_complete and result.pending_models:
        raise ValueError("missing explicit metadata for models: " + ", ".join(result.pending_models))
    return result


def report_payload(report: ImportReport) -> dict:
    """JSON-safe sidecar evidence, intentionally excluding in-memory source masks."""
    def encode(value):
        if is_dataclass(value):
            return {f.name: encode(getattr(value, f.name)) for f in fields(value) if f.name != "mask"}
        if isinstance(value, (tuple, list)):
            return [encode(v) for v in value]
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        return value
    return encode(report)
