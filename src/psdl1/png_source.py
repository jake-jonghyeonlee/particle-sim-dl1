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
import unicodedata
import zlib

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



class _PNGScanlines:
    """Stream inflated bytes into scanlines; retain at most two packed rows.

    Palette rows are unfiltered in raw sample units before index validation.
    Other colour types need only a valid filter byte and exact scanline lengths.
    """
    def __init__(self, width, height, depth, colour, interlace):
        self.depth, self.colour = depth, colour
        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[colour]
        geometry = ((0, 0, 1, 1),) if not interlace else (
            (0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
            (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2))
        self.passes = []
        for x, y, dx, dy in geometry:
            w = max(0, (width - x + dx - 1) // dx)
            h = max(0, (height - y + dy - 1) // dy)
            if w and h:
                self.passes.append((w, h, (w * channels * depth + 7) // 8))
        self.expected = sum(h * (row_bytes + 1) for _, h, row_bytes in self.passes)
        self.received = self.pass_index = self.row_index = 0
        self.row = bytearray()
        self.previous = None
        self.palette_entries = None

    def feed(self, data):
        self.received += len(data)
        if self.received > self.expected:
            raise ValueError("IDAT scanline byte count is longer than IHDR requires")
        offset = 0
        while offset < len(data):
            width, height, size = self.passes[self.pass_index]
            take = min(size + 1 - len(self.row), len(data) - offset)
            self.row.extend(data[offset:offset + take])
            offset += take
            if len(self.row) == size + 1:
                filter_type = self.row[0]
                if filter_type > 4:
                    raise ValueError("invalid PNG scanline filter type")
                if self.colour == 3 and self.palette_entries != 1 << self.depth:
                    packed = bytearray(self.row[1:])
                    prior = self.previous
                    for i in range(len(packed)) if filter_type else ():
                        value = packed[i]
                        left = packed[i - 1] if i else 0
                        above = prior[i] if prior is not None else 0
                        upper_left = prior[i - 1] if prior is not None and i else 0
                        if filter_type == 1:
                            value += left
                        elif filter_type == 2:
                            value += above
                        elif filter_type == 3:
                            value += (left + above) // 2
                        elif filter_type == 4:
                            predictor = left + above - upper_left
                            dl, da, du = abs(predictor - left), abs(predictor - above), abs(predictor - upper_left)
                            value += left if dl <= da and dl <= du else above if da <= du else upper_left
                        packed[i] = value & 255
                    values = np.frombuffer(packed, dtype=np.uint8)
                    if self.depth == 8:
                        invalid_index = bool(np.any(values >= self.palette_entries))
                    else:
                        shifts = np.arange(8 - self.depth, -1, -self.depth, dtype=np.uint8)
                        samples = ((values[:, None] >> shifts) & ((1 << self.depth) - 1)).ravel()[:width]
                        invalid_index = bool(np.any(samples >= self.palette_entries))
                    if invalid_index:
                        raise ValueError("palette pixel index has no PLTE entry")
                    self.previous = packed
                self.row.clear()
                self.row_index += 1
                if self.row_index == height:
                    self.pass_index += 1
                    self.row_index = 0
                    self.previous = None

    def finish(self):
        if self.received != self.expected or self.row:
            raise ValueError(f"IDAT scanline byte count is short: expected {self.expected}, received {self.received}")


def _validate_png_structure(path, *, policy=None):
    """Independent chunk/CRC/critical-order/zlib/sample validation, no Pillow internals.

    Compressed chunks and inflate output are consumed in bounded blocks. No full
    compressed stream or full inflated image is retained by this check.
    """
    seen_ihdr = seen_plte = seen_idat = ended_idat = seen_trns = False
    inflater = zlib.decompressobj()
    scanlines = None
    colour = depth = None

    def inflate(block):
        while block:
            if inflater.eof:
                raise ValueError("extra compressed bytes after IDAT zlib stream")
            output = inflater.decompress(block, 65536)
            scanlines.feed(output)
            if inflater.unused_data:
                raise ValueError("extra compressed bytes after IDAT zlib stream")
            block = inflater.unconsumed_tail

    with path.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError("invalid PNG signature")
        while True:
            header = handle.read(8)
            if len(header) != 8:
                raise ValueError("PNG missing/truncated chunk or IEND")
            length, kind = struct.unpack(">I4s", header)
            if kind in {b"acTL", b"fcTL", b"fdAT"}:
                raise ValueError("APNG is unsupported")
            if length > 0x7fffffff or not all(65 <= c <= 90 or 97 <= c <= 122 for c in kind) or kind[2] & 32:
                raise ValueError("invalid PNG chunk length/type")
            if not seen_ihdr and kind != b"IHDR":
                raise ValueError("IHDR must be the first PNG chunk")
            if kind == b"IHDR" and (seen_ihdr or length != 13):
                raise ValueError("PNG requires exactly one 13-byte IHDR first")
            if kind == b"PLTE" and (seen_plte or seen_idat or colour in {0, 4} or
                                    not 0 < length <= 768 or length % 3):
                raise ValueError("invalid PLTE length/order/colour type")
            if kind == b"tRNS":
                if colour != 3:
                    raise ValueError("non-palette tRNS is nonopaque/unsupported")
                if seen_trns or seen_idat or not seen_plte or not 0 < length <= scanlines.palette_entries:
                    raise ValueError("invalid palette tRNS length/order")
            if kind == b"IDAT":
                if ended_idat or colour == 3 and not seen_plte:
                    raise ValueError("IDAT order invalid or palette PNG missing PLTE")
                seen_idat = True
            elif seen_idat:
                ended_idat = True
            if kind == b"IEND" and (length or not seen_idat):
                raise ValueError("invalid IEND length/order or missing IDAT")
            if not kind[0] & 32 and kind not in {b"IHDR", b"PLTE", b"IDAT", b"IEND"}:
                raise ValueError("unknown critical PNG chunk")
            crc = zlib.crc32(kind)
            payload = bytearray()  # only bounded IHDR/PLTE/tRNS metadata retained
            remaining = length
            while remaining:
                block = handle.read(min(remaining, 65536))
                if not block:
                    raise ValueError("truncated PNG chunk payload")
                remaining -= len(block)
                crc = zlib.crc32(block, crc)
                if kind == b"IDAT":
                    inflate(block)
                elif kind in {b"IHDR", b"PLTE", b"tRNS"}:
                    payload.extend(block)
            expected_crc = handle.read(4)
            if len(expected_crc) != 4 or struct.unpack(">I", expected_crc)[0] != crc:
                raise ValueError(f"PNG {kind.decode()} CRC mismatch/truncation")
            if kind == b"IHDR":
                width, height, depth, colour, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
                allowed = {0: {1, 2, 4, 8}, 2: {8}, 3: {1, 2, 4, 8}, 4: {8}, 6: {8}}
                if not 0 < width <= 0x7fffffff or not 0 < height <= 0x7fffffff or depth not in allowed.get(colour, set()):
                    raise ValueError("invalid/unsupported IHDR dimensions/bit depth/colour type")
                if compression != 0 or filtering != 0 or interlace not in {0, 1}:
                    raise ValueError("unsupported IHDR compression/filter/interlace method")
                if policy is not None:
                    Image, _ = _pillow()
                    pixels = width * height
                    limit = policy.effective_max_image_pixels
                    if pixels > 2 * limit:
                        raise Image.DecompressionBombError(f"Image size ({pixels} pixels) exceeds limit of {2 * limit} pixels")
                    if pixels > limit:
                        raise Image.DecompressionBombWarning(f"Image size ({pixels} pixels) exceeds limit of {limit} pixels")
                scanlines = _PNGScanlines(width, height, depth, colour, interlace)
                seen_ihdr = True
            elif kind == b"PLTE":
                entries = length // 3
                if colour == 3 and entries > 1 << depth:
                    raise ValueError("PLTE entries exceed palette bit depth")
                scanlines.palette_entries = entries
                seen_plte = True
            elif kind == b"tRNS":
                if any(alpha != 255 for alpha in payload):
                    raise ValueError("nonopaque palette tRNS is unsupported (including unused entries)")
                seen_trns = True
            elif kind == b"IEND":
                if not inflater.eof:
                    raise ValueError("truncated/incomplete IDAT zlib stream")
                scanlines.finish()
                if handle.read(1):
                    raise ValueError("trailing bytes after IEND")
                return

def _decode(path, diagnostic_path, policy):
    Image, _ = _pillow()
    shape = None
    try:
        shape = _header_shape(path)
        # Validate structure and the finite pixel guard before Pillow can decode
        # (PNG getexif() may itself trigger image.load()).
        _validate_png_structure(path, policy=policy)
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
            image.load()
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

    def warn_name(name, path, warning_list):
        if not unicodedata.is_normalized("NFC", name):
            warning_list.append(SourceIssue(rel(path), "name is not NFC-normalised; byte-exact identity preserved"))

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
            warn_name(model_dir.name, model_dir, model_warnings)
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
                warn_name(folder.name, folder, model_warnings)
                if folder.name != "cno" and folder.name.casefold() == "cno":
                    model_errors.append(SourceIssue(rel(folder), "reserved CNT folder is exactly 'cno'; rename this case variant explicitly"))
                    continue
                is_cnt = folder.name == "cno"
                cno_present |= is_cnt
                png_count = 0
                for path in sorted(folder.iterdir()):
                    if not safe(path, model_errors):
                        continue
                    if path.is_dir() or path.suffix.lower() != ".png":
                        unexpected(path, model_errors, model_warnings)
                        continue
                    png_count += 1
                    try:
                        stack, layer = parse_png_name(path.name)
                    except ValueError as exc:
                        model_errors.append(SourceIssue(rel(path), str(exc)))
                        continue
                    warn_name(layer, path, model_warnings)
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
                if not is_cnt and not png_count:
                    model_warnings.append(SourceIssue(rel(folder), "electrode folder contains no PNG assets"))
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
