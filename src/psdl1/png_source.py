"""Read-only, original-resolution PNG inventory. Pillow is imported on demand.

Decoding is serialized because Pillow's pixel guard is process-global. Neither
geometry nor source names are resampled, registered, thresholded or aliased.
"""
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
from io import BytesIO
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
                           SourceAsset, SourceInventory, SourceIssue, canonical_id, mask_hash,
                           StackOrderInput, StackOrderResolution, OrderEntry, STACK_ORDER_SCHEMA_VERSION, order_digest)
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
    match = re.fullmatch(r"([0-9]+)[-_](.+)\.[pP][nN][gG]", filename)
    if match is None or int(match[1]) <= 0:
        raise ValueError(f"{filename}: expected positive numeric prefix and first '-'/'_' separator")
    return int(match[1]), identifier(match[2], "layer")


def _header_shape(path):
    # IHDR gives dimensions even when Pillow raises before returning an Image.
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) == 24 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR":
        w, h = struct.unpack(">II", header[16:24])
        return h, w
    return None


class _PNGBuffer:
    """Immutable compressed snapshot shared by all decoder/validator reads."""
    def __init__(self, data):
        self.data = data

    def open(self, mode="rb"):
        if mode != "rb":
            raise ValueError("PNG snapshot is read-only")
        return BytesIO(self.data)


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
        data = path.read_bytes()
        snapshot = _PNGBuffer(data)
        shape = _header_shape(snapshot)
        # Validate structure and the finite pixel guard before Pillow can decode
        # (PNG getexif() may itself trigger image.load()).
        _validate_png_structure(snapshot, policy=policy)
        with Image.open(BytesIO(data)) as image:
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
            return mask, mode, hashlib.sha256(data).hexdigest()
    except Exception as exc:
        raise PNGSourceError(SourceIssue(diagnostic_path, str(exc), type(exc).__name__, shape,
                                         shape[0] * shape[1] if shape else None,
                                         policy.effective_max_image_pixels)) from exc


def decode_png(path, *, png_max_image_pixels=None):
    """Standalone strict decoder returning (bool mask, original mode, policy)."""
    path = Path(path)
    with decoder_session(png_max_image_pixels) as policy:
        mask, mode, _ = _decode(path, str(path), policy)
        return mask, mode, policy


class _SourceScanner:
    """Single-use shared scanner. Global state contains scalar vocabulary/IDs only.

    Consume models one at a time, releasing each before advancing. Call finish
    only after exhaustion, passing eager or detached records in traversal order.
    Closing the iterator also restores the decoder policy after interruption.
    """
    def __init__(self, root, *, png_max_image_pixels=None, stack_order=None, progress=None):
        self.progress = progress or (lambda **record: None)
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError(f"{self.root}: source root must be a directory")
        self.reference = stack_order if isinstance(stack_order, StackOrderInput) else load_stack_order(stack_order)
        self.png_max_image_pixels = png_max_image_pixels
        self.errors, self.notices, self.names = [], [], []
        self.decoder = None
        self.complete = False
        self.started = False

    def __iter__(self):
        if self.started:
            raise ValueError("source scanner is single-use")
        self.started = True
        return _scan_source(self)

    def finish(self, models):
        if not self.complete:
            raise ValueError("source scanner must be exhausted before finalization")
        models = tuple(models)
        if tuple(m.model_id for m in models) != tuple(m[0] for m in self.names):
            raise ValueError("scanner models must preserve traversal order")
        return _finish_inventory(models, self)


def _scan_source(scanner):
    root, reference = scanner.root, scanner.reference
    image_only = reference.source is None
    errors, notices = scanner.errors, scanner.notices
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

    def scan_model(model_dir, policy):
        scanner.progress(stage="reading", state="started", model_id=model_dir.name)
        model_errors, model_warnings, assets = [], [], []
        warn_name(model_dir.name, model_dir, model_warnings)
        try:
            identifier(model_dir.name, "model")
        except ValueError as exc:
            model_errors.append(SourceIssue(rel(model_dir), str(exc)))
        bindings = {}
        files_done = 0
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
                    filename_number, layer = parse_png_name(path.name)
                except ValueError as exc:
                    model_errors.append(SourceIssue(rel(path), str(exc)))
                    continue
                warn_name(layer, path, model_warnings)
                binding = (folder.name, layer)
                if not is_cnt and binding in bindings:
                    model_errors.append(SourceIssue(rel(path), f"duplicate normalized binding; also {bindings[binding]}"))
                    model_errors.append(SourceIssue(bindings[binding], f"duplicate normalized binding; also {rel(path)}"))
                elif not is_cnt:
                    bindings[binding] = rel(path)
                try:
                    mask, mode, file_hash = _decode(path, rel(path), policy)
                    source_binding = (model_dir.name, "cno" if is_cnt else "electrode",
                                      None if is_cnt else folder.name, layer)
                    geometry_hash = mask_hash(mask)
                    if is_cnt:
                        source_binding = (model_dir.name, "cno", None, layer, geometry_hash)
                    asset_id = canonical_id("asset", source_binding)
                    duplicate_content = is_cnt and image_only and any(a.kind == 'cno' and a.binding == source_binding for a in assets)
                    if not duplicate_content:
                        claim_id(asset_id, rel(path), model_errors)
                    assets.append(SourceAsset(
                        asset_id, model_dir.name, source_binding[1],
                        source_binding[2], source_binding[2], layer, filename_number, rel(path), path.name,
                        file_hash, geometry_hash, mode,
                        tuple(mask.shape), None if duplicate_content else mask))
                    del mask
                except PNGSourceError as exc:
                    model_errors.append(exc.issue)
                files_done += 1
                if files_done == 1 or files_done % 100 == 0:
                    scanner.progress(stage="reading", state="advanced", model_id=model_dir.name, files_done=files_done)
            if not is_cnt and not png_count:
                model_warnings.append(SourceIssue(rel(folder), "electrode folder contains no PNG assets"))
        scanner.progress(stage="reading", state="completed", model_id=model_dir.name, files_done=files_done, files_total=files_done)
        assets.sort(key=lambda a: (a.kind, a.layer, a.electrode or "", a.mask_sha256))
        cnt_groups = {}
        for a in assets:
            if a.kind == "cno":
                cnt_groups.setdefault((a.layer, a.mask_sha256), []).append(a)
        for (name, digest), group in sorted(cnt_groups.items()):
            if len(group) > 1:
                paths = sorted(a.relative_path for a in group)
                code = 'duplicate_cnt_mask_same_name' if image_only else 'ambiguous_cnt_asset_identity'
                (model_warnings if image_only else model_errors).append(SourceIssue(rel(model_dir),
                    f"{code}: model {json.dumps(model_dir.name, ensure_ascii=False)}, CNT {json.dumps(name, ensure_ascii=False)}, mask_sha256 {digest}, files {json.dumps(paths, ensure_ascii=False)}" +
                    ("; shared content identity, all files retained as provenance" if image_only else "; separate stable source identifiers are required")))
                if image_only:
                    # One content identity; filename numbers never distinguish assets.
                    representative = group[0]
                    files = tuple({k: getattr(a, k) for k in ('relative_path', 'original_filename', 'filename_number', 'file_sha256', 'decoded_mode')} for a in group)
                    assets = [a for a in assets if a not in group] + [replace(representative, provenance_files=files)]
        assets.sort(key=lambda a: (a.kind, a.layer, a.electrode or "", a.mask_sha256))
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
        folded_names = {}
        for a in assets:
            folded_names.setdefault(a.layer.casefold(), set()).add(a.layer)
        for names in folded_names.values():
            if len(names) > 1:
                code = 'layer_name_case_variants' if image_only else 'stack_order_case_collision'
                (model_warnings if image_only else model_errors).append(SourceIssue(rel(model_dir),
                    f"{code}: model {json.dumps(model_dir.name, ensure_ascii=False)}, names {json.dumps(sorted(names), ensure_ascii=False)}, files {json.dumps(sorted(a.relative_path for a in assets if a.layer in names), ensure_ascii=False)}"))
        estimate = ResourceEstimate(unknown_reasons=("invalid_source",) if model_errors else ("extraction_not_run",))
        return ModelSource(model_dir.name, tuple(assets),
                           tuple(sorted({a.role for a in assets if a.role is not None})),
                           tuple(sorted({a.layer for a in assets})), frame, cno_present,
                           tuple(model_errors), tuple(model_warnings), estimate)

    with decoder_session(scanner.png_max_image_pixels) as policy:
        scanner.decoder = policy
        model_index = 0
        for model_dir in sorted(root.iterdir()):
            if not safe(model_dir, errors):
                continue
            if not model_dir.is_dir():
                unexpected(model_dir, errors, notices)
                continue
            model_index += 1
            scanner.progress(stage="model", state="started", model_id=model_dir.name, model_index=model_index)
            model = scan_model(model_dir, policy)
            model = replace(model, stack_order=resolve_model_order(model, reference),
                            cnt_mode="image_only" if image_only else "relations")
            scanner.names.append((model.model_id, model.roles, model.layers))
            yield model
            del model
            scanner.progress(stage="model", state="completed", model_id=model_dir.name, model_index=model_index)
    scanner.complete = True


def _finish_inventory(models, scanner):
    errors, notices = list(scanner.errors), list(scanner.notices)
    policy, image_only = scanner.decoder, scanner.reference.source is None
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
    if image_only:
        folded = {}
        for name in layers:
            folded.setdefault(name.casefold(), set()).add(name)
        for variants in folded.values():
            involved = sorted(m.model_id for m in models if set(m.layers) & variants)
            if len(variants) > 1 and len(involved) > 1:
                notices.append(SourceIssue('.', f'layer_name_case_variants: names {json.dumps(sorted(variants), ensure_ascii=False)}, models {json.dumps(involved, ensure_ascii=False)}; exact source vocabulary preserved'))
    inventory = SourceInventory(tuple(models), policy, roles, layers, unique_roles, unique_layers,
                           tuple(errors), tuple(notices))
    return inventory


def discover_source(root, *, png_max_image_pixels=None, stack_order=None) -> SourceInventory:
    """Eager compatibility API; arrays remain resident for all returned models."""
    scanner = _SourceScanner(root, png_max_image_pixels=png_max_image_pixels, stack_order=stack_order)
    return scanner.finish(tuple(scanner))


def _detach_model(model):
    """Private report snapshot: None marks detached, never an empty observation."""
    return replace(model, assets=tuple(replace(a, mask=None) for a in model.assets))


def _require_model_masks(model):
    if any(a.mask is None for a in model.assets):
        raise ValueError("detached source model requires verified reload before mask consumption")


def _reload_model(root, model, policy, *, progress=None):
    """Reload only recorded paths, checking each immutable decoder snapshot."""
    try:
        root = Path(root).resolve(strict=True)
    except (OSError, ValueError, RuntimeError) as exc:
        raise ValueError(f"source changed: source root unavailable: {root}") from exc
    progress = progress or (lambda **record: None)
    files_total = sum(len(a.provenance_files) or 1 for a in model.assets)
    files_done = 0
    progress(stage="reading", state="started", model_id=model.model_id, files_total=files_total)
    assets = []
    for asset in model.assets:
        representative = None
        files = asset.provenance_files or ({"relative_path": asset.relative_path,
                  "file_sha256": asset.file_sha256, "decoded_mode": asset.decoded_mode},)
        for provenance in files:
            relative = provenance["relative_path"]
            path = root / relative
            try:
                path.resolve(strict=True).relative_to(root)
                mask, mode, digest = _decode(path, relative, policy)
                if (digest != provenance["file_sha256"] or mode != provenance["decoded_mode"] or
                        tuple(mask.shape) != asset.shape_yx or mask_hash(mask) != asset.mask_sha256):
                    raise ValueError("snapshot differs from inventory")
            except (OSError, ValueError, RuntimeError) as exc:
                raise ValueError(f"source changed: {relative}: {exc}") from exc
            if relative == asset.relative_path:
                representative = mask
            del mask
            files_done += 1
            if files_done == 1 or files_done % 100 == 0 or files_done == files_total:
                progress(stage="reading", state="advanced", model_id=model.model_id, files_done=files_done, files_total=files_total)
        if representative is None:
            raise ValueError(f"source changed: missing representative provenance: {asset.relative_path}")
        assets.append(replace(asset, mask=representative))
    progress(stage="reading", state="completed", model_id=model.model_id, files_done=files_done, files_total=files_total)
    return replace(model, assets=tuple(assets))


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
            reason = exc.strerror if isinstance(exc, OSError) else str(exc)
            raise ValueError(f"metadata: {reason}") from exc
    if require_complete and result.pending_models:
        raise ValueError("missing explicit metadata for models: " + ", ".join(result.pending_models))
    return result


def report_payload(report: ImportReport) -> dict:
    """JSON-safe sidecar evidence, intentionally excluding in-memory source masks."""
    def encode(value):
        if is_dataclass(value):
            result = {f.name: encode(getattr(value, f.name)) for f in fields(value) if f.name != "mask"}
            if "extraction_status" in result:
                status = result.pop("extraction_status")
                result["extractor_run_status"] = "ran" if status in {"complete", "extractor_ran"} else status
            if isinstance(value, ModelSource):
                # Distinguish electrode vocabulary, CNT inventory and resolved order.
                result["electrode_layers"] = list(value.electrode_layers)
                result["cno_layers"] = list(value.cno_layers)
                result["all_source_layers"] = result.pop("layers")
            if isinstance(value, SourceAsset):
                if not value.provenance_files:
                    result.pop('provenance_files')
                result["order_index"] = next((m.order_index(value) for m in report.inventory.models if m.model_id == value.model), None)
            if result.get('geometry_only_component_count', False) is None:
                result.pop('geometry_only_component_count')
            if isinstance(value, SourceInventory):
                for name in ("layer_union", "unique_layers"):
                    result["all_source_" + name] = result.pop(name)
                for kind in ("electrode", "cno"):
                    counts = {}
                    for model in value.models:
                        for name in getattr(model, kind + "_layers"):
                            counts[name] = counts.get(name, 0) + 1
                    result[kind + "_layer_union"] = sorted(counts)
                    result["unique_" + kind + "_layers"] = sorted(n for n, count in counts.items() if count == 1)
            return result
        if isinstance(value, (tuple, list)):
            return [encode(v) for v in value]
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        return value
    return encode(report)


def _order_document(doc):
    """Strict structural validation shared with self-contained report readers."""
    q = lambda v: json.dumps(v, ensure_ascii=True)
    if not isinstance(doc, dict):
        raise ValueError('stack_order_schema: expected object')
    required = {'order', 'stack_order_schema_version'}
    if set(doc) != required:
        raise ValueError('stack_order_fields: required ["order", "stack_order_schema_version"]; missing ' +
                         q(sorted(required - set(doc))) + '; unknown ' + q(sorted(set(doc) - required)))
    if doc['stack_order_schema_version'] != STACK_ORDER_SCHEMA_VERSION:
        raise ValueError('stack_order_version: expected "dl1_png_stack_order_v1", got ' + q(doc['stack_order_schema_version']))
    names = doc['order']
    if not isinstance(names, list):
        raise ValueError('stack_order_list_type: order must be an array of layer names')
    for i, name in enumerate(names):
        if (not isinstance(name, str) or not name or name.strip() != name or
                any(ord(c) < 32 or ord(c) == 127 or 0xD800 <= ord(c) <= 0xDFFF for c in name)):
            raise ValueError(f'stack_order_identifier: order[{i}] must be a nonempty trimmed identifier without controls or unpaired surrogates')
    for name in names:
        indexes = [i for i, n in enumerate(names) if n == name]
        if len(indexes) > 1:
            raise ValueError(f'stack_order_duplicate_name: order repeats {q(name)} at indexes {q(indexes)}')
    for i, name in enumerate(names):
        colliding = [n for n in names if n.casefold() == name.casefold()]
        if len(colliding) > 1:
            indexes = [j for j, n in enumerate(names) if n.casefold() == name.casefold()]
            raise ValueError(f'stack_order_case_collision: order names {q(colliding)} at indexes {q(indexes)}')
    return doc


def load_stack_order(path):
    if path is None:
        return StackOrderInput()
    digest = None
    try:
        try:
            raw = Path(path).read_bytes()
        except OSError as exc:
            raise ValueError(f'stack_order_read_error: cannot read stack order file ({type(exc).__name__})') from exc
        digest = hashlib.sha256(raw).hexdigest()
        if b'\x00' in raw or raw.startswith((b'\xff\xfe', b'\xfe\xff', b'\x00\x00\xfe\xff')):
            raise ValueError('stack_order_encoding: expected UTF-8 with optional leading BOM')
        try:
            text = raw.decode('utf-8-sig')
        except UnicodeError as exc:
            raise ValueError('stack_order_encoding: expected UTF-8 with optional leading BOM') from exc
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('stack_order_duplicate_key: duplicate key ' + json.dumps(key))
                result[key] = value
            return result
        def nonfinite(token):
            raise ValueError('stack_order_nonfinite: invalid JSON constant ' + token)
        def finite_float(token):
            import math
            value = float(token)
            if not math.isfinite(value):
                nonfinite(token)
            return value
        try:
            doc = json.loads(text, object_pairs_hook=unique, parse_constant=nonfinite, parse_float=finite_float)
        except json.JSONDecodeError as exc:
            raise ValueError(f'stack_order_json: invalid JSON at line {exc.lineno}, column {exc.colno}') from exc
        _order_document(doc)
        return StackOrderInput(STACK_ORDER_SCHEMA_VERSION, 'user_supplied_order_file', digest, doc)
    except ValueError as exc:
        return StackOrderInput(source='user_supplied_order_file', file_sha256=digest, errors=(str(exc),))


def resolve_model_order(model, reference):
    q = lambda v: json.dumps(v, ensure_ascii=False)
    if reference.source is None:
        return StackOrderResolution()
    if reference.errors:
        return StackOrderResolution(status='invalid', errors=reference.errors)
    if not model.valid:
        return StackOrderResolution(status="invalid", errors=("invalid_source_inventory",))
    errors, groups = [], {}
    for a in model.assets:
        groups.setdefault((a.kind, a.layer), []).append(a)
    for (kind, name), assets in sorted(groups.items()):
        if kind == 'electrode' and ('cno', name) in groups:
            errors.append(f'stack_order_name_kind_collision: model {q(model.model_id)}, name {q(name)}, electrode files {q(sorted(a.relative_path for a in assets))}, CNT files {q(sorted(a.relative_path for a in groups[("cno", name)]))}')
    if errors:
        return StackOrderResolution(status='invalid', errors=tuple(errors))
    names = reference.document['order']
    reference_keys = {n.casefold() for n in names}
    for kind, code in [('electrode', 'electrode'), ('cno', 'cnt')]:
        missing = sorted(name for k, name in groups if k == kind and name.casefold() not in reference_keys)
        if missing:
            errors.append(f'stack_order_missing_{code}_names: model {q(model.model_id)}, missing {q(missing)}')
    if errors:
        return StackOrderResolution(status='invalid', errors=tuple(errors))
    by_name = {name.casefold(): (kind, name, assets) for (kind, name), assets in groups.items()}
    filtered = [n for n in names if n.casefold() in by_name]
    entries = tuple(OrderEntry(i, by_name[n.casefold()][0], by_name[n.casefold()][1],
                              tuple(sorted(a.asset_id for a in by_name[n.casefold()][2]))) for i, n in enumerate(filtered))
    return StackOrderResolution('resolved', 'user_supplied_order_file', 'order', entries,
                                tuple(n for n in names if n.casefold() not in by_name), order_digest(entries),
                                matched_pairs=tuple((n, by_name[n.casefold()][1]) for n in filtered))


def resolve_stack_order(inventory, reference):
    return replace(inventory, models=tuple(replace(m, stack_order=resolve_model_order(m, reference), cnt_mode="image_only" if reference.source is None else "relations") for m in inventory.models))
