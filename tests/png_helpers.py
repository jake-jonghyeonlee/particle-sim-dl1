"""Synthetic PNG writing helpers; no expected results use importer algorithms."""
from pathlib import Path
import numpy as np
from PIL import Image


def write_png(root, relative_path, *, shape=(1001, 1203), pixels=((0, 0),), mode="L",
              compress_level=6, **save_options):
    path = Path(root) / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full(shape, 255, dtype=np.uint8)
    for y, x in pixels:
        data[y, x] = 0
    image = Image.fromarray(data)
    if mode == "P":
        indices = np.where(data == 0, 0, 1).astype(np.uint8)
        image = Image.fromarray(indices).convert("P")
        image.putpalette([0, 0, 0, 255, 255, 255] + [0] * (256 * 3 - 6))
    elif mode != "L":
        image = image.convert(mode)
    image.save(path, compress_level=compress_level, **save_options)
    return path


def write_model(root, assets, *, shape=(1001, 1203), model="M"):
    """Assets are hand-written (folder, filename, foreground y/x pixels)."""
    for folder, filename, pixels in assets:
        write_png(root, f"{model}/{folder}/{filename}", shape=shape, pixels=pixels)


def png_chunk(kind, payload, *, corrupt_crc=False):
    """Hand-built PNG chunk, independent of the decoder and Pillow writer."""
    import struct
    import zlib
    crc = zlib.crc32(kind + payload) ^ int(corrupt_crc)
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def raw_png(width, height, depth, colour, raw, *, interlace=0, compression=0, filtering=0,
            before_idat=(), idat_parts=None, after_idat=()):
    import struct
    import zlib
    header = struct.pack(">IIBBBBB", width, height, depth, colour, compression, filtering, interlace)
    parts = (zlib.compress(raw),) if idat_parts is None else idat_parts
    return (b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", header) +
            b"".join(png_chunk(kind, data) for kind, data in before_idat) +
            b"".join(png_chunk(b"IDAT", data) for data in parts) +
            b"".join(png_chunk(kind, data) for kind, data in after_idat) + png_chunk(b"IEND", b""))


def encode_adam7_binary(pattern, depth, colour):
    """Independent fixture encoder: explicit PNG Adam7 pass coordinates, filter 0."""
    height, width = pattern.shape
    rows = bytearray()
    passes = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
              (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
    for start_x, start_y, step_x, step_y in passes:
        columns = list(range(start_x, width, step_x))
        if not columns: continue
        for y in range(start_y, height, step_y):
            samples = []
            for x in columns:
                # Pattern True is black. Palette entry 0 black, entry 1 white.
                sample = 0 if pattern[y, x] else (1 if colour == 3 else (1 << depth) - 1)
                if colour in {0, 3}: samples.append(sample)
                elif colour == 2: samples.extend([sample] * 3)
                elif colour == 4: samples.extend([sample, 255])
                elif colour == 6: samples.extend([sample] * 3 + [255])
            rows.append(0)
            for offset in range(0, len(samples), 8 // depth):
                value = 0
                for i, sample in enumerate(samples[offset:offset + 8 // depth]):
                    value |= sample << (8 - depth * (i + 1))
                rows.append(value)
    return bytes(rows)


def write_import_fixture(root, *, model="SYNTHETIC-M", shape=(1001, 1203)):
    """Four disjoint contacts: direct, skipped/confirmation, ambiguous and margin.

    Expected relation centres are (601,601), (701,701), (21,21); the contact at
    (801,801) overlaps two lower electrodes and stays unresolved.
    """
    def square(y, x):
        return tuple((y + dy, x + dx) for dy in range(3) for dx in range(3))
    direct, skipped, ambiguous, margin = square(600, 600), square(700, 700), square(800, 800), square(20, 20)
    write_model(root, [
        ("A", "1-U.png", skipped), ("A", "5-U5.png", direct + ambiguous + margin),
        ("N", "3-N.png", ((900, 900),)),
        ("B", "7-L.png", direct + skipped + ambiguous + margin),
        ("D", "7-L.png", ambiguous),
        ("cno", "4_C4.png", ()),
        ("cno", "006_C6.png", direct + skipped + ambiguous + margin),
    ], shape=shape, model=model)


def write_wide_import_fixture(root, *, model="SYNTHETIC-WIDE"):
    """Hand-designed 32px CNT pads and 80px traces in a 1401×1603 frame.

    CNT bboxes: margin [24,24,56,56], direct [584,584,616,616],
    skipped [784,784,816,816], ambiguous [984,984,1016,1016].
    Representative centres (ties y then x): (39,39), (599,599), (799,799).
    """
    shape = (1401, 1603)
    pads = [(24, 24, 56, 56), (584, 584, 616, 616),
            (784, 784, 816, 816), (984, 984, 1016, 1016)]
    def rectangles(boxes):
        result = np.zeros(shape, dtype=bool)
        for x0, y0, x1, y1 in boxes: result[y0:y1, x0:x1] = True
        return result
    traces = [(0, 0, 220, 80), (440, 560, 720, 640),
              (640, 760, 920, 840), (840, 960, 1120, 1040)]
    lower_traces = [(0, 0, 80, 220), (560, 440, 640, 720),
                    (760, 640, 840, 920), (960, 840, 1040, 1120)]
    assets = [("A", "1-U.png", [traces[2]]),
              ("A", "5-U5.png", [traces[i] for i in (0, 1, 3)]),
              ("N", "3-N.png", [(1200, 1200, 1320, 1280)]),
              ("B", "7-L.png", lower_traces), ("D", "7-L.png", [lower_traces[3]]),
              ("cno", "4_C4.png", []), ("cno", "6_C6.png", pads)]
    for folder, name, boxes in assets:
        path = Path(root) / model / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.where(rectangles(boxes), 0, 255).astype(np.uint8)).save(path)
    return pads


def write_stack_order(path, order):
    """Syntax-only synthetic reference: caller supplies literal positions."""
    import json
    path = Path(path)
    path.write_text(json.dumps({'stack_order_schema_version': 'dl1_png_stack_order_v1', 'order': order}), encoding='utf-8')
    return path
