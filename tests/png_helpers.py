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
