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
