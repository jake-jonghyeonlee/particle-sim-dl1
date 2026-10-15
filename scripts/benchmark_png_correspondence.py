#!/usr/bin/env python3
"""Synthetic single-source correspondence benchmark; no whole-import claims.

Use --baseline-source with a saved png_diagnostic.py to compare in fresh processes.
Peak traced bytes include NumPy buffers allocated during correspondence; RSS is
whole-process high-water (including imports and source/target preparation).
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
import tracemalloc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from psdl1 import png_diagnostic as diag
from psdl1 import data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline-source', type=Path)
    ap.add_argument('--case', choices=('dense', 'sparse'), required=True)
    ap.add_argument('--size', type=int, default=1500)
    ap.add_argument('--grid', type=int, default=384)
    args = ap.parse_args()
    torch.set_num_threads(1)
    implementation = diag
    if args.baseline_source:
        spec = importlib.util.spec_from_file_location('psdl1._benchmark_baseline', args.baseline_source)
        implementation = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = implementation
        spec.loader.exec_module(implementation)
    rng = np.random.default_rng(42)
    mask = np.ones((args.size, args.size), bool) if args.case == 'dense' else rng.random((args.size, args.size)) < .03
    if args.case == 'sparse':
        # Noise plus two retained islands exercises lost material AND real edges.
        side = max(2, args.size // 12)
        for origin in (args.size // 5, args.size * 3 // 5):
            mask[origin:origin + side, origin:origin + side] = True
    measured = diag.measure_source(mask, np.ones_like(mask))
    transform = data.frame_transform(mask.shape, args.grid)
    target = diag.label_components(data.normalize_unit(measured.source, args.grid) >= .5)
    tracemalloc.start()
    start = time.perf_counter()
    result = implementation.correspondence(measured.labels, target, transform)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    digest = hashlib.sha256()
    for array in result:
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        implementation.correspondence(measured.labels, target, transform)
        timings.append(time.perf_counter() - start)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin':
        rss *= 1024
    print(json.dumps(dict(case=args.case, size=args.size, grid=args.grid, foreground=measured.active,
        edges=len(result[0]), result_sha256=digest.hexdigest(), traced_peak_bytes=peak, process_peak_rss_bytes=rss,
        traced_seconds=elapsed, median_seconds=float(np.median(timings)),
        implementation='baseline' if args.baseline_source else 'current')))


if __name__ == '__main__':
    main()
