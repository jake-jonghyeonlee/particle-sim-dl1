"""dl1-size/T4b 진단용: 한 조합의 입력 이동·unit 이동·tile 오프셋과 coverage 구멍을 본다. 인수: U ΔU axis phase path [TILE_DIVISOR]"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("measure_unit_stretch", Path(__file__).resolve().parent / "measure_unit_stretch.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
from psdl1 import screen_tile as st  # noqa: E402
from psdl1.screen_relations import tile_inputs  # noqa: E402

unit, delta, axis, phase, path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
if len(sys.argv) > 6:
    st.TILE_DIVISOR = int(sys.argv[6])
sa, ra = m.signature("A", unit, unit, 0)
uy, ux = m.dims(unit, delta, axis)
sb, rb = m.signature("B", uy, ux, phase)
out = m.measure_pair(sa, ra, sb, rb, path)
sources, source, keys = tile_inputs(sa, sb, out["anchor"])
origin = st.origin_shift(sa, sb, out["shift"])
print("shift", out["shift"], "origin", origin, "anchor", out["anchor"], "shapes", sa["shape"], sb["shape"])
plan = st.tile_plan(st.unit_geometry(sa["full_shape"], 8), st.unit_geometry(sb["full_shape"], 8), 8, 1)
layers = {k: (np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)) for k, (a, b) in sorted(sources.items())}
al = st.align_tiles(plan, layers, np.asarray(sa["valid"], dtype=bool), np.asarray(sb["valid"], dtype=bool), origin)
print("unit_shifts", al["unit_shifts"], "unit_failed", al["unit_failed"], "applied", sum(al["applied"]), "of", len(al["applied"]))
cov, p = st._score(sa, sb, plan, al["offsets"], keys, 1)
print("coverage", cov, "p", p, "tile", plan["tile_cells"], "radius", plan["radius_cells"], "unit radius", plan["unit_radius_cells"])
bounds = plan["geometry_a"]["bounds"]
box = (slice(bounds[0][0], bounds[0][2]), slice(bounds[1][0], bounds[1][2]))
wv = st.warp_raster(sb["valid"], plan, al["offsets"])[box] & np.asarray(sa["valid"], dtype=bool)[box]
print("valid frac per tile (min over tiles)")
worst = sorted(((float(wv[t["region"][0] - bounds[0][0]:t["region"][1] - bounds[0][0], t["region"][2] - bounds[1][0]:t["region"][3] - bounds[1][0]].mean()), t["unit_index"], t["tile_index"], t["region"], al["offsets"][i], al["predictions"][i]) for i, t in enumerate(plan["tiles"])))[:8]
for w in worst:
    print(" ", w)
print("A shape", sa["shape"], "B shape", sb["shape"], "valid A frac in box", np.asarray(sa["valid"])[box].mean(), "valid B all", np.asarray(sb["valid"]).mean())
