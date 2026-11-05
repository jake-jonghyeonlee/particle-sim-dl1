"""dl1-size/T4b 정책 탐색 재현용: tile 반경·unit 반경·unit 실패 처리·tile 폭 나눔수를 바꿔 가며 (a)(b) 게이트를 ΔU 별로 잰다(게이트 자체는 measure_unit_stretch.py --tile).

실행: PYTHONPATH=$PWD/src python scripts/probe_tile_policy.py TILE_R UNIT_R KEEP DELTAS [UNITS] [--div=N] [--min=N] [--cov=F] [--maxtiles=N] [--maxdelta=N] [--detail] [--detail-b]
  TILE_R, UNIT_R: cur(src 상수) | 정수(칸)   KEEP: keep(unit 실패를 기하 예측으로 대체하고 계속) | fail(전부 실패면 unit_align_failed)
  DELTAS: 쉼표 목록   UNITS: 쉼표 목록(기본 560,1000)   --div/--min/--cov/--maxtiles/--maxdelta: TILE_DIVISOR/TILE_MIN_CELLS/COVERAGE_MIN/MAX_TILES/MAX_DELTA_U_PX 덮어쓰기
"""
import collections
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("measure_unit_stretch", Path(__file__).resolve().parent / "measure_unit_stretch.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
from psdl1 import screen_tile as st  # noqa: E402

tile_r, unit_r, keep, deltas = sys.argv[1], sys.argv[2], sys.argv[3], tuple(int(v) for v in sys.argv[4].split(","))
units = tuple(int(v) for v in sys.argv[5].split(",")) if len(sys.argv) > 5 and not sys.argv[5].startswith("--") else (560, 1000)
for flag in sys.argv:
    if flag.startswith("--div="):
        st.TILE_DIVISOR = int(flag[6:])
    if flag.startswith("--min="):
        st.TILE_MIN_CELLS = int(flag[6:])
    if flag.startswith("--cov="):
        st.COVERAGE_MIN = float(flag[6:])
    if flag.startswith("--maxtiles="):
        st.MAX_TILES = int(flag[11:])
    if flag.startswith("--maxdelta="):
        st.MAX_DELTA_U_PX = int(flag[11:])

original = st.tile_plan


def plan(*a, **k):
    p = original(*a, **k)
    if p.get("status") == "ok":
        if tile_r != "cur":
            p["radius_cells"] = (int(tile_r), int(tile_r))
        if unit_r not in ("cur", "prop"):
            p["unit_radius_cells"] = (int(unit_r), int(unit_r))
    return p


st.tile_plan = plan
if keep == "keep":
    original_align = st.align_tiles

    def align(*a, **k):
        r = original_align(*a, **k)
        r["unit_failed"] = {}
        return r

    st.align_tiles = align

same = m.tile_measure(units=units, deltas=deltas)
false = m.false_measure(units=units, deltas=deltas)
print(f"tile_r={tile_r} unit_r={unit_r} keep={keep} deltas={deltas} units={units}")
print(m.delta_table(same, false))
bad_same, bad_false = m.same_failures(same), m.false_failures(false)
print("(a) fail", len(bad_same), "of", len(same), sorted(collections.Counter((r["unit"], r["delta"], r["tile_status"]) for r in bad_same).items()))
if "--detail" in sys.argv:
    for r in bad_same:
        print("  (a)", r["unit"], r["delta"], r["axis"], r["phase"], r["path"], r["tile_status"], "tile_p", r["tile_p"], "p0", r["p0"], "applied", r["tile_applied"], "of", r["tile_count"],
              "replaced", r["tile_step_replaced"], "shift y", r["tile_shift_min_y"], r["tile_shift_max_y"], "x", r["tile_shift_min_x"], r["tile_shift_max_x"], "tile", r["tile_size_y_cells"], "rad", r["tile_radius_y_cells"])
print("shift_error > 1.5:", sorted((r["unit"], r["delta"], r["axis"], r["phase"], r["path"], round(r["shift_error"], 1)) for r in same if r["shift_error"] > 1.5))
if "--detail-b" in sys.argv:
    for r in false:
        if r["delta"] and r["tile_p"] is not None and r["p_var0"] is not None and r["tile_p"] - r["p_var0"] > 0.02:
            print("  (b+)", r["variant"], r["unit"], r["delta"], r["axis"], r["path"], "tile_p", round(r["tile_p"], 4), "p_var0", round(r["p_var0"], 4),
                  "diff", round(r["tile_p"] - r["p_var0"], 4), "single_p", r["single_p"], "applied", r["tile_applied"], "of", r["tile_count"])
print("(b) fail", len(bad_false), "of", len(false))
for r in bad_false:
    print("  (b)", r["variant"], r["unit"], r["delta"], r["axis"], r["path"], "tile_p", r["tile_p"], "p_var0", r["p_var0"], r["tier"], r["reason"])
