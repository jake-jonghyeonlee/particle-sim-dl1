"""dl1-judge/T2: unit 크기 차(ΔU)가 판정 축 P(axis_p)를 얼마나 깎는지 합성 제품으로 잰다.

합성 제품: 프레임 = 2×2 unit + 사방 MARGIN px(0.1 µm/px). 선폭(CD)은 고정이고 배선 길이·간격만 unit 크기 U 에 비례해 늘어난다
(rescale 하지 않는다). 공개 함수(product_signature, compare, layer_columns, pair_metric_rows, pair_axes)만 쓴다.

실행: PYTHONPATH=$PWD/src python scripts/measure_unit_stretch.py [--unit 1000] [--phases 0-7] [--quick]
"""
import argparse
import math
import sys
import time
from fractions import Fraction

import numpy as np

from psdl1 import screen_align
from psdl1.schema import Element, Product
from psdl1.screen_relations import (block_reduce, compare, layer_columns, layer_item, pair_axes, pair_metric_rows, product_signature)

UNIT = 1000
MARGIN = 500
CD = 12  # 선폭(px). unit 크기와 무관하다
FACTOR = 8
TOLERANCE = 1
MARGIN_P = 0.05  # three-axis 판정 margin 기본값(PLAN v1.1)
JUDGE_SIZE_SUPPORT_PX = 7  # PLAN v1.2 G-size 지원 범위(T2 측정 근거)
GATE_DELTAS = tuple(range(8))
RECORD_DELTAS = (8, 12, 20, 40)
AXES = ("x", "y", "xy")
PATHS = ("anchor", "layer")
LAYERS = ("M1", "M2", "M3")


def fill(mask, y0, y1, x0, x1):
    mask[max(0, int(y0)):max(0, int(y1)), max(0, int(x0)):max(0, int(x1))] = True


def draw_unit(m, oy, ox, uy, ux):
    """unit 하나(원점 oy, ox · 크기 uy × ux)의 배선. 위치는 unit 크기의 비율이고 선폭·패드는 고정이다."""
    r = lambda f, size: int(round(f * size))
    for f in (0.2, 0.6):  # gate/M1: 가로 배선 + 왼쪽 패드
        y = oy + r(f, uy)
        fill(m["gate/M1"], y, y + CD, ox + r(0.1, ux), ox + r(0.9, ux))
        fill(m["gate/M1"], y - 14, y + CD + 14, ox + r(0.1, ux), ox + r(0.1, ux) + 40)
    x = ox + r(0.5, ux)  # gate/M2: 세로 stub
    fill(m["gate/M2"], oy + r(0.2, uy), oy + r(0.8, uy), x, x + CD)
    for f in (0.3, 0.7):  # data/M2: 세로 배선 + 위쪽 패드
        x = ox + r(f, ux)
        fill(m["data/M2"], oy + r(0.1, uy), oy + r(0.9, uy), x, x + CD)
        fill(m["data/M2"], oy + r(0.1, uy), oy + r(0.1, uy) + 40, x - 14, x + CD + 14)
    for (fy, fx, size) in ((0.4, 0.5, 100), (0.8, 0.8, 60)):  # pixel/M3: 고정 크기 판
        y, x = oy + r(fy, uy), ox + r(fx, ux)
        fill(m["pixel/M3"], y, y + size, x, x + size)
    for (fy, fx) in ((0.2, 0.3), (0.6, 0.7)):  # CNT: 고정 크기 점
        y, x = oy + r(fy, uy), ox + r(fx, ux)
        fill(m["cnt"], y - 4, y + 56, x - 4, x + 16)


def build(name, uy, ux, offset=0):
    """2×2 unit 제품. offset 은 내용을 프레임 안에서 (offset, offset) px 옮긴다(프레임 크기는 그대로 2*U+2*MARGIN)."""
    shape = (2 * uy + 2 * MARGIN, 2 * ux + 2 * MARGIN)
    m = {k: np.zeros(shape, dtype=bool) for k in ("gate/M1", "gate/M2", "data/M2", "pixel/M3", "cnt")}
    oy, ox = MARGIN + offset, MARGIN + offset
    for i in range(2):
        for j in range(2):
            draw_unit(m, oy + i * uy, ox + j * ux, uy, ux)
    fill(m["gate/M1"], oy + uy - 6, oy + uy + 6, ox, ox + 2 * ux)  # unit 사이 가로 버스: ΔU 만큼 길어진다
    fill(m["data/M2"], oy, oy + 2 * uy, ox + ux - 6, ox + ux + 6)  # unit 사이 세로 버스
    elements = tuple(Element(*key.split("/"), m[key]) for key in ("gate/M1", "gate/M2", "data/M2", "pixel/M3"))
    product = Product(name, (shape[0] * 0.1, shape[1] * 0.1), elements, m["cnt"].astype(np.float32), lineage_id="L", design_version="v")
    layer_masks = {"m1": m["gate/M1"], "m2": m["gate/M2"] | m["data/M2"], "m3": m["pixel/M3"]}
    return product, layer_masks


def signature(name, uy, ux, offset=0):
    product, layer_masks = build(name, uy, ux, offset)
    sig = product_signature(product, FACTOR, TOLERANCE)
    rasters = {k: block_reduce(v, FACTOR, "any") for k, v in layer_masks.items()}
    sig["layers"] = {k: layer_item(rasters[k], screen_align.pack_mask(layer_masks[k]), sig["full_shape"], TOLERANCE) for k in layer_masks}
    return sig, rasters


def measure_pair(sa, rasters_a, sb, rasters_b, path):
    """정렬 경로 하나로 쌍을 비교하고 판정 축·유사도·Dice 를 돌려준다. path: anchor = 기존 compare 위치, layer = Layer 사이드카 정렬."""
    result = compare(sa, sb, tolerance=TOLERANCE)
    out = {"similarity": result["similarity"], "layer_applied": None}
    if path == "layer":
        align, columns = layer_columns(sa, sb, result, rasters_a, rasters_b, max_shift=0.35, min_overlap=screen_align.ALIGN_MIN_OVERLAP,
                                       min_support=screen_align.ALIGN_MIN_SUPPORT, min_dice=screen_align.ALIGN_MIN_DICE)
        on_layer = columns["metrics_anchor"] == "LAYER"
        out["layer_applied"] = on_layer
        out["similarity"] = columns["similarity_layer"]
    else:
        on_layer = False
    anchor = "LAYER" if on_layer else result["anchor"]
    shift = align["shift"] if on_layer else result["shift"]
    rows = pair_metric_rows(sa, sb, shift, tolerance=TOLERANCE, use_layers=on_layer)
    dice = [r["dice"] for r in rows if r["kind"] == "binding" and r["status"] == "both" and r["dice"] is not None]
    axes = pair_axes(rows, anchor, sa, sb)
    out.update(axes, dice=round(float(np.mean(dice)), 6) if dice else None, shift=tuple(int(v) for v in shift), anchor=anchor)
    return out


def supported(anchor, size_delta_y, size_delta_x):
    """PLAN v1.2 지원 범위: metrics_anchor 가 LAYER 이고 max|size_delta_unit_px| ≤ JUDGE_SIZE_SUPPORT_PX."""
    return anchor == "LAYER" and max(abs(size_delta_y), abs(size_delta_x)) <= JUDGE_SIZE_SUPPORT_PX


def tier(p, p_floor, margin=MARGIN_P, size_tag="same", in_support=True):
    """PLAN v1.1 판정표(다른 계보 쌍의 초기 tier, P 만 본다) + v1.2 G-size: P < p_floor − margin → different, 그 밖은 unknown. P 없음 → unknown.
    size_differs 이고 지원 범위 밖이면 different 대신 unknown(size_differs_out_of_support)."""
    if p is None:
        return "unknown", "no_pattern_value"
    exact = lambda value: Fraction(repr(float(value)))  # src judge_verdict 와 같은 십진 정확 비교(0.52 - 0.05 = 0.47000000000000003 오차 방지)
    if exact(p) < exact(p_floor) - exact(margin):
        if size_tag == "size_differs" and not in_support:
            return "unknown", "size_differs_out_of_support"
        return "different", "below_pattern_floor"
    return "unknown", "close_to_same_lineage" if exact(p) >= exact(p_floor) else "near_boundary"


def classify(records, base):
    """base = {(경로, 위상): ΔU=0 의 axis_p}. 레코드마다 p0, 지원 범위 안/밖, tier 를 채운다."""
    for r in records:
        r["p0"] = base.get((r["path"], r["phase"]))
        r["in_support"] = supported(r["anchor"], r["size_delta_unit_px_y"], r["size_delta_unit_px_x"])
        r["tier"], r["reason"] = (tier(r["axis_p"], r["p0"], size_tag=r["size_tag"], in_support=r["in_support"]) if r["p0"] is not None
                                  else ("unknown", "no_baseline"))


def gate_failures(records):
    """(a) 지원 범위 안(size_differs 가 아니거나 LAYER 이고 |ΔU| ≤ 7): axis_p ≥ P0 − 0.05 이고 tier ≠ different. (b) 밖: tier ≠ different."""
    bad = []
    for r in records:
        inside = r["size_tag"] == "same" or r["in_support"]
        if r["axis_p"] is None or r["p0"] is None or r["tier"] == "different" or (inside and r["axis_p"] < r["p0"] - MARGIN_P):
            bad.append(r)
    return bad


def dims(unit, delta, axis):
    return unit + (delta if "y" in axis else 0), unit + (delta if "x" in axis else 0)


def measure(unit=UNIT, phases=range(8), deltas=GATE_DELTAS, axes=AXES, paths=PATHS, progress=None):
    """레코드 목록. A 는 ΔU=0 · offset 0 제품이고 B 가 (ΔU, 축, 위상)마다 바뀐다. 같은 (경로, 위상)의 ΔU=0 쌍이 기준(P0)이다."""
    sa, ra = signature("A", unit, unit, 0)
    records = []
    for delta in deltas:
        for axis in (AXES[:1] if delta == 0 else axes):
            uy, ux = dims(unit, delta, axis)
            for phase in phases:
                sb, rb = signature("B", uy, ux, phase)
                for path in paths:
                    rec = measure_pair(sa, ra, sb, rb, path)
                    rec.update(delta=delta, axis="-" if delta == 0 else axis, phase=phase, path=path)
                    records.append(rec)
                if progress:
                    progress(delta, axis, phase)
    base = {(r["path"], r["phase"]): r["axis_p"] for r in records if r["delta"] == 0}
    classify(records, base)
    return records


def fmt(v):
    return "—" if v is None else f"{v:.4f}"


def table(records):
    lines = ["| ΔU px | 축 | 경로 | n 위상 | P min | P max | P0 min | (P−P0) min | 유사도 min | Dice min | size_delta_unit_px (y,x) | layer 적용 | 지원 범위 | tier(최악, G-size 포함) | tier(G-size 없이) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    keys = sorted({(r["delta"], r["axis"], r["path"]) for r in records}, key=lambda k: (k[0], k[1], k[2]))
    for delta, axis, path in keys:
        group = [r for r in records if (r["delta"], r["axis"], r["path"]) == (delta, axis, path)]
        ps = [r["axis_p"] for r in group if r["axis_p"] is not None]
        diffs = [r["axis_p"] - r["p0"] for r in group if r["axis_p"] is not None and r["p0"] is not None]
        sizes = {(r["size_delta_unit_px_y"], r["size_delta_unit_px_x"]) for r in group}
        applied = "" if path == "anchor" else f"{sum(bool(r['layer_applied']) for r in group)}/{len(group)}"
        worst = "different" if any(r["tier"] == "different" for r in group) else "unknown"
        raw = "different" if any(tier(r["axis_p"], r["p0"])[0] == "different" for r in group if r["p0"] is not None) else "unknown"
        inside = "안" if all(r["size_tag"] == "same" or r["in_support"] for r in group) else "밖"
        mn = lambda key: fmt(min((r[key] for r in group if r[key] is not None), default=None))
        lines.append(f"| {delta} | {axis} | {path} | {len(group)} | {fmt(min(ps, default=None))} | {fmt(max(ps, default=None))} | {mn('p0')} | "
                     f"{fmt(min(diffs, default=None))} | {mn('similarity')} | {mn('dice')} | {', '.join(str(s) for s in sorted(sizes))} | {applied} | {inside} | {worst} | {raw} |")
    return "\n".join(lines)


def parse_phases(text):
    if "-" in text:
        lo, hi = text.split("-")
        return range(int(lo), int(hi) + 1)
    return [int(v) for v in text.split(",")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--unit", type=int, default=UNIT)
    parser.add_argument("--phases", default="0-7", help="ΔU 0–7 에 쓸 위상(예 0-7 또는 0,4)")
    parser.add_argument("--record-phases", default="0,4", help="ΔU 8·12·20·40(기록만)에 쓸 위상")
    args = parser.parse_args(argv)
    start = time.time()
    done = []
    progress = lambda d, a, p: done.append(1) or (len(done) % 20 == 0 and print(f"  ... {len(done)} B products, {time.time() - start:.0f}s", file=sys.stderr))
    gate = measure(args.unit, parse_phases(args.phases), GATE_DELTAS, progress=progress)
    record = measure(args.unit, parse_phases(args.record_phases), RECORD_DELTAS, progress=progress)
    print(f"# unit stretch (U={args.unit} px, frame={2 * args.unit + 2 * MARGIN}², CD={CD} px, factor={FACTOR}, tolerance={TOLERANCE}, margin={MARGIN_P})\n")
    print("## 게이트 ΔU 0–7 (P0 = 같은 경로·위상의 ΔU=0 쌍)\n")
    print(table(gate))
    print("\n## 기록만 ΔU 8·12·20·40 (P0 는 위 ΔU=0 같은 위상)\n")
    base = {(r["path"], r["phase"]): r["axis_p"] for r in gate if r["delta"] == 0}
    classify(record, base)
    print(table(record))
    gate_fail = gate_failures(gate) + gate_failures(record)
    print(f"\ngate: {'PASS' if not gate_fail else 'FAIL'} ({len(gate_fail)} failing of {len(gate) + len(record)}); elapsed {time.time() - start:.0f}s")
    for r in gate_fail[:20]:
        print(f"  FAIL ΔU={r['delta']} axis={r['axis']} phase={r['phase']} path={r['path']} P={r['axis_p']} P0={r['p0']}")
    return 0 if not gate_fail else 1


if __name__ == "__main__":
    sys.exit(main())
