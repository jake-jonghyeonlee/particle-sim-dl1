"""screen-metrics/T5: binding 별 판단 지표, 쌍 단위 지표, bindings.csv, summary AUC. 손계산 반례 중심이다."""
import csv
import json

import numpy as np
import pytest

from psdl1 import screen_relations
from psdl1.data import Element, Product
from psdl1.screen_align import inner_boundary
from psdl1.screen_relations import evaluate_shift, main, pair_metric_columns, pair_metric_rows, product_signature, screen, metric_context
from test_screen_relations import FACTOR, case, incremental_inputs  # noqa: F401  (case is a fixture)

TOL = 1


def rect(shape, y0, y1, x0, x1):
    out = np.zeros(shape, dtype=np.float32)
    out[y0:y1, x0:x1] = 1
    return out


def product(name, shape, elements, cnt=None, valid=None):
    """elements: {(role, layer): 마스크}."""
    zero = np.zeros(shape, dtype=np.float32)
    return Product(name, (shape[0] * 0.05, shape[1] * 0.05), tuple(Element(r, l, np.asarray(m, dtype=np.float32)) for (r, l), m in elements.items()),
                   zero if cnt is None else cnt, lineage_id="L", design_version="v", valid=valid)


def sig(p, factor=2):
    return product_signature(p, factor, TOL)


def rows_by(rows):
    return {(r["kind"], r["key"]): r for r in rows}


def metrics(pa, pb, factor=2, shift=(0, 0), **kw):
    return pair_metric_rows(sig(pa, factor), sig(pb, factor), shift, tolerance=TOL, **kw)


# ---- 형상: 채움 F1 은 높고 경계 F1 은 낮다 -----------------------------------------------------------------------------
def test_plate_electrodes_that_differ_only_at_the_edge_have_high_fill_f1_and_low_boundary_f1():
    shape = (256, 256)
    a = product("A", shape, {("gate", "l1"): rect(shape, 16, 240, 16, 240)})
    b = product("B", shape, {("gate", "l1"): rect(shape, 20, 236, 20, 236)})  # 래스터(factor 2)에서 가장자리가 2 칸 안쪽
    row = rows_by(metrics(a, b))[("binding", "gate/l1")]
    assert row["status"] == "both" and row["fill_f1"] > 0.95 and row["boundary_f1"] < 0.1 and row["dice"] > 0.95
    assert row["boundary_precision"] == 0.0 and row["boundary_recall"] == 0.0
    assert row["area_a_px"] == 224 * 224 and row["area_b_px"] == 216 * 216 and row["area_ratio"] == pytest.approx((216 / 224) ** 2, abs=1e-6)
    assert rows_by(metrics(a, b))[("role", "gate")]["boundary_f1"] == row["boundary_f1"]


def test_identical_shapes_have_boundary_f1_one():
    shape = (64, 64)
    m = rect(shape, 10, 40, 12, 50)
    row = rows_by(metrics(product("A", shape, {("gate", "l1"): m}), product("B", shape, {("gate", "l1"): m})))[("binding", "gate/l1")]
    assert row["boundary_f1"] == 1.0 and row["dice"] == 1.0 and row["fill_f1"] == 1.0 and row["area_ratio"] == 1.0


# ---- presence 문턱 ----------------------------------------------------------------------------------------------------
def test_a_single_pixel_binding_against_a_large_one_is_not_present_and_the_electrode_jaccard_drops():
    shape = (64, 64)
    big = rect(shape, 8, 40, 8, 40)
    dot = rect(shape, 20, 21, 20, 21)
    a = product("A", shape, {("gate", "l1"): big, ("data", "l2"): dot})
    b = product("B", shape, {("gate", "l1"): big, ("data", "l2"): big})
    rows = metrics(a, b)
    assert rows_by(rows)[("binding", "data/l2")]["status"] == "only_b" and rows_by(rows)[("binding", "gate/l1")]["status"] == "both"
    columns = pair_metric_columns(rows, "CNT")
    old = evaluate_shift(sig(a), sig(b), 0, 0)["usage_jaccard"]
    assert old == 1.0 and columns["usage_jaccard_electrodes"] == 0.5 < old
    assert rows_by(metrics(a, b, presence_fraction=0.0))[("binding", "data/l2")]["status"] == "both"


def test_one_sided_binding_has_the_x4_values_and_never_calls_tolerant_distance(monkeypatch):
    shape = (64, 64)
    a = product("A", shape, {("gate", "l1"): rect(shape, 8, 40, 8, 40)})
    b = product("B", shape, {("gate", "l1"): np.zeros(shape)})

    def forbidden(*args):
        raise AssertionError("tolerant_distance must not be called for a one-sided binding")
    monkeypatch.setattr(screen_relations, "tolerant_distance", forbidden)
    row = rows_by(metrics(a, b))[("binding", "gate/l1")]
    assert (row["status"], row["area_ratio"], row["dice"], row["boundary_f1"], row["fill_f1"]) == ("only_a", 0.0, 0.0, 0.0, 0.0)
    assert row["boundary_precision"] is None and row["boundary_recall"] is None and row["area_b_px"] == 0
    assert rows_by(metrics(b, a))[("binding", "gate/l1")]["status"] == "only_b"
    assert metrics(product("E", shape, {("gate", "l1"): np.zeros(shape)}), product("F", shape, {("gate", "l1"): np.zeros(shape)})) == []  # 양쪽 없음 → 행 없음


# ---- CNT -------------------------------------------------------------------------------------------------------------
def test_a_pair_that_differs_only_in_cnt_has_low_cnt_similarity_and_the_electrode_metrics_do_not_move():
    shape = (96, 96)
    plate = {("gate", "l1"): rect(shape, 8, 80, 8, 80)}
    same = product("S", shape, plate, cnt=rect(shape, 20, 40, 20, 40))
    other = product("O", shape, plate, cnt=rect(shape, 50, 74, 52, 76))
    none = product("N", shape, plate)
    rows = metrics(same, other)
    columns = pair_metric_columns(rows, "CNT")
    assert columns["cnt_similarity"] == 0.0 and rows_by(rows)[("cnt", "CNT")]["status"] == "both"
    assert columns["usage_jaccard_electrodes"] == 1.0 and columns["electrode_similarity"] == pair_metric_columns(metrics(same, none), "CNT")["electrode_similarity"]
    assert pair_metric_columns(metrics(same, none), "CNT")["cnt_similarity"] == 0.0 and rows_by(metrics(same, none))[("cnt", "CNT")]["status"] == "only_a"
    assert pair_metric_columns(metrics(none, none), "CNT")["cnt_similarity"] is None  # 둘 다 없음 → 빈칸
    assert all(r["kind"] != "cnt" for r in metrics(none, none))
    assert pair_metric_columns(metrics(same, same), "CNT")["cnt_similarity"] == 1.0


# ---- X1: 면적은 원본 해상도 ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("factor", [1, 2, 8])
def test_original_resolution_presence_does_not_depend_on_the_raster_factor(factor):
    shape = (64, 64)
    a = product("A", shape, {("gate", "l1"): rect(shape, 30, 31, 30, 31)})   # 원본 1 px
    b = product("B", shape, {("gate", "l1"): rect(shape, 28, 36, 24, 40)})   # 연속 8×16 = 128 px
    row = rows_by(metrics(a, b, factor=factor))[("binding", "gate/l1")]
    assert (row["status"], row["area_a_px"], row["area_b_px"]) == ("only_b", 1, 128)  # 래스터 면적으로 보면 factor 8 에서 1 칸 대 1~2 칸이라 both 가 된다
    both = metrics(product("A", shape, {("gate", "l1"): rect(shape, 28, 36, 24, 40)}), b, factor=factor)
    assert rows_by(both)[("binding", "gate/l1")]["status"] == "both" and rows_by(both)[("binding", "gate/l1")]["area_a_px"] == 128


def random_mask(shape, seed, density=0.2):
    return (np.random.RandomState(seed).rand(*shape) < density).astype(np.float32)


@pytest.mark.parametrize("shape_a,shape_b,shift,valid", [((96, 80), (64, 64), (3, -2), False), ((64, 64), (64, 64), (0, 0), False), ((64, 56), (96, 88), (-4, 5), False),
                                                           ((60, 52), (44, 50), (2, 1), False), ((64, 64), (64, 64), (1, 1), True)])
def test_block_sum_counts_equal_the_direct_original_sum(shape_a, shape_b, shift, valid):
    """개수 래스터(정확한 경우)와 원본 직접 합이 같다. 정확하지 않은 경우(프레임이 factor 로 안 나뉨, valid 구멍)는 직접 합만 쓴다."""
    def build(name, shape, seed):
        hole = None
        if valid:
            hole = np.ones(shape, dtype=np.uint8)
            hole[10:30, 10:30] = 0
        return product(name, shape, {("gate", "l1"): random_mask(shape, seed), ("data", "l2"): random_mask(shape, seed + 1, 0.05)}, cnt=random_mask(shape, seed + 2, 0.1), valid=hole)
    a, b = build("A", shape_a, 1), build("B", shape_b, 4)
    sa, sb = sig(a, 8), sig(b, 8)
    fast = pair_metric_rows(sa, sb, shift, tolerance=TOL)
    direct = pair_metric_rows(sa, sb, shift, tolerance=TOL, method="direct")
    assert fast == direct and fast
    exact = metric_context(sa, sb, shift, tolerance=TOL, presence_fraction=0.01)["block_sum"]
    assert exact == (shape_a[0] % 8 == shape_a[1] % 8 == shape_b[0] % 8 == shape_b[1] % 8 == 0 and not valid)
    # 독립 계산: 원본 좌표에서 직접 센다
    dy, dx = shift[0] * 8, shift[1] * 8
    y0, y1, x0, x1 = max(0, dy), min(shape_a[0], shape_b[0] + dy), max(0, dx), min(shape_a[1], shape_b[1] + dx)
    ma, mb = (random_mask(shape_a, 1) > 0), (random_mask(shape_b, 4) > 0)
    omega = np.ones((y1 - y0, x1 - x0), dtype=bool)
    if valid:
        va, vb = np.ones(shape_a, bool), np.ones(shape_b, bool)
        va[10:30, 10:30] = vb[10:30, 10:30] = False
        omega = va[y0:y1, x0:x1] & vb[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
        ma, mb = ma & va, mb & vb
    row = rows_by(fast).get(("binding", "gate/l1"))
    assert row["area_a_px"] == int((ma[y0:y1, x0:x1] & omega).sum()) and row["area_b_px"] == int((mb[y0 - dy:y1 - dy, x0 - dx:x1 - dx] & omega).sum())


# ---- Ω: 경계에 인공 경계가 없다 -----------------------------------------------------------------------------------------
def test_a_mask_that_fills_omega_has_no_boundary_even_with_a_valid_hole():
    shape = (64, 64)
    full = np.ones(shape, dtype=np.float32)
    row = rows_by(metrics(product("A", shape, {("gate", "l1"): full}), product("B", shape, {("gate", "l1"): full})))[("binding", "gate/l1")]
    assert row["status"] == "both" and row["dice"] == 1.0 and row["fill_f1"] == 1.0
    assert row["boundary_f1"] is None and row["boundary_precision"] is None and row["boundary_recall"] is None
    hole = np.ones(shape, dtype=np.uint8)
    hole[16:48, 16:48] = 0  # 안쪽 valid 구멍: 마스크는 Ω 를 꽉 채우지만 구멍 둘레에 인공 경계가 생기면 안 된다
    holed = metrics(product("A", shape, {("gate", "l1"): full}, valid=hole), product("B", shape, {("gate", "l1"): full}, valid=hole))
    assert rows_by(holed)[("binding", "gate/l1")]["boundary_f1"] is None and rows_by(holed)[("binding", "gate/l1")]["dice"] == 1.0
    omega = np.ones((20, 20), bool)
    omega[5:15, 5:15] = False
    assert not inner_boundary(omega, omega).any()  # 구멍과 배열 밖은 전경으로 본다
    inside = np.zeros((20, 20), bool)
    inside[5:15, 5:15] = True
    assert inner_boundary(inside, np.ones((20, 20), bool)).sum() == 36  # 10×10 사각형의 둘레(8 근방 침식)


def test_shifted_comparison_inside_the_overlap_uses_the_shifted_omega():
    shape_a, shape_b = (64, 64), (48, 48)
    a = product("A", shape_a, {("gate", "l1"): rect(shape_a, 0, 64, 0, 64)})
    b = product("B", shape_b, {("gate", "l1"): rect(shape_b, 0, 48, 0, 48)})
    row = rows_by(metrics(a, b, shift=(4, 6)))[("binding", "gate/l1")]  # 래스터 이동 (4, 6) = 원본 (8, 12): B 는 A 안 (8,12)~(56,60) 에 통째로 놓인다
    assert row["area_a_px"] == row["area_b_px"] == 48 * 48 and row["dice"] == 1.0 and row["boundary_f1"] is None
    clipped = rows_by(metrics(a, b, shift=(20, 20)))[("binding", "gate/l1")]  # 원본 (40, 40): 겹침은 24×24
    assert clipped["area_a_px"] == clipped["area_b_px"] == 24 * 24 and clipped["boundary_f1"] is None


# ---- 쌍 집계 ----------------------------------------------------------------------------------------------------------
def test_aggregates_skip_blank_items_and_are_blank_without_valid_items():
    shape = (64, 64)
    full = np.ones(shape, dtype=np.float32)
    same_full = metrics(product("A", shape, {("gate", "l1"): full}), product("B", shape, {("gate", "l1"): full}))
    columns = pair_metric_columns(same_full, "CNT")
    assert columns["electrode_similarity"] is None and columns["usage_jaccard_electrodes"] == 1.0  # 경계 없는 role 은 평균에서 빠지고 유효 항목이 없으면 빈칸
    assert pair_metric_columns([], "CNT") == {"usage_jaccard_electrodes": None, "electrode_similarity": None, "pattern_similarity": None, "cnt_similarity": None}
    empty = {("gate", "l1"): np.zeros(shape)}
    only_cnt = metrics(product("A", shape, empty, cnt=rect(shape, 10, 30, 10, 30)), product("B", shape, empty, cnt=rect(shape, 10, 30, 10, 30)))
    assert [r["kind"] for r in only_cnt] == ["cnt"] and pair_metric_columns(only_cnt, "CNT")["usage_jaccard_electrodes"] is None  # 전극 Jaccard 분모 0 → 빈칸
    mixed = metrics(product("A", shape, {("gate", "l1"): full, ("data", "l2"): rect(shape, 8, 40, 8, 40)}),
                    product("B", shape, {("gate", "l1"): full, ("data", "l2"): rect(shape, 8, 40, 8, 40)}))
    assert pair_metric_columns(mixed, "CNT")["electrode_similarity"] == 1.0  # gate 는 빈칸이라 빼고 data 만 평균
    one_sided = metrics(product("A", shape, {("gate", "l1"): rect(shape, 8, 40, 8, 40)}), product("B", shape, empty))
    assert pair_metric_columns(one_sided, "CNT")["electrode_similarity"] == 0.0  # 한쪽만 role → 0
    assert pair_metric_columns(same_full, "LAYER")["pattern_similarity"] is None  # Layer 행이 없으면 빈칸


def test_rows_are_sorted_by_kind_then_key():
    shape = (64, 64)
    elements = {("pixel", "l2"): rect(shape, 8, 20, 8, 20), ("gate", "l1"): rect(shape, 30, 50, 30, 50)}
    rows = metrics(product("A", shape, elements, cnt=rect(shape, 2, 6, 2, 6)), product("B", shape, elements, cnt=rect(shape, 2, 6, 2, 6)))
    assert [(r["kind"], r["key"]) for r in rows] == [("binding", "gate/l1"), ("binding", "pixel/l2"), ("cnt", "CNT"), ("role", "gate"), ("role", "pixel")]


# ---- CLI·출력 ---------------------------------------------------------------------------------------------------------
def run_cli(root, name, split, manifest, *extra):
    split_path = root / f"{name}-split.json"
    split_path.write_text(json.dumps(split))
    out = root / f"{name}.csv"
    main(["--manifest", str(manifest), "--split", str(split_path), "--out", str(out), "--factor", str(FACTOR), "--margin", "0.0", *extra])
    return out


def test_cli_writes_bindings_csv_and_new_columns_without_nan_or_inf(case, capsys):
    manifest, split, root, _ = case
    out = run_cli(root, "a", split, manifest)
    capsys.readouterr()
    bindings = Path_text(str(out) + ".bindings.csv")
    scores = Path_text(str(out) + ".scores.csv")
    summary_text = Path_text(str(out) + ".summary.json")
    for text in (bindings, scores):
        for cell in (c.strip().lower() for line in csv.reader(text.splitlines()) for c in line):
            assert cell not in {"nan", "inf", "-inf", "+inf", "infinity"}
    json.loads(summary_text, parse_constant=lambda token: pytest.fail(f"non-finite JSON constant {token}"))
    rows = list(csv.DictReader(bindings.splitlines()))
    assert list(rows[0]) == list(screen_relations.BINDING_FIELDS) and rows
    keys = [(r["partition"], r["product_a"], r["product_b"], r["kind"], r["key"]) for r in rows]
    assert keys == sorted(keys) and set(r["kind"] for r in rows) <= {"binding", "role", "layer", "cnt"} and {r["status"] for r in rows} <= {"both", "only_a", "only_b"}
    score_rows = list(csv.DictReader(scores.splitlines()))
    assert list(score_rows[0])[-5:] == ["usage_jaccard_electrodes", "electrode_similarity", "pattern_similarity", "cnt_similarity", "layer_align_decided_dice"]
    assert all(r["pattern_similarity"] == "" for r in score_rows)  # 이 fixture 에는 Layer 가 없다
    assert all(0 <= float(r["usage_jaccard_electrodes"]) <= 1 for r in score_rows if r["usage_jaccard_electrodes"])
    summary = json.loads(summary_text)
    auc_block = summary["separation_auc_same_vs_other_lineage"]
    for name in ("similarity_layer", "usage_jaccard_electrodes", "electrode_similarity", "pattern_similarity", "cnt_similarity"):
        assert name in auc_block and name in auc_block["n_pairs_with_value"]
    assert auc_block["n_pairs_with_value"]["pattern_similarity"] == 0 and auc_block["pattern_similarity"] is None
    assert auc_block["n_pairs_with_value"]["similarity_layer"] == len(score_rows)
    assert auc_block["similarity_layer"] == auc_block["similarity"]  # Layer 가 없으면 같은 위치·같은 값


def Path_text(path):
    with open(path, encoding="utf-8") as stream:
        return stream.read()


def test_output_conflict_on_bindings_csv_alone_writes_nothing(case):
    manifest, split, root, _ = case
    out = root / "x.csv"
    blocker = root / "x.csv.bindings.csv"
    blocker.write_text("keep\n")
    split_path = root / "split.json"
    split_path.write_text(json.dumps(split))
    with pytest.raises(FileExistsError, match="bindings.csv"):
        main(["--manifest", str(manifest), "--split", str(split_path), "--out", str(out), "--factor", str(FACTOR)])
    assert sorted(p.name for p in root.glob("x.csv*")) == ["x.csv.bindings.csv"] and blocker.read_text() == "keep\n"
    custom = root / "elsewhere.csv"
    main(["--manifest", str(manifest), "--split", str(split_path), "--out", str(out), "--factor", str(FACTOR), "--bindings-out", str(custom)])
    assert custom.exists() and blocker.read_text() == "keep\n"
    with pytest.raises(FileExistsError, match="elsewhere.csv"):
        main(["--manifest", str(manifest), "--split", str(split_path), "--out", str(root / "y.csv"), "--factor", str(FACTOR), "--bindings-out", str(custom)])
    assert not list(root.glob("y.csv*"))


def test_presence_fraction_argument_and_incremental_delta_rows(case, capsys):
    manifest, split, root, _ = case
    base_split, base_split_path, base_relations, floor_json = incremental_inputs(root, split)
    out = run_cli(root, "inc", split, manifest, "--partitions", "train", "test", "--base-split", str(base_split_path), "--base-relations", str(base_relations),
                  "--floor-json", str(floor_json), "--presence-fraction", "0.5")
    capsys.readouterr()
    rows = list(csv.DictReader(Path_text(str(out) + ".bindings.csv").splitlines()))
    pairs = {(r["product_a"], r["product_b"]) for r in rows}
    assert pairs <= {("BUS_R00", "COMB_T00"), ("BUS_T00", "COMB_T00"), ("COMB_R00", "COMB_T00"), ("RING_R00", "RING_T00")} and pairs  # 기존×기존 쌍은 없다
    with pytest.raises(ValueError, match="presence_fraction"):
        screen(manifest, split, factor=FACTOR, presence_fraction=1.5)
    default = screen(manifest, split, factor=FACTOR, margin=0.0)
    strict = screen(manifest, split, factor=FACTOR, margin=0.0, presence_fraction=1.0)
    assert [p["similarity"] for p in default["pairs"]] == [p["similarity"] for p in strict["pairs"]]  # 판정 경로는 문턱과 무관하다
