"""dl1-judge/T3: three-axis 판정표·기준값·증분 고정·CLI. P·U·크기 축은 합성 값으로 정하고(pair_axes 를 바꿔 끼움) 판정 쪽만 본다."""
import csv
import json

import pytest

import psdl1.screen_relations as sr
from conftest import make_product
from psdl1.screen_relations import judge_verdict, load_floor_spec, main, screen, write_csvs
from psdl1.split import SPLIT_VERSION
from test_pretrain import write_manifest
from test_screen_relations import FACTOR, embed

OFFSETS = {"A1": (10, 10), "A2": (30, 22), "A3": (20, 40), "A4": (40, 8), "B1": (14, 30), "B2": (36, 36)}
NAMES = ["A1", "A2", "A3", "A4", "B1", "B2"]
MARGIN = 0.125  # p_floor 0.5 → different 은 P < 0.375. 둘 다 2 진 분수라 경계가 정확하다
SAME_VALUES = {("A1", "A2"): 0.50, ("A1", "A3"): 0.90, ("A1", "A4"): 0.60, ("A2", "A3"): 0.80, ("A2", "A4"): 0.70, ("A3", "A4"): 0.95, ("B1", "B2"): 0.85}
SAME = {pair: {"axis_p": v, "axis_u": v} for pair, v in SAME_VALUES.items()}  # 정렬 값 .50 .60 .70 .80 .85 .90 .95 → floor(최솟값) .5
DEFAULT = {"axis_p": 0.1, "axis_u": 0.1}  # 값을 안 정한 쌍(다른 계보)은 멀다
SMALL = {"size_tag": "size_differs", "size_ratio_y": 0.9, "size_ratio_x": 0.9, "size_delta_unit_px_y": 3, "size_delta_unit_px_x": 3}
BIG = {**SMALL, "size_delta_unit_px_y": 12, "size_delta_unit_px_x": 12}


def pair_order(names, old=()):
    ordered = sorted(names)
    return [(a, b) for i, a in enumerate(ordered) for b in ordered[i + 1:] if not (a in old and b in old)]


@pytest.fixture
def world(tmp_path, monkeypatch):
    real = sr.pair_axes  # 실행마다 바꿔 끼우므로 처음 것을 붙잡아 둔다

    def build(names):
        root = tmp_path / ("m_" + "_".join(names))
        root.mkdir(exist_ok=True)
        products = [embed(make_product("bus", 48), n, n[0], 96, OFFSETS[n]) for n in names]
        manifest = write_manifest(root, products, 64)
        split = {"schema_version": SPLIT_VERSION, "train": list(names), "validation": [], "test": []}
        path = root / "split.json"
        path.write_text(json.dumps(split))
        return manifest, split, path

    def run(axes, names=NAMES, old=(), groups=None, **kwargs):
        """axes = {(a, b): 축 값 덮어쓰기}. 쌍은 screen() 이 pair_axes 를 부르는 순서(정렬 순서)대로 대응한다."""
        manifest, split, _ = kwargs.pop("built", None) or build(names)
        if "floor_spec" not in kwargs:  # 증분은 base summary 의 모드를 잇는다. 그 밖에는 three-axis 가 기본이고 judge=None 이면 legacy 다
            kwargs.setdefault("judge", "three-axis")
        order = iter(pair_order(names, old))
        monkeypatch.setattr(sr, "pair_axes", lambda rows, anchor, sa, sb: {**real(rows, anchor, sa, sb), **axes.get(next(order), DEFAULT)})
        group_path = None
        if groups is not None:
            group_path = tmp_path / f"groups_{len(list(tmp_path.iterdir()))}.json"
            group_path.write_text(json.dumps(groups))
        return screen(manifest, split, groups=group_path, factor=FACTOR, **kwargs)
    run.build = build
    return run


def by_pair(result):
    return {(p["product_a"], p["product_b"]): p for p in result["pairs"]}


def verdict(p):
    return p["tier"], p["reason"], p["judge_case"]


def test_every_row_of_the_judge_table(world):
    axes = {**SAME,
            ("A1", "B1"): {"axis_p": None, "axis_u": 0.9},                      # P 없음
            ("A1", "B2"): {"axis_p": 0.2, "axis_u": 0.9},                       # 경우 2
            ("A2", "B1"): {"axis_p": 0.375, "axis_u": 0.9},                     # P = p_floor − margin 은 unknown
            ("A2", "B2"): {"axis_p": 0.45, "axis_u": 0.9},                      # 경계 부근
            ("A3", "B1"): {"axis_p": 0.7, "axis_u": None},                      # U 없음
            ("A3", "B2"): {"axis_p": 0.7, "axis_u": 0.3},                       # 경우 1
            ("A4", "B1"): {"axis_p": 0.7, "axis_u": 0.8},                       # 일반 close
            ("A4", "B2"): {"axis_p": 0.7, "axis_u": 0.8, **SMALL}}              # 경우 3
    pairs = by_pair(world(axes, margin=MARGIN))
    assert verdict(pairs[("A1", "B1")]) == ("unknown", "no_pattern_value", None)
    assert verdict(pairs[("A1", "B2")]) == ("different", "below_pattern_floor", 2)
    assert verdict(pairs[("A2", "B1")]) == ("unknown", "near_boundary", None)
    assert verdict(pairs[("A2", "B2")]) == ("unknown", "near_boundary", None)
    assert verdict(pairs[("A3", "B1")]) == ("unknown", "pattern_same_usage_unknown", None)
    assert verdict(pairs[("A3", "B2")]) == ("unknown", "pattern_same_usage_differs", 1)
    assert verdict(pairs[("A4", "B1")]) == ("unknown", "close_to_same_lineage", None)
    assert verdict(pairs[("A4", "B2")]) == ("unknown", "close_to_same_lineage", 3)
    assert all(pairs[k]["tier"] == "similar" and pairs[k]["reason"] == "same_lineage" for k in SAME)


def test_floors_are_recorded_per_pair_and_in_the_judge_block(world):
    result = world({**SAME}, margin=MARGIN)
    p = by_pair(result)[("A1", "B2")]
    assert p["judge_p_floor"] == 0.5 and p["judge_u_floor"] == 0.5
    judge = result["summary"]["judge"]
    assert judge["schema"] == "dl1_judge_v1" and judge["mode"] == "three-axis" and judge["margin"] == MARGIN and judge["quantile"] is None
    assert judge["p"] == {"global": 0.5, "per_group": {"-": 0.5}, "source": "calibrated", "n_values": 7}
    assert judge["u"] == {"global": 0.5, "per_group": {"-": 0.5}, "source": "calibrated", "n_values": 7}
    assert set(judge["cases"]) == {"case_1", "case_2", "case_3", "size_differs_out_of_support"} and set(judge["auc"]) == {"axis_p", "axis_u", "n_pairs_with_value"}
    assert result["summary"]["floor"]["global"] is not None  # similarity floor 도 계산되면 기록한다(L5)


def test_same_lineage_pair_is_similar_with_case_3_when_the_size_differs():
    axes = {"axis_p": 0.1, "axis_u": 0.1, **SMALL}
    assert judge_verdict(axes, "LAYER", True, 0.5, 0.5, 0.125) == ("similar", "same_lineage", 3)
    assert judge_verdict({**axes, "size_tag": "same"}, "LAYER", True, 0.5, 0.5, 0.125) == ("similar", "same_lineage", None)


@pytest.mark.parametrize("anchor, delta, expected", [("LAYER", 3, "different"), ("LAYER", 7, "different"), ("LAYER", 8, "unknown"), ("LAYER", 12, "unknown"),
                                                      ("CNT", 3, "unknown"), ("ELECTRODES", 0, "unknown")])
def test_g_size_support_range_for_size_differs(anchor, delta, expected):
    axes = {"axis_p": 0.2, "axis_u": 0.9, "size_tag": "size_differs", "size_delta_unit_px_y": delta, "size_delta_unit_px_x": 0}
    tier, reason, case = judge_verdict(axes, anchor, False, 0.5, 0.5, 0.125)
    assert tier == expected
    assert (reason, case) == (("below_pattern_floor", 2) if expected == "different" else ("size_differs_out_of_support", 3))


def test_g_size_does_not_touch_same_size_pairs_and_checks_both_axes():
    same_size = {"axis_p": 0.2, "axis_u": 0.9, "size_tag": "same", "size_delta_unit_px_y": 0, "size_delta_unit_px_x": 0}
    assert judge_verdict(same_size, "CNT", False, 0.5, 0.5, 0.125) == ("different", "below_pattern_floor", 2)
    x_only = {**same_size, "size_tag": "size_differs", "size_delta_unit_px_x": -9}
    assert judge_verdict(x_only, "LAYER", False, 0.5, 0.5, 0.125)[1] == "size_differs_out_of_support"


def test_out_of_support_size_pair_is_unknown_in_screen_and_counted(world):
    # 합성 실행에는 Layer 사이드카가 없어 metrics_anchor 가 LAYER 가 아니다 → 지원 범위 밖
    axes = {**SAME, ("A1", "B2"): {"axis_p": 0.2, "axis_u": 0.9, **SMALL}, ("A1", "B1"): {"axis_p": 0.2, "axis_u": 0.9}}
    result = world(axes, margin=MARGIN)
    pairs = by_pair(result)
    assert verdict(pairs[("A1", "B2")]) == ("unknown", "size_differs_out_of_support", 3)
    assert verdict(pairs[("A1", "B1")]) == ("different", "below_pattern_floor", 2)  # 같은 크기는 영향 없음
    assert result["summary"]["judge"]["cases"]["size_differs_out_of_support"] == 1
    assert result["summary"]["unknown_by_reason"]["size_differs_out_of_support"] == 1


def test_cross_group_cap_is_applied_in_p_order_and_weakest_uses_p(world):
    groups = {"A1": "g1", "A2": "g1", "A3": "g1", "A4": "g1", "B1": "g1", "B2": "g2"}
    low = {"A1": 0.10, "A2": 0.20, "A3": 0.25, "A4": 0.30}  # P 순서는 이름 순서와 반대: 유사도가 같으면 이름 순이 앞서 A1 이 남는다
    axes = {**SAME, **{(a, "B1"): {"axis_p": 0.1, "axis_u": 0.1} for a in low}, **{(a, "B2"): {"axis_p": v, "axis_u": 0.1} for a, v in low.items()}}
    result = world(axes, groups=groups, margin=MARGIN)
    pairs = by_pair(result)
    assert all(pairs[(a, "B1")]["tier"] == "different" for a in low)  # 그룹 안 different 4 → 그룹 밖 허용 ceil(0.2*4) = 1
    assert pairs[("A4", "B2")]["tier"] == "different"
    assert [pairs[(a, "B2")]["reason"] for a in ("A1", "A2", "A3")] == ["cross_group_capped"] * 3
    assert "weakest_same_lineage" in pairs[("A1", "A2")]["review"] and "weakest_same_lineage" in pairs[("A1", "A4")]["review"]  # P .50 .60
    assert "weakest_same_lineage" not in pairs[("A3", "A4")]["review"]


def test_different_near_boundary_review_uses_p_and_the_p_floor(world):
    axes = {**SAME, ("A1", "B1"): {"axis_p": 0.3, "axis_u": 0.1}, ("A2", "B1"): {"axis_p": 0.05, "axis_u": 0.1}}
    pairs = by_pair(world(axes, margin=MARGIN))
    assert "different_near_boundary" in pairs[("A1", "B1")]["review"]  # 0.3 >= 0.5 − 0.125 − 0.15
    assert "different_near_boundary" not in pairs[("A2", "B1")]["review"]


def test_no_same_lineage_pair_names_the_axis_that_needs_an_explicit_floor(world):
    with pytest.raises(ValueError, match=r"axis_p.*--judge-floor-p"):
        world({}, names=["A1", "B1"], judge="three-axis")
    with pytest.raises(ValueError, match=r"axis_u.*--judge-floor-u"):
        world({}, names=["A1", "B1"], judge="three-axis", judge_floor_p=0.5)
    with pytest.raises(ValueError, match=r"axis_p.*--judge-floor-p"):
        world({}, names=["A1", "B1"], judge="three-axis", judge_floor_u=0.5)


def test_both_axes_explicit_needs_no_same_lineage_and_similarity_floor_may_be_missing(world):
    result = world({("A1", "B1"): {"axis_p": 0.4, "axis_u": 0.9}}, names=["A1", "B1"], judge="three-axis", judge_floor_p=0.5, judge_floor_u=0.6, margin=MARGIN)
    assert result["summary"]["floor"]["global"] is None and result["floor"]["global"] is None
    judge = result["summary"]["judge"]
    assert judge["p"] == {"global": 0.5, "per_group": {}, "source": "explicit", "n_values": 0} and judge["u"]["global"] == 0.6
    assert verdict(result["pairs"][0]) == ("unknown", "near_boundary", None)  # 0.4 는 [0.375, 0.5)


def test_one_explicit_axis_leaves_the_other_calibrated_and_gets_no_group_floor(world):
    groups = {n: "g1" if n[0] == "A" else "g2" for n in NAMES}
    judge = world(SAME, groups=groups, margin=MARGIN, judge="three-axis", judge_floor_p=0.4)["summary"]["judge"]
    assert judge["p"]["global"] == 0.4 and judge["p"]["per_group"] == {} and judge["p"]["source"] == "explicit"
    assert judge["u"]["global"] == 0.5 and judge["u"]["per_group"] == {"g1": 0.5} and judge["u"]["source"] == "calibrated"


def test_group_floor_needs_enough_valid_values_inside_the_group(world):
    groups = {n: "g1" if n[0] == "A" else "g2" for n in NAMES}
    full = world(SAME, groups=groups, margin=MARGIN)["summary"]["judge"]
    assert full["p"]["per_group"] == {"g1": 0.5} and full["p"]["n_values"] == 7  # g2 는 같은 계보 1쌍(< min_group_pairs)
    missing = {**SAME, **{k: {"axis_p": None, "axis_u": SAME_VALUES[k]} for k in (("A1", "A2"), ("A1", "A3"), ("A1", "A4"), ("A2", "A3"))}}
    thin = world(missing, groups=groups, margin=MARGIN)["summary"]["judge"]
    assert thin["p"]["per_group"] == {} and thin["p"]["global"] == 0.7 and thin["p"]["n_values"] == 3  # g1 유효 값 2 < 3, global 은 유효 값 .7 .85 .95 의 최솟값
    assert thin["u"]["per_group"] == {"g1": 0.5} and thin["u"]["n_values"] == 7  # u 는 영향 없다


def test_floor_quantile_applies_only_to_calibrated_axes(world):
    judge = world(SAME, margin=MARGIN, judge="three-axis", floor_quantile=0.5, judge_floor_p=0.4)["summary"]["judge"]
    assert judge["p"]["global"] == 0.4 and judge["u"]["global"] == 0.8 and judge["quantile"] == 0.5  # u: floor(0.5*6)=3 → .80


def test_judge_floor_in_legacy_is_an_error(world):
    with pytest.raises(ValueError, match="three-axis"):
        world(SAME, judge=None, judge_floor_p=0.5)  # 생략 = legacy
    with pytest.raises(ValueError, match="three-axis"):
        world(SAME, judge="legacy", judge_floor_u=0.5)
    with pytest.raises(ValueError, match="in \\[0,1\\]"):
        world(SAME, judge="three-axis", judge_floor_p=1.5)
    with pytest.raises(ValueError, match="judge must be"):
        world(SAME, judge="other")


# ---- 증분 ----------------------------------------------------------------------------------------------------------
R1 = ["A1", "A2", "A3", "B1"]
R1_AXES = {("A1", "A2"): {"axis_p": 0.5, "axis_u": 0.5}, ("A1", "A3"): {"axis_p": 0.9, "axis_u": 0.9}, ("A2", "A3"): {"axis_p": 0.8, "axis_u": 0.8}}
R2_AXES = {("A1", "A4"): {"axis_p": 0.45, "axis_u": 0.45}, ("A2", "A4"): {"axis_p": 0.45, "axis_u": 0.45}, ("A3", "A4"): {"axis_p": 0.45, "axis_u": 0.45},
           ("B1", "B2"): {"axis_p": 0.85, "axis_u": 0.85}, ("A1", "B2"): {"axis_p": 0.45, "axis_u": 0.9}}


@pytest.fixture
def rounds(world, tmp_path):
    """r01(제품 4개)을 three-axis 로 돌려 summary 와 relations 를 쓰고, r02 를 돌리는 함수를 돌려준다."""
    r01 = world(R1_AXES, names=R1, margin=MARGIN, judge="three-axis")
    out = tmp_path / "r01.csv"
    write_csvs(r01, out, tmp_path / "r01.scores.csv")
    summary_path = tmp_path / "r01.summary.json"
    summary_path.write_text(json.dumps(r01["summary"]))
    base_split = {"schema_version": SPLIT_VERSION, "train": list(R1), "validation": [], "test": []}
    built = world.build(NAMES)

    def r02(spec=None, **kwargs):
        spec = load_floor_spec(summary_path) if spec is None else spec
        return world(R2_AXES, names=NAMES, old=tuple(R1), built=built, base_split=base_split, base_relations=out, floor_spec=spec, **kwargs)
    r02.r01, r02.summary_path, r02.base_split, r02.base_relations = r01, summary_path, base_split, out
    return r02


def test_r01_three_axis_summary_is_loadable_and_stores_the_fixed_values(rounds):
    spec = load_floor_spec(rounds.summary_path)
    assert spec["judge"]["margin"] == MARGIN and spec["judge"]["p"] == {"global": 0.5, "groups": {"-": 0.5}, "source": "calibrated", "n_values": 3}


def test_incremental_uses_stored_p_u_margin_and_does_not_recalibrate(rounds):
    result = rounds()
    judge = result["summary"]["judge"]
    assert judge["margin"] == MARGIN and judge["p"]["global"] == 0.5 and judge["p"]["per_group"] == {"-": 0.5} and judge["u"]["per_group"] == {"-": 0.5}
    assert judge["p"]["n_values"] == 3  # 저장된 n_values 를 그대로 잇는다
    pairs = by_pair(result)
    # 재보정했다면 같은 계보 delta 최솟값 .45 가 새 기준이 되어 close_to_same_lineage 가 된다. 저장값 .5 에서는 경계 부근이다.
    assert verdict(pairs[("A1", "B2")]) == ("unknown", "near_boundary", None)
    assert result["summary"]["mode"] == "incremental" and result["summary"]["delta_pairs"] == {"train": 9}  # 새×새 1 + 새×기존 2×4


def test_incremental_margin_is_stored_and_a_different_one_is_rejected(rounds):
    with pytest.raises(ValueError, match="margin differs from the base round summary"):
        rounds(margin=0.05)
    assert rounds(margin=MARGIN)["summary"]["judge"]["margin"] == MARGIN


def test_incremental_inherits_the_mode_and_rejects_a_different_explicit_one(rounds):
    assert "judge" in rounds()["summary"] and "judge" in rounds(judge="three-axis")["summary"]
    with pytest.raises(ValueError, match="judge mode differs from the base round summary"):
        rounds(judge="legacy")


def test_incremental_legacy_summary_inherits_legacy_and_rejects_three_axis(rounds):
    legacy_spec = {"global": 0.5, "groups": {}}
    assert "judge" not in rounds(spec=legacy_spec)["summary"]
    with pytest.raises(ValueError, match="judge mode differs from the base round summary"):
        rounds(spec=legacy_spec, judge="three-axis")


def test_incremental_rejects_explicit_judge_floors_and_quantile(rounds):
    with pytest.raises(ValueError, match="exclusive with floor_spec"):
        rounds(judge_floor_p=0.5)
    with pytest.raises(ValueError, match="exclusive with floor and floor_spec"):
        rounds(floor_quantile=0.5)


# ---- load_floor_spec ----------------------------------------------------------------------------------------------
def judge_document(**changes):
    block = {"global": 0.5, "per_group": {"g": 0.4}, "source": "calibrated", "n_values": 3}
    document = {"floor": {"global": None, "per_group": {}, "margin": 0.05},
                "judge": {"schema": "dl1_judge_v1", "mode": "three-axis", "margin": 0.05, "quantile": None, "p": dict(block), "u": dict(block), "cases": {}, "auc": {}}}
    for key, value in changes.items():
        target = document["judge"] if key != "floor" else document
        target[key] = value
    return document


def write_spec(tmp_path, document):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(document))
    return path


def test_load_floor_spec_accepts_a_null_similarity_floor_only_with_a_judge_block(tmp_path):
    spec = load_floor_spec(write_spec(tmp_path, judge_document()))
    assert spec["global"] is None and spec["judge"]["p"]["groups"] == {"g": 0.4}
    legacy = {"floor": {"global": None, "per_group": {}}}
    with pytest.raises(ValueError, match="floor.global"):
        load_floor_spec(write_spec(tmp_path, legacy))
    assert "judge" not in load_floor_spec(write_spec(tmp_path, {"floor": {"global": 0.5, "per_group": {}}}))


@pytest.mark.parametrize("changes", [{"schema": "x"}, {"mode": "legacy"}, {"margin": "0.1"}, {"margin": float("nan")}, {"quantile": True},
                                      {"p": {"global": 0.5, "per_group": {}, "source": 1, "n_values": 3}},
                                      {"u": {"global": 0.5, "per_group": {}, "source": "x", "n_values": -1}},
                                      {"u": {"global": 0.5, "per_group": {}, "source": "x", "n_values": True}},
                                      {"p": {"global": None, "per_group": {}, "source": "x", "n_values": 1}},
                                      {"p": {"global": 0.5, "per_group": {"g": "x"}, "source": "x", "n_values": 1}}, {"u": None}])
def test_load_floor_spec_validates_the_judge_block(tmp_path, changes):
    with pytest.raises(ValueError, match="judge block"):
        load_floor_spec(write_spec(tmp_path, judge_document(**changes)))


# ---- CLI·출력 ------------------------------------------------------------------------------------------------------
def cli_base(manifest, split_path):
    return ["--manifest", str(manifest), "--split", str(split_path), "--factor", str(FACTOR)]


def five(out):
    return [out, out.with_name(out.name + ".scores.csv"), out.with_name(out.name + ".bindings.csv"), out.with_name(out.name + ".summary.json"),
            out.with_name(out.name + ".review.html")]


def test_three_axis_run_without_a_similarity_floor_completes_all_five_outputs(world, tmp_path, capsys):
    manifest, _, split_path = world.build(["A1", "B1"])
    out = tmp_path / "o3" / "r.csv"
    main([*cli_base(manifest, split_path), "--out", str(out), "--judge", "three-axis", "--judge-floor-p", "0.5", "--judge-floor-u", "0.5"])
    assert all(path.exists() for path in five(out))
    summary = json.loads(five(out)[3].read_text(encoding="utf-8"))
    assert summary["floor"]["global"] is None and summary["judge"]["p"]["source"] == "explicit" and summary["review"]["html_bytes"] > 0
    assert "—" in five(out)[4].read_text(encoding="utf-8")  # similarity floor 를 "—" 로 쓴다
    header = five(out)[1].read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[:len(sr.SCORE_FIELDS)] == list(sr.SCORE_FIELDS) and header[len(sr.SCORE_FIELDS):] == list(sr.JUDGE_SCORE_FIELDS)
    capsys.readouterr()


def test_three_axis_scores_csv_has_the_axis_columns(world, tmp_path):
    manifest, split, split_path = world.build(["A1", "A2", "B1"])
    out = tmp_path / "o4" / "r.csv"
    main([*cli_base(manifest, split_path), "--out", str(out), "--judge", "three-axis", "--judge-floor-p", "0.5", "--judge-floor-u", "0.5"])
    rows = list(csv.DictReader(five(out)[1].open(encoding="utf-8", newline="")))
    assert len(rows) == 3 and all(r["axis_p"] and r["axis_u"] and r["size_tag"] == "same" and r["judge_p_floor"] == "0.5" and r["judge_u_floor"] == "0.5" for r in rows)
    assert {r["size_ratio_y"] for r in rows} == {"1.0"} and {r["size_delta_unit_px_y"] for r in rows} == {"0"}


def test_cli_judge_floor_errors(world, tmp_path):
    manifest, split, split_path = world.build(["A1", "A2", "B1"])
    base = cli_base(manifest, split_path)
    floor_json = tmp_path / "f.json"
    floor_json.write_text(json.dumps(judge_document()))
    with pytest.raises(SystemExit) as stopped:
        main([*base, "--out", str(tmp_path / "x1.csv"), "--floor-json", str(floor_json), "--judge-floor-p", "0.5"])
    assert stopped.value.code == 2
    with pytest.raises(ValueError, match="three-axis"):
        main([*base, "--out", str(tmp_path / "x2.csv"), "--judge-floor-p", "0.5"])  # 비증분 + 생략 = legacy
    with pytest.raises(ValueError, match="three-axis"):
        main([*base, "--out", str(tmp_path / "x3.csv"), "--judge", "legacy", "--judge-floor-u", "0.5"])
    with pytest.raises(SystemExit) as stopped:
        main([*base, "--out", str(tmp_path / "x4.csv"), "--judge", "other"])
    assert stopped.value.code == 2
    assert not any((tmp_path / name).exists() for name in ("x1.csv", "x2.csv", "x3.csv", "x4.csv"))


def test_legacy_default_and_explicit_legacy_write_identical_bytes_and_no_judge_keys(world, tmp_path):
    manifest, split, split_path = world.build(NAMES)
    base = cli_base(manifest, split_path)
    default, explicit = tmp_path / "d" / "r.csv", tmp_path / "e" / "r.csv"
    main([*base, "--out", str(default)])
    main([*base, "--out", str(explicit), "--judge", "legacy"])
    for a, b in zip(five(default), five(explicit)):
        assert a.read_bytes().replace(str(default).encode(), b"OUT") == b.read_bytes().replace(str(explicit).encode(), b"OUT"), a.name
    assert "judge" not in json.loads(five(default)[3].read_text(encoding="utf-8"))
    header = five(default)[1].read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(sr.SCORE_FIELDS) and "axis_p" not in header and "judge_case" not in header
    assert "size_differs" not in five(default)[1].read_text(encoding="utf-8") and "pattern" not in " ".join(json.loads(five(default)[3].read_text(encoding="utf-8"))["unknown_by_reason"])
