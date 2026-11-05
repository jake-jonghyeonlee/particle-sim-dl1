"""dl1-size/T2: B 판정표(크기용 U 기준)·인수 검증·summary.judge.size·scores 새 열·증분 고정. tile 은 아직 연결하지 않았다(tile_p=None 경로)."""
import csv
import json

import pytest

import psdl1.screen_relations as sr
from psdl1.screen_report import REASON_TEXT
from psdl1.screen_relations import judge_verdict, load_floor_spec, main, write_csvs
from psdl1.split import SPLIT_VERSION
from test_screen_judge import (BIG, MARGIN, NAMES, R1, R1_AXES, R2_AXES, SAME, SMALL, by_pair, cli_base, five, judge_document, rounds, verdict,  # noqa: F401
                               world, write_spec)

import test_screen_judge as base_tests  # noqa: E402


@pytest.fixture(autouse=True)
def factor_eight(monkeypatch):
    """world 픽스처는 모듈 전역 FACTOR(2)로 screen() 을 부른다. size 기준은 factor 8 에서만 허용되므로 이 모듈에서는 8 로 바꾼다."""
    monkeypatch.setattr(base_tests, "FACTOR", 8)


U_FLOOR = 0.75
P_FLOOR, MARGIN_V = 0.5, 0.125  # P < 0.375 이면 단일 P 가 낮다. tile_p 경계는 0.375 / 0.5
OUT_OF_SUPPORT = {"size_tag": "size_differs", "size_delta_unit_px_y": 12, "size_delta_unit_px_x": 12}


def axes(p, u, **extra):
    return {"axis_p": p, "axis_u": u, **OUT_OF_SUPPORT, **extra}


def vd(ax, tile_p=None, size_u_floor=U_FLOOR, anchor="LAYER", same_lineage=False, u_floor=0.9):
    return judge_verdict(ax, anchor, same_lineage, P_FLOOR, u_floor, MARGIN_V, size_u_floor=size_u_floor, tile_p=tile_p)


# ---- judge_verdict 표 --------------------------------------------------------------------------------------------------
def test_size_u_floor_none_is_the_current_table():
    for ax in (axes(0.2, 0.9), axes(0.2, None), axes(0.2, 0.1), {**axes(0.2, 0.9), "size_delta_unit_px_y": 3, "size_delta_unit_px_x": 3}):
        assert vd(ax, size_u_floor=None) == judge_verdict(ax, "LAYER", False, P_FLOOR, 0.9, MARGIN_V)
    assert vd(axes(0.2, 0.9), tile_p=0.1, size_u_floor=None) == ("unknown", "size_differs_out_of_support", 3)  # tile_p 만으로는 아무것도 바뀌지 않는다


def test_u_none_stays_unknown():
    assert vd(axes(0.2, None)) == ("unknown", "size_differs_out_of_support", 3)
    assert vd(axes(0.2, None), tile_p=0.1) == ("unknown", "size_differs_out_of_support", 3)


@pytest.mark.parametrize("u, expected", [(0.5, "different"), (0.7499999, "different"), (0.75, "unknown"), (0.7500001, "unknown"), (1.0, "unknown")])
def test_size_u_floor_boundary_is_exact_and_u_equal_passes(u, expected):
    tier, reason, case = vd(axes(0.2, u))
    assert tier == expected
    assert (reason, case) == (("size_differs_usage_below_floor", 2) if expected == "different" else ("size_differs_out_of_support", 3))


def test_u_at_or_above_the_size_floor_without_tile_p_is_out_of_support():
    assert vd(axes(0.2, 0.9)) == ("unknown", "size_differs_out_of_support", 3)


@pytest.mark.parametrize("tile_p, expected", [
    (0.0, ("different", "size_differs_below_pattern_floor_tiled", 2)),
    (0.3749999, ("different", "size_differs_below_pattern_floor_tiled", 2)),
    (0.375, ("unknown", "near_boundary_tiled", None)),   # P = p_floor − margin 은 different 가 아니다
    (0.4999999, ("unknown", "near_boundary_tiled", None)),
    (0.5, ("unknown", "close_to_same_lineage", 3)),     # P = p_floor 는 경계가 아니다
    (0.9, ("unknown", "close_to_same_lineage", 3)),
])
def test_tile_p_rows_with_u_at_the_size_floor_and_above_the_usage_floor(tile_p, expected):
    assert vd(axes(0.2, 0.95), tile_p=tile_p) == expected


def test_tile_p_at_the_pattern_floor_with_low_usage_is_case_1():
    assert vd(axes(0.2, 0.8), tile_p=0.5) == ("unknown", "pattern_same_usage_differs", 1)  # 0.75 <= U < u_floor 0.9
    assert vd(axes(0.2, 0.9), tile_p=0.5) == ("unknown", "close_to_same_lineage", 3)       # U == u_floor 는 통과
    assert vd(axes(0.2, 0.8), tile_p=0.4) == ("unknown", "near_boundary_tiled", None)      # 경계 부근이 먼저다
    assert vd(axes(0.2, 0.8), tile_p=0.2) == ("different", "size_differs_below_pattern_floor_tiled", 2)


def test_tile_p_is_compared_decimal_exact():
    # 실제 기준 p_floor 0.45, margin 0.05: 0.45 - 0.05 는 float 로 0.4 가 아니어도 경계는 십진 0.40 / 0.45 이다
    def at(tile_p):
        return judge_verdict(axes(0.2, 0.95), "LAYER", False, 0.45, 0.9, 0.05, size_u_floor=U_FLOOR, tile_p=tile_p)[:2]
    assert at(0.39999999) == ("different", "size_differs_below_pattern_floor_tiled")
    assert at(0.40) == ("unknown", "near_boundary_tiled")
    assert at(0.44999999) == ("unknown", "near_boundary_tiled")
    assert at(0.45) == ("unknown", "close_to_same_lineage")


def test_adv4_a_single_p_at_or_above_the_p_floor_ignores_the_size_floor():
    for p, u in ((0.5, 0.1), (0.6, 0.1), (0.9, 0.3), (0.45, 0.1)):
        ax = axes(p, u)
        assert vd(ax) == judge_verdict(ax, "LAYER", False, P_FLOOR, 0.9, MARGIN_V), (p, u)
    assert vd(axes(0.6, 0.1))[:2] == ("unknown", "pattern_same_usage_differs")  # 크기 기준이 아니라 현행 규칙


def test_size_floor_does_not_touch_same_size_supported_or_same_lineage_pairs():
    same_size = {"axis_p": 0.2, "axis_u": 0.1, "size_tag": "same", "size_delta_unit_px_y": 0, "size_delta_unit_px_x": 0}
    assert vd(same_size) == ("different", "below_pattern_floor", 2)
    supported = axes(0.2, 0.1, size_delta_unit_px_y=3, size_delta_unit_px_x=3)
    assert vd(supported) == ("different", "below_pattern_floor", 2)
    assert vd(axes(0.2, 0.1), same_lineage=True) == ("similar", "same_lineage", 3)
    assert vd({"axis_p": None, "axis_u": 0.1, **OUT_OF_SUPPORT}) == ("unknown", "no_pattern_value", None)


def test_new_reasons_are_known_to_the_report():
    assert "near_boundary_tiled" in sr.JUDGE_REVIEW_REASONS
    for reason in ("near_boundary_tiled", "size_differs_usage_below_floor", "size_differs_below_pattern_floor_tiled"):
        assert REASON_TEXT[reason]


# ---- screen(): 인수 검증 ------------------------------------------------------------------------------------------------
def test_size_arguments_are_rejected_in_legacy(world):
    with pytest.raises(ValueError, match="three-axis"):
        world(SAME, judge=None, judge_size_u_floor=0.75)
    with pytest.raises(ValueError, match="three-axis"):
        world(SAME, judge="legacy", judge_size_u_floor=0.75)
    with pytest.raises(ValueError, match="three-axis"):
        world(SAME, judge="legacy", judge_tile_max_pairs=5)


def test_size_arguments_reject_other_factor_and_tolerance(world, monkeypatch):
    monkeypatch.setattr(base_tests, "FACTOR", 4)
    with pytest.raises(ValueError, match="factor 8 and tolerance 1"):
        world(SAME, margin=MARGIN, judge_size_u_floor=0.75)
    monkeypatch.setattr(base_tests, "FACTOR", 8)
    with pytest.raises(ValueError, match="factor 8 and tolerance 1"):
        world(SAME, margin=MARGIN, judge_size_u_floor=0.75, tolerance=0)
    with pytest.raises(ValueError, match="factor 8 and tolerance 1"):
        world(SAME, margin=MARGIN, judge_size_u_floor=0.75, tolerance=2)


def test_size_arguments_range_and_pairing(world):
    for bad in (-0.1, 1.5, float("nan"), True, "0.5"):
        with pytest.raises(ValueError, match="in \\[0,1\\]"):
            world(SAME, margin=MARGIN, judge_size_u_floor=bad)
    for bad in (-1, 1.5, True, "3"):
        with pytest.raises(ValueError, match="integer >= 0"):
            world(SAME, margin=MARGIN, judge_size_u_floor=0.75, judge_tile_max_pairs=bad)
    with pytest.raises(ValueError, match="needs judge_size_u_floor"):
        world(SAME, margin=MARGIN, judge_tile_max_pairs=10)  # max_pairs 단독
    for ok in (0, 1, 2000):
        assert "size" in world(SAME, margin=MARGIN, judge_size_u_floor=0.75, judge_tile_max_pairs=ok)["summary"]["judge"]
    for edge in (0, 1, 0.0, 1.0):
        assert world(SAME, margin=MARGIN, judge_size_u_floor=edge)["summary"]["judge"]["size"]["config"]["u_floor"] == float(edge)


def test_size_arguments_are_exclusive_with_floor_spec(rounds):
    with pytest.raises(ValueError, match="exclusive with floor_spec"):
        rounds(judge_size_u_floor=0.75)
    with pytest.raises(ValueError, match="exclusive with floor_spec"):
        rounds(judge_tile_max_pairs=3)
    spec_with_size = load_floor_spec(rounds.summary_path)
    with pytest.raises(ValueError, match="exclusive with floor_spec"):
        rounds(spec=spec_with_size, judge_size_u_floor=0.75, judge_tile_max_pairs=5)


# ---- screen(): 기능 on ----------------------------------------------------------------------------------------------
SIZE_AXES = {**SAME,
             ("A1", "B1"): axes(0.2, 0.5, **{}),                 # U < 0.75 → different (B)
             ("A1", "B2"): axes(0.2, 0.75),                      # U == 0.75 → 통과, tile 안 돌림 → out_of_support
             ("A2", "B1"): axes(0.2, None),                      # U 없음 → unknown
             ("A2", "B2"): {"axis_p": 0.2, "axis_u": 0.1},       # 같은 크기 → 현행 different
             ("A3", "B1"): axes(0.7, 0.1),                       # 단일 P >= p_floor → 현행 (U 낮음 case 1)
             ("A3", "B2"): axes(0.2, 0.95, **SMALL)}             # 지원 범위 안(ΔU 3 px)이지만 합성 world 는 LAYER 가 아니라 범위 밖


def size_run(world, **kwargs):
    kwargs.setdefault("judge_size_u_floor", U_FLOOR)
    return world(SIZE_AXES, margin=MARGIN, **kwargs)


def test_size_on_verdicts_and_tile_status(world):
    result = size_run(world)
    pairs = by_pair(result)
    assert verdict(pairs[("A1", "B1")]) == ("different", "size_differs_usage_below_floor", 2)
    assert verdict(pairs[("A1", "B2")]) == ("unknown", "size_differs_out_of_support", 3)
    assert verdict(pairs[("A2", "B1")]) == ("unknown", "size_differs_out_of_support", 3)
    assert verdict(pairs[("A2", "B2")]) == ("different", "below_pattern_floor", 2)
    assert verdict(pairs[("A3", "B1")]) == ("unknown", "pattern_same_usage_differs", 1)  # adv:4
    assert verdict(pairs[("A3", "B2")]) == ("unknown", "size_differs_out_of_support", 3)
    assert pairs[("A1", "B1")]["tile_status"] == "usage_below_floor"
    assert pairs[("A1", "B2")]["tile_status"] == "geometry_unsupported"  # 합성 world 의 프레임은 margin 1000 px 보다 작아 tile 을 지원하지 않는다
    assert pairs[("A3", "B2")]["tile_status"] == "geometry_unsupported"
    assert pairs[("A2", "B1")]["tile_status"] == "not_candidate"  # U 없음
    assert pairs[("A2", "B2")]["tile_status"] == "not_candidate"
    assert pairs[("A3", "B1")]["tile_status"] == "not_candidate"
    assert all(pairs[k]["tile_status"] == "not_candidate" for k in SAME)  # 같은 계보
    assert all(pairs[k]["judge_size_u_floor"] == U_FLOOR for k in pairs)
    assert all(pairs[k][key] is None for k in pairs for key in sr.JUDGE_SIZE_SCORE_FIELDS if key not in ("judge_size_u_floor", "tile_status", "tile_source"))
    assert pairs[("A1", "B2")]["tile_source"] in ("layer", "binding") and pairs[("A1", "B1")]["tile_source"] is None


def test_size_on_summary_structure(world):
    judge = size_run(world, judge_tile_max_pairs=7)["summary"]["judge"]
    assert judge["schema"] == "dl1_judge_v1"
    assert judge["size"]["config"] == {"u_floor": U_FLOOR, "enabled": True, "max_pairs": 7, "support_px": 60, "s_max_cells": 2, "tile_min_cells": 8,
                                       "factor": 8, "tolerance": 1}
    stats = judge["size"]["stats"]
    assert set(stats) == {"pairs_usage_below_floor", "pairs_tiled", "pairs_capped", "tile_status"}
    assert stats["pairs_usage_below_floor"] == 1 and stats["pairs_tiled"] == 2 and stats["pairs_capped"] == 0  # tile 을 부른 쌍 수(실패 포함)
    assert stats["tile_status"] == {"not_candidate": 12, "geometry_unsupported": 2, "usage_below_floor": 1}
    cases = judge["cases"]
    assert cases["size_differs_usage_below_floor"] == 1 and cases["size_differs_tiled"] == 0 and cases["size_differs_tiled_different"] == 0
    assert cases["size_differs_tile_unsupported"] == 2 and cases["size_differs_out_of_support"] == 3  # U 없음 쌍도 out_of_support
    assert cases["case_2"] == 4  # B 쌍 1 + 같은 크기 쌍 1 + 값을 안 정한 (A4, B1)·(A4, B2) 2


def test_default_tile_max_pairs_is_2000(world):
    assert size_run(world)["summary"]["judge"]["size"]["config"]["max_pairs"] == 2000 == sr.JUDGE_TILE_MAX_PAIRS


def test_size_off_leaves_no_trace(world):
    summary = world(SIZE_AXES, margin=MARGIN)["summary"]["judge"]
    assert "size" not in summary
    assert set(summary["cases"]) == {"case_1", "case_2", "case_2_capped", "case_3", "size_differs_out_of_support"}
    assert not any(key in p for p in world(SIZE_AXES, margin=MARGIN)["pairs"] for key in ("tile_status", "judge_size_u_floor"))


def test_scores_csv_gets_the_size_columns_only_when_on(world, tmp_path):
    on, off = world(SIZE_AXES, margin=MARGIN, judge_size_u_floor=U_FLOOR), world(SIZE_AXES, margin=MARGIN)
    write_csvs(on, tmp_path / "on.csv", tmp_path / "on.scores.csv")
    write_csvs(off, tmp_path / "off.csv", tmp_path / "off.scores.csv")
    header_on = (tmp_path / "on.scores.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    header_off = (tmp_path / "off.scores.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header_off == list(sr.SCORE_FIELDS + sr.JUDGE_SCORE_FIELDS)
    assert header_on == list(sr.SCORE_FIELDS + sr.JUDGE_SCORE_FIELDS + sr.JUDGE_SIZE_SCORE_FIELDS)
    assert sr.JUDGE_SIZE_SCORE_FIELDS == ("judge_size_u_floor", "tile_status", "tile_source", "tile_p", "tile_size_y_cells", "tile_size_x_cells", "tile_count",
                                          "tile_applied", "tile_step_replaced", "tile_radius_y_cells", "tile_radius_x_cells", "tile_shift_min_y", "tile_shift_max_y",
                                          "tile_shift_min_x", "tile_shift_max_x")
    rows = {(r["product_a"], r["product_b"]): r for r in csv.DictReader((tmp_path / "on.scores.csv").open(encoding="utf-8", newline=""))}
    assert rows[("A1", "B1")]["tile_status"] == "usage_below_floor" and rows[("A1", "B2")]["tile_status"] == "geometry_unsupported"
    assert {r["judge_size_u_floor"] for r in rows.values()} == {"0.75"}
    assert all(r[key] == "" for r in rows.values() for key in sr.JUDGE_SIZE_SCORE_FIELDS if key not in ("judge_size_u_floor", "tile_status", "tile_source"))
    assert "size_differs_usage_below_floor" in (tmp_path / "on.scores.csv").read_text(encoding="utf-8")
    off_text = (tmp_path / "off.scores.csv").read_text(encoding="utf-8")
    assert "tile_status" not in off_text and "usage_below_floor" not in off_text


def test_review_page_renders_with_the_new_reasons(world):
    from psdl1.screen_report import render_review
    result = size_run(world)
    page = render_review(result, 60, 6, 4)
    assert page.startswith("<!doctype html>") and "size_differs_out_of_support" in page
    assert result["summary"]["review"]["by_reason"]["size_differs_out_of_support"] >= 1


# ---- 증분 ------------------------------------------------------------------------------------------------------------
def r01_with_size(world, tmp_path, **kwargs):
    r01 = world(R1_AXES, names=R1, margin=MARGIN, judge="three-axis", **kwargs)
    out = tmp_path / "s_r01.csv"
    write_csvs(r01, out, tmp_path / "s_r01.scores.csv")
    path = tmp_path / "s_r01.summary.json"
    path.write_text(json.dumps(r01["summary"]))
    return r01, out, path


def r02_run(world, out, path, spec=None, **kwargs):
    base_split = {"schema_version": SPLIT_VERSION, "train": list(R1), "validation": [], "test": []}
    return world(R2_AXES, names=NAMES, old=tuple(R1), built=world.build(NAMES), base_split=base_split, base_relations=out,
                 floor_spec=load_floor_spec(path) if spec is None else spec, **kwargs)


def test_incremental_carries_the_size_config_and_recounts_the_stats(world, tmp_path):
    r01, out, path = r01_with_size(world, tmp_path, judge_size_u_floor=U_FLOOR, judge_tile_max_pairs=9)
    spec = load_floor_spec(path)
    assert spec["judge"]["size"] == {"u_floor": U_FLOOR, "enabled": True, "max_pairs": 9, "support_px": 60, "s_max_cells": 2, "tile_min_cells": 8, "factor": 8, "tolerance": 1}
    r02 = r02_run(world, out, path)
    size = r02["summary"]["judge"]["size"]
    assert size["config"] == r01["summary"]["judge"]["size"]["config"]
    assert size["stats"]["tile_status"] == {"not_candidate": 9}  # r01 이 아니라 이 라운드의 delta 쌍으로 다시 센다
    assert {p["judge_size_u_floor"] for p in r02["pairs"]} == {U_FLOOR}
    assert size["stats"] != r01["summary"]["judge"]["size"]["stats"]


def test_incremental_applies_the_stored_size_floor_to_new_pairs(world, tmp_path):
    _, out, path = r01_with_size(world, tmp_path, judge_size_u_floor=U_FLOOR)
    r02 = r02_run(world, out, path)
    assert "size" in r02["summary"]["judge"]
    # 같은 R2_AXES 에서 크기 쌍을 하나 끼우면 저장된 U 기준(0.75)으로 판정된다
    axes2 = {**R2_AXES, ("A1", "B2"): axes(0.2, 0.5)}
    base_split = {"schema_version": SPLIT_VERSION, "train": list(R1), "validation": [], "test": []}
    result = world(axes2, names=NAMES, old=tuple(R1), built=world.build(NAMES), base_split=base_split, base_relations=out, floor_spec=load_floor_spec(path))
    assert verdict(by_pair(result)[("A1", "B2")]) == ("different", "size_differs_usage_below_floor", 2)


def test_incremental_without_size_in_the_base_stays_off(world, tmp_path):
    _, out, path = r01_with_size(world, tmp_path)
    assert "size" not in load_floor_spec(path)["judge"]
    r02 = r02_run(world, out, path)
    assert "size" not in r02["summary"]["judge"]
    assert not any("tile_status" in p for p in r02["pairs"])


def test_incremental_rejects_explicit_size_arguments_with_the_base_spec(world, tmp_path):
    _, out, path = r01_with_size(world, tmp_path, judge_size_u_floor=U_FLOOR)
    with pytest.raises(ValueError, match="exclusive with floor_spec"):
        r02_run(world, out, path, judge_size_u_floor=U_FLOOR)
    with pytest.raises(ValueError, match="exclusive with floor_spec"):
        r02_run(world, out, path, judge_tile_max_pairs=1)
    plain = tmp_path / "plain"
    plain.mkdir()
    _, out2, path2 = r01_with_size(world, plain)
    with pytest.raises(ValueError, match="exclusive with floor_spec"):
        r02_run(world, out2, path2, judge_size_u_floor=0.5)  # base 에 size 가 없어도 명시 인수는 거부한다


def test_incremental_rejects_other_factor_or_tolerance_for_a_stored_size_config(world, tmp_path):
    _, out, path = r01_with_size(world, tmp_path, judge_size_u_floor=U_FLOOR)
    with pytest.raises(ValueError, match="factor 8 and tolerance 1"):
        r02_run(world, out, path, tolerance=0)


@pytest.mark.parametrize("changes, message", [
    ({"u_floor": 1.5}, "u_floor"), ({"u_floor": True}, "u_floor"), ({"u_floor": None}, "u_floor"), ({"enabled": False}, "enabled"),
    ({"max_pairs": -1}, "max_pairs"), ({"max_pairs": 1.5}, "max_pairs"), ({"max_pairs": True}, "max_pairs"),
    ({"factor": 4}, "factor"), ({"tolerance": 0}, "tolerance"), ({"support_px": 30}, "support_px"), ({"s_max_cells": 3}, "s_max_cells"),
    ({"tile_min_cells": 4}, "tile_min_cells"),
])
def test_load_floor_spec_validates_the_size_block(tmp_path, changes, message):
    config = {"u_floor": 0.75, "enabled": True, "max_pairs": 2000, "support_px": 60, "s_max_cells": 2, "tile_min_cells": 8, "factor": 8, "tolerance": 1, **changes}
    document = judge_document()
    document["judge"]["size"] = {"config": config, "stats": {}}
    with pytest.raises(ValueError, match=message):
        load_floor_spec(write_spec(tmp_path, document))


def test_load_floor_spec_rejects_a_size_block_without_config(tmp_path):
    for block in ({}, {"stats": {}}, {"config": {"u_floor": 0.5}}, [], None):
        document = judge_document()
        document["judge"]["size"] = block
        with pytest.raises(ValueError, match="size"):
            load_floor_spec(write_spec(tmp_path, document))


# ---- CLI ------------------------------------------------------------------------------------------------------------
def test_cli_size_arguments(world, tmp_path):
    manifest, split, split_path = world.build(["A1", "A2", "B1"])
    base = [*cli_base(manifest, split_path), "--judge", "three-axis", "--judge-floor-p", "0.5", "--judge-floor-u", "0.5"]
    out = tmp_path / "c1" / "r.csv"
    main([*base, "--out", str(out), "--judge-size-u-floor", "0.75", "--judge-tile-max-pairs", "3"])
    summary = json.loads(five(out)[3].read_text(encoding="utf-8"))
    assert summary["judge"]["size"]["config"]["u_floor"] == 0.75 and summary["judge"]["size"]["config"]["max_pairs"] == 3
    header = five(out)[1].read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[-len(sr.JUDGE_SIZE_SCORE_FIELDS):] == list(sr.JUDGE_SIZE_SCORE_FIELDS)
    out2 = tmp_path / "c2" / "r.csv"
    main([*base, "--out", str(out2)])  # 기능 off
    assert "size" not in json.loads(five(out2)[3].read_text(encoding="utf-8"))["judge"]
    assert "tile_status" not in five(out2)[1].read_text(encoding="utf-8").splitlines()[0]


def test_cli_size_argument_errors(world, tmp_path):
    manifest, split, split_path = world.build(["A1", "A2", "B1"])
    base = cli_base(manifest, split_path)
    floor_json = tmp_path / "f.json"
    floor_json.write_text(json.dumps(judge_document()))
    three = ["--judge", "three-axis", "--judge-floor-p", "0.5", "--judge-floor-u", "0.5"]
    for n, args in enumerate((["--floor-json", str(floor_json), "--judge-size-u-floor", "0.75"], ["--floor-json", str(floor_json), "--judge-tile-max-pairs", "1"],
                              ["--judge", "legacy", "--judge-size-u-floor", "0.75"], [*three, "--judge-size-u-floor", "1.5"], [*three, "--judge-size-u-floor", "-0.1"],
                              [*three, "--judge-size-u-floor", "nan"], [*three, "--judge-tile-max-pairs", "3"], [*three, "--judge-size-u-floor", "0.5", "--judge-tile-max-pairs", "-1"],
                              [*three, "--judge-size-u-floor", "0.5", "--judge-tile-max-pairs", "x"])):
        with pytest.raises(SystemExit) as stopped:
            main([*base, "--out", str(tmp_path / f"e{n}.csv"), *args])
        assert stopped.value.code == 2, args
    plain = cli_base(manifest, split_path)[:4]  # --factor 없이: 기본 factor 8 은 합성 world 의 factor 와 다르지만 인수 검증은 그 앞에서 걸린다
    other_factor = [*plain, "--factor", "4", *three, "--judge-size-u-floor", "0.75"]
    other_tolerance = [*base, *three, "--judge-size-u-floor", "0.75", "--tolerance", "0"]
    omitted_judge = [*base, "--judge-size-u-floor", "0.75"]  # judge 생략 = legacy
    for n, argv in enumerate((other_factor, other_tolerance, omitted_judge)):
        with pytest.raises(ValueError):
            main([*argv, "--out", str(tmp_path / f"v{n}.csv")])
    assert not any(path.name.startswith(("e", "v")) for path in tmp_path.iterdir() if path.suffix == ".csv")


# ---- 기능 off 는 현행 출력 ----------------------------------------------------------------------------------------------
def test_off_three_axis_and_legacy_have_no_size_keys_and_the_current_columns(world, tmp_path):
    manifest, split, split_path = world.build(NAMES)
    base = cli_base(manifest, split_path)
    legacy, three = tmp_path / "l" / "r.csv", tmp_path / "t" / "r.csv"
    main([*base, "--out", str(legacy)])
    main([*base, "--out", str(three), "--judge", "three-axis", "--judge-floor-p", "0.5", "--judge-floor-u", "0.5"])
    legacy_summary, three_summary = (json.loads(five(path)[3].read_text(encoding="utf-8")) for path in (legacy, three))
    assert "judge" not in legacy_summary and "size" not in three_summary["judge"]
    assert set(three_summary["judge"]["cases"]) == {"case_1", "case_2", "case_2_capped", "case_3", "size_differs_out_of_support"}
    assert five(legacy)[1].read_text(encoding="utf-8").splitlines()[0] == ",".join(sr.SCORE_FIELDS)
    assert five(three)[1].read_text(encoding="utf-8").splitlines()[0] == ",".join(sr.SCORE_FIELDS + sr.JUDGE_SCORE_FIELDS)
    for path in (legacy, three):
        text = "".join(p.read_text(encoding="utf-8") for p in five(path)[:4])
        assert "tile_" not in text and "judge_size_u_floor" not in text and "usage_below_floor" not in text
