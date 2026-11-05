"""실행기(psdl1_assets.py)의 --judge-size-u-floor·--judge-tile-max-pairs 인수 식별·조합 거부·재사용 전 검사·증분·test 구획 전달 (dl1-size/T5)."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from psdl1 import screen_relations  # noqa: E402
from test_runner_judge import (JUDGE_BLOCK, THREE, Booted, boot_stops, make_round_one, new_work, png, read_summary, run_assets,  # noqa: E402,F401
                               tree_digest)
from test_single_file import stage_globals, stages  # noqa: E402,F401

U = ["--judge-size-u-floor", "0.5"]


def size_block(u_floor=0.5, max_pairs=2000):
    config = {"u_floor": u_floor, "enabled": True, "max_pairs": max_pairs, **screen_relations.JUDGE_SIZE_FIXED}
    return {"config": config, "stats": {"pairs_usage_below_floor": 1, "pairs_tiled": 2, "pairs_capped": 0, "tile_status": {"ok": 2}}}


def judge_with_size(**kwargs):
    judge = copy.deepcopy(JUDGE_BLOCK)
    judge["size"] = size_block(**kwargs)
    return judge


def stops(g, function, capsys):
    with pytest.raises(SystemExit) as stopped:
        g[function]()
    err = capsys.readouterr().err
    assert stopped.value.code == 2 and "STOP:" in err
    return err


# ---- 인수 식별 ----

@pytest.mark.parametrize("args,expected", [
    (["--judge-size-u-floor", "0.5"], (0.5, None)),
    (["--judge-size-u-floor=0.25"], (0.25, None)),
    (["--judge-size-u-floor", "0.5", "--judge-tile-max-pairs", "10"], (0.5, 10)),
    (["--judge-tile-max-pairs=0", "--judge-size-u-floor=1"], (1.0, 0)),
    (["--judge-size-u-floor", "0.5", "--judge-size-u-floor=0.75", "--judge-tile-max-pairs", "3", "--judge-tile-max-pairs=4"], (0.75, 4)),
    (["--groups", "g.json"], (None, None))])
def test_size_arguments_are_identified_in_space_and_equals_forms_and_the_last_value_wins(stages, args, expected):
    g = stage_globals(stages, "psdl1_assets.py")
    assert g["_size_args"](args) == expected
    assert g["_judge_args"](args) == (None, None, None)  # 정확한 이름은 약어 검사에 걸리지 않는다


def test_size_request_fills_the_default_max_pairs_and_the_default_matches_screen_relations(stages):
    g = stage_globals(stages, "psdl1_assets.py")
    assert g["SIZE_MAX_PAIRS_DEFAULT"] == screen_relations.JUDGE_TILE_MAX_PAIRS
    assert g["_size_request"]([]) is None
    assert g["_size_request"](U) == {"u_floor": 0.5, "max_pairs": screen_relations.JUDGE_TILE_MAX_PAIRS}
    assert g["_size_request"]([*U, "--judge-tile-max-pairs=7"]) == {"u_floor": 0.5, "max_pairs": 7}


@pytest.mark.parametrize("rnd", [None, 1, 2])
@pytest.mark.parametrize("args", [["--judge-size", "0.5"], ["--judge-size-u", "0.5"], ["--judge-size-u-", "0.5"], ["--judge-size-u-flo=0.5"], ["--judge-size-u-floo", "0.5"],
                                   ["--judge-tile", "5"], ["--judge-tile-max", "5"], ["--judge-tile-max-", "5"], ["--judge-tile-max-pair=5"], ["--judge-tile-max-pair", "5"],
                                   ["--judge-s", "0.5"], ["--judge-t=5"], ["--judge-", "x"], ["--judge-size-u-floor"], ["--judge-tile-max-pairs"]])
def test_size_abbreviations_and_missing_values_stop_before_anything_starts(stages, tmp_path, rnd, args, capsys):
    g = stage_globals(stages, "psdl1_assets.py")
    g.update(WORK=str(tmp_path / "w"), ROUND=rnd, SCREEN_ARGS=[*THREE, *args])
    err = stops(g, "stage", capsys)
    assert " start " not in err and not (tmp_path / "w").exists()


@pytest.mark.parametrize("rnd", [None, 1, 2])
@pytest.mark.parametrize("args", [["--judge-size-u-floor", "1.5"], ["--judge-size-u-floor=-0.1"], ["--judge-size-u-floor", "nope"], ["--judge-size-u-floor=nan"],
                                   ["--judge-tile-max-pairs", "-1"], ["--judge-tile-max-pairs=2.5"], ["--judge-tile-max-pairs", "many"]])
def test_size_value_range_stops_before_anything_starts(stages, tmp_path, rnd, args, capsys):
    g = stage_globals(stages, "psdl1_assets.py")
    g.update(WORK=str(tmp_path / "w"), ROUND=rnd, SCREEN_ARGS=[*THREE, "--judge-size-u-floor", "0.5", *args])
    err = stops(g, "stage", capsys)
    assert " start " not in err and not (tmp_path / "w").exists()


def test_exact_size_arguments_are_accepted_in_round_one(stages):
    g = stage_globals(stages, "psdl1_assets.py")
    for rnd in (None, 1):
        g.update(ROUND=rnd, SCREEN_ARGS=[*THREE, *U, "--judge-tile-max-pairs=0"])
        g["_check_screen_args"]()


def test_shorter_judge_abbreviations_keep_their_old_message(stages, capsys):
    """size 인수가 없는 기존 접두어 약어의 STOP 문구는 그대로다."""
    g = stage_globals(stages, "psdl1_assets.py")
    g.update(ROUND=None, SCREEN_ARGS=["--judge-floor", "0.5"])
    assert "write the full name (--judge, --judge-floor-p or --judge-floor-u)" in stops(g, "_check_screen_args", capsys)


# ---- 조합 거부 ----

@pytest.mark.parametrize("args", [
    [*U],  # judge 없음 → 기본 legacy
    ["--judge", "legacy", *U],
    ["--judge=legacy", "--judge-tile-max-pairs", "5", *U]])
def test_size_arguments_need_the_three_axis_judge(stages, capsys, args):
    g = stage_globals(stages, "psdl1_assets.py")
    g.update(ROUND=None, SCREEN_ARGS=args)
    assert "without --judge three-axis" in stops(g, "_check_screen_args", capsys)


def test_max_pairs_alone_stops(stages, capsys):
    g = stage_globals(stages, "psdl1_assets.py")
    g.update(ROUND=None, SCREEN_ARGS=[*THREE, "--judge-tile-max-pairs", "5"])
    assert "--judge-tile-max-pairs without --judge-size-u-floor" in stops(g, "_check_screen_args", capsys)


@pytest.mark.parametrize("floor_json", [["--floor-json", "s.json"], ["--floor-json=s.json"], ["--floor-j", "s.json"]])
def test_size_arguments_with_a_user_floor_json_stop(stages, capsys, floor_json):
    g = stage_globals(stages, "psdl1_assets.py")
    g.update(ROUND=None, SCREEN_ARGS=[*THREE, *U, *floor_json])
    assert "together with --floor-json" in stops(g, "_check_screen_args", capsys)


# ---- 재사용 전 검사: 판정 기대값 ----

def test_expectation_reads_the_size_config_and_the_problem_compares_only_the_config(stages):
    g = stage_globals(stages, "psdl1_assets.py")
    sized, plain = {"judge": judge_with_size(u_floor=0.5, max_pairs=10)}, {"judge": copy.deepcopy(JUDGE_BLOCK)}
    expect = g["_judge_expectation"](sized, "three-axis")
    assert expect["size"] == {"u_floor": 0.5, "max_pairs": 10} and g["_judge_expectation"](plain, "three-axis")["size"] is None
    assert g["_judge_problem"](sized, expect) is None
    other_stats = copy.deepcopy(sized)
    other_stats["judge"]["size"]["stats"] = {"pairs_tiled": 99}
    assert g["_judge_problem"](other_stats, expect) is None  # stats 는 비교하지 않는다(adv:3)
    for wanted in (None, {"u_floor": 0.4, "max_pairs": 10}, {"u_floor": 0.5, "max_pairs": 11}):
        assert "judge.size.config" in g["_judge_problem"](sized, {**expect, "size": wanted})
    assert "judge.size.config" in g["_judge_problem"](plain, expect)
    assert g["_judge_problem"](plain, {**g["_judge_expectation"](plain, "three-axis")}) is None


def test_expectation_stops_on_an_unusable_size_config(stages, capsys):
    g = stage_globals(stages, "psdl1_assets.py")
    for broken in ({"config": {"u_floor": 2.0, "max_pairs": 1}}, {"config": {"u_floor": 0.5, "max_pairs": -1}}, {"config": {"u_floor": 0.5}}, {"config": 3}):
        judge = copy.deepcopy(JUDGE_BLOCK)
        judge["size"] = broken
        with pytest.raises(SystemExit):
            g["_judge_expectation"]({"judge": judge}, "three-axis")
        assert "STOP:" in capsys.readouterr().err


# ---- 재사용 전 검사: 실제 실행 ----

def digest_after_stop(stages, tmp_path, png, work, args, text):
    before = tree_digest(work)
    result = run_assets(stages, tmp_path, png, work, args)
    assert result.returncode == 2 and "STOP:" in result.stderr and "judge.size.config" in result.stderr and text in result.stderr, (args, result.stderr[-2000:])
    assert str(work / "relations") in result.stderr
    assert tree_digest(work) == before


def test_size_work_rerun_skips_and_every_other_size_request_stops_and_keeps_files(stages, tmp_path, png):
    work = new_work(tmp_path)
    first = run_assets(stages, tmp_path, png, work, [*THREE, *U])
    assert first.returncode == 0, first.stderr[-3000:]
    for part in ("train", "test"):
        assert read_summary(work, part)["judge"]["size"]["config"]["u_floor"] == 0.5
        assert read_summary(work, part)["judge"]["size"]["config"]["max_pairs"] == screen_relations.JUDGE_TILE_MAX_PAIRS
    before = tree_digest(work)
    for args in ([*THREE, *U], [*THREE, "--judge-size-u-floor=0.5", "--judge-tile-max-pairs", str(screen_relations.JUDGE_TILE_MAX_PAIRS)]):  # 기본값을 적어도 같은 설정이다
        again = run_assets(stages, tmp_path, png, work, args)
        assert again.returncode == 0 and "skip relations" in again.stderr and "STOP" not in again.stderr, again.stderr[-2000:]
    assert tree_digest(work) == before
    digest_after_stop(stages, tmp_path, png, work, THREE, "no size judge")  # 있음 → 없음
    digest_after_stop(stages, tmp_path, png, work, [*THREE, "--judge-size-u-floor", "0.4"], "u_floor 0.4")  # u_floor 다름
    digest_after_stop(stages, tmp_path, png, work, [*THREE, *U, "--judge-tile-max-pairs", "5"], "max_pairs 5")  # max_pairs 다름
    assert run_assets(stages, tmp_path, png, work, [*THREE, *U]).returncode == 0  # 원래 요청이면 다시 건너뛴다


def test_work_without_size_asked_for_size_stops_and_keeps_files(stages, tmp_path, png):
    work = new_work(tmp_path)
    assert run_assets(stages, tmp_path, png, work, THREE).returncode == 0
    assert "size" not in read_summary(work, "train")["judge"]
    digest_after_stop(stages, tmp_path, png, work, [*THREE, *U], "but this run fixes u_floor 0.5")  # 없음 → 있음
    assert run_assets(stages, tmp_path, png, work, THREE).returncode == 0


def test_train_done_and_test_missing_with_changed_size_stops(stages, tmp_path, png):
    import shutil
    work = new_work(tmp_path)
    assert run_assets(stages, tmp_path, png, work, [*THREE, *U]).returncode == 0
    (work / "relations" / "relations.csv").unlink()
    shutil.rmtree(work / "relations" / "test")
    digest_after_stop(stages, tmp_path, png, work, [*THREE, "--judge-size-u-floor", "0.4"], "u_floor 0.4")
    done = run_assets(stages, tmp_path, png, work, [*THREE, *U])  # 같은 요청이면 test 구획을 마저 만든다
    assert done.returncode == 0, done.stderr[-3000:]
    assert read_summary(work, "test")["judge"]["size"]["config"] == read_summary(work, "train")["judge"]["size"]["config"]


# ---- test 구획 전달 ----

def run_round_one_with_spy(stages, tmp_path, screen_args, train_summary):
    g = stage_globals(stages, "psdl1_assets.py")
    seen = {}

    def spy(manifest, split_path, part, out_dir, extra, drop_quantile=False, expect=None):
        seen[part] = ([str(a) for a in extra], drop_quantile, expect)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "relations.csv").write_text("a,b,similar\n" if part == "train" else "c,d,different\n", encoding="utf-8")
        return train_summary

    work = tmp_path / "w"
    (work / "export").mkdir(parents=True)
    (work / "export" / "manifest.json").write_text("{}", encoding="utf-8")
    (work / "reports").mkdir()
    (work / "reports" / "grid-report.json").write_text("{}", encoding="utf-8")
    g.update(WORK=str(work), ROUND=None, SCREEN_ARGS=screen_args, _screen=spy, _boot=lambda: None, _call=lambda *a, **k: None,
             _write_json_once=lambda *a, **k: None, STACK_ORDER=None,
             _split=lambda *a, **k: {"train": ["W1", "W2"], "validation": [], "test": ["W3", "W4"]})
    g["stage"]()
    return seen


@pytest.mark.parametrize("user,expected_size", [
    ([], ["--judge-size-u-floor", "0.5", "--judge-tile-max-pairs", "7"]),
    (["--judge-size-u-floor", "0.5"], ["--judge-tile-max-pairs", "7"]),
    (["--judge-size-u-floor=0.5", "--judge-tile-max-pairs=7"], [])])
def test_test_partition_gets_the_train_size_config_unless_the_user_gave_it(stages, tmp_path, user, expected_size):
    train = {"floor": {"global": None}, "judge": judge_with_size(u_floor=0.5, max_pairs=7)}
    train["judge"]["mode"] = "three-axis"
    seen = run_round_one_with_spy(stages, tmp_path, [*THREE, *user], train)
    assert seen["train"][0] == []
    assert seen["test"][0] == ["--judge-floor-p", "0.8", "--judge-floor-u", "0.6", *expected_size] and seen["test"][1] is True
    assert seen["test"][2]["size"] == {"u_floor": 0.5, "max_pairs": 7}


def test_test_partition_gets_no_size_arguments_without_a_train_size_config(stages, tmp_path):
    train = {"floor": {"global": None}, "judge": copy.deepcopy(JUDGE_BLOCK)}
    seen = run_round_one_with_spy(stages, tmp_path, THREE, train)
    assert seen["test"][0] == ["--judge-floor-p", "0.8", "--judge-floor-u", "0.6"] and seen["test"][2]["size"] is None
    assert seen["train"][2]["size"] is None


def test_round_one_expectation_for_the_train_part_carries_the_requested_size(stages, tmp_path):
    train = {"floor": {"global": None}, "judge": judge_with_size(u_floor=0.5, max_pairs=3)}
    seen = run_round_one_with_spy(stages, tmp_path, [*THREE, *U, "--judge-tile-max-pairs", "3"], train)
    assert seen["train"][2]["size"] == {"u_floor": 0.5, "max_pairs": 3}
    assert seen["test"][0] == ["--judge-floor-p", "0.8", "--judge-floor-u", "0.6"]


# ---- 증분 ----

@pytest.mark.parametrize("args", [U, ["--judge-size-u-floor=0.5"], ["--judge-tile-max-pairs", "5"], [*THREE, *U], ["--judge", "legacy", *U]])
def test_round_two_refuses_explicit_size_arguments(stages, tmp_path, args, capsys):
    g = stage_globals(stages, "psdl1_assets.py")
    work = tmp_path / "w"
    make_round_one(work, judge_with_size())
    before = tree_digest(work)
    g.update(WORK=str(work), ROUND=2, SCREEN_ARGS=args)
    boot_stops(g)
    err = stops(g, "stage", capsys)
    assert "round-1" in err and "judge.size.config" in err and "start" not in err
    assert tree_digest(work) == before and not (work / "rounds" / "r02").exists()


def run_round_two_with_spy(stages, tmp_path, judge, screen_args):
    g = stage_globals(stages, "psdl1_assets.py")
    work = tmp_path / "w"
    r1 = make_round_one(work, judge)
    seen = {}

    def spy(manifest, split_path, part, out_dir, extra, drop_quantile=False, expect=None):
        seen[part] = ([str(a) for a in extra], expect)
        raise Booted()

    g.update(WORK=str(work), ROUND=2, SCREEN_ARGS=screen_args, _screen=spy, _boot=lambda: None, _call=lambda *a, **k: None, _call_keeping_failed=lambda *a, **k: None,
             _merge_step=lambda *a, **k: None, _write_json_once=lambda *a, **k: None, STACK_ORDER=None,
             _split=lambda *a, **k: {"train": ["W1", "W2"], "validation": [], "test": ["W3", "W4"]})
    with pytest.raises(Booted):
        g["stage"]()
    return seen, r1


def test_round_two_uses_the_saved_size_config_through_the_round_one_floor_json(stages, tmp_path):
    seen, r1 = run_round_two_with_spy(stages, tmp_path, judge_with_size(u_floor=0.5, max_pairs=7), [])
    extra, expect = seen["train"]
    assert extra[extra.index("--floor-json") + 1] == str(r1 / "relations" / "train" / "relations.csv.summary.json")
    assert not any(a.startswith("--judge-size") or a.startswith("--judge-tile") for a in extra)  # 인수는 따로 넘기지 않고 floor-json 의 저장값을 쓴다
    assert expect["size"] == {"u_floor": 0.5, "max_pairs": 7}


def test_round_two_without_round_one_size_expects_none(stages, tmp_path):
    seen, _ = run_round_two_with_spy(stages, tmp_path, copy.deepcopy(JUDGE_BLOCK), [])
    assert seen["train"][1]["size"] is None


def test_round_two_legacy_round_one_has_no_size_expectation(stages, tmp_path):
    seen, _ = run_round_two_with_spy(stages, tmp_path, None, [])
    assert seen["train"][1] == {"mode": "legacy"}
