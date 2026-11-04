"""실행기(psdl1_assets.py)의 --judge 인수 식별·test 구획 기준 전달·재사용 전 모드 검증 (dl1-judge/T4)."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from png_helpers import write_wide_import_fixture  # noqa: E402
from psdl1.split import SPLIT_VERSION  # noqa: E402
from test_single_file import assets_values, run_script, stage_globals, stage_script, stages  # noqa: E402,F401

NAMES = ["W1", "W2", "W3", "W4"]
METADATA = {"metadata_schema_version": "dl1_png_metadata_v1", "models": {
    n: {"lineage_id": "F1" if n in ("W1", "W2") else "F2", "design_version": "V1"} for n in NAMES}}
THREE = ["--judge", "three-axis"]
LEGACY = ["--floor", "2"]


def summary_path(work, part):
    return Path(work) / "relations" / part / "relations.csv.summary.json"


def read_summary(work, part):
    return json.loads(summary_path(work, part).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_digest(work):
    return {str(p): digest(p) for p in sorted((Path(work) / "relations").rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def png(tmp_path_factory):
    root = tmp_path_factory.mktemp("judge-png")
    for name in NAMES:
        write_wide_import_fixture(root, model=name)
    return root


def new_work(tmp_path, name="work"):
    work = tmp_path / name
    (work / "inputs").mkdir(parents=True)
    (work / "inputs" / "split.json").write_text(json.dumps({"schema_version": SPLIT_VERSION, "train": ["W1", "W2"], "validation": [], "test": ["W3", "W4"]}))
    return work


def run_assets(stages, tmp_path, png, work, screen_args, tag="s"):
    folder = tmp_path / tag
    folder.mkdir(exist_ok=True)
    script = stage_script(stages, "psdl1_assets.py", folder, **assets_values(png, work, METADATA=METADATA, NO_PREVIEW=True, SCREEN_ARGS=screen_args))
    return run_script(script, tmp_path)


# ---- 인수 식별 (X5) ----

@pytest.mark.parametrize("args,expected", [
    (["--judge", "three-axis"], ("three-axis", None, None)),
    (["--judge=three-axis"], ("three-axis", None, None)),
    (["--judge=legacy", "--judge", "three-axis"], ("three-axis", None, None)),
    (["--judge", "three-axis", "--judge-floor-p", "0.5", "--judge-floor-u=0.25"], ("three-axis", 0.5, 0.25)),
    (["--groups", "g.json"], (None, None, None))])
def test_judge_arguments_are_identified_in_space_and_equals_forms(stages, args, expected):
    g = stage_globals(stages, "psdl1_assets.py")
    assert g["_judge_args"](args) == expected


@pytest.mark.parametrize("rnd", [None, 1, 2])
@pytest.mark.parametrize("args", [["--judge", "three-axis"], ["--judge=three-axis"]])
def test_exact_judge_is_accepted_in_every_round(stages, rnd, args):
    g = stage_globals(stages, "psdl1_assets.py")
    g.update(ROUND=rnd, SCREEN_ARGS=args)
    g["_check_screen_args"]()


@pytest.mark.parametrize("rnd", [None, 1, 2])
@pytest.mark.parametrize("args", [["--jud", "three-axis"], ["--judge-floor", "0.5"], ["--judge-f=x"], ["--j", "x"], ["--judge-floor-", "0.5"], ["--judge-fl", "0.5"],
                                   ["--judge", "bogus"], ["--judge"], ["--judge-floor-p", "nope"], ["--judge-floor-u=2"]])
def test_judge_abbreviations_and_bad_values_stop_before_anything_starts(stages, tmp_path, rnd, args, capsys):
    g = stage_globals(stages, "psdl1_assets.py")
    g.update(WORK=str(tmp_path / "w"), ROUND=rnd, SCREEN_ARGS=args)
    with pytest.raises(SystemExit) as stopped:
        g["stage"]()
    err = capsys.readouterr().err
    assert stopped.value.code == 2 and "STOP:" in err and " start " not in err and not (tmp_path / "w").exists()


@pytest.mark.parametrize("args", [["--judge-floor-p", "0.5"], ["--judge-floor-u=0.5"], [*THREE, "--judge-floor-p", "0.5"], [*THREE, "--judge-floor-u=0.5"],
                                   ["--floor-quantile", "0.5"], ["--floor-quantile=0.5", *THREE]])
def test_round_two_refuses_new_criteria(stages, tmp_path, args, capsys):
    g = stage_globals(stages, "psdl1_assets.py")
    g.update(WORK=str(tmp_path / "w"), ROUND=2, SCREEN_ARGS=args)
    with pytest.raises(SystemExit) as stopped:
        g["stage"]()
    err = capsys.readouterr().err
    assert stopped.value.code == 2 and "STOP:" in err and "round-1" in err and not (tmp_path / "w").exists()


@pytest.mark.parametrize("args", [["--judge-floor-p", "0.5"], ["--judge", "legacy", "--judge-floor-u", "0.5"], [*THREE, "--judge-floor-p", "0.5", "--floor-json", "s.json"]])
def test_round_one_judge_floor_needs_three_axis_and_no_floor_json(stages, args, capsys):
    g = stage_globals(stages, "psdl1_assets.py")
    g.update(ROUND=None, SCREEN_ARGS=args)
    with pytest.raises(SystemExit) as stopped:
        g["_check_screen_args"]()
    assert stopped.value.code == 2 and "STOP:" in capsys.readouterr().err
    g.update(ROUND=None, SCREEN_ARGS=[*THREE, "--judge-floor-p", "0.5", "--judge-floor-u=0.25"])
    g["_check_screen_args"]()


# ---- ROUND >= 2: r01 모드 상속·불일치 ----

def make_round_one(work, judge):
    """stage() 가 r01 파일 요구를 통과하도록 최소한만 만든다. judge 가 None 이면 legacy 요약."""
    r1 = Path(work) / "rounds" / "r01"
    for rel in ("export/manifest.json", "export/bundle-integrity.json", "inputs/split.json", "relations/relations.csv"):
        (r1 / rel).parent.mkdir(parents=True, exist_ok=True)
        (r1 / rel).write_text("", encoding="utf-8")
    summary = {"floor": {"global": 0.3}}
    if judge is not None:
        summary["judge"] = judge
    (r1 / "relations" / "train").mkdir(parents=True, exist_ok=True)
    (r1 / "relations" / "train" / "relations.csv.summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return r1


JUDGE_BLOCK = {"schema": "dl1_judge_v1", "mode": "three-axis", "margin": 0.05, "quantile": None,
               "p": {"global": 0.8, "per_group": {}, "source": "calibrated", "n_values": 3}, "u": {"global": 0.6, "per_group": {}, "source": "calibrated", "n_values": 3}}


class Booted(Exception):
    pass


def boot_stops(g):
    def boot():
        raise Booted()
    g["_boot"] = boot


@pytest.mark.parametrize("judge,args", [(JUDGE_BLOCK, []), (JUDGE_BLOCK, THREE), (JUDGE_BLOCK, ["--judge=three-axis"])])
def test_round_two_inherits_or_accepts_the_round_one_three_axis_mode(stages, tmp_path, capsys, judge, args):
    g = stage_globals(stages, "psdl1_assets.py")
    make_round_one(tmp_path / "w", judge)
    g.update(WORK=str(tmp_path / "w"), ROUND=2, SCREEN_ARGS=args)
    boot_stops(g)
    with pytest.raises(Booted):
        g["stage"]()
    err = capsys.readouterr().err
    assert ("inheriting three-axis" in err) == (args == [])


def test_round_two_legacy_round_one_inherits_silently_and_accepts_explicit_legacy(stages, tmp_path, capsys):
    g = stage_globals(stages, "psdl1_assets.py")
    make_round_one(tmp_path / "w", None)
    boot_stops(g)
    for args in ([], ["--judge", "legacy"]):
        g.update(WORK=str(tmp_path / "w"), ROUND=2, SCREEN_ARGS=args)
        with pytest.raises(Booted):
            g["stage"]()
        assert "inheriting" not in capsys.readouterr().err


@pytest.mark.parametrize("judge,args", [(JUDGE_BLOCK, ["--judge", "legacy"]), (None, THREE), (None, ["--judge=three-axis"])])
def test_round_two_explicit_mode_that_differs_from_round_one_stops(stages, tmp_path, capsys, judge, args):
    g = stage_globals(stages, "psdl1_assets.py")
    work = tmp_path / "w"
    r1 = make_round_one(work, judge)
    before = tree_digest(work)
    g.update(WORK=str(work), ROUND=2, SCREEN_ARGS=args)
    boot_stops(g)
    with pytest.raises(SystemExit) as stopped:
        g["stage"]()
    err = capsys.readouterr().err
    assert stopped.value.code == 2 and "fixed in round 1" in err and str(r1) in err
    assert tree_digest(work) == before


# ---- r01 test 구획 인수 (X3·L5) ----

def test_test_partition_gets_the_train_p_u_floor_and_no_quantile(stages, tmp_path, png):
    work = new_work(tmp_path)
    result = run_assets(stages, tmp_path, png, work, [*THREE, "--floor-quantile", "0.5"])
    assert result.returncode == 0, result.stderr[-3000:]
    train, test = read_summary(work, "train"), read_summary(work, "test")
    assert train["judge"]["mode"] == "three-axis" and test["judge"]["mode"] == "three-axis"
    assert train["judge"]["quantile"] == 0.5 and test["judge"]["quantile"] is None
    for axis in ("p", "u"):
        assert test["judge"][axis]["global"] == train["judge"][axis]["global"] and test["judge"][axis]["source"] == "explicit"
    assert train["floor"]["quantile"] == 0.5 and "quantile" not in test["floor"]
    if train["floor"]["global"] is not None:
        assert test["floor"]["global"] == train["floor"]["global"]


@pytest.mark.parametrize("train_floor,user,expected", [
    (0.25, [], ["--judge-floor-p", "0.8", "--judge-floor-u", "0.6", "--floor", "0.25"]),
    (None, [], ["--judge-floor-p", "0.8", "--judge-floor-u", "0.6"]),
    (0.25, ["--judge-floor-p", "0.8"], ["--judge-floor-u", "0.6", "--floor", "0.25"]),
    (0.25, ["--floor", "0.7"], ["--judge-floor-p", "0.8", "--judge-floor-u", "0.6"]),
    (0.25, ["--floor-json", "s.json"], [])])
def test_three_axis_stage_hands_the_test_partition_the_train_criteria(stages, tmp_path, train_floor, user, expected):
    """실제 stage() 의 f 절을 _screen 대체로 돌려 test 호출에 간 인수를 본다: p·u 값, quantile 없음, train floor 가 있을 때만 --floor."""
    g = stage_globals(stages, "psdl1_assets.py")
    seen = {}

    def spy(manifest, split_path, part, out_dir, extra, drop_quantile=False, expect=None):
        seen[part] = ([str(a) for a in extra], drop_quantile)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "relations.csv").write_text("a,b,similar\n" if part == "train" else "c,d,different\n", encoding="utf-8")
        return {"floor": {"global": train_floor}, "judge": {"mode": "three-axis", "margin": 0.05, "p": {"global": 0.8}, "u": {"global": 0.6}}}

    work = tmp_path / "w"
    (work / "export").mkdir(parents=True)
    (work / "export" / "manifest.json").write_text("{}", encoding="utf-8")
    (work / "reports").mkdir()
    (work / "reports" / "grid-report.json").write_text("{}", encoding="utf-8")
    g.update(WORK=str(work), ROUND=None, SCREEN_ARGS=[*THREE, "--floor-quantile", "0.5", *user], _screen=spy, _boot=lambda: None, _call=lambda *a, **k: None,
             _write_json_once=lambda *a, **k: None, STACK_ORDER=None,
             _split=lambda *a, **k: {"train": ["W1", "W2"], "validation": [], "test": ["W3", "W4"]})
    if "--floor-json" in user or "--floor" in user:
        g["SCREEN_ARGS"] = [*THREE, *user]  # the quantile conflicts with a user floor
    g["stage"]()
    assert seen["train"] == ([], False)
    assert seen["test"] == (expected, True)


# ---- 재사용 전 검증 (X4) ----

def test_legacy_complete_work_rerun_with_legacy_skips_as_before(stages, tmp_path, png):
    work = new_work(tmp_path)
    assert run_assets(stages, tmp_path, png, work, LEGACY).returncode == 0
    before = tree_digest(work)
    result = run_assets(stages, tmp_path, png, work, LEGACY)
    assert result.returncode == 0 and "skip relations" in result.stderr and "STOP" not in result.stderr
    assert tree_digest(work) == before


def test_legacy_complete_work_asked_for_three_axis_stops_and_keeps_files(stages, tmp_path, png):
    work = new_work(tmp_path)
    assert run_assets(stages, tmp_path, png, work, LEGACY).returncode == 0
    before = tree_digest(work)
    result = run_assets(stages, tmp_path, png, work, THREE)
    assert result.returncode == 2 and "STOP:" in result.stderr and "legacy judge" in result.stderr and str(work / "relations") in result.stderr
    assert tree_digest(work) == before


def test_three_axis_complete_work_rerun_skips_and_other_modes_stop(stages, tmp_path, png):
    work = new_work(tmp_path)
    assert run_assets(stages, tmp_path, png, work, THREE).returncode == 0
    before = tree_digest(work)
    result = run_assets(stages, tmp_path, png, work, THREE)
    assert result.returncode == 0 and "skip relations" in result.stderr and "STOP" not in result.stderr
    for args in (LEGACY, ["--judge", "legacy"], []):
        result = run_assets(stages, tmp_path, png, work, args)
        assert result.returncode == 2 and "three-axis judge" in result.stderr, args
    assert tree_digest(work) == before


def test_train_done_and_test_missing_checks_the_train_mode(stages, tmp_path, png):
    work = new_work(tmp_path)
    assert run_assets(stages, tmp_path, png, work, THREE).returncode == 0
    (work / "relations" / "relations.csv").unlink()
    shutil.rmtree(work / "relations" / "test")
    before = tree_digest(work)
    result = run_assets(stages, tmp_path, png, work, LEGACY)
    assert result.returncode == 2 and "STOP:" in result.stderr and "three-axis judge" in result.stderr
    assert tree_digest(work) == before
    result = run_assets(stages, tmp_path, png, work, THREE)  # the same mode finishes the missing test partition
    assert result.returncode == 0, result.stderr[-3000:]
    assert read_summary(work, "test")["judge"]["p"]["global"] == read_summary(work, "train")["judge"]["p"]["global"]


def test_explicit_floor_that_differs_from_the_finished_train_summary_stops(stages, tmp_path, png):
    work = new_work(tmp_path)
    assert run_assets(stages, tmp_path, png, work, [*THREE, "--judge-floor-p", "0.5", "--judge-floor-u", "0.4"]).returncode == 0
    train = read_summary(work, "train")
    assert train["judge"]["p"]["global"] == 0.5 and read_summary(work, "test")["judge"]["u"]["global"] == 0.4
    before = tree_digest(work)
    result = run_assets(stages, tmp_path, png, work, [*THREE, "--judge-floor-p", "0.6", "--judge-floor-u", "0.4"])
    assert result.returncode == 2 and "judge.p.global" in result.stderr
    assert tree_digest(work) == before
    assert run_assets(stages, tmp_path, png, work, [*THREE, "--judge-floor-p", "0.5", "--judge-floor-u", "0.4"]).returncode == 0
