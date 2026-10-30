import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_single_file as builder  # noqa: E402
from conftest import FAMILIES, make_product  # noqa: E402
from png_helpers import write_stack_order, write_wide_import_fixture  # noqa: E402
from psdl1.split import SPLIT_VERSION, validate_split  # noqa: E402
from test_pretrain import write_manifest  # noqa: E402


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    out = tmp_path_factory.mktemp("single") / "psdl1_job.py"
    info = builder.build(out)
    return out, info


def with_jobs(text, jobs, log, work):
    block = re.compile(r"JOBS = \[.*?\nLOG_FILE = .*?\nWORK_DIR = .*?\n", re.S)
    replacement = f"JOBS = {jobs!r}\nLOG_FILE = {str(log)!r}\nWORK_DIR = {str(work)!r}\n"
    assert block.search(text)
    return block.sub(lambda m: replacement, text, count=1)


def test_bundle_embeds_every_module_byte_for_byte_and_refuses_overwrite(bundle):
    out, info = bundle
    modules = sorted((ROOT / "src/psdl1").glob("*.py"))
    assert info["modules"] == len(modules) and out.stat().st_size == info["bytes"]
    namespace = __import__("runpy").run_path(str(out), run_name="psdl1_job")  # defines the functions, does not run the jobs
    files = namespace["_files"]()
    assert files.keys() == {p.name for p in modules} and all(files[p.name] == p.read_bytes() for p in modules)
    with pytest.raises(FileExistsError):
        builder.build(out)
    assert builder.build(out, overwrite=True)["source_sha256"] == info["source_sha256"]  # deterministic
    assert "EDIT BELOW" in out.read_text(encoding="utf-8") and "EDIT ABOVE" in out.read_text(encoding="utf-8")


def test_extraction_reuses_the_folder_and_repairs_a_modified_file(bundle, tmp_path):
    out, _ = bundle
    text = with_jobs(out.read_text(encoding="utf-8"), [], tmp_path / "log.txt", tmp_path / "work")
    script = tmp_path / "job.py"
    script.write_text(text, encoding="utf-8")
    ns = __import__("runpy").run_path(str(script), run_name="psdl1_job")
    root = ns["_extract"]()
    assert root.parent == tmp_path / "work" and (root / "psdl1" / "train.py").read_bytes() == (ROOT / "src/psdl1/train.py").read_bytes()
    (root / "psdl1" / "train.py").write_text("# tampered")
    assert ns["_extract"]() == root and (root / "psdl1" / "train.py").read_bytes() == (ROOT / "src/psdl1/train.py").read_bytes()


def test_console_encoding_that_cannot_represent_a_character_does_not_stop_the_job(bundle, tmp_path):
    out, _ = bundle
    code = (
        "import runpy, sys\n"
        "ns = runpy.run_path(sys.argv[1], run_name='psdl1_job')\n"
        "ns['_safe_streams']()\n"
        "print('17\\u201318 bytes', flush=True)\n"
        "print('17\\u201318 bytes', file=sys.stderr, flush=True)\n"
    )
    env = {**os.environ, "PYTHONIOENCODING": "euc_kr:strict"}
    result = subprocess.run([sys.executable, "-c", code, str(out)], capture_output=True, env=env, cwd=tmp_path)
    assert result.returncode == 0, result.stderr.decode("ascii", "replace")
    assert b"\\u2013" in result.stdout and b"\\u2013" in result.stderr


def test_unknown_job_name_is_rejected_before_anything_runs(bundle, tmp_path):
    out, _ = bundle
    script = tmp_path / "job.py"
    script.write_text(with_jobs(out.read_text(encoding="utf-8"), [("rm_rf", [])], tmp_path / "log.txt", tmp_path / "work"), encoding="utf-8")
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode != 0 and "unknown job" in result.stderr


def test_pretrain_then_train_run_end_to_end_from_the_single_file(bundle, tmp_path):
    from dataclasses import replace
    out, _ = bundle
    products = [replace(make_product(f, s), product_id=f"{f.upper()}_{v}", lineage_id=f.upper()) for f in FAMILIES for s, v in ((40, "R00"), (64, "T00"))]
    manifest = write_manifest(tmp_path, products, 64)
    split = {"schema_version": SPLIT_VERSION, "train": ["BUS_R00", "BUS_T00", "COMB_R00", "COMB_T00"], "validation": [], "test": ["RING_R00", "RING_T00"]}
    (tmp_path / "split.json").write_text(json.dumps(split))
    (tmp_path / "relations.csv").write_text("BUS_R00,BUS_T00,similar\nCOMB_R00,COMB_T00,similar\nBUS_R00,COMB_R00,different\nBUS_T00,COMB_T00,different\n")
    pre, run = tmp_path / "pretrain", tmp_path / "train"
    jobs = [("pretrain", ["--manifest", str(manifest), "--split", str(tmp_path / "split.json"), "--out", str(pre), "--epochs", "1", "--batch-size", "3", "--mask-ratio", "0.5"]),
            ("train", ["--manifest", str(manifest), "--split", str(tmp_path / "split.json"), "--relations", str(tmp_path / "relations.csv"), "--weights", str(pre / "backbone.pt"),
                       "--backbone", "tiny", "--out", str(run), "--epochs", "1", "--batch-size", "4"])]
    log = tmp_path / "logs" / "job.log"
    script = tmp_path / "psdl1_job.py"
    script.write_text(with_jobs(out.read_text(encoding="utf-8"), jobs, log, tmp_path / "work"), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=tmp_path, env=env)
    assert result.returncode == 0, result.stderr[-3000:]
    assert (pre / "backbone.pt").is_file() and (run / "dl1.pt").is_file()
    text = log.read_text(encoding="utf-8")
    assert str(tmp_path / "work") in text.splitlines()[0]  # psdl1 came from the extracted copy, not from an installed one
    assert "start pretrain" in text and "done pretrain" in text and "done train" in text and '"event": "progress"' in text
    saved = __import__("torch").load(run / "dl1.pt", map_location="cpu", weights_only=True)
    assert saved["metadata"]["initial_weights_hash"] is not None and saved["epoch"] == 1


def test_job_failure_is_logged_and_stops_the_chain(bundle, tmp_path):
    out, _ = bundle
    log = tmp_path / "job.log"
    jobs = [("train", ["--manifest", str(tmp_path / "missing.json"), "--split", "x", "--out", str(tmp_path / "o")]), ("pretrain", ["--help"])]
    script = tmp_path / "job.py"
    script.write_text(with_jobs(out.read_text(encoding="utf-8"), jobs, log, tmp_path / "work"), encoding="utf-8")
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode != 0 and "Traceback" in log.read_text(encoding="utf-8") and "start pretrain" not in log.read_text(encoding="utf-8")


# ---- 단계별 파일 (--out-dir): inspect / assets / train ----

STAGE_FILES = ("psdl1_inspect.py", "psdl1_assets.py", "psdl1_train.py")
METADATA = {"metadata_schema_version": "dl1_png_metadata_v1", "models": {
    "SYNTHETIC-WIDE": {"lineage_id": "SYNTHETIC-FAMILY", "design_version": "SYNTHETIC-V1"},
    "SYNTHETIC-WIDE-Z": {"lineage_id": "SYNTHETIC-OTHER-FAMILY", "design_version": "SYNTHETIC-OTHER-V1"}}}
STACK_ORDER = {"stack_order_schema_version": "dl1_png_stack_order_v1", "order": ["U", "N", "C4", "U5", "C6", "L"]}


def edit_block(text, block):
    pattern = re.compile(r"# =+ EDIT BELOW.*?# =+ EDIT ABOVE =+\n", re.S)
    assert pattern.search(text)
    return pattern.sub(lambda m: block, text, count=1)


def stage_script(stages, name, tmp_path, **values):
    """EDIT 블록을 값으로 바꾼 사본을 tmp_path 에 쓴다. 값은 repr 로 적는다."""
    lines = [f"{key} = {value!r}" for key, value in values.items()]
    lines += ["LOG_FILE = WORK + '/logs/stage.log'", "WORK_DIR = None"]
    script = tmp_path / name
    script.write_text(edit_block((stages / name).read_text(encoding="utf-8"), "\n".join(lines) + "\n"), encoding="utf-8")
    return script


def run_script(script, cwd):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=cwd, env=env)


def assets_values(png, work, **overrides):
    values = dict(PNG_ROOT=str(png), WORK=str(work), GRID=128, NO_PREVIEW=False, ROUND=None, METADATA=METADATA, STACK_ORDER=STACK_ORDER, SPLIT_SEED=0,
                  VALIDATION_FRACTION=0.2, TEST_FRACTION=0.2, SCREEN_ARGS=["--floor", "2"])
    return {**values, **overrides}


def train_values(work, **overrides):
    values = dict(WORK=str(work), PRETRAIN=False, PRETRAIN_ARGS=[], TRAIN_ARGS=["--backbone", "tiny", "--epochs", "1", "--batch-size", "2", "--allow-grid-check-failure"],
                  EVALUATE=True, EVALUATE_ARGS=[], EVALUATE_ONLY=False, ROUND=None, ROUND_EPOCHS=30, WARM_START_LR=1e-4, NEW_EDGE_REPEAT=1)
    return {**values, **overrides}


def skips(log_text):
    return [line for line in log_text.splitlines() if "] " in line and " skip " in line]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def stages(tmp_path_factory):
    out = tmp_path_factory.mktemp("stages")
    builder.build_stages(out)
    return out


@pytest.fixture(scope="module")
def png_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("png-source")
    write_wide_import_fixture(root)
    write_wide_import_fixture(root, model="SYNTHETIC-WIDE-Z")
    return root


@pytest.fixture(scope="module")
def assets_done(stages, png_source, tmp_path_factory):
    """assets 단계를 한 번 실행한 WORK. 읽기만 하는 테스트가 공유한다."""
    base = tmp_path_factory.mktemp("assets-run")
    script = stage_script(stages, "psdl1_assets.py", base, **assets_values(png_source, base / "work"))
    result = run_script(script, base)
    assert result.returncode == 0, result.stderr[-3000:]
    return base, script, base / "work"


def test_psdl1_comes_from_this_worktree():
    import psdl1
    assert Path(psdl1.__file__).resolve().is_relative_to(ROOT / "src")


def test_nonzero_return_stops_the_chain_and_the_process(bundle, tmp_path):
    out, _ = bundle
    log = tmp_path / "job.log"
    jobs = [("import_png", ["--root", str(tmp_path / "missing"), "--out", str(tmp_path / "o"), "--export"]), ("pretrain", ["--help"])]  # import_png returns 2, it does not raise
    script = tmp_path / "job.py"
    script.write_text(with_jobs(out.read_text(encoding="utf-8"), jobs, log, tmp_path / "work"), encoding="utf-8")
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=tmp_path)
    text = log.read_text(encoding="utf-8")
    assert result.returncode == 2 and "import_png exited with 2" in text and "start pretrain" not in text


def test_out_dir_builds_three_deterministic_files_and_refuses_overwrite(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    info = builder.build_stages(first)
    builder.build_stages(second)
    assert sorted(p.name for p in first.iterdir()) == sorted(STAGE_FILES) and set(info["files"]) == set(STAGE_FILES)
    assert all((first / n).read_bytes() == (second / n).read_bytes() for n in STAGE_FILES)
    for name in STAGE_FILES:
        text = (first / name).read_text(encoding="utf-8")
        assert "EDIT BELOW" in text and "EDIT ABOVE" in text and (__import__("runpy").run_path(str(first / name), run_name="psdl1_stage")["_files"]().keys() ==
                                                                  {p.name for p in (ROOT / "src/psdl1").glob("*.py")})
    with pytest.raises(FileExistsError):
        builder.build_stages(first)
    (first / "psdl1_train.py").unlink()
    with pytest.raises(FileExistsError):  # one remaining file is enough to refuse, and nothing is written
        builder.build_stages(first)
    assert not (first / "psdl1_train.py").exists()
    assert builder.build_stages(first, overwrite=True)["source_sha256"] == info["source_sha256"]
    assert main_cli(["--out-dir", str(first), "--out", str(first / "x.py")]) != 0  # mutually exclusive


def main_cli(argv):
    try:
        builder.main(argv)
    except SystemExit as exc:
        return exc.code
    return 0


def test_inspect_runs_then_skips(stages, png_source, tmp_path):
    script = stage_script(stages, "psdl1_inspect.py", tmp_path, PNG_ROOT=str(png_source), WORK=str(tmp_path / "work"), GRID_CANDIDATES=[64, 128], NO_PREVIEW=True)
    first = run_script(script, tmp_path)
    assert first.returncode == 0, first.stderr[-3000:]
    out = tmp_path / "work" / "inspect-g64-128"
    assert out.is_dir() and any(out.iterdir())
    second = run_script(script, tmp_path)
    assert second.returncode == 0, second.stderr[-3000:]
    assert f"skip inspect (exists: {out})" in (tmp_path / "work" / "logs" / "stage.log").read_text(encoding="utf-8")


def test_assets_produces_every_file_without_inspect(assets_done):
    _, _, work = assets_done
    import psdl1.data as data
    products = data.load_manifest(work / "export" / "manifest.json")
    split = json.loads((work / "inputs" / "split.json").read_text(encoding="utf-8"))
    validate_split(products, split)
    assert json.loads((work / "inputs" / "metadata.json").read_text(encoding="utf-8")) == METADATA
    assert json.loads((work / "inputs" / "stack-order.json").read_text(encoding="utf-8")) == STACK_ORDER
    rows = (work / "relations" / "relations.csv").read_text(encoding="utf-8").split()
    assert (work / "reports" / "grid-report.json").is_file() and [r.rsplit(",", 1)[1] for r in rows] == ["different"]
    assert all((work / "relations" / "train" / f"relations.csv{suffix}").is_file() for suffix in (".review.html", ".scores.csv", ".summary.json"))
    assert not (work / "relations" / "test").exists()  # no test products in this split
    text = (work / "logs" / "stage.log").read_text(encoding="utf-8")
    assert "split train: 2 products, 2 lineages" in text and "check these files before training" in text and "relations.csv.review.html" in text


def test_assets_second_run_skips_everything_except_verify(assets_done, tmp_path):
    base, script, work = assets_done
    log = work / "logs" / "stage.log"
    before = log.read_text(encoding="utf-8")
    snapshot = {p: sha256(p) for p in work.rglob("*") if p.is_file() and p.parent.name != "logs"}
    result = run_script(script, base)
    assert result.returncode == 0, result.stderr[-3000:]
    second = log.read_text(encoding="utf-8")[len(before):]
    starts = [line.split("start ", 1)[1].split()[0:2] for line in second.splitlines() if "] " in line and " start " in line]
    assert starts == [["import_png", "--verify-bundle"]]  # the only job that runs again
    kinds = " ".join(skips(second))
    for step in ("METADATA", "STACK_ORDER", "export", "grid report", "split", "relations"):
        assert f"skip {step} (exists:" in kinds
    assert snapshot == {p: sha256(p) for p in work.rglob("*") if p.is_file() and p.parent.name != "logs"}


def test_changed_metadata_stops_without_overwrite(stages, assets_done, tmp_path):
    _, _, work = assets_done
    saved = sha256(work / "inputs" / "metadata.json")
    changed = {**METADATA, "models": {**METADATA["models"], "SYNTHETIC-WIDE": {"lineage_id": "OTHER", "design_version": "SYNTHETIC-V1"}}}
    script = stage_script(stages, "psdl1_assets.py", tmp_path, **assets_values(tmp_path / "no-png", work, METADATA=changed))
    result = run_script(script, tmp_path)
    assert result.returncode != 0 and "STOP:" in result.stderr and "differs from METADATA" in result.stderr
    assert sha256(work / "inputs" / "metadata.json") == saved


def test_train_requires_assets_first(stages, tmp_path):
    script = stage_script(stages, "psdl1_train.py", tmp_path, **train_values(tmp_path / "empty-work"))
    result = run_script(script, tmp_path)
    assert result.returncode != 0 and "psdl1_assets.py first" in result.stderr and not (tmp_path / "empty-work" / "train").exists()


def test_train_runs_after_assets_then_refuses_to_overwrite_the_checkpoint(stages, assets_done, tmp_path):
    _, _, work = assets_done
    script = stage_script(stages, "psdl1_train.py", tmp_path, **train_values(work))
    result = run_script(script, tmp_path)
    assert result.returncode == 0, result.stderr[-3000:]
    checkpoint = work / "train" / "dl1.pt"
    assert checkpoint.is_file() and "skip evaluate (no test products" in result.stderr
    digest = sha256(checkpoint)
    again = run_script(script, tmp_path)
    assert again.returncode != 0 and "already exists" in again.stderr and "EVALUATE_ONLY" in again.stderr and "--resume" not in again.stderr
    assert sha256(checkpoint) == digest


def test_train_stops_on_an_unfinished_pretrain_folder(stages, tmp_path):
    work = tmp_path / "work"
    for rel in ("export/manifest.json", "inputs/split.json", "relations/relations.csv", "pretrain/backbone.pt"):
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_text("x")  # backbone.pt alone: pretrain.json (written last) is missing
    script = stage_script(stages, "psdl1_train.py", tmp_path, **train_values(work, PRETRAIN=True))
    result = run_script(script, tmp_path)
    assert result.returncode != 0 and "not a completed pretraining" in result.stderr and not (work / "train").exists()


def test_train_file_pretrains_trains_evaluates_and_skips_finished_steps(stages, tmp_path):
    from dataclasses import replace
    work = tmp_path / "work"
    (work / "export").mkdir(parents=True)
    products = [replace(make_product(f, s), product_id=f"{f.upper()}_{v}", lineage_id=f.upper()) for f in FAMILIES for s, v in ((40, "R00"), (64, "T00"))]
    write_manifest(work / "export", products, 64)
    (work / "inputs").mkdir()
    split = {"schema_version": SPLIT_VERSION, "train": ["BUS_R00", "BUS_T00", "COMB_R00", "COMB_T00"], "validation": [], "test": ["RING_R00", "RING_T00"]}
    (work / "inputs" / "split.json").write_text(json.dumps(split))
    (work / "relations").mkdir()
    (work / "relations" / "relations.csv").write_text("BUS_R00,BUS_T00,similar\nCOMB_R00,COMB_T00,similar\nBUS_R00,COMB_R00,different\nBUS_T00,COMB_T00,different\n")
    script = stage_script(stages, "psdl1_train.py", tmp_path, **train_values(work, PRETRAIN=True, PRETRAIN_ARGS=["--epochs", "1", "--batch-size", "3", "--mask-ratio", "0.5"],
                                                                             TRAIN_ARGS=["--backbone", "tiny", "--epochs", "1", "--batch-size", "4"]))
    result = run_script(script, tmp_path)
    assert result.returncode == 0, result.stderr[-3000:]
    assert (work / "pretrain" / "pretrain.json").is_file() and (work / "train" / "dl1.pt").is_file() and (work / "train" / "evaluation.json").is_file()
    saved = __import__("torch").load(work / "train" / "dl1.pt", map_location="cpu", weights_only=True)
    assert saved["metadata"]["initial_weights_hash"] is not None  # the pretrained backbone was used
    again = run_script(script, tmp_path)  # dl1.pt exists: stops before pretrain/train
    assert again.returncode != 0 and "already exists" in again.stderr and "start pretrain" not in again.stderr
    only = stage_script(stages, "psdl1_train.py", tmp_path / "only", **train_values(work, PRETRAIN=True, EVALUATE_ONLY=True)) if (tmp_path / "only").mkdir() is None else None
    done = run_script(only, tmp_path)
    assert done.returncode == 0, done.stderr[-3000:]
    assert "skip evaluate (exists:" in " ".join(skips(done.stderr)) and "start train" not in done.stderr and "start pretrain" not in done.stderr
    evaluation = work / "train" / "evaluation.json"
    document = json.loads(evaluation.read_text(encoding="utf-8"))
    document["checkpoint_weights_hash"] = "0" * 64
    evaluation.write_text(json.dumps(document))
    stale = sha256(evaluation)
    refused = run_script(only, tmp_path)
    assert refused.returncode != 0 and "does not match" in refused.stderr and sha256(evaluation) == stale
    evaluation.unlink()
    fresh = run_script(only, tmp_path)  # EVALUATE_ONLY runs only evaluate
    assert fresh.returncode == 0 and "start evaluate" in fresh.stderr and "start train" not in fresh.stderr and evaluation.is_file()
    assert "was trained for 1 epochs (TRAIN_ARGS --epochs: 1)" in fresh.stderr and "WARNING" not in fresh.stderr
    short = stage_script(stages, "psdl1_train.py", tmp_path / "short", **train_values(work, EVALUATE_ONLY=True, TRAIN_ARGS=["--epochs", "5"])) if (tmp_path / "short").mkdir() is None else None
    warned = run_script(short, tmp_path)  # unfinished-checkpoint warning only; it still evaluates (here: skips the finished evaluation)
    assert warned.returncode == 0 and "WARNING: the checkpoint has 1 of 5 epochs" in warned.stderr


def test_evaluate_only_needs_a_checkpoint(stages, tmp_path):
    work = tmp_path / "work"
    for rel in ("export/manifest.json", "inputs/split.json", "relations/relations.csv"):
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_text("x")
    script = stage_script(stages, "psdl1_train.py", tmp_path, **train_values(work, EVALUATE_ONLY=True))
    result = run_script(script, tmp_path)
    assert result.returncode != 0 and "EVALUATE_ONLY needs" in result.stderr


@pytest.mark.parametrize("name,key,args", [("psdl1_train.py", "TRAIN_ARGS", ["--out", "x"]), ("psdl1_train.py", "PRETRAIN_ARGS", ["--manifest=x"]),
                                          ("psdl1_train.py", "EVALUATE_ARGS", ["--ckpt", "x"]), ("psdl1_assets.py", "SCREEN_ARGS", ["--partitions=train"])])
def test_reserved_arguments_stop_before_anything_starts(stages, assets_done, tmp_path, name, key, args):
    _, _, work = assets_done
    values = assets_values(tmp_path / "no-png", tmp_path / "w") if name == "psdl1_assets.py" else train_values(work)
    values[key] = args
    result = run_script(stage_script(stages, name, tmp_path, **values), tmp_path)
    assert result.returncode != 0 and "must not contain" in result.stderr and " start " not in result.stderr


def test_checkpoint_present_stops_before_pretrain(stages, tmp_path):
    work = tmp_path / "work"
    for rel in ("export/manifest.json", "inputs/split.json", "relations/relations.csv", "train/dl1.pt"):
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_text("x")
    result = run_script(stage_script(stages, "psdl1_train.py", tmp_path, **train_values(work, PRETRAIN=True)), tmp_path)
    assert result.returncode != 0 and "already exists" in result.stderr and not (work / "pretrain").exists() and "start pretrain" not in result.stderr


def test_failed_inspect_is_kept_aside_and_the_next_run_starts_fresh(stages, png_source, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    work = tmp_path / "work"
    failing = stage_script(stages, "psdl1_inspect.py", tmp_path / "a", PNG_ROOT=str(empty), WORK=str(work), GRID_CANDIDATES=[64], NO_PREVIEW=True) if (tmp_path / "a").mkdir() is None else None
    assert run_script(failing, tmp_path).returncode != 0
    kept = [p.name for p in work.iterdir() if p.name.startswith("inspect-g64.failed-")]
    assert len(kept) == 1 and not (work / "inspect-g64").exists()
    good = stage_script(stages, "psdl1_inspect.py", tmp_path / "b", PNG_ROOT=str(png_source), WORK=str(work), GRID_CANDIDATES=[64], NO_PREVIEW=True) if (tmp_path / "b").mkdir() is None else None
    result = run_script(good, tmp_path)
    assert result.returncode == 0, result.stderr[-2000:]
    assert (work / "inspect-g64").is_dir() and "skip inspect" not in result.stderr


def test_failed_export_is_kept_aside_and_stale_inputs_are_replaced_before_export(stages, png_source, tmp_path):
    work = tmp_path / "work"
    bad = {"metadata_schema_version": "dl1_png_metadata_v1", "models": {"NO-SUCH-MODEL": {"lineage_id": "L", "design_version": "V"}}}
    failing = stage_script(stages, "psdl1_assets.py", tmp_path / "a", **assets_values(png_source, work, METADATA=bad)) if (tmp_path / "a").mkdir() is None else None
    assert run_script(failing, tmp_path).returncode != 0
    assert [p.name for p in work.iterdir() if p.name.startswith("export.failed-")] and not (work / "export").exists()
    good = stage_script(stages, "psdl1_assets.py", tmp_path / "b", **assets_values(png_source, work, NO_PREVIEW=True)) if (tmp_path / "b").mkdir() is None else None
    result = run_script(good, tmp_path)
    assert result.returncode == 0, result.stderr[-3000:]
    assert "replaced (no export yet) METADATA" in result.stderr
    assert json.loads((work / "inputs" / "metadata.json").read_text(encoding="utf-8")) == METADATA


def test_relations_are_screened_per_partition_with_the_train_floor(stages, tmp_path):
    png = tmp_path / "png"
    names = ["W1", "W2", "W3", "W4"]
    for name in names:
        write_wide_import_fixture(png, model=name)
    metadata = {"metadata_schema_version": "dl1_png_metadata_v1", "models": {
        n: {"lineage_id": "F1" if n in ("W1", "W2") else "F2", "design_version": "V1"} for n in names}}
    work = tmp_path / "work"
    (work / "inputs").mkdir(parents=True)
    split = {"schema_version": SPLIT_VERSION, "train": ["W1", "W2"], "validation": [], "test": ["W3", "W4"]}
    (work / "inputs" / "split.json").write_text(json.dumps(split))
    script = stage_script(stages, "psdl1_assets.py", tmp_path, **assets_values(png, work, METADATA=metadata, NO_PREVIEW=True, SCREEN_ARGS=[]))
    result = run_script(script, tmp_path)
    assert result.returncode == 0, result.stderr[-3000:]
    train = json.loads((work / "relations" / "train" / "relations.csv.summary.json").read_text(encoding="utf-8"))
    test = json.loads((work / "relations" / "test" / "relations.csv.summary.json").read_text(encoding="utf-8"))
    assert test["floor"]["global"] == train["floor"]["global"]  # the held-out partition is judged by the train floor
    rows = [line.split(",") for line in (work / "relations" / "relations.csv").read_text(encoding="utf-8").split()]
    assert sorted(r[2] for r in rows) == ["similar", "similar"] and {frozenset(r[:2]) for r in rows} == {frozenset(("W1", "W2")), frozenset(("W3", "W4"))}
    assert (work / "relations" / "test" / "relations.csv.review.html").is_file() and "relations/test/relations.csv.review.html" in result.stderr


# ---- 리뷰 권고 반영: 약어·--config·안전 이동·실패 경로·CSV 합치기 ----

def stage_globals(stages, name):
    """묶음 파일을 import 해(실행하지 않고) 함수들이 쓰는 globals 를 돌려준다. 여기서 바꾼 값이 함수에 반영된다."""
    return __import__("runpy").run_path(str(stages / name), run_name="psdl1_stage")["_say"].__globals__


@pytest.mark.parametrize("name,key,args", [
    ("psdl1_assets.py", "SCREEN_ARGS", ["--partition", "test"]), ("psdl1_assets.py", "SCREEN_ARGS", ["--manif=x"]), ("psdl1_assets.py", "SCREEN_ARGS", ["--sp", "x"]),
    ("psdl1_train.py", "TRAIN_ARGS", ["--resum", "x"]), ("psdl1_train.py", "PRETRAIN_ARGS", ["--wei", "x"]), ("psdl1_train.py", "EVALUATE_ARGS", ["--ckp", "x"]),
    ("psdl1_train.py", "TRAIN_ARGS", ["--config", "x.json"]), ("psdl1_train.py", "TRAIN_ARGS", ["--conf=x.json"]), ("psdl1_train.py", "TRAIN_ARGS", ["--c", "x"]),
    ("psdl1_assets.py", "SCREEN_ARGS", ["--flo", "2"]), ("psdl1_assets.py", "SCREEN_ARGS", ["--fl=2"])])
def test_abbreviations_and_config_stop_before_anything_starts(stages, assets_done, tmp_path, name, key, args):
    _, _, work = assets_done
    values = assets_values(tmp_path / "no-png", tmp_path / "w") if name == "psdl1_assets.py" else train_values(work)
    values[key] = args
    result = run_script(stage_script(stages, name, tmp_path, **values), tmp_path)
    assert result.returncode != 0 and "STOP:" in result.stderr and " start " not in result.stderr and "psdl1 from" not in result.stderr
    assert not [p.name for p in (tmp_path / "w").glob("*") if p.name != "logs"]  # nothing but the log was written
    if "config" in " ".join(args) or "conf" in " ".join(args):
        assert "JOBS-style" in result.stderr


def test_floor_abbreviation_is_refused_and_the_full_name_is_read(stages):
    g = stage_globals(stages, "psdl1_assets.py")
    assert g["_floor_arg"](["--floor", "2"]) == "2" and g["_floor_arg"](["--floor=3"]) == "3" and g["_floor_arg"](["--groups", "g.json"]) is None
    for bad in (["--fl", "2"], ["--flo=2"], ["--floo", "2"]):
        with pytest.raises(SystemExit) as stopped:
            g["_floor_arg"](bad)
        assert stopped.value.code == 2


def test_user_floor_is_used_for_both_partitions(stages, tmp_path):
    png = tmp_path / "png"
    names = ["W1", "W2", "W3", "W4"]
    for name in names:
        write_wide_import_fixture(png, model=name)
    metadata = {"metadata_schema_version": "dl1_png_metadata_v1", "models": {
        n: {"lineage_id": "F1" if n in ("W1", "W2") else "F2", "design_version": "V1"} for n in names}}
    work = tmp_path / "work"
    (work / "inputs").mkdir(parents=True)
    (work / "inputs" / "split.json").write_text(json.dumps({"schema_version": SPLIT_VERSION, "train": ["W1", "W2"], "validation": [], "test": ["W3", "W4"]}))
    script = stage_script(stages, "psdl1_assets.py", tmp_path, **assets_values(png, work, METADATA=metadata, NO_PREVIEW=True, SCREEN_ARGS=["--floor", "0.7"]))
    result = run_script(script, tmp_path)
    assert result.returncode == 0, result.stderr[-3000:]
    for part in ("train", "test"):
        summary = json.loads((work / "relations" / part / "relations.csv.summary.json").read_text(encoding="utf-8"))
        assert summary["floor"]["global"] == 0.7


def test_stack_order_none_moves_the_old_file_aside(stages, tmp_path):
    work = tmp_path / "work"
    (work / "inputs").mkdir(parents=True)
    old = work / "inputs" / "stack-order.json"
    old.write_text(json.dumps(STACK_ORDER))
    script = stage_script(stages, "psdl1_assets.py", tmp_path, **assets_values(tmp_path / "no-png", work, STACK_ORDER=None))
    result = run_script(script, tmp_path)  # the export fails later (no PNGs); the old file was already moved, not deleted
    moved = [p for p in (work / "inputs").iterdir() if p.name.startswith("stack-order.json.removed-")]
    assert not old.exists() and len(moved) == 1 and json.loads(moved[0].read_text(encoding="utf-8")) == STACK_ORDER
    assert f"moved {old} to {moved[0]}" in result.stderr


def test_failure_names_do_not_collide_and_the_cause_is_logged_first(stages, tmp_path, capsys):
    g = stage_globals(stages, "psdl1_assets.py")
    g["time"] = type("T", (), {"strftime": staticmethod(lambda fmt: "T"), "time": staticmethod(lambda: 0.0)})
    out = tmp_path / "export"
    out.mkdir()
    (out / "new.txt").write_text("new")
    older = tmp_path / "export.failed-T"
    older.mkdir()
    (older / "old.txt").write_text("old")
    assert g["_aside"](out, "failed") == tmp_path / "export.failed-T-2"
    g["_call"] = lambda name, args: (_ for _ in ()).throw(SystemExit(3))
    with pytest.raises(SystemExit) as stopped:
        g["_call_keeping_failed"]("import_png", [], out)
    assert stopped.value.code == 3 and not out.exists()
    assert (older / "old.txt").read_text() == "old" and (tmp_path / "export.failed-T-2" / "new.txt").read_text() == "new"
    err = capsys.readouterr().err
    assert err.index("import_png failed: exit code 3") < err.index("kept its output at")


def test_failure_when_the_move_fails_keeps_the_original_exit_code(stages, tmp_path, capsys):
    g = stage_globals(stages, "psdl1_assets.py")

    def no_replace(src, dst):
        raise OSError("read-only")
    g["os"] = type("O", (), {"replace": staticmethod(no_replace), "path": os.path})
    g["_call"] = lambda name, args: (_ for _ in ()).throw(RuntimeError("boom"))
    out = tmp_path / "export"
    out.mkdir()
    with pytest.raises(RuntimeError, match="boom"):
        g["_call_keeping_failed"]("import_png", [], out)
    err = capsys.readouterr().err
    assert out.is_dir() and "import_png failed: RuntimeError: boom" in err and f"이동 실패: {out}" in err and "손으로 옮긴 뒤 다시 실행" in err


def test_partial_or_broken_screen_output_stops_with_guidance(stages, tmp_path, capsys):
    g = stage_globals(stages, "psdl1_assets.py")
    part = tmp_path / "train"
    part.mkdir()
    (part / "relations.csv").write_text("A,B,similar\n")  # only one of the three outputs
    with pytest.raises(SystemExit) as stopped:
        g["_screen"]("m.json", "s.json", "train", part, [])
    assert stopped.value.code == 2 and "partial output" in capsys.readouterr().err
    (part / "relations.csv.review.html").write_text("<html>")
    (part / "relations.csv.summary.json").write_text("{not json")
    with pytest.raises(SystemExit) as stopped:
        g["_screen"]("m.json", "s.json", "train", part, [])
    err = capsys.readouterr().err
    assert stopped.value.code == 2 and f"부분 출력: {part} 를 옮긴 뒤 다시 실행" in err and "Traceback" not in err
    (part / "relations.csv.summary.json").write_text("[1]")
    with pytest.raises(SystemExit):
        g["_screen"]("m.json", "s.json", "train", part, [])
    (part / "relations.csv.summary.json").write_text(json.dumps({"floor": {"global": 1.5}}))
    assert g["_screen"]("m.json", "s.json", "train", part, [])["floor"]["global"] == 1.5


def test_bindings_csv_alone_is_partial_output_and_three_required_files_still_skip(stages, tmp_path, capsys):
    g = stage_globals(stages, "psdl1_assets.py")
    part = tmp_path / "train"
    part.mkdir()
    (part / "relations.csv.bindings.csv").write_text("product_a\n")
    with pytest.raises(SystemExit) as stopped:
        g["_screen"]("m.json", "s.json", "train", part, [])
    err = capsys.readouterr().err
    assert stopped.value.code == 2 and "partial output" in err and "relations.csv.bindings.csv" in err
    (part / "relations.csv").write_text("A,B,similar\n")
    (part / "relations.csv.review.html").write_text("<html>")
    (part / "relations.csv.summary.json").write_text(json.dumps({"floor": {"global": 1.5}}))
    assert g["_screen"]("m.json", "s.json", "train", part, [])["floor"]["global"] == 1.5  # 필수 3개가 있으면 bindings 가 있어도 건너뛴다


def test_merge_stops_on_a_pair_in_two_partitions_and_keeps_csv_quoting(stages, tmp_path):
    import csv
    g = stage_globals(stages, "psdl1_assets.py")
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text('"P,1","P 2",similar\nQ1,Q2,different\n')
    b.write_text('X1,"X,2",similar\n')
    rows = g["_merge_rows"]([a, b])
    merged = tmp_path / "merged.csv"
    g["_write_rows"](merged, rows)
    with open(merged, newline="", encoding="utf-8") as handle:
        assert list(csv.reader(handle)) == [["P,1", "P 2", "similar"], ["Q1", "Q2", "different"], ["X1", "X,2", "similar"]]
    assert '"P,1"' in merged.read_text(encoding="utf-8") and not (tmp_path / "merged.csv.tmp").exists()
    b.write_text("Q2,Q1,similar\n")  # the same pair, reversed, in the other partition
    with pytest.raises(SystemExit) as stopped:
        g["_merge_rows"]([a, b])
    assert stopped.value.code == 2


def test_target_epochs_reads_train_args(stages):
    g = stage_globals(stages, "psdl1_train.py")
    assert g["_target_epochs"](["--backbone", "tiny"]) == 100 and g["_target_epochs"](["--epochs", "7"]) == 7
    assert g["_target_epochs"](["--epochs=9"]) == 9 and g["_target_epochs"](["--epoch", "4"]) == 4 and g["_target_epochs"](["--epochs", "x"]) is None


# ---- 증분 라운드 (ROUND) ----
# X6 fixture: r01 = 계보 FA(A1,A2) FB(B1,B2) FC(C1,C2). split 은 손으로 정한다: train = FA+FB(서로 다른 계보라 negative 가 생긴다, --floor 2 로 동일 이미지도 different),
# test = FC(C1-C2 similar 쌍이 평가 가능한 positive). r02 = D1(FA) E1,E2(새 계보 FE). 계보 4개이므로 extend_split 의 목표 수 int(4*0.2)-기존 = 0 이라 새 계보는 train 으로 간다.
# 그래서 r02 train 에는 새 제품이 낀 negative 쌍이 있고 test 의 C1-C2 쌍이 평가된다.

ROUND_MODELS_1 = {"A1": "FA", "A2": "FA", "B1": "FB", "B2": "FB", "C1": "FC", "C2": "FC"}
ROUND_MODELS_2 = {"D1": "FA", "E1": "FE", "E2": "FE"}
ROUND_SPLIT_1 = {"schema_version": SPLIT_VERSION, "train": ["A1", "A2", "B1", "B2"], "validation": [], "test": ["C1", "C2"]}
ROUND_EVALUATE_ARGS = ["--allow-grid-check-failure"]


def lineage_metadata(models):
    return {"metadata_schema_version": "dl1_png_metadata_v1", "models": {name: {"lineage_id": lineage, "design_version": "V1"} for name, lineage in models.items()}}


def script_in(stages, name, folder, **values):
    folder.mkdir(parents=True, exist_ok=True)
    return stage_script(stages, name, folder, **values)


def clone_tree(source, destination):
    """폴더를 복사한다. 큰 마스크(.npy)는 hard link 로 걸어 시간을 줄이고, 나머지는 복사해 사본을 고쳐도 원본이 변하지 않게 한다."""
    shutil.copytree(source, destination, copy_function=lambda s, d: os.link(s, d) if s.endswith(".npy") else shutil.copy2(s, d))


def run_inprocess(stages, name, folder, monkeypatch, patch=None, **values):
    """실행기의 stage() 를 이 프로세스에서 돌린다(_boot 는 건너뛴다). 종료 코드를 돌려준다."""
    g = __import__("runpy").run_path(str(script_in(stages, name, folder, **values)), run_name="psdl1_stage")["_say"].__globals__  # the live module globals
    g["_boot"] = lambda: None
    if patch:
        patch(g)
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    try:
        g["stage"]()
    except SystemExit as exc:
        return exc.code
    return 0


def fail_first_merge(g):
    """첫 merge_bundles 호출이 출력 폴더를 만든 뒤 실패하게 한다(나머지 호출은 그대로)."""
    real = g["_call"]

    def flaky(name, args):
        argv = [str(a) for a in args]
        if name == "merge_bundles" and "--out" in argv and not flaky.done:
            flaky.done = True
            out = Path(argv[argv.index("--out") + 1])
            out.mkdir()
            (out / "merge-report.json").write_text("{}")
            raise SystemExit(2)
        return real(name, args)
    flaky.done = False
    g["_call"] = flaky


@pytest.fixture(scope="module")
def pngs(tmp_path_factory):
    roots = {}
    for key, models in (("one", ROUND_MODELS_1), ("two", ROUND_MODELS_2), ("alt", {"G1": "FG", "G2": "FG"}), ("three", {"F1": "FF"})):
        roots[key] = tmp_path_factory.mktemp(f"png-{key}")
        for model in models:
            write_wide_import_fixture(roots[key], model=model)
    return roots


def build_round_one(stages, pngs, base, rounds):
    """r01 을 만든다(assets + train + evaluate). rounds=False 면 평면 WORK(ROUND=None), True 면 WORK/rounds/r01(ROUND=1)."""
    work = base / "work"
    round_value = 1 if rounds else None
    folder = (work / "rounds" / "r01" if rounds else work) / "inputs"
    folder.mkdir(parents=True)
    (folder / "split.json").write_text(json.dumps(ROUND_SPLIT_1))
    assets = script_in(stages, "psdl1_assets.py", base / "s-assets-1", **assets_values(pngs["one"], work, ROUND=round_value, METADATA=lineage_metadata(ROUND_MODELS_1), NO_PREVIEW=True))
    result = run_script(assets, base)
    assert result.returncode == 0, result.stderr[-3000:]
    train = script_in(stages, "psdl1_train.py", base / "s-train-1", **train_values(work, ROUND=round_value, EVALUATE_ARGS=ROUND_EVALUATE_ARGS))
    result = run_script(train, base)
    assert result.returncode == 0, result.stderr[-3000:]
    return work


def round_two_values(png, work, models, **overrides):
    return assets_values(png, work, ROUND=2, METADATA=lineage_metadata(models), SCREEN_ARGS=[], NO_PREVIEW=True, **overrides)


def round_two_train_values(work, **overrides):
    return train_values(work, **{**dict(ROUND=2, ROUND_EPOCHS=2, WARM_START_LR=5e-4, NEW_EDGE_REPEAT=2, EVALUATE_ARGS=ROUND_EVALUATE_ARGS), **overrides})


@pytest.fixture(scope="module")
def rounds_env(stages, pngs, tmp_path_factory):
    """r01(ROUND=1) 와 r02(ROUND=2) 를 assets·train 까지 한 번 돌린 WORK. 읽기만 하는 테스트가 공유한다."""
    base = tmp_path_factory.mktemp("rounds-run")
    work = build_round_one(stages, pngs, base, rounds=True)
    assets = script_in(stages, "psdl1_assets.py", base / "s-assets-2", **round_two_values(pngs["two"], work, ROUND_MODELS_2))
    result = run_script(assets, base)
    assert result.returncode == 0, result.stderr[-3000:]
    train = script_in(stages, "psdl1_train.py", base / "s-train-2", **round_two_train_values(work))
    result = run_script(train, base)
    assert result.returncode == 0, result.stderr[-3000:]
    return SimpleNamespace(base=base, work=work, assets=assets, train=train, r1=work / "rounds" / "r01", r2=work / "rounds" / "r02", log=work / "logs" / "stage.log")


@pytest.fixture(scope="module")
def flat_env(stages, pngs, tmp_path_factory):
    base = tmp_path_factory.mktemp("flat-run")
    return SimpleNamespace(base=base, work=build_round_one(stages, pngs, base, rounds=False))


@pytest.fixture(scope="module")
def alt_env(stages, pngs, rounds_env, tmp_path_factory):
    """r01 의 합친 relations.csv 를 손으로 고친(마지막 개행 제거, 한 행의 쌍 순서 바꿈) 사본에서 다른 모델(G1, G2)로 r02 assets 를 돌린다."""
    base = tmp_path_factory.mktemp("alt-run")
    work = base / "work"
    clone_tree(rounds_env.r1, work / "rounds" / "r01")
    path = work / "rounds" / "r01" / "relations" / "relations.csv"
    original = path.read_bytes()
    lines = original.decode().splitlines()
    index = next(i for i, line in enumerate(lines) if line.endswith(",different"))
    a, b, kind = lines[index].split(",")
    lines[index] = f"{b},{a},{kind}"
    edited = "\n".join(lines).encode()  # no trailing newline
    assert edited != original.rstrip(b"\n")
    path.write_bytes(edited)
    script = script_in(stages, "psdl1_assets.py", base / "s-assets-2", **round_two_values(pngs["alt"], work, {"G1": "FG", "G2": "FG"}))
    result = run_script(script, base)
    assert result.returncode == 0, result.stderr[-3000:]
    return SimpleNamespace(base=base, work=work, r2=work / "rounds" / "r02", edited=edited)


def test_round_one_in_rounds_layout_matches_the_flat_layout(rounds_env, flat_env):
    def files(root):
        return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    flat = [f for f in files(flat_env.work) if not f.startswith("logs/")]
    assert not (flat_env.work / "rounds").exists()  # ROUND = None keeps the flat WORK
    assert flat == files(rounds_env.r1)
    assert {p.name for p in rounds_env.r1.iterdir()} == {"inputs", "export", "reports", "relations", "train"}
    for rel in ("relations/relations.csv", "inputs/split.json", "inputs/metadata.json"):
        assert (flat_env.work / rel).read_bytes() == (rounds_env.r1 / rel).read_bytes()


def test_round_two_assets_merges_extends_split_and_screens_only_delta(rounds_env):
    import psdl1.data as data
    from psdl1.merge_bundles import verify_merged_bundle
    ids = sorted(p.product_id for p in data.load_manifest(rounds_env.r2 / "export" / "manifest.json"))
    assert ids == sorted({**ROUND_MODELS_1, **ROUND_MODELS_2})
    assert verify_merged_bundle(rounds_env.r2 / "export")["product_count"] == 9
    old, new = (json.loads((r / "inputs" / "split.json").read_text(encoding="utf-8")) for r in (rounds_env.r1, rounds_env.r2))
    for part in ("train", "validation", "test"):
        assert set(old[part]) <= set(new[part])
    assert {"D1", "E1", "E2"} <= set(new["train"])  # D1 joins its lineage FA; the new lineage FE goes to train (int(4 * 0.2) - 0 = 0 validation/test lineages)
    read = lambda r, part: json.loads((r / "relations" / part / "relations.csv.summary.json").read_text(encoding="utf-8"))
    train, test, first = read(rounds_env.r2, "train"), read(rounds_env.r2, "test"), read(rounds_env.r1, "train")
    assert train["mode"] == "incremental" and train["delta_pairs"] == train["expected_delta_pairs"] == {"train": 15}  # 3 new * 4 old + 3 new pairs
    assert test["delta_pairs"] == {"test": 0}
    assert train["floor"]["global"] == test["floor"]["global"] == first["floor"]["global"] == 2.0 and train["floor"]["source"] == "base_summary"
    before, after = (r / "relations" / "relations.csv" for r in (rounds_env.r1, rounds_env.r2))
    assert after.read_bytes().startswith(before.read_bytes())
    assert len(_rows(after)) == len(_rows(before)) + 15
    text = rounds_env.log.read_text(encoding="utf-8")
    assert "relations/train/relations.csv.review.html" in text and "export/merge-report.json" in text and "start merge_bundles --base" in text


def _rows(path):
    import csv
    with open(path, newline="", encoding="utf-8") as handle:
        return [row for row in csv.reader(handle) if row]


def test_round_two_train_warm_starts_and_compares_rounds(rounds_env):
    import torch
    from psdl1.checkpoint import read_checkpoint
    saved = torch.load(rounds_env.r2 / "train" / "dl1.pt", map_location="cpu", weights_only=True)
    meta = saved["metadata"]
    assert meta["parent"]["weights_hash"] == read_checkpoint(rounds_env.r1 / "train" / "dl1.pt")["weights_hash"]
    assert meta["round"] == 2 and meta["contract"]["lr"] == 5e-4 and meta["contract"]["new_edge_repeat"] == 2 and saved["epoch"] == 2
    comparison = json.loads((rounds_env.r2 / "train" / "round-comparison.json").read_text(encoding="utf-8"))
    assert [q["query_id"] for q in comparison["common_queries"]] == ["C1", "C2"]
    text = rounds_env.log.read_text(encoding="utf-8")
    assert "round override: --epochs 2 --lr 0.0005" in text and "start compare_rounds" in text and "start train" in text and "--init-from" in text


def test_round_two_second_run_skips_everything_except_verify(rounds_env):
    work, log = rounds_env.work, rounds_env.log
    snapshot = lambda: {p: sha256(p) for p in work.rglob("*") if p.is_file() and p.parent.name != "logs"}
    before, taken = log.read_text(encoding="utf-8"), snapshot()
    result = run_script(rounds_env.assets, rounds_env.base)
    assert result.returncode == 0, result.stderr[-3000:]
    second = log.read_text(encoding="utf-8")[len(before):]
    starts = [line.split("start ", 1)[1].split()[0:2] for line in second.splitlines() if "] " in line and " start " in line]
    assert starts == [["import_png", "--verify-bundle"], ["import_png", "--verify-bundle"], ["merge_bundles", "--verify"]]  # the only jobs that run again
    kinds = " ".join(skips(second))
    for step in ("METADATA", "STACK_ORDER", "export-new", "merge", "grid report", "split", "relations"):
        assert f"skip {step} (exists:" in kinds
    assert snapshot() == taken
    again = run_script(rounds_env.train, rounds_env.base)  # train stops on the existing checkpoint and points at --resume without --init-from
    assert again.returncode != 0 and "already exists" in again.stderr and "train --resume" in again.stderr and "without --init-from" in again.stderr and " start " not in again.stderr
    assert snapshot() == taken


def test_round_two_evaluate_only_reuses_evaluation_and_comparison(stages, rounds_env):
    work = rounds_env.work
    snapshot = lambda: {p: sha256(p) for p in work.rglob("*") if p.is_file() and p.parent.name != "logs"}
    taken = snapshot()
    only = script_in(stages, "psdl1_train.py", rounds_env.base / "s-train-only", **round_two_train_values(work, EVALUATE_ONLY=True))
    result = run_script(only, rounds_env.base)
    assert result.returncode == 0, result.stderr[-3000:]
    kinds = " ".join(skips(result.stderr))
    assert "skip evaluate (exists:" in kinds and "skip compare (exists:" in kinds
    assert " start train" not in result.stderr and " start evaluate" not in result.stderr and " start compare_rounds" not in result.stderr
    assert "was trained for 2 epochs (target --epochs: 2)" in result.stderr and "WARNING" not in result.stderr
    assert snapshot() == taken


def test_round_two_requires_the_previous_round_and_refuses_pretrain(stages, rounds_env, tmp_path):
    empty = run_script(script_in(stages, "psdl1_assets.py", tmp_path / "a", **round_two_values(tmp_path / "no-png", tmp_path / "w", ROUND_MODELS_2)), tmp_path)
    assert empty.returncode != 0 and "STOP:" in empty.stderr and "needs the finished round 1" in empty.stderr and " start " not in empty.stderr and "psdl1 from" not in empty.stderr
    pretrain = run_script(script_in(stages, "psdl1_train.py", tmp_path / "b", **round_two_train_values(rounds_env.work, PRETRAIN=True)), tmp_path)
    assert pretrain.returncode != 0 and "STOP:" in pretrain.stderr and "replaces pretraining" in pretrain.stderr and " start " not in pretrain.stderr and "psdl1 from" not in pretrain.stderr
    work = tmp_path / "work"
    for rel in ("rounds/r02/export/manifest.json", "rounds/r02/inputs/split.json", "rounds/r02/relations/relations.csv"):
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_text("x")
    nothing = run_script(script_in(stages, "psdl1_train.py", tmp_path / "c", **round_two_train_values(work)), tmp_path)
    assert nothing.returncode != 0 and "STOP:" in nothing.stderr and "previous round's checkpoint" in nothing.stderr and " start " not in nothing.stderr and "psdl1 from" not in nothing.stderr


def test_previous_checkpoint_round_must_match_when_recorded(stages, monkeypatch, capsys):
    import psdl1.checkpoint as checkpoint
    g = stage_globals(stages, "psdl1_train.py")
    g["ROUND"] = 3
    for saved in ({"metadata": {}}, {"metadata": {"round": 2}}):
        monkeypatch.setattr(checkpoint, "read_checkpoint", lambda path, saved=saved: saved)
        g["_previous_checkpoint"](Path("dl1.pt"))
    for saved in ({"metadata": {"round": 5}}, {"metadata": {"round": 3}}):
        monkeypatch.setattr(checkpoint, "read_checkpoint", lambda path, saved=saved: saved)
        with pytest.raises(SystemExit) as stopped:
            g["_previous_checkpoint"](Path("dl1.pt"))
        assert stopped.value.code == 2 and "needs round 2" in capsys.readouterr().err
    monkeypatch.setattr(checkpoint, "read_checkpoint", lambda path: (_ for _ in ()).throw(ValueError("broken")))
    with pytest.raises(SystemExit):
        g["_previous_checkpoint"](Path("dl1.pt"))
    assert "cannot read the previous round's checkpoint" in capsys.readouterr().err


@pytest.mark.parametrize("value", [0, -1, True, "1", 1.0])
def test_round_must_be_none_or_a_positive_integer(stages, value, capsys):
    g = stage_globals(stages, "psdl1_assets.py")
    with pytest.raises(SystemExit) as stopped:
        g["_check_round"](value)
    assert stopped.value.code == 2 and "ROUND must be None or an integer >= 1" in capsys.readouterr().err
    for fine in (None, 1, 7):
        g["_check_round"](fine)


@pytest.mark.parametrize("key", ["TRAIN_ARGS", "PRETRAIN_ARGS", "EVALUATE_ARGS"])
@pytest.mark.parametrize("flag", ["--init-from", "--init", "--round", "--new-edge"])
def test_round_reserved_train_arguments_stop_before_anything_starts(stages, rounds_env, key, flag, capsys):
    g = stage_globals(stages, "psdl1_train.py")
    g.update(WORK=str(rounds_env.work), ROUND=2, PRETRAIN=False, **{key: [flag, "x"]})
    with pytest.raises(SystemExit) as stopped:
        g["stage"]()
    err = capsys.readouterr().err
    assert stopped.value.code == 2 and "must not contain" in err and " start " not in err


@pytest.mark.parametrize("arg", ["--base-sp", "--base-split", "--base-relations=x", "--floor-j", "--floor-json", "--floor-json=x"])
def test_round_reserved_screen_arguments_stop_before_anything_starts(stages, tmp_path, arg, capsys):
    g = stage_globals(stages, "psdl1_assets.py")
    g.update(WORK=str(tmp_path / "w"), ROUND=2, SCREEN_ARGS=[arg, "x"])
    with pytest.raises(SystemExit) as stopped:
        g["stage"]()
    err = capsys.readouterr().err
    assert stopped.value.code == 2 and "must not contain" in err and " start " not in err and not (tmp_path / "w").exists()
    g.update(ROUND=1)  # the incremental flags are not reserved before round 2 (only the exact --floor family keeps today's rules)
    g["_check_screen_args"]()


@pytest.mark.parametrize("round_value", [None, 1, 2])
@pytest.mark.parametrize("args,flat_stops", [(["--floor", "2"], False), (["--floor=2"], False), (["--flo", "2"], True), (["--floor-j", "x"], False)])
def test_floor_arguments_follow_the_round_mode(stages, round_value, args, flat_stops, capsys):
    g = stage_globals(stages, "psdl1_assets.py")
    g.update(ROUND=round_value, SCREEN_ARGS=args)
    if round_value == 2 or flat_stops:
        with pytest.raises(SystemExit) as stopped:
            g["_check_screen_args"]()
        err = capsys.readouterr().err
        assert stopped.value.code == 2
        if round_value == 2:  # an exact --floor (and its abbreviations) is refused; --floor-j is an abbreviation of the reserved --floor-json
            assert ("abbreviation of --floor-json" in err) == (args[0] == "--floor-j") and ("round-1 floor" in err) == (args[0] != "--floor-j")
    else:
        g["_check_screen_args"]()  # ROUND None/1 keep today's behaviour


def test_incremental_relations_keep_base_bytes_and_stop_on_bad_rows(stages, tmp_path, capsys):
    g = stage_globals(stages, "psdl1_assets.py")
    base, train, test = tmp_path / "base.csv", tmp_path / "train.csv", tmp_path / "test.csv"
    base.write_bytes(b"A,B,similar\r\nA,C,different")  # hand-edited bytes: CRLF and no trailing newline
    train.write_text("D,A,different\n")
    test.write_text("")
    data, count = g["_merge_incremental"](base, [train, test], ["A", "B", "C", "D"])
    assert data == b"A,B,similar\r\nA,C,different\nD,A,different\n" and count == 3
    for rows, ids, message in (("C,A,similar\n", ["A", "B", "C", "D"], "more than once"), ("D,Z,different\n", ["A", "B", "C", "D"], "invalid"), ("D,A,maybe\n", ["A", "B", "C", "D"], "not product_a")):
        train.write_text(rows)
        with pytest.raises(SystemExit) as stopped:
            g["_merge_incremental"](base, [train, test], ids)
        assert stopped.value.code == 2 and message in capsys.readouterr().err


def test_hand_edited_round_one_relations_are_the_byte_prefix_of_round_two(alt_env):
    after = (alt_env.r2 / "relations" / "relations.csv").read_bytes()
    assert after.startswith(alt_env.edited) and after[len(alt_env.edited):len(alt_env.edited) + 1] == b"\n"  # the missing last newline was added, nothing else changed
    summary = json.loads((alt_env.r2 / "relations" / "train" / "relations.csv.summary.json").read_text(encoding="utf-8"))
    assert summary["delta_pairs"] == {"train": 9} and len(_rows(alt_env.r2 / "relations" / "relations.csv")) == 7 + 9


def test_flat_work_moved_into_r01_supports_round_two(stages, pngs, flat_env, tmp_path):
    work = tmp_path / "work"
    for name in ("inputs", "export", "reports", "relations", "train"):
        clone_tree(flat_env.work / name, work / "rounds" / "r01" / name)
    assets = run_script(script_in(stages, "psdl1_assets.py", tmp_path / "a", **round_two_values(pngs["two"], work, ROUND_MODELS_2)), tmp_path)
    assert assets.returncode == 0, assets.stderr[-3000:]
    train = run_script(script_in(stages, "psdl1_train.py", tmp_path / "b", **round_two_train_values(work, ROUND_EPOCHS=1)), tmp_path)
    assert train.returncode == 0, train.stderr[-3000:]
    import torch
    from psdl1.checkpoint import read_checkpoint
    meta = torch.load(work / "rounds" / "r02" / "train" / "dl1.pt", map_location="cpu", weights_only=True)["metadata"]
    assert meta["parent"]["weights_hash"] == read_checkpoint(flat_env.work / "train" / "dl1.pt")["weights_hash"] and meta["round"] == 2
    assert (work / "rounds" / "r02" / "train" / "round-comparison.json").is_file()


def test_failed_merge_is_kept_aside_and_the_next_run_merges_again(stages, pngs, rounds_env, tmp_path, monkeypatch, capsys):
    from psdl1.merge_bundles import verify_merged_bundle
    work = tmp_path / "work"
    clone_tree(rounds_env.r1, work / "rounds" / "r01")
    values = round_two_values(pngs["two"], work, ROUND_MODELS_2)
    assert run_inprocess(stages, "psdl1_assets.py", tmp_path / "a", monkeypatch, patch=fail_first_merge, **values) == 2
    r2 = work / "rounds" / "r02"
    kept = [p for p in r2.iterdir() if p.name.startswith("export.failed-")]
    assert len(kept) == 1 and (kept[0] / "merge-report.json").read_text() == "{}" and not (r2 / "export").exists() and (r2 / "export-new" / "manifest.json").is_file()
    assert "merge_bundles failed" in capsys.readouterr().err
    assert run_inprocess(stages, "psdl1_assets.py", tmp_path / "b", monkeypatch, **values) == 0
    assert verify_merged_bundle(r2 / "export")["product_count"] == 9 and kept[0].is_dir()
    assert "skip export-new (exists:" in capsys.readouterr().err


def test_round_two_metadata_is_frozen_once_export_new_exists(stages, pngs, rounds_env, tmp_path, monkeypatch, capsys):
    work = tmp_path / "work"
    clone_tree(rounds_env.r1, work / "rounds" / "r01")
    values = round_two_values(pngs["two"], work, ROUND_MODELS_2)
    assert run_inprocess(stages, "psdl1_assets.py", tmp_path / "a", monkeypatch, patch=fail_first_merge, **values) == 2  # export-new succeeded, the merge failed
    capsys.readouterr()
    r2 = work / "rounds" / "r02"
    meta, before = r2 / "inputs" / "metadata.json", sha256(r2 / "inputs" / "metadata.json")
    changed = lineage_metadata({**ROUND_MODELS_2, "D1": "FZ"})
    assert run_inprocess(stages, "psdl1_assets.py", tmp_path / "b", monkeypatch, **{**values, "METADATA": changed}) == 2
    err = capsys.readouterr().err
    assert "STOP:" in err and "differs from METADATA" in err and "export-new" in err and "start import_png" not in err and "start merge_bundles" not in err
    assert sha256(meta) == before and (r2 / "export-new" / "manifest.json").is_file() and any(p.name.startswith("export.failed-") for p in r2.iterdir())
    stack = dict(values, STACK_ORDER={**STACK_ORDER, "order": ["U", "N", "C4", "U5", "C6"]})  # STACK_ORDER is frozen the same way
    assert run_inprocess(stages, "psdl1_assets.py", tmp_path / "c", monkeypatch, **stack) == 2
    assert "differs from STACK_ORDER" in capsys.readouterr().err


def test_merge_is_not_reused_when_it_was_made_from_other_inputs(stages, pngs, rounds_env, alt_env, tmp_path, monkeypatch, capsys):
    work = tmp_path / "work"
    clone_tree(rounds_env.r1, work / "rounds" / "r01")
    clone_tree(rounds_env.r2 / "export", work / "rounds" / "r02" / "export")  # a valid merged bundle of r01 + (D1, E1, E2)
    clone_tree(alt_env.r2 / "export-new", work / "rounds" / "r02" / "export-new")  # a valid, different new bundle (G1, G2)
    snapshot = lambda: {p: sha256(p) for p in (work / "rounds" / "r02" / "export").rglob("*") if p.is_file()}
    taken = snapshot()
    values = round_two_values(pngs["alt"], work, {"G1": "FG", "G2": "FG"})
    assert run_inprocess(stages, "psdl1_assets.py", tmp_path / "a", monkeypatch, **values) == 2
    err = capsys.readouterr().err
    assert "STOP:" in err and "was not made from the current" in err and "manifest_sha256 differs" in err and " start merge_bundles" not in err
    assert snapshot() == taken


def test_round_three_merges_onto_a_merged_base(stages, pngs, rounds_env, tmp_path):
    work = tmp_path / "work"
    clone_tree(rounds_env.work / "rounds", work / "rounds")
    values = assets_values(pngs["three"], work, ROUND=3, METADATA=lineage_metadata({"F1": "FF"}), SCREEN_ARGS=[], NO_PREVIEW=True, VALIDATION_FRACTION=0.0)
    result = run_script(script_in(stages, "psdl1_assets.py", tmp_path / "a", **values), tmp_path)
    assert result.returncode == 0, result.stderr[-3000:]
    report = json.loads((work / "rounds" / "r03" / "export" / "merge-report.json").read_text(encoding="utf-8"))
    assert report["round"] == 3 and report["base"]["kind"] == "merge" and report["product_count"] == 10 and "start merge_bundles --verify" in result.stderr
    assert "--base-round" not in result.stderr  # only an import base bundle needs it
    summary = json.loads((work / "rounds" / "r03" / "relations" / "train" / "relations.csv.summary.json").read_text(encoding="utf-8"))
    assert summary["delta_pairs"] == {"train": 7} and summary["floor"]["global"] == 2.0


def test_round_comparison_is_reused_only_for_the_same_evaluation_files(stages, rounds_env, tmp_path, monkeypatch, capsys):
    g = stage_globals(stages, "psdl1_train.py")
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    previous, current, out = tmp_path / "previous.json", tmp_path / "current.json", tmp_path / "round-comparison.json"
    shutil.copy(rounds_env.r1 / "train" / "evaluation.json", previous)
    shutil.copy(rounds_env.r2 / "train" / "evaluation.json", current)
    g["_compare_rounds"](tmp_path / "missing.json", current, out)
    assert not out.exists() and "skip compare (no previous evaluation" in capsys.readouterr().err
    g["_compare_rounds"](previous, current, out)
    assert out.is_file()
    capsys.readouterr()
    made = sha256(out)
    g["_compare_rounds"](previous, current, out)
    assert "skip compare (exists:" in capsys.readouterr().err and sha256(out) == made
    document = json.loads(current.read_text(encoding="utf-8"))  # same checkpoint weights, different results
    document["recall_at_k"] = 0.0
    for query in document["queries"]:
        query["recall_at_k"] = 0.0
    current.write_text(json.dumps(document))
    with pytest.raises(SystemExit) as stopped:
        g["_compare_rounds"](previous, current, out)
    assert stopped.value.code == 2 and "does not match the current evaluations" in capsys.readouterr().err and sha256(out) == made


def test_round_one_checkpoint_records_round_one_and_round_two_checks_it(stages, rounds_env, tmp_path, monkeypatch, capsys):
    import torch
    assert torch.load(rounds_env.r1 / "train" / "dl1.pt", map_location="cpu", weights_only=True)["metadata"]["round"] == 1
    g = stage_globals(stages, "psdl1_train.py")  # --round is filled by the runner in ROUND = 1 as well
    g.update(WORK=str(rounds_env.work), ROUND=1, TRAIN_ARGS=["--round", "1"])
    with pytest.raises(SystemExit) as stopped:
        g["stage"]()
    assert stopped.value.code == 2 and "must not contain" in capsys.readouterr().err
    work = tmp_path / "work"  # a parent trained as round 2 cannot be the parent of round 2
    clone_tree(rounds_env.r2 / "train", work / "rounds" / "r01" / "train")
    for rel in ("rounds/r02/export/manifest.json", "rounds/r02/inputs/split.json", "rounds/r02/relations/relations.csv"):
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_text("x")
    assert run_inprocess(stages, "psdl1_train.py", tmp_path / "a", monkeypatch, **round_two_train_values(work)) == 2
    assert "was trained as round 2, but ROUND = 2 needs round 1" in capsys.readouterr().err


def test_compare_is_skipped_when_the_evaluation_belongs_to_an_older_checkpoint(stages, rounds_env, tmp_path, monkeypatch, capsys):
    work = tmp_path / "work"
    clone_tree(rounds_env.r1, work / "rounds" / "r01")
    clone_tree(rounds_env.r2, work / "rounds" / "r02")
    train = work / "rounds" / "r02" / "train"
    (train / "dl1.pt").unlink()
    (train / "round-comparison.json").unlink()  # the older checkpoint's evaluation.json stays
    assert run_inprocess(stages, "psdl1_train.py", tmp_path / "a", monkeypatch, **round_two_train_values(work, EVALUATE=False, ROUND_EPOCHS=1)) == 0
    err = capsys.readouterr().err
    assert (train / "dl1.pt").is_file() and not (train / "round-comparison.json").exists()
    assert "skip compare (evaluation was not requested" in err and "start compare_rounds" not in err
    none = run_inprocess(stages, "psdl1_train.py", tmp_path / "b", monkeypatch, **round_two_train_values(work, EVALUATE=False, EVALUATE_ONLY=False))
    assert none == 2  # the checkpoint exists now, training stops as always
    (train / "evaluation.json").unlink()
    assert run_inprocess(stages, "psdl1_train.py", tmp_path / "c", monkeypatch, **round_two_train_values(work, EVALUATE=False, EVALUATE_ONLY=True)) == 0
    assert (train / "round-comparison.json").is_file() and "start compare_rounds" in capsys.readouterr().err  # a fresh evaluation of the current checkpoint is compared


def test_floor_json_counts_as_a_user_floor_source(stages):
    g = stage_globals(stages, "psdl1_assets.py")
    for args in (["--floor", "2"], ["--floor=2"], ["--floor-json", "s.json"], ["--floor-j=s.json"]):
        assert g["_user_floor_source"](args) is True
    for args in ([], ["--factor", "8"], ["--groups", "g.json"]):
        assert g["_user_floor_source"](args) is False


def test_merged_folder_without_bundle_integrity_stops_as_a_partial_merge(stages, pngs, rounds_env, tmp_path, monkeypatch, capsys):
    work = tmp_path / "work"
    clone_tree(rounds_env.r1, work / "rounds" / "r01")
    clone_tree(rounds_env.r2 / "export", work / "rounds" / "r02" / "export")
    clone_tree(rounds_env.r2 / "export-new", work / "rounds" / "r02" / "export-new")
    (work / "rounds" / "r02" / "export" / "bundle-integrity.json").unlink()
    assert run_inprocess(stages, "psdl1_assets.py", tmp_path / "a", monkeypatch, **round_two_values(pngs["two"], work, ROUND_MODELS_2)) == 2
    err = capsys.readouterr().err
    assert "no bundle-integrity.json" in err and "partial merge" in err and "start merge_bundles" not in err


def test_round_two_lists_the_export_new_preview(stages, pngs, rounds_env, tmp_path, monkeypatch, capsys):
    work = tmp_path / "work"
    clone_tree(rounds_env.r1, work / "rounds" / "r01")
    clone_tree(rounds_env.r2, work / "rounds" / "r02")
    preview = work / "rounds" / "r02" / "export-new" / "preview.html"
    assert not preview.exists()  # the shared run used NO_PREVIEW
    preview.write_text("<html>")

    def skip_bundle_verify(g):  # a hand-added file would fail the real hash verification; only the listing is under test
        real = g["_call"]
        g["_call"] = lambda name, args: None if name == "import_png" and "--verify-bundle" in [str(a) for a in args] else real(name, args)
    assert run_inprocess(stages, "psdl1_assets.py", tmp_path / "a", monkeypatch, patch=skip_bundle_verify, **round_two_values(pngs["two"], work, ROUND_MODELS_2)) == 0
    assert f"  {preview}" in capsys.readouterr().err


def test_merge_and_compare_jobs_run_from_the_single_file(bundle, rounds_env, tmp_path):
    out, _ = bundle
    namespace = __import__("runpy").run_path(str(out), run_name="psdl1_job")
    assert "merge_bundles" in namespace["ALLOWED"] and "compare_rounds" in namespace["ALLOWED"]
    comparison = tmp_path / "comparison.json"
    jobs = [("merge_bundles", ["--verify", str(rounds_env.r2 / "export")]),
            ("compare_rounds", ["--previous", str(rounds_env.r1 / "train" / "evaluation.json"), "--current", str(rounds_env.r2 / "train" / "evaluation.json"), "--out", str(comparison)])]
    script = tmp_path / "job.py"
    script.write_text(with_jobs(out.read_text(encoding="utf-8"), jobs, tmp_path / "job.log", tmp_path / "pkg"), encoding="utf-8")
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=tmp_path, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    assert result.returncode == 0, result.stderr[-3000:]
    assert json.loads(comparison.read_text(encoding="utf-8"))["schema_version"] == "dl1_round_comparison_v1"


# ---- --floor-quantile (T6) ----

QUANTILE_FORMS = [["--floor-quantile", "0.5"], ["--floor-quantile=0.5"]]


@pytest.mark.parametrize("args", QUANTILE_FORMS)
def test_floor_quantile_helpers_split_the_train_only_argument(stages, args):
    g = stage_globals(stages, "psdl1_assets.py")
    mixed = ["--groups", "g.json", *args, "--margin", "0.1"]
    assert g["_floor_quantile_arg"](mixed) is True and g["_floor_quantile_arg"](["--groups", "g.json"]) is False
    assert g["_without_floor_quantile"](mixed) == ["--groups", "g.json", "--margin", "0.1"]
    assert g["_user_floor_source"](args) is False  # a rule that computes the floor, not a user floor
    assert g["_user_floor_source"]([*args, "--floor", "1"]) is True


@pytest.mark.parametrize("args", QUANTILE_FORMS)
def test_floor_quantile_goes_to_train_only_and_test_gets_the_train_floor(stages, tmp_path, args):
    png = tmp_path / "png"
    names = ["W1", "W2", "W3", "W4"]
    for name in names:
        write_wide_import_fixture(png, model=name)
    metadata = {"metadata_schema_version": "dl1_png_metadata_v1", "models": {
        n: {"lineage_id": "F1" if n in ("W1", "W2") else "F2", "design_version": "V1"} for n in names}}
    work = tmp_path / "work"
    (work / "inputs").mkdir(parents=True)
    (work / "inputs" / "split.json").write_text(json.dumps({"schema_version": SPLIT_VERSION, "train": ["W1", "W2"], "validation": [], "test": ["W3", "W4"]}))
    script = stage_script(stages, "psdl1_assets.py", tmp_path, **assets_values(png, work, METADATA=metadata, NO_PREVIEW=True, SCREEN_ARGS=args))
    result = run_script(script, tmp_path)
    assert result.returncode == 0, result.stderr[-3000:]
    train, test = (json.loads((work / "relations" / part / "relations.csv.summary.json").read_text(encoding="utf-8")) for part in ("train", "test"))
    assert train["floor"]["quantile"] == 0.5 and train["floor"]["quantile_method"] == "lower_index"
    assert "quantile" not in test["floor"] and test["floor"]["global"] == train["floor"]["global"]


@pytest.mark.parametrize("args", QUANTILE_FORMS)
def test_floor_quantile_is_in_the_train_call_arguments_and_not_in_the_test_call(stages, tmp_path, args):
    g = stage_globals(stages, "psdl1_assets.py")
    calls = []

    def fake_call(name, call_args):
        calls.append([str(a) for a in call_args])
        csv = Path(call_args[call_args.index("--out") + 1])
        csv.write_text("", encoding="utf-8")
        csv.with_name(csv.name + ".review.html").write_text("", encoding="utf-8")
        csv.with_name(csv.name + ".summary.json").write_text(json.dumps({"floor": {"global": 0.25}}), encoding="utf-8")

    g.update(_call=fake_call, SCREEN_ARGS=["--groups", "g.json", *args])
    train = g["_screen"]("m.json", "s.json", "train", tmp_path / "train", [])
    floor = ["--floor", repr(train["floor"]["global"])]
    g["_screen"]("m.json", "s.json", "test", tmp_path / "test", floor, drop_quantile=True)
    train_call, test_call = calls
    assert any(a.split("=")[0] == "--floor-quantile" for a in train_call) and (len(args) == 1 or "0.5" in train_call)
    assert not any(a.split("=")[0] == "--floor-quantile" for a in test_call) and "0.5" not in test_call
    assert test_call[-2:] == ["--floor", "0.25"] and "--groups" in test_call


@pytest.mark.parametrize("args", [
    ["--floor-quantile", "0.2", "--floor", "0.5"], ["--floor-quantile=0.2", "--floor=0.5"], ["--floor-quantile", "0.2", "--floor-json", "s.json"],
    ["--floor-quantile", "0.2", "--floor-j=s.json"], ["--floor-q", "0.2"], ["--floor-qu=0.2"], ["--floor-quantil", "0.2"], ["--floor-", "x"]])
def test_floor_quantile_conflicts_and_abbreviations_stop_before_anything_starts(stages, tmp_path, args, capsys):
    g = stage_globals(stages, "psdl1_assets.py")
    g.update(WORK=str(tmp_path / "w"), ROUND=None, SCREEN_ARGS=args)
    with pytest.raises(SystemExit) as stopped:
        g["stage"]()
    err = capsys.readouterr().err
    assert stopped.value.code == 2 and "STOP:" in err and " start " not in err and not (tmp_path / "w").exists()


@pytest.mark.parametrize("args", QUANTILE_FORMS + [["--floor-q", "0.5"], ["--floor-", "x"]])
def test_floor_quantile_is_refused_from_round_two_before_export(stages, tmp_path, args, capsys):
    g = stage_globals(stages, "psdl1_assets.py")
    g.update(WORK=str(tmp_path / "w"), ROUND=2, SCREEN_ARGS=args)
    with pytest.raises(SystemExit) as stopped:
        g["stage"]()
    err = capsys.readouterr().err
    assert stopped.value.code == 2 and "STOP:" in err and " start " not in err and not (tmp_path / "w").exists()
    if args[0] == "--floor-quantile" or args[0].startswith("--floor-quantile="):
        assert "round-1 floor" in err


def test_floor_quantile_alone_is_accepted_in_round_one(stages):
    g = stage_globals(stages, "psdl1_assets.py")
    for rnd in (None, 1):
        g.update(ROUND=rnd, SCREEN_ARGS=["--floor-quantile", "0.2"])
        g["_check_screen_args"]()
