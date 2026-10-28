import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

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
    values = dict(PNG_ROOT=str(png), WORK=str(work), GRID=128, NO_PREVIEW=False, METADATA=METADATA, STACK_ORDER=STACK_ORDER, SPLIT_SEED=0,
                  VALIDATION_FRACTION=0.2, TEST_FRACTION=0.2, SCREEN_ARGS=["--floor", "2"])
    return {**values, **overrides}


def train_values(work, **overrides):
    values = dict(WORK=str(work), PRETRAIN=False, PRETRAIN_ARGS=[], TRAIN_ARGS=["--backbone", "tiny", "--epochs", "1", "--batch-size", "2", "--allow-grid-check-failure"],
                  EVALUATE=True, EVALUATE_ARGS=[], EVALUATE_ONLY=False)
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
