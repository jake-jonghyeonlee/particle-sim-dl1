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
from psdl1.split import SPLIT_VERSION  # noqa: E402
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
