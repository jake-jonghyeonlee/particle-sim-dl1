#!/usr/bin/env python3
"""src/psdl1/*.py 전체를 .py 파일 하나로 묶는다. 명령줄 옵션을 넘길 수 없고 .py만 올릴 수 있는 서버용.

--out FILE: 작업 목록(JOBS) 방식. 만들어진 파일 맨 위의 EDIT 블록(JOBS, LOG_FILE, WORK_DIR)만 고치면 된다.
--out-dir DIR: 단계별 파일 3개(psdl1_inspect.py, psdl1_assets.py, psdl1_train.py). 각 파일의 EDIT 블록에 그 단계의 값만 있고,
모든 출력은 WORK 아래에 생기며, 다시 실행하면 끝난 단계는 건너뛴다.
실행하면 패키지를 실제 폴더에 풀고(바이트 그대로라 code_hash가 저장소와 같다) psdl1.<작업>.main(인자)을 호출한다.
서버에는 torch, numpy, Pillow(timm은 선택)만 있으면 된다."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]

DOC = '''#!/usr/bin/env python3
"""@@TITLE@@ (자동 생성: scripts/build_single_file.py). 소스 해시 @@SHA@@.
서버 wrapper가 명령줄 인자를 넘기지 못할 때 쓴다. 아래 EDIT 블록만 고치고 이 파일을 그대로 실행한다."""
'''

HEAD_JOBS = r'''# ============================== EDIT BELOW: 이 블록만 고치세요 ==============================
# 작업 이름은 train, pretrain, screen_relations, evaluate, embed_plot, search, import_png, data, relations_draft 중 하나다.
# 인자는 명령줄에서 쓰던 그대로 문자열 리스트로 적는다. 경로는 서버 기준 절대 경로를 권장한다. 작업은 위에서부터 차례로 실행되고 하나가 실패하면 멈춘다.
JOBS = [
    ("train", [
        "--manifest", "/data/export/manifest.json",
        "--split", "/data/split.json",
        "--relations", "/data/relations_final.csv",
        "--weights", "/data/runs/pretrain1/backbone.pt",   # 사전학습 가중치를 안 쓰면 이 두 줄을 지운다
        "--backbone", "tiny",
        "--out", "/data/runs/train_pre",
        "--device", "cuda",
        "--batch-size", "2",
        "--epochs", "100",
        "--allow-grid-check-failure",                      # 얇은 선 경고를 승인한 경우에만
        # "--resume", "/data/runs/train_pre/dl1.pt",       # 중단 후 이어서 할 때(--epochs를 더 크게)
    ]),
    # ("pretrain", ["--manifest", "/data/export/manifest.json", "--split", "/data/split.json", "--out", "/data/runs/pretrain1",
    #               "--device", "cuda", "--batch-size", "2", "--epochs", "100"]),
    # ("screen_relations", ["--manifest", "/data/export/manifest.json", "--split", "/data/split.json", "--out", "/data/relations_new.csv",
    #                       "--groups", "/data/groups.json", "--partitions", "train"]),
    # ("evaluate", ["--ckpt", "/data/runs/train_pre/dl1.pt", "--manifest", "/data/export/manifest.json", "--split", "/data/split.json",
    #               "--relations", "/data/relations_final.csv", "--topk", "5", "--out", "/data/runs/evaluation1.json"]),
    # ("embed_plot", ["--ckpt", "/data/runs/train_pre/dl1.pt", "--manifest", "/data/export/manifest.json", "--out", "/data/runs/report1"]),
]
LOG_FILE = "/data/runs/psdl1_job.log"   # stdout/stderr(진행 줄)를 이 파일에도 남긴다. 필요 없으면 None
WORK_DIR = None                          # 패키지를 풀 폴더. None이면 이 파일 옆, 쓸 수 없으면 임시 폴더
# ============================== EDIT ABOVE ==============================================
'''

HEAD_INSPECT = r'''# ============================== EDIT BELOW: 이 블록만 고치세요 ==============================
# grid(해상도)를 모를 때만 쓰는 1단계다. 후보 grid마다 검사 결과(미리보기 HTML 포함)를 WORK/inspect-g<grid>... 에 만든다.
# 결과를 보고 grid를 정한 다음 psdl1_assets.py 의 GRID 에 적는다. 이미 아는 grid가 있으면 이 파일은 건너뛴다.
PNG_ROOT = "/data/png"                  # 모델별 PNG 폴더가 들어 있는 입력 폴더
WORK = "/data/work"                     # 모든 출력이 생기는 폴더(쓸 수 있어야 한다)
GRID_CANDIDATES = [128, 256, 512]       # 비교해 볼 grid 후보
NO_PREVIEW = False                      # True 면 미리보기 이미지·HTML 을 만들지 않는다(빠르지만 눈으로 확인하기 어렵다)
LOG_FILE = WORK + "/logs/inspect.log"   # stdout/stderr(진행 줄)를 이 파일에도 남긴다. 필요 없으면 None
WORK_DIR = None                         # 패키지를 풀 폴더. None이면 이 파일 옆, 쓸 수 없으면 임시 폴더
# ============================== EDIT ABOVE ==============================================
'''

HEAD_ASSETS = r'''# ============================== EDIT BELOW: 이 블록만 고치세요 ==============================
# 2단계: PNG -> export(manifest) -> 검증 -> 격자 보고서 -> 자동 split -> relations 초안(검토용 HTML 포함). 학습은 하지 않는다.
# grid 를 알면 inspect 없이 바로 실행한다. 다시 실행하면 끝난 단계는 건너뛴다(로그에 skip 으로 남는다).
PNG_ROOT = "/data/png"                  # 모델별 PNG 폴더가 들어 있는 입력 폴더
WORK = "/data/work"                     # 모든 출력이 생기는 폴더(쓸 수 있어야 한다)
GRID = 256                              # candidate grid (psdl1_inspect.py 결과를 보고 정한다)
NO_PREVIEW = False                      # True 면 export 미리보기를 만들지 않는다
# 모델별 계보(lineage)와 설계 버전. WORK/inputs/metadata.json 으로 저장된다.
METADATA = {
    "metadata_schema_version": "dl1_png_metadata_v1",
    "models": {
        "MODEL_A": {"lineage_id": "FAMILY_A", "design_version": "V1"},
    },
}
# 층 쌓는 순서를 알면 적는다(CNT 관계가 생긴다). 모르면 None. 저장 위치: WORK/inputs/stack-order.json
# 예: {"stack_order_schema_version": "dl1_png_stack_order_v1", "order": ["U", "N", "C4", "U5", "C6", "L"]}
STACK_ORDER = None
SPLIT_SEED = 0                          # 자동 split(계보 단위) 설정. WORK/inputs/split.json 이 있으면 다시 만들지 않고 재사용한다
VALIDATION_FRACTION = 0.2
TEST_FRACTION = 0.2
SCREEN_PARTITIONS = ["train", "validation", "test"]   # relations 스크리닝 대상 구획
SCREEN_ARGS = []                        # screen_relations 추가 인자. 예: ["--groups", "/data/groups.json", "--margin", "0.05"]
LOG_FILE = WORK + "/logs/assets.log"    # stdout/stderr(진행 줄)를 이 파일에도 남긴다. 필요 없으면 None
WORK_DIR = None                         # 패키지를 풀 폴더. None이면 이 파일 옆, 쓸 수 없으면 임시 폴더
# ============================== EDIT ABOVE ==============================================
'''

HEAD_TRAIN = r'''# ============================== EDIT BELOW: 이 블록만 고치세요 ==============================
# 3단계: (선택) 사전학습 -> 학습 -> (선택) 평가. psdl1_assets.py 가 만든 WORK 를 입력으로 쓴다.
# --manifest/--split/--relations/--out/--weights 는 이 파일이 채운다. 아래 인자 리스트에는 나머지 옵션만 적는다.
WORK = "/data/work"                     # psdl1_assets.py 와 같은 폴더
PRETRAIN = False                        # True 면 backbone 사전학습을 먼저 하고 그 가중치로 학습한다
PRETRAIN_ARGS = ["--backbone", "tiny", "--device", "cuda", "--batch-size", "2", "--epochs", "100"]
TRAIN_ARGS = ["--backbone", "tiny", "--device", "cuda", "--batch-size", "2", "--epochs", "100"]
EVALUATE = True                         # validation/test 제품이 있을 때만 평가한다
EVALUATE_ARGS = ["--topk", "5"]
LOG_FILE = WORK + "/logs/train.log"     # stdout/stderr(진행 줄)를 이 파일에도 남긴다. 필요 없으면 None
WORK_DIR = None                         # 패키지를 풀 폴더. None이면 이 파일 옆, 쓸 수 없으면 임시 폴더
# ============================== EDIT ABOVE ==============================================
'''

COMMON = r'''
import base64
import hashlib
import importlib
import inspect
import json
import os
import sys
import tempfile
import time
import traceback
import zlib
from pathlib import Path
from types import SimpleNamespace

ALLOWED = ("train", "pretrain", "screen_relations", "evaluate", "embed_plot", "search", "import_png", "data", "relations_draft")
PAYLOAD_SHA256 = "@@SHA@@"
PAYLOAD = (
@@PAYLOAD@@
)


def _files():
    files = json.loads(zlib.decompress(base64.b64decode("".join(PAYLOAD))))
    for name, entry in files.items():
        data = base64.b64decode(entry["data"])
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"embedded file is corrupted: {name}")
        files[name] = data
    return files


def _extract():
    """psdl1 패키지를 실제 폴더에 푼다(바이트 그대로). 이미 같은 내용이 있으면 재사용한다. 풀린 위치(패키지의 상위 폴더)를 돌려준다."""
    files = _files()
    here = globals().get("__file__")
    bases = [Path(WORK_DIR)] if WORK_DIR else []
    bases += ([Path(here).resolve().parent] if here else [Path.cwd()]) + [Path(tempfile.gettempdir())]
    errors = []
    for base in bases:
        root = base / f".psdl1_pkg_{PAYLOAD_SHA256[:12]}"
        package = root / "psdl1"
        try:
            package.mkdir(parents=True, exist_ok=True)
            for name, data in files.items():
                target = package / name
                if not target.is_file() or target.read_bytes() != data:
                    target.write_bytes(data)
            return root
        except OSError as exc:
            errors.append(f"{base}: {exc}")
    raise RuntimeError("cannot extract the psdl1 package anywhere: " + "; ".join(errors))


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


def _stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _say(message):
    print(f"[psdl1_job] {_stamp()} {message}", file=sys.stderr, flush=True)


def _boot():
    """패키지를 풀고 import 한다. 묶음 쪽 psdl1 이 우선이다."""
    root = _extract()
    sys.path.insert(0, str(root))
    import psdl1
    if not str(Path(psdl1.__file__).resolve()).startswith(str(root.resolve())):
        raise RuntimeError(f"another psdl1 shadows the bundled one: {psdl1.__file__}")
    try:
        import torch
        device_info = f"torch {torch.__version__} cuda {torch.version.cuda} available {torch.cuda.is_available()}" + (f" gpu {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "")
    except Exception as exc:  # 로그용 정보일 뿐이다
        device_info = f"torch unavailable: {exc}"
    _say(f"python {sys.version.split()[0]} | {device_info} | psdl1 from {psdl1.__file__}")
    return root


def _call(name, args):
    """psdl1.<name>.main 을 부른다. 0 아닌 반환값이나 SystemExit 은 로그를 남기고 0 아닌 코드로 멈춘다."""
    if name not in ALLOWED:
        raise ValueError(f"unknown job {name!r}; choose from {ALLOWED}")
    argv = [str(a) for a in args]
    _say(f"start {name} {' '.join(argv)}")
    started = time.time()
    main = importlib.import_module(f"psdl1.{name}").main
    sys.argv = [f"psdl1.{name}", *argv]
    try:
        rc = main(argv) if "argv" in inspect.signature(main).parameters else main()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            _say(f"{name} exited with {exc.code!r}")
            raise
        rc = None
    if rc not in (None, 0):
        _say(f"{name} exited with {rc!r}")
        raise SystemExit(rc if isinstance(rc, int) else 1)
    _say(f"done {name} in {time.time() - started:.0f}s")


def _skip(step, path):
    _say(f"skip {step} (exists: {path})")


def _stop(message):
    _say("STOP: " + message)
    raise SystemExit(2)


def _run_with_log(entry):
    log = None
    if LOG_FILE:
        Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        log = open(LOG_FILE, "a", buffering=1, encoding="utf-8")
        sys.stdout, sys.stderr = _Tee(sys.__stdout__, log), _Tee(sys.__stderr__, log)
    try:
        entry()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if log is not None:
            log.flush()

'''

TAIL_JOBS = r'''
def run_jobs(jobs):
    for name, _ in jobs:
        if name not in ALLOWED:
            raise ValueError(f"unknown job {name!r}; choose from {ALLOWED}")
    _boot()
    for name, args in jobs:
        _call(name, args)


def _main():
    _run_with_log(lambda: run_jobs(JOBS))


if __name__ == "__main__":
    _main()
'''

TAIL_INSPECT = r'''
def stage():
    _boot()
    work = Path(WORK)
    work.mkdir(parents=True, exist_ok=True)
    grids = [int(g) for g in GRID_CANDIDATES]
    if not grids:
        _stop("GRID_CANDIDATES is empty")
    out = work / ("inspect-g" + "-".join(str(g) for g in grids))
    if out.exists():
        _skip("inspect", out)
    else:
        _call("import_png", ["--root", PNG_ROOT, "--out", out, "--inspect-grids", *grids, *(["--no-preview"] if NO_PREVIEW else [])])
    preview = out / "preview.html"
    _say(f"check: {preview if preview.is_file() else out}")
    _say("pick the grid from the inspection result, then set GRID in psdl1_assets.py and run it")


def _main():
    _run_with_log(stage)


if __name__ == "__main__":
    _main()
'''

TAIL_ASSETS = r'''
def _write_json_once(path, obj, what):
    """파일이 없으면 쓰고, 같은 내용이면 재사용하고, 다르면 덮어쓰지 않고 멈춘다."""
    new = json.loads(json.dumps(obj))
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            old = object()
        if old != new:
            _stop(f"{path} differs from {what} in the EDIT block; not overwriting. Restore the EDIT value, or move {path} "
                  f"and the dependent WORK/export folder away to start over.")
        _skip(what, path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    _say(f"wrote {what}: {path}")


def _split(manifest, split_path):
    from psdl1.data import _iter_manifest_products
    from psdl1.screen_relations import CHECK_GRID
    from psdl1.split import PARTITIONS, lineage_split, validate_split
    products = [SimpleNamespace(product_id=p.product_id, lineage_id=p.lineage_id) for p in _iter_manifest_products(manifest, grid=CHECK_GRID)]
    if split_path.is_file():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        validate_split(products, split)
        _skip("split", split_path)
    else:
        split = lineage_split(products, seed=SPLIT_SEED, validation_fraction=VALIDATION_FRACTION, test_fraction=TEST_FRACTION)
        tmp = split_path.with_name(split_path.name + ".tmp")
        tmp.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, split_path)
        _say(f"wrote split: {split_path}")
    lineage = {p.product_id: p.lineage_id for p in products}
    for part in PARTITIONS:
        _say(f"split {part}: {len(split[part])} products, {len({lineage[pid] for pid in split[part]})} lineages")
    return split


def stage():
    _boot()
    work = Path(WORK)
    work.mkdir(parents=True, exist_ok=True)
    inputs, export, reports, relations = work / "inputs", work / "export", work / "reports", work / "relations"
    meta_path, order_path = inputs / "metadata.json", inputs / "stack-order.json"
    manifest, split_path, grid_report = export / "manifest.json", inputs / "split.json", reports / "grid-report.json"
    csv = relations / "relations.csv"

    # a. 입력 파일
    _write_json_once(meta_path, METADATA, "METADATA")
    if STACK_ORDER is not None:
        _write_json_once(order_path, STACK_ORDER, "STACK_ORDER")
    elif order_path.exists():
        _stop(f"STACK_ORDER is None but {order_path} exists; set STACK_ORDER to its content or move the file away")

    # b. export
    if manifest.is_file():
        _skip("export", manifest)
    elif export.exists():
        _stop(f"{export} exists without manifest.json (failed or partial export). Read its diagnostic report, fix the cause, move the folder away, and run again")
    else:
        _call("import_png", ["--root", PNG_ROOT, "--out", export, "--export", "--metadata", meta_path, "--candidate-grid", GRID,
                             *(["--stack-order", order_path] if STACK_ORDER is not None else []), *(["--no-preview"] if NO_PREVIEW else [])])

    # c. 묶음 검증(매번)
    _call("import_png", ["--verify-bundle", export])

    # d. 격자 보고서
    reports.mkdir(parents=True, exist_ok=True)
    if grid_report.is_file():
        _skip("grid report", grid_report)
    else:
        _call("data", [manifest, "--report", grid_report])

    # e. 자동 split
    inputs.mkdir(parents=True, exist_ok=True)
    _split(manifest, split_path)

    # f. relations 초안과 검토 페이지
    outputs = [csv, csv.with_name(csv.name + ".review.html"), csv.with_name(csv.name + ".summary.json")]
    found = [p for p in outputs if p.exists()]
    if len(found) == len(outputs):
        _skip("screen_relations", csv)
    elif found:
        _stop("screen_relations left partial output: " + ", ".join(str(p) for p in found) + ". Move the relations folder away and run again")
    else:
        relations.mkdir(parents=True, exist_ok=True)
        _call("screen_relations", ["--manifest", manifest, "--split", split_path, "--out", csv, "--partitions", *SCREEN_PARTITIONS, *SCREEN_ARGS])

    # g. 사람이 확인할 파일
    _say("check these files before training:")
    for path in (export / "preview.html", split_path, outputs[1], grid_report):
        if path.exists():
            _say(f"  {path}")
    _say("after checking, run psdl1_train.py")


def _main():
    _run_with_log(stage)


if __name__ == "__main__":
    _main()
'''

TAIL_TRAIN = r'''
def _pretrain_complete(pdir):
    """끝난 사전학습이면 None, 아니면 이유를 돌려준다. pretrain.json 은 마지막에 쓰이므로 가중치 해시까지 맞아야 끝난 것으로 본다."""
    import torch
    from psdl1.checkpoint import state_hash
    try:
        done = json.loads((pdir / "pretrain.json").read_text(encoding="utf-8"))
        saved = torch.load(pdir / "pretrain.pt", map_location="cpu", weights_only=True)
        backbone = torch.load(pdir / "backbone.pt", map_location="cpu", weights_only=True)
        expected = done["backbone_weights_hash"]
        if saved["metadata"].get("backbone_weights_hash") != expected or state_hash(backbone) != expected:
            return "weights hash differs from pretrain.json (interrupted run)"
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def stage():
    work = Path(WORK)
    manifest, split_path, csv = work / "export" / "manifest.json", work / "inputs" / "split.json", work / "relations" / "relations.csv"
    missing = [str(p) for p in (manifest, split_path, csv) if not p.is_file()]
    if missing:
        _stop("missing " + ", ".join(missing) + ". Run psdl1_assets.py first (same WORK)")
    _boot()
    pdir, tdir, evaluation = work / "pretrain", work / "train", work / "reports" / "evaluation.json"
    weights = []
    if PRETRAIN:
        if pdir.exists():
            reason = _pretrain_complete(pdir)
            if reason:
                _stop(f"{pdir} exists but is not a completed pretraining ({reason}). Resume it with a JOBS-mode file "
                      f"(pretrain --resume {pdir / 'pretrain.pt'}) or move the folder away")
            _skip("pretrain", pdir / "pretrain.json")
        else:
            _call("pretrain", ["--manifest", manifest, "--split", split_path, "--out", pdir, *PRETRAIN_ARGS])
        weights = ["--weights", pdir / "backbone.pt"]
    checkpoint = tdir / "dl1.pt"
    if checkpoint.exists():
        _stop(f"{checkpoint} already exists; psdl1.train would overwrite it, so this file stops. To continue that run, use a JOBS-mode "
              f"file (--out FILE) with train --resume {checkpoint} and a larger --epochs. To train again, move {tdir} away")
    _call("train", ["--manifest", manifest, "--split", split_path, "--relations", csv, "--out", tdir, *weights, *TRAIN_ARGS])
    if EVALUATE:
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if not (split.get("validation") or split.get("test")):
            _say("skip evaluate (no validation or test products in the split)")
        elif evaluation.is_file():
            _skip("evaluate", evaluation)
        else:
            evaluation.parent.mkdir(parents=True, exist_ok=True)
            _call("evaluate", ["--ckpt", checkpoint, "--manifest", manifest, "--split", split_path, "--relations", csv, "--out", evaluation, *EVALUATE_ARGS])
    _say(f"checkpoint: {checkpoint}")


def _main():
    _run_with_log(stage)


if __name__ == "__main__":
    _main()
'''

TITLE_JOBS = "DL1 단일 파일 실행기"
STAGES = {
    "psdl1_inspect.py": ("DL1 1단계 inspect 실행기", HEAD_INSPECT, TAIL_INSPECT),
    "psdl1_assets.py": ("DL1 2단계 assets 실행기", HEAD_ASSETS, TAIL_ASSETS),
    "psdl1_train.py": ("DL1 3단계 train 실행기", HEAD_TRAIN, TAIL_TRAIN),
}


def _payload():
    sources = sorted(p for p in (ROOT / "src/psdl1").glob("*.py"))
    if not sources:
        raise ValueError("no source files found under src/psdl1")
    files = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "data": base64.b64encode(p.read_bytes()).decode()} for p in sources}
    packed = base64.b64encode(zlib.compress(json.dumps(files, sort_keys=True).encode(), 9)).decode()
    sha = hashlib.sha256(json.dumps({n: e["sha256"] for n, e in files.items()}, sort_keys=True).encode()).hexdigest()
    lines = "\n".join(f'    "{packed[i:i + 100]}"' for i in range(0, len(packed), 100))
    return files, sha, lines


def _render(title, head, tail, sha, lines, target):
    text = DOC.replace("@@TITLE@@", title) + head + COMMON + tail
    text = text.replace("@@SHA@@", sha).replace("@@PAYLOAD@@", lines)
    compile(text, str(target), "exec")
    return text


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def build(out, *, overwrite=False):
    out = Path(out)
    if out.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {out}; pass --overwrite or choose a new path")
    files, sha, lines = _payload()
    _write(out, _render(TITLE_JOBS, HEAD_JOBS, TAIL_JOBS, sha, lines, out))
    return {"out": str(out), "bytes": out.stat().st_size, "modules": len(files), "source_sha256": sha}


def build_stages(out_dir, *, overwrite=False):
    out_dir = Path(out_dir)
    targets = {name: out_dir / name for name in STAGES}
    existing = [str(p) for p in targets.values() if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite {', '.join(existing)}; pass --overwrite or choose a new directory")
    files, sha, lines = _payload()
    rendered = {name: _render(title, head, tail, sha, lines, targets[name]) for name, (title, head, tail) in STAGES.items()}  # 모두 만들어 본 뒤에 쓴다
    for name, text in rendered.items():
        _write(targets[name], text)
    return {"out_dir": str(out_dir), "files": {name: {"out": str(p), "bytes": p.stat().st_size} for name, p in targets.items()},
            "modules": len(files), "source_sha256": sha}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--out", help="single .py file (JOBS style) to write, e.g. outputs/psdl1_job.py")
    where.add_argument("--out-dir", help="folder for psdl1_inspect.py, psdl1_assets.py and psdl1_train.py")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    result = build(a.out, overwrite=a.overwrite) if a.out else build_stages(a.out_dir, overwrite=a.overwrite)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
