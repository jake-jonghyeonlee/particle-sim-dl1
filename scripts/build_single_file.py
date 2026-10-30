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
# ROUND: None = 지금까지와 같은 평면 WORK. 1 = 같은 단계를 WORK/rounds/r01 아래에서 한다. 2 이상 = 증분 라운드: WORK/rounds/r{k-1:02d} 의 결과가 있어야 하고,
# PNG_ROOT 와 METADATA 에는 새 모델만 적는다. 새 묶음을 앞 라운드 묶음에 합치고, split 은 앞 배정을 유지한 채 넓히며, 새 제품이 낀 쌍만 스크리닝한다.
ROUND = None
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
# screen_relations 추가 인자. 예: ["--groups", "/data/groups.json", "--margin", "0.05"]. --out/--manifest/--split/--partitions 는 쓰지 않는다.
# train 구획에 같은 계보 쌍이 없으면 ["--floor", "<값>"] 이 필요하다. 있으면 두 구획(train, test)에 같은 값을 쓴다.
# ROUND 2 이상에서는 --floor 를 쓸 수 없다. 항상 WORK/rounds/r01 의 floor 를 고정해 쓴다.
# ["--floor-quantile", "0.2"] 는 train 의 floor 를 같은 계보 유사도의 분위수로 계산한다(--floor 와 함께 쓸 수 없고 ROUND 2 이상에서는 쓰지 않는다). test 는 계산된 train floor 를 쓴다.
SCREEN_ARGS = []
LOG_FILE = WORK + "/logs/assets.log"    # stdout/stderr(진행 줄)를 이 파일에도 남긴다. 필요 없으면 None
WORK_DIR = None                         # 패키지를 풀 폴더. None이면 이 파일 옆, 쓸 수 없으면 임시 폴더
# ============================== EDIT ABOVE ==============================================
'''

HEAD_TRAIN = r'''# ============================== EDIT BELOW: 이 블록만 고치세요 ==============================
# 3단계: (선택) 사전학습 -> 학습 -> (선택) 평가. psdl1_assets.py 가 만든 WORK 를 입력으로 쓴다.
# --manifest/--split/--relations/--out/--weights/--resume/--ckpt 는 이 파일이 채운다. 아래 인자 리스트에는 나머지 옵션만 적는다.
WORK = "/data/work"                     # psdl1_assets.py 와 같은 폴더
PRETRAIN = False                        # True 면 backbone 사전학습을 먼저 하고 그 가중치로 학습한다
PRETRAIN_ARGS = ["--backbone", "tiny", "--device", "cuda", "--batch-size", "2", "--epochs", "100"]
TRAIN_ARGS = ["--backbone", "tiny", "--device", "cuda", "--batch-size", "2", "--epochs", "100"]
EVALUATE = True                         # split 에 test 제품이 있을 때만 평가한다
EVALUATE_ARGS = ["--topk", "5"]         # TRAIN_ARGS 에 --allow-grid-check-failure 를 넣었다면 여기에도 넣는다
EVALUATE_ONLY = False                   # True 면 학습하지 않고 WORK/train/dl1.pt 를 평가만 한다(dl1.pt 가 있어야 한다)
# ROUND 는 psdl1_assets.py 와 같은 값으로 맞춘다. None = 평면 WORK, 1 = WORK/rounds/r01, 2 이상 = 앞 라운드 dl1.pt 에서 이어 배우는 warm start(PRETRAIN 은 False).
# 아래 세 값은 ROUND 2 이상에서만 쓴다. ROUND_EPOCHS/WARM_START_LR 이 None 이면 TRAIN_ARGS 의 --epochs/--lr 를 그대로 쓴다.
ROUND = None
ROUND_EPOCHS = 30                       # 증분 라운드의 epoch 수(TRAIN_ARGS 의 --epochs 를 덮어쓴다)
WARM_START_LR = 1e-4                    # 증분 라운드의 학습률(TRAIN_ARGS 의 --lr 를 덮어쓴다)
NEW_EDGE_REPEAT = 1                     # 앞 라운드가 학습하지 않은 negative 쌍을 epoch 마다 이 횟수만큼 일정에 넣는다
LOG_FILE = WORK + "/logs/train.log"    # stdout/stderr(진행 줄)를 이 파일에도 남긴다. 필요 없으면 None
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

ALLOWED = ("train", "pretrain", "screen_relations", "evaluate", "embed_plot", "search", "import_png", "data", "relations_draft", "merge_bundles", "compare_rounds")
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


def _aside(path, tag):
    """<path>.<tag>-<시각> 이름을 돌려준다. 이미 있으면 -2, -3 … 을 붙여 기존 것과 겹치지 않게 한다."""
    base = path.name + "." + tag + "-" + time.strftime("%Y%m%d-%H%M%S")
    target, n = path.with_name(base), 1
    while os.path.lexists(target):
        n += 1
        target = path.with_name(f"{base}-{n}")
    return target


def _why(exc):
    return f"exit code {exc.code!r}" if isinstance(exc, SystemExit) else f"{type(exc).__name__}: {exc}"


def _call_keeping_failed(name, args, out):
    """_call 이 실패하면 원인을 먼저 로그에 남기고, 출력 폴더를 <out>.failed-<시각> 으로 옮겨 증거를 보존한다. 다음 실행은 새로 만든다."""
    try:
        _call(name, args)
    except BaseException as exc:
        _say(f"{name} failed: {_why(exc)}")
        out = Path(out)
        if out.exists():
            try:
                moved = _aside(out, "failed")
                os.replace(out, moved)
            except OSError as move_error:
                _say(f"이동 실패: {out} ({move_error}), 손으로 옮긴 뒤 다시 실행")
            else:
                _say(f"{name} failed; kept its output at {moved}")
        raise


def _check_reserved(label, args, flags):
    """실행기가 채우는 인자가 EDIT 인자 리스트에 있으면 시작 전에 멈춘다. --flag=value 와 argparse 약어(--manif 등)도 막는다."""
    for arg in args:
        name = str(arg).split("=", 1)[0]
        if not name.startswith("--") or len(name) < 3:
            continue
        for flag in flags:
            if flag.startswith(name):
                what = "" if name == flag else "an abbreviation of "
                if flag == "--config":
                    _stop(f"{label} must not contain {name}: it is {what}--config, whose JSON can set resume/weights/out behind this file's back. "
                          f"Write the options in {label} itself, or use a JOBS-style file (--out) for psdl1.train --config")
                _stop(f"{label} must not contain {name}: it is {what}{flag}, which this file sets itself")


def _skip(step, path):
    _say(f"skip {step} (exists: {path})")


def _stop(message):
    _say("STOP: " + message)
    raise SystemExit(2)


def _check_round(value):
    """ROUND 는 None 이거나 1 이상의 정수여야 한다."""
    if value is not None and (type(value) is not int or value < 1):
        _stop(f"ROUND must be None or an integer >= 1, got {value!r}")


def _round_dir(work, number):
    """라운드 number 의 폴더: WORK/rounds/rNN."""
    return Path(work) / "rounds" / f"r{number:02d}"


def _require(paths, what):
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        _stop(what + ": missing " + ", ".join(missing))


def _safe_streams():
    """콘솔 인코딩(예: euc_kr)이 표현하지 못하는 문자(예: en dash)가 있어도 멈추지 않고 대체 표기로 출력한다."""
    for stream in (sys.__stdout__, sys.__stderr__):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass


def _run_with_log(entry):
    _safe_streams()
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
        _call_keeping_failed("import_png", ["--root", PNG_ROOT, "--out", out, "--inspect-grids", *grids, *(["--no-preview"] if NO_PREVIEW else [])], out)
    preview = out / "preview.html"
    _say(f"check: {preview if preview.is_file() else out}")
    _say("pick the grid from the inspection result, then set GRID in psdl1_assets.py and run it")


def _main():
    _run_with_log(stage)


if __name__ == "__main__":
    _main()
'''

TAIL_ASSETS = r'''
def _write_json_once(path, obj, what, replace=False, away="the dependent WORK/export folder"):
    """파일이 없으면 쓴다. 같으면 재사용한다. 다르면 replace(export 전)일 때만 덮어쓰고 로그에 남기고, 아니면 멈춘다."""
    new = json.loads(json.dumps(obj))
    replaced = False
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            old = object()
        if old == new:
            _skip(what, path)
            return
        if not replace:
            _stop(f"{path} differs from {what} in the EDIT block; not overwriting. Restore the EDIT value, or move {path} "
                  f"and {away} away to start over.")
        replaced = True
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    _say(f"{'replaced (no export yet)' if replaced else 'wrote'} {what}: {path}")


def _floor_arg(args):
    """사용자가 준 --floor 값(없으면 None). argparse 약어(--flo 등)는 인식하지 않고 전체 이름을 쓰라며 멈춘다."""
    for i, arg in enumerate(args):
        name = str(arg).split("=", 1)[0]
        if name.startswith("--") and len(name) > 2 and name != "--floor" and "--floor".startswith(name):
            _stop(f"SCREEN_ARGS has {name}: write the full name --floor")
        if name == "--floor":
            if "=" in str(arg):
                return str(arg).split("=", 1)[1]
            if i + 1 < len(args):
                return str(args[i + 1])
    return None


def _floor_quantile_arg(args):
    """SCREEN_ARGS 에 --floor-quantile 이 있는가. 약어(--floor-q 등)와 모호한 --floor- 는 전체 이름을 쓰라며 멈춘다."""
    found = False
    for arg in args:
        name = str(arg).split("=", 1)[0]
        if name == "--floor-quantile":
            found = True
        elif name.startswith("--floor-") and "--floor-quantile".startswith(name):
            _stop(f"SCREEN_ARGS has {name}: write the full name --floor-quantile (or --floor-json)")
    return found


def _without_floor_quantile(args):
    """--floor-quantile Q / --floor-quantile=Q 를 뺀 인자 목록. test 구획은 train 에서 계산한 floor 를 쓰므로 quantile 을 받지 않는다."""
    kept, skip = [], False
    for arg in args:
        if skip:
            skip = False
            continue
        name = str(arg).split("=", 1)[0]
        if name == "--floor-quantile":
            skip = "=" not in str(arg)
            continue
        kept.append(arg)
    return kept


def _user_floor_source(args):
    """사용자가 기준(floor)을 직접 정했는가: --floor 이거나 --floor-json(약어 포함)이면 True. 이때 실행기는 test 구획에 train 기준을 따로 넣지 않는다.
    --floor-quantile 은 기준값이 아니라 train 에서 floor 를 계산하는 규칙이라 False 다."""
    _floor_quantile_arg(args)
    if _floor_arg(args) is not None:
        return True
    for arg in args:
        name = str(arg).split("=", 1)[0]
        if name.startswith("--floor-") and "--floor-json".startswith(name):
            return True
    return False


def _screen(manifest, split_path, part, out_dir, extra, drop_quantile=False):
    """한 구획을 스크리닝한다. 세 산출물이 모두 있으면 건너뛰고, 일부만 있으면 멈춘다. 요약(summary) dict 를 돌려준다. drop_quantile 이면 --floor-quantile 을 빼고 넘긴다."""
    csv = out_dir / "relations.csv"
    outputs = [csv, csv.with_name(csv.name + ".review.html"), csv.with_name(csv.name + ".summary.json")]
    found = [p for p in outputs + [csv.with_name(csv.name + ".scores.csv"), csv.with_name(csv.name + ".bindings.csv")] if p.exists()]
    if all(p.exists() for p in outputs):
        _skip(f"screen_relations {part}", csv)
    elif found:
        _stop("screen_relations left partial output: " + ", ".join(str(p) for p in found) + f". Move {out_dir} away and run again")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        _call("screen_relations", ["--manifest", manifest, "--split", split_path, "--out", csv, "--partitions", part, *(_without_floor_quantile(SCREEN_ARGS) if drop_quantile else SCREEN_ARGS), *extra])
    try:
        summary = json.loads(outputs[2].read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:
        _stop(f"부분 출력: {out_dir} 를 옮긴 뒤 다시 실행 ({outputs[2].name}: {exc})")
    return summary


def _rows(path):
    import csv as csv_module
    with open(path, newline="", encoding="utf-8") as handle:
        return [row for row in csv_module.reader(handle) if row]


def _merge_rows(parts):
    """구획별 relations.csv 의 행을 순서대로 모은다. 한 쌍이 둘 이상의 구획에 나오면 멈춘다."""
    seen, rows = set(), []
    for part in parts:
        for row in _rows(part):
            key = frozenset(row[:2])
            if key in seen:
                _stop(f"pair {row[:2]} appears in more than one partition's relations; not merging")
            seen.add(key)
            rows.append(row)
    return rows


def _write_rows(path, rows):
    import csv as csv_module
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        csv_module.writer(handle, lineterminator="\n").writerows(rows)
    os.replace(tmp, path)


def _merge_incremental(base, deltas, product_ids):
    """앞 라운드의 합친 relations.csv(base) 바이트 + 구획별 delta 행. 전체를 검사한다. (바이트, 행 수)를 돌려준다.
    base 바이트는 그대로 두고 마지막 개행만 보장한다. 같은 쌍이 두 번 나오거나 모순·미등록 ID 가 있으면 멈춘다."""
    import csv as csv_module
    import io
    from psdl1.losses import validate_relations
    raw = Path(base).read_bytes()
    if raw and not raw.endswith(b"\n"):
        raw += b"\n"
    base_rows = _rows(base)
    rows = list(base_rows)
    for delta in deltas:
        rows += _rows(delta)
    seen, similar, different = set(), set(), set()
    for row in rows:
        if len(row) != 3 or row[2].strip() not in ("similar", "different"):
            _stop(f"relation row {row} is not product_a,product_b,similar|different; not merging")
        key = frozenset(v.strip() for v in row[:2])
        if key in seen:
            _stop(f"pair {sorted(key)} appears more than once in the previous round's relations and the new delta rows; not merging")
        seen.add(key)
        (similar if row[2].strip() == "similar" else different).add(key)
    try:
        validate_relations(similar, different, product_ids=product_ids)
    except ValueError as exc:
        _stop(f"merged relations are invalid ({exc}); not merging")
    buffer = io.StringIO()
    csv_module.writer(buffer, lineterminator="\n").writerows(rows[len(base_rows):])
    return raw + buffer.getvalue().encode("utf-8"), len(rows)


def _write_bytes(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _split(manifest, split_path, previous=None):
    from psdl1.data import _iter_manifest_products
    from psdl1.screen_relations import CHECK_GRID
    from psdl1.split import PARTITIONS, extend_split, lineage_split, validate_split
    products = [SimpleNamespace(product_id=p.product_id, lineage_id=p.lineage_id) for p in _iter_manifest_products(manifest, grid=CHECK_GRID)]
    if split_path.is_file():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        validate_split(products, split)
        _skip("split", split_path)
    else:
        if previous is None:
            split = lineage_split(products, seed=SPLIT_SEED, validation_fraction=VALIDATION_FRACTION, test_fraction=TEST_FRACTION)
        else:
            split = extend_split(json.loads(previous.read_text(encoding="utf-8")), products, seed=SPLIT_SEED,
                                 validation_fraction=VALIDATION_FRACTION, test_fraction=TEST_FRACTION)
        tmp = split_path.with_name(split_path.name + ".tmp")
        tmp.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, split_path)
        _say(f"wrote split: {split_path}")
    lineage = {p.product_id: p.lineage_id for p in products}
    for part in PARTITIONS:
        _say(f"split {part}: {len(split[part])} products, {len({lineage[pid] for pid in split[part]})} lineages")
    if previous is not None:
        before = json.loads(previous.read_text(encoding="utf-8"))
        old = {lineage[pid] for part in PARTITIONS for pid in before[part] if pid in lineage}
        _say(f"split extends {previous}: {len(set(lineage.values()) - old)} new lineages, {sum(len(split[p]) for p in PARTITIONS) - sum(len(before[p]) for p in PARTITIONS)} new products")
    return split


def _check_screen_args():
    """SCREEN_ARGS 의 예약 인자를 시작 전에 검사한다. ROUND 2 이상은 --floor 를 받지 않고(r01 floor 를 고정한다) 증분 인자도 예약한다."""
    reserved = ("--out", "--manifest", "--split", "--partitions", "--bindings-out")
    if ROUND is not None and ROUND >= 2:
        if _floor_quantile_arg(SCREEN_ARGS):
            _stop("SCREEN_ARGS must not contain --floor-quantile: ROUND >= 2 always uses the round-1 floor (--floor-json from WORK/rounds/r01), "
                  "so a new quantile would change the floor between rounds. Remove it")
        for arg in SCREEN_ARGS:
            name = str(arg).split("=", 1)[0]
            if name.startswith("--") and len(name) > 2 and "--floor".startswith(name):
                _stop(f"SCREEN_ARGS must not contain {name}: ROUND >= 2 always uses the round-1 floor (--floor-json from WORK/rounds/r01). Remove it")
        _check_reserved("SCREEN_ARGS", SCREEN_ARGS, reserved + ("--base-split", "--base-relations", "--floor-json"))
    else:
        _check_reserved("SCREEN_ARGS", SCREEN_ARGS, reserved)
        _floor_arg(SCREEN_ARGS)  # --floor 약어는 시작 전에 멈춘다
        if _floor_quantile_arg(SCREEN_ARGS) and _user_floor_source(SCREEN_ARGS):
            _stop("SCREEN_ARGS has --floor-quantile together with --floor or --floor-json: the quantile computes the floor, so use only one")


def _merge_step(prev, export, export_new):
    """증분 라운드: 앞 라운드 묶음에 새 묶음을 합친다(검증은 매번). 입력이 다른 병합본이나 부분 출력은 멈춘다."""
    from psdl1.merge_bundles import merge_inputs_match
    merged_base = (prev / "export" / "merge-report.json").is_file()
    base_round = None if merged_base else ROUND - 1
    manifest = export / "manifest.json"
    if merged_base:
        _call("merge_bundles", ["--verify", prev / "export"])
    else:
        _call("import_png", ["--verify-bundle", prev / "export"])
    existed = manifest.is_file()
    if existed and not (export / "bundle-integrity.json").is_file():
        _stop(f"{export} has manifest.json but no bundle-integrity.json (killed or partial merge). Move the folder away and run again")
    if existed:
        reason = merge_inputs_match(export, prev / "export", export_new, ROUND, base_round)
        if reason:
            _stop(f"{export} was not made from the current {prev / 'export'} and {export_new} ({reason}); not reusing it. "
                  f"Restore the inputs, or move {export} away and run again")
    elif export.exists():
        _stop(f"{export} exists without manifest.json (killed or partial merge). Read its *.failed.json files, fix the cause, move the folder away, and run again")
    else:
        _call_keeping_failed("merge_bundles", ["--base", prev / "export", "--add", export_new, "--out", export, "--round", ROUND,
                                               *([] if base_round is None else ["--base-round", base_round])], export)
    _call("merge_bundles", ["--verify", export])
    if existed:
        _skip("merge", manifest)


def stage():
    _check_round(ROUND)
    _check_screen_args()
    incremental = ROUND is not None and ROUND >= 2
    top = Path(WORK)
    work = top if ROUND is None else _round_dir(top, ROUND)
    prev, first = (_round_dir(top, ROUND - 1), _round_dir(top, 1)) if incremental else (None, None)
    if incremental:
        _require([prev / "export" / "manifest.json", prev / "export" / "bundle-integrity.json", prev / "inputs" / "split.json",
                  prev / "relations" / "relations.csv", first / "relations" / "train" / "relations.csv.summary.json"],
                 f"ROUND = {ROUND} needs the finished round {ROUND - 1} (and round 1's floor)")
    _boot()
    work.mkdir(parents=True, exist_ok=True)
    inputs, export, reports, relations = work / "inputs", work / "export", work / "reports", work / "relations"
    export_new = work / "export-new"
    meta_path, order_path = inputs / "metadata.json", inputs / "stack-order.json"
    manifest, split_path, grid_report = export / "manifest.json", inputs / "split.json", reports / "grid-report.json"
    csv = relations / "relations.csv"

    # a. 입력 파일. export 가 끝나기 전에는 EDIT 값으로 바꿔 쓰고, 끝난 뒤에는 다르면 멈춘다. 증분 라운드는 export-new 가 끝난 뒤부터 고정한다.
    exported = (export_new / "manifest.json").is_file() if incremental else manifest.is_file()
    away = f"the export-new folder {export_new} (and the merged {export} if it exists)" if incremental else "the dependent WORK/export folder"
    _write_json_once(meta_path, METADATA, "METADATA", replace=not exported, away=away)
    if STACK_ORDER is not None:
        _write_json_once(order_path, STACK_ORDER, "STACK_ORDER", replace=not exported, away=away)
    elif order_path.exists():
        if exported:
            _stop(f"STACK_ORDER is None but {order_path} exists; set STACK_ORDER to its content or move the file away")
        moved = _aside(order_path, "removed")
        os.replace(order_path, moved)
        _say(f"moved {order_path} to {moved} (STACK_ORDER is None and there is no export yet)")

    # b. export. 실패하면 출력 폴더를 .failed-* 로 보존한다. 증분 라운드는 새 모델만 export-new 로 내보낸 뒤 앞 묶음과 합친다.
    out = export_new if incremental else export
    if out.joinpath("manifest.json").is_file():
        _skip("export-new" if incremental else "export", out / "manifest.json")
    elif out.exists():
        _stop(f"{out} exists without manifest.json (killed or partial export). Read its diagnostic report, fix the cause, move the folder away, and run again")
    else:
        _call_keeping_failed("import_png", ["--root", PNG_ROOT, "--out", out, "--export", "--metadata", meta_path, "--candidate-grid", GRID,
                                            *(["--stack-order", order_path] if STACK_ORDER is not None else []), *(["--no-preview"] if NO_PREVIEW else [])], out)

    # c. 묶음 검증(매번). 증분 라운드는 합친 묶음까지 검증한다.
    _call("import_png", ["--verify-bundle", out])
    if incremental:
        _merge_step(prev, export, export_new)

    # d. 격자 보고서
    reports.mkdir(parents=True, exist_ok=True)
    if grid_report.is_file():
        try:
            json.loads(grid_report.read_text(encoding="utf-8"))
        except ValueError as exc:
            _stop(f"{grid_report} is not valid JSON ({exc}); move it away and run again")
        _skip("grid report", grid_report)
    else:
        _call("data", [manifest, "--report", grid_report])

    # e. 자동 split
    inputs.mkdir(parents=True, exist_ok=True)
    split = _split(manifest, split_path, prev / "inputs" / "split.json" if incremental else None)

    # f. relations. held-out 이 train 기준값(floor)에 섞이지 않도록 train 과 test 를 따로 만들어 합친다.
    review = []
    if csv.is_file():
        _skip("relations", csv)
    elif incremental:
        # 새 제품이 낀 쌍만 스크리닝한다. 기준(floor)은 r01 train 요약에서 고정한다. 앞 라운드의 합친 파일(사용자 수정 포함)이 바이트 그대로 앞부분이 된다.
        train_dir, test_dir = relations / "train", relations / "test"
        carried = ["--base-split", prev / "inputs" / "split.json", "--base-relations", prev / "relations" / "relations.csv",
                   "--floor-json", first / "relations" / "train" / "relations.csv.summary.json"]
        _screen(manifest, split_path, "train", train_dir, carried)
        deltas = [train_dir / "relations.csv"]
        review.append(train_dir / "relations.csv.review.html")
        if split.get("test"):
            _screen(manifest, split_path, "test", test_dir, carried)
            deltas.append(test_dir / "relations.csv")
            review.append(test_dir / "relations.csv.review.html")
        else:
            _say("no test products in the split; relations come from train only")
        data, count = _merge_incremental(prev / "relations" / "relations.csv", deltas, [pid for part in ("train", "validation", "test") for pid in split[part]])
        _write_bytes(csv, data)
        _say(f"wrote merged relations: {csv} ({count} pairs; the previous round's file is its first bytes)")
    else:
        train_dir, test_dir = relations / "train", relations / "test"
        train_summary = _screen(manifest, split_path, "train", train_dir, [])
        parts = [train_dir / "relations.csv"]
        review.append(train_dir / "relations.csv.review.html")
        if split.get("test"):
            floor = [] if _user_floor_source(SCREEN_ARGS) else ["--floor", repr(train_summary["floor"]["global"])]
            _screen(manifest, split_path, "test", test_dir, floor, drop_quantile=True)
            parts.append(test_dir / "relations.csv")
            review.append(test_dir / "relations.csv.review.html")
        else:
            _say("no test products in the split; relations come from train only")
        rows = _merge_rows(parts)
        _write_rows(csv, rows)
        _say(f"wrote merged relations: {csv} ({len(rows)} pairs)")
    review = review or [p for p in (relations / "train" / "relations.csv.review.html", relations / "test" / "relations.csv.review.html")]

    # g. 사람이 확인할 파일
    _say("check these files before training:" + (" (the review pages list the new pairs only)" if incremental else ""))
    for path in (export / "preview.html", export_new / "preview.html", export / "merge-report.json", split_path, *review, grid_report):
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


def _evaluation_problem(evaluation, checkpoint):
    """evaluation.json 이 현재 dl1.pt 의 평가면 None, 아니면 이유를 돌려준다."""
    from psdl1.checkpoint import read_checkpoint
    try:
        recorded = json.loads(evaluation.read_text(encoding="utf-8"))["checkpoint_weights_hash"]
        current = read_checkpoint(checkpoint)["weights_hash"]
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None if recorded == current else "it was made from a different checkpoint"


def _target_epochs(args):
    """TRAIN_ARGS 의 --epochs 값(약어 --epoch 도 허용). 없으면 psdl1.train 기본값 100. 읽을 수 없으면 None."""
    value = 100
    for i, arg in enumerate(args):
        name = str(arg).split("=", 1)[0]
        if name.startswith("--ep") and "--epochs".startswith(name):
            raw = str(arg).split("=", 1)[1] if "=" in str(arg) else (str(args[i + 1]) if i + 1 < len(args) else "")
            try:
                value = int(raw)
            except ValueError:
                return None
    return value


def _report_epochs(checkpoint, args=None):
    """평가 전에 체크포인트가 몇 epoch 까지 학습됐는지 남긴다. 목표(TRAIN_ARGS --epochs, 증분 라운드는 덮어쓴 값)보다 작으면 경고만 하고 멈추지 않는다."""
    from psdl1.checkpoint import read_checkpoint
    try:
        done = int(read_checkpoint(checkpoint)["epoch"])
    except Exception as exc:
        _say(f"cannot read the epoch of {checkpoint}: {type(exc).__name__}: {exc}")
        return
    target = _target_epochs(TRAIN_ARGS if args is None else args)
    _say(f"checkpoint {checkpoint} was trained for {done} epochs" + (f" ({'TRAIN_ARGS ' if args is None else 'target '}--epochs: {target})" if target is not None else ""))
    if target is not None and done < target:
        _say(f"WARNING: the checkpoint has {done} of {target} epochs (training may have stopped early); evaluating it anyway")


def _previous_checkpoint(path):
    """증분 라운드의 부모 dl1.pt 를 열어 보고, 기록된 round 가 있으면 ROUND-1 과 같은지 본다."""
    from psdl1.checkpoint import read_checkpoint
    try:
        recorded = read_checkpoint(path)["metadata"].get("round")
    except Exception as exc:
        _stop(f"cannot read the previous round's checkpoint {path}: {type(exc).__name__}: {exc}")
    if recorded is not None and recorded != ROUND - 1:
        _stop(f"{path} was trained as round {recorded}, but ROUND = {ROUND} needs round {ROUND - 1}")


def _compare_rounds(previous, evaluation, comparison):
    """이전 라운드 평가가 있으면 쿼리 단위로 비교한다. 이미 있는 결과는 두 평가 파일이 같을 때만 재사용하고, 다르면 덮어쓰지 않고 멈춘다."""
    from psdl1.compare_rounds import comparison_inputs_match
    if not previous.is_file():
        _say(f"skip compare (no previous evaluation: {previous})")
    elif not evaluation.is_file():
        _say(f"skip compare (no evaluation in this round: {evaluation})")
    elif comparison.exists():
        reason = comparison_inputs_match(comparison, previous, evaluation)
        if reason:
            _stop(f"{comparison} does not match the current evaluations ({reason}); not overwriting. Move it away and run again")
        _skip("compare", comparison)
    else:
        _call("compare_rounds", ["--previous", previous, "--current", evaluation, "--out", comparison])


def stage():
    _check_round(ROUND)
    incremental = ROUND is not None and ROUND >= 2
    top = Path(WORK)
    work = top if ROUND is None else _round_dir(top, ROUND)
    prev = _round_dir(top, ROUND - 1) if incremental else None
    manifest, split_path, csv = work / "export" / "manifest.json", work / "inputs" / "split.json", work / "relations" / "relations.csv"
    missing = [str(p) for p in (manifest, split_path, csv) if not p.is_file()]
    if missing:
        _stop("missing " + ", ".join(missing) + ". Run psdl1_assets.py first (same WORK" + (", same ROUND)" if ROUND is not None else ")"))
    reserved = ("--manifest", "--split", "--relations", "--out", "--weights", "--resume", "--ckpt")
    warm = ("--init-from", "--round", "--new-edge-repeat") if incremental else ("--round",) if ROUND == 1 else ()  # 이 파일이 채운다
    _check_reserved("PRETRAIN_ARGS", PRETRAIN_ARGS, reserved + warm)
    _check_reserved("TRAIN_ARGS", TRAIN_ARGS, reserved + warm + ("--config",))  # psdl1.train --config JSON 은 resume/weights 를 줄 수 있다
    _check_reserved("EVALUATE_ARGS", EVALUATE_ARGS, reserved + warm)
    pdir, tdir = work / "pretrain", work / "train"
    checkpoint, evaluation = tdir / "dl1.pt", tdir / "evaluation.json"
    if incremental:
        if PRETRAIN:
            _stop("warm-start replaces pretraining in round >= 2; set PRETRAIN = False")
        if not EVALUATE_ONLY and not (prev / "train" / "dl1.pt").is_file():
            _stop(f"ROUND = {ROUND} needs the previous round's checkpoint {prev / 'train' / 'dl1.pt'}; train round {ROUND - 1} first")
    if EVALUATE_ONLY:
        if not checkpoint.is_file():
            _stop(f"EVALUATE_ONLY needs {checkpoint}; train first (EVALUATE_ONLY = False)")
    elif checkpoint.exists():
        _stop(f"{checkpoint} already exists; psdl1.train would overwrite it, so this file stops. To train again, move {tdir} to another name. "
              f"To only evaluate this checkpoint, set EVALUATE_ONLY = True"
              + (f". An interrupted warm-start run is resumed from a JOBS-style file: train --resume {checkpoint} with the same --epochs/--lr/--new-edge-repeat, "
                 f"without --init-from" if incremental else ""))
    _boot()
    train_args = [*TRAIN_ARGS]
    if ROUND == 1:
        train_args += ["--round", 1]
    if incremental:
        if not EVALUATE_ONLY:
            _previous_checkpoint(prev / "train" / "dl1.pt")
        override = (["--epochs", ROUND_EPOCHS] if ROUND_EPOCHS is not None else []) + (["--lr", WARM_START_LR] if WARM_START_LR is not None else [])
        train_args += ["--init-from", prev / "train" / "dl1.pt", "--round", ROUND, "--new-edge-repeat", NEW_EDGE_REPEAT, *override]
        if override and not EVALUATE_ONLY:
            _say("round override: " + " ".join(str(v) for v in override) + " (replaces the --epochs/--lr in TRAIN_ARGS)")
    if not EVALUATE_ONLY:
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
        _call("train", ["--manifest", manifest, "--split", split_path, "--relations", csv, "--out", tdir, *weights, *train_args])
    if EVALUATE or EVALUATE_ONLY:
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if not split.get("test"):
            _say("skip evaluate (no test products in the split; evaluate uses only the test partition)")
        else:
            _report_epochs(checkpoint, train_args if incremental else None)
            if evaluation.exists():
                problem = _evaluation_problem(evaluation, checkpoint)
                if problem:
                    _stop(f"{evaluation} does not match {checkpoint} ({problem}); not overwriting. Move it away and run again")
                _skip("evaluate", evaluation)
            else:
                _call("evaluate", ["--ckpt", checkpoint, "--manifest", manifest, "--split", split_path, "--relations", csv, "--out", evaluation, *EVALUATE_ARGS])
    if incremental:
        problem = _evaluation_problem(evaluation, checkpoint) if evaluation.exists() else None
        if not (EVALUATE or EVALUATE_ONLY):
            _say("skip compare (evaluation was not requested; EVALUATE = False)")
        elif problem:
            _say(f"skip compare ({evaluation} does not belong to {checkpoint}: {problem})")
        else:
            _compare_rounds(prev / "train" / "evaluation.json", evaluation, tdir / "round-comparison.json")
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
