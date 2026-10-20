#!/usr/bin/env python3
"""src/psdl1/*.py 전체를 .py 파일 하나로 묶는다. 명령줄 옵션을 넘길 수 없고 .py만 올릴 수 있는 서버용.

만들어진 파일 맨 위의 EDIT 블록(JOBS, LOG_FILE, WORK_DIR)만 고치면 된다. 실행하면 패키지를 실제 폴더에 풀고(바이트 그대로라
code_hash가 저장소와 같다) psdl1.<작업>.main(인자)을 순서대로 호출한다. 서버에는 torch, numpy, Pillow(timm은 선택)만 있으면 된다."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]

TEMPLATE = '''#!/usr/bin/env python3
"""DL1 단일 파일 실행기 (자동 생성: scripts/build_single_file.py). 소스 해시 @@SHA@@.
서버 wrapper가 명령줄 인자를 넘기지 못할 때 쓴다. 아래 EDIT 블록만 고치고 이 파일을 그대로 실행한다."""
# ============================== EDIT BELOW: 이 블록만 고치세요 ==============================
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


def run_jobs(jobs):
    for name, _ in jobs:
        if name not in ALLOWED:
            raise ValueError(f"unknown job {name!r}; choose from {ALLOWED}")
    root = _extract()
    sys.path.insert(0, str(root))
    import psdl1
    if not str(Path(psdl1.__file__).resolve()).startswith(str(root.resolve())):
        raise RuntimeError(f"another psdl1 shadows the bundled one: {psdl1.__file__}")
    stamp = lambda: time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        import torch
        device_info = f"torch {torch.__version__} cuda {torch.version.cuda} available {torch.cuda.is_available()}" + (f" gpu {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "")
    except Exception as exc:  # 로그용 정보일 뿐이다
        device_info = f"torch unavailable: {exc}"
    print(f"[psdl1_job] {stamp()} python {sys.version.split()[0]} | {device_info} | psdl1 from {psdl1.__file__}", file=sys.stderr, flush=True)
    for name, args in jobs:
        argv = [str(a) for a in args]
        print(f"[psdl1_job] {stamp()} start {name} {' '.join(argv)}", file=sys.stderr, flush=True)
        started = time.time()
        main = importlib.import_module(f"psdl1.{name}").main
        sys.argv = [f"psdl1.{name}", *argv]
        try:
            main(argv) if "argv" in inspect.signature(main).parameters else main()
        except SystemExit as exc:
            if exc.code not in (None, 0):
                print(f"[psdl1_job] {stamp()} {name} exited with {exc.code!r}", file=sys.stderr, flush=True)
                raise
        print(f"[psdl1_job] {stamp()} done {name} in {time.time() - started:.0f}s", file=sys.stderr, flush=True)


def _main():
    log = None
    if LOG_FILE:
        Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        log = open(LOG_FILE, "a", buffering=1, encoding="utf-8")
        sys.stdout, sys.stderr = _Tee(sys.__stdout__, log), _Tee(sys.__stderr__, log)
    try:
        run_jobs(JOBS)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if log is not None:
            log.flush()


if __name__ == "__main__":
    _main()
'''


def build(out, *, overwrite=False):
    out = Path(out)
    if out.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {out}; pass --overwrite or choose a new path")
    sources = sorted(p for p in (ROOT / "src/psdl1").glob("*.py"))
    if not sources:
        raise ValueError("no source files found under src/psdl1")
    files = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "data": base64.b64encode(p.read_bytes()).decode()} for p in sources}
    packed = base64.b64encode(zlib.compress(json.dumps(files, sort_keys=True).encode(), 9)).decode()
    sha = hashlib.sha256(json.dumps({n: e["sha256"] for n, e in files.items()}, sort_keys=True).encode()).hexdigest()
    lines = "\n".join(f'    "{packed[i:i + 100]}"' for i in range(0, len(packed), 100))
    text = TEMPLATE.replace("@@SHA@@", sha).replace("@@PAYLOAD@@", lines)
    compile(text, str(out), "exec")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    return {"out": str(out), "bytes": out.stat().st_size, "modules": len(files), "source_sha256": sha}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="single .py file to write, e.g. outputs/psdl1_job.py")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(build(a.out, overwrite=a.overwrite), indent=2))


if __name__ == "__main__":
    main()
