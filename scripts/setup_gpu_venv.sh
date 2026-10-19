#!/usr/bin/env bash
# Closed-network GPU venv via the pip mirror (run from Git Bash / sh on Windows; no PowerShell).
# Creates a NEW venv, installs pinned packages with pip, then checks CUDA torch and timm.
# Existing venvs are never touched. timm builds architectures only: no weight downloads.
#
#   bash scripts/setup_gpu_venv.sh [VENV_DIR]
#
# Environment overrides:
#   PYTHON            python to build the venv with (default: python)
#   TORCH_SPEC        torch version (default 2.7.1+cu118, same as the main env); TV_SPEC for torchvision (0.22.1+cu118)
#   TORCH_INDEX_URL   index/mirror that hosts the +cu118 wheels (PyPI's Windows torch is CPU-only).
#                     Default: use the pip config as-is. If that mirror lacks +cu118, set this.
#
# Mirror address: pip reads PIP_INDEX_URL / PIP_TRUSTED_HOST / PIP_CERT from the environment, e.g.
#   export PIP_INDEX_URL=http://<mirror>/simple PIP_TRUSTED_HOST=<mirror-host>
# A new venv does not inherit a pip.conf that lives inside another venv; find the main env's with
#   <main python> -m pip config list -v
set -euo pipefail

PYTHON="${PYTHON:-python}"
VENV="${1:-.venv-gpu}"
TORCH_SPEC="${TORCH_SPEC:-2.7.1+cu118}"
TV_SPEC="${TV_SPEC:-0.22.1+cu118}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Git Bash gives /c/... paths that a Windows python.exe cannot resolve; hand it a C:/... path.
if command -v cygpath >/dev/null 2>&1; then ROOT="$(cygpath -m "$ROOT")"; fi

if [ -e "$VENV" ]; then
  echo "Venv already exists and was preserved: $VENV (choose a new path)" >&2
  exit 1
fi

"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "CPython 3.12 required, got %s" % sys.version' >&2

"$PYTHON" -m venv "$VENV"
if [ -x "$VENV/Scripts/python.exe" ]; then VPY="$VENV/Scripts/python.exe"; else VPY="$VENV/bin/python"; fi
pip() { "$VPY" -I -m pip --disable-pip-version-check "$@"; }

index_args=()
if [ -n "${TORCH_INDEX_URL:-}" ]; then index_args=(--extra-index-url "$TORCH_INDEX_URL"); fi

pip install "numpy==2.5.3" "Pillow==12.3.0" "safetensors==0.8.0" "hatchling==1.27.0"
pip install "${index_args[@]}" "torch==$TORCH_SPEC" "torchvision==$TV_SPEC"
pip install "timm==1.0.30"
# Application from this checkout, editable: a later git pull updates the installed code. Deps are pinned above.
pip install --no-deps --no-build-isolation -e "$ROOT"
pip check

"$VPY" -I - <<'PY'
import sys
import torch, torchvision, timm, safetensors, psdl1, PIL, numpy
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda build", torch.version.cuda,
      "| timm", timm.__version__, "| torchvision", torchvision.__version__)
if not torch.version.cuda:
    sys.exit("CPU-only torch was installed. Set TORCH_INDEX_URL to a mirror hosting the +cu118 wheels and use a new venv.")
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA build installed but no usable GPU; check the NVIDIA driver (nvidia-smi).")
from psdl1.model import TinyBackbone
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("psdl1 TinyBackbone forward on", dev, tuple(TinyBackbone(4).to(dev)(torch.zeros(1, 4, 64, 64, device=dev)).shape))
timm.create_model("convnextv2_atto", pretrained=False, in_chans=4, num_classes=0)  # architecture only, offline
print("timm convnextv2_atto (in_chans=4, pretrained=False) OK")
PY
echo "Done. Activate: source \"$VENV/Scripts/activate\"  (Git Bash)  or  source \"$VENV/bin/activate\""
