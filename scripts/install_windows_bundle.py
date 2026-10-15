#!/usr/bin/env python3
"""CMD/installed-Python offline installer; no shell, PowerShell, uv, or downloads."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import uuid


def check_runtime():
    if (sys.platform != 'win32' or platform.python_implementation() != 'CPython'
            or sys.version_info[:2] != (3, 12) or struct.calcsize('P') != 8
            or platform.machine().upper() != 'AMD64'):
        raise ValueError('Preinstalled CPython 3.12 Windows x64 required (not x86/ARM64).')


def offline_environment():
    env = dict(os.environ)
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1',
               PIP_NO_CACHE_DIR='1', PYTHONNOUSERSITE='1', PYTHONIOENCODING='utf-8',
               PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1')
    return env


def run(label, command, *, cwd, env):
    print(f'[{label}] {subprocess.list2cmdline(list(map(str, command)))}', flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)


def install(bundle, venv, *, reuse=False, smoke=False, smoke_out=None):
    check_runtime()
    bundle, venv = Path(bundle).resolve(), Path(venv).resolve()
    # The default .venv-cmd is the one allowed mutable directory in the bundle.
    if venv == bundle or (venv.is_relative_to(bundle) and venv != bundle / '.venv-cmd'):
        raise ValueError('Use bundle/.venv-cmd, or a venv path outside the bundle.')
    if venv.exists() and not reuse:
        raise ValueError('Venv already exists and was preserved. Use --reuse-venv for exact-match checks without installation.')
    if reuse and not (venv / 'pyvenv.cfg').is_file():
        raise ValueError('--reuse-venv requires an existing isolated venv.')
    if smoke_out is not None:
        smoke_out = Path(smoke_out).resolve()
        if smoke_out.is_relative_to(bundle) or smoke_out.exists():
            raise ValueError('Smoke output must be a new directory outside the bundle.')
    env = offline_environment()
    execute = lambda label, *command: run(label, command, cwd=bundle, env=env)
    python = venv / 'Scripts/python.exe'
    if not reuse:
        execute('create venv', sys.executable, '-I', '-m', 'venv', venv)
    execute('Python version/architecture', python, '-I', bundle / 'install_windows_bundle.py', '--check-runtime')
    execute('bundle hashes/inventory/target/dependencies', python, '-I', bundle / 'verify_windows_bundle.py', bundle)
    if not reuse:
        execute('local wheels only', python, '-I', '-m', 'pip', '--isolated', '--disable-pip-version-check',
                'install', '--no-index', '--no-cache-dir', '--only-binary=:all:', '--require-hashes',
                '--find-links=wheelhouse', '-r', 'requirements-windows.txt')
    execute('installed versions/isolation/DL1 source', python, '-I', bundle / 'verify_windows_bundle.py',
            bundle, '--installed-prefix', venv)
    execute('pip check', python, '-I', '-m', 'pip', '--isolated', '--disable-pip-version-check', 'check')
    execute('DL1 CPU imports', python, '-I', '-c',
            "import psdl1,numpy,torch,PIL; import psdl1.train,psdl1.search; assert torch.version.cuda is None; print('DL1 CPU imports OK')")
    execute('DL1 help', python, '-I', '-m', 'psdl1.import_png', '--help')
    print('Offline installation/imports/help passed. Numerical smoke and real data are separate checks.', flush=True)
    if smoke:
        if smoke_out is None:
            smoke_out = bundle.parent / ('psdl1-smoke-' + uuid.uuid4().hex)
        execute('optional synthetic DL1 smoke', python, '-I', bundle / 'smoke_windows_bundle.py', '--out', smoke_out)
        print(f'Synthetic DL1 smoke passed. Evidence: {smoke_out}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--venv', type=Path, help='default: .venv-cmd beside this installer; existing paths are preserved')
    p.add_argument('--reuse-venv', action='store_true', help='validate exact existing environment; never install/update')
    p.add_argument('--smoke', action='store_true', help='also run synthetic PNG/diagnostic/export/CPU train/search smoke')
    p.add_argument('--smoke-out', type=Path, help='new output outside bundle, requires --smoke')
    p.add_argument('--check-runtime', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    try:
        if args.check_runtime:
            check_runtime()
            print('CPython 3.12 Windows x64 OK')
            return 0
        if args.smoke_out and not args.smoke:
            p.error('--smoke-out requires --smoke')
        bundle = Path(__file__).resolve().parent
        install(bundle, args.venv or bundle / '.venv-cmd', reuse=args.reuse_venv,
                smoke=args.smoke, smoke_out=args.smoke_out)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'Offline installation stopped: {exc}\nExisting files/environment were preserved. '
              'For DLL failures check the preinstalled Microsoft VC++ x64 runtime. '
              'Keep the error and retry after fixing the cause; no online fallback is used.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
