#!/usr/bin/env python3
"""Connected-machine-only builder. Runtime installation uses pip, never uv or the network."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile

from verify_windows_bundle import SCHEMA, TARGET, read_wheel, sha256, verify

ROOT = Path(__file__).resolve().parents[1]
UV_VERSION = '0.11.23'


def run(*args, cwd=ROOT, env=None):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True)


def provenance(wheel, metadata):
    digest = sha256(wheel)
    if metadata['name'].lower() == 'torch':
        index = 'https://download.pytorch.org/whl/cpu/torch/'
        content = urllib.request.urlopen(index, timeout=120).read().decode()
        for url in re.findall(r'href="([^"]+)"', content):
            from urllib.parse import unquote, urlsplit
            if unquote(urlsplit(url).path.rsplit('/', 1)[-1]) == wheel.name and url.endswith('#sha256=' + digest):
                return {'index': index, 'url': url.split('#')[0], 'published_sha256': digest}
        raise ValueError('torch wheel/hash not listed by the official CPU index')
    url = f"https://pypi.org/pypi/{metadata['name']}/{metadata['version']}/json"
    info = json.load(urllib.request.urlopen(url, timeout=120))
    for entry in info['urls']:
        if entry['filename'] == wheel.name and entry['digests']['sha256'] == digest:
            return {'index': url, 'url': entry['url'], 'published_sha256': digest}
    raise ValueError(f'wheel/hash not listed on PyPI: {wheel.name}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/windows_cp312_win_amd64')
    p.add_argument('--download-cache', type=Path, default=ROOT / 'outputs/windows-downloads')
    p.add_argument('--parts', type=Path, help='optional NEW Git-transfer directory (48 MB per part)')
    args = p.parse_args()
    output = args.output.resolve()
    archive = output.with_suffix('.zip')
    if output.exists() or archive.exists() or (args.parts and args.parts.exists()):
        raise SystemExit('Output/archive/parts already exists; choose fresh paths. Nothing was removed.')
    uv = shutil.which('uv')
    if not uv or subprocess.check_output([uv, '--version'], text=True).split()[1] != UV_VERSION:
        raise SystemExit(f'Connected builder requires uv {UV_VERSION}. Offline host does not need uv.')
    output.mkdir(parents=True, exist_ok=False)
    wheelhouse = output / 'wheelhouse'
    wheelhouse.mkdir()
    evidence = output / 'build-evidence'
    evidence.mkdir()
    resolved = evidence / 'resolved-windows.txt'
    run(uv, 'pip', 'compile', ROOT / 'scripts/windows-runtime.in', '--quiet', '--no-header', '--no-annotate',
        '--python-platform', 'x86_64-pc-windows-msvc', '--python-version', '3.12',
        '--only-binary', ':all:', '--torch-backend', 'cpu', '--default-index', 'https://pypi.org/simple',
        '--no-config', '--no-python-downloads', '--generate-hashes', '--output-file', resolved)
    cache = args.download_cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    # uv resolved all deps for Windows. --no-deps prevents pip reinterpreting markers on macOS.
    run(sys.executable, '-m', 'pip', '--isolated', '--disable-pip-version-check', 'download',
        '--index-url', 'https://pypi.org/simple', '--extra-index-url', 'https://download.pytorch.org/whl/cpu',
        '--require-hashes', '--only-binary=:all:', '--no-deps', '--platform', 'win_amd64',
        '--python-version', '3.12', '--implementation', 'cp', '--abi', 'cp312', '--abi', 'abi3',
        '--abi', 'none', '--dest', cache, '-r', resolved)
    # Copy only resolved packages, never stale/unrelated cached wheels.
    pins = dict(re.findall(r'^([\w.-]+)==([^\s\\]+)', resolved.read_text(), re.M))
    canon = lambda s: re.sub(r'[-_.]+', '-', s).lower()
    pins = {canon(k): v for k, v in pins.items()}
    for name, version in pins.items():
        matches = [w for w in cache.glob('*.whl') if canon(w.name.split('-')[0]) == name and w.name.split('-')[1] == version]
        if len(matches) != 1:
            raise ValueError(f'exactly one cached target wheel required for {name} {version}')
        shutil.copy2(matches[0], wheelhouse / matches[0].name)
    built = evidence / 'built'
    run(uv, 'build', '--wheel', '--no-config', '--out-dir', built,
        '--build-constraints', ROOT / 'scripts/windows-build-constraints.txt', ROOT)
    app, = built.glob('psdl1-*.whl')
    shutil.move(app, wheelhouse / app.name)
    (built / ".gitignore").unlink(missing_ok=True)
    built.rmdir()
    # Source content check includes the diagnostic improvements in the supplied baseline.
    source_hashes = {p.relative_to(ROOT / 'src').as_posix(): sha256(p)
                     for p in sorted((ROOT / 'src/psdl1').rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    with zipfile.ZipFile(wheelhouse / app.name) as z:
        wheel_sources = {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('psdl1/') and not n.endswith('/')}
    if wheel_sources != source_hashes:
        raise ValueError('built application differs from complete src/psdl1 tree')
    for name in ('verify_windows_bundle.py', 'install_and_smoke.ps1', 'smoke_windows_bundle.py',
                 'windows-runtime.in', 'windows-build-constraints.txt'):
        shutil.copy2(ROOT / 'scripts' / name, output / name)
    shutil.copy2(ROOT / 'docs/WINDOWS_OFFLINE.md', output / 'README.md')
    wheels, lines = [], []
    for path in sorted(wheelhouse.glob('*.whl')):
        meta = read_wheel(path)
        origin = {'index': 'local source build', 'url': None, 'published_sha256': None} if path.name == app.name else provenance(path, meta)
        wheels.append({'file': path.name, 'metadata': meta, 'source': origin})
        extra = '[png]' if path.name == app.name else ''
        lines.append(f"{meta['name']}{extra}=={meta['version']} --hash=sha256:{sha256(path)}")
    (output / 'requirements-windows.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    git = lambda *a: subprocess.check_output(['git', *a], cwd=ROOT, text=True).strip()
    manifest = {'schema_version': SCHEMA, 'target': TARGET,
                'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'builder': {'python': sys.version, 'uv': UV_VERSION, 'hatchling': '1.27.0'},
                'application': {'baseline': git('rev-parse', 'HEAD'), 'branch': git('branch', '--show-current'),
                                'git_status_at_build': git('status', '--short')},
                'reference': {'branch': 'rewrite/line-pixel-rate', 'commit': 'd417ad847244ac0cd986f6195e23c4ba8d88d08c'},
                'application_sources': source_hashes, 'wheels': wheels,
                'files': {p.relative_to(output).as_posix(): {'size_bytes': p.stat().st_size, 'sha256': sha256(p)}
                          for p in sorted(output.rglob('*')) if p.is_file()}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    verify(output)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(output.parent))
    # Verify archive CRC and extracted layout, not just the pre-archive directory.
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('archive CRC failed')
    result = {'archive': str(archive), 'size_bytes': archive.stat().st_size, 'sha256': sha256(archive),
              'wheels': len(wheels), 'expanded_bytes': sum(r['size_bytes'] for r in manifest['files'].values()),
              'installed_wheel_bytes': sum(i.file_size for p in wheelhouse.glob('*.whl') for i in zipfile.ZipFile(p).infolist())}
    archive.with_suffix('.zip.sha256').write_text(f"{result['sha256']}  {archive.name}\n", encoding='utf-8')
    if args.parts:
        from transfer_windows_bundle import split
        split(archive, args.parts.resolve())
        result['parts'] = str(args.parts.resolve())
    archive.with_suffix('.build-result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
