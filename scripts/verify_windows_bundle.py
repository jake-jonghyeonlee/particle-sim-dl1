#!/usr/bin/env python3
"""Offline inventory, hash, target tag and dependency-closure verification (pip required)."""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile

SCHEMA = 'psdl1_windows_bundle_v1'
TARGET = 'cp312-win_amd64'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def packaging_api():
    # pip's vendored packaging works before runtime installation; developer venvs
    # created by uv may instead have standalone packaging and no pip.
    import importlib
    try:
        root = 'pip._vendor.packaging'
        importlib.import_module(root)
    except ImportError:
        root = 'packaging'
    module = lambda part: importlib.import_module(root + '.' + part)
    return (module('requirements').Requirement, module('specifiers').SpecifierSet,
            module('tags').compatible_tags, module('tags').cpython_tags,
            module('utils').canonicalize_name, module('utils').parse_wheel_filename)



def safe_relative(name):
    p = PurePosixPath(name)
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name
            or p.is_absolute() or any(x in ('', '.', '..') for x in name.split('/'))):
        raise ValueError(f'unsafe bundle path: {name!r}')
    return p


def read_wheel(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('duplicate ZIP members')
        for name in names:
            safe_relative(name.rstrip('/'))
        metadata = [n for n in names if n.count('/') == 1 and n.endswith('.dist-info/METADATA')]
        if len(metadata) != 1:
            raise ValueError(f'invalid wheel metadata: {path.name}')
        m = BytesParser().parsebytes(z.read(metadata[0]))
        return {'name': m['Name'], 'version': m['Version'],
                'requires_python': m.get('Requires-Python', ''),
                'requires_dist': m.get_all('Requires-Dist', [])}


def check_closure(wheels):
    Requirement, SpecifierSet, compatible_tags, cpython_tags, canon, parse = packaging_api()
    tags = set(cpython_tags((3, 12), abis=['cp312'], platforms=['win_amd64']))
    tags.update(compatible_tags((3, 12), interpreter='cp312', platforms=['win_amd64']))
    packages = {}
    for entry in wheels:
        name, version, _, wheel_tags = parse(entry['file'])
        if not tags.intersection(wheel_tags):
            raise ValueError(f'incompatible target wheel: {entry["file"]}')
        meta = entry['metadata']
        if name != canon(meta['name']) or str(version) != meta['version'] or name in packages:
            raise ValueError('duplicate or mismatched wheel identity')
        if not SpecifierSet(meta['requires_python']).contains('3.12.0'):
            raise ValueError(f'Python 3.12 unsupported: {name}')
        packages[name] = meta
    env = dict(os_name='nt', sys_platform='win32', platform_machine='AMD64',
               platform_python_implementation='CPython', implementation_name='cpython',
               implementation_version='3.12.0', python_version='3.12', python_full_version='3.12.0',
               platform_release='10', platform_system='Windows', platform_version='10.0', extra='')
    extras = {name: set() for name in packages}
    if 'psdl1' not in packages:
        raise ValueError('missing psdl1 wheel')
    extras['psdl1'].add('png')
    changed = True
    while changed:
        changed = False
        for name, meta in packages.items():
            for raw in meta['requires_dist']:
                req = Requirement(raw)
                if req.marker and not any(req.marker.evaluate(dict(env, extra=e)) for e in {''} | extras[name]):
                    continue
                dependency = canon(req.name)
                if req.url or dependency not in packages or not req.specifier.contains(packages[dependency]['version']):
                    raise ValueError(f'dependency closure failure: {name} -> {raw}')
                if not req.extras <= extras[dependency]:
                    extras[dependency].update(req.extras)
                    changed = True
    return packages


def verify(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != SCHEMA or manifest.get('target') != TARGET:
        raise ValueError('unsupported bundle schema/target')
    files = manifest['files']
    if not isinstance(files, dict) or not files:
        raise ValueError('empty inventory')
    expected_dirs = set()
    for name in files:
        p = safe_relative(name)
        expected_dirs.update(str(parent) for parent in p.parents if str(parent) != '.')
    # Match the field-tested CMD order: create .venv-cmd, then verify from
    # that interpreter. Only that active, isolated venv may be excluded.
    # An arbitrary pre-existing folder or a venv used from another interpreter
    # remains unexpected inventory. Payload files stay strict.
    active_venv = folder / '.venv-cmd'
    ignore_venv = False
    if active_venv.exists() and Path(sys.prefix).resolve() == active_venv.resolve() and sys.prefix != sys.base_prefix:
        if active_venv.is_symlink():
            raise ValueError('symlink venv forbidden inside bundle')
        cfg = (active_venv / 'pyvenv.cfg').read_text(encoding='utf-8').lower()
        if not re.search(r'include-system-site-packages\s*=\s*false', cfg):
            raise ValueError('system-site-packages forbidden')
        ignore_venv = True
    actual_files, actual_dirs = set(), set()
    for base, dirs, names in os.walk(folder, followlinks=False):
        base = Path(base)
        for name in list(dirs):
            path = base / name
            if path.is_symlink():
                raise ValueError('symlinks forbidden in bundle')
            if ignore_venv and path == active_venv:
                dirs.remove(name)
                continue
            actual_dirs.add(path.relative_to(folder).as_posix())
        for name in names:
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise ValueError('only regular payload files allowed in bundle')
            actual_files.add(path.relative_to(folder).as_posix())
    expected_files = set(files) | {'manifest.json'}
    if actual_files != expected_files or actual_dirs != expected_dirs:
        raise ValueError(f'inventory differs; missing={sorted(expected_files-actual_files)} '
                         f'unexpected={sorted(actual_files-expected_files)} directories={sorted(actual_dirs ^ expected_dirs)}')
    for name, record in files.items():
        path = folder / name
        if path.stat().st_size != record['size_bytes'] or sha256(path) != record['sha256']:
            raise ValueError(f'integrity check failed: {name}')
    wheels = manifest['wheels']
    wheel_paths = ['wheelhouse/' + e['file'] for e in wheels]
    if len(wheel_paths) != len(set(wheel_paths)) or set(wheel_paths) != {n for n in files if n.startswith('wheelhouse/')}:
        raise ValueError('wheel inventory mismatch')
    locked = []
    for entry in wheels:
        path = folder / 'wheelhouse' / entry['file']
        if read_wheel(path) != entry['metadata']:
            raise ValueError('wheel metadata mismatch')
        name = entry['metadata']['name']
        extra = '[png]' if name.lower() == 'psdl1' else ''
        locked.append(f"{name}{extra}=={entry['metadata']['version']} --hash=sha256:{files['wheelhouse/' + entry['file']]['sha256']}")
    lines = (folder / 'requirements-windows.txt').read_text(encoding='utf-8').splitlines()
    if sorted(lines) != sorted(locked):
        raise ValueError('requirements must exactly pin every bundled wheel and hash; no URLs/options')
    packages = check_closure(wheels)
    app = next(e for e in wheels if e['metadata']['name'].lower() == 'psdl1')
    with zipfile.ZipFile(folder / 'wheelhouse' / app['file']) as z:
        sources = {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('psdl1/') and not n.endswith('/')}
    if sources != manifest['application_sources']:
        raise ValueError('application wheel source mismatch')
    return manifest, packages


def check_installed(folder, prefix):
    manifest, packages = verify(folder)
    prefix = Path(prefix).resolve()
    if (sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != prefix
            or sys.version_info[:2] != (3, 12)):
        raise ValueError('must use the requested isolated CPython 3.12 venv')
    cfg = (prefix / 'pyvenv.cfg').read_text(encoding='utf-8').lower()
    if not re.search(r'include-system-site-packages\s*=\s*false', cfg):
        raise ValueError('system-site-packages forbidden')
    canon = packaging_api()[4]
    installed = {}
    for dist in importlib.metadata.distributions():
        name = canon(dist.metadata['Name'])
        if name in installed:
            raise ValueError(f'duplicate installed distribution: {name}')
        installed[name] = dist
    if set(installed) - set(packages) - {'pip', 'setuptools'}:
        raise ValueError('existing environment contains unbundled packages; use a fresh path')
    for name, meta in packages.items():
        dist = installed.get(name)
        if dist is None or dist.version != meta['version']:
            raise ValueError(f'existing environment version differs: {name}')
        if not Path(dist.locate_file('')).resolve().is_relative_to(prefix) or dist.read_text('direct_url.json'):
            raise ValueError(f'non-isolated/editable installation forbidden: {name}')
    # Verify all application code, so a stale editable checkout cannot pass reuse.
    app = installed['psdl1']
    for name, digest in manifest['application_sources'].items():
        if sha256(app.locate_file(name)) != digest:
            raise ValueError(f'installed psdl1 source differs: {name}')
    return packages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--installed-prefix', type=Path)
    args = parser.parse_args()
    try:
        packages = check_installed(args.folder, args.installed_prefix) if args.installed_prefix else verify(args.folder)[1]
        print(f'Bundle integrity / Windows target dependency closure OK ({len(packages)} wheels)')
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        print(f'bundle verification failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
