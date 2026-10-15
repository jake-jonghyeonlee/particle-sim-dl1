"""Offline bundle integrity, target dependency closure and lossless Git transfer contracts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = load('verify_windows_bundle')
transfer = load('transfer_windows_bundle')


def seal(root, manifest):
    manifest['files'] = {p.relative_to(root).as_posix(): {'size_bytes': p.stat().st_size, 'sha256': verify.sha256(p)}
                         for p in root.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')


def bundle(root, *, dependency='numpy>=1.26', numpy_tag='cp312-cp312-win_amd64', numpy_python='>=3.12'):
    (root / 'wheelhouse').mkdir()
    records = []
    for name, version, tag, requires, requires_python in [
            ('psdl1', '0.1.0', 'py3-none-any', [dependency, 'Pillow==12.3.0; extra == "png"'], '>=3.11'),
            ('numpy', '2.5.3', numpy_tag, [], numpy_python),
            ('pillow', '12.3.0', 'cp312-cp312-win_amd64', [], '>=3.10')]:
        wheel = root / 'wheelhouse' / f'{name}-{version}-{tag}.whl'
        metadata = f'Name: {name}\nVersion: {version}\nRequires-Python: {requires_python}\n'
        metadata += ''.join(f'Requires-Dist: {r}\n' for r in requires)
        with zipfile.ZipFile(wheel, 'w') as z:
            z.writestr(f'{name}-{version}.dist-info/METADATA', metadata)
            if name == 'psdl1':
                z.writestr('psdl1/__init__.py', b'# synthetic application\n')
        records.append({'file': wheel.name, 'metadata': verify.read_wheel(wheel)})
    lines = [f"{e['metadata']['name']}{'[png]' if e['metadata']['name']=='psdl1' else ''}=={e['metadata']['version']} --hash=sha256:{verify.sha256(root/'wheelhouse'/e['file'])}"
             for e in records]
    (root / 'requirements-windows.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (root / 'install_and_smoke.ps1').write_text('# installer fixture', encoding='utf-8')
    manifest = {'schema_version': verify.SCHEMA, 'target': verify.TARGET, 'wheels': records,
                'application_sources': {'psdl1/__init__.py': hashlib.sha256(b'# synthetic application\n').hexdigest()}}
    seal(root, manifest)
    return manifest


def test_complete_bundle_accepts_spaces_unicode_and_png_extra(tmp_path):
    root = tmp_path / '폐쇄망 bundle'; root.mkdir()
    bundle(root)
    _, packages = verify.verify(root)
    assert set(packages) == {'psdl1', 'numpy', 'pillow'}


@pytest.mark.parametrize('damage', ['wheel-tamper', 'missing', 'extra-wheel', 'extra-file', 'extra-dir',
                                     'installer-tamper', 'requirements-tamper', 'symlink'])
def test_strict_inventory_and_hashes(tmp_path, damage):
    bundle(tmp_path)
    wheel = next((tmp_path / 'wheelhouse').glob('numpy*'))
    if damage == 'wheel-tamper': wheel.write_bytes(wheel.read_bytes() + b'x')
    elif damage == 'missing': wheel.unlink()
    elif damage == 'extra-wheel': shutil.copy2(wheel, wheel.with_name('extra.whl'))
    elif damage == 'extra-file': (tmp_path / 'unexpected.txt').touch()
    elif damage == 'extra-dir': (tmp_path / 'empty').mkdir()
    elif damage == 'installer-tamper': (tmp_path / 'install_and_smoke.ps1').write_text('changed')
    elif damage == 'requirements-tamper': (tmp_path / 'requirements-windows.txt').write_text('numpy')
    elif damage == 'symlink': wheel.with_name('linked').symlink_to(wheel)
    with pytest.raises(ValueError): verify.verify(tmp_path)


@pytest.mark.parametrize('dependency,accepted', [
    ('numpy>=1.26', True), ('numpy>=3', False), ('missing; sys_platform == "win32"', False),
    ('missing; sys_platform == "darwin"', True), ('missing; platform_system == "Linux"', True),
    ('missing; python_version == "3.12"', False), ('missing; extra == "png"', False),
    ('missing; extra == "timm"', True), ('numpy @ https://example.invalid/wheel.whl', False)])
def test_target_marker_and_version_closure(tmp_path, dependency, accepted):
    bundle(tmp_path, dependency=dependency)
    if accepted: verify.verify(tmp_path)
    else:
        with pytest.raises(ValueError, match='closure'): verify.verify(tmp_path)


@pytest.mark.parametrize('tag', ['cp311-cp311-win_amd64', 'cp312-cp312-win32',
                                  'cp312-cp312-win_arm64', 'cp312-cp312-macosx_11_0_arm64'])
def test_wrong_target_tag_rejected(tmp_path, tag):
    bundle(tmp_path, numpy_tag=tag)
    with pytest.raises(ValueError, match='incompatible'): verify.verify(tmp_path)


def test_python_requirement_is_target_python(tmp_path):
    bundle(tmp_path, numpy_python='>=3.13')
    with pytest.raises(ValueError, match='Python 3.12'): verify.verify(tmp_path)


@pytest.mark.parametrize('damage', ['url', 'extra-pin', 'no-hash', 'wrong-hash', 'unactivate-png'])
def test_requirements_cannot_reintroduce_online_or_unhashed_input(tmp_path, damage):
    manifest = bundle(tmp_path)
    path = tmp_path / 'requirements-windows.txt'
    content = path.read_text()
    if damage == 'url': content += '--find-links https://example.invalid\n'
    elif damage == 'extra-pin': content += 'unexpected==1.0 --hash=sha256:'+'0'*64+'\n'
    elif damage == 'no-hash': content = content.split(' --hash')[0] + '\n'
    elif damage == 'wrong-hash': content = content.replace('--hash=sha256:', '--hash=sha256:0')
    else: content = content.replace('[png]', '')
    path.write_text(content)
    seal(tmp_path, manifest)
    with pytest.raises(ValueError, match='requirements'): verify.verify(tmp_path)


def test_wheel_metadata_and_app_source_are_independently_checked(tmp_path):
    manifest = bundle(tmp_path)
    manifest['application_sources']['psdl1/__init__.py'] = '0' * 64
    seal(tmp_path, manifest)
    with pytest.raises(ValueError, match='source'): verify.verify(tmp_path)


def test_missing_png_dependency_rejected_even_with_consistent_inventory(tmp_path):
    manifest = bundle(tmp_path)
    pillow = manifest['wheels'].pop()
    (tmp_path / 'wheelhouse' / pillow['file']).unlink()
    path = tmp_path / 'requirements-windows.txt'
    path.write_text('\n'.join(x for x in path.read_text().splitlines() if not x.startswith('pillow'))+'\n')
    seal(tmp_path, manifest)
    with pytest.raises(ValueError, match='closure'): verify.verify(tmp_path)


@pytest.mark.parametrize('path', ['../escape', '/absolute', 'wheelhouse/../escape', 'C:/escape', 'a\\b', 'a//b'])
def test_manifest_path_traversal_rejected(tmp_path, path):
    manifest = bundle(tmp_path)
    manifest['files'][path] = {'size_bytes': 0, 'sha256': '0' * 64}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='unsafe'): verify.verify(tmp_path)


def parts(tmp_path):
    archive = tmp_path / 'bundle.zip'
    archive.write_bytes(bytes(range(256)) * 100)
    folder = tmp_path / '전달 parts'
    transfer.split(archive, folder, 5000)
    return archive, folder


def test_transfer_exact_roundtrip_and_no_overwrite(tmp_path):
    archive, folder = parts(tmp_path)
    output = tmp_path / 'rejoined.zip'
    transfer.join(folder, output)
    assert output.read_bytes() == archive.read_bytes()
    output.write_bytes(b'existing output')
    with pytest.raises(FileExistsError): transfer.join(folder, output)
    assert output.read_bytes() == b'existing output'


@pytest.mark.parametrize('damage', ['tamper', 'missing', 'unexpected', 'order', 'archive-hash', 'script'])
def test_transfer_rejects_damage_before_publication(tmp_path, damage):
    _, folder = parts(tmp_path)
    paths = sorted(folder.glob('*.part*'))
    manifest_path = folder / 'parts.json'
    manifest = json.loads(manifest_path.read_text())
    if damage == 'tamper': paths[0].write_bytes(b'tampered')
    elif damage == 'missing': paths[0].unlink()
    elif damage == 'unexpected': (folder / 'extra').touch()
    elif damage == 'order':
        manifest['parts'].reverse(); manifest_path.write_text(json.dumps(manifest))
    elif damage == 'archive-hash':
        manifest['sha256'] = '0' * 64; manifest_path.write_text(json.dumps(manifest))
    else: (folder / 'reassemble.py').write_text('changed')
    output = tmp_path / 'output.zip'
    with pytest.raises(ValueError): transfer.join(folder, output)
    assert not output.exists()


def test_reuse_does_not_accept_the_active_source_environment(tmp_path):
    bundle(tmp_path)
    with pytest.raises(ValueError, match='isolated CPython'):
        verify.check_installed(tmp_path, tmp_path / 'different-prefix')


@pytest.mark.parametrize('implementation,version,bits,machine,system,accepted', [
    ('CPython', (3,12), 8, 'AMD64', 'win32', True),
    ('CPython', (3,11), 8, 'AMD64', 'win32', False),
    ('CPython', (3,13), 8, 'AMD64', 'win32', False),
    ('CPython', (3,12), 4, 'x86', 'win32', False),
    ('CPython', (3,12), 8, 'ARM64', 'win32', False),
    ('PyPy', (3,12), 8, 'AMD64', 'win32', False),
    ('CPython', (3,12), 8, 'AMD64', 'darwin', False)])
def test_installer_interpreter_acceptance_matrix(monkeypatch, implementation, version, bits, machine, system, accepted):
    import platform
    import re
    import struct
    import sys
    text = (ROOT / 'scripts/install_and_smoke.ps1').read_text(encoding='ascii')
    predicate = re.search(r'^\$targetCheck = "(.*)"$', text, re.M).group(1)
    monkeypatch.setattr(platform, 'python_implementation', lambda: implementation)
    monkeypatch.setattr(platform, 'machine', lambda: machine)
    monkeypatch.setattr(sys, 'version_info', version)
    monkeypatch.setattr(sys, 'platform', system)
    monkeypatch.setattr(struct, 'calcsize', lambda _: bits)
    if accepted: exec(predicate, {})
    else:
        with pytest.raises(AssertionError): exec(predicate, {})
