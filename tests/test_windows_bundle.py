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


installer = load('install_windows_bundle')
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
    (root / 'install_windows_bundle.cmd').write_text('# installer fixture', encoding='utf-8')
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
    elif damage == 'installer-tamper': (tmp_path / 'install_windows_bundle.cmd').write_text('changed')
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
    import struct
    import sys
    monkeypatch.setattr(platform, 'python_implementation', lambda: implementation)
    monkeypatch.setattr(platform, 'machine', lambda: machine)
    monkeypatch.setattr(sys, 'version_info', version)
    monkeypatch.setattr(sys, 'platform', system)
    monkeypatch.setattr(struct, 'calcsize', lambda _: bits)
    if accepted: installer.check_runtime()
    else:
        with pytest.raises(ValueError): installer.check_runtime()


def test_active_cmd_venv_is_allowed_but_payload_is_still_strict(tmp_path, monkeypatch):
    import sys
    bundle(tmp_path)
    venv = tmp_path / '.venv-cmd'
    venv.mkdir(); (venv / 'pyvenv.cfg').write_text('include-system-site-packages = false\n')
    (venv / 'Scripts').mkdir(); (venv / 'Scripts/python.exe').write_bytes(b'fixture')
    with pytest.raises(ValueError, match='inventory'): verify.verify(tmp_path)
    monkeypatch.setattr(sys, 'prefix', str(venv))
    monkeypatch.setattr(sys, 'base_prefix', str(tmp_path.parent / 'base'))
    verify.verify(tmp_path)
    (tmp_path / 'unexpected').mkdir()
    with pytest.raises(ValueError, match='inventory'): verify.verify(tmp_path)


def test_cmd_venv_with_system_packages_rejected(tmp_path, monkeypatch):
    import sys
    bundle(tmp_path)
    venv = tmp_path / '.venv-cmd'; venv.mkdir()
    (venv / 'pyvenv.cfg').write_text('include-system-site-packages = true\n')
    monkeypatch.setattr(sys, 'prefix', str(venv))
    monkeypatch.setattr(sys, 'base_prefix', str(tmp_path.parent / 'base'))
    with pytest.raises(ValueError, match='system-site-packages'): verify.verify(tmp_path)


def test_installer_existing_venv_is_preserved_without_running_commands(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, 'check_runtime', lambda: None)
    monkeypatch.setattr(installer, 'run', lambda *a, **kw: pytest.fail('must not run against existing venv'))
    venv = tmp_path / '.venv-cmd'; venv.mkdir()
    marker = venv / 'existing-file'; marker.write_bytes(b'preserve')
    with pytest.raises(ValueError, match='preserved'): installer.install(tmp_path, venv)
    assert marker.read_bytes() == b'preserve'


@pytest.mark.parametrize('reuse,smoke', [(False, False), (True, False), (False, True)])
def test_installer_cmd_order_offline_flags_and_optional_smoke(tmp_path, monkeypatch, reuse, smoke):
    monkeypatch.setattr(installer, 'check_runtime', lambda: None)
    root = tmp_path / '한글 bundle'; root.mkdir()
    venv = root / '.venv-cmd'
    if reuse:
        venv.mkdir(); (venv / 'pyvenv.cfg').write_text('include-system-site-packages = false\n')
    calls = []
    monkeypatch.setattr(installer, 'run', lambda label, command, **kw: calls.append((label, list(map(str, command)), kw)))
    installer.install(root, venv, reuse=reuse, smoke=smoke)
    labels = [c[0] for c in calls]
    assert ('create venv' in labels) == (not reuse)
    assert ('local wheels only' in labels) == (not reuse)
    assert ('optional synthetic DL1 smoke' in labels) == smoke
    assert labels.index('bundle hashes/inventory/target/dependencies') < labels.index('pip check') < labels.index('DL1 help')
    for _, command, options in calls:
        assert options['cwd'] == root
        assert options['env']['PIP_CONFIG_FILE'] == __import__('os').devnull
        assert options['env']['PIP_NO_INDEX'] == '1'
        assert options['env']['HF_HUB_OFFLINE'] == '1'
        assert not any('powershell' in part.lower() or part == 'uv' or 'https:' in part for part in command)
    if not reuse:
        command = next(c[1] for c in calls if c[0] == 'local wheels only')
        assert {'--no-index', '--only-binary=:all:', '--require-hashes', '--find-links=wheelhouse', '--isolated'} <= set(command)
    if smoke:
        assert labels[-1] == 'optional synthetic DL1 smoke'
        path = Path(calls[-1][1][-1])
        assert not path.is_relative_to(root)


def test_failed_venv_creation_never_installs_and_preserves_partial_work(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(installer, 'check_runtime', lambda: None)
    venv = tmp_path / '.venv-cmd'
    def fail(label, command, **kw):
        assert label == 'create venv'
        venv.mkdir(); (venv / 'partial').write_text('retain')
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(installer, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError): installer.install(tmp_path, venv)
    assert (venv / 'partial').read_text() == 'retain'


@pytest.mark.parametrize('protect_bytes', [False, True])
def test_git_windows_autocrlf_transfer_with_and_without_attributes(tmp_path, protect_bytes):
    """Reproduce the field failure via real Git checkout, without a Windows shell."""
    import subprocess
    repo = tmp_path / 'repo'; repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    archive = repo / 'input.zip'; archive.write_bytes(bytes(range(256)) * 100)
    folder = repo / 'bundles/windows_cp312_win_amd64.parts'
    transfer.split(archive, folder, 5000)
    if protect_bytes:
        shutil.copy2(ROOT / '.gitattributes', repo / '.gitattributes')
    git = ['git', '-C', str(repo), '-c', 'core.autocrlf=true']
    subprocess.run([*git, 'add', '.'], check=True, capture_output=True)
    exported = tmp_path / 'Windows checkout'; exported.mkdir()
    subprocess.run([*git, 'checkout-index', '--all', '--prefix', str(exported) + '/'], check=True)
    checkout_parts = exported / folder.relative_to(repo)
    output = tmp_path / 'rejoined.zip'
    if protect_bytes:
        transfer.join(checkout_parts, output)
        assert output.read_bytes() == archive.read_bytes()
        assert (checkout_parts / 'reassemble.py').read_bytes() == (folder / 'reassemble.py').read_bytes()
    else:
        assert b'\r\n' in (checkout_parts / 'reassemble.py').read_bytes()
        before = (checkout_parts / 'reassemble.py').read_bytes()
        transfer.join(checkout_parts, output)
        assert output.read_bytes() == archive.read_bytes()
        assert (checkout_parts / 'reassemble.py').read_bytes() == before


@pytest.mark.parametrize('bom,crlf', [(False, True), (True, False), (True, True)])
def test_reassembly_accepts_only_hash_identical_text_representations(tmp_path, bom, crlf):
    archive, folder = parts(tmp_path)
    path = folder / 'reassemble.py'
    data = path.read_bytes()
    if crlf: data = data.replace(b'\n', b'\r\n')
    if bom: data = b'\xef\xbb\xbf' + data
    path.write_bytes(data)
    out = tmp_path / 'joined.zip'
    transfer.join(folder, out)
    assert out.read_bytes() == archive.read_bytes()
    assert path.read_bytes() == data


def test_reassembly_rejects_changed_code_even_after_crlf_normalization(tmp_path):
    _, folder = parts(tmp_path)
    path = folder / 'reassemble.py'
    data = path.read_bytes().replace(b'\n', b'\r\n') + b'\r\n# different release\r\n'
    path.write_bytes(data)
    out = tmp_path / 'joined.zip'
    with pytest.raises(ValueError, match=r'file=.*expected=.*actual=.*canonical_LF='):
        transfer.join(folder, out)
    assert path.read_bytes() == data
    assert not out.exists()


def test_transfer_cli_version_and_actionable_mismatch(tmp_path):
    import subprocess
    import sys
    tool = ROOT / 'scripts/transfer_windows_bundle.py'
    result = subprocess.run([sys.executable, str(tool), '--version'], capture_output=True, text=True)
    assert result.returncode == 0 and 'psdl1-transfer 2' in result.stdout
    _, folder = parts(tmp_path)
    script = folder / 'reassemble.py'; script.write_bytes(script.read_bytes() + b'\n# changed\n')
    result = subprocess.run([sys.executable, str(tool), 'join', str(folder), '--output', str(tmp_path/'joined.zip')],
                            capture_output=True, text=True)
    assert result.returncode == 1 and 'Traceback' not in result.stderr
    assert 'expected=' in result.stderr and 'actual=' in result.stderr and 'canonical_LF=' in result.stderr
    assert not (tmp_path/'joined.zip').exists()
