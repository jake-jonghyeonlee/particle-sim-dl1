#!/usr/bin/env python3
"""Split into Git-sized pieces; reassemble only after exact inventory/SHA256 checks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def split(archive, folder, part_bytes=48_000_000):
    if not 1 <= part_bytes <= 50_000_000:
        raise ValueError('part size must be 1..50,000,000 bytes')
    folder.mkdir(parents=True, exist_ok=False)
    records = []
    with archive.open('rb') as stream:
        index = 1
        while block := stream.read(part_bytes):
            name = f'{archive.name}.part{index:03d}'
            (folder / name).write_bytes(block)
            records.append({'file': name, 'size_bytes': len(block), 'sha256': hashlib.sha256(block).hexdigest()})
            index += 1
    shutil.copy2(__file__, folder / 'reassemble.py')
    manifest = {'schema_version': 'psdl1_split_zip_v1', 'archive': archive.name,
                'size_bytes': archive.stat().st_size, 'sha256': digest(archive), 'parts': records,
                'reassemble_sha256': digest(folder / 'reassemble.py')}
    (folder / 'parts.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def join(folder, output):
    manifest = json.loads((folder / 'parts.json').read_text(encoding='utf-8'))
    if manifest['schema_version'] != 'psdl1_split_zip_v1' or not manifest['parts']:
        raise ValueError('unsupported/empty parts manifest')
    names = [p['file'] for p in manifest['parts']]
    expected_names = [f"{manifest['archive']}.part{i:03d}" for i in range(1, len(names) + 1)]
    if (names != expected_names or any('/' in n or '\\' in n or ':' in n or n.startswith('.') for n in names)
            or len(set(names)) != len(names)):
        raise ValueError('invalid part names/order')
    if {p.name for p in folder.iterdir()} != set(names) | {'parts.json', 'reassemble.py'}:
        raise ValueError('parts inventory differs')
    for path in folder.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError('parts must be regular files')
    if digest(folder / 'reassemble.py') != manifest['reassemble_sha256']:
        raise ValueError('reassembly script hash differs')
    for record in manifest['parts']:
        path = folder / record['file']
        if path.stat().st_size != record['size_bytes'] or digest(path) != record['sha256']:
            raise ValueError(f'part integrity failed: {path.name}')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: an existing archive is never replaced.
    with output.open('xb') as stream:
        try:
            for name in names:
                with (folder / name).open('rb') as part:
                    shutil.copyfileobj(part, stream, 1024 * 1024)
            stream.flush()
            if output.stat().st_size != manifest['size_bytes'] or digest(output) != manifest['sha256']:
                raise ValueError('reassembled archive hash differs')
        except BaseException:
            stream.close()
            output.unlink()
            raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('split')
    s.add_argument('archive', type=Path)
    s.add_argument('folder', type=Path)
    s.add_argument('--part-bytes', type=int, default=48_000_000)
    j = sub.add_parser('join')
    j.add_argument('folder', type=Path)
    j.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = split(args.archive, args.folder, args.part_bytes) if args.command == 'split' else join(args.folder, args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
