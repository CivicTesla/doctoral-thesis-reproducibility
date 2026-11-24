#!/usr/bin/env python3
"""Verify copied V1 artifacts without executing experiments or changing files."""
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    rows = list(csv.DictReader((ROOT / 'GITHUB_FILE_MANIFEST.csv').open(newline='')))
    failures = []
    for row in rows:
        rel = Path(row['release_path'])
        if rel.is_absolute() or '..' in rel.parts:
            failures.append(f'unsafe path: {rel}')
            continue
        path = ROOT / rel
        if not path.is_file() or path.is_symlink():
            failures.append(f'missing: {rel}')
        elif path.stat().st_size != int(row['bytes']) or sha256(path) != row['sha256']:
            failures.append(f'mismatch: {rel}')
    for item in failures:
        print(item)
    print(f'{len(rows)-len(failures)}/{len(rows)} copied files passed manifest integrity')
    raise SystemExit(bool(failures))

if __name__ == '__main__':
    main()
