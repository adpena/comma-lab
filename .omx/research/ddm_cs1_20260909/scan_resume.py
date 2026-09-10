"""Run the existing SSD audit with a resumable, stat-validated hash journal.

Apparatus only. Never writes to scan roots. Resume by repeating this command.
"""
import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-from', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('ssd_audit', 'tools/audit_ssd_authored_signal.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    records = {}
    if args.resume_from.exists():
        for line in args.resume_from.read_text().splitlines():
            try:
                row = json.loads(line)
                records[row['path']] = row
            except (ValueError, KeyError):
                continue
    original = audit.blob_sha1
    with args.resume_from.open('a', buffering=1) as journal:
        def cached_hash(path):
            try:
                stat = path.stat()
                signature = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
                old = records.get(str(path))
                if old and old['signature'] == signature:
                    return old['blob_sha1']
                digest = original(path)
                after = path.stat()
                if signature != [after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns]:
                    return None
                if digest is not None:
                    row = dict(path=str(path), signature=signature, blob_sha1=digest)
                    journal.write(json.dumps(row) + '\n')
                    records[str(path)] = row
                return digest
            except OSError:
                return None
        audit.blob_sha1 = cached_hash
        return audit.main(['--write-cache', '--manifest', str(args.manifest)])


if __name__ == '__main__':
    raise SystemExit(main())
