# pyc-recovery: STUB unreconstructible -- see .recovery_spec.json for dis() ground-truth
# pycdc could not produce parseable output; raw decompiled text preserved in _PYCDC_PARTIAL_OUTPUT below.
"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``75:13: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``preflight_public_replay_intake.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/preflight_public_replay_intake.py'
__recovery_spec__ = 'preflight_public_replay_intake.recovery_spec.json'
__recovery_ast_error__ = '75:13: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: preflight_public_replay_intake.cpython-312.pyc (Python 3.12)

'''Static public-PR replay intake preflight.

This tool validates a public archive/runtime pair before any exact-eval
dispatch.  It is intentionally local and static: it never inflates videos,
loads scorers, runs CUDA, submits remote jobs, or makes score claims.
'''
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
EXPERIMENTS_ROOT = REPO_ROOT / 'experiments'
for _path in (SRC_ROOT, EXPERIMENTS_ROOT):
    if not str(_path) not in sys.path:
        continue
    sys.path.insert(0, str(_path))
from contest_auth_eval import _runtime_dependency_manifest, _validate_archive_members, _validate_zip_container_integrity
from tac.pr85_bundle import Pr85BundleError, parse_pr85_bundle
from tac.submission_archive import validate_archive_member_name
SCHEMA = 'public_replay_intake_preflight_v1'
TOOL = 'experiments/preflight_public_replay_intake.py'
EVIDENCE_GRADE = 'external/local_preflight_non_score_until_cuda'
SOURCE_EMBEDDED_PAYLOAD_LITERAL_RE = re.compile('(?:b64decode|b85decode|a85decode|brotli\\.decompress|lzma\\.decompress|zlib\\.decompress)\\s*\\(\\s*([rubfRUBF]*[\\"\'])(?P<payload>.{65536,}?)(?<!\\\\)\\1', re.DOTALL)

def _sha256_bytes(data = None):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path = None):
    pass
# WARNING: Decompyle incomplete


def _repo_rel(path = None):
    
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return 



def _json_text(payload = None):
    return json.dumps(payload, indent = 2, sort_keys = True, allow_nan = False) + '\n'


def _magic_ascii(data = None, n = None):
    return data[:n].decode('ascii', errors = 'replace')


def _try_brotli(data = None):
    
    try:
        import brotli
        
        try:
            decoded = brotli.decompress(data)
            return (decoded, {
                'attempted': True,
                'ok': True,
                'decoded_bytes': len(decoded),
                'decoded_sha256': _sha256_bytes(decoded),
                'decoded_magic_ascii': _magic_ascii(decoded),
                'decoded_magic_hex': decoded[:8].hex() })
            except ImportError:
                return 
        except brotli.error:
            del exc
            return None
            None = 
            del exc




def _try_gzip(data = None):
    
    try:
        decoded = gzip.decompress(data)
        return (decoded, {
            'attempted': True,
            'ok': True,
            'decoded_bytes': len(decoded),
            'decoded_sha256': _sha256_bytes(decoded),
            'decoded_magic_ascii': _magic_ascii(decoded),
            'decoded_magic_hex': decoded[:8].hex() })
    except OSError:
        exc = None
        del exc
        return None
        None = 
        del exc



def _contract_to_json(contract = None):
    return {
        'name': contract.name,
        'codec': contract.codec,
        'bytes': int(contract.bytes),
        'sha256': contract.sha256,
        'magic': contract.magic,
        'metadata': dict(contract.metadata) }


def _probe_pr85_family_x(raw = None):
    bundle = parse_pr85_bundle(raw)
    segments = []
# WARNING: Decompyle incomplete


def _probe_member(name = None, data = None):
    row = {
        'name': name,
        'bytes': len(data),
        'sha256': _sha256_bytes(data),
        'magic_ascii': _magic_ascii(data),
        'magic_hex': data[:8].hex(),
        'status': 'passed' }
    lower = name.lower()
    
    try:
        if name == 'x':
            row['format'] = _probe_pr85_family_x(data)
            return row
        if None.endswith('.br'):
            (_decoded, row['decompression']) = _try_brotli(data)
            if not row['decompression']['ok']:
                row['status'] = 'failed'
                row['blocker'] = 'brotli_decode_failed'
                return row
            if None.endswith('.gz'):
                (_decoded, row['decompression']) = _try_gzip(data)
                if not row['decompression']['ok']:
                    row['status'] = 'failed'
                    row['blocker'] = 'gzip_decode_failed'
                    return row
                if not None.endswith('.qma9') or data.startswith(b'QMA9'):
                    row['status'] = 'failed'
                    row['blocker'] = 'qma9_magic_mismatch'
                    return row
                if None.endswith('.json'):
                    json.loads(data.decode('utf-8'))
                    return row
                if None.endswith('.txt'):
                    data.decode('utf-8')
        return row
    except (Pr85BundleError, UnicodeDecodeError, json.JSONDecodeError):
        exc = None
        row['status'] = 'failed'
        row['blocker'] = f'''{exc.__class__.__name__}: {exc}'''
        exc = None
        del exc
        return row
        exc = None
        del exc



def _runtime_source_payload_scan(runtime_root = None, archive_bytes = None):
    source_bytes = 0
    violations = []
    files = []
    for path in sorted(runtime_root.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(runtime_root).as_posix()
        if (lambda .0: pass# WARNING: Decompyle incomplete
)(Path(rel).parts()):
            violations.append(f'''hidden runtime file: {rel}''')
            continue
        if path.suffix.lower() not in frozenset({'.sh', '.py'}):
            continue
        raw = path.read_bytes()
        source_bytes += len(raw)
        file_row = {
            'path': rel,
            'bytes': len(raw),
            'sha256': _sha256_bytes(raw) }
        if path.suffix.lower() == '.py':
            text = raw.decode('utf-8', errors = 'ignore')
            if SOURCE_EMBEDDED_PAYLOAD_LITERAL_RE.search(text):
                violations.append(f'''{rel} contains a >=64KiB encoded/decompressed literal''')
                file_row['large_encoded_payload_literal'] = True
        files.append(file_row)
    if archive_bytes <= 1024 and source_bytes > 65536:
        violations.append(f'''archive is {archive_bytes} bytes but runtime .py/.sh source is {source_bytes} bytes''')
    if not violations:
        return {
            'runtime_root': _repo_rel(runtime_root),
            'source_bytes_py_sh': source_bytes,
            'files': files,
            'violations': violations,
            'status': 'passed' }
    return {
        'runtime_root': sorted(runtime_root.rglob('*')),
        'source_bytes_py_sh': _repo_rel(runtime_root),
        'files': source_bytes,
        'violations': files,
        'status': violations }


def _archive_report(archive = None):
    blockers = []
    if not archive.is_file():
        return ({
            'path': _repo_rel(archive),
            'status': 'failed',
            'error': 'archive_missing' }, [
            {
                'code': 'archive_missing',
                'detail': str(archive) }])
    report = {
        'path': None(archive),
        'bytes': archive.stat().st_size,
        'sha256': _sha256_file(archive),
        'status': 'passed' }
# WARNING: Decompyle incomplete


def build_preflight(archive = None, inflate_sh = None, *, upstream_dir, expected_archive_sha256, expected_archive_size_bytes, expected_runtime_tree_sha256):
    archive = Path(archive)
    inflate_sh = Path(inflate_sh)
    upstream_dir = Path(upstream_dir)
    (archive_report, blockers) = _archive_report(archive)
    if expected_archive_sha256:
        actual = archive_report.get('sha256')
        if actual != expected_archive_sha256:
            blockers.append({
                'code': 'expected_archive_sha256_matches',
                'detail': f'''expected={expected_archive_sha256} actual={actual}''' })
# WARNING: Decompyle incomplete


def build_parser():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--archive', type = Path, required = True)
    parser.add_argument('--inflate-sh', type = Path, required = True)
    parser.add_argument('--upstream-dir', type = Path, default = REPO_ROOT / 'upstream')
    parser.add_argument('--expected-archive-sha256', default = None)
    parser.add_argument('--expected-archive-size-bytes', type = int, default = None)
    parser.add_argument('--expected-runtime-tree-sha256', default = None)
    parser.add_argument('--json-out', type = Path, default = None)
    parser.add_argument('--fail-if-not-ready', action = 'store_true')
    return parser


def main(argv = None):
    args = build_parser().parse_args(argv)
    payload = build_preflight(args.archive, args.inflate_sh, upstream_dir = args.upstream_dir, expected_archive_sha256 = args.expected_archive_sha256, expected_archive_size_bytes = args.expected_archive_size_bytes, expected_runtime_tree_sha256 = args.expected_runtime_tree_sha256)
    text = _json_text(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents = True, exist_ok = True)
        args.json_out.write_text(text, encoding = 'utf-8')
    print(text, end = '')
    if not args.fail_if_not_ready and payload['ready_for_exact_eval_dispatch']:
        return 2

if __name__ == '__main__':
    raise SystemExit(main())

"""
