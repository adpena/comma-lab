"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``51:9: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``pre_submission_compliance_check.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'scripts/pre_submission_compliance_check.py'
__recovery_spec__ = 'pre_submission_compliance_check.recovery_spec.json'
__recovery_ast_error__ = '51:9: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: pre_submission_compliance_check.cpython-312.pyc (Python 3.12)

'''Strict pre-submission compliance gate for contest release packets.

This script is intentionally provider-agnostic. It validates the exact files
that would be uploaded or published, records deterministic expectations, and
fails closed on archive/runtime/auth-eval custody gaps. It does not run a
scorer and does not make a new score claim.
'''
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from tac.public_submission_refs import parse_public_pr_refs_csv
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
ORIGINAL_VIDEO_BYTES = 37545489
SCHEMA = 'pre_submission_compliance_check_v1'
TOOL = 'scripts/pre_submission_compliance_check.py'
DEFAULT_REQUIRED_FILES = ('archive.zip', 'inflate.sh', 'report.txt')
SECRET_SCAN_SUFFIXES = {
    '.md',
    '.py',
    '.sh',
    '.txt',
    '.yml',
    '.json',
    '.toml',
    '.yaml'}
SHA256_RE = re.compile('^[0-9a-fA-F]{64}$')
PACKED_PAYLOAD_MEMBER_NAMES = ('p', 'renderer_payload.bin', 'renderer_payload.bin.br')
TERMINAL_DISPATCH_STATUS_PREFIXES = ('completed', 'failed', 'stopped', 'refused_dispatch', 'stale_superseded')

class ComplianceError(RuntimeError):
    '''Raised for malformed inputs before a report can be built.'''
    pass

Check = <NODE:12>()

def _sha256(path = None):
    pass
# WARNING: Decompyle incomplete


def _rel(path = None, root = None):
    
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return 



def _check(checks = None, name = None, passed = None, details = None, *, severity):
    checks.append(Check(name = name, passed = bool(passed), details = details, severity = severity))


def _read_json(path = None):
    
    try:
        payload = json.loads(path.read_text(encoding = 'utf-8'))
        if not isinstance(payload, dict):
            raise ComplianceError(f'''{path} must contain a JSON object''')
        return payload
    except json.JSONDecodeError:
        exc = None
        raise ComplianceError(f'''{path} is not valid JSON: {exc}'''), exc
        exc = None
        del exc



def _score(seg_dist = None, pose_dist = None, archive_bytes = None):
    return 100 * seg_dist + math.sqrt(10 * pose_dist) + 25 * archive_bytes / ORIGINAL_VIDEO_BYTES


def _unsafe_zip_name(name = None):
    if not name:
        return 'empty_member_name'
    if '\\' in name:
        return 'backslash_member_name'
    if '\x00' in name or (lambda .0: pass# WARNING: Decompyle incomplete
)(name()):
        return 'control_character_member_name'
    if re.match('^[A-Za-z]:', name):
        return 'windows_drive_member_name'
    member = PurePosixPath(name)
    if member.is_absolute() or '..' in member.parts:
        return 'zip_slip_member_name'
    if '__MACOSX' in member.parts:
        return 'macosx_resource_directory'
    base = member.name
    if base in frozenset({'.DS_Store', 'Thumbs.db'}) or (lambda .0: pass# WARNING: Decompyle incomplete
)(member.parts()):
        return 'resource_fork_or_hidden_sidecar'
    if (lambda .0: pass# WARNING: Decompyle incomplete
)(member.parts()):
        return 'hidden_sidecar_member_name'


def _decode_zip_name(raw = None, flag_bits = None):
    encoding = 'utf-8' if flag_bits & 2048 else 'cp437'
    
    try:
        return raw.decode(encoding, errors = 'strict')
    except UnicodeDecodeError:
        return None



def _local_header_name(path = None, info = None):
    handle = path.open('rb')
    handle.seek(info.header_offset)
    header = handle.read(30)
    if len(header) != 30 or header[:4] != b'PK\x03\x04':
        None(None, None)
        return (None, None)
    local_flag_bits = struct.unpack_from('<H', header, 6)[0]
    (name_len, extra_len) = struct.unpack_from('<HH', header, 26)
    raw_name = handle.read(name_len)
    _ = handle.read(extra_len)
    None(None, None)
# WARNING: Decompyle incomplete


def inspect_archive(path = None, *, expect_single_member):
    checks = []
    record = {
        'path': _rel(path),
        'exists': path.is_file(),
        'bytes': path.stat().st_size if path.is_file() else None,
        'sha256': _sha256(path) if path.is_file() else None,
        'members': [] }
    _check(checks, 'archive_exists', path.is_file(), f'''archive={_rel(path)}''')
    if not path.is_file():
        return (record, checks)
# WARNING: Decompyle incomplete


def _json_sha_candidates(payload = None):
    archive = payload.get('archive') if isinstance(payload.get('archive'), dict) else { }
    frontier = payload.get('frontier_summary') if isinstance(payload.get('frontier_summary'), dict) else { }
    return {
        'archive_sha256': payload.get('archive_sha256'),
        'sha256': payload.get('sha256'),
        'archive.sha256': archive.get('sha256'),
        'archive.archive_sha256': archive.get('archive_sha256'),
        'frontier_summary.archive_sha256': frontier.get('archive_sha256') }


def _json_size_candidates(payload = None):
    archive = payload.get('archive') if isinstance(payload.get('archive'), dict) else { }
    frontier = payload.get('frontier_summary') if isinstance(payload.get('frontier_summary'), dict) else { }
    return {
        'archive_size_bytes': payload.get('archive_size_bytes'),
        'size_bytes': payload.get('size_bytes'),
        'bytes': payload.get('bytes'),
        'archive.archive_size_bytes': archive.get('archive_size_bytes'),
        'archive.size_bytes': archive.get('size_bytes'),
        'archive.bytes': archive.get('bytes'),
        'frontier_summary.archive_size_bytes': frontier.get('archive_size_bytes') }


def _first_present(mapping = None):
    pass
# WARNING: Decompyle incomplete


def _manifest_member_rows(payload = None):
    rows = payload.get('members')
    if not isinstance(rows, list):
        archive = payload.get('archive') if isinstance(payload.get('archive'), dict) else { }
        rows = archive.get('members')
    if not isinstance(rows, list):
        return []
# WARNING: Decompyle incomplete


def inspect_archive_manifest(path = None, archive = None):
    pass
# WARNING: Decompyle incomplete


def inspect_submission_dir(submission_dir = None, required_files = None):
    checks = []
    files = { }
    for rel in required_files:
        path = submission_dir / rel
        exists = path.is_file()
        files[rel] = {
            'path': _rel(path),
            'exists': exists,
            'bytes': path.stat().st_size if exists else None,
            'sha256': _sha256(path) if exists else None }
        _check(checks, f'''required_file_present:{rel}''', exists, _rel(path))
    inflate = submission_dir / 'inflate.sh'
    if inflate.is_file():
        mode = inflate.stat().st_mode
        executable = bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
        _check(checks, 'inflate_sh_executable', executable, f'''mode={oct(stat.S_IMODE(mode))}''')
    return ({
        'path': _rel(submission_dir),
        'required_files': files }, checks)


def _runtime_tree_candidates_from_auth(payload = None):
    candidates = { }
    provenance = payload.get('provenance')
    if isinstance(provenance, dict):
        runtime = provenance.get('inflate_runtime_manifest')
        if isinstance(runtime, dict) and isinstance(runtime.get('runtime_tree_sha256'), str):
            candidates['provenance.inflate_runtime_manifest.runtime_tree_sha256'] = runtime['runtime_tree_sha256']
        value = provenance.get('runtime_tree_sha256')
        if isinstance(value, str):
            candidates['provenance.runtime_tree_sha256'] = value
    runtime = payload.get('inflate_runtime_manifest')
    if isinstance(runtime, dict) and isinstance(runtime.get('runtime_tree_sha256'), str):
        candidates['inflate_runtime_manifest.runtime_tree_sha256'] = runtime['runtime_tree_sha256']
    return candidates


def _runtime_file_manifest_candidates_from_auth(payload = None):
    candidates = { }
    provenance = payload.get('provenance')
# WARNING: Decompyle incomplete


def _runtime_tree_from_auth(payload = None):
    candidates = _runtime_tree_candidates_from_auth(payload)
    for key in ('provenance.inflate_runtime_manifest.runtime_tree_sha256', 'provenance.runtime_tree_sha256', 'inflate_runtime_manifest.runtime_tree_sha256'):
        if not key in candidates:
            continue
        
        return ('provenance.inflate_runtime_manifest.runtime_tree_sha256', 'provenance.runtime_tree_sha256', 'inflate_runtime_manifest.runtime_tree_sha256'), candidates[key]


def _runtime_files_from_auth(payload = None):
    candidates = _runtime_file_manifest_candidates_from_auth(payload)
    for key in ('provenance.inflate_runtime_manifest.files', 'inflate_runtime_manifest.files'):
        if not key in candidates:
            continue
        
        return ('provenance.inflate_runtime_manifest.files', 'inflate_runtime_manifest.files'), candidates[key]
    return []


def _runtime_file_identity(row = None):
    return (row.get('relative_path'), row.get('bytes'), row.get('sha256'))


def _runtime_file_manifest_values_consistent(candidates = None):
    pass
# WARNING: Decompyle incomplete


def _looks_like_sha256(value = None):
    if isinstance(value, str):
        isinstance(value, str)
    return bool(SHA256_RE.fullmatch(value))


def _present_consistent(left = None, right = None):
    if not left is None:
        left is None
        if not right is None:
            right is None
    return left == right


def _is_nonnegative_int(value = None):
    if type(value) is int:
        type(value) is int
    return value >= 0


def inspect_auth_eval(path = None, *, archive, expected_samples, require_t4_equivalent, expected_runtime_tree_sha256, require_submission_runtime_match, submission_dir):
    pass
# WARNING: Decompyle incomplete


def _iter_scan_files(paths = None):
    out = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in SECRET_SCAN_SUFFIXES:
            out.append(path)
            continue
        if not path.is_dir():
            continue
        for child in path.rglob('*'):
            if not child.is_file():
                continue
            if not child.suffix.lower() in SECRET_SCAN_SUFFIXES:
                continue
            out.append(child)
    return sorted(dict.fromkeys(out))


def run_public_hygiene(paths = None):
    checks = []
    files = _iter_scan_files(paths)
# WARNING: Decompyle incomplete


def inspect_report_linkage(report_path = None, *, archive, auth_eval, expected_lane_id, expected_job_id, require_archive_link, require_auth_link):
    checks = []
    record = {
        'path': _rel(report_path),
        'present': report_path.is_file() }
    _check(checks, 'report_txt_present', report_path.is_file(), _rel(report_path))
    if not report_path.is_file():
        return (record, checks)
    text = None.read_text(encoding = 'utf-8', errors = 'replace')
    archive_sha = archive.get('sha256')
    archive_bytes = archive.get('bytes')
    if isinstance(archive_sha, str):
        isinstance(archive_sha, str)
    contains_archive_sha = archive_sha in text
    if archive_bytes is not None:
        archive_bytes is not None
    contains_archive_bytes = str(archive_bytes) in text
    if expected_lane_id:
        expected_lane_id
    if expected_job_id:
        expected_job_id
    record.update({
        'contains_archive_sha256': contains_archive_sha,
        'contains_archive_size_bytes': contains_archive_bytes,
        'contains_expected_lane_id': bool(expected_lane_id in text),
        'contains_expected_job_id': bool(expected_job_id in text) })
    if require_archive_link:
        _check(checks, 'report_links_exact_archive_sha256', contains_archive_sha, 'report.txt must include the exact archive SHA-256')
        _check(checks, 'report_links_exact_archive_size_bytes', contains_archive_bytes, 'report.txt must include the exact archive byte size')
# WARNING: Decompyle incomplete


def inspect_dispatch_claim_linkage(claims_path = None, *, expected_lane_id, expected_job_id):
    pass
# WARNING: Decompyle incomplete


def build_report(args = None):
    submission_dir = args.submission_dir.resolve()
    if not args.archive:
        args.archive
    archive_path = (submission_dir / 'archive.zip').resolve()
    if args.contest_final:
        args.require_auth_eval = True
        args.require_t4_equivalent = True
        if not args.expect_single_member:
            args.expect_single_member
        args.expect_single_member = 'x'
        args.require_archive_manifest = True
        args.require_report_archive_link = True
        args.require_submission_runtime_match = True
        if not args.archive_manifest_json:
            args.archive_manifest_json
        args.archive_manifest_json = submission_dir / 'archive_manifest.json'
        if not args.public_scan_path:
            args.public_scan_path = [
                submission_dir]
    if not args.required_file:
        args.required_file
    required_files = tuple(DEFAULT_REQUIRED_FILES)
    all_checks = []
    (submission, checks) = inspect_submission_dir(submission_dir, required_files)
    all_checks.extend(checks)
    (archive, checks) = inspect_archive(archive_path, expect_single_member = args.expect_single_member)
    all_checks.extend(checks)
    archive_manifest = None
    if args.archive_manifest_json or args.require_archive_manifest:
        (archive_manifest, checks) = inspect_archive_manifest(args.archive_manifest_json, archive)
        all_checks.extend(checks)
    submission_archive = submission['required_files'].get('archive.zip')
    if submission_archive and submission_archive.get('exists') and archive.get('exists'):
        if submission_archive.get('sha256') == archive.get('sha256'):
            submission_archive.get('sha256') == archive.get('sha256')
        _check(all_checks, 'submission_archive_matches_inspected_archive', submission_archive.get('bytes') == archive.get('bytes'), f'''submission_sha={submission_archive.get('sha256')} submission_bytes={submission_archive.get('bytes')} inspected_sha={archive.get('sha256')} inspected_bytes={archive.get('bytes')}''')
    if args.expected_archive_sha256:
        _check(all_checks, 'expected_archive_sha256_format', _looks_like_sha256(args.expected_archive_sha256), f'''expected_archive_sha256={args.expected_archive_sha256!r}''')
        _check(all_checks, 'expected_archive_sha256', archive.get('sha256') == args.expected_archive_sha256, f'''expected={args.expected_archive_sha256} observed={archive.get('sha256')}''')
    elif args.contest_final:
        _check(all_checks, 'contest_final_expected_archive_sha256_present', False, '--contest-final requires --expected-archive-sha256')
# WARNING: Decompyle incomplete


def build_arg_parser():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--submission-dir', type = Path, required = True)
    parser.add_argument('--archive', type = Path, default = None)
    parser.add_argument('--auth-eval-json', type = Path, default = None)
    parser.add_argument('--require-auth-eval', action = 'store_true')
    parser.add_argument('--require-t4-equivalent', action = 'store_true')
    parser.add_argument('--expected-archive-sha256', default = None)
    parser.add_argument('--expected-archive-size-bytes', type = int, default = None)
    parser.add_argument('--expected-samples', type = int, default = 600)
    parser.add_argument('--expected-runtime-tree-sha256', default = None)
    parser.add_argument('--require-submission-runtime-match', action = 'store_true', help = 'Require auth-eval runtime manifest files to match files in --submission-dir.')
    parser.add_argument('--expect-single-member', default = None)
    parser.add_argument('--archive-manifest-json', type = Path, default = None)
    parser.add_argument('--require-archive-manifest', action = 'store_true')
    parser.add_argument('--require-report-archive-link', action = 'store_true')
    parser.add_argument('--require-report-auth-score-link', action = 'store_true')
    parser.add_argument('--dispatch-claims-md', type = Path, default = None)
    parser.add_argument('--expected-lane-id', default = None)
    parser.add_argument('--expected-job-id', default = None)
    parser.add_argument('--contest-final', action = 'store_true', help = 'Enable final-submission strict mode: require auth eval, T4/equivalent custody, single member x, public hygiene scan, archive manifest, report archive linkage, submission-runtime match, and explicit expected archive SHA/bytes.')
    parser.add_argument('--required-file', action = 'append', default = None)
    parser.add_argument('--public-scan-path', action = 'append', type = Path, default = [])
    parser.add_argument('--source-prs', default = None, help = 'Comma-separated public PR refs used as provenance signal, e.g. PR85,PR91.')
    parser.add_argument('--output-json', type = Path, default = None)
    return parser


def main(argv = None):
    args = build_arg_parser().parse_args(argv)
    report = build_report(args)
    text = json.dumps(report, indent = 2, sort_keys = True, allow_nan = False) + '\n'
# WARNING: Decompyle incomplete

if __name__ == '__main__':
    raise SystemExit(main())

"""
