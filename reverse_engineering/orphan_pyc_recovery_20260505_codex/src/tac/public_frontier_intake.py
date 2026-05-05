"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``33:18: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``public_frontier_intake.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'src/tac/public_frontier_intake.py'
__recovery_spec__ = 'public_frontier_intake.recovery_spec.json'
__recovery_ast_error__ = '33:18: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: public_frontier_intake.cpython-312.pyc (Python 3.12)

'''Static intake and byte diffing for public frontier archives.

This module is intentionally offline and byte-only. It validates ZIP custody,
identifies PR85-family bundle segments, records charged side-info members, and
diffs segment identities against named baselines. It never inflates contest
videos, loads scorers, submits jobs, or makes score claims.
'''
from __future__ import annotations
import argparse
import hashlib
import json
import math
import struct
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence
from tac.archive_byte_profile import CONTEST_ORIGINAL_BYTES, contest_rate_term
from tac.pr85_bundle import Pr85BundleError, parse_pr85_bundle
SCHEMA = 'public_frontier_archive_intake_v1'
TOOL = 'experiments/profile_public_frontier_intake.py'
EVIDENCE_GRADE = 'external_archive_byte_intake_only'
LOCAL_FILE_HEADER = 67324752

class PublicFrontierIntakeError(ValueError):
    '''Raised when a public-frontier intake input is malformed.'''
    pass

_ParsedArchive = <NODE:12>()

def _json_text(payload = None):
    return json.dumps(payload, indent = 2, sort_keys = True, allow_nan = False) + '\n'


def _sha256_bytes(data = None):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path = None):
    pass
# WARNING: Decompyle incomplete


def _byte_entropy(data = None):
    if not data:
        return 0
    counts = Counter(data)
    entropy = 0
    total = len(data)
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return round(entropy, 12)


def _display_ascii(data = None, *, limit):
    return (lambda .0: pass# WARNING: Decompyle incomplete
)(data[:limit]())


def _safe_member_blockers(name = None):
    blockers = []
    if not name:
        blockers.append('empty_member_name')
    if '\x00' in name:
        blockers.append('nul_in_member_name')
    if '\\' in name:
        blockers.append('backslash_in_member_name')
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if posix.is_absolute() and windows.is_absolute() or windows.drive:
        blockers.append('absolute_or_drive_member_path')
    if (lambda .0: pass# WARNING: Decompyle incomplete
)(posix.parts()):
        blockers.append('zip_slip_member_path')
    if (lambda .0: pass# WARNING: Decompyle incomplete
)(posix.parts()):
        blockers.append('hidden_or_resource_fork_member')
    return blockers


def _decode_zip_name(raw = None, flag_bits = None):
    encoding = 'utf-8' if flag_bits & 2048 else 'cp437'
    return raw.decode(encoding)


def _local_header_name(path = None, info = None):
    handle = path.open('rb')
    handle.seek(info.header_offset)
    header = handle.read(30)
    if len(header) != 30:
        raise PublicFrontierIntakeError(f'''truncated local header for {info.filename!r} at {info.header_offset}''')
    (signature, _version_needed, _flag_bits, _compress_type, _mod_time, _mod_date, _crc, _compress_size, _file_size, name_len, extra_len) = struct.unpack('<IHHHHHIIIHH', header)
    if signature != LOCAL_FILE_HEADER:
        raise PublicFrontierIntakeError(f'''bad local header signature for {info.filename!r} at {info.header_offset}''')
    raw_name = handle.read(name_len)
    if len(raw_name) != name_len:
        raise PublicFrontierIntakeError(f'''truncated local filename for {info.filename!r}''')
    handle.seek(extra_len, 1)
    None(None, None)
# WARNING: Decompyle incomplete


def _codec_label(segment_name = None, data = None):
    if segment_name == 'mask':
        if data.startswith(b'QMA9'):
            return 'QMA9_range_mask'
        if data.startswith(b'STBM1BR\x00'):
            return 'STBM1BR_lossless_mask_recode'
        if data.startswith(b'HPM1'):
            return 'HPM1_hpac_mask'
    if segment_name == 'randmulti' and data.startswith(b'RMB1'):
        return 'RMB1_side_info_backed_randmulti'
    if segment_name in frozenset({'bias', 'frac', 'pose', 'post', 'frac2', 'frac3', 'model', 'shift', 'region', 'randmulti'}):
        return 'opaque_pr85_segment'
    return 'unknown'


def _segment_row(name = None, data = None, offset = None):
    return {
        'name': name,
        'bytes': len(data),
        'sha256': _sha256_bytes(data),
        'offset': int(offset),
        'magic_hex': data[:8].hex(),
        'magic_ascii': _display_ascii(data),
        'codec': _codec_label(name, data),
        'entropy_bits_per_byte': _byte_entropy(data) }


def _inspect_member_data(name = None, data = None):
    row = {
        'name': name,
        'bytes': len(data),
        'sha256': _sha256_bytes(data),
        'magic_hex': data[:8].hex(),
        'magic_ascii': _display_ascii(data),
        'entropy_bits_per_byte': _byte_entropy(data) }
# WARNING: Decompyle incomplete


def _archive_members(path = None):
    pass
# WARNING: Decompyle incomplete


def _primary_pr85_member(member_data = None):
    candidates = []
# WARNING: Decompyle incomplete


def _diff_against_baselines(primary_segments = None, baselines = None, *, candidate_archive_bytes):
    rows = []
# WARNING: Decompyle incomplete


def _parse_baseline_arg(value = None):
    if '=' not in value:
        raise PublicFrontierIntakeError(f'''baseline must be LABEL=PATH, got {value!r}''')
    (label, path) = value.split('=', 1)
    if not label.strip() or path.strip():
        raise PublicFrontierIntakeError(f'''baseline must be LABEL=PATH, got {value!r}''')
    return (label.strip(), Path(path))


def _analyze_one(path = None, *, label, candidate_bytes):
    if not path.is_file():
        raise FileNotFoundError(f'''archive not found: {path}''')
    archive_bytes = path.stat().st_size
# WARNING: Decompyle incomplete


def profile_public_frontier_archive(archive = None, *, label, baselines):
    '''Build a deterministic public-frontier intake report for one archive.'''
    archive_path = Path(archive)
    candidate = _analyze_one(archive_path, label = label)
    baseline_reports = { }
    if not baselines:
        baselines
    for baseline_label, baseline_path in { }.items():
        baseline_reports[baseline_label] = _analyze_one(Path(baseline_path), label = baseline_label, candidate_bytes = int(candidate.report['archive']['bytes']))
    candidate.report['baseline_diffs'] = _diff_against_baselines(candidate.primary_segments, baseline_reports, candidate_archive_bytes = int(candidate.report['archive']['bytes']))
# WARNING: Decompyle incomplete


def render_markdown(report = None):
    '''Render a compact Markdown view of an intake report.'''
    lines = [
        '# Public Frontier Archive Intake',
        '',
        f'''- label: `{report['label']}`''',
        f'''- evidence_grade: `{report['evidence_grade']}`''',
        f'''- score_claim: `{report['score_claim']}`''',
        f'''- archive bytes: `{report['archive']['bytes']}`''',
        f'''- archive sha256: `{report['archive']['sha256']}`''',
        f'''- strict ZIP valid: `{report['strict_zip']['valid']}`''',
        f'''- side-info bytes: `{report['side_info']['charged_bytes']}`''',
        '',
        'This report is byte-only. It does not inflate videos, load scorers, dispatch jobs, promote methods, or claim a contest score.',
        '',
        '## Members',
        '',
        '| name | bytes | compressed | local name match | sha256 |',
        '|---|---:|---:|---|---|']
    for member in report['strict_zip']['members']:
        if not member['sha256']:
            member['sha256']
        lines.append(f'''| {member['name']} | {member['bytes']} | {member['compressed_bytes']} | {member['central_local_name_match']} | {''} |''')
    if report['strict_zip']['blockers']:
        lines.extend([
            '',
            '## Strict ZIP Blockers',
            ''])
        for blocker in report['strict_zip']['blockers']:
            lines.append(f'''- `{blocker}`''')
    primary = report.get('primary_member')
    if primary:
        lines.extend([
            '',
            '## Primary PR85-Family Bundle',
            '',
            f'''- member: `{primary['name']}`''',
            f'''- format: `{primary['bundle_format']}`''',
            '',
            '| segment | bytes | codec | sha256 |',
            '|---|---:|---|---|'])
        for segment in primary['segments']:
            lines.append(f'''| {segment['name']} | {segment['bytes']} | {segment['codec']} | {segment['sha256']} |''')
    if report['side_info']['members']:
        lines.extend([
            '',
            '## Charged Side Info',
            '',
            '| member | bytes | magic | sha256 |',
            '|---|---:|---|---|'])
        for member in report['side_info']['members']:
            lines.append(f'''| {member['name']} | {member['bytes']} | {member['magic_ascii']} | {member['sha256']} |''')
    if report.get('baseline_diffs'):
        lines.extend([
            '',
            '## Baseline Diffs',
            ''])
        for diff in report['baseline_diffs']:
            lines.extend([
                f'''### {diff['baseline_label']}''',
                '',
                f'''- archive delta bytes: `{diff['candidate_minus_baseline_archive_bytes']}`''',
                f'''- rate score delta: `{diff['candidate_minus_baseline_rate_score_delta']}`''',
                '',
                '| segment | delta bytes | baseline codec | candidate codec | same sha256 |',
                '|---|---:|---|---|---|'])
            for segment in diff['changed_segments']:
                lines.append(f'''| {segment['segment']} | {segment['delta_bytes']} | {segment['baseline_codec']} | {segment['candidate_codec']} | {segment['same_sha256']} |''')
    return '\n'.join(lines) + '\n'


def write_outputs(report = None, *, json_out, markdown_out):
    pass
# WARNING: Decompyle incomplete


def build_arg_parser():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--archive', type = Path, required = True)
    parser.add_argument('--label', default = 'candidate')
    parser.add_argument('--baseline', action = 'append', default = [], help = 'Named baseline as LABEL=PATH. May be repeated.')
    parser.add_argument('--json-out', type = Path)
    parser.add_argument('--markdown-out', type = Path)
    return parser


def main(argv = None):
    args = build_arg_parser().parse_args(argv)
    baselines = (lambda .0: pass# WARNING: Decompyle incomplete
)(args.baseline())
    report = profile_public_frontier_archive(args.archive, label = args.label, baselines = baselines)
    write_outputs(report, json_out = args.json_out, markdown_out = args.markdown_out)
# WARNING: Decompyle incomplete

__all__ = [
    'PublicFrontierIntakeError',
    'profile_public_frontier_archive',
    'render_markdown',
    'write_outputs',
    'main']

"""
