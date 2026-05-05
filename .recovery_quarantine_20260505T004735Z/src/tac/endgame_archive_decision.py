"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``37:18: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``endgame_archive_decision.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'src/tac/endgame_archive_decision.py'
__recovery_spec__ = 'endgame_archive_decision.recovery_spec.json'
__recovery_ast_error__ = '37:18: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: endgame_archive_decision.cpython-312.pyc (Python 3.12)

'''Byte-level endgame decision support for PR85-family archives.

This module is local and deterministic. It validates ZIP custody, slices
PR85-family single-member bundles, probes cheap codec contracts, and estimates
rate-only transplant deltas against a named frontier archive. It never inflates
contest videos, loads scorers, dispatches jobs, or makes score claims.
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
from tac.archive_byte_profile import contest_rate_term
from tac.pr85_bundle import HPM1_MAGIC, PR85_HEADERLESS_RANDMULTI_SPECS, Pr85BundleError, SEGMENT_ORDER, parse_hpm1_mask_segment, parse_pr85_bundle
SCHEMA = 'endgame_archive_decision_profile_v1'
TOOL = 'experiments/profile_endgame_archive_decision.py'
EVIDENCE_GRADE = 'byte_level_decision_support_only'
LOCAL_FILE_HEADER = 67324752
QMA9_HEADER_BYTES = 20
STBM1BR_MAGIC = b'STBM1BR\x00'
RMB1_MAGIC = b'RMB1'
RSB1_MAGIC = b'RSB1'

class EndgameArchiveDecisionError(ValueError):
    '''Raised when endgame archive profiling inputs are malformed.'''
    pass

_MemberPayload = <NODE:12>()
_ArchiveProfile = <NODE:12>()

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
    total = len(data)
    entropy = 0
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
        raise EndgameArchiveDecisionError(f'''truncated local header for {info.filename!r} at {info.header_offset}''')
    (signature, _version_needed, _flag_bits, _compress_type, _mod_time, _mod_date, _crc, _compress_size, _file_size, name_len, extra_len) = struct.unpack('<IHHHHHIIIHH', header)
    if signature != LOCAL_FILE_HEADER:
        raise EndgameArchiveDecisionError(f'''bad local header signature for {info.filename!r} at {info.header_offset}''')
    raw_name = handle.read(name_len)
    if len(raw_name) != name_len:
        raise EndgameArchiveDecisionError(f'''truncated local filename for {info.filename!r}''')
    handle.seek(extra_len, 1)
    None(None, None)
# WARNING: Decompyle incomplete


def _import_brotli():
    
    try:
        import brotli
        return brotli
    except ImportError:
        exc = None
        raise EndgameArchiveDecisionError('brotli package is required for codec validation'), exc
        exc = None
        del exc



def _brotli_probe(data = None, *, expected_magics, decoded_size):
    pass
# WARNING: Decompyle incomplete


def _qma9_probe(data = None):
    if not len(data) < QMA9_HEADER_BYTES or data.startswith(b'QMA9'):
        return {
            'status': 'failed',
            'blockers': [
                'bad_qma9_magic_or_header'] }
    (frames, width, height, bitstream_bytes) = None.unpack_from('<IIII', data, 4)
    payload_bytes = len(data) - QMA9_HEADER_BYTES
    blockers = []
    if frames <= 0 and width <= 0 or height <= 0:
        blockers.append('nonpositive_qma9_shape')
    if bitstream_bytes != payload_bytes:
        blockers.append('qma9_bitstream_length_mismatch')
    return {
        'status': 'ok' if not blockers else 'failed',
        'frames': int(frames),
        'width': int(width),
        'height': int(height),
        'header_bytes': QMA9_HEADER_BYTES,
        'declared_bitstream_bytes': int(bitstream_bytes),
        'actual_bitstream_bytes': int(payload_bytes),
        'bitstream_sha256': _sha256_bytes(data[QMA9_HEADER_BYTES:]),
        'blockers': blockers }


def _hpm1_probe(data = None):
    pass
# WARNING: Decompyle incomplete


def _stbm1br_probe(data = None):
    pass
# WARNING: Decompyle incomplete


def _parse_rmb1(data = None):
    if not len(data) < 6 or data.startswith(RMB1_MAGIC):
        return {
            'status': 'failed',
            'blockers': [
                'bad_rmb1_magic_or_header'] }
    mask_len = None.from_bytes(data[4:6], 'little')
    if 6 + mask_len > len(data):
        return {
            'status': 'failed',
            'blockers': [
                'rmb1_mask_brotli_overruns_segment'] }
    mask_br = None[6:6 + mask_len]
    vals_br = data[6 + mask_len:]
# WARNING: Decompyle incomplete


def _parse_rsb1(data = None):
    if not len(data) < 8 or data.startswith(RSB1_MAGIC):
        return {
            'status': 'failed',
            'blockers': [
                'bad_rsb1_magic_or_header'] }
    count = None.from_bytes(data[4:6], 'little')
    table_id = int(data[6])
    reserved = int(data[7])
# WARNING: Decompyle incomplete


def _codec_label(segment_name = None, data = None):
    if segment_name == 'mask':
        if data.startswith(b'QMA9'):
            return 'QMA9_range_mask'
        if data.startswith(STBM1BR_MAGIC):
            return 'STBM1BR_lossless_mask_recode'
        if data.startswith(HPM1_MAGIC):
            return 'HPM1_hpac_mask'
    if segment_name == 'randmulti' and data.startswith(RMB1_MAGIC):
        return 'RMB1_bitmask_randmulti'
    if segment_name == 'model':
        return 'brotli_qh_model'
    if segment_name == 'pose':
        return 'brotli_p1d1_pose'
    if segment_name in frozenset({'bias', 'frac', 'post', 'frac2', 'frac3', 'shift', 'region', 'randmulti'}):
        return 'brotli_pr85_sidechannel'
    return 'opaque'


def _segment_validation(name = None, data = None):
    if name == 'mask' and data.startswith(b'QMA9'):
        return _qma9_probe(data)
    if None == 'mask' and data.startswith(HPM1_MAGIC):
        return _hpm1_probe(data)
    if None == 'mask' and data.startswith(STBM1BR_MAGIC):
        return _stbm1br_probe(data)
    if None == 'model':
        return _brotli_probe(data, expected_magics = (b'QH0', b'QH1'))
    if None == 'pose':
        return _brotli_probe(data, expected_magics = (b'P1D1',))
    if None == 'post':
        return _brotli_probe(data, decoded_size = 2400)
    if None == 'shift':
        return _brotli_probe(data, expected_magics = (b'SD4',))
    if None == 'frac':
        return _brotli_probe(data, expected_magics = (b'FV1',))
    if None == 'frac2':
        return _brotli_probe(data, expected_magics = (b'FH2',))
    if None == 'frac3':
        return _brotli_probe(data, expected_magics = (b'FD3',))
    if None == 'bias':
        return _brotli_probe(data, expected_magics = (b'BD1',))
    if None == 'region':
        return _brotli_probe(data, expected_magics = (b'RH1',))
    if None == 'randmulti' and data.startswith(RMB1_MAGIC):
        return _parse_rmb1(data)
    if None == 'randmulti':
        return _brotli_probe(data)
    if None.startswith(RSB1_MAGIC):
        return _parse_rsb1(data)
    return {
        'status': None,
        'reason': 'no cheap local contract for payload' }


def _segment_row(name = None, data = None, offset = None):
    validation = _segment_validation(name, data)
    return {
        'name': name,
        'bytes': len(data),
        'sha256': _sha256_bytes(data),
        'offset': int(offset),
        'magic_hex': data[:8].hex(),
        'magic_ascii': _display_ascii(data),
        'entropy_bits_per_byte': _byte_entropy(data),
        'codec': _codec_label(name, data),
        'validation': validation }


def _side_member_row(member = None):
    data = member.data
    validation = _segment_validation(member.name, data)
    codec = 'RSB1_router_side_actions' if data.startswith(RSB1_MAGIC) else 'opaque_side_member'
    return {
        'name': member.name,
        'occurrence': member.occurrence,
        'bytes': len(data),
        'sha256': _sha256_bytes(data),
        'magic_hex': data[:8].hex(),
        'magic_ascii': _display_ascii(data),
        'entropy_bits_per_byte': _byte_entropy(data),
        'codec': codec,
        'validation': validation }


def _member_payloads(path = None):
    pass
# WARNING: Decompyle incomplete


def _try_primary_bundle(member = None):
    pass
# WARNING: Decompyle incomplete


def _profile_one_archive(path = None, *, label):
    if not path.is_file():
        raise FileNotFoundError(f'''archive not found: {path}''')
# WARNING: Decompyle incomplete


def _segment_delta_rows(candidate = None, frontier = None):
    rows = []
# WARNING: Decompyle incomplete


def _comparison(candidate = None, frontier = None):
    cand_report = candidate.report
    front_report = frontier.report
    cand_archive_bytes = int(cand_report['archive']['bytes'])
    front_archive_bytes = int(front_report['archive']['bytes'])
    archive_delta = cand_archive_bytes - front_archive_bytes
    cand_primary = int(cand_report['byte_accounting']['primary_member_bytes'])
    front_primary = int(front_report['byte_accounting']['primary_member_bytes'])
    cand_side = int(cand_report['side_info']['charged_bytes'])
    front_side = int(front_report['side_info']['charged_bytes'])
    cand_overhead = int(cand_report['byte_accounting']['archive_zip_overhead_bytes'])
    front_overhead = int(front_report['byte_accounting']['archive_zip_overhead_bytes'])
    primary_delta = cand_primary - front_primary
    side_delta = cand_side - front_side
    overhead_delta = cand_overhead - front_overhead
    segment_rows = _segment_delta_rows(candidate.segments_by_name, frontier.segments_by_name)
# WARNING: Decompyle incomplete


def _rank_actions(comparisons = None):
    actions = []
    for comp in comparisons:
        for estimate in comp['transplant_estimates']:
            if estimate['byte_positive']:
                actions.append({
                    'candidate_label': comp['candidate_label'],
                    'frontier_label': comp['frontier_label'],
                    'surface': estimate['segment'],
                    'estimated_archive_delta_bytes': estimate['estimated_archive_delta_bytes'],
                    'estimated_rate_term_delta': estimate['estimated_rate_term_delta'],
                    'blocked_by': estimate['runtime_blockers'],
                    'advice': estimate['dispatch_advice'] })
                continue
            if not estimate['requires_candidate_side_info']:
                continue
            None({
                'candidate_label': actions.append,
                'frontier_label': comp['candidate_label'],
                'surface': comp['frontier_label'],
                'estimated_archive_delta_bytes': f'''{estimate['segment']}+side_info''',
                'estimated_rate_term_delta': estimate['estimated_archive_delta_bytes'],
                'blocked_by': estimate['estimated_rate_term_delta'],
                'advice': estimate['dispatch_advice'] })
    actions.sort(key = (lambda row: (row['estimated_archive_delta_bytes'] >= 0, int(row['estimated_archive_delta_bytes']), row['candidate_label'], row['surface'])))
    return actions


def _parse_label_path(value = None):
    if '=' not in value:
        raise EndgameArchiveDecisionError(f'''expected LABEL=PATH, got {value!r}''')
    (label, raw_path) = value.split('=', 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or raw_path:
        raise EndgameArchiveDecisionError(f'''expected LABEL=PATH, got {value!r}''')
    return (label, Path(raw_path))


def build_endgame_decision_profile(candidates = None, *, frontier_label):
    '''Build a deterministic byte-level decision profile for candidate archives.'''
    pass
# WARNING: Decompyle incomplete


def render_markdown(profile = None):
    lines = [
        '# Endgame Archive Decision Profile',
        '',
        f'''- schema: `{profile['schema']}`''',
        f'''- evidence_grade: `{profile['evidence_grade']}`''',
        f'''- score_claim: `{profile['score_claim']}`''',
        f'''- frontier_label: `{profile['frontier_label']}`''',
        '',
        'This report is byte-only. It does not inflate videos, load scorers, dispatch jobs, promote methods, or claim contest score.',
        '',
        '## Archives',
        '',
        '| label | bytes | sha256 | strict ZIP | decision-valid | side bytes |',
        '|---|---:|---|---|---|---:|']
    for archive in profile['archives']:
        lines.append(f'''| {archive['label']} | {archive['archive']['bytes']} | {archive['archive']['sha256']} | {archive['strict_zip']['valid']} | {archive['decision_support']['valid_for_byte_decision']} | {archive['side_info']['charged_bytes']} |''')
    for archive in profile['archives']:
        primary = archive.get('primary_member')
        if not primary:
            continue
        lines.extend([
            '',
            f'''## {archive['label']} Segments''',
            '',
            '| segment | bytes | codec | validation | sha256 |',
            '|---|---:|---|---|---|'])
        for segment in primary['segments']:
            lines.append(f'''| {segment['name']} | {segment['bytes']} | {segment['codec']} | {segment['validation'].get('status')} | {segment['sha256']} |''')
        if not archive['side_info']['members']:
            continue
        lines.extend([
            '',
            '### Side Info',
            '',
            '| member | bytes | codec | validation | sha256 |',
            '|---|---:|---|---|---|'])
        for member in archive['side_info']['members']:
            lines.append(f'''| {member['name']} | {member['bytes']} | {member['codec']} | {member['validation'].get('status')} | {member['sha256']} |''')
    if profile['comparisons_to_frontier']:
        lines.extend([
            '',
            '## Frontier Comparisons',
            ''])
        for comp in profile['comparisons_to_frontier']:
            lines.extend([
                f'''### {comp['candidate_label']} vs {comp['frontier_label']}''',
                '',
                f'''- archive delta bytes: `{comp['archive_delta_bytes']}`''',
                f'''- primary member delta bytes: `{comp['primary_member_delta_bytes']}`''',
                f'''- side-info delta bytes: `{comp['side_info_delta_bytes']}`''',
                f'''- ZIP overhead delta bytes: `{comp['zip_overhead_delta_bytes']}`''',
                '',
                '| segment | delta bytes | frontier codec | candidate codec | validation |',
                '|---|---:|---|---|---|'])
            for row in comp['changed_segments']:
                lines.append(f'''| {row['segment']} | {row['delta_bytes']} | {row['frontier_codec']} | {row['candidate_codec']} | {row['candidate_validation_status']} |''')
            if not comp['transplant_estimates']:
                continue
            lines.extend([
                '',
                '#### Transplant Estimates',
                '',
                '| surface | est. delta bytes | side info | advice |',
                '|---|---:|---|---|'])
            for estimate in comp['transplant_estimates']:
                lines.append(f'''| {estimate['segment']} | {estimate['estimated_archive_delta_bytes']} | {estimate['requires_candidate_side_info']} | {estimate['dispatch_advice']} |''')
    if profile['ranked_actions']:
        lines.extend([
            '',
            '## Ranked Actions',
            '',
            '| candidate | surface | est. delta bytes | advice | blockers |',
            '|---|---|---:|---|---|'])
        for action in profile['ranked_actions']:
            blockers = ', '.join(action['blocked_by'])
            lines.append(f'''| {action['candidate_label']} | {action['surface']} | {action['estimated_archive_delta_bytes']} | {action['advice']} | {blockers} |''')
    return '\n'.join(lines) + '\n'


def write_outputs(profile = None, *, json_out, markdown_out):
    pass
# WARNING: Decompyle incomplete


def build_arg_parser():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--candidate', action = 'append', default = [], metavar = 'LABEL=PATH', help = 'Candidate archive to profile. May be repeated.')
    parser.add_argument('--frontier-label', help = 'Label to use as the byte frontier for comparison. Defaults to the smallest archive.')
    parser.add_argument('--json-out', type = Path)
    parser.add_argument('--markdown-out', type = Path)
    return parser


def main(argv = None):
    args = build_arg_parser().parse_args(argv)
    candidates = (lambda .0: pass# WARNING: Decompyle incomplete
)(args.candidate())
    profile = build_endgame_decision_profile(candidates, frontier_label = args.frontier_label)
    write_outputs(profile, json_out = args.json_out, markdown_out = args.markdown_out)
# WARNING: Decompyle incomplete

__all__ = [
    'EndgameArchiveDecisionError',
    'build_endgame_decision_profile',
    'render_markdown',
    'write_outputs',
    'main']

"""
