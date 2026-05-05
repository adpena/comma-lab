# pyc-recovery: STUB unreconstructible -- see .recovery_spec.json for dis() ground-truth
# pycdc could not produce parseable output; raw decompiled text preserved in _PYCDC_PARTIAL_OUTPUT below.
"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``42:18: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``profile_pr75_minp_archive.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/profile_pr75_minp_archive.py'
__recovery_spec__ = 'profile_pr75_minp_archive.recovery_spec.json'
__recovery_ast_error__ = '42:18: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: profile_pr75_minp_archive.cpython-312.pyc (Python 3.12)

'''Forensic byte/profile support for the current PR75/minp public archive.

This is a local reverse-engineering tool only. It does not load the scorer,
does not require CUDA, does not dispatch jobs, and never claims score. Its
purpose is to make the current public PR75/minp single-blob grammar explicit so
runtime parity work can be implemented against measured bytes instead of chat
notes.
'''
from __future__ import annotations
import argparse
import hashlib
import importlib.util as importlib
import json
import struct
import subprocess
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import brotli
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = REPO_ROOT / 'experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/archive.zip'
DEFAULT_COMPARE_ARCHIVE = REPO_ROOT / 'experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip'
DEFAULT_OUTPUT_JSON = REPO_ROOT / 'experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/pr75_minp_grammar_profile.json'
DEFAULT_PR75_SOURCE_ROOT = Path('/tmp/pr75-minp')
DEFAULT_UNPACKER = REPO_ROOT / 'submissions/robust_current/unpack_renderer_payload.py'
SCHEMA = 'pr75_minp_archive_grammar_profile_v1'
TOOL = 'experiments/profile_pr75_minp_archive.py'
EVIDENCE_GRADE = 'empirical_reverse_engineering_profile'
MEMBER_NAME = 'p'
MASK_BR_LEN = 219472
SEG_H = 384
SEG_W = 512
SEG_TILE_SIZE = 32
MAX_TILE_ID = (SEG_H // SEG_TILE_SIZE) * (SEG_W // SEG_TILE_SIZE)
PUBLIC_ACTION_COUNT = 108
FixedSlicePlan = <NODE:12>()
FIXED_SLICE_PLANS: 'dict[int, FixedSlicePlan]' = {
    276641: FixedSlicePlan(276641, 'pr75_fixed_qpose14_r55_actions236_model56034', MASK_BR_LEN, 56034, 236),
    276520: FixedSlicePlan(276520, 'pr75_fixed_qpose14_r55_actions236_model55914', MASK_BR_LEN, 55914, 236),
    276381: FixedSlicePlan(276381, 'pr75_minp_fixed_actions255_model55756', MASK_BR_LEN, 55756, 255),
    276379: FixedSlicePlan(276379, 'pr75_minp_fixed_actions253_model55756', MASK_BR_LEN, 55756, 253),
    276362: FixedSlicePlan(276362, 'pr75_minp_fixed_actions236_model55756', MASK_BR_LEN, 55756, 236),
    277247: FixedSlicePlan(277247, 'pr79_minp_s1_split_actions1121_model55756', MASK_BR_LEN, 55756, 1121),
    277288: FixedSlicePlan(277288, 'pr79_minp_v2_fixed_actions1162_model55756', MASK_BR_LEN, 55756, 1162) }

def _sha256_bytes(data = None):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path = None):
    pass
# WARNING: Decompyle incomplete


def _json_bytes(payload = None):
    return (json.dumps(payload, indent = 2, sort_keys = True, allow_nan = False) + '\n').encode('utf-8')


def _read_single_payload_zip(path = None):
    zf = zipfile.ZipFile(path, 'r')
# WARNING: Decompyle incomplete


def _safe_prefix(data = None, n = None):
    return data[:n].hex()


def _stream_summary(name = None, charged = None, decoded = None, codec = ('name', 'str', 'charged', 'bytes', 'decoded', 'bytes', 'codec', 'str', 'return', 'dict[str, Any]')):
    return {
        'name': name,
        'charged_bytes': int(len(charged)),
        'charged_sha256': _sha256_bytes(charged),
        'codec': codec,
        'decoded_bytes': int(len(decoded)),
        'decoded_sha256': _sha256_bytes(decoded),
        'decoded_prefix_hex': _safe_prefix(decoded) }


def fixed_slice_plan_for_payload(payload = None):
    plan = FIXED_SLICE_PLANS.get(len(payload))
# WARNING: Decompyle incomplete


def split_fixed_public_payload(payload = None):
    '''Split the public current-workflow PR75/minp stored-payload wire bytes.'''
    plan = fixed_slice_plan_for_payload(payload)
    cursor = 0
    mask = payload[cursor:cursor + plan.mask_br_bytes]
    cursor += plan.mask_br_bytes
    renderer = payload[cursor:cursor + plan.renderer_br_bytes]
    cursor += plan.renderer_br_bytes
    actions = payload[cursor:cursor + plan.actions_br_bytes]
    cursor += plan.actions_br_bytes
    pose = payload[cursor:]
    if len(pose) != plan.pose_br_bytes:
        raise ValueError(f'''pose slice length mismatch: {len(pose)} != {plan.pose_br_bytes}''')
    return (plan, {
        'masks.mkv.br': mask,
        'renderer.bin.br': renderer,
        'seg_tile_actions.br': actions,
        'optimized_poses.qp1.br': pose })


def _read_uvarint(raw = None, cursor = None):
    value = 0
    shift = 0
    start = cursor
    if cursor < len(raw):
        byte = raw[cursor]
        cursor += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return (value, cursor)
        None += 7
        if shift > 63:
            pass
        elif cursor < len(raw):
            continue
    raise ValueError(f'''truncated or overlong uvarint at byte {start}''')


def decode_seg_tile_actions_raw(raw = None):
    '''Decode public PR75 action wire forms to runtime (pair, tile, action).'''
    records = []
    if raw.startswith(b'TA4'):
        body = raw[3:]
        if len(body) % 4:
            raise ValueError('TA4 action body length is not divisible by 4')
        for offset in range(0, len(body), 4):
            records.append((int.from_bytes(body[offset:offset + 2], 'little'), body[offset + 2], body[offset + 3]))
        wire_kind = 'TA4_raw_u16pair_u8tile_u8action'
    elif raw.startswith(b'TA5'):
        body = raw[3:]
        if len(body) % 5:
            raise ValueError('TA5 action body length is not divisible by 5')
        for offset in range(0, len(body), 5):
            records.append((int.from_bytes(body[offset:offset + 2], 'little'), int.from_bytes(body[offset + 2:offset + 4], 'little'), body[offset + 4]))
        wire_kind = 'TA5_raw_u16pair_u16tile_u8action'
    elif raw.startswith(b'S1'):
        cursor = 2
        (group_count, cursor) = _read_uvarint(raw, cursor)
        groups = []
        tile = 0
        for group_index in range(group_count):
            (tile_delta, cursor) = _read_uvarint(raw, cursor)
            tile = tile_delta if group_index == 0 else tile + tile_delta
            (count, cursor) = _read_uvarint(raw, cursor)
            groups.append((tile, count))
        pairs = []
        for tile, count in groups:
            frame = 0
            for idx in range(count):
                (delta, cursor) = _read_uvarint(raw, cursor)
                frame = delta if idx == 0 else frame + delta
                pairs.append((frame, tile))
        if cursor + len(pairs) != len(raw):
            raise ValueError('S1 split action stream length mismatch')
        for frame, tile in pairs:
            action = raw[cursor]
            cursor += 1
            records.append((frame, tile, action))
        wire_kind = 'S1_split_tile_delta_count_pair_delta_actions'
    elif (raw.startswith(b'SG2') or len(raw) % 4 != 0) and len(raw) % 5 != 0:
        cursor = 3 if raw.startswith(b'SG2') else 0
        if cursor < len(raw):
            (tile, cursor) = _read_uvarint(raw, cursor)
            (count, cursor) = _read_uvarint(raw, cursor)
            frame = 0
            for idx in range(count):
                (delta, cursor) = _read_uvarint(raw, cursor)
                frame = delta if idx == 0 else frame + delta
                if cursor >= len(raw):
                    raise ValueError('SG2 action stream ended before action byte')
                action = raw[cursor]
                cursor += 1
                records.append((frame, tile, action))
            if cursor < len(raw):
                continue
        wire_kind = 'SG2_grouped_tile_frame_delta_varint'
    elif len(raw) % 4 == 0 and len(raw) % 5 != 0:
        for offset in range(0, len(raw), 4):
            records.append((int.from_bytes(raw[offset:offset + 2], 'little'), raw[offset + 2], raw[offset + 3]))
        wire_kind = 'raw4_u16pair_u8tile_u8action'
    elif len(raw) % 5 == 0 and len(raw) % 4 != 0:
        for offset in range(0, len(raw), 5):
            records.append((int.from_bytes(raw[offset:offset + 2], 'little'), int.from_bytes(raw[offset + 2:offset + 4], 'little'), raw[offset + 4]))
        wire_kind = 'raw5_u16pair_u16tile_u8action'
    elif not raw:
        wire_kind = 'empty_raw4'
    else:
        raise ValueError(f'''ambiguous public action body length without TA4/TA5 header: {len(raw)}''')
    for frame, tile, action in records:
        if not  <= 0, frame or 0, frame < 600:
            pass
        else:
            records
        raise ValueError(f'''public seg action frame out of range: {frame}''')
        if not  <= 0, tile or 0, tile < MAX_TILE_ID:
            pass
        
        raise ValueError(f'''public seg action tile out of range: {tile}''')
        if  <= 0, action:
            if 0, action < PUBLIC_ACTION_COUNT:
                continue
            
        raise ValueError(f'''public seg action id out of range: {action}''')
    return (wire_kind, records)


def encode_runtime_action_records(records = None):
    out = bytearray()
    for frame, tile, action in records:
        out.extend(int(frame).to_bytes(2, 'little'))
        out.append(int(tile))
        out.append(int(action))
    return bytes(out)


def _top_counts(counter = None, n = None):
    pass
# WARNING: Decompyle incomplete


def summarize_action_records(*, raw_wire, charged, records, wire_kind):
    runtime_bytes = encode_runtime_action_records(records)
    pair_counts = (lambda .0: pass# WARNING: Decompyle incomplete
)(records())
    tile_counts = (lambda .0: pass# WARNING: Decompyle incomplete
)(records())
    action_counts = (lambda .0: pass# WARNING: Decompyle incomplete
)(records())
    pairs = sorted(pair_counts)
# WARNING: Decompyle incomplete


def summarize_qp1_pose(raw = None):
    if not raw.startswith(b'QP1'):
        return {
            'codec': 'unknown',
            'raw_bytes': int(len(raw)),
            'raw_sha256': _sha256_bytes(raw),
            'raw_prefix_hex': _safe_prefix(raw) }
    if None(raw) < 5:
        raise ValueError('QP1 stream is too short')
    values = [
        int.from_bytes(raw[3:5], 'little')]
    cursor = 5
    if cursor < len(raw):
        (acc, cursor) = _read_uvarint(raw, cursor)
        delta = acc >> 1 ^ -(acc & 1)
        values.append(values[-1] + delta)
        if cursor < len(raw):
            continue
    return {
        'codec': 'QP1_col0_delta_varint',
        'raw_bytes': int(len(raw)),
        'raw_sha256': _sha256_bytes(raw),
        'row_count': int(len(values)),
        'q0_min': int(min(values)),
        'q0_max': int(max(values)),
        'q0_first': int(values[0]),
        'q0_last': int(values[-1]),
        'raw_prefix_hex': _safe_prefix(raw) }


def _load_unpacker(path = None):
    spec = importlib.util.spec_from_file_location('profile_pr75_minp_unpacker', path)
# WARNING: Decompyle incomplete


def robust_parse_payload(payload = None, unpacker_path = None):
    pass
# WARNING: Decompyle incomplete


def profile_public_minp_archive(path = None, *, unpacker_path):
    (payload, zip_info) = _read_single_payload_zip(path)
    (plan, slices) = split_fixed_public_payload(payload)
    mask_raw = brotli.decompress(slices['masks.mkv.br'])
    renderer_raw = brotli.decompress(slices['renderer.bin.br'])
    actions_raw = brotli.decompress(slices['seg_tile_actions.br'])
    pose_raw = brotli.decompress(slices['optimized_poses.qp1.br'])
    if not mask_raw.startswith(b'\x12\x00'):
        raise ValueError('decoded mask stream does not look like AV1 OBU')
    if not renderer_raw.startswith(b'QZS3'):
        raise ValueError(f'''decoded renderer does not start with QZS3: {renderer_raw[:4]!r}''')
    if not pose_raw.startswith(b'QP1'):
        raise ValueError(f'''decoded pose stream does not start with QP1: {pose_raw[:3]!r}''')
    (action_wire_kind, action_records) = decode_seg_tile_actions_raw(actions_raw)
    return {
        'archive': {
            'path': str(path),
            'bytes': int(path.stat().st_size),
            'sha256': _sha256_file(path),
            'zip': zip_info },
        'payload': {
            'member': MEMBER_NAME,
            'bytes': int(len(payload)),
            'sha256': _sha256_bytes(payload),
            'prefix_hex': _safe_prefix(payload),
            'fixed_slice_plan': {
                'label': plan.label,
                'payload_bytes': plan.payload_bytes,
                'mask_br_bytes': plan.mask_br_bytes,
                'renderer_br_bytes': plan.renderer_br_bytes,
                'actions_br_bytes': plan.actions_br_bytes,
                'pose_br_bytes': plan.pose_br_bytes } },
        'decoded_streams': {
            'masks.mkv': _stream_summary('masks.mkv', slices['masks.mkv.br'], mask_raw, 'brotli_av1_obu'),
            'renderer.bin': _stream_summary('renderer.bin', slices['renderer.bin.br'], renderer_raw, 'brotli_qzs3'),
            'seg_tile_actions.bin': _stream_summary('seg_tile_actions.bin', slices['seg_tile_actions.br'], actions_raw, 'brotli_public_seg_tile_actions'),
            'optimized_poses.qp1': _stream_summary('optimized_poses.qp1', slices['optimized_poses.qp1.br'], pose_raw, 'brotli_qp1') },
        'renderer': {
            'magic': renderer_raw[:4].decode('ascii', errors = 'replace'),
            'qzs3_block_size': int.from_bytes(renderer_raw[4:6], 'little') if len(renderer_raw) >= 6 else None },
        'actions': summarize_action_records(raw_wire = actions_raw, charged = slices['seg_tile_actions.br'], records = action_records, wire_kind = action_wire_kind),
        'pose': summarize_qp1_pose(pose_raw),
        'robust_current_parse': robust_parse_payload(payload, unpacker_path) }


def profile_compare_archive(path = None, *, unpacker_path):
    (payload, zip_info) = _read_single_payload_zip(path)
    parsed = robust_parse_payload(payload, unpacker_path)
    out = {
        'archive': {
            'path': str(path),
            'bytes': int(path.stat().st_size),
            'sha256': _sha256_file(path),
            'zip': zip_info },
        'payload': {
            'member': MEMBER_NAME,
            'bytes': int(len(payload)),
            'sha256': _sha256_bytes(payload),
            'prefix_hex': _safe_prefix(payload) },
        'robust_current_parse': parsed }
    decoded = parsed.get('decoded_members') if parsed.get('ok') else None
    if isinstance(decoded, dict) and 'seg_tile_actions.bin' in decoded:
        unpacker = _load_unpacker(unpacker_path)
        (_header, decoded_bytes) = unpacker._parse_payload(payload)
        raw_actions = decoded_bytes['seg_tile_actions.bin']
        records = []
        if len(raw_actions) % 4 == 0:
            for offset in range(0, len(raw_actions), 4):
                records.append((int.from_bytes(raw_actions[offset:offset + 2], 'little'), raw_actions[offset + 2], raw_actions[offset + 3]))
            out['actions'] = summarize_action_records(raw_wire = raw_actions, charged = raw_actions, records = records, wire_kind = 'robust_runtime_raw4_after_unpack')
    return out


def compare_public_to_archive(public = None, other = None):
    other_decoded = other.get('robust_current_parse', { }).get('decoded_members', { })
    stream_comparison = { }
    for public_name, other_name in (('masks.mkv', 'masks.mkv'), ('renderer.bin', 'renderer.bin'), ('optimized_poses.qp1', 'optimized_poses.qp1'), ('seg_tile_actions.bin', 'seg_tile_actions.bin')):
        p_stream = public.get('decoded_streams', { }).get(public_name)
        o_stream = other_decoded.get(other_name)
        if not p_stream or o_stream:
            stream_comparison[public_name] = {
                'available': False,
                'public_available': bool(p_stream),
                'compare_available': bool(o_stream) }
            continue
        stream_comparison[public_name] = {
            'available': True,
            'decoded_bytes_delta_public_minus_compare': int(p_stream['decoded_bytes']) - int(o_stream['decoded_bytes']),
            'decoded_sha256_equal': p_stream['decoded_sha256'] == o_stream['decoded_sha256'],
            'public_decoded_sha256': p_stream['decoded_sha256'],
            'compare_decoded_sha256': o_stream['decoded_sha256'] }
    action_overlap = None
    public_actions = public.get('actions')
    other_actions = other.get('actions')
    if public_actions and other_actions:
        action_overlap = {
            'public_record_count': public_actions['record_count'],
            'compare_record_count': other_actions['record_count'],
            'runtime_record_sha256_equal': public_actions['runtime_record_sha256'] == other_actions['runtime_record_sha256'],
            'public_runtime_record_sha256': public_actions['runtime_record_sha256'],
            'compare_runtime_record_sha256': other_actions['runtime_record_sha256'] }
    return {
        'public_minus_compare_archive_bytes': int(public['archive']['bytes']) - int(other['archive']['bytes']),
        'public_minus_compare_payload_bytes': int(public['payload']['bytes']) - int(other['payload']['bytes']),
        'stream_comparison': stream_comparison,
        'action_record_comparison': action_overlap }


def _git_source_info(path = None):
    if not path.exists():
        return {
            'path': str(path),
            'exists': False }
    info = {
        'path': None(path),
        'exists': True }
    if (path / '.git').exists():
        for key, args in (('commit', [
            'rev-parse',
            'HEAD']), ('branch', [
            'rev-parse',
            '--abbrev-ref',
            'HEAD'])):
            completed = None(subprocess.run, check = True, text = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
            info[key] = completed.stdout.strip()
    inflate = path / 'submissions/qpose14_r55_segactions_minp/inflate.py'
    info['inflate_py'] = {
        'path': str(inflate),
        'exists': inflate.exists(),
        'sha256': _sha256_file(inflate) if inflate.exists() else None }
    return info
    except (OSError, subprocess.CalledProcessError):
        exc = None
        info[f'''{key}_error'''] = str(exc)
        exc = None
        del exc
        continue
        exc = None
        del exc


def build_profile(*, archive, compare_archive, pr75_source_root, unpacker_path):
    public = profile_public_minp_archive(archive, unpacker_path = unpacker_path)
    compare = None
    comparison = None
# WARNING: Decompyle incomplete


def build_arg_parser():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--archive', type = Path, default = DEFAULT_ARCHIVE)
    parser.add_argument('--compare-archive', type = Path, default = DEFAULT_COMPARE_ARCHIVE)
    parser.add_argument('--no-compare', action = 'store_true')
    parser.add_argument('--pr75-source-root', type = Path, default = DEFAULT_PR75_SOURCE_ROOT)
    parser.add_argument('--robust-unpacker', type = Path, default = DEFAULT_UNPACKER)
    parser.add_argument('--output-json', type = Path, default = DEFAULT_OUTPUT_JSON)
    return parser


def main(argv = None):
    args = build_arg_parser().parse_args(argv)
    compare_archive = None if args.no_compare else args.compare_archive
    profile = build_profile(archive = args.archive, compare_archive = compare_archive, pr75_source_root = args.pr75_source_root, unpacker_path = args.robust_unpacker)
    args.output_json.parent.mkdir(parents = True, exist_ok = True)
    args.output_json.write_bytes(_json_bytes(profile))
    print(json.dumps(profile, indent = 2, sort_keys = True, allow_nan = False))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

"""
