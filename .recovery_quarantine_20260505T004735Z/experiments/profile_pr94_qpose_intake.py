"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``30:17: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``profile_pr94_qpose_intake.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/profile_pr94_qpose_intake.py'
__recovery_spec__ = 'profile_pr94_qpose_intake.recovery_spec.json'
__recovery_ast_error__ = '30:17: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: profile_pr94_qpose_intake.cpython-312.pyc (Python 3.12)

'''Static PR94 qpose archive intake profiler.

This tool performs local byte and grammar accounting only. It does not run the
contest scorer, does not load scorer models, and does not dispatch GPU work.
'''
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import brotli
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = REPO_ROOT / 'experiments/results/public_pr94_qpose_intake_20260504_codex/archive.zip'
DEFAULT_PR_JSON = REPO_ROOT / 'experiments/results/public_pr94_hpac_contract_probe_20260504_codex/pr94_api.json'
DEFAULT_OUT_DIR = REPO_ROOT / 'experiments/results/public_pr94_qpose_intake_20260504_codex'
DEFAULT_JSON_OUT = DEFAULT_OUT_DIR / 'profile_pr94_qpose_intake.json'
DEFAULT_MARKDOWN_OUT = DEFAULT_OUT_DIR / 'profile_pr94_qpose_intake.md'
CONTEST_ORIGINAL_BYTES = 37545489
TOOL = 'experiments/profile_pr94_qpose_intake.py'
SCHEMA = 'pr94_qpose_static_intake_profile_v1'
SegmentLayout = <NODE:12>()

def _sha256_bytes(data = None):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path = None):
    pass
# WARNING: Decompyle incomplete


def _json_text(payload = None):
    return json.dumps(payload, indent = 2, sort_keys = True, allow_nan = False) + '\n'


def _rel(path = None):
    
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return 



def _rate_score_delta(byte_delta = None):
    return 25 * byte_delta / CONTEST_ORIGINAL_BYTES


def _read_single_member_zip(path = None):
    zf = zipfile.ZipFile(path, 'r')
# WARNING: Decompyle incomplete


def infer_pr94_layout(payload = None):
    '''Mirror the PR94 ``inflate.py`` packed-payload branch without decoding.'''
    if payload.startswith(b'P3'):
        if len(payload) < 10:
            raise ValueError('truncated P3 qpose payload')
        (mask_len, model_len, actions_len) = struct.unpack_from('<IHH', payload, 2)
        header_bytes = 2 + struct.calcsize('<IHH')
        pose_len = len(payload) - header_bytes - mask_len - model_len - actions_len
        if min(mask_len, model_len, actions_len, pose_len) <= 0:
            raise ValueError('invalid P3 qpose segment length')
        return SegmentLayout(payload_format = 'p3_self_describing_qpose_tile_actions', boundary_authority = 'pr94_inflate_p3_header', header_bytes = header_bytes, mask_len = int(mask_len), model_len = int(model_len), actions_len = int(actions_len), pose_len = int(pose_len))
    if None.startswith(b'P2'):
        if len(payload) < 8:
            raise ValueError('truncated P2 qpose payload')
        (mask_len, model_len) = struct.unpack_from('<IH', payload, 2)
        header_bytes = 2 + struct.calcsize('<IH')
        pose_len = len(payload) - header_bytes - mask_len - model_len
        if min(mask_len, model_len, pose_len) <= 0:
            raise ValueError('invalid P2 qpose segment length')
        return SegmentLayout(payload_format = 'p2_self_describing_qpose_no_tile_actions', boundary_authority = 'pr94_inflate_p2_header', header_bytes = header_bytes, mask_len = int(mask_len), model_len = int(model_len), actions_len = 0, pose_len = int(pose_len))
    if None(payload) == 276641:
        return SegmentLayout(payload_format = 'public_pr75_fixed_qpose_tile_actions', boundary_authority = 'pr94_inflate_exact_len_276641', header_bytes = 0, mask_len = 219472, model_len = 56034, actions_len = 236, pose_len = len(payload) - 219472 - 56034 - 236)
    if not None(payload) in (276574, 276749):
        if  <= 276900, len(payload) or 276900, len(payload) <= 278000:
            pass
        
    else:
        55756 = 219472
        pose_len = 898
        actions_len = len(payload) - mask_len - model_len - pose_len
        if actions_len <= 0:
            raise ValueError('PR94 fixed-range payload has no action stream')
        return SegmentLayout(payload_format = 'pr94_fixed_range_qpose_tile_actions', boundary_authority = 'pr94_inflate_len_range_276900_278000', header_bytes = 0, mask_len = mask_len, model_len = model_len, actions_len = actions_len, pose_len = pose_len)
    raise None(f'''unsupported PR94 qpose payload length: {len(payload)}''')


def _slice_payload(payload = None, layout = None):
    cursor = layout.header_bytes
    mask = payload[cursor:cursor + layout.mask_len]
    cursor += layout.mask_len
    model = payload[cursor:cursor + layout.model_len]
    cursor += layout.model_len
    actions = payload[cursor:cursor + layout.actions_len]
    cursor += layout.actions_len
    pose = payload[cursor:cursor + layout.pose_len]
    cursor += layout.pose_len
    if cursor != len(payload):
        raise ValueError(f'''payload slice accounting mismatch: cursor={cursor}, len={len(payload)}''')
    return {
        'masks.mkv.br': mask,
        'renderer.bin.br': model,
        'seg_tile_actions.br': actions,
        'optimized_poses.qp1.br': pose }


def _decode_uvarint(raw = None, cursor = None):
    shift = 0
    value = 0
    if cursor < len(raw):
        byte = raw[cursor]
        cursor += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return (value, cursor)
        None += 7
        if shift > 63:
            raise ValueError('truncated or overlong uvarint')
        if cursor < len(raw):
            continue
    raise ValueError('truncated or overlong uvarint')


def parse_seg_tile_actions(decoded = None):
    records = []
    if (decoded.startswith(b'SG2') or len(decoded) % 4 != 0) and len(decoded) % 5 != 0:
        cursor = 3 if decoded.startswith(b'SG2') else 0
        if cursor < len(decoded):
            (tile, cursor) = _decode_uvarint(decoded, cursor)
            (count, cursor) = _decode_uvarint(decoded, cursor)
            frame = 0
            for idx in range(count):
                (delta, cursor) = _decode_uvarint(decoded, cursor)
                frame = delta if idx == 0 else frame + delta
                if cursor >= len(decoded):
                    raise ValueError('seg tile action stream ended before action byte')
                action = decoded[cursor]
                cursor += 1
                records.append((frame, tile, action))
            if cursor < len(decoded):
                continue
        fmt = 'sg2_tile_group_varint' if decoded.startswith(b'SG2') else 'tile_group_varint'
    elif len(decoded) % 4 == 0:
        for offset in range(0, len(decoded), 4):
            records.append((int.from_bytes(decoded[offset:offset + 2], 'little'), decoded[offset + 2], decoded[offset + 3]))
        fmt = 'fixed4_frame_u16_tile_u8_action_u8'
    elif len(decoded) % 5 == 0:
        for offset in range(0, len(decoded), 5):
            records.append((int.from_bytes(decoded[offset:offset + 2], 'little'), int.from_bytes(decoded[offset + 2:offset + 4], 'little'), decoded[offset + 4]))
        fmt = 'fixed5_frame_u16_tile_u16_action_u8'
    else:
        raise ValueError(f'''unsupported seg tile action payload length: {len(decoded)}''')
# WARNING: Decompyle incomplete


def parse_qp1_pose(decoded = None):
    pass
# WARNING: Decompyle incomplete


def np_uint16_le(data = None):
    return struct.unpack('<H', data)[0]


def _segment_profile(name = None, charged = None, decoded = None, codec = ('name', 'str', 'charged', 'bytes', 'decoded', 'bytes | None', 'codec', 'str', 'return', 'dict[str, Any]')):
    profile = {
        'name': name,
        'codec': codec,
        'charged_bytes': len(charged),
        'charged_sha256': _sha256_bytes(charged),
        'charged_prefix_hex': charged[:8].hex() }
# WARNING: Decompyle incomplete


def profile_payload(payload = None, *, layout):
    if not layout:
        layout
    layout = infer_pr94_layout(payload)
    slices = _slice_payload(payload, layout)
    decoded_mask = brotli.decompress(slices['masks.mkv.br'])
    decoded_model = brotli.decompress(slices['renderer.bin.br'])
    decoded_actions = brotli.decompress(slices['seg_tile_actions.br']) if slices['seg_tile_actions.br'] else b''
    decoded_pose = brotli.decompress(slices['optimized_poses.qp1.br'])
    segments = [
        _segment_profile('masks.mkv', slices['masks.mkv.br'], decoded_mask, 'brotli_av1_obu_mask_video'),
        _segment_profile('renderer.bin', slices['renderer.bin.br'], decoded_model, 'brotli_qzs_renderer'),
        _segment_profile('seg_tile_actions.bin', slices['seg_tile_actions.br'], decoded_actions, 'brotli_seg_tile_actions'),
        _segment_profile('optimized_poses.qp1', slices['optimized_poses.qp1.br'], decoded_pose, 'brotli_qpose_qp1')]
# WARNING: Decompyle incomplete


def _reported_mps_summary(pr_json = None):
    pass
# WARNING: Decompyle incomplete


def stackability_findings(profile = None):
    layout = profile['layout']
    return [
        {
            'surface': 'pose_side_qp1_velocity',
            'charged_bytes': layout['pose_bytes'],
            'static_delta_vs_pr85_stbm_pose_bytes': layout['pose_bytes'] - 1487,
            'rate_score_delta_if_isolated': _rate_score_delta(layout['pose_bytes'] - 1487),
            'verdict': 'blocked_not_isolated',
            'reason': 'PR94 encodes only velocity col0 and relies on its own qpose runtime path; PR85/STBM uses a different pose contract, so this is not a drop-in pose-side stack without decode/reencode and runtime-output parity.' },
        {
            'surface': 'renderer_qzs3_model',
            'charged_bytes': layout['model_bytes'],
            'static_delta_vs_pr85_stbm_model_bytes': layout['model_bytes'] - 57074,
            'rate_score_delta_if_isolated': _rate_score_delta(layout['model_bytes'] - 57074),
            'verdict': 'blocked_coupled_model_mask_pose',
            'reason': 'The smaller QZS3 model is trained for PR94 masks/pose/actions; transplanting it onto PR85_STBM1BR/RMB1 would be a renderer replacement, not an isolated non-mask recode.' },
        {
            'surface': 'tile_actions_control',
            'charged_bytes': layout['actions_bytes'],
            'static_delta_vs_no_tile_actions': layout['actions_bytes'],
            'rate_score_delta_if_added': _rate_score_delta(layout['actions_bytes']),
            'verdict': 'not_rate_stackable',
            'reason': 'Tile actions add charged bytes and only become useful if exact CUDA component gain exceeds their rate cost; PR94 only supplies MPS evidence.' },
        {
            'surface': 'mask_stream',
            'charged_bytes': layout['mask_bytes'],
            'static_delta_vs_pr85_stbm_mask_bytes': layout['mask_bytes'] - 152439,
            'rate_score_delta_if_replacing_stbm_mask': _rate_score_delta(layout['mask_bytes'] - 152439),
            'verdict': 'do_not_stack_onto_stbm',
            'reason': "Replacing PR85_STBM1BR's lossless mask recode with PR94's full mask stream gives back the STBM byte win and changes the scorer-visible mask basin." }]


def build_profile(archive = None, *, pr_json):
    (archive_info, payload) = _read_single_member_zip(archive)
    payload_profile = profile_payload(payload)
# WARNING: Decompyle incomplete


def build_markdown(profile = None):
    lines = [
        '# PR94 Qpose Static Intake',
        '',
        f'''- archive: `{profile['archive']['path']}`''',
        f'''- archive_bytes: `{profile['archive']['archive_bytes']}`''',
        f'''- archive_sha256: `{profile['archive']['archive_sha256']}`''',
        f'''- member: `{profile['archive']['member_name']}` stored={profile['archive']['zip_stored']} bytes={profile['archive']['member_bytes']}''',
        f'''- payload_format: `{profile['payload']['payload_format']}`''',
        f'''- evidence_grade: `{profile['evidence_grade']}`''',
        f'''- score_claim: `{profile['score_claim']}`''',
        '',
        '## Segments',
        '',
        '| segment | charged bytes | charged sha256 | decoded bytes | decoded prefix |',
        '| --- | ---: | --- | ---: | --- |']
    for row in profile['segments']:
        lines.append(f'''| `{row['name']}` | {row['charged_bytes']} | `{row['charged_sha256']}` | {row.get('decoded_bytes')} | `{row.get('decoded_prefix_hex')}` |''')
    lines.extend([
        '',
        '## Stackability',
        ''])
    for row in profile['stackability']:
        delta = row.get('static_delta_vs_pr85_stbm_pose_bytes')
        delta = row.get('static_delta_vs_pr85_stbm_model_bytes', delta)
        delta = row.get('static_delta_vs_pr85_stbm_mask_bytes', delta)
        delta = row.get('static_delta_vs_no_tile_actions', delta)
        lines.append(f'''- `{row['surface']}`: `{row['verdict']}`, byte_delta `{delta}`. {row['reason']}''')
    return '\n'.join(lines) + '\n'


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--archive', type = Path, default = DEFAULT_ARCHIVE)
    parser.add_argument('--pr-json', type = Path, default = DEFAULT_PR_JSON)
    parser.add_argument('--json-out', type = Path, default = DEFAULT_JSON_OUT)
    parser.add_argument('--markdown-out', type = Path, default = DEFAULT_MARKDOWN_OUT)
    args = parser.parse_args(argv)
    profile = build_profile(args.archive, pr_json = args.pr_json)
    if args.json_out:
        args.json_out.parent.mkdir(parents = True, exist_ok = True)
        args.json_out.write_text(_json_text(profile), encoding = 'utf-8')
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents = True, exist_ok = True)
        args.markdown_out.write_text(build_markdown(profile), encoding = 'utf-8')
    print(_json_text(profile))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

"""
