"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``190:25: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``build_pr85_stbm1br_pr92_rmb1_randmulti_candidate.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/build_pr85_stbm1br_pr92_rmb1_randmulti_candidate.py'
__recovery_spec__ = 'build_pr85_stbm1br_pr92_rmb1_randmulti_candidate.recovery_spec.json'
__recovery_ast_error__ = '190:25: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: build_pr85_stbm1br_pr92_rmb1_randmulti_candidate.cpython-312.pyc (Python 3.12)

\"\"\"Build the PR85_STBM1BR + PR92 RMB1 randmulti decoded-parity candidate.

This is a local deterministic candidate builder. It replaces only the STBM
frontier archive's randmulti segment with PR92's charged RMB1 randmulti segment
after proving decoded sparse-row parity. It does not run scorers, dispatch
remote work, or claim a score.
\"\"\"
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import io
import json
import shutil
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from tac.pr85_bundle import SEGMENT_ORDER, compare_pr85_randmulti_decoded_rows, decode_pr85_randmulti_to_headerless_rows, pack_pr85_bundle, parse_pr85_bundle, validate_pr85_member_name
TOOL = 'experiments/build_pr85_stbm1br_pr92_rmb1_randmulti_candidate.py'
SCHEMA = 'pr85_stbm1br_pr92_rmb1_randmulti_candidate_v1'
SUMMARY_SCHEMA = 'pr85_stbm1br_pr92_rmb1_randmulti_summary_v1'
DEFAULT_PR85_ARCHIVE = REPO_ROOT / 'experiments/results/public_pr85_intake_20260503_codex/archive.zip'
DEFAULT_STBM_ARCHIVE = REPO_ROOT / 'experiments/results/pr85_stbm1br_mask_recode_20260504_worker/pr90_stbm1br_lossless_pr85_mask_recode/archive.zip'
DEFAULT_STBM_MANIFEST = DEFAULT_STBM_ARCHIVE.parent / 'manifest.json'
DEFAULT_PR92_ARCHIVE = REPO_ROOT / 'experiments/results/public_pr92_intake_20260504_codex/archive.zip'
DEFAULT_PR92_PROFILE = DEFAULT_PR92_ARCHIVE.parent / 'public_frontier_intake_profile.json'
DEFAULT_STBM_REPLAY_RUNTIME = REPO_ROOT / 'experiments/results/pr85_stbm1br_mask_recode_20260504_worker/replay_submission_stbm'
DEFAULT_OUT_DIR = REPO_ROOT / 'experiments/results/pr85_stbm1br_pr92_rmb1_randmulti_20260504_worker'
DEFAULT_LEDGER = REPO_ROOT / '.omx/research/pr85_stbm1br_pr92_rmb1_randmulti_20260504_worker.md'
DEFAULT_STBM_EXACT_T4 = REPO_ROOT / 'experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/contest_auth_eval.adjudicated.json'
CANDIDATE_ID = 'pr85_stbm1br_plus_pr92_rmb1_randmulti_recode'
LANE_ID = 'pr85_stbm1br_pr92_rmb1_randmulti'
EXPECTED = {
    'pr85_archive_bytes': 236328,
    'pr85_archive_sha256': 'eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e',
    'stbm_archive_bytes': 229756,
    'stbm_archive_sha256': 'c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6',
    'pr92_archive_bytes': 236516,
    'pr92_archive_sha256': 'f0dedeb7ad3c019ab3f4e2dea317f3192408e47a573897adb208030709d01490',
    'stbm_randmulti_bytes': 16101,
    'pr92_rmb1_randmulti_bytes': 15825,
    'pr92_rmb1_randmulti_sha256': '4b10018eab64d8755da3def355881f51f2d450c9f19dd1457a6ec813cddd6f7c',
    'decoded_randmulti_rows_bytes': 27105,
    'decoded_randmulti_rows_sha256': '87bcc720c1e80afb9adad5ee01477423ced526f31c54d461d69dbf26e08eecc9',
    'stbm_mask_sha256': '1b1ec60b64e284aae11e838dc3d9996bce00125df5712a8ba9c3e8f739c9d313' }
RMB1_RUNTIME_HELPER = '\n\ndef decode_randmulti_bitmask_payload(encoded_randmulti: bytes) -> bytes:\n    \"\"\"Decode PR92 RMB1 bitmask+value randmulti to headerless sparse rows.\"\"\"\n    if len(encoded_randmulti) < 6 or encoded_randmulti[:4] != b"RMB1":\n        raise ValueError("bad RMB1 randmulti payload")\n    mask_len = int.from_bytes(encoded_randmulti[4:6], "little")\n    mask_br = encoded_randmulti[6:6 + mask_len]\n    vals_br = encoded_randmulti[6 + mask_len:]\n    if not mask_br or not vals_br:\n        raise ValueError("truncated RMB1 randmulti payload")\n    mask = brotli.decompress(mask_br)\n    vals = brotli.decompress(vals_br)\n    if len(mask) % 75:\n        raise ValueError("bad RMB1 mask length")\n    out = bytearray()\n    vals_pos = 0\n    for row_start in range(0, len(mask), 75):\n        row_mask = mask[row_start:row_start + 75]\n        indices = []\n        row_values = []\n        for byte_i, byte in enumerate(row_mask):\n            for bit in range(8):\n                frame_i = byte_i * 8 + bit\n                if frame_i >= 600:\n                    break\n                if byte & (1 << bit):\n                    if vals_pos >= len(vals):\n                        raise ValueError("truncated RMB1 values")\n                    indices.append(frame_i)\n                    row_values.append(vals[vals_pos])\n                    vals_pos += 1\n        count = len(indices)\n        if count < 255:\n            out.append(count)\n        else:\n            out.append(255)\n            out.extend(count.to_bytes(2, "little"))\n        last = -1\n        for idx in indices:\n            delta = idx - last - 1\n            last = idx\n            while True:\n                byte = delta & 0x7F\n                delta >>= 7\n                if delta:\n                    out.append(byte | 0x80)\n                else:\n                    out.append(byte)\n                    break\n        out.extend(row_values)\n    if vals_pos != len(vals):\n        raise ValueError("unused RMB1 values")\n    return bytes(out)\n'

class CandidateBuildError(ValueError):
    '''Raised when the candidate must fail closed.'''
    pass


def _json_text(payload = None):
    return json.dumps(payload, indent = 2, sort_keys = True, allow_nan = False) + '\n'


def _write_json(path = None, payload = None):
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(_json_text(payload), encoding = 'utf-8')


def _load_json(path = None):
    if not path.is_file():
        raise CandidateBuildError(f'''missing JSON file: {_rel(path)}''')
    payload = json.loads(path.read_text(encoding = 'utf-8'))
    if not isinstance(payload, dict):
        raise CandidateBuildError(f'''JSON file must contain an object: {_rel(path)}''')
    return payload


def _sha256_bytes(data = None):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path = None):
    pass
# WARNING: Decompyle incomplete


def _rel(path = None):
    resolved = Path(path).resolve()
    
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return 



def _utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond = 0)


def _read_archive_member(path = None, *, member, allow_extra_members):
    pass
# WARNING: Decompyle incomplete


def _expect_archive(meta = None, *, bytes_key, sha_key):
    if meta.get('archive_bytes') != EXPECTED[bytes_key]:
        raise CandidateBuildError(f'''{bytes_key} mismatch: {meta.get('archive_bytes')} != {EXPECTED[bytes_key]}''')
    if meta.get('archive_sha256') != EXPECTED[sha_key]:
        raise CandidateBuildError(f'''{sha_key} mismatch: {meta.get('archive_sha256')} != {EXPECTED[sha_key]}''')


def _zip_single_x_bytes(payload = None):
    buffer = io.BytesIO()
    info = zipfile.ZipInfo('x', (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 27525120
    info.create_system = 3
    zf = zipfile.ZipFile(buffer, 'w')
    zf.writestr(info, payload)
    None(None, None)
    return buffer.getvalue()
    with None:
        if not None:
            pass
    return buffer.getvalue()


def _local_header_name(archive_path = None, info = None):
    data = archive_path.read_bytes()
    offset = int(info.header_offset)
    if data[offset:offset + 4] != b'PK\x03\x04':
        raise CandidateBuildError('candidate ZIP local header signature mismatch')
    (name_len, extra_len) = struct.unpack_from('<HH', data, offset + 26)
    name_start = offset + 30
    name_end = name_start + int(name_len)
    _extra_end = name_end + int(extra_len)
    return data[name_start:name_end].decode('utf-8')


def _strict_zip_report(archive_path = None):
    zf = zipfile.ZipFile(archive_path, 'r')
# WARNING: Decompyle incomplete


def _segment_diff(left_raw = None, right_raw = None):
    left = parse_pr85_bundle(left_raw)
    right = parse_pr85_bundle(right_raw)
    diffs = []
    for name in SEGMENT_ORDER:
        l_bytes = bytes(left.segments[name])
        r_bytes = bytes(right.segments[name])
        if not l_bytes != r_bytes:
            continue
        diffs.append({
            'segment': name,
            'left_bytes': len(l_bytes),
            'right_bytes': len(r_bytes),
            'left_sha256': _sha256_bytes(l_bytes),
            'right_sha256': _sha256_bytes(r_bytes) })
    return diffs


def _stbm_manifest_report(path = None, stbm_meta = None, pr85_meta = None):
    payload = _load_json(path)
    checks = {
        'score_claim_false': payload.get('score_claim') is False,
        'dispatch_performed_false': payload.get('dispatch_performed') is False,
        'candidate_sha_matches': payload.get('candidate_archive', { }).get('archive_sha256') == stbm_meta.get('archive_sha256'),
        'source_sha_matches': payload.get('source_archive', { }).get('archive_sha256') == pr85_meta.get('archive_sha256'),
        'decoded_mask_equal': payload.get('parity', { }).get('decoded_mask_equal') is True,
        'diff_pixels_zero': payload.get('parity', { }).get('diff_pixels') == 0,
        'stbm_exact_runtime_ready': payload.get('exact_eval_runtime_contract', { }).get('ready_for_exact_eval_runtime') is True }
    return {
        'path': _rel(path),
        'sha256': _sha256_file(path),
        'checks': checks,
        'status': 'passed' if all(checks.values()) else 'failed',
        'candidate_id': payload.get('candidate_id'),
        'runtime_tree_sha256': payload.get('exact_eval_runtime_contract', { }).get('runtime_tree_sha256'),
        'render_order_sha256': payload.get('parity', { }).get('candidate_render_order_sha256'),
        'mask_sha256': payload.get('segments', { }).get('candidate_mask', { }).get('sha256') }


def _pr92_profile_report(path = None):
    payload = _load_json(path)
    rows = payload.get('primary_member', { }).get('segments', [])
    randmulti = (lambda .0: pass# WARNING: Decompyle incomplete
)(rows(), { })
    checks = {
        'score_claim_false': payload.get('score_claim') is False,
        'promotion_eligible_false': payload.get('promotion_eligible') is False,
        'randmulti_bytes_expected': randmulti.get('bytes') == EXPECTED['pr92_rmb1_randmulti_bytes'],
        'randmulti_sha_expected': randmulti.get('sha256') == EXPECTED['pr92_rmb1_randmulti_sha256'],
        'randmulti_codec_rmb1': randmulti.get('codec') == 'RMB1_side_info_backed_randmulti' }
    if all(checks.values()):
        return {
            'path': _rel(path),
            'sha256': _sha256_file(path),
            'label': payload.get('label'),
            'evidence_grade': payload.get('evidence_grade'),
            'side_info_charged_bytes': payload.get('side_info', { }).get('charged_bytes'),
            'randmulti_segment': dict(randmulti),
            'checks': checks,
            'status': 'passed' }
    return {
        'path': next,
        'sha256': _rel(path),
        'label': _sha256_file(path),
        'evidence_grade': payload.get('label'),
        'side_info_charged_bytes': payload.get('evidence_grade'),
        'randmulti_segment': payload.get('side_info', { }).get('charged_bytes'),
        'checks': dict(randmulti),
        'status': checks }


def _runtime_tree_manifest(runtime_dir = None):
    files = []
    tree = hashlib.sha256()
    for path in (lambda .0: pass# WARNING: Decompyle incomplete
)(runtime_dir.rglob('*')()):
        rel = path.relative_to(runtime_dir).as_posix()
        sha = _sha256_file(path)
        files.append({
            'path': rel,
            'bytes': path.stat().st_size,
            'sha256': sha })
        tree.update(rel.encode('utf-8') + b'\x00' + sha.encode('ascii') + b'\x00')
    return {
        'runtime_dir': _rel(runtime_dir),
        'runtime_file_count': len(files),
        'runtime_tree_sha256': tree.hexdigest(),
        'files': files }


def _patch_replay_runtime_for_rmb1(source_dir = None, out_dir = None):
    out_dir.mkdir(parents = True, exist_ok = True)
    for child in sorted(source_dir.iterdir()):
        if child.is_file() and child.name != 'README.md':
            shutil.copy2(child, out_dir / child.name)
            continue
        if not child.is_file():
            continue
        if not child.name == 'README.md':
            continue
        text = child.read_text(encoding = 'utf-8', errors = 'replace')
        text += '\nPR92 RMB1 randmulti support: candidate-local copy generated by ' + TOOL + '\n'
        (out_dir / child.name).write_text(text, encoding = 'utf-8')
    inflate_py = out_dir / 'inflate.py'
    if not inflate_py.is_file():
        raise CandidateBuildError('runtime copy is missing inflate.py')
    text = inflate_py.read_text(encoding = 'utf-8', errors = 'replace')
    if 'def decode_randmulti_bitmask_payload' not in text:
        marker = '\ndef main():'
        if marker not in text:
            raise CandidateBuildError('cannot locate replay runtime main() for RMB1 helper insertion')
        text = text.replace(marker, RMB1_RUNTIME_HELPER + marker, 1)
    old = '        raw_n = brotli.decompress(bundle["randmulti"])'
    new = '        encoded_n = bundle["randmulti"]\n        if encoded_n[:4] == b"RMB1":\n            raw_n = decode_randmulti_bitmask_payload(encoded_n)\n        else:\n            raw_n = brotli.decompress(encoded_n)'
    if old in text:
        text = text.replace(old, new, 1)
    inflate_py.write_text(text, encoding = 'utf-8')
    if 'STBM1BR' in text:
        'STBM1BR' in text
    if 'def decode_randmulti_bitmask_payload' in text:
        'def decode_randmulti_bitmask_payload' in text
    checks = {
        'source_runtime_exists': source_dir.is_dir(),
        'inflate_sh_present': (out_dir / 'inflate.sh').is_file(),
        'inflate_py_present': inflate_py.is_file(),
        'stbm_support_present': 'load_stbm1br_mask' in text,
        'rmb1_helper_present': 'RMB1' in text,
        'rmb1_randmulti_branch_present': 'encoded_n[:4] == b"RMB1"' in text }
    report = _runtime_tree_manifest(out_dir)
    report.update({
        'schema': 'pr85_stbm1br_pr92_rmb1_replay_runtime_v1',
        'source_runtime_dir': _rel(source_dir),
        'checks': checks,
        'status': 'passed' if all(checks.values()) else 'failed',
        'score_claim': False,
        'dispatch_performed': False })
    return report


def _robust_current_rmb1_report():
    pass
# WARNING: Decompyle incomplete


def _stbm_exact_t4_report(path = None, stbm_meta = None):
    if not path.is_file():
        return {
            'path': _rel(path),
            'status': 'missing',
            'checks': {
                'artifact_present': False } }
    payload = None(path)
    provenance = payload.get('provenance', { }) if isinstance(payload.get('provenance'), Mapping) else { }
    runtime = provenance.get('inflate_runtime_manifest', { }) if isinstance(provenance.get('inflate_runtime_manifest'), Mapping) else { }
    checks = {
        'artifact_present': True,
        'archive_sha_matches_stbm': provenance.get('archive_sha256') == stbm_meta.get('archive_sha256'),
        'archive_bytes_matches_stbm': payload.get('archive_size_bytes') == stbm_meta.get('archive_bytes'),
        'cuda_device': provenance.get('device') == 'cuda',
        't4_match': provenance.get('gpu_t4_match') is True,
        'full_sample_count': payload.get('n_samples') == 600,
        'runtime_tree_recorded': isinstance(runtime.get('runtime_tree_sha256'), str) }
    return {
        'path': _rel(path),
        'sha256': _sha256_file(path),
        'status': 'passed' if all(checks.values()) else 'failed',
        'checks': checks,
        'canonical_score': payload.get('canonical_score'),
        'avg_posenet_dist': payload.get('avg_posenet_dist'),
        'avg_segnet_dist': payload.get('avg_segnet_dist'),
        'runtime_tree_sha256': runtime.get('runtime_tree_sha256') }


def build_candidate(*, pr85_archive, stbm_archive, stbm_manifest, pr92_archive, pr92_profile, stbm_replay_runtime, stbm_exact_t4_json, out_dir):
    (pr85_meta, pr85_raw) = _read_archive_member(pr85_archive)
    (stbm_meta, stbm_raw) = _read_archive_member(stbm_archive)
    (pr92_meta, pr92_raw) = _read_archive_member(pr92_archive, allow_extra_members = True)
    _expect_archive(pr85_meta, bytes_key = 'pr85_archive_bytes', sha_key = 'pr85_archive_sha256')
    _expect_archive(stbm_meta, bytes_key = 'stbm_archive_bytes', sha_key = 'stbm_archive_sha256')
    _expect_archive(pr92_meta, bytes_key = 'pr92_archive_bytes', sha_key = 'pr92_archive_sha256')
    pr85_bundle = parse_pr85_bundle(pr85_raw)
    stbm_bundle = parse_pr85_bundle(stbm_raw)
    pr92_bundle = parse_pr85_bundle(pr92_raw)
    stbm_randmulti = bytes(stbm_bundle.segments['randmulti'])
    pr92_randmulti = bytes(pr92_bundle.segments['randmulti'])
    stbm_mask = bytes(stbm_bundle.segments['mask'])
    if len(stbm_randmulti) != EXPECTED['stbm_randmulti_bytes']:
        raise CandidateBuildError('STBM randmulti encoded byte count mismatch')
    if len(pr92_randmulti) != EXPECTED['pr92_rmb1_randmulti_bytes']:
        raise CandidateBuildError('PR92 RMB1 randmulti encoded byte count mismatch')
    if not pr92_randmulti.startswith(b'RMB1'):
        raise CandidateBuildError('PR92 randmulti segment is not RMB1')
    if _sha256_bytes(pr92_randmulti) != EXPECTED['pr92_rmb1_randmulti_sha256']:
        raise CandidateBuildError('PR92 RMB1 randmulti SHA mismatch')
    (decoded_rows, decoded_profile) = decode_pr85_randmulti_to_headerless_rows(pr92_randmulti)
    if len(decoded_rows) != EXPECTED['decoded_randmulti_rows_bytes']:
        raise CandidateBuildError('PR92 RMB1 decoded rows byte count mismatch')
    if _sha256_bytes(decoded_rows) != EXPECTED['decoded_randmulti_rows_sha256']:
        raise CandidateBuildError('PR92 RMB1 decoded rows SHA mismatch')
    stbm_manifest_report = _stbm_manifest_report(stbm_manifest, stbm_meta, pr85_meta)
    pr92_profile_report = _pr92_profile_report(pr92_profile)
    if stbm_manifest_report['status'] != 'passed':
        raise CandidateBuildError('STBM source manifest did not pass review checks')
    if pr92_profile_report['status'] != 'passed':
        raise CandidateBuildError('PR92 intake profile did not pass randmulti checks')
    if stbm_manifest_report['mask_sha256'] != _sha256_bytes(stbm_mask):
        raise CandidateBuildError('STBM manifest mask SHA does not match source archive')
    if _sha256_bytes(stbm_mask) != EXPECTED['stbm_mask_sha256']:
        raise CandidateBuildError('STBM source mask SHA mismatch')
    stbm_vs_pr85 = _segment_diff(pr85_raw, stbm_raw)
# WARNING: Decompyle incomplete


def render_ledger(summary = None):
    candidate = summary.get('candidate_archive', { }) if isinstance(summary.get('candidate_archive'), Mapping) else { }
    lines = [
        '# PR85 STBM1BR + PR92 RMB1 Randmulti Recode - 2026-05-04',
        '',
        f'''- tool: `{TOOL}`''',
        '- score_claim: false',
        '- dispatch_performed: false',
        '- remote_jobs_dispatched: false',
        '',
        '## Candidate',
        '',
        f'''- archive: `{candidate.get('path')}`''',
        f'''- bytes: `{candidate.get('archive_bytes')}`''',
        f'''- sha256: `{candidate.get('archive_sha256')}`''',
        f'''- manifest: `{summary.get('candidate_manifest')}`''',
        f'''- archive_delta_bytes_vs_stbm: `{summary.get('archive_delta_bytes_vs_stbm')}`''',
        f'''- randmulti decoded rows SHA: `{summary.get('randmulti_decoded_rows_sha256')}`''',
        '',
        '## Readiness',
        '',
        f'''- strict_zip_valid: `{summary.get('strict_zip_valid')}`''',
        f'''- exact_t4_dispatch_justified_after_claim: `{summary.get('exact_t4_dispatch_justified_after_claim')}`''',
        '- exact CUDA eval is required before any score claim.',
        '',
        '## Exact Next Claim Command',
        '',
        '```bash',
        str(summary.get('next_claim_command')),
        '```',
        '']
    return '\n'.join(lines)


def write_ledger(path = None, summary = None):
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(render_ledger(summary), encoding = 'utf-8')


def parse_args(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--pr85-archive', type = Path, default = DEFAULT_PR85_ARCHIVE)
    parser.add_argument('--stbm-archive', type = Path, default = DEFAULT_STBM_ARCHIVE)
    parser.add_argument('--stbm-manifest', type = Path, default = DEFAULT_STBM_MANIFEST)
    parser.add_argument('--pr92-archive', type = Path, default = DEFAULT_PR92_ARCHIVE)
    parser.add_argument('--pr92-profile', type = Path, default = DEFAULT_PR92_PROFILE)
    parser.add_argument('--stbm-replay-runtime', type = Path, default = DEFAULT_STBM_REPLAY_RUNTIME)
    parser.add_argument('--stbm-exact-t4-json', type = Path, default = DEFAULT_STBM_EXACT_T4)
    parser.add_argument('--out-dir', type = Path, default = DEFAULT_OUT_DIR)
    parser.add_argument('--ledger-md', type = Path, default = DEFAULT_LEDGER)
    parser.add_argument('--stdout', action = 'store_true')
    return parser.parse_args(argv)


def main(argv = None):
    args = parse_args(argv)
    summary = build_candidate(pr85_archive = args.pr85_archive, stbm_archive = args.stbm_archive, stbm_manifest = args.stbm_manifest, pr92_archive = args.pr92_archive, pr92_profile = args.pr92_profile, stbm_replay_runtime = args.stbm_replay_runtime, stbm_exact_t4_json = args.stbm_exact_t4_json, out_dir = args.out_dir)
    write_ledger(args.ledger_md, summary)
    if args.stdout:
        sys.stdout.write(_json_text(summary))
        return 0
    print(_json_text({
        'candidate_archive': summary['candidate_archive'],
        'candidate_manifest': summary['candidate_manifest'],
        'exact_t4_dispatch_justified_after_claim': summary['exact_t4_dispatch_justified_after_claim'] }), end = '')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

"""
