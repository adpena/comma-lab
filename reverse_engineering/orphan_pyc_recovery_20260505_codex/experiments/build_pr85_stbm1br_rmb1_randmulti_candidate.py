# Source Generated with Decompyle++
# File: build_pr85_stbm1br_rmb1_randmulti_candidate.cpython-312.pyc (Python 3.12)

"""Build PR85 STBM1BR plus PR92 RMB1 randmulti lossless recode candidate.

This local builder replaces only the ``randmulti`` segment in the current
STBM1BR frontier archive with PR92's byte-smaller ``RMB1`` recode, then proves
the decoded headerless sparse randmulti rows are identical. It does not run
scorers, dispatch remote work, or claim a score.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from tac.pr85_bundle import Pr85BundleError, decode_pr85_randmulti_to_headerless_rows, pack_pr85_bundle, parse_pr85_bundle, validate_pr85_member_name
from tac.stbm1br_mask_codec import STBM1BR_MAGIC
TOOL = 'experiments/build_pr85_stbm1br_rmb1_randmulti_candidate.py'
SCHEMA = 'pr85_stbm1br_rmb1_randmulti_candidate_summary_v1'
MANIFEST_SCHEMA = 'pr85_stbm1br_rmb1_randmulti_candidate_v1'
DEFAULT_STBM_ARCHIVE = REPO_ROOT / 'experiments/results/pr85_stbm1br_mask_recode_20260504_worker/pr90_stbm1br_lossless_pr85_mask_recode/archive.zip'
DEFAULT_PR92_ARCHIVE = REPO_ROOT / 'experiments/results/public_pr92_intake_20260504_codex/archive.zip'
DEFAULT_OUT_DIR = REPO_ROOT / 'experiments/results/pr85_stbm1br_rmb1_randmulti_20260504_codex'
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
EXPECTED_STBM_SHA256 = 'c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6'
EXPECTED_STBM_BYTES = 229756
EXPECTED_PR92_SHA256 = 'f0dedeb7ad3c019ab3f4e2dea317f3192408e47a573897adb208030709d01490'
DEFAULT_DISPATCH_LANE_ID = 'pr85_stbm1br_pr92_rmb1_randmulti'
DEFAULT_DISPATCH_PLATFORM = 'lightning'
DISPATCH_CLAIMS_PATH = '.omx/state/active_lane_dispatch_claims.md'
ROBUST_CURRENT_DIR = REPO_ROOT / 'submissions/robust_current'
CANONICAL_RMB1_BUILDER = 'experiments/build_pr85_stbm1br_pr92_rmb1_randmulti_candidate.py'

class Rmb1CandidateBuildError(ValueError):
    '''Raised when the candidate must fail closed.'''
    pass


def _json_text(payload = None):
    return json.dumps(payload, indent = 2, sort_keys = True, allow_nan = False) + '\n'


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



def _write_json(path = None, payload = None):
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(_json_text(payload), encoding = 'utf-8')


def _read_single_x_archive(path = None, *, allow_extra_members):
    if not path.is_file():
        raise Rmb1CandidateBuildError(f'''archive is missing: {_rel(path)}''')
    zf = zipfile.ZipFile(path, 'r')
# WARNING: Decompyle incomplete


def _zip_member_bytes(member_name = None, payload = None):
    buffer = io.BytesIO()
    info = zipfile.ZipInfo(member_name, FIXED_ZIP_TIMESTAMP)
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


def _write_single_x_archive(path = None, payload = None):
    first = _zip_member_bytes('x', payload)
    second = _zip_member_bytes('x', payload)
    if first != second:
        raise Rmb1CandidateBuildError('deterministic ZIP writer produced non-identical bytes')
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(first)
    (meta, readback) = _read_single_x_archive(path)
    if readback != payload:
        raise Rmb1CandidateBuildError('candidate archive readback differs from written payload')
    meta['deterministic_rewrite_identical'] = True
    return meta


def _segment_meta(segment = None, *, codec):
    return {
        'bytes': len(segment),
        'sha256': _sha256_bytes(segment),
        'magic_hex': segment[:8].hex(),
        'codec': codec }


def _robust_current_runtime_support_report(runtime_dir = None):
    """Check that the default submission runtime has this archive's decoders."""
    pass
# WARNING: Decompyle incomplete


def _module_string_constants(path = None, names = None):
    constants = { }
    tree = ast.parse(path.read_text(encoding = 'utf-8'), filename = str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if not target.id in names:
                continue
            constants[target.id] = node.value.value
    return constants


def _duplicate_builder_coordination_report():
    '''Fail closed if the legacy and canonical RMB1 builders drift lanes.'''
    canonical_path = REPO_ROOT / CANONICAL_RMB1_BUILDER
    checks = {
        'canonical_builder_exists': canonical_path.is_file() }
    constants = { }
    if canonical_path.is_file():
        constants = _module_string_constants(canonical_path, {
            'TOOL',
            'SCHEMA',
            'LANE_ID'})
        checks.update({
            'canonical_tool_matches_path': constants.get('TOOL') == CANONICAL_RMB1_BUILDER,
            'lane_id_matches_canonical_builder': constants.get('LANE_ID') == DEFAULT_DISPATCH_LANE_ID,
            'canonical_schema_is_pr92_rmb1': str(constants.get('SCHEMA', '')).startswith('pr85_stbm1br_pr92_rmb1_randmulti') })
    return {
        'schema': 'duplicate_rmb1_builder_coordination_v1',
        'status': 'passed' if all(checks.values()) else 'failed',
        'legacy_builder': TOOL,
        'canonical_builder': CANONICAL_RMB1_BUILDER,
        'legacy_lane_id': DEFAULT_DISPATCH_LANE_ID,
        'canonical_constants': constants,
        'checks': checks }


def _dispatch_claim_template(*, candidate_id, archive_sha256, manifest_path):
    command = [
        '.venv/bin/python',
        'tools/claim_lane_dispatch.py',
        'claim',
        '--claims-path',
        DISPATCH_CLAIMS_PATH,
        '--lane-id',
        DEFAULT_DISPATCH_LANE_ID,
        '--platform',
        DEFAULT_DISPATCH_PLATFORM,
        '--instance-job-id',
        f'''exact_eval_{candidate_id}_t4_${{UTC_STAMP}}''',
        '--agent',
        '${AGENT_ID}',
        '--predicted-eta-utc',
        '${PREDICTED_ETA_UTC}',
        '--status',
        'exact_eval_ready',
        '--notes',
        f'''archive_sha256={archive_sha256} manifest={_rel(manifest_path)}''']
    return {
        'claim_required': True,
        'command_not_executed_by_builder': True,
        'command_template': command,
        'lane_id': DEFAULT_DISPATCH_LANE_ID,
        'platform': DEFAULT_DISPATCH_PLATFORM,
        'claims_path': DISPATCH_CLAIMS_PATH,
        'status': 'exact_eval_ready',
        'required_placeholders': [
            'AGENT_ID',
            'PREDICTED_ETA_UTC',
            'UTC_STAMP'] }


def build_pr85_stbm1br_rmb1_randmulti_candidate(*, stbm_archive, pr92_archive, out_dir, candidate_id):
    (stbm_meta, stbm_raw) = _read_single_x_archive(stbm_archive)
    (pr92_meta, pr92_raw) = _read_single_x_archive(pr92_archive, allow_extra_members = True)
    stbm_bundle = parse_pr85_bundle(stbm_raw)
    pr92_bundle = parse_pr85_bundle(pr92_raw)
# WARNING: Decompyle incomplete


def build_arg_parser():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--stbm-archive', type = Path, default = DEFAULT_STBM_ARCHIVE)
    parser.add_argument('--pr92-archive', type = Path, default = DEFAULT_PR92_ARCHIVE)
    parser.add_argument('--out-dir', type = Path, default = DEFAULT_OUT_DIR)
    parser.add_argument('--candidate-id', default = 'pr85_stbm1br_plus_pr92_rmb1_randmulti')
    parser.add_argument('--stdout', action = 'store_true', help = 'Compatibility no-op; the builder always prints the JSON summary.')
    return parser


def main(argv = None):
    args = build_arg_parser().parse_args(argv)
    summary = build_pr85_stbm1br_rmb1_randmulti_candidate(stbm_archive = args.stbm_archive, pr92_archive = args.pr92_archive, out_dir = args.out_dir, candidate_id = args.candidate_id)
    print(_json_text(summary), end = '')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
