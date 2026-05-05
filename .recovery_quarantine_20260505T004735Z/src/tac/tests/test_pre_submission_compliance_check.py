# Source Generated with Decompyle++
# File: test_pre_submission_compliance_check.cpython-312.pyc (Python 3.12)

from __future__ import annotations
import importlib.util as importlib
import json
import os
import zipfile
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / 'scripts' / 'pre_submission_compliance_check.py'

def _load_module():
    spec = importlib.util.spec_from_file_location('pre_submission_compliance_check', SCRIPT)
# WARNING: Decompyle incomplete


def _write_submission(root = None, *, unsafe_zip_member, include_runtime_tree, runtime_file_mismatch, top_level_runtime_manifest_only):
    root.mkdir(parents = True)
    archive = root / 'archive.zip'
    zf = zipfile.ZipFile(archive, 'w', compression = zipfile.ZIP_STORED)
    name = '../x' if unsafe_zip_member else 'x'
    info = zipfile.ZipInfo(name, date_time = (1980, 1, 1, 0, 0, 0))
    info.external_attr = 27525120
    zf.writestr(info, b'payload')
    None(None, None)
    inflate = root / 'inflate.sh'
    inflate.write_text('#!/usr/bin/env bash\nset -euo pipefail\n', encoding = 'utf-8')
    inflate.chmod(493)
    archive_sha = module_sha256(archive)
    archive_bytes = archive.stat().st_size
    zf = zipfile.ZipFile(archive)
    members = []
    for info in zf.infolist():
        members.append({
            'name': info.filename,
            'file_size': info.file_size,
            'sha256': module_bytes_sha256(zf.read(info.filename)) })
    None(None, None)
    seg = 0.00057185
    pose = 0.0001894
    score = 100 * seg + (10 * pose) ** 0.5 + 25 * archive_bytes / 37545489
    (root / 'report.txt').write_text(f'''report\narchive_sha256: {archive_sha}\narchive_size_bytes: {archive_bytes}\nscore_recomputed_from_components: {score}\n''', encoding = 'utf-8')
# WARNING: Decompyle incomplete


def module_sha256(path = None):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_bytes_sha256(payload = None):
    import hashlib
    return hashlib.sha256(payload).hexdigest()


def _rewrite_auth_archive_identity(auth_path = None, archive_path = None):
    payload = json.loads(auth_path.read_text(encoding = 'utf-8'))
    archive_sha = module_sha256(archive_path)
    archive_bytes = archive_path.stat().st_size
    payload['archive_size_bytes'] = archive_bytes
    payload['score_recomputed_from_components'] = 100 * payload['avg_segnet_dist'] + (10 * payload['avg_posenet_dist']) ** 0.5 + 25 * archive_bytes / 37545489
    payload['provenance']['archive_sha256'] = archive_sha
    payload['provenance']['archive_size_bytes'] = archive_bytes
    auth_path.write_text(json.dumps(payload, indent = 2) + '\n', encoding = 'utf-8')


def _rewrite_archive_manifest_identity(manifest_path = None, archive_path = None):
    archive_sha = module_sha256(archive_path)
    archive_bytes = archive_path.stat().st_size
    members = []
    zf = zipfile.ZipFile(archive_path)
    for info in zf.infolist():
        members.append({
            'name': info.filename,
            'file_size': info.file_size,
            'sha256': module_bytes_sha256(zf.read(info.filename)) })
    None(None, None)
    manifest_path.write_text(json.dumps({
        'schema': 'unit_test_archive_manifest_v1',
        'archive': {
            'sha256': archive_sha,
            'size_bytes': archive_bytes,
            'members': members } }, indent = 2, sort_keys = True) + '\n', encoding = 'utf-8')
    return None
    with None:
        if not None:
            pass
    continue


def test_pre_submission_check_passes_strict_happy_path(tmp_path = None):
    mod = _load_module()
    expected = _write_submission(tmp_path / 'submission')
    report = mod.build_report(mod.build_arg_parser().parse_args([
        '--submission-dir',
        str(tmp_path / 'submission'),
        '--auth-eval-json',
        str(tmp_path / 'submission' / 'contest_auth_eval.json'),
        '--require-auth-eval',
        '--require-t4-equivalent',
        '--expect-single-member',
        'x',
        '--expected-archive-sha256',
        str(expected['archive_sha256']),
        '--expected-archive-size-bytes',
        str(expected['archive_size_bytes']),
        '--expected-runtime-tree-sha256',
        'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa']))
# WARNING: Decompyle incomplete


def test_pre_submission_check_contest_final_implies_strict_gates(tmp_path = None):
    mod = _load_module()
    expected = _write_submission(tmp_path / 'submission')
    report = mod.build_report(mod.build_arg_parser().parse_args([
        '--submission-dir',
        str(tmp_path / 'submission'),
        '--auth-eval-json',
        str(tmp_path / 'submission' / 'contest_auth_eval.json'),
        '--contest-final',
        '--expected-archive-sha256',
        str(expected['archive_sha256']),
        '--expected-archive-size-bytes',
        str(expected['archive_size_bytes'])]))
# WARNING: Decompyle incomplete


def test_pre_submission_check_contest_final_requires_expected_archive_identity(tmp_path = None):
    mod = _load_module()
    _write_submission(tmp_path / 'submission')
    report = mod.build_report(mod.build_arg_parser().parse_args([
        '--submission-dir',
        str(tmp_path / 'submission'),
        '--auth-eval-json',
        str(tmp_path / 'submission' / 'contest_auth_eval.json'),
        '--contest-final']))
# WARNING: Decompyle incomplete


def test_pre_submission_check_contest_final_requires_submission_runtime_match(tmp_path = None):
    mod = _load_module()
    expected = _write_submission(tmp_path / 'submission', runtime_file_mismatch = True)
    report = mod.build_report(mod.build_arg_parser().parse_args([
        '--submission-dir',
        str(tmp_path / 'submission'),
        '--auth-eval-json',
        str(tmp_path / 'submission' / 'contest_auth_eval.json'),
        '--contest-final',
        '--expected-archive-sha256',
        str(expected['archive_sha256']),
        '--expected-archive-size-bytes',
        str(expected['archive_size_bytes'])]))
# WARNING: Decompyle incomplete


def test_pre_submission_check_runtime_match_flag_fails_without_contest_final(tmp_path = None):
    mod = _load_module()
    expected = _write_submission(tmp_path / 'submission', runtime_file_mismatch = True)
    report = mod.build_report(mod.build_arg_parser().parse_args([
        '--submission-dir',
        str(tmp_path / 'submission'),
        '--auth-eval-json',
        str(tmp_path / 'submission' / 'contest_auth_eval.json'),
        '--require-auth-eval',
        '--require-submission-runtime-match',
        '--expected-archive-sha256',
        str(expected['archive_sha256']),
        '--expected-archive-size-bytes',
        str(expected['archive_size_bytes'])]))
# WARNING: Decompyle incomplete


def test_pre_submission_check_runtime_match_accepts_top_level_manifest(tmp_path = None):
    mod = _load_module()
    expected = _write_submission(tmp_path / 'submission', top_level_runtime_manifest_only = True)
    report = mod.build_report(mod.build_arg_parser().parse_args([
        '--submission-dir',
        str(tmp_path / 'submission'),
        '--auth-eval-json',
        str(tmp_path / 'submission' / 'contest_auth_eval.json'),
        '--require-auth-eval',
        '--require-submission-runtime-match',
        '--expected-archive-sha256',
        str(expected['archive_sha256']),
        '--expected-archive-size-bytes',
        str(expected['archive_size_bytes'])]))
# WARNING: Decompyle incomplete


def test_pre_submission_check_rejects_malformed_expected_hash(tmp_path = None):
    mod = _load_module()
    expected = _write_submission(tmp_path / 'submission')
    report = mod.build_report(mod.build_arg_parser().parse_args([
        '--submission-dir',
        str(tmp_path / 'submission'),
        '--auth-eval-json',
        str(tmp_path / 'submission' / 'contest_auth_eval.json'),
        '--expected-archive-sha256',
        'not-a-sha',
        '--expected-archive-size-bytes',
        str(expected['archive_size_bytes'])]))
# WARNING: Decompyle incomplete


def test_pre_submission_check_allows_adjudicator_display_contribution_rounding(tmp_path = None):
    mod = _load_module()
    expected = _write_submission(tmp_path / 'submission')
    auth_path = tmp_path / 'submission' / 'contest_auth_eval.json'
    payload = json.loads(auth_path.read_text(encoding = 'utf-8'))
    payload['score_seg_contribution'] = 100 * payload['avg_segnet_dist']
    payload['score_pose_contribution'] = (10 * payload['avg_posenet_dist']) ** 0.5
    payload['score_rate_contribution'] = round(25 * payload['archive_size_bytes'] / 37545489, 6)
    payload['score_recomputed_from_components'] = payload['score_seg_contribution'] + payload['score_pose_contribution'] + payload['score_rate_contribution']
    auth_path.write_text(json.dumps(payload, indent = 2) + '\n', encoding = 'utf-8')
    report = mod.build_report(mod.build_arg_parser().parse_args([
        '--submission-dir',
        str(tmp_path / 'submission'),
        '--auth-eval-json',
        str(auth_path),
        '--require-auth-eval',
        '--require-t4-equivalent',
        '--expected-archive-sha256',
        str(expected['archive_sha256']),
        '--expected-archive-size-bytes',
        str(expected['archive_size_bytes'])]))
# WARNING: Decompyle incomplete


def test_pre_submission_check_fails_zip_slip_member(tmp_path = None):
    mod = _load_module()
    _write_submission(tmp_path / 'submission', unsafe_zip_member = True)
    report = mod.build_report(mod.build_arg_parser().parse_args([
        '--submission-dir',
        str(tmp_path / 'submission'),
        '--auth-eval-json',
        str(tmp_path / 'submission' / 'contest_auth_eval.json'),
        '--require-auth-eval']))
# WARNING: Decompyle incomplete


def test_pre_submission_check_fails_hidden_zip_member(tmp_path = None):
    mod = _load_module()
    _write_submission(tmp_path / 'submission')
    archive = tmp_path / 'submission' / 'archive.zip'
    zf = zipfile.ZipFile(archive, 'w', compression = zipfile.ZIP_STORED)
    zf.writestr('x', b'payload')
    zf.writestr('.env', b'SECRET=1\n')
    None(None, None)
    _rewrite_auth_archive_identity(tmp_path / 'submission' / 'contest_auth_eval.json', archive)
    report = mod.build_report(mod.build_arg_parser().parse_args([
        '--submission-dir',
        str(tmp_path / 'submission'),
        '--auth-eval-json',
        str(tmp_path / 'submission' / 'contest_auth_eval.json'),
        '--require-auth-eval']))
# WARNING: Decompyle incomplete


def test_pre_submission_check_fails_hidden_directory_even_with_single_member(tmp_path = None):
    mod = _load_module()
    _write_submission(tmp_path / 'submission')
    archive = tmp_path / 'submission' / 'archive.zip'
    zf = zipfile.ZipFile(archive, 'w', compression = zipfile.ZIP_STORED)
    zf.writestr('.cache/', b'')
    zf.writestr('x', b'payload')
    None(None, None)
    _rewrite_auth_archive_identity(tmp_path / 'submission' / 'contest_auth_eval.json', archive)
    report = mod.build_report(mod.build_arg_parser().parse_args([
        '--submission-dir',
        str(tmp_path / 'submission'),
        '--auth-eval-json',
        str(tmp_path / 'submission' / 'contest_auth_eval.json'),
        '--expect-single-member',
        'x']))
# WARNING: Decompyle incomplete


def test_pre_submission_check_contest_final_rejects_stale_archive_manifest(tmp_path = None):
    mod = _load_module()
    expected = _write_submission(tmp_path / 'submission')
    manifest_path = tmp_path / 'submission' / 'archive_manifest.json'
    payload = json.loads(manifest_path.read_text(encoding = 'utf-8'))
    payload['archive']['sha256'] = 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
    manifest_path.write_text(json.dumps(payload, indent = 2) + '\n', encoding = 'utf-8')
    report = mod.build_report(mod.build_arg_parser().parse_args([
        '--submission-dir',
        str(tmp_path / 'submission'),
        '--auth-eval-json',
        str(tmp_path / 'submission' / 'contest_auth_eval.json'),
        '--contest-final',
        '--expected-archive-sha256',
        str(expected['archive_sha256']),
        '--expected-archive-size-bytes',
        str(expected['archive_size_bytes'])]))
# WARNING: Decompyle incomplete


def test_pre_submission_check_contest_final_requires_report_archive_link(tmp_path = None):
    mod = _load_module()
    expected = _write_submission(tmp_path / 'submission')
    (tmp_path / 'submission' / 'report.txt').write_text('report without custody\n', encoding = 'utf-8')
    report = mod.build_report(mod.build_arg_parser().parse_args([
        '--submission-dir',
        str(tmp_path / 'submission'),
        '--auth-eval-json',
        str(tmp_path / 'submission' / 'contest_auth_eval.json'),
        '--contest-final',
        '--expected-archive-sha256',
        str(expected['archive_sha256']),
        '--expected-archive-size-bytes',
        str(expected['archive_size_bytes'])]))
# WARNING: Decompyle incomplete


def test_pre_submission_check_fails_multiple_packed_payload_containers(tmp_path = None):
    mod = _load_module()
    _write_submission(tmp_path / 'submission')
    archive = tmp_path / 'submission' / 'archive.zip'
    zf = zipfile.ZipFile(archive, 'w', compression = zipfile.ZIP_STORED)
    zf.writestr('p', b'payload-a')
    zf.writestr('renderer_payload.bin', b'payload-b')
    None(None, None)
    _rewrite_auth_archive_identity(tmp_path / 'submission' / 'contest_auth_eval.json', archive)
    _rewrite_archive_manifest_identity(tmp_path / 'submission' / 'archive_manifest.json', archive)
    report = mod.build_report(mod.build_arg_parser().parse_args([
        '--submission-dir',
        str(tmp_path / 'submission'),
        '--auth-eval-json',
        str(tmp_path / 'submission' / 'contest_auth_eval.json'),
        '--archive-manifest-json',
        str(tmp_path / 'submission' / 'archive_manifest.json'),
        '--require-auth-eval']))
# WARNING: Decompyle incomplete


def test_pre_submission_check_dispatch_claim_linkage_requires_terminal_row(tmp_path = None):
    mod = _load_module()
    expected = _write_submission(tmp_path / 'submission')
    claims = tmp_path / 'active_lane_dispatch_claims.md'
    claims.write_text('| ts | lane_id | platform | instance/job_id | status | notes |\n| 2026-05-04T00:00:00Z | lane-a | lightning | job-a | running | active |\n', encoding = 'utf-8')
    report = mod.build_report(mod.build_arg_parser().parse_args([
        '--submission-dir',
        str(tmp_path / 'submission'),
        '--auth-eval-json',
        str(tmp_path / 'submission' / 'contest_auth_eval.json'),
        '--require-auth-eval',
        '--archive-manifest-json',
        str(tmp_path / 'submission' / 'archive_manifest.json'),
        '--expected-archive-sha256',
        str(expected['archive_sha256']),
        '--expected-archive-size-bytes',
        str(expected['archive_size_bytes']),
        '--dispatch-claims-md',
        str(claims),
        '--expected-lane-id',
        'lane-a',
        '--expected-job-id',
        'job-a']))
# WARNING: Decompyle incomplete


def test_pre_submission_check_public_hygiene_flags_modal_provider_ids(tmp_path = None):
    mod = _load_module()
    _write_submission(tmp_path / 'submission')
    public_doc = tmp_path / 'submission' / 'supplement.md'
    public_doc.write_text('Modal call fc-01KQS22WSZ7YR3ZJYXVPPYE4VB app ap-KoGUy9mB8TVViZbp6BIoJX\n', encoding = 'utf-8')
    report = mod.build_report(mod.build_arg_parser().parse_args([
        '--submission-dir',
        str(tmp_path / 'submission'),
        '--public-scan-path',
        str(public_doc)]))
# WARNING: Decompyle incomplete


def test_pre_submission_check_fails_archive_override_mismatch(tmp_path = None):
    mod = _load_module()
    _write_submission(tmp_path / 'submission')
    _write_submission(tmp_path / 'other')
    zf = zipfile.ZipFile(tmp_path / 'other' / 'archive.zip', 'w', compression = zipfile.ZIP_STORED)
    zf.writestr('x', b'different-payload')
    None(None, None)
    _rewrite_auth_archive_identity(tmp_path / 'other' / 'contest_auth_eval.json', tmp_path / 'other' / 'archive.zip')
    report = mod.build_report(mod.build_arg_parser().parse_args([
        '--submission-dir',
        str(tmp_path / 'submission'),
        '--archive',
        str(tmp_path / 'other' / 'archive.zip'),
        '--auth-eval-json',
        str(tmp_path / 'other' / 'contest_auth_eval.json'),
        '--require-auth-eval']))
# WARNING: Decompyle incomplete


def test_pre_submission_check_fails_auth_eval_archive_identity_conflict(tmp_path = None):
    mod = _load_module()
    _write_submission(tmp_path / 'submission')
    auth_path = tmp_path / 'submission' / 'contest_auth_eval.json'
    payload = json.loads(auth_path.read_text(encoding = 'utf-8'))
    payload['archive_sha256'] = 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
    payload['archive_size_bytes'] = payload['archive_size_bytes'] + 1
    auth_path.write_text(json.dumps(payload, indent = 2) + '\n', encoding = 'utf-8')
    report = mod.build_report(mod.build_arg_parser().parse_args([
        '--submission-dir',
        str(tmp_path / 'submission'),
        '--auth-eval-json',
        str(auth_path),
        '--require-auth-eval']))
# WARNING: Decompyle incomplete


def test_pre_submission_check_fails_auth_eval_runtime_tree_conflict(tmp_path = None):
    mod = _load_module()
    _write_submission(tmp_path / 'submission')
    auth_path = tmp_path / 'submission' / 'contest_auth_eval.json'
    payload = json.loads(auth_path.read_text(encoding = 'utf-8'))
    payload['inflate_runtime_manifest'] = {
        'runtime_tree_sha256': 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb' }
    auth_path.write_text(json.dumps(payload, indent = 2) + '\n', encoding = 'utf-8')
    report = mod.build_report(mod.build_arg_parser().parse_args([
        '--submission-dir',
        str(tmp_path / 'submission'),
        '--auth-eval-json',
        str(auth_path),
        '--require-auth-eval']))
# WARNING: Decompyle incomplete


def test_pre_submission_check_fails_auth_eval_without_runtime_tree(tmp_path = None):
    mod = _load_module()
    _write_submission(tmp_path / 'submission', include_runtime_tree = False)
    report = mod.build_report(mod.build_arg_parser().parse_args([
        '--submission-dir',
        str(tmp_path / 'submission'),
        '--auth-eval-json',
        str(tmp_path / 'submission' / 'contest_auth_eval.json'),
        '--require-auth-eval']))
# WARNING: Decompyle incomplete


def test_pre_submission_check_detects_non_executable_inflate(tmp_path = None):
    mod = _load_module()
    _write_submission(tmp_path / 'submission')
    os.chmod(tmp_path / 'submission' / 'inflate.sh', 420)
    report = mod.build_report(mod.build_arg_parser().parse_args([
        '--submission-dir',
        str(tmp_path / 'submission')]))
# WARNING: Decompyle incomplete

