# Source Generated with Decompyle++
# File: test_build_contest_submission_packet.cpython-312.pyc (Python 3.12)

from __future__ import annotations
import hashlib
import importlib.util as importlib
import json
import os
import struct
import zipfile
from pathlib import Path
import pytest
REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / 'scripts' / 'build_contest_submission_packet.py'

def _load_module():
    spec = importlib.util.spec_from_file_location('build_contest_submission_packet', SCRIPT)
# WARNING: Decompyle incomplete


def _rpk1_payload_with_members(members = None):
    pass
# WARNING: Decompyle incomplete


def _write_artifact(root = None, *, device, gpu_t4_match, n_samples, include_adjudication, trace_all_match, include_sjkl_payload, include_cdo1_payload, include_amr1_payload, include_packed_cdo1_payload, sjkl_auth_log, extra_auth_log):
    root.mkdir(parents = True)
    archive = root / 'archive.zip'
    if include_sjkl_payload and include_cdo1_payload and include_amr1_payload or include_packed_cdo1_payload:
        zf = zipfile.ZipFile(archive, 'w')
        members = []
        if include_packed_cdo1_payload:
            import brotli
            packed = _rpk1_payload_with_members([
                ('renderer.bin', b'renderer'),
                ('masks.mkv', b'masks'),
                ('masks.cdo1.xz', b'charged packed overlay'),
                ('optimized_poses.bin', b'poses')])
            members.append(('p', brotli.compress(packed, quality = 11)))
        else:
            members.append(('p', b'deterministic archive bytes'))
        if include_sjkl_payload:
            members.append(('sjkl.bin', b'charged residual'))
        if include_cdo1_payload:
            members.append(('masks.cdo1.xz', b'charged overlay'))
        if include_amr1_payload:
            members.append(('alpha4_residual_repair.amr1.xz', b'charged repair'))
        for name, payload in members:
            info = zipfile.ZipInfo(name)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.external_attr = 27525120
            zf.writestr(info, payload, compress_type = zipfile.ZIP_STORED)
        None(None, None)
    else:
        zf = zipfile.ZipFile(archive, 'w', compression = zipfile.ZIP_STORED)
        info = zipfile.ZipInfo('x')
        info.date_time = (1980, 1, 1, 0, 0, 0)
        info.external_attr = 27525120
        zf.writestr(info, b'deterministic archive bytes')
        None(None, None)
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive_bytes = archive.stat().st_size
    avg_segnet = 0.00061244
    avg_posenet = 0.00049637
    score = 100 * avg_segnet + (10 * avg_posenet) ** 0.5 + 25 * archive_bytes / 37545489
    provenance = {
        'schema_version': 1,
        'started_at_utc': '2026-05-02T03:45:50Z',
        'tool': 'experiments/contest_auth_eval.py',
        'archive_path': '/remote/pact/archive.zip',
        'archive_sha256': archive_sha,
        'archive_size_bytes': archive_bytes,
        'device': device,
        'gpu_model': 'Tesla T4' if gpu_t4_match else 'L40S',
        'gpu_t4_match': gpu_t4_match,
        'cuda_available': device == 'cuda',
        'cuda_device_count': 1 if device == 'cuda' else 0,
        'inflate_timeout_seconds': 1800,
        'evaluate_timeout_seconds': 1800,
        'sys_argv': [
            'experiments/contest_auth_eval.py',
            '--device',
            device],
        'upstream_commit': '11ad728f563d8970929e8947a1cf6124ee6303e4' }
    contest = {
        'schema_version': 1,
        'final_score': round(score, 2),
        'avg_posenet_dist': avg_posenet,
        'avg_segnet_dist': avg_segnet,
        'score_recomputed_from_components': score,
        'score_pose_contribution': (10 * avg_posenet) ** 0.5,
        'score_seg_contribution': 100 * avg_segnet,
        'score_rate_contribution': 25 * archive_bytes / 37545489,
        'archive_size_bytes': archive_bytes,
        'n_samples': n_samples,
        'inflate_elapsed_seconds': 10,
        'evaluate_elapsed_seconds': 20,
        'contest_auth_eval_elapsed_seconds': 30,
        'provenance': provenance }
    (root / 'contest_auth_eval.json').write_text(json.dumps(contest, indent = 2, sort_keys = True) + '\n')
    (root / 'eval_provenance.json').write_text(json.dumps(provenance, indent = 2, sort_keys = True) + '\n')
    (root / 'report.txt').write_text('contest report\n')
# WARNING: Decompyle incomplete


def _attach_runtime_manifest(artifact = None, runtime = None):
    rows = []
    for rel in ('inflate.py', 'inflate.sh'):
        path = runtime / rel
        rows.append({
            'relative_path': rel,
            'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest() })
    payload = json.loads((artifact / 'contest_auth_eval.json').read_text())
    payload['provenance']['inflate_runtime_manifest'] = {
        'schema': 'contest_auth_eval_runtime_dependency_manifest_v1',
        'runtime_tree_sha256': 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
        'files': rows }
    (artifact / 'contest_auth_eval.json').write_text(json.dumps(payload, indent = 2, sort_keys = True) + '\n')
    trace = json.loads((artifact / 'component_trace.json').read_text())
    trace['contest_auth_eval_cross_check']['contest_auth_eval_json_sha256'] = hashlib.sha256((artifact / 'contest_auth_eval.json').read_bytes()).hexdigest()
    (artifact / 'component_trace.json').write_text(json.dumps(trace, indent = 2, sort_keys = True) + '\n')
    return {
        'runtime_tree_sha256': 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
        'files': rows }


def test_build_packet_writes_deterministic_manifest_and_checklist(tmp_path = None):
    mod = _load_module()
    artifact = tmp_path / 'artifact'
    expected = _write_artifact(artifact)
    output = tmp_path / 'packet'
    manifest = mod.build_packet(artifact, output, repo_root = tmp_path, expected_archive_sha256 = str(expected['archive_sha']), expected_archive_size_bytes = int(expected['archive_bytes']), expected_samples = 600)
    manifest_path = output / 'submission_packet_manifest.json'
    checklist_path = output / 'submission_packet_checklist.md'
    first_manifest = manifest_path.read_text()
    first_checklist = checklist_path.read_text()
    manifest_again = mod.build_packet(artifact, output, repo_root = tmp_path, expected_archive_sha256 = str(expected['archive_sha']), expected_archive_size_bytes = int(expected['archive_bytes']), expected_samples = 600)
# WARNING: Decompyle incomplete


def test_build_packet_copies_selected_runtime_archive_report_with_release_modes(tmp_path = None):
    mod = _load_module()
    artifact = tmp_path / 'artifact'
    expected = _write_artifact(artifact)
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    (runtime / 'inflate.py').write_text("print('runtime')\n")
    inflate = runtime / 'inflate.sh'
    inflate.write_text('#!/usr/bin/env bash\npython inflate.py\n')
    inflate.chmod(493)
    (runtime / 'README.md').write_text('not selected by auth runtime manifest\n')
    _attach_runtime_manifest(artifact, runtime)
    manifest = mod.build_packet(artifact, tmp_path / 'packet', repo_root = tmp_path, runtime_dir = runtime, expected_archive_sha256 = str(expected['archive_sha']), expected_archive_size_bytes = int(expected['archive_bytes']), expected_samples = 600, expected_lane_id = 'test_lane', expected_job_id = 'test_job')
    submission = tmp_path / 'packet' / 'submission'
# WARNING: Decompyle incomplete


def test_build_packet_normalizes_inflate_sh_executable_even_if_source_is_not(tmp_path = None):
    mod = _load_module()
    artifact = tmp_path / 'artifact'
    expected = _write_artifact(artifact)
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    (runtime / 'inflate.py').write_text("print('runtime')\n")
    inflate = runtime / 'inflate.sh'
    inflate.write_text('#!/usr/bin/env bash\npython inflate.py\n')
    inflate.chmod(420)
    _attach_runtime_manifest(artifact, runtime)
    manifest = mod.build_packet(artifact, tmp_path / 'packet', repo_root = tmp_path, runtime_dir = runtime, expected_archive_sha256 = str(expected['archive_sha']), expected_archive_size_bytes = int(expected['archive_bytes']), expected_samples = 600)
# WARNING: Decompyle incomplete


def test_build_packet_rejects_runtime_file_hash_mismatch(tmp_path = None):
    mod = _load_module()
    artifact = tmp_path / 'artifact'
    _write_artifact(artifact)
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    (runtime / 'inflate.py').write_text("print('runtime')\n")
    inflate = runtime / 'inflate.sh'
    inflate.write_text('#!/usr/bin/env bash\npython inflate.py\n')
    inflate.chmod(493)
    _attach_runtime_manifest(artifact, runtime)
    (runtime / 'inflate.py').write_text("print('changed')\n")
    pytest.raises(mod.PacketError, match = 'runtime file does not match auth eval manifest')
    mod.build_packet(artifact, tmp_path / 'packet', repo_root = tmp_path, runtime_dir = runtime)
    None(None, None)
    return None
    with None:
        if not None:
            pass


def test_build_packet_rejects_archive_sha_mismatch(tmp_path = None):
    mod = _load_module()
    artifact = tmp_path / 'artifact'
    _write_artifact(artifact)
    payload = json.loads((artifact / 'contest_auth_eval.json').read_text())
    payload['provenance']['archive_sha256'] = '0000000000000000000000000000000000000000000000000000000000000000'
    (artifact / 'contest_auth_eval.json').write_text(json.dumps(payload) + '\n')
    pytest.raises(mod.PacketError, match = 'archive_sha256_matches_contest_auth_eval')
    mod.build_packet(artifact, tmp_path / 'packet', repo_root = tmp_path)
    None(None, None)
    return None
    with None:
        if not None:
            pass


def test_build_packet_rejects_component_trace_cross_check_failure(tmp_path = None):
    mod = _load_module()
    artifact = tmp_path / 'artifact'
    _write_artifact(artifact, trace_all_match = False)
    pytest.raises(mod.PacketError, match = 'component_trace_cross_check')
    mod.build_packet(artifact, tmp_path / 'packet', repo_root = tmp_path)
    None(None, None)
    return None
    with None:
        if not None:
            pass


def test_build_packet_classifies_cuda_non_t4_without_adjudication_as_a(tmp_path = None):
    mod = _load_module()
    artifact = tmp_path / 'artifact'
    _write_artifact(artifact, gpu_t4_match = False, include_adjudication = False)
    manifest = mod.build_packet(artifact, tmp_path / 'packet', repo_root = tmp_path)
# WARNING: Decompyle incomplete


def test_build_packet_records_non_score_supporting_artifacts(tmp_path = None):
    mod = _load_module()
    artifact = tmp_path / 'artifact'
    expected = _write_artifact(artifact)
    adjudicated = json.loads((artifact / 'contest_auth_eval.adjudicated.json').read_text())
    adjudicated['adjudication_status'] = 'passed'
    (artifact / 'contest_auth_eval.adjudicated.json').write_text(json.dumps(adjudicated, indent = 2, sort_keys = True) + '\n')
    planner = tmp_path / 'plans' / 'planner.json'
    visualization = tmp_path / 'reports' / 'target_gap.svg'
    next_actions = tmp_path / 'docs' / 'next_actions.md'
    planner.parent.mkdir(parents = True)
    visualization.parent.mkdir(parents = True)
    next_actions.parent.mkdir(parents = True)
    planner.write_text('{"schema": "planner-ledger", "score_claim": false}\n')
    visualization.write_text('<svg><title>target gap</title></svg>\n')
    next_actions.write_text('# Next Action Tranche\n\nLocal byte win before exact eval.\n')
    manifest = mod.build_packet(artifact, tmp_path / 'packet', repo_root = tmp_path, score_authority = 'contest_auth_eval.adjudicated.json', expected_archive_sha256 = str(expected['archive_sha']), expected_archive_size_bytes = int(expected['archive_bytes']), expected_samples = 600, planner_ledgers = [
        planner], visualizations = [
        visualization], next_action_tranches = [
        next_actions])
    checklist = (tmp_path / 'packet' / 'submission_packet_checklist.md').read_text()
# WARNING: Decompyle incomplete


def test_build_packet_rejects_missing_supporting_artifact(tmp_path = None):
    mod = _load_module()
    artifact = tmp_path / 'artifact'
    _write_artifact(artifact)
    pytest.raises(mod.PacketError, match = 'missing planner ledger')
    mod.build_packet(artifact, tmp_path / 'packet', repo_root = tmp_path, planner_ledgers = [
        tmp_path / 'missing-planner.json'])
    None(None, None)
    return None
    with None:
        if not None:
            pass


def test_build_packet_rejects_charged_sjkl_without_apply_proof(tmp_path = None):
    mod = _load_module()
    artifact = tmp_path / 'artifact'
    _write_artifact(artifact, include_sjkl_payload = True)
    pytest.raises(mod.PacketError, match = 'sjkl_auth_eval_log_present')
    mod.build_packet(artifact, tmp_path / 'packet', repo_root = tmp_path)
    None(None, None)
    return None
    with None:
        if not None:
            pass


def test_build_packet_accepts_charged_sjkl_with_strict_apply_proof(tmp_path = None):
    mod = _load_module()
    artifact = tmp_path / 'artifact'
    _write_artifact(artifact, include_sjkl_payload = True, sjkl_auth_log = 'Loaded SJ-KL residual payload: 16 pairs, k=1, target=384x512, alpha_bits=3 (250 charged bytes)\nApplying SJ-KL residuals to JointFrameGenerator fake1 (384x512, device=cuda:0)\nSJ-KL strict contract passed: applied to 16 pair(s).\n')
    manifest = mod.build_packet(artifact, tmp_path / 'packet', repo_root = tmp_path)
    sjkl = manifest['archive_payload_contracts']['sjkl']
# WARNING: Decompyle incomplete


def test_build_packet_rejects_charged_cdo1_without_apply_proof(tmp_path = None):
    mod = _load_module()
    artifact = tmp_path / 'artifact'
    _write_artifact(artifact, include_cdo1_payload = True)
    pytest.raises(mod.PacketError, match = 'cdo1_auth_eval_log_present')
    mod.build_packet(artifact, tmp_path / 'packet', repo_root = tmp_path)
    None(None, None)
    return None
    with None:
        if not None:
            pass


def test_build_packet_accepts_charged_cdo1_with_apply_proof(tmp_path = None):
    mod = _load_module()
    artifact = tmp_path / 'artifact'
    _write_artifact(artifact, include_cdo1_payload = True, extra_auth_log = '  Applied CDO1 decoded-mask overlay masks.cdo1.xz: 1,024 raw bytes\n')
    manifest = mod.build_packet(artifact, tmp_path / 'packet', repo_root = tmp_path)
    cdo1 = manifest['archive_payload_contracts']['cdo1']
# WARNING: Decompyle incomplete


def test_build_packet_detects_packed_charged_cdo1_member(tmp_path = None):
    mod = _load_module()
    artifact = tmp_path / 'artifact'
    _write_artifact(artifact, include_packed_cdo1_payload = True, extra_auth_log = '  Applied CDO1 decoded-mask overlay masks.cdo1.xz: 2,048 raw bytes\n')
    manifest = mod.build_packet(artifact, tmp_path / 'packet', repo_root = tmp_path)
    cdo1 = manifest['archive_payload_contracts']['cdo1']
# WARNING: Decompyle incomplete


def test_build_packet_rejects_charged_amr1_without_apply_proof(tmp_path = None):
    mod = _load_module()
    artifact = tmp_path / 'artifact'
    _write_artifact(artifact, include_amr1_payload = True)
    pytest.raises(mod.PacketError, match = 'amr1_auth_eval_log_present')
    mod.build_packet(artifact, tmp_path / 'packet', repo_root = tmp_path)
    None(None, None)
    return None
    with None:
        if not None:
            pass


def test_build_packet_accepts_charged_amr1_with_apply_proof(tmp_path = None):
    mod = _load_module()
    artifact = tmp_path / 'artifact'
    _write_artifact(artifact, include_amr1_payload = True, extra_auth_log = '  Applied Alpha residual repair alpha4_residual_repair.amr1.xz: 16,384 raw AMR1 bytes\n')
    manifest = mod.build_packet(artifact, tmp_path / 'packet', repo_root = tmp_path)
    amr1 = manifest['archive_payload_contracts']['amr1']
# WARNING: Decompyle incomplete


def test_build_packet_rejects_duplicate_zip_member(tmp_path = None):
    mod = _load_module()
    artifact = tmp_path / 'artifact'
    _write_artifact(artifact)
    archive = artifact / 'archive.zip'
    zf = zipfile.ZipFile(archive, 'w', compression = zipfile.ZIP_STORED)
    zf.writestr('x', b'a')
    zf.writestr('x', b'b')
    None(None, None)
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive_bytes = archive.stat().st_size
    payload = json.loads((artifact / 'contest_auth_eval.json').read_text())
    payload['archive_size_bytes'] = archive_bytes
    payload['score_rate_contribution'] = 25 * archive_bytes / 37545489
    payload['score_recomputed_from_components'] = payload['score_seg_contribution'] + payload['score_pose_contribution'] + payload['score_rate_contribution']
    payload['provenance']['archive_sha256'] = archive_sha
    payload['provenance']['archive_size_bytes'] = archive_bytes
    (artifact / 'contest_auth_eval.json').write_text(json.dumps(payload, indent = 2) + '\n')
    pytest.raises(mod.PacketError, match = 'archive_no_duplicate_members')
    mod.build_packet(artifact, tmp_path / 'packet', repo_root = tmp_path)
    None(None, None)
    return None
    with None:
        if not None:
            pass
    continue
    with None:
        if not None:
            pass


def test_build_packet_rejects_multiple_packed_payload_containers(tmp_path = None):
    mod = _load_module()
    artifact = tmp_path / 'artifact'
    _write_artifact(artifact)
    archive = artifact / 'archive.zip'
    zf = zipfile.ZipFile(archive, 'w', compression = zipfile.ZIP_STORED)
    zf.writestr('p', b'a')
    zf.writestr('renderer_payload.bin.br', b'b')
    None(None, None)
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive_bytes = archive.stat().st_size
    payload = json.loads((artifact / 'contest_auth_eval.json').read_text())
    payload['archive_size_bytes'] = archive_bytes
    payload['score_rate_contribution'] = 25 * archive_bytes / 37545489
    payload['score_recomputed_from_components'] = payload['score_seg_contribution'] + payload['score_pose_contribution'] + payload['score_rate_contribution']
    payload['provenance']['archive_sha256'] = archive_sha
    payload['provenance']['archive_size_bytes'] = archive_bytes
    (artifact / 'contest_auth_eval.json').write_text(json.dumps(payload, indent = 2) + '\n')
    pytest.raises(mod.PacketError, match = 'archive_packed_payload_singleton')
    mod.build_packet(artifact, tmp_path / 'packet', repo_root = tmp_path)
    None(None, None)
    return None
    with None:
        if not None:
            pass
    continue
    with None:
        if not None:
            pass

