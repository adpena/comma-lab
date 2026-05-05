# Source Generated with Decompyle++
# File: test_preflight_candidate_manifest_dispatch_readiness.cpython-312.pyc (Python 3.12)

from __future__ import annotations
import importlib.util as importlib
import json
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / 'experiments' / 'preflight_candidate_manifest_dispatch_readiness.py'

def _load_script():
    spec = importlib.util.spec_from_file_location('candidate_manifest_dispatch_readiness_test', SCRIPT)
# WARNING: Decompyle incomplete

module = _load_script()

def _write_manifest(path = None, payload = None):
    path.write_text(json.dumps(payload, sort_keys = True) + '\n', encoding = 'utf-8')
    return path


def test_blocks_builder_specific_local_only_dispatch_gate(tmp_path = None):
    manifest = _write_manifest(tmp_path / 'manifest.json', {
        'candidate_id': 'stack_candidate',
        'score_claim': False,
        'dispatch_gate': 'blocked_local_only_until_standalone_exact_positives_and_lane_claim',
        'dispatch_unlocked': False })
    payload = module.build_preflight(manifest)
# WARNING: Decompyle incomplete


def test_blocks_unsafe_exact_eval_dispatch_gate(tmp_path = None):
    manifest = _write_manifest(tmp_path / 'manifest.json', {
        'candidate_id': 'renderer_transplant',
        'score_claim': False,
        'exact_eval_dispatch_gate': {
            'required': True,
            'status': 'missing_pose_safety_report',
            'safe_for_exact_eval_dispatch': False,
            'blockers': [
                'missing renderer transplant pose-safety preflight'] } })
    payload = module.build_preflight(manifest)
# WARNING: Decompyle incomplete


def test_blocks_nested_exact_runtime_contract_blockers(tmp_path = None):
    manifest = _write_manifest(tmp_path / 'manifest.json', {
        'candidate_id': 'stbm_candidate',
        'score_claim': False,
        'dispatch_gate': 'eligible_for_exact_eval_after_lane_claim',
        'exact_eval_runtime_contract': {
            'ready_for_exact_eval_runtime': False,
            'remaining_blockers': [
                {
                    'code': 'exact_runtime:missing_explicit_pr85_replay_runtime' }] } })
    payload = module.build_preflight(manifest)
# WARNING: Decompyle incomplete


def test_blocks_runtime_changing_manifest_missing_exact_runtime_contract(tmp_path = None):
    manifest = _write_manifest(tmp_path / 'manifest.json', {
        'candidate_id': 'pr90_stbm1br_lossless_pr85_mask_recode',
        'score_claim': False,
        'fail_closed_preflight': {
            'status': 'passed',
            'exact_eval_requires_lane_claim': True },
        'runtime_support': {
            'support_scope': 'local_runtime_only',
            'format': 'STBM1BR' } })
    payload = module.build_preflight(manifest)
# WARNING: Decompyle incomplete


def test_blocks_formula_only_qfq4_model_recode_without_runtime_loader(tmp_path = None):
    manifest = _write_manifest(tmp_path / 'qfq4_readiness.json', {
        'candidate_id': 'pr85_stbm1br_qfq4_model_recode',
        'score_claim': False,
        'dispatch_gate': 'eligible_for_exact_eval_after_lane_claim',
        'exact_eval_readiness': {
            'ready': True,
            'requires_lane_claim_before_remote_eval': True },
        'qfq4_screen': {
            'best_byte_screen': {
                'candidate_id': 'qfq4_pr85_shifted_int8_rows',
                'archive_delta_bytes_vs_source_formula': -659,
                'rate_score_delta_if_components_identical_formula_only': -0.000438801,
                'decoded_tensor_parity': {
                    'decoded_tensor_parity': False },
                'runtime_compatibility': {
                    'runtime_can_decode_without_edits': False,
                    'public_pr85_replay_qfq4_model_loader': False } } } })
    payload = module.build_preflight(manifest)
# WARNING: Decompyle incomplete


def test_blocks_hpm1_projection_when_decode_parity_failed(tmp_path = None):
    manifest = _write_manifest(tmp_path / 'hpm1_readiness.json', {
        'candidate_id': 'pr91_hpm1_parity_recovery',
        'score_claim': False,
        'dispatch_gate': 'eligible_for_exact_eval_after_lane_claim',
        'byte_faithful_fusion': {
            'rate_score_delta_if_archive_is_valid': -0.0048954,
            'component_identity_required': True },
        'hpm1_replay_gate': {
            'status': 'failed_closed',
            'failure_reason': 'hpac_entropy_decode_contract_mismatch',
            'full_decode_byte_parity_proven': False },
        'score_projection_if_hpm1_mask_is_semantically_identical': {
            'evidence_grade': 'prediction',
            'score_claim': False } })
    payload = module.build_preflight(manifest)
# WARNING: Decompyle incomplete


def test_allows_eligible_manifest_but_warns_lane_claim_required(tmp_path = None):
    manifest = _write_manifest(tmp_path / 'manifest.json', {
        'candidate_id': 'byte_closed_candidate',
        'score_claim': False,
        'dispatch_gate': 'eligible_for_cuda_auth_eval_after_lane_claim',
        'dispatch_unlocked': True,
        'ready_for_exact_eval_dispatch_claim': True,
        'fixed_runtime_preflight': {
            'ready_for_fixed_runtime_exact_eval': True,
            'remaining_blockers': [] } })
    payload = module.build_preflight(manifest)
# WARNING: Decompyle incomplete


def test_allows_explanatory_rate_projection_when_runtime_contract_is_ready(tmp_path = None):
    manifest = _write_manifest(tmp_path / 'manifest.json', {
        'candidate_id': 'pr85_stbm1br_plus_pr92_rmb1_randmulti_recode',
        'score_claim': False,
        'dispatch_readiness': {
            'lane_id': 'pr85_stbm1br_pr92_rmb1_randmulti',
            'checks': {
                'strict_zip_single_member_x': True,
                'candidate_changes_only_randmulti_vs_stbm': True,
                'randmulti_decoded_rows_match_stbm': True,
                'candidate_score_claim_false': True } },
        'exact_eval_runtime_contract': {
            'ready_for_exact_eval_runtime': True,
            'runtime_tree_sha256': 'abc123',
            'remaining_blockers': [] },
        'formula_only_rate_delta_vs_stbm': {
            'archive_delta_bytes': -276,
            'score_delta_from_rate_only': -0.000183778,
            'score_claim': False } })
    payload = module.build_preflight(manifest)
# WARNING: Decompyle incomplete


def test_blocks_active_same_lane_claim_when_claims_path_is_supplied(tmp_path = None):
    manifest = _write_manifest(tmp_path / 'manifest.json', {
        'candidate_id': 'pr85_stbm1br_plus_pr92_rmb1_randmulti_recode',
        'score_claim': False,
        'dispatch_readiness': {
            'lane_id': 'pr85_stbm1br_pr92_rmb1_randmulti',
            'checks': {
                'candidate_score_claim_false': True } },
        'exact_eval_runtime_contract': {
            'ready_for_exact_eval_runtime': True,
            'runtime_tree_sha256': 'abc123',
            'remaining_blockers': [] } })
    claims = tmp_path / 'active_lane_dispatch_claims.md'
    claims.write_text('# claims\n\n| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n|---|---|---|---|---|---|---|---|\n| 2026-05-04T08:22:20Z | codex:gpt-5.5 | pr85_stbm1br_pr92_rmb1_randmulti | lightning | exact_eval_active | 2026-05-04T09:37:20Z | eval | active |\n', encoding = 'utf-8')
    payload = module.build_preflight(manifest, claims_path = claims, now_utc = '2026-05-04T08:26:36Z')
# WARNING: Decompyle incomplete


def test_terminal_newer_claim_closes_older_active_same_job(tmp_path = None):
    manifest = _write_manifest(tmp_path / 'manifest.json', {
        'candidate_id': 'pr85_stbm1br_plus_pr92_rmb1_randmulti_recode',
        'score_claim': False,
        'dispatch_readiness': {
            'lane_id': 'pr85_stbm1br_pr92_rmb1_randmulti',
            'checks': {
                'candidate_score_claim_false': True } },
        'exact_eval_runtime_contract': {
            'ready_for_exact_eval_runtime': True,
            'runtime_tree_sha256': 'abc123',
            'remaining_blockers': [] } })
    claims = tmp_path / 'active_lane_dispatch_claims.md'
    claims.write_text('# claims\n\n| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |\n|---|---|---|---|---|---|---|---|\n| 2026-05-04T08:30:00Z | codex:gpt-5.5 | pr85_stbm1br_pr92_rmb1_randmulti | lightning | exact_eval_closed | 2026-05-04T08:30:00Z | completed_score_0.2535 | done |\n| 2026-05-04T08:22:20Z | codex:gpt-5.5 | pr85_stbm1br_pr92_rmb1_randmulti | lightning | exact_eval_closed | 2026-05-04T09:37:20Z | eval | old active |\n', encoding = 'utf-8')
    payload = module.build_preflight(manifest, claims_path = claims, now_utc = '2026-05-04T08:31:00Z')
# WARNING: Decompyle incomplete


def test_cli_fail_if_not_ready_returns_two_and_writes_report(tmp_path = None, capsys = None):
    manifest = _write_manifest(tmp_path / 'manifest.json', {
        'candidate_id': 'planning_candidate',
        'score_claim': False,
        'dispatch_gate': 'planning_only/no_remote_dispatch' })
    out = tmp_path / 'report.json'
    rc = module.main([
        '--manifest',
        str(manifest),
        '--json-out',
        str(out),
        '--fail-if-not-ready'])
# WARNING: Decompyle incomplete

