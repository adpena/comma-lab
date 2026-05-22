# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_production_contract import (
    FAIL_VERDICT,
    PASS_VERDICT,
    build_mlx_scorer_production_contract_manifest,
)
from tac.local_acceleration.mlx_scorer_response import GPU_BATCH_SHAPE_BLOCKER

REPO = Path(__file__).resolve().parents[3]


def test_mlx_production_contract_accepts_cpu_local_acceleration_signal() -> None:
    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
    )

    assert manifest["passed"] is True
    assert manifest["verdict"] == PASS_VERDICT
    assert manifest["production_deployment_role"] == "local_mlx_scorer_acceleration_non_authoritative"
    assert manifest["score_authority"] is False
    assert manifest["contest_authority"] is False
    assert manifest["required_gates"]["cache_auth_audit"] is True
    assert manifest["required_gates"]["torch_parity"] is True
    assert manifest["required_gates"]["profile_stability"] is True
    assert manifest["required_gates"]["batch_invariance"] is True


def test_mlx_production_contract_requires_gates_by_default() -> None:
    manifest = build_mlx_scorer_production_contract_manifest(_response_payload())

    assert manifest["passed"] is False
    assert "cache_auth_audit_manifest_not_supplied" in manifest["blockers"]
    assert "torch_parity_manifest_not_supplied" in manifest["blockers"]
    assert "profile_stability_manifest_not_supplied" in manifest["blockers"]
    assert "batch_invariance_manifest_not_supplied" in manifest["blockers"]


def test_mlx_production_contract_advisory_mode_warns_on_missing_gates() -> None:
    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        require_cache_auth_audit=False,
        require_torch_parity=False,
        require_profile_stability=False,
        require_batch_invariance=False,
    )

    assert manifest["passed"] is True
    assert "cache_auth_audit_manifest_not_supplied" in manifest["warnings"]
    assert "torch_parity_manifest_not_supplied" in manifest["warnings"]
    assert "profile_stability_manifest_not_supplied" in manifest["warnings"]
    assert "batch_invariance_manifest_not_supplied" in manifest["warnings"]


def test_mlx_production_contract_rejects_failing_supplied_gate() -> None:
    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        profile_stability=_profile_stability(passed=False),
        batch_invariance=_batch_invariance(),
    )

    assert manifest["passed"] is False
    assert "profile_stability_not_passing" in manifest["blockers"]


def test_mlx_production_contract_rejects_false_authority() -> None:
    payload = _response_payload()
    payload["score_claim"] = True
    payload["promotion_eligible"] = True

    manifest = build_mlx_scorer_production_contract_manifest(
        payload,
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
    )

    assert manifest["passed"] is False
    assert manifest["verdict"] == FAIL_VERDICT
    assert "response_score_claim_not_false" in manifest["blockers"]
    assert "response_promotion_eligible_not_false" in manifest["blockers"]


def test_mlx_production_contract_accepts_reference_array_hash_identity() -> None:
    payload = _response_payload()
    reference = payload["cache_identity"]["reference"]
    reference["archive_sha256"] = None
    reference["inflated_outputs_aggregate_sha256"] = None
    reference["raw_sha256"] = None
    reference["array_sha256"] = {
        "pair_indices": "3" * 64,
        "posenet_yuv6_pair": "4" * 64,
        "segnet_last_rgb": "5" * 64,
    }

    manifest = build_mlx_scorer_production_contract_manifest(
        payload,
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
    )

    assert manifest["passed"] is True


def test_mlx_production_contract_rejects_failing_cache_auth_audit() -> None:
    audit = _cache_auth_audit(passed=False)

    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=audit,
        torch_parity=_torch_parity(),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
    )

    assert manifest["passed"] is False
    assert "cache_auth_audit_not_passing" in manifest["blockers"]
    assert "cache_auth_audit_verdict_not_pass" in manifest["blockers"]


def test_mlx_production_contract_rejects_cache_auth_audit_identity_mismatch() -> None:
    audit = _cache_auth_audit(archive_sha256="0" * 64)

    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=audit,
        torch_parity=_torch_parity(),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
    )

    assert manifest["passed"] is False
    assert "cache_auth_audit_archive_sha256_mismatch" in manifest["blockers"]
    assert "cache_auth_audit_candidate_archive_sha256_mismatch" in manifest["blockers"]


def test_mlx_production_contract_rejects_failed_torch_parity() -> None:
    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(passed=False),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
    )

    assert manifest["passed"] is False
    assert "torch_parity_not_passing" in manifest["blockers"]
    assert "torch_parity_verdict_not_pass" in manifest["blockers"]


def test_mlx_production_contract_rejects_torch_parity_identity_mismatch() -> None:
    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(archive_sha256="0" * 64),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
    )

    assert manifest["passed"] is False
    assert "torch_parity_cache_identity_archive_sha256_mismatch" in manifest["blockers"]


def test_mlx_production_contract_can_require_score_calibration() -> None:
    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
        score_calibration=_score_calibration(),
        require_score_calibration=True,
    )

    assert manifest["passed"] is True
    assert manifest["required_gates"]["score_calibration"] is True


def test_mlx_production_contract_rejects_uncertain_score_calibration() -> None:
    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
        score_calibration=_score_calibration(uncertain_count=1),
        require_score_calibration=True,
    )

    assert manifest["passed"] is False
    assert "score_calibration_uncertain_pairwise_triage" in manifest["blockers"]


def test_mlx_production_contract_rejects_gpu_non_singleton_batch() -> None:
    payload = _response_payload(device="gpu", batch_pairs=2)

    manifest = build_mlx_scorer_production_contract_manifest(
        payload,
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
    )

    assert manifest["passed"] is False
    assert GPU_BATCH_SHAPE_BLOCKER in manifest["blockers"]


def test_mlx_production_contract_rejects_multi_pair_batch_invariance_mismatch() -> None:
    payload = _response_payload(device="cpu", batch_pairs=4)

    manifest = build_mlx_scorer_production_contract_manifest(
        payload,
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(device="cpu", batch_pairs=2),
    )

    assert manifest["passed"] is False
    assert "batch_invariance_batch_pairs_mismatch:response=4:gate=2" in manifest["blockers"]


def test_mlx_production_contract_rejects_multi_pair_batch_invariance_device_mismatch() -> None:
    payload = _response_payload(device="cpu", batch_pairs=4)

    manifest = build_mlx_scorer_production_contract_manifest(
        payload,
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(device="gpu", batch_pairs=4),
    )

    assert manifest["passed"] is False
    assert "batch_invariance_device_type_mismatch:response=cpu:gate=gpu" in manifest["blockers"]


def test_mlx_production_contract_cli_writes_manifest(tmp_path: Path) -> None:
    response_path = tmp_path / "response.json"
    cache_audit_path = tmp_path / "cache_audit.json"
    torch_parity_path = tmp_path / "torch_parity.json"
    stability_path = tmp_path / "stability.json"
    invariance_path = tmp_path / "invariance.json"
    out_path = tmp_path / "contract.json"
    response_path.write_text(json.dumps(_response_payload()), encoding="utf-8")
    cache_audit_path.write_text(json.dumps(_cache_auth_audit()), encoding="utf-8")
    torch_parity_path.write_text(json.dumps(_torch_parity()), encoding="utf-8")
    stability_path.write_text(json.dumps(_profile_stability()), encoding="utf-8")
    invariance_path.write_text(json.dumps(_batch_invariance()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "check_mlx_scorer_production_contract.py"),
            "--response",
            str(response_path),
            "--output",
            str(out_path),
            "--cache-auth-audit",
            str(cache_audit_path),
            "--torch-parity",
            str(torch_parity_path),
            "--profile-stability",
            str(stability_path),
            "--batch-invariance",
            str(invariance_path),
            "--run-id",
            "unit",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert '"passed": true' in completed.stdout
    manifest = json.loads(out_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "unit"
    assert manifest["passed"] is True


def _response_payload(*, device: str = "cpu", batch_pairs: int = 1) -> dict:
    return {
        "schema_version": "mlx_scorer_response.v1",
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_axis": EVIDENCE_TAG_MLX,
        "hardware_substrate": f"MLX {device}",
        "gpu_research_signal_allowed": device == "gpu",
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "canonical_score": 0.1,
        "score_recomputed_from_components": 0.1,
        "canonical_score_source": "score_recomputed_from_components",
        "avg_posenet_dist": 0.0,
        "avg_segnet_dist": 0.0,
        "batch_pairs": batch_pairs,
        "n_samples": 1,
        "pair_window": [0, 1],
        "components": {
            "posenet_shape": [1],
            "segnet_shape": [1],
            "posenet_sha256": "a" * 64,
            "segnet_sha256": "b" * 64,
        },
        "cache_identity": {
            "pair_indices_equal": True,
            "reference": {
                "archive_sha256": "c" * 64,
                "inflated_outputs_aggregate_sha256": "d" * 64,
                "raw_sha256": "e" * 64,
            },
            "candidate": {
                "archive_sha256": "f" * 64,
                "inflated_outputs_aggregate_sha256": "1" * 64,
                "raw_sha256": "2" * 64,
            },
        },
        "archive_sha256": "f" * 64,
        "inflated_outputs_aggregate_sha256": "1" * 64,
        "device_contract": {
            "gpu_research_signal_blocker": "mlx_gpu_scorer_response_requires_explicit_research_signal_allowance",
            "gpu_batch_shape_blocker": "mlx_gpu_scorer_response_requires_singleton_batches_until_invariance_passes",
            "forbidden_uses": ["auth_eval", "score_claim", "promotion", "rank_or_kill"],
        },
    }


def _profile_stability(*, passed: bool = True) -> dict:
    return {
        "schema_version": "mlx_scorer_response_profile_stability.v1",
        "passed": passed,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "candidate_generation_only": True,
    }


def _cache_auth_audit(*, passed: bool = True, archive_sha256: str = "f" * 64) -> dict:
    return {
        "schema_version": "mlx_scorer_input_cache_auth_eval_audit.v1",
        "passed": passed,
        "verdict": (
            "PASS_CACHE_AUTH_EVAL_IDENTITY"
            if passed
            else "FAIL_CACHE_AUTH_EVAL_IDENTITY"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "cache": {
            "archive_sha256": archive_sha256,
            "inflated_outputs_aggregate_sha256": "1" * 64,
            "raw_sha256": "2" * 64,
        },
        "allowed_use": (
            ["local_mlx_training_transfer_calibration"] if passed else ["debug_only"]
        ),
    }


def _torch_parity(*, passed: bool = True, archive_sha256: str = "f" * 64) -> dict:
    return {
        "schema_version": "mlx_scorer_torch_parity_sweep.v1",
        "passed": passed,
        "verdict": (
            "PASS_MLX_TORCH_SCORER_PARITY_SWEEP"
            if passed
            else "FAIL_MLX_TORCH_SCORER_PARITY_SWEEP"
        ),
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_axis": EVIDENCE_TAG_MLX,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "cache_identity": {
            "archive_sha256": archive_sha256,
        },
        "summary": {"failed_windows": 0 if passed else 1},
    }


def _score_calibration(*, uncertain_count: int = 0) -> dict:
    return {
        "schema_version": "mlx_score_calibration.v1",
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "decision_policy": {
            "allowed_use": "local_spend_triage_only_after_strict_auth_axis_calibration",
            "recommended_min_mlx_gap_for_spend_triage": 1.0e-4,
        },
        "summary": {
            "mlx_spend_triage_pairwise_uncertain_count": uncertain_count,
        },
    }


def _batch_invariance(*, passed: bool = True, device: str = "cpu", batch_pairs: int = 1) -> dict:
    return {
        "schema_version": "mlx_scorer_batch_invariance.v1",
        "passed": passed,
        "device_type": device,
        "batch_pairs": batch_pairs,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "candidate_generation_only": True,
    }
