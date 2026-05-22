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
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
    )

    assert manifest["passed"] is True
    assert manifest["verdict"] == PASS_VERDICT
    assert manifest["production_deployment_role"] == "local_mlx_scorer_acceleration_non_authoritative"
    assert manifest["score_authority"] is False
    assert manifest["contest_authority"] is False
    assert manifest["required_gates"]["profile_stability"] is True
    assert manifest["required_gates"]["batch_invariance"] is True


def test_mlx_production_contract_requires_gates_by_default() -> None:
    manifest = build_mlx_scorer_production_contract_manifest(_response_payload())

    assert manifest["passed"] is False
    assert "profile_stability_manifest_not_supplied" in manifest["blockers"]
    assert "batch_invariance_manifest_not_supplied" in manifest["blockers"]


def test_mlx_production_contract_advisory_mode_warns_on_missing_gates() -> None:
    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        require_profile_stability=False,
        require_batch_invariance=False,
    )

    assert manifest["passed"] is True
    assert "profile_stability_manifest_not_supplied" in manifest["warnings"]
    assert "batch_invariance_manifest_not_supplied" in manifest["warnings"]


def test_mlx_production_contract_rejects_failing_supplied_gate() -> None:
    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
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
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
    )

    assert manifest["passed"] is False
    assert manifest["verdict"] == FAIL_VERDICT
    assert "response_score_claim_not_false" in manifest["blockers"]
    assert "response_promotion_eligible_not_false" in manifest["blockers"]


def test_mlx_production_contract_rejects_gpu_non_singleton_batch() -> None:
    payload = _response_payload(device="gpu", batch_pairs=2)

    manifest = build_mlx_scorer_production_contract_manifest(
        payload,
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
    )

    assert manifest["passed"] is False
    assert GPU_BATCH_SHAPE_BLOCKER in manifest["blockers"]


def test_mlx_production_contract_rejects_multi_pair_batch_invariance_mismatch() -> None:
    payload = _response_payload(device="cpu", batch_pairs=4)

    manifest = build_mlx_scorer_production_contract_manifest(
        payload,
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(device="cpu", batch_pairs=2),
    )

    assert manifest["passed"] is False
    assert "batch_invariance_batch_pairs_mismatch:response=4:gate=2" in manifest["blockers"]


def test_mlx_production_contract_rejects_multi_pair_batch_invariance_device_mismatch() -> None:
    payload = _response_payload(device="cpu", batch_pairs=4)

    manifest = build_mlx_scorer_production_contract_manifest(
        payload,
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(device="gpu", batch_pairs=4),
    )

    assert manifest["passed"] is False
    assert "batch_invariance_device_type_mismatch:response=cpu:gate=gpu" in manifest["blockers"]


def test_mlx_production_contract_cli_writes_manifest(tmp_path: Path) -> None:
    response_path = tmp_path / "response.json"
    stability_path = tmp_path / "stability.json"
    invariance_path = tmp_path / "invariance.json"
    out_path = tmp_path / "contract.json"
    response_path.write_text(json.dumps(_response_payload()), encoding="utf-8")
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
