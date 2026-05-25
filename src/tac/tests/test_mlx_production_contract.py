# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.auth_eval_schema import contest_formula_score
from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_production_contract import (
    ADVISORY_VERDICT,
    BUNDLE_FAIL_VERDICT,
    BUNDLE_PASS_VERDICT,
    FAIL_VERDICT,
    GATE_SET_VERSION,
    PASS_VERDICT,
    build_mlx_scorer_production_contract_bundle_manifest,
    build_mlx_scorer_production_contract_manifest,
)
from tac.local_acceleration.mlx_scorer_response import GPU_BATCH_SHAPE_BLOCKER

REPO = Path(__file__).resolve().parents[3]


def test_mlx_production_contract_accepts_cpu_local_acceleration_signal() -> None:
    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        reference_torch_parity=_torch_parity(side="reference"),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
        conv2d_accumulation_probe=_conv2d_accumulation_probe(),
        downstream_scorer_drift=_downstream_scorer_drift(),
    )

    assert manifest["passed"] is True
    assert manifest["verdict"] == PASS_VERDICT
    assert manifest["gate_set_version"] == GATE_SET_VERSION
    assert manifest["production_deployment_role"] == "local_mlx_scorer_acceleration_non_authoritative"
    assert manifest["score_authority"] is False
    assert manifest["contest_authority"] is False
    assert manifest["score_claim"] is False
    assert manifest["score_claim_valid"] is False
    assert manifest["candidate_generation_only"] is True
    assert manifest["requires_exact_eval_before_promotion"] is True
    assert manifest["score_axis"] == EVIDENCE_TAG_MLX
    assert manifest["required_gates"]["cache_auth_audit"] is True
    assert manifest["required_gates"]["torch_parity"] is True
    assert manifest["required_gates"]["profile_stability"] is True
    assert manifest["response_summary"]["archive_size_bytes"] == 123
    assert manifest["response_summary"]["canonical_score"] == _fixture_score()
    assert manifest["response_summary"]["avg_posenet_dist"] == 0.0
    assert manifest["response_summary"]["avg_segnet_dist"] == 0.0
    formula = manifest["response_summary"]["score_formula_recompute"]
    assert formula["passed"] is True
    assert formula["blocker"] is None
    assert formula["expected_score"] == _fixture_score()
    assert formula["actual_score"] == _fixture_score()
    assert formula["abs_delta"] == 0.0
    assert manifest["response_summary"]["candidate_cache_array_sha256"] == {
        "pair_indices": "9" * 64,
        "posenet_yuv6_pair": "a" * 64,
        "segnet_last_rgb": "b" * 64,
    }
    assert manifest["response_summary"]["reference_cache_array_sha256"] == {
        "pair_indices": "6" * 64,
        "posenet_yuv6_pair": "7" * 64,
        "segnet_last_rgb": "8" * 64,
    }
    assert manifest["response_summary"]["posenet_sha256"] == "a" * 64
    assert manifest["response_summary"]["segnet_sha256"] == "b" * 64
    assert manifest["required_gates"]["batch_invariance"] is False
    assert manifest["required_gates"]["batch_invariance_policy_requested"] is True
    assert manifest["required_gates"]["conv2d_accumulation_probe"] is False
    assert (
        manifest["numerical_mitigation_summary"]["conv2d_accumulation_probe"]["passed"]
        is True
    )
    assert (
        manifest["numerical_mitigation_summary"]["downstream_scorer_drift"][
            "aggregate_verdict"
        ]
        == "BELOW_SCORER_PRECISION"
    )
    assert manifest["numerical_mitigation_summary"]["downstream_scorer_drift"][
        "stages"
    ][0]["stage_name"] == "stage_5_contest_score_aggregation"


def test_mlx_production_contract_rejects_failed_conv2d_accumulation_probe() -> None:
    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        reference_torch_parity=_torch_parity(side="reference"),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
        conv2d_accumulation_probe=_conv2d_accumulation_probe(passed=False),
    )

    assert manifest["passed"] is False
    assert "conv2d_accumulation_probe_not_passing" in manifest["blockers"]
    assert "conv2d_accumulation_probe_verdict_not_pass" in manifest["blockers"]
    assert "conv2d_accumulation_probe_mode_not_passing:fixed_fp64" in manifest["blockers"]


def test_mlx_production_contract_can_require_conv2d_accumulation_probe() -> None:
    missing = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        reference_torch_parity=_torch_parity(side="reference"),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
        require_conv2d_accumulation_probe=True,
    )

    assert missing["passed"] is False
    assert "conv2d_accumulation_probe_manifest_not_supplied" in missing["blockers"]
    assert missing["required_gates"]["conv2d_accumulation_probe"] is True

    present = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        reference_torch_parity=_torch_parity(side="reference"),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
        conv2d_accumulation_probe=_conv2d_accumulation_probe(),
        require_conv2d_accumulation_probe=True,
    )

    assert present["passed"] is True
    assert present["required_gates"]["conv2d_accumulation_probe"] is True


def test_mlx_production_contract_can_require_downstream_scorer_drift() -> None:
    missing = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        reference_torch_parity=_torch_parity(side="reference"),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
        require_downstream_scorer_drift=True,
    )

    assert missing["passed"] is False
    assert "downstream_scorer_drift_manifest_not_supplied" in missing["blockers"]
    assert missing["required_gates"]["downstream_scorer_drift"] is True

    present = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        reference_torch_parity=_torch_parity(side="reference"),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
        downstream_scorer_drift=_downstream_scorer_drift(),
        require_downstream_scorer_drift=True,
    )

    assert present["passed"] is True
    assert present["required_gates"]["downstream_scorer_drift"] is True


def test_mlx_production_contract_rejects_failed_downstream_scorer_drift() -> None:
    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        reference_torch_parity=_torch_parity(side="reference"),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
        downstream_scorer_drift=_downstream_scorer_drift(
            aggregate_verdict="ABOVE_SCORER_PRECISION",
            drift_units=0.02,
        ),
    )

    assert manifest["passed"] is False
    assert "downstream_scorer_drift_verdict_not_below_precision" in manifest["blockers"]
    assert any(
        blocker.startswith("downstream_scorer_drift_units_exceed_threshold")
        for blocker in manifest["blockers"]
    )


def test_mlx_production_contract_preserves_downstream_failure_routing() -> None:
    drift = _downstream_scorer_drift(
        aggregate_verdict="ABOVE_SCORER_PRECISION",
        drift_units=0.02,
    )
    drift["operator_routable_per_verdict"] = (
        "run kahan/fp64/MLX determinism/cudnn mitigation ladder"
    )
    drift["stages"] = [
        {
            "stage_name": "stage_5_contest_score_aggregation",
            "metric": "aggregate_contest_score_drift_due_to_mlx_vs_pytorch_framework",
            "max_abs": 0.02,
            "mean_abs": 0.02,
            "rms": 0.02,
            "extra": {"pose_contest_delta_units": 0.02},
        }
    ]
    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        reference_torch_parity=_torch_parity(side="reference"),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
        downstream_scorer_drift=drift,
    )

    summary = manifest["numerical_mitigation_summary"]["downstream_scorer_drift"]
    assert summary["operator_routable_per_verdict"] == (
        "run kahan/fp64/MLX determinism/cudnn mitigation ladder"
    )
    assert summary["stages"][0]["stage_name"] == "stage_5_contest_score_aggregation"
    assert summary["stages"][0]["extra"]["pose_contest_delta_units"] == 0.02


def test_mlx_production_contract_rejects_unbound_downstream_scorer_drift() -> None:
    cases = [
        (
            "downstream_scorer_drift_checkpoint_mode_not_trained_archive",
            _downstream_scorer_drift(checkpoint_mode="random_init"),
        ),
        (
            "downstream_scorer_drift_archive_sha256_mismatch",
            _downstream_scorer_drift(archive_sha256="0" * 64),
        ),
        (
            "downstream_scorer_drift_posenet_sha256_mismatch",
            _downstream_scorer_drift(posenet_sha256="0" * 64),
        ),
        (
            "downstream_scorer_drift_segnet_sha256_mismatch",
            _downstream_scorer_drift(segnet_sha256="0" * 64),
        ),
        (
            "downstream_scorer_drift_units_negative",
            _downstream_scorer_drift(drift_units=-1.0),
        ),
        (
            "downstream_scorer_drift_pair_coverage_below_minimum:1<100",
            _downstream_scorer_drift(n_pairs_actual=1, covered_pair_window=[0, 1]),
        ),
        (
            "downstream_scorer_drift_response_pair_window_not_covered",
            _downstream_scorer_drift(covered_pair_window=[10, 110]),
        ),
    ]

    for expected_blocker, drift in cases:
        manifest = build_mlx_scorer_production_contract_manifest(
            _response_payload(),
            cache_auth_audit=_cache_auth_audit(),
            torch_parity=_torch_parity(),
            reference_torch_parity=_torch_parity(side="reference"),
            profile_stability=_profile_stability(),
            batch_invariance=_batch_invariance(),
            downstream_scorer_drift=drift,
        )

        assert manifest["passed"] is False
        assert expected_blocker in manifest["blockers"]


def test_mlx_production_contract_bundle_rejects_stale_no_downstream_gate() -> None:
    contract = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        reference_torch_parity=_torch_parity(side="reference"),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
        score_calibration=_score_calibration(),
        conv2d_accumulation_probe=_conv2d_accumulation_probe(),
        require_score_calibration=True,
        require_conv2d_accumulation_probe=True,
    )

    bundle = build_mlx_scorer_production_contract_bundle_manifest([contract])

    assert bundle["passed"] is False
    assert "mlx_production_contract_bundle_child_blocked:0" in bundle["blockers"]
    assert (
        "required_gate_downstream_scorer_drift_not_true"
        in bundle["summary"]["child_contracts"][0]["blockers"]
    )


def test_mlx_production_contract_bundle_accepts_strict_contracts() -> None:
    contract = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        reference_torch_parity=_torch_parity(side="reference"),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
        score_calibration=_score_calibration(),
        conv2d_accumulation_probe=_conv2d_accumulation_probe(),
        downstream_scorer_drift=_downstream_scorer_drift(),
        require_score_calibration=True,
        require_conv2d_accumulation_probe=True,
        require_downstream_scorer_drift=True,
    )

    bundle = build_mlx_scorer_production_contract_bundle_manifest(
        [contract],
        run_id="unit-bundle",
    )

    assert bundle["schema"] == "mlx_scorer_production_contract_bundle.v1"
    assert bundle["passed"] is True
    assert bundle["verdict"] == BUNDLE_PASS_VERDICT
    assert bundle["score_claim"] is False
    assert bundle["summary"]["contract_count"] == 1
    assert bundle["summary"]["strict_contract_count"] == 1
    assert bundle["contracts"][0]["run_id"] == contract["run_id"]


def test_mlx_production_contract_bundle_blocks_failed_contract() -> None:
    contract = build_mlx_scorer_production_contract_manifest(_response_payload())

    bundle = build_mlx_scorer_production_contract_bundle_manifest([contract])

    assert bundle["passed"] is False
    assert bundle["verdict"] == BUNDLE_FAIL_VERDICT
    assert "mlx_production_contract_bundle_child_blocked:0" in bundle["blockers"]
    assert bundle["summary"]["strict_contract_count"] == 0


def test_mlx_production_contract_requires_gates_by_default() -> None:
    manifest = build_mlx_scorer_production_contract_manifest(_response_payload())

    assert manifest["passed"] is False
    assert "cache_auth_audit_manifest_not_supplied" in manifest["blockers"]
    assert "torch_parity_manifest_not_supplied" in manifest["blockers"]
    assert "reference_torch_parity_manifest_not_supplied" in manifest["blockers"]
    assert "profile_stability_manifest_not_supplied" in manifest["blockers"]
    assert "batch_invariance_manifest_not_supplied" not in manifest["blockers"]
    assert "batch_invariance_not_required_for_singleton_response" in manifest["warnings"]


def test_mlx_production_contract_requires_batch_invariance_for_multi_pair_response() -> None:
    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(batch_pairs=4),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(batch_pairs=4),
        profile_stability=_profile_stability(batch_pairs=4),
    )

    assert manifest["passed"] is False
    assert "batch_invariance_manifest_not_supplied" in manifest["blockers"]
    assert manifest["required_gates"]["batch_invariance"] is True


def test_mlx_production_contract_advisory_mode_warns_on_missing_gates() -> None:
    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        require_cache_auth_audit=False,
        require_torch_parity=False,
        require_profile_stability=False,
        require_batch_invariance=False,
    )

    assert manifest["passed"] is False
    assert manifest["advisory_passed"] is True
    assert manifest["verdict"] == ADVISORY_VERDICT
    assert "cache_auth_audit_manifest_not_supplied" in manifest["warnings"]
    assert "torch_parity_manifest_not_supplied" in manifest["warnings"]
    assert "profile_stability_manifest_not_supplied" in manifest["warnings"]
    assert "batch_invariance_manifest_not_supplied" in manifest["warnings"]
    assert "production_required_gate_policy_bypassed" in manifest["warnings"]


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
        reference_torch_parity=_torch_parity(side="reference"),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
    )

    assert manifest["passed"] is False
    assert manifest["verdict"] == FAIL_VERDICT
    assert "response_score_claim_not_false" in manifest["blockers"]
    assert "response_promotion_eligible_not_false" in manifest["blockers"]


def test_mlx_production_contract_rejects_self_consistent_wrong_score_formula() -> None:
    payload = _response_payload()
    payload["canonical_score"] = 0.1
    payload["score_recomputed_from_components"] = 0.1

    manifest = build_mlx_scorer_production_contract_manifest(
        payload,
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        reference_torch_parity=_torch_parity(side="reference"),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
    )

    assert manifest["passed"] is False
    assert any(
        blocker.startswith("response_score_formula_recompute_mismatch")
        for blocker in manifest["blockers"]
    )


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
    reference_parity = _torch_parity(side="reference")
    reference_parity["cache_identity"]["archive_sha256"] = None
    reference_parity["cache_identity"]["inflated_outputs_aggregate_sha256"] = None
    reference_parity["cache_identity"]["raw_sha256"] = None
    reference_parity["cache_identity"]["array_sha256"] = reference["array_sha256"]

    manifest = build_mlx_scorer_production_contract_manifest(
        payload,
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        reference_torch_parity=reference_parity,
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
        reference_torch_parity=_torch_parity(side="reference"),
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


def test_mlx_production_contract_rejects_minimal_cache_auth_audit() -> None:
    audit = _cache_auth_audit()
    audit.pop("canonical_equation")
    audit["cache"].pop("array_sha256")

    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=audit,
        torch_parity=_torch_parity(),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
    )

    assert manifest["passed"] is False
    assert "cache_auth_audit_canonical_equation_missing" in manifest["blockers"]
    assert "cache_auth_audit_array_sha256_missing" in manifest["blockers"]


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


def test_mlx_production_contract_rejects_torch_parity_window_mismatch() -> None:
    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(batch_pairs=4),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(batch_pairs=4, covered_pair_window=[10, 14]),
        profile_stability=_profile_stability(batch_pairs=4),
        batch_invariance=_batch_invariance(batch_pairs=4),
    )

    assert manifest["passed"] is False
    assert (
        "torch_parity_window_does_not_cover_response:response=[0, 4]:covered=[10, 14]"
        in manifest["blockers"]
    )


def test_mlx_production_contract_rejects_torch_parity_batch_shape_mismatch() -> None:
    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(batch_pairs=4),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(batch_pairs=1, covered_pair_window=[0, 4]),
        profile_stability=_profile_stability(batch_pairs=4),
        batch_invariance=_batch_invariance(batch_pairs=4),
    )

    assert manifest["passed"] is False
    assert (
        "torch_parity_window_pairs_mismatch:"
        "response_batch_pairs=4:parity_window_pairs=1"
    ) in manifest["blockers"]


def test_mlx_production_contract_rejects_torch_parity_threshold_too_loose() -> None:
    parity = _torch_parity()
    parity["thresholds"]["max_segnet_argmax_diff_pixels"] = 2

    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=parity,
        reference_torch_parity=_torch_parity(side="reference"),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
    )

    assert manifest["passed"] is False
    assert (
        "torch_parity_threshold_max_segnet_argmax_diff_pixels_too_loose:2>1"
        in manifest["blockers"]
    )


def test_mlx_production_contract_rejects_torch_parity_sweep_gaps() -> None:
    response = _response_payload(batch_pairs=1)
    response["n_samples"] = 4
    response["pair_window"] = [0, 4]
    response["components"]["posenet_shape"] = [4]
    response["components"]["segnet_shape"] = [4]
    parity = _torch_parity(batch_pairs=1, covered_pair_window=[0, 4])
    parity["rows"] = [
        {**parity["rows"][0], "pair_window": [0, 1], "n_samples": 1},
        {**parity["rows"][0], "index": 1, "pair_window": [2, 3], "n_samples": 1},
        {**parity["rows"][0], "index": 2, "pair_window": [3, 4], "n_samples": 1},
    ]

    manifest = build_mlx_scorer_production_contract_manifest(
        response,
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=parity,
        profile_stability=_profile_stability(batch_pairs=4),
        batch_invariance=_batch_invariance(),
    )

    assert manifest["passed"] is False
    assert (
        "torch_parity_sweep_rows_not_contiguous:"
        "index=1:expected_start=1:window=[2, 3]"
    ) in manifest["blockers"]


def test_mlx_production_contract_can_require_score_calibration() -> None:
    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        reference_torch_parity=_torch_parity(side="reference"),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
        score_calibration=_score_calibration(),
        conv2d_accumulation_probe=_conv2d_accumulation_probe(),
        downstream_scorer_drift=_downstream_scorer_drift(),
        require_score_calibration=True,
        require_conv2d_accumulation_probe=True,
        require_downstream_scorer_drift=True,
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


def test_mlx_production_contract_rejects_incomplete_score_calibration() -> None:
    calibration = _score_calibration()
    calibration["summary"].pop("mlx_spend_triage_pairwise_certified_count")

    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
        score_calibration=calibration,
        require_score_calibration=True,
    )

    assert manifest["passed"] is False
    assert "score_calibration_certified_pairwise_count_missing" in manifest["blockers"]


def test_mlx_production_contract_rejects_score_calibration_for_different_response() -> None:
    calibration = _score_calibration()
    calibration["rows"][0]["mlx_components"]["posenet_sha256"] = "0" * 64

    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        reference_torch_parity=_torch_parity(side="reference"),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
        score_calibration=calibration,
        require_score_calibration=True,
    )

    assert manifest["passed"] is False
    assert "score_calibration_no_row_matches_response_identity" in manifest["blockers"]


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
        torch_parity=_torch_parity(batch_pairs=4),
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
        torch_parity=_torch_parity(batch_pairs=4),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(device="gpu", batch_pairs=4),
    )

    assert manifest["passed"] is False
    assert "batch_invariance_device_type_mismatch:response=cpu:gate=gpu" in manifest["blockers"]


def test_mlx_production_contract_rejects_profile_window_mismatch() -> None:
    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        profile_stability=_profile_stability(start_pair=5),
        batch_invariance=_batch_invariance(),
    )

    assert manifest["passed"] is False
    assert (
        "profile_stability_pair_window_mismatch:response=[0, 1]:profile=[5, 6]"
        in manifest["blockers"]
    )


def test_mlx_production_contract_rejects_vacuous_profile_stability() -> None:
    stability = _profile_stability()
    stability["profile_summary"]["row_count"] = 1

    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        profile_stability=stability,
        batch_invariance=_batch_invariance(),
    )

    assert manifest["passed"] is False
    assert "profile_stability_row_count_lt_2" in manifest["blockers"]


def test_mlx_production_contract_rejects_profile_recommended_shape_mismatch() -> None:
    stability = _profile_stability()
    stability["selection"]["recommended_row"]["batch_pairs"] = 2

    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        profile_stability=stability,
        batch_invariance=_batch_invariance(),
    )

    assert manifest["passed"] is False
    assert (
        "profile_stability_recommended_batch_pairs_mismatch:"
        "response=1:profile=2"
    ) in manifest["blockers"]


def test_mlx_production_contract_rejects_batch_invariance_cache_mismatch() -> None:
    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(batch_pairs=4),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(batch_pairs=4),
        profile_stability=_profile_stability(batch_pairs=4),
        batch_invariance=_batch_invariance(batch_pairs=4, cache_dir="other/cache"),
    )

    assert manifest["passed"] is False
    assert "batch_invariance_candidate_cache_dir_mismatch" in manifest["blockers"]


def test_mlx_production_contract_checks_optional_batch_invariance_authority() -> None:
    batch_invariance = _batch_invariance()
    batch_invariance["score_claim"] = True

    manifest = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        profile_stability=_profile_stability(),
        batch_invariance=batch_invariance,
    )

    assert manifest["passed"] is False
    assert "batch_invariance_score_claim_not_false" in manifest["blockers"]
    assert manifest["required_gates"]["batch_invariance"] is False


def test_mlx_production_contract_cli_writes_manifest(tmp_path: Path) -> None:
    response_path = tmp_path / "response.json"
    cache_audit_path = tmp_path / "cache_audit.json"
    torch_parity_path = tmp_path / "torch_parity.json"
    reference_torch_parity_path = tmp_path / "reference_torch_parity.json"
    stability_path = tmp_path / "stability.json"
    invariance_path = tmp_path / "invariance.json"
    conv2d_probe_path = tmp_path / "conv2d_probe.json"
    downstream_drift_path = tmp_path / "downstream_drift.json"
    out_path = tmp_path / "contract.json"
    response_path.write_text(json.dumps(_response_payload()), encoding="utf-8")
    cache_audit_path.write_text(json.dumps(_cache_auth_audit()), encoding="utf-8")
    torch_parity_path.write_text(json.dumps(_torch_parity()), encoding="utf-8")
    reference_torch_parity_path.write_text(
        json.dumps(_torch_parity(side="reference")),
        encoding="utf-8",
    )
    stability_path.write_text(json.dumps(_profile_stability()), encoding="utf-8")
    invariance_path.write_text(json.dumps(_batch_invariance()), encoding="utf-8")
    conv2d_probe_path.write_text(
        json.dumps(_conv2d_accumulation_probe()),
        encoding="utf-8",
    )
    downstream_drift_path.write_text(
        json.dumps(_downstream_scorer_drift()),
        encoding="utf-8",
    )

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
            "--reference-torch-parity",
            str(reference_torch_parity_path),
            "--profile-stability",
            str(stability_path),
            "--batch-invariance",
            str(invariance_path),
            "--conv2d-accumulation-probe",
            str(conv2d_probe_path),
            "--downstream-scorer-drift",
            str(downstream_drift_path),
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
    assert (
        manifest["numerical_mitigation_summary"]["conv2d_accumulation_probe"]["passed"]
        is True
    )
    assert (
        manifest["numerical_mitigation_summary"]["downstream_scorer_drift"][
            "aggregate_verdict"
        ]
        == "BELOW_SCORER_PRECISION"
    )


def test_mlx_production_contract_cli_requires_conv2d_accumulation_probe_when_requested(
    tmp_path: Path,
) -> None:
    response_path = tmp_path / "response.json"
    cache_audit_path = tmp_path / "cache_audit.json"
    torch_parity_path = tmp_path / "torch_parity.json"
    reference_torch_parity_path = tmp_path / "reference_torch_parity.json"
    stability_path = tmp_path / "stability.json"
    invariance_path = tmp_path / "invariance.json"
    out_path = tmp_path / "contract.json"
    response_path.write_text(json.dumps(_response_payload()), encoding="utf-8")
    cache_audit_path.write_text(json.dumps(_cache_auth_audit()), encoding="utf-8")
    torch_parity_path.write_text(json.dumps(_torch_parity()), encoding="utf-8")
    reference_torch_parity_path.write_text(
        json.dumps(_torch_parity(side="reference")),
        encoding="utf-8",
    )
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
            "--reference-torch-parity",
            str(reference_torch_parity_path),
            "--profile-stability",
            str(stability_path),
            "--batch-invariance",
            str(invariance_path),
            "--require-conv2d-accumulation-probe",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    manifest = json.loads(out_path.read_text(encoding="utf-8"))
    assert "conv2d_accumulation_probe_manifest_not_supplied" in manifest["blockers"]
    assert manifest["required_gates"]["conv2d_accumulation_probe"] is True


def test_mlx_production_contract_cli_requires_downstream_scorer_drift_when_requested(
    tmp_path: Path,
) -> None:
    response_path = tmp_path / "response.json"
    cache_audit_path = tmp_path / "cache_audit.json"
    torch_parity_path = tmp_path / "torch_parity.json"
    reference_torch_parity_path = tmp_path / "reference_torch_parity.json"
    stability_path = tmp_path / "stability.json"
    invariance_path = tmp_path / "invariance.json"
    out_path = tmp_path / "contract.json"
    response_path.write_text(json.dumps(_response_payload()), encoding="utf-8")
    cache_audit_path.write_text(json.dumps(_cache_auth_audit()), encoding="utf-8")
    torch_parity_path.write_text(json.dumps(_torch_parity()), encoding="utf-8")
    reference_torch_parity_path.write_text(
        json.dumps(_torch_parity(side="reference")),
        encoding="utf-8",
    )
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
            "--reference-torch-parity",
            str(reference_torch_parity_path),
            "--profile-stability",
            str(stability_path),
            "--batch-invariance",
            str(invariance_path),
            "--require-downstream-scorer-drift",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    manifest = json.loads(out_path.read_text(encoding="utf-8"))
    assert "downstream_scorer_drift_manifest_not_supplied" in manifest["blockers"]
    assert manifest["required_gates"]["downstream_scorer_drift"] is True


def test_mlx_production_contract_bundle_cli_writes_manifest(tmp_path: Path) -> None:
    contract = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        reference_torch_parity=_torch_parity(side="reference"),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
        score_calibration=_score_calibration(),
        conv2d_accumulation_probe=_conv2d_accumulation_probe(),
        downstream_scorer_drift=_downstream_scorer_drift(),
        require_score_calibration=True,
        require_conv2d_accumulation_probe=True,
        require_downstream_scorer_drift=True,
    )
    contract_path = tmp_path / "contract.json"
    out_path = tmp_path / "bundle.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_mlx_production_contract_bundle.py"),
            "--contract",
            str(contract_path),
            "--output",
            str(out_path),
            "--run-id",
            "unit-bundle",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert '"passed": true' in completed.stdout
    bundle = json.loads(out_path.read_text(encoding="utf-8"))
    assert bundle["run_id"] == "unit-bundle"
    assert bundle["passed"] is True
    assert bundle["verdict"] == BUNDLE_PASS_VERDICT
    assert bundle["summary"]["contract_count"] == 1
    assert bundle["summary"]["strict_contract_count"] == 1


def test_mlx_production_contract_bundle_cli_validates_dataset_coverage(
    tmp_path: Path,
) -> None:
    contract = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        reference_torch_parity=_torch_parity(side="reference"),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
        score_calibration=_score_calibration(),
        conv2d_accumulation_probe=_conv2d_accumulation_probe(),
        downstream_scorer_drift=_downstream_scorer_drift(),
        require_score_calibration=True,
        require_conv2d_accumulation_probe=True,
        require_downstream_scorer_drift=True,
    )
    contract_path = tmp_path / "contract.json"
    dataset_path = tmp_path / "dataset.json"
    out_path = tmp_path / "bundle.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    dataset_path.write_text(
        json.dumps(
            {
                "schema": "scorer_response_dataset.v1",
                "rows": [_mlx_dataset_row_from_contract(contract)],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_mlx_production_contract_bundle.py"),
            "--contract",
            str(contract_path),
            "--dataset",
            str(dataset_path),
            "--output",
            str(out_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert '"passed": true' in completed.stdout
    bundle = json.loads(out_path.read_text(encoding="utf-8"))
    assert bundle["passed"] is True
    assert bundle["dataset_coverage_gate"]["status"] == "strict_pass"
    assert bundle["summary"]["dataset_mlx_row_count"] == 1
    assert bundle["summary"]["dataset_matched_row_count"] == 1


def test_mlx_production_contract_bundle_cli_blocks_uncovered_dataset_row(
    tmp_path: Path,
) -> None:
    contract = build_mlx_scorer_production_contract_manifest(
        _response_payload(),
        cache_auth_audit=_cache_auth_audit(),
        torch_parity=_torch_parity(),
        reference_torch_parity=_torch_parity(side="reference"),
        profile_stability=_profile_stability(),
        batch_invariance=_batch_invariance(),
        score_calibration=_score_calibration(),
        conv2d_accumulation_probe=_conv2d_accumulation_probe(),
        require_score_calibration=True,
        require_conv2d_accumulation_probe=True,
    )
    row = _mlx_dataset_row_from_contract(contract)
    row["source_pair_window"] = [1, 2]
    contract_path = tmp_path / "contract.json"
    dataset_path = tmp_path / "dataset.json"
    out_path = tmp_path / "bundle.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    dataset_path.write_text(
        json.dumps({"schema": "scorer_response_dataset.v1", "rows": [row]}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_mlx_production_contract_bundle.py"),
            "--contract",
            str(contract_path),
            "--dataset",
            str(dataset_path),
            "--output",
            str(out_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert '"passed": false' in completed.stdout
    bundle = json.loads(out_path.read_text(encoding="utf-8"))
    assert bundle["passed"] is False
    assert "dataset_mlx_row_coverage_gate_not_strict_pass" in bundle["blockers"]
    assert bundle["dataset_coverage_gate"]["status"] == "blocked"
    assert "mlx_production_contract_bundle_row_unmatched:mlx-row-1" in bundle[
        "blockers"
    ]


def _mlx_dataset_row_from_contract(contract: dict) -> dict:
    summary = contract["response_summary"]
    return {
        "schema": "scorer_response_row.v1",
        "row_id": "mlx-row-1",
        "family": "mlx_scorer_response",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "archive_sha256": summary["archive_sha256"],
        "source_inflated_outputs_aggregate_sha256": summary[
            "inflated_outputs_aggregate_sha256"
        ],
        "source_batch_pairs": summary["batch_pairs"],
        "source_n_samples": summary["n_samples"],
        "source_pair_window": summary["pair_window"],
        "source_candidate_cache_array_sha256": summary[
            "candidate_cache_array_sha256"
        ],
        "source_reference_cache_array_sha256": summary[
            "reference_cache_array_sha256"
        ],
        "source_posenet_sha256": summary["posenet_sha256"],
        "source_segnet_sha256": summary["segnet_sha256"],
    }


def _response_payload(*, device: str = "cpu", batch_pairs: int = 1) -> dict:
    n_samples = int(batch_pairs)
    score = _fixture_score()
    return {
        "schema_version": "mlx_scorer_response.v1",
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_axis": EVIDENCE_TAG_MLX,
        "response_family": "pr101_pose_axis",
        "hardware_substrate": f"MLX {device}",
        "gpu_research_signal_allowed": device == "gpu",
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "canonical_score": score,
        "score_recomputed_from_components": score,
        "canonical_score_source": "score_recomputed_from_components",
        "avg_posenet_dist": 0.0,
        "avg_segnet_dist": 0.0,
        "archive_size_bytes": 123,
        "batch_pairs": batch_pairs,
        "n_samples": n_samples,
        "pair_window": [0, n_samples],
        "components": {
            "posenet_shape": [n_samples],
            "segnet_shape": [n_samples],
            "posenet_sha256": "a" * 64,
            "segnet_sha256": "b" * 64,
        },
        "cache_identity": {
            "pair_indices_equal": True,
            "reference": {
                "archive_sha256": "c" * 64,
                "inflated_outputs_aggregate_sha256": "d" * 64,
                "raw_sha256": "e" * 64,
                "path": "reference/cache",
                "pair_count": 10,
                "hash_domain": "_array_sha256(dtype_string + json_shape + contiguous_bytes)",
                "array_sha256": {
                    "pair_indices": "6" * 64,
                    "posenet_yuv6_pair": "7" * 64,
                    "segnet_last_rgb": "8" * 64,
                },
                "pair_indices_shape": [10, 2],
                "posenet_yuv6_pair_shape": [10, 12, 192, 256],
                "segnet_last_rgb_shape": [10, 3, 384, 512],
            },
            "candidate": {
                "archive_sha256": "f" * 64,
                "inflated_outputs_aggregate_sha256": "1" * 64,
                "raw_sha256": "2" * 64,
                "path": "candidate/cache",
                "pair_count": 10,
                "hash_domain": "_array_sha256(dtype_string + json_shape + contiguous_bytes)",
                "array_sha256": {
                    "pair_indices": "9" * 64,
                    "posenet_yuv6_pair": "a" * 64,
                    "segnet_last_rgb": "b" * 64,
                },
                "pair_indices_shape": [10, 2],
                "posenet_yuv6_pair_shape": [10, 12, 192, 256],
                "segnet_last_rgb_shape": [10, 3, 384, 512],
                "auth_eval_identity_audit": {
                    "schema_version": "mlx_scorer_input_cache_auth_eval_audit.v1",
                    "verdict": "PASS_CACHE_AUTH_EVAL_IDENTITY",
                    "passed": True,
                    "identity_residual": 0,
                    "score_claim": False,
                    "score_claim_valid": False,
                    "promotion_eligible": False,
                    "promotable": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
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


def _profile_stability(
    *,
    passed: bool = True,
    start_pair: int = 0,
    batch_pairs: int = 1,
    candidate_cache_dir: str = "candidate/cache",
    reference_cache_dir: str = "reference/cache",
) -> dict:
    score = _fixture_score()
    return {
        "schema_version": "mlx_scorer_response_profile_stability.v1",
        "passed": passed,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "candidate_generation_only": True,
        "profile_summary": {
            "reference_cache_dir": reference_cache_dir,
            "candidate_cache_dir": candidate_cache_dir,
            "archive_size_bytes": 123,
            "start_pair": start_pair,
            "max_pairs": int(batch_pairs),
            "row_count": 2,
        },
        "selection": {
            "recommended_row": {
                "index": 0,
                "device": "cpu",
                "batch_pairs": int(batch_pairs),
                "n_samples": int(batch_pairs),
                "pair_window": [start_pair, start_pair + int(batch_pairs)],
                "canonical_score": score,
                "avg_posenet_dist": 0.0,
                "avg_segnet_dist": 0.0,
                "posenet_sha256": "a" * 64,
                "segnet_sha256": "b" * 64,
                "pairs_per_second": 1.0,
            },
        },
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
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "cache": {
            "archive_sha256": archive_sha256,
            "inflated_outputs_aggregate_sha256": "1" * 64,
            "raw_sha256": "2" * 64,
            "pair_count": 10,
            "hash_domain": "_array_sha256(dtype_string + json_shape + contiguous_bytes)",
            "array_sha256": {
                "pair_indices": "9" * 64,
                "posenet_yuv6_pair": "a" * 64,
                "segnet_last_rgb": "b" * 64,
            },
            "pair_indices_shape": [10, 2],
            "posenet_yuv6_pair_shape": [10, 12, 192, 256],
            "segnet_last_rgb_shape": [10, 3, 384, 512],
        },
        "auth_eval": {
            "n_samples": 10,
            "scorer_input_hash_domain": "_array_sha256(dtype_string + json_shape + contiguous_bytes)",
            "scorer_input_array_sha256": {
                "pair_indices": "9" * 64,
                "posenet_yuv6_pair": "a" * 64,
                "segnet_last_rgb": "b" * 64,
            },
            "scorer_input_shapes": {
                "pair_indices": [10, 2],
                "posenet_yuv6_pair": [10, 12, 192, 256],
                "segnet_last_rgb": [10, 3, 384, 512],
            },
        },
        "canonical_equation": {
            "eligible_for_local_mlx_transfer_calibration": passed,
            "identity_residual": 0 if passed else 1,
            "hash_domain": "_array_sha256(dtype_string + json_shape + contiguous_bytes)",
            "compared_scorer_input_hashes": {
                "pair_indices": True,
                "posenet_yuv6_pair": True,
                "segnet_last_rgb": True,
            },
            "compared_scorer_input_shapes": {
                "pair_indices": True,
                "posenet_yuv6_pair": True,
                "segnet_last_rgb": True,
            },
            "score_claim": False,
        },
        "allowed_use": (
            ["local_mlx_training_transfer_calibration"] if passed else ["debug_only"]
        ),
    }


def _conv2d_accumulation_probe(*, passed: bool = True) -> dict:
    return {
        "schema_version": "mlx_conv2d_accumulation_probe.v1",
        "passed": passed,
        "verdict": (
            "PASS_MLX_CONV2D_ACCUMULATION_PROBE"
            if passed
            else "FAIL_MLX_CONV2D_ACCUMULATION_PROBE"
        ),
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_axis": EVIDENCE_TAG_MLX,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "device_type": "cpu",
        "torch_device_type": "cpu",
        "mlx_backend": {
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "classification": "framework_different_no_public_deterministic_reduction_flag",
        },
        "rows": [
            _conv2d_accumulation_row("optimized_mlx_conv2d", passed=True),
            _conv2d_accumulation_row("fixed_fp32", passed=True),
            _conv2d_accumulation_row("kahan_fp32", passed=True),
            _conv2d_accumulation_row("fixed_fp64", passed=passed),
        ],
    }


def _conv2d_accumulation_row(mode: str, *, passed: bool) -> dict:
    return {
        "mode": mode,
        "passed": passed,
        "shape_match": True,
        "max_abs_delta": 0.0 if passed else 1.0,
        "blockers": [] if passed else ["max_abs_delta_exceeds_threshold:1.0>0.0"],
    }


def _downstream_scorer_drift(
    *,
    aggregate_verdict: str = "BELOW_SCORER_PRECISION",
    drift_units: float = 7.4e-5,
    scorer_input_mode: str = "contest_uint8",
    archive_sha256: str = "f" * 64,
    posenet_sha256: str = "a" * 64,
    segnet_sha256: str = "b" * 64,
    checkpoint_mode: str = "trained_archive",
    n_pairs_actual: int = 100,
    covered_pair_window: list[int] | None = None,
) -> dict:
    window = covered_pair_window or [0, int(n_pairs_actual)]
    return {
        "schema": "pr95_mlx_pytorch_full_decoder_downstream_scorer_drift_v1",
        "schema_version": "pr95_mlx_pytorch_full_decoder_downstream_scorer_drift_v1",
        "aggregate_verdict": aggregate_verdict,
        "aggregate_contest_score_drift_units": drift_units,
        "scorer_input_mode": scorer_input_mode,
        "checkpoint_mode": checkpoint_mode,
        "n_pairs_actual": n_pairs_actual,
        "covered_pair_window": window,
        "covered_pair_indices_sha256": "4" * 64,
        "archive_zip_sha256": archive_sha256,
        "posenet_sha256": posenet_sha256,
        "segnet_sha256": segnet_sha256,
        "scorer_components": {
            "posenet_sha256": posenet_sha256,
            "segnet_sha256": segnet_sha256,
        },
        "stages": [
            {
                "stage_name": "stage_5_contest_score_aggregation",
                "metric": "aggregate_contest_score_drift_due_to_mlx_vs_pytorch_framework",
                "max_abs": drift_units,
                "mean_abs": drift_units,
                "rms": drift_units,
                "extra": {"verdict_per_stage": aggregate_verdict},
            }
        ],
        "operator_routable_per_verdict": "unit-test-only",
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "axis_tag": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "blockers": [
            "macos_mlx_research_signal_not_contest_authority",
            "requires_paired_contest_cpu_plus_cuda_for_score_claim",
            "drift_measurement_is_implementation_parity_not_scorer_response",
        ],
    }


def _torch_parity(
    *,
    passed: bool = True,
    archive_sha256: str = "f" * 64,
    covered_pair_window: list[int] | None = None,
    batch_pairs: int = 1,
    side: str = "candidate",
) -> dict:
    window = covered_pair_window or [0, int(batch_pairs)]
    cache_path = "reference/cache" if side == "reference" else "candidate/cache"
    cache_archive_sha256 = "c" * 64 if side == "reference" else archive_sha256
    cache_inflated_sha256 = "d" * 64 if side == "reference" else "1" * 64
    cache_raw_sha256 = "e" * 64 if side == "reference" else "2" * 64
    array_sha256 = (
        {
            "pair_indices": "6" * 64,
            "posenet_yuv6_pair": "7" * 64,
            "segnet_last_rgb": "8" * 64,
        }
        if side == "reference"
        else {
            "pair_indices": "9" * 64,
            "posenet_yuv6_pair": "a" * 64,
            "segnet_last_rgb": "b" * 64,
        }
    )
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
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "score_axis": EVIDENCE_TAG_MLX,
        "candidate_generation_only": True,
        "requires_exact_eval_before_promotion": True,
        "cache_identity": {
            "archive_sha256": cache_archive_sha256,
            "inflated_outputs_aggregate_sha256": cache_inflated_sha256,
            "raw_sha256": cache_raw_sha256,
            "path": cache_path,
            "pair_count": 10,
            "hash_domain": "_array_sha256(dtype_string + json_shape + contiguous_bytes)",
            "array_sha256": array_sha256,
            "pair_indices_shape": [10, 2],
            "posenet_yuv6_pair_shape": [10, 12, 192, 256],
            "segnet_last_rgb_shape": [10, 3, 384, 512],
        },
        "cache_dir": "candidate/cache",
        "total_pair_count": 10,
        "covered_pair_window": window,
        "thresholds": {
            "max_posenet_output_abs_delta": 2.0e-3,
            "max_segnet_logit_abs_delta": 1.0e-2,
            "max_posenet_component_abs_delta": 2.0e-5,
            "max_segnet_argmax_diff_pixels": 0,
        },
        "window_pairs": int(batch_pairs),
        "stride_pairs": int(batch_pairs),
        "window_count": 1,
        "summary": {"failed_windows": 0 if passed else 1},
        "rows": [
            {
                "index": 0,
                "passed": passed,
                "verdict": (
                    "PASS_MLX_TORCH_SCORER_PARITY"
                    if passed
                    else "FAIL_MLX_TORCH_SCORER_PARITY"
                ),
                "blockers": [] if passed else ["synthetic_failure"],
                "pair_window": window,
                "n_samples": window[1] - window[0],
                "deltas": {},
            }
        ],
    }


def _score_calibration(*, uncertain_count: int = 0) -> dict:
    score = _fixture_score()
    return {
        "schema_version": "mlx_score_calibration.v1",
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "decision_policy": {
            "allowed_use": "local_spend_triage_only_after_strict_auth_axis_calibration",
            "recommended_min_mlx_gap_for_spend_triage": 1.0e-4,
            "calibration_uncertainty_score": 2.0e-5,
        },
        "summary": {
            "mlx_spend_triage_pairwise_uncertain_count": uncertain_count,
            "mlx_spend_triage_pairwise_certified_count": 3,
            "mlx_spend_triage_pairwise_total_count": 3,
            "recommended_min_mlx_gap_for_spend_triage": 1.0e-4,
            "calibration_uncertainty_score": 2.0e-5,
        },
        "rows": [
            {
                "archive_sha256": "f" * 64,
                "inflated_outputs_aggregate_sha256": "1" * 64,
                "archive_size_bytes": 123,
                "response_family": "pr101_pose_axis",
                "pair_window": [0, 1],
                "n_samples": 1,
                "batch_pairs": 1,
                "mlx_score": score,
                "mlx_avg_posenet_dist": 0.0,
                "mlx_avg_segnet_dist": 0.0,
                "mlx_components": {
                    "posenet_sha256": "a" * 64,
                    "segnet_sha256": "b" * 64,
                    "posenet_shape": [1],
                    "segnet_shape": [1],
                },
                "candidate_cache_identity": {
                    "hash_domain": "_array_sha256(dtype_string + json_shape + contiguous_bytes)",
                    "array_sha256": {
                        "pair_indices": "9" * 64,
                        "posenet_yuv6_pair": "a" * 64,
                        "segnet_last_rgb": "b" * 64,
                    },
                    "pair_indices_shape": [10, 2],
                    "posenet_yuv6_pair_shape": [10, 12, 192, 256],
                    "segnet_last_rgb_shape": [10, 3, 384, 512],
                },
                "reference_cache_identity": {
                    "hash_domain": "_array_sha256(dtype_string + json_shape + contiguous_bytes)",
                    "array_sha256": {
                        "pair_indices": "6" * 64,
                        "posenet_yuv6_pair": "7" * 64,
                        "segnet_last_rgb": "8" * 64,
                    },
                    "pair_indices_shape": [10, 2],
                    "posenet_yuv6_pair_shape": [10, 12, 192, 256],
                    "segnet_last_rgb_shape": [10, 3, 384, 512],
                },
            }
        ],
    }


def _fixture_score() -> float:
    return contest_formula_score(seg_dist=0.0, pose_dist=0.0, archive_bytes=123)


def _batch_invariance(
    *,
    passed: bool = True,
    device: str = "cpu",
    batch_pairs: int = 1,
    cache_dir: str = "candidate/cache",
) -> dict:
    return {
        "schema_version": "mlx_scorer_batch_invariance.v1",
        "passed": passed,
        "device_type": device,
        "batch_pairs": batch_pairs,
        "cache_dir": cache_dir,
        "start_pair": 0,
        "total_pair_count": 10,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "candidate_generation_only": True,
    }
