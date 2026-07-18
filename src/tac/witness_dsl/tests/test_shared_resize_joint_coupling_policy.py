# SPDX-License-Identifier: MIT
from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.measure_shared_resize_seg_pose_coupling_20260718 import (
    aggregate_response_metrics,
)
from tac.witness_dsl.shared_resize_joint_coupling_policy import (
    INVERSE_SOLVE_FACTOR_IDS,
    INVERSE_SOLVE_FACTOR_LEAF_IDS,
    compile_inverse_solve_completeness_manifest,
    compile_shared_resize_joint_coupling_policy,
    deterministic_stride_sample_ids,
    validate_measurement_receipt,
)


def _receipt(*, n: int = 8) -> dict[str, object]:
    counts = {"nonzero_requested": 2, "realized_changed": 1, "boundary_clipped": 1}
    baseline = {"d_seg": 0.2, "d_pose": 0.4}
    arm_scores = {
        "seg_plus": {"d_seg": 0.19, "d_pose": 0.41},
        "seg_minus": {"d_seg": 0.21, "d_pose": 0.39},
        "pose_plus": {"d_seg": 0.18, "d_pose": 0.38},
        "pose_minus": {"d_seg": 0.22, "d_pose": 0.42},
        "joint_plus": {"d_seg": 0.19, "d_pose": 0.39},
    }
    derived = aggregate_response_metrics(baseline, **arm_scores)
    return {
        "schema": "shared_resize_joint_coupling_measurement.v2",
        "captured_at_utc": "2026-07-18T12:00:00+00:00",
        "axis": "[macOS-CPU advisory]",
        "evidence_status": (
            "LIVENESS_ONLY_NOT_A_MEASUREMENT_VERDICT" if n == 1 else "MEASURED_ADVISORY_SUBSET"
        ),
        "n_pairs_total": 600,
        "sample": {
            "method": "deterministic_cyclic_stride",
            "n_of_600": n,
            "pair_ids": list(deterministic_stride_sample_ids(600, n, 538)),
            "seed": 538,
        },
        "input_custody": {
            "checkpoint_path": "/evidence/checkpoint.npz",
            "checkpoint_sha256": "a" * 64,
            "gt_cache_path": "/evidence/gt.npz",
            "gt_cache_sha256": "b" * 64,
            "segnet_path": "/evidence/segnet.safetensors",
            "segnet_sha256": "c" * 64,
            "posenet_path": "/evidence/posenet.safetensors",
            "posenet_sha256": "d" * 64,
            "checkpoint_payload": {
                "carrier_absent": True,
                "base_inr_only": True,
                "detected_carrier_keys": [],
                "checkpoint_key_count": 1,
                "checkpoint_key_manifest_sha256": "f" * 64,
            },
        },
        "execution_custody": {
            "git_head": "e" * 40,
            "argv": ["measure.py", "--n-sample", str(n)],
            "config": {"n_sample": n, "seed": 538},
            "input_bytes": {
                "checkpoint": 1,
                "gt_cache": 2,
                "segnet": 3,
                "posenet": 4,
                "modules_py": 5,
                "frame_utils_py": 6,
                "evaluate_py": 7,
            },
            "upstream_source_sha256": {
                "modules_py": "1" * 64,
                "frame_utils_py": "2" * 64,
                "evaluate_py": "3" * 64,
            },
            "upstream_source_paths": {
                "modules_py": "/evidence/modules.py",
                "frame_utils_py": "/evidence/frame_utils.py",
                "evaluate_py": "/evidence/evaluate.py",
            },
        },
        "evidence_labels": {
            "smooth_gram": "B1_LOCAL_DERIVED",
            "finite_response": "B32_DUPLICATE_LAST_SUBSET_ADVISORY",
            "contest_score": "NOT_MEASURED",
        },
        "gt_target_custody": {
            "targets_used": "rederived_from_gt_frames",
            "scorer_batch_size": 32,
            "last_batch_padding": "duplicate-last then discard padded outputs",
            "cache_vs_rederived": {
                "seg_mismatched_pixels": 0,
                "seg_total_pixels": n * 384 * 512,
                "seg_mismatch_fraction": 0.0,
                "pose_mismatched_elements": 0,
                "pose_total_elements": n * 6,
                "pose_max_abs": 0.0,
                "pose_mse": 0.0,
            },
        },
        "shared_A": {
            "seg_pose_operator_identical": True,
            "camera_hw": [874, 1164],
            "scorer_hw": [384, 512],
        },
        "smooth_coupling": {
            "evidence_label": "B1_LOCAL_DERIVED",
            "primary_surface": "shared_frame1",
            "shared_frame1": {
                "raw_gram_2x2": [[2.0, 0.5], [0.5, 1.0]],
                "score_priced_gram_2x2": [[20000.0, 50.0], [50.0, 5.0]],
            },
            "full_pair_context": {
                "raw_gram_2x2": [[2.0, 0.5], [0.5, 2.0]],
                "score_priced_gram_2x2": [[20000.0, 50.0], [50.0, 10.0]],
            },
        },
        "actual_response": {
            "evidence_label": "B32_DUPLICATE_LAST_SUBSET_ADVISORY",
            "native_or_full_n600_comparable": False,
            "perturbation_surface": "shared_frame1_only",
            "scorer_batch_size": 32,
            "last_batch_padding": "duplicate-last then discard padded outputs",
            "by_support_fraction": [
                {
                    "support_fraction": 0.001,
                    "baseline": baseline,
                    "arm_scores": arm_scores,
                    **derived,
                    "realized_lsb_counts": {
                        arm: dict(counts)
                        for arm in (
                            "seg_plus",
                            "seg_minus",
                            "pose_plus",
                            "pose_minus",
                            "joint_plus",
                        )
                    },
                }
            ],
        },
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "paid_dispatch": False,
        "trainer_activation": False,
        "sacred_c2_mutated": False,
        "research_only": True,
        "verdict_scope": "instance and subset only",
    }


def test_policy_is_argv_inert_and_preserves_exact_factor_manifest() -> None:
    policy = compile_shared_resize_joint_coupling_policy()
    assert policy["live_trainer_argv"] == ()
    assert len(INVERSE_SOLVE_FACTOR_IDS) == 10
    assert len(INVERSE_SOLVE_FACTOR_LEAF_IDS) == 11
    assert INVERSE_SOLVE_FACTOR_LEAF_IDS[2:4] == (
        "3a_camera_resize_A",
        "3b_shared_A_coupling",
    )
    assert INVERSE_SOLVE_FACTOR_LEAF_IDS[8:10] == (
        "8_frame0_seg_freedom",
        "9_kerA_MDL",
    )


def test_completeness_manifest_never_presence_greens() -> None:
    rows = [
        {"factor_id": factor, "term": factor, "owning_task": "#538", "state": "have"}
        for factor in INVERSE_SOLVE_FACTOR_LEAF_IDS
    ]
    manifest = compile_inverse_solve_completeness_manifest(rows)
    assert manifest["disposition_rows_complete"] is True
    assert manifest["complete_by_construction"] is False
    assert manifest["live_v10_integration"] is False


def test_completeness_manifest_rejects_leaf_merging_or_reordering() -> None:
    rows = [
        {"factor_id": factor, "term": factor, "owning_task": "#538", "state": "have"}
        for factor in reversed(INVERSE_SOLVE_FACTOR_LEAF_IDS)
    ]
    with pytest.raises(ValueError, match="leaves/order"):
        compile_inverse_solve_completeness_manifest(rows)


def test_receipt_accepts_advisory_subset_and_liveness_only() -> None:
    assert validate_measurement_receipt(_receipt(n=8))["evidence_status"] == "MEASURED_ADVISORY_SUBSET"
    assert "LIVENESS_ONLY" in validate_measurement_receipt(_receipt(n=1))["evidence_status"]


@pytest.mark.parametrize("n", range(2, 8))
def test_receipt_rejects_unclassified_sample_sizes(n: int) -> None:
    with pytest.raises(ValueError, match="neither allowed liveness"):
        validate_measurement_receipt(_receipt(n=n))


def test_receipt_rejects_authority_escalation_and_bad_batch_geometry() -> None:
    escalated = _receipt()
    escalated["score_claim"] = True
    with pytest.raises(ValueError, match="score_claim"):
        validate_measurement_receipt(escalated)
    bad_batch = _receipt()
    bad_batch["actual_response"]["scorer_batch_size"] = 8  # type: ignore[index]
    with pytest.raises(ValueError, match="batch size 32"):
        validate_measurement_receipt(bad_batch)


def test_receipt_rejects_non_psd_or_nonconserving_evidence() -> None:
    bad_gram = _receipt()
    bad_gram["smooth_coupling"]["shared_frame1"]["raw_gram_2x2"] = [  # type: ignore[index]
        [1.0, 2.0],
        [2.0, 1.0],
    ]
    with pytest.raises(ValueError, match="positive semidefinite"):
        validate_measurement_receipt(bad_gram)
    bad_counts = deepcopy(_receipt())
    row = bad_counts["actual_response"]["by_support_fraction"][0]  # type: ignore[index]
    row["realized_lsb_counts"]["seg_plus"]["realized_changed"] = 2
    with pytest.raises(ValueError, match="do not conserve"):
        validate_measurement_receipt(bad_counts)


def test_receipt_rejects_fabricated_sample_or_carrier_custody() -> None:
    fabricated = _receipt()
    fabricated["sample"]["pair_ids"][0] += 1  # type: ignore[index]
    with pytest.raises(ValueError, match="deterministic stride"):
        validate_measurement_receipt(fabricated)
    carrier = _receipt()
    carrier["input_custody"]["checkpoint_payload"]["carrier_absent"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="carrier absence"):
        validate_measurement_receipt(carrier)


def test_receipt_seals_primary_surface_and_evidence_labels() -> None:
    wrong_primary = _receipt()
    wrong_primary["smooth_coupling"]["primary_surface"] = "full_pair"  # type: ignore[index]
    with pytest.raises(ValueError, match="shared_frame1"):
        validate_measurement_receipt(wrong_primary)
    wrong_label = _receipt()
    wrong_label["actual_response"]["evidence_label"] = "MEASURED"  # type: ignore[index]
    with pytest.raises(ValueError, match=r"evidence labels|measured advisory"):
        validate_measurement_receipt(wrong_label)


def test_receipt_rejects_noop_response_labeled_help() -> None:
    receipt = _receipt()
    row = receipt["actual_response"]["by_support_fraction"][0]  # type: ignore[index]
    baseline = {"d_seg": 0.2, "d_pose": 0.4}
    no_op_arms = {name: dict(baseline) for name in row["arm_scores"]}
    no_op = aggregate_response_metrics(baseline, **no_op_arms)
    row.update({"baseline": baseline, "arm_scores": no_op_arms, **no_op})
    row["measured_direction_classification"]["seg_direction"]["quality"] = "MEASURED_HELP"
    with pytest.raises(ValueError, match="exact arm-score recomputation"):
        validate_measurement_receipt(receipt)


def test_receipt_requires_realized_change_for_measured_help() -> None:
    receipt = _receipt()
    row = receipt["actual_response"]["by_support_fraction"][0]  # type: ignore[index]
    row["realized_lsb_counts"]["seg_plus"] = {
        "nonzero_requested": 0,
        "realized_changed": 0,
        "boundary_clipped": 0,
    }
    with pytest.raises(ValueError, match="without any realized changed pixels"):
        validate_measurement_receipt(receipt)


def test_receipt_requires_finite_baseline_and_every_arm() -> None:
    missing = _receipt()
    row = missing["actual_response"]["by_support_fraction"][0]  # type: ignore[index]
    del row["arm_scores"]["joint_plus"]
    with pytest.raises(ValueError, match=r"arm_scores\.joint_plus"):
        validate_measurement_receipt(missing)
    nonfinite = _receipt()
    row = nonfinite["actual_response"]["by_support_fraction"][0]  # type: ignore[index]
    row["baseline"]["d_pose"] = float("nan")
    with pytest.raises(ValueError, match=r"baseline\.d_pose"):
        validate_measurement_receipt(nonfinite)
