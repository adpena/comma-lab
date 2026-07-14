from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_equations.throughput_authority_anchors_20260714 import (
    build_exact_int64_segnet_anchor,
    build_full_r_anchor,
    build_mixed_int64_segnet_anchor,
    build_qdq_anchor,
    build_weight_l1_class_pair_tie_snap_segnet_anchor,
    build_weight_l1_int64_segnet_anchor,
    build_weight_l1_tie_snap_segnet_anchor,
)


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _qdq() -> dict[str, object]:
    indices_hash = "a" * 64
    return {
        "schema": "fixedpoint_scorer_forward_n600.v2",
        "contract": {
            "native_integer_speed_claim": False,
            "activation_scale_mode": "fixed_calibration",
            "calibration_split": [0, 120],
            "heldout_split": [120, 600],
            "accumulation": "QDQ fp32",
        },
        "summary": {
            "status": "MEASURED",
            "full_real_n600": True,
            "minimum_argmax_exact_arm": "w22a22",
            "minimum_training_tolerance_arm": "w18a18",
            "rung2_verdict": "ARGMAX_FIXEDPOINT_FEASIBLE",
            "verdict_scope": "n600 instance",
            "cache_custody": {
                "status": "MEASURED",
                "pairs": 600,
                "unique_pair_indices": 600,
                "observed_pair_indices_sha256": indices_hash,
                "expected_pair_indices_sha256": indices_hash,
            },
            "arms": {
                "w22a22": {
                    "status": "MEASURED",
                    "argmax_exact_admitted": True,
                    "segnet": {
                        "full": {
                            "aggregate_flip_fraction": 0.0,
                            "worst_pair_flip_fraction": 0.0,
                            "uncertified_pixels": 0,
                            "argmax_corpus_sha256": "b" * 64,
                        }
                    },
                }
            },
        },
    }


def test_qdq_anchor_requires_exact_n600_custody(tmp_path: Path) -> None:
    path = tmp_path / "qdq.json"
    payload = _qdq()
    _write(path, payload)
    anchor = build_qdq_anchor(path, payload, repo=tmp_path)
    assert anchor.residual == 0.0
    assert anchor.empirical_output["minimum_argmax_exact_arm"] == "w22a22"
    assert anchor.inputs["activation_scale_mode"] == "fixed_calibration"
    payload["summary"]["cache_custody"]["pairs"] = 599  # type: ignore[index]
    with pytest.raises(ValueError, match=r"0\.\.599"):
        build_qdq_anchor(path, payload, repo=tmp_path)


def test_full_r_anchor_requires_complete_real_corpus(tmp_path: Path) -> None:
    path = tmp_path / "full_r.json"
    payload = {
        "schema": "pythagorean_exact_arithmetic_full_r_n600.v2",
        "contract": {"q_weight_bits": 15, "state_bits_by_boundary": [7, 5]},
        "summary": {
            "complete": True,
            "authority": {
                "coverage_exact": True,
                "frames": 1200,
                "within_derived_bound": True,
            },
            "fixed_q15_int32_atomic": {
                "cross_process_identical": True,
                "exact_numpy_int_corpus_parity": True,
            },
            "float_atomic": {"cross_process_identical": False},
            "overall_verdict": "REAL-L70-LEVER-FULL-R-N600",
            "verdict_scope": "n600 instance",
        },
    }
    _write(path, payload)
    anchor = build_full_r_anchor(path, payload, repo=tmp_path)
    assert anchor.residual == 0.0
    payload["summary"]["authority"]["frames"] = 1199  # type: ignore[index]
    with pytest.raises(ValueError, match="custody"):
        build_full_r_anchor(path, payload, repo=tmp_path)


def test_exact_int64_anchor_keeps_empirical_argmax_and_certificate_separate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "exact_int64.json"
    indices_hash = "d" * 64
    payload = {
        "schema": "exact_int64_fixedpoint_scorer_n600.v1",
        "contract": {
            "native_integer_speed_claim": True,
            "activation_scale_mode": "dynamic_exact_absmax",
        },
        "model_manifest": {
            "converted_conv2d_count": 125,
            "accumulation": "exact_signed_int64",
            "finalization": "single_fp32_scale_and_bias_per_output",
        },
        "summary": {
            "status": "MEASURED",
            "full_real_n600": True,
            "bits": 26,
            "argmax_exact_admitted": True,
            "training_tolerance_admitted": True,
            "rung2_integer_verdict": "EXACT_INT64_ARGMAX_FEASIBLE",
            "verdict_scope": "n600 instance",
            "cache_custody": {
                "status": "MEASURED",
                "pairs": 600,
                "unique_pair_indices": 600,
                "observed_pair_indices_sha256": indices_hash,
                "expected_pair_indices_sha256": indices_hash,
            },
            "candidate": {
                "full": {
                    "aggregate_flip_fraction": 0.0,
                    "uncertified_pixels": 2,
                }
            },
            "timing": {"candidate_speedup_vs_reference_x": 0.1},
        },
    }
    _write(path, payload)
    anchor = build_exact_int64_segnet_anchor(path, payload, repo=tmp_path)
    assert anchor.residual == 0.0
    assert anchor.empirical_output["argmax_exact_admitted"] is True
    assert anchor.empirical_output["candidate_full"]["uncertified_pixels"] == 2


def test_mixed_int64_anchor_requires_geometry_only_static_bound_manifest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mixed_int64.json"
    indices_hash = "e" * 64
    payload = {
        "schema": "mixed_int64_fixedpoint_scorer_n600.v1",
        "contract": {"native_integer_speed_claim": True},
        "model_manifest": {
            "minimum_bits": 26,
            "maximum_bits": 30,
            "converted_conv2d_count": 125,
            "accumulation": "exact_signed_int64",
            "finalization": "single_fp32_scale_and_bias_per_output",
            "assignment_rule": "largest_geometry_safe_bits_with_signed_int64_static_bound",
            "precision_histogram": {"26": 5, "27": 30, "28": 22, "29": 19, "30": 49},
        },
        "summary": {
            "status": "MEASURED",
            "full_real_n600": True,
            "argmax_exact_admitted": True,
            "training_tolerance_admitted": True,
            "rung2_mixed_integer_verdict": "MIXED_INT64_ARGMAX_FEASIBLE",
            "verdict_scope": "n600 instance",
            "cache_custody": {
                "status": "MEASURED",
                "pairs": 600,
                "unique_pair_indices": 600,
                "observed_pair_indices_sha256": indices_hash,
                "expected_pair_indices_sha256": indices_hash,
            },
            "candidate": {"full": {"aggregate_flip_fraction": 0.0}},
            "timing": {"candidate_speedup_vs_reference_x": 0.1},
        },
    }
    _write(path, payload)
    anchor = build_mixed_int64_segnet_anchor(path, payload, repo=tmp_path)
    assert anchor.residual == 0.0
    assert anchor.inputs["precision_histogram"]["30"] == 49
    payload["model_manifest"]["assignment_rule"] = "label_tuned"  # type: ignore[index]
    with pytest.raises(ValueError, match="integer custody"):
        build_mixed_int64_segnet_anchor(path, payload, repo=tmp_path)


def test_weight_l1_anchor_requires_label_free_bound(tmp_path: Path) -> None:
    path = tmp_path / "weight_l1.json"
    indices_hash = "f" * 64
    payload = {
        "schema": "weight_l1_int64_fixedpoint_scorer_n600.v1",
        "contract": {"native_integer_speed_claim": True},
        "model_manifest": {
            "minimum_bits": 26,
            "maximum_bits": 31,
            "converted_conv2d_count": 125,
            "accumulation": "exact_signed_int64",
            "finalization": "single_fp32_scale_and_bias_per_output",
            "assignment_rule": "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound",
            "bound_kind": "activation_qmax_times_max_output_quantized_weight_l1",
            "label_or_frame_dependent": False,
            "precision_histogram": {"27": 4, "28": 28, "29": 32, "30": 41, "31": 20},
        },
        "summary": {
            "status": "MEASURED",
            "full_real_n600": True,
            "argmax_exact_admitted": True,
            "training_tolerance_admitted": True,
            "rung2_weight_l1_integer_verdict": "WEIGHT_L1_INT64_ARGMAX_FEASIBLE",
            "verdict_scope": "n600 instance",
            "cache_custody": {
                "status": "MEASURED",
                "pairs": 600,
                "unique_pair_indices": 600,
                "observed_pair_indices_sha256": indices_hash,
                "expected_pair_indices_sha256": indices_hash,
            },
            "candidate": {"full": {"aggregate_flip_fraction": 0.0}},
            "timing": {"candidate_speedup_vs_reference_x": 0.1},
        },
    }
    _write(path, payload)
    anchor = build_weight_l1_int64_segnet_anchor(path, payload, repo=tmp_path)
    assert anchor.residual == 0.0
    assert anchor.inputs["bound_kind"] == (
        "activation_qmax_times_max_output_quantized_weight_l1"
    )
    payload["model_manifest"]["label_or_frame_dependent"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="integer custody"):
        build_weight_l1_int64_segnet_anchor(path, payload, repo=tmp_path)


def test_weight_l1_tie_snap_anchor_requires_split_honesty(tmp_path: Path) -> None:
    path = tmp_path / "tie_snap.json"
    indices_hash = "e" * 64
    epsilon = 2.0**-19
    payload = {
        "schema": "weight_l1_tie_snap_scorer_n600.v1",
        "contract": {
            "epsilon_ladder": [0.0, epsilon],
            "epsilon_selection": (
                "minimum calibration-exact epsilon; no heldout reselection"
            ),
            "decision_rule": "lowest class index within epsilon of candidate maximum",
            "runtime_label_or_frame_dependent": False,
            "calibration_split": [0, 120],
            "heldout_start": 120,
        },
        "model_manifest": {
            "minimum_bits": 26,
            "maximum_bits": 31,
            "converted_conv2d_count": 125,
            "accumulation": "exact_signed_int64",
            "assignment_rule": (
                "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound"
            ),
            "bound_kind": "activation_qmax_times_max_output_quantized_weight_l1",
            "label_or_frame_dependent": False,
            "precision_histogram": {
                "27": 4,
                "28": 28,
                "29": 32,
                "30": 41,
                "31": 20,
            },
        },
        "summary": {
            "status": "MEASURED",
            "full_real_n600": True,
            "argmax_exact_admitted": True,
            "minimum_calibration_exact_arm": "epsilon_2m19",
            "minimum_calibration_exact_epsilon": epsilon,
            "rung2_tie_snap_verdict": "TIE_SNAP_ARGMAX_FEASIBLE",
            "verdict_scope": "n600 instance",
            "cache_custody": {
                "status": "MEASURED",
                "pairs": 600,
                "unique_pair_indices": 600,
                "observed_pair_indices_sha256": indices_hash,
                "expected_pair_indices_sha256": indices_hash,
            },
            "arms": {
                "epsilon_2m19": {
                    "calibration": {"flips": 0},
                    "heldout": {"flips": 0},
                    "full": {"flips": 0, "aggregate_flip_fraction": 0.0},
                }
            },
        },
    }
    _write(path, payload)
    anchor = build_weight_l1_tie_snap_segnet_anchor(path, payload, repo=tmp_path)
    assert anchor.residual == 0.0
    assert anchor.empirical_output["selected_epsilon"] == epsilon
    payload["contract"]["epsilon_selection"] = "full-corpus reselection"  # type: ignore[index]
    with pytest.raises(ValueError, match="custody"):
        build_weight_l1_tie_snap_segnet_anchor(path, payload, repo=tmp_path)


def test_class_pair_tie_snap_anchor_preserves_second_validation(tmp_path: Path) -> None:
    path = tmp_path / "class_pair_tie_snap.json"
    indices_hash = "e" * 64
    payload = {
        "schema": "weight_l1_class_pair_tie_snap_scorer_n600.v1",
        "contract": {
            "design_split": [0, 264],
            "second_validation_split": [264, 600],
            "epsilon": 2.0**-19,
            "candidate_winner_class": 4,
            "candidate_runner_class": 0,
            "replacement_class": 0,
            "rule_frozen_before_second_validation_access": True,
            "second_validation_reselection": False,
            "runtime_label_or_frame_dependent": False,
        },
        "model_manifest": {
            "minimum_bits": 26,
            "maximum_bits": 31,
            "converted_conv2d_count": 125,
            "accumulation": "exact_signed_int64",
            "assignment_rule": (
                "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound"
            ),
            "bound_kind": "activation_qmax_times_max_output_quantized_weight_l1",
            "label_or_frame_dependent": False,
            "precision_histogram": {
                "27": 4,
                "28": 28,
                "29": 32,
                "30": 41,
                "31": 20,
            },
        },
        "summary": {
            "status": "MEASURED",
            "full_real_n600": True,
            "argmax_exact_admitted": True,
            "design_exact": True,
            "second_validation_exact": True,
            "rung2_class_pair_tie_snap_verdict": (
                "CLASS_PAIR_TIE_SNAP_ARGMAX_FEASIBLE"
            ),
            "verdict_scope": "n600 instance",
            "cache_custody": {
                "status": "MEASURED",
                "pairs": 600,
                "unique_pair_indices": 600,
                "observed_pair_indices_sha256": indices_hash,
                "expected_pair_indices_sha256": indices_hash,
            },
            "class_pair_tie_snap": {
                "design": {"flips": 0},
                "second_validation": {"flips": 0},
                "full": {"flips": 0, "aggregate_flip_fraction": 0.0},
            },
        },
    }
    _write(path, payload)
    anchor = build_weight_l1_class_pair_tie_snap_segnet_anchor(
        path,
        payload,
        repo=tmp_path,
    )
    assert anchor.residual == 0.0
    assert anchor.inputs["ordered_candidate_top2"] == [4, 0]
    payload["contract"]["second_validation_reselection"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="honest"):
        build_weight_l1_class_pair_tie_snap_segnet_anchor(
            path,
            payload,
            repo=tmp_path,
        )
