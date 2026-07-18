# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from experiments.measure_shared_resize_seg_pose_coupling_20260718 import (
    aggregate_response_metrics,
)
from tac.canonical_equations.shared_resize_joint_coupling_20260718 import (
    EQUATION_ID,
    NON_RESOLVING_DISPLAY_ALIAS,
    build_shared_resize_joint_coupling_through_A_v1,
    joint_costate_coefficients,
    load_bound_measurement_receipt,
    normalized_overlap,
    populate_shared_resize_joint_coupling_equation,
    pose_score_marginal,
    smooth_coupling_summary,
)
from tac.witness_dsl.shared_resize_joint_coupling_policy import (
    deterministic_stride_sample_ids,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bound_receipt(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, Path]]:
    contents = {
        "gt_cache": b"gt-cache",
        "segnet": b"segnet",
        "posenet": b"posenet",
        "modules_py": b"modules",
        "frame_utils_py": b"frame-utils",
        "evaluate_py": b"evaluate",
    }
    paths: dict[str, Path] = {}
    for name, data in contents.items():
        path = tmp_path / f"{name}.bin"
        path.write_bytes(data)
        paths[name] = path
    checkpoint_path = tmp_path / "checkpoint.npz"
    np.savez(checkpoint_path, code=np.zeros((1,), dtype=np.float32))
    paths["checkpoint"] = checkpoint_path
    contents["checkpoint"] = checkpoint_path.read_bytes()
    count = {"nonzero_requested": 1, "realized_changed": 1, "boundary_clipped": 0}
    baseline = {"d_seg": 0.2, "d_pose": 0.4}
    arm_scores = {
        "seg_plus": {"d_seg": 0.19, "d_pose": 0.41},
        "seg_minus": {"d_seg": 0.21, "d_pose": 0.39},
        "pose_plus": {"d_seg": 0.18, "d_pose": 0.38},
        "pose_minus": {"d_seg": 0.22, "d_pose": 0.42},
        "joint_plus": {"d_seg": 0.19, "d_pose": 0.39},
    }
    derived = aggregate_response_metrics(baseline, **arm_scores)
    pair_ids = list(deterministic_stride_sample_ids(600, 8, 538))
    receipt: dict[str, object] = {
        "schema": "shared_resize_joint_coupling_measurement.v2",
        "captured_at_utc": "2026-07-18T12:00:00+00:00",
        "axis": "[macOS-CPU advisory]",
        "evidence_status": "MEASURED_ADVISORY_SUBSET",
        "n_pairs_total": 600,
        "sample": {
            "method": "deterministic_cyclic_stride",
            "n_of_600": 8,
            "pair_ids": pair_ids,
            "seed": 538,
        },
        "input_custody": {
            "checkpoint_path": str(paths["checkpoint"]),
            "checkpoint_sha256": _sha(contents["checkpoint"]),
            "gt_cache_path": str(paths["gt_cache"]),
            "gt_cache_sha256": _sha(contents["gt_cache"]),
            "segnet_path": str(paths["segnet"]),
            "segnet_sha256": _sha(contents["segnet"]),
            "posenet_path": str(paths["posenet"]),
            "posenet_sha256": _sha(contents["posenet"]),
            "checkpoint_payload": {
                "carrier_absent": True,
                "base_inr_only": True,
                "detected_carrier_keys": [],
                "checkpoint_key_count": 1,
                "checkpoint_key_manifest_sha256": hashlib.sha256(b'["code"]').hexdigest(),
            },
        },
        "execution_custody": {
            "git_head": "a" * 40,
            "argv": ["measure.py"],
            "config": {"n_sample": 8, "seed": 538},
            "input_bytes": {name: len(data) for name, data in contents.items()},
            "upstream_source_sha256": {
                name: _sha(contents[name])
                for name in ("modules_py", "frame_utils_py", "evaluate_py")
            },
            "upstream_source_paths": {
                name: str(paths[name])
                for name in ("modules_py", "frame_utils_py", "evaluate_py")
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
                "seg_total_pixels": 8 * 384 * 512,
                "seg_mismatch_fraction": 0.0,
                "pose_mismatched_elements": 0,
                "pose_total_elements": 48,
                "pose_max_abs": 0.0,
                "pose_mse": 0.0,
            },
        },
        "shared_A": {
            "seg_pose_operator_identical": True,
            "operator": "torch.nn.functional.interpolate(mode=bilinear,align_corners=False)",
            "camera_hw": [874, 1164],
            "scorer_hw": [384, 512],
            "seg_preprocess_tensor_equal": True,
            "pose_yuv6_clone_max_abs": 0.0,
        },
        "smooth_coupling": {
            "evidence_label": "B1_LOCAL_DERIVED",
            "primary_surface": "shared_frame1",
            "shared_frame1": {
                "raw_gram_2x2": [[2.0, 0.5], [0.5, 1.0]],
                "score_priced_gram_2x2": [[20000.0, 50.0], [50.0, 5.0]],
                "normalized_overlap": 0.35355339059327373,
            },
            "full_pair_context": {
                "raw_gram_2x2": [[2.0, 0.5], [0.5, 2.0]],
                "score_priced_gram_2x2": [[20000.0, 50.0], [50.0, 10.0]],
                "normalized_overlap": 0.25,
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
                        name: dict(count)
                        for name in (
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
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    return receipt_path, receipt, paths


def test_score_derived_coefficients_have_no_tunable_coupling_constant() -> None:
    expected = 5.0 / math.sqrt(20.0)
    assert pose_score_marginal(2.0) == pytest.approx(expected)
    assert joint_costate_coefficients(2.0) == {
        "lambda_seg": 100.0,
        "lambda_pose": pytest.approx(expected),
    }
    with pytest.raises(ValueError, match="strictly positive"):
        pose_score_marginal(0.0)


def test_smooth_summary_prices_gram_and_preserves_cosine() -> None:
    row = smooth_coupling_summary([[4.0, -1.0], [-1.0, 9.0]], d_pose_baseline=2.0)
    assert row["normalized_overlap"] == pytest.approx(-1.0 / 6.0)
    assert row["score_priced_gram_2x2"][0][0] == pytest.approx(40000.0)
    assert row["score_priced_gram_2x2"][0][1] == pytest.approx(
        -100.0 * pose_score_marginal(2.0)
    )
    assert normalized_overlap(0.0, 0.0, 3.0) == 0.0


def test_nonphysical_gram_fails_closed() -> None:
    with pytest.raises(ValueError, match="Cauchy-Schwarz"):
        smooth_coupling_summary([[1.0, 2.0], [2.0, 1.0]], d_pose_baseline=1.0)
    with pytest.raises(ValueError, match="symmetric"):
        smooth_coupling_summary([[1.0, 0.0], [1.0, 1.0]], d_pose_baseline=1.0)


def test_pending_equation_is_structural_only_and_registers_in_temp_registry(tmp_path) -> None:
    equation = build_shared_resize_joint_coupling_through_A_v1()
    assert EQUATION_ID == "shared_resize_joint_coupling_through_a_v1"
    assert equation.equation_id == EQUATION_ID
    assert NON_RESOLVING_DISPLAY_ALIAS == "shared_resize_joint_coupling_through_A_v1"
    assert equation.domain_of_validity["display_alias_support"] == (
        "NON_RESOLVING_DISPLAY_ALIAS"
    )
    assert not equation.empirical_anchors
    assert equation.domain_of_validity["subset_evidence_only"] is False
    assert equation.domain_of_validity["score_claim"] is False

    registry = tmp_path / "equations.jsonl"
    lock = tmp_path / "equations.lock"
    populate_shared_resize_joint_coupling_equation(path=registry, lock_path=lock)
    rows = [json.loads(line) for line in registry.read_text().splitlines()]
    assert rows[-1]["equation_id"] == EQUATION_ID
    assert rows[-1]["equation_payload"]["domain_of_validity"][
        "empirical_verification_status"
    ] == (
        "ASSUMED_AWAITING_VERIFICATION"
    )


def test_unbound_mapping_remains_pending_and_cannot_form_anchor() -> None:
    equation = build_shared_resize_joint_coupling_through_A_v1(
        measurement_receipt={"fabricated": True}
    )
    assert not equation.empirical_anchors
    assert equation.domain_of_validity["unbound_mapping_supplied_pending_only"] is True
    with pytest.raises(ValueError, match="receipt path only"):
        build_shared_resize_joint_coupling_through_A_v1(
            measurement_receipt={"fabricated": True},
            measurement_receipt_path="receipt.json",
        )


def test_bound_receipt_rehashes_inputs_and_binds_digest(tmp_path) -> None:
    receipt_path, _receipt, _paths = _write_bound_receipt(tmp_path)
    bound = load_bound_measurement_receipt(receipt_path)
    assert bound["receipt_sha256"] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert set(bound["verified_artifacts"]) == {
        "checkpoint",
        "gt_cache",
        "segnet",
        "posenet",
        "modules_py",
        "frame_utils_py",
        "evaluate_py",
    }
    equation = build_shared_resize_joint_coupling_through_A_v1(
        measurement_receipt_path=receipt_path
    )
    anchor = equation.empirical_anchors[0]
    assert anchor.inputs["measurement_receipt_sha256"] == bound["receipt_sha256"]
    assert anchor.residual == 0.0
    assert anchor.predicted_output["shared_A_parity_expected"] == {
        "seg_pose_operator_identical": True,
        "seg_preprocess_tensor_equal": True,
        "pose_yuv6_clone_max_abs": 0.0,
    }
    assert anchor.empirical_output["shared_A_parity"] == {
        "seg_pose_operator_identical": True,
        "seg_preprocess_tensor_equal": True,
        "pose_yuv6_clone_max_abs": 0.0,
    }
    assert (
        anchor.empirical_output["comparison_contract"]
        == "NONCOMMENSURATE_NO_CROSS_SURFACE_RESIDUAL"
    )
    assert "smooth_input_gradient_overlap_surrogate" in anchor.empirical_output
    assert "finite_lattice_response_column_overlap_unpriced" in anchor.empirical_output
    assert equation.predicted_vs_empirical_residual == {
        "shared_A_YUV6_parity_max_abs": 0.0
    }
    assert anchor.provenance.source_sha256 == bound["receipt_sha256"]


def test_bound_receipt_rejects_input_tamper(tmp_path) -> None:
    receipt_path, _receipt, paths = _write_bound_receipt(tmp_path)
    paths["segnet"].write_bytes(b"tampered-segnet")
    with pytest.raises(ValueError, match="custody mismatch for segnet"):
        load_bound_measurement_receipt(receipt_path)


def test_bound_receipt_rejects_fabricated_sample_ids(tmp_path) -> None:
    receipt_path, receipt, _paths = _write_bound_receipt(tmp_path)
    receipt["sample"]["pair_ids"][0] += 1  # type: ignore[index]
    receipt_path.write_text(json.dumps(receipt, sort_keys=True))
    with pytest.raises(ValueError, match="deterministic stride"):
        load_bound_measurement_receipt(receipt_path)
