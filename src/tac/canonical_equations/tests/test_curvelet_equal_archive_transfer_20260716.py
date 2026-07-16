from __future__ import annotations

from copy import deepcopy

import pytest

from tac.canonical_equations.curvelet_equal_archive_transfer_20260716 import (
    CurveletTransferReceiptError,
    evaluate_curvelet_equal_archive_transfer,
)

PROGRAM_SHA = "1" * 64
CONTROL_TREE_SHA = "c" * 64
TREATMENT_TREE_SHA = "d" * 64


def _custody(*, output_tree_sha256: str) -> dict:
    return {
        "n_pairs": 600,
        "n_samples": 600,
        "scorer_batch_size": 32,
        "through_r": True,
        "official_evaluator": True,
        "parse_back": True,
        "output_tree_sha256": output_tree_sha256,
        "upstream_snapshot_sha256": "3" * 64,
        "runtime_sha256": "4" * 64,
        "checkpoint_sha256": "5" * 64,
        "evaluate_report_sha256": "6" * 64,
        "segnet_weights_sha256": "7" * 64,
        "posenet_weights_sha256": "8" * 64,
        "git_sha": "9" * 40,
        "measurement_utc": "2026-07-16T20:00:00Z",
        "hardware_substrate": "contest-linux-x86_64-cpu",
        "torch_version": "2.8.0",
        "device": "cpu",
        "evaluator_argv": ["python3", "upstream/evaluate.py", "--device", "cpu"],
        "seed": 0,
    }


def _receipt(*, axis: str = "contest-CPU", treatment_d_seg: float = 0.003) -> dict:
    return {
        "schema": "curvelet_equal_archive_transfer.v1",
        "lawref": "curvelet_equal_archive_transfer_v1",
        "basis_program_sha256": PROGRAM_SHA,
        "equal_budget_receipt_verified": True,
        "output_trees_preserved": True,
        "output_tree_custody": {
            "control_sha256": CONTROL_TREE_SHA,
            "treatment_sha256": TREATMENT_TREE_SHA,
        },
        "equal_budget": {
            "equal_archive_bytes": True,
            "target_archive_bytes": 1234,
            "left": {"matched_archive_bytes": 1234, "matched_archive_sha256": "a" * 64},
            "right": {"matched_archive_bytes": 1234, "matched_archive_sha256": "b" * 64},
        },
        "measurements": {
            "control": {
                **_custody(output_tree_sha256=CONTROL_TREE_SHA),
                "family": "legacy_fourier_ab_control",
                "archive_sha256": "a" * 64,
                "archive_bytes": 1234,
                "axis": axis,
                "d_seg": 0.004,
                "d_pose": 0.02,
            },
            "treatment": {
                **_custody(output_tree_sha256=TREATMENT_TREE_SHA),
                "family": "literal_polar_curvelet",
                "basis_program_sha256": PROGRAM_SHA,
                "archive_sha256": "b" * 64,
                "archive_bytes": 1234,
                "axis": axis,
                "d_seg": treatment_d_seg,
                "d_pose": 0.019,
            },
        },
    }


def test_equal_archive_transfer_is_instance_scoped_and_pointer_inert() -> None:
    verdict = evaluate_curvelet_equal_archive_transfer(_receipt())
    assert verdict.status == "MEASURED_TRANSFER_FORMULATION_INSTANCE"
    assert verdict.family_verdict == "OPEN"
    assert verdict.equal_archive_bytes == 1234
    assert verdict.delta_d_seg_treatment_minus_control == pytest.approx(-0.001)
    assert verdict.delta_nonrate_score < 0.0
    assert verdict.pointer_authorized is False


def test_advisory_axis_never_promotes() -> None:
    verdict = evaluate_curvelet_equal_archive_transfer(_receipt(axis="macOS-CPU advisory"))
    assert verdict.status == "ADVISORY_ONLY_NO_PROMOTION"
    assert verdict.family_verdict == "OPEN"


@pytest.mark.parametrize(
    ("path", "value", "match"),
    (
        (("equal_budget_receipt_verified",), False, "not re-derived"),
        (("measurements", "control", "n_pairs"), 6, "full-n600"),
        (("measurements", "treatment", "archive_sha256"), "c" * 64, "matched ZIP"),
        (("measurements", "treatment", "basis_program_sha256"), "2" * 64, "BasisProgramConfig"),
    ),
)
def test_custody_drift_fails_closed(path: tuple[str, ...], value: object, match: str) -> None:
    receipt = deepcopy(_receipt())
    target = receipt
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(CurveletTransferReceiptError, match=match):
        evaluate_curvelet_equal_archive_transfer(receipt)
