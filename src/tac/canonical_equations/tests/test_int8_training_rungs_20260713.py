from __future__ import annotations

import json

import pytest

from tac.canonical_equations.int8_training_rungs_20260713 import (
    EXPECTED_CHECKPOINT_SHA256,
    build_int8_teacher_admission_v1,
    build_int8_witness_posthoc_gap_v1,
    heterogeneous_overlap_seconds,
    int8_teacher_admission,
    int8_witness_posthoc_gap,
)


def test_teacher_gate_is_conjunctive_and_blocked_without_measurement() -> None:
    assert (
        int8_teacher_admission(
            speedup_x=None,
            global_gradient_cosine=None,
            minimum_pair_gradient_cosine=None,
            quality_pairs=0,
        )["verdict"]
        == "NO_VERDICT_BLOCKED"
    )
    assert (
        int8_teacher_admission(
            speedup_x=1.5,
            global_gradient_cosine=0.99,
            minimum_pair_gradient_cosine=0.99,
            quality_pairs=600,
        )["verdict"]
        == "GO"
    )
    assert (
        int8_teacher_admission(
            speedup_x=1.49,
            global_gradient_cosine=1.0,
            minimum_pair_gradient_cosine=1.0,
            quality_pairs=600,
        )["verdict"]
        == "NO_GO"
    )
    assert build_int8_teacher_admission_v1().empirical_anchors == ()


def test_qat_prize_is_signed_and_overlap_does_not_sum_independent_paths() -> None:
    row = int8_witness_posthoc_gap(d_seg_fp32=0.03, d_seg_int8=0.032)
    assert row["d_seg_gap_int8_minus_fp32"] == pytest.approx(0.002)
    assert row["seg_score_unit_gap_100x"] == pytest.approx(0.2)
    assert heterogeneous_overlap_seconds(
        gpu_witness_seconds=0.03, ane_forward_seconds=0.01, synchronization_seconds=0.002
    ) == pytest.approx(0.032)


def test_b_equation_requires_terminal_n600_parseback_receipt(tmp_path) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "status": "MEASURED",
                "completed_at_utc": "2026-07-13T12:00:00Z",
                "provenance": {
                    "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                    "packet": {"parse_back_equals_direct_int8_dequant": True},
                },
                "measurement": {
                    "n_pairs": 600,
                    "n600_evidence": True,
                    "d_seg_fp32_ema": 0.03,
                    "d_seg_parsed_int8": 0.032,
                    "d_seg_gap_int8_minus_fp32": 0.002,
                    "seg_score_unit_gap_100x": 0.2,
                    "positive_recovery_prize_ceiling": 0.002,
                },
            }
        )
    )
    equation = build_int8_witness_posthoc_gap_v1(receipt_path)
    assert equation.empirical_anchors[0].empirical_output["d_seg_gap_int8_minus_fp32"] == 0.002

    payload = json.loads(receipt_path.read_text())
    payload["measurement"]["n_pairs"] = 2
    receipt_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="n600"):
        build_int8_witness_posthoc_gap_v1(receipt_path)
