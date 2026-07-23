# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest
from pydantic import ValidationError

from tac.optimization.ddm_dr2b_tolerance_costate import (
    DDMDR2BMeasurementError,
    exact_n600_rebase,
    frequency_band_admission,
    head_flip_distance,
    ordered_redundancy_matrix,
    rank_costate_rows,
    require_description_crosswalk,
)
from tac.optimization.ddm_runtime_sensitivity import RuntimeSensitivityError
from tools.measure_ddm_dr2b_tolerance_costate import (
    DDMDR2BToleranceCostateConfigV1,
    PerturbationProbeV1,
    _receiver_blocker_row,
)


def test_exact_n600_rebase_scales_one_canonical_window() -> None:
    row = exact_n600_rebase(
        baseline_d_seg=0.2,
        baseline_d_pose=4.0,
        window_d_seg_before=0.25,
        window_d_seg_after=0.125,
        window_d_pose_before=5.0,
        window_d_pose_after=5.6,
        window_pair_count=16,
        delta_bytes=3,
    )
    assert row["delta_d_seg"] == pytest.approx(-0.125 * 16 / 600)
    assert row["delta_d_pose"] == pytest.approx(0.6 * 16 / 600)
    assert row["d_seg"] == pytest.approx(0.2 - 0.125 * 16 / 600)
    assert row["n_pairs"] == 600
    assert row["exact_pair_local_rebase"] is True
    assert row["bytes_added"] == 3


def test_exact_n600_rebase_refuses_noncanonical_batch_geometry() -> None:
    with pytest.raises(
        DDMDR2BMeasurementError,
        match="canonical batch-16",
    ):
        exact_n600_rebase(
            baseline_d_seg=0.2,
            baseline_d_pose=4.0,
            window_d_seg_before=0.2,
            window_d_seg_after=0.2,
            window_d_pose_before=4.0,
            window_d_pose_after=4.0,
            window_pair_count=8,
            delta_bytes=0,
        )


def test_fisher_head_flip_distance_is_margin_over_normal() -> None:
    assert head_flip_distance(
        margin=-3.0,
        head_normal_norm=4.0,
    ) == pytest.approx(0.75)
    with pytest.raises(DDMDR2BMeasurementError, match="positive"):
        head_flip_distance(margin=1.0, head_normal_norm=0.0)


def test_frequency_band_admission_requires_exact_r_or_measured_debt() -> None:
    assert (
        frequency_band_admission(
            exact_r_transfer_zero=True,
            emitted_description_bytes=0,
        )["status"]
        == "ADMITTED_EXACT_R_NULL_TRUNCATION"
    )
    with pytest.raises(DDMDR2BMeasurementError, match="must emit zero"):
        frequency_band_admission(
            exact_r_transfer_zero=True,
            emitted_description_bytes=1,
        )
    with pytest.raises(DDMDR2BMeasurementError, match="cannot claim zero"):
        frequency_band_admission(
            exact_r_transfer_zero=False,
            emitted_description_bytes=0,
        )


def test_receiver_blocker_preserves_exact_cause_chain() -> None:
    probe = PerturbationProbeV1(
        probe_id="anchor",
        stream="base/chart.anchors",
        flat_index=0,
        delta=-4,
        expected_original_value=20,
        pair_id=0,
        purpose="chart_tolerance",
    )
    try:
        raise ValueError("chart reconstruction escaped uint8")
    except ValueError as cause:
        error = RuntimeSensitivityError("perturbed state failed receiver realization")
        error.__cause__ = cause
    row = _receiver_blocker_row(probe=probe, error=error)
    assert row["measurement_status"] == "BLOCKED_RECEIVER_INVALID"
    assert row["receiver_blocker"]["cause_type"] == "ValueError"
    assert row["receiver_blocker"]["cause_reason"] == "chart reconstruction escaped uint8"


def test_ordered_redundancy_matrix_is_directional_and_complete() -> None:
    streams = {
        "base": b"abc" * 40,
        "template": b"abc" * 10 + b"z",
        "sparse": b"q" * 25,
    }
    rows = ordered_redundancy_matrix(
        streams,
        decode_order=("base", "template", "sparse"),
    )
    assert len(rows) == 6
    assert {(row["conditioner"], row["stream"]) for row in rows} == {
        (left, right) for left in streams for right in streams if left != right
    }
    assert all(row["first_rung"] is True for row in rows)


def test_costate_rank_uses_reduced_cost_and_keeps_first_rung() -> None:
    rows = [
        {
            "probe_id": "higher",
            "first_rung": True,
            "n600_rebase": {"joint_delta": 0.2},
        },
        {
            "probe_id": "lower",
            "first_rung": True,
            "n600_rebase": {"joint_delta": -0.1},
        },
    ]
    ranked = rank_costate_rows(rows)
    assert [row["probe_id"] for row in ranked] == ["lower", "higher"]
    assert [row["costate_rank"] for row in ranked] == [1, 2]


def test_description_crosswalk_fails_closed_when_missing() -> None:
    with pytest.raises(
        DDMDR2BMeasurementError,
        match="SDWL1-fact to E2-runtime",
    ):
        require_description_crosswalk(None)


def _minimal_config() -> dict:
    bound = {"path": "x", "bytes": 1, "sha256": "a" * 64}
    return {
        "schema": "DDMDR2BToleranceCostateConfigV1",
        "run_id": "ddm_dr2b_tolerance_ladder_and_costate_rows_20260723",
        "authority_sha256": "b" * 64,
        "delegation_checkpoint_key": ("codex_delegate:ddm_dr2b_tolerance_ladder_and_costate_rows:20260723T184128Z"),
        "e2_archive": bound,
        "e2_verification_receipt": bound,
        "e2_findings_receipt": bound,
        "dr2_receipt": bound,
        "dr1_receipt": bound,
        "v19b_receipt": bound,
        "dv2_receipt": bound,
        "dv2_fact_inventory": bound,
        "dv2_selected_payload": bound,
        "scorer_config": bound,
        "output_directory": ".omx/research/test",
        "probes": [
            {
                "probe_id": "chart",
                "stream": "base/chart.anchors",
                "flat_index": 0,
                "delta": 1,
                "expected_original_value": 0,
                "pair_id": 0,
                "purpose": "chart_tolerance",
            }
        ],
    }


def test_config_rejects_quarantined_lineage_reference() -> None:
    value = _minimal_config()
    value["e2_archive"] = {
        "path": "forbidden_pr110_archive.zip",
        "bytes": 1,
        "sha256": "a" * 64,
    }
    with pytest.raises(ValidationError, match="quarantined"):
        DDMDR2BToleranceCostateConfigV1.model_validate(value)


def test_config_rejects_semantic_probe_without_cell() -> None:
    value = _minimal_config()
    value["probes"] = [
        {
            "probe_id": "semantic",
            "stream": "semantic/composed",
            "flat_index": 0,
            "delta": 1,
            "expected_original_value": 0,
            "pair_id": 0,
            "purpose": "semantic_boundary",
        }
    ]
    with pytest.raises(ValidationError, match="require one scorer cell"):
        DDMDR2BToleranceCostateConfigV1.model_validate(value)
