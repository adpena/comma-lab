# SPDX-License-Identifier: MIT
"""Focused checks for the Einstein--Kolmogorov action/rate contract."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tac.canonical_equations.einstein_kolmogorov_crux_20260719 import (
    EQUATION_ID,
    SOURCE_FRONTIER_MAGNITUDE,
    SOURCE_MEASUREMENT,
    InfeasibleByteBudgetError,
    MeasuredHardRReceipt,
    build_einstein_kolmogorov_crux_action_rate_contract_v1,
    contest_action,
    derive_research_only_decision,
    fixed_byte_palette_delta,
    frontier_feasible_at_zero_pose_and_rate,
    inclusive_maximum_byte_budget,
    maximum_byte_budget,
    populate_einstein_kolmogorov_crux_action_rate_contract_v1,
    validate_frontier_magnitude_chart,
)


@pytest.mark.parametrize(
    ("target", "d_pose", "expected"),
    [
        (0.1910828242, 0.0, 264_150),
        (0.1910828242, 1.0184e-4, 216_223),
        (0.15, 0.0, 202_451),
        (0.15, 1.0184e-4, 154_524),
    ],
)
def test_reported_center_byte_cap_examples(target: float, d_pose: float, expected: int) -> None:
    # Formula examples only. The C1 216,223 B row is a reported-center point estimate,
    # not an exact cap because the official distortion components have eight decimals.
    budget = maximum_byte_budget(target_action=target, d_seg=1.5196e-4, d_pose=d_pose)
    assert budget == expected


def test_strict_and_inclusive_byte_caps_diverge_at_exact_equality() -> None:
    target = 25 / 37_545_489
    assert inclusive_maximum_byte_budget(target_action=target, d_seg=0.0, d_pose=0.0) == 1
    assert maximum_byte_budget(target_action=target, d_seg=0.0, d_pose=0.0) == 0
    assert contest_action(d_seg=0.0, d_pose=0.0, archive_bytes=0) < target
    assert contest_action(d_seg=0.0, d_pose=0.0, archive_bytes=1) == target


def test_action_and_budget_are_monotone() -> None:
    assert contest_action(d_seg=0.001, d_pose=0.0, archive_bytes=101) > contest_action(
        d_seg=0.001, d_pose=0.0, archive_bytes=100
    )
    assert maximum_byte_budget(target_action=0.2, d_seg=0.001, d_pose=0.0) > maximum_byte_budget(
        target_action=0.19, d_seg=0.001, d_pose=0.0
    )


def test_wrong_operating_point_fails_seg_only_frontier_necessity_gate() -> None:
    pointer = 0.1910828242
    assert frontier_feasible_at_zero_pose_and_rate(d_seg=0.00015196, target_action=pointer)
    assert not frontier_feasible_at_zero_pose_and_rate(
        d_seg=0.0056786007351345485,
        target_action=pointer,
    )
    assert pointer < 100 * 0.0056786007351345485


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_action": 0.1, "d_seg": 0.002, "d_pose": 0.0},
        {"target_action": float("nan"), "d_seg": 0.0, "d_pose": 0.0},
    ],
)
def test_budget_rejects_infeasible_or_invalid_inputs(kwargs: dict[str, float]) -> None:
    expected = InfeasibleByteBudgetError if kwargs["target_action"] == 0.1 else ValueError
    with pytest.raises(expected):
        maximum_byte_budget(**kwargs)


def test_fixed_byte_palette_delta_has_no_rate_term() -> None:
    delta = fixed_byte_palette_delta(
        before_d_seg=0.02,
        before_d_pose=0.01,
        after_d_seg=0.015,
        after_d_pose=0.01,
        before_bytes=19_859,
        after_bytes=19_859,
    )
    assert math.isclose(delta, -0.5, abs_tol=1e-12)
    with pytest.raises(ValueError, match="identical packet bytes"):
        fixed_byte_palette_delta(
            before_d_seg=0.02,
            before_d_pose=0.0,
            after_d_seg=0.01,
            after_d_pose=0.0,
            before_bytes=1,
            after_bytes=2,
        )


def test_receipt_derivation_edges_keep_research_only_scope() -> None:
    receipt = MeasuredHardRReceipt(
        receipt_id="hard-r-receipt:caller-supplied",
        verdict_scope="n24 hard-R research-only",
        d_seg=0.001,
        d_pose=0.0,
        archive_bytes=100,
    )
    decision = derive_research_only_decision(receipt=receipt, target_action=0.2)
    assert decision.equation_id == EQUATION_ID
    assert decision.research_only is True
    assert decision.promotion_eligible is False
    assert [edge.relation for edge in decision.derivation_edges] == ["MEASURED_HARD_R_INPUT", "DERIVES", "SCOPES"]


def test_hash_bound_canonical_equation_builds_from_frozen_measurement() -> None:
    equation = build_einstein_kolmogorov_crux_action_rate_contract_v1()
    assert equation.equation_id == EQUATION_ID
    assert equation.provenance.source_path == SOURCE_MEASUREMENT
    assert equation.domain_of_validity["anchor_measurement_sha256"] == (equation.provenance.source_sha256)
    assert equation.domain_of_validity["research_only"] is True
    assert equation.domain_of_validity["promotion_eligible"] is False
    assert equation.domain_of_validity["n600_explicit_target_launch_eligible"] is False
    assert len(equation.empirical_anchors) == 2
    anchor = equation.empirical_anchors[0]
    assert anchor.provenance.source_sha256 == equation.provenance.source_sha256
    assert anchor.empirical_output["winner_hard_mismatch_px"] < (anchor.empirical_output["source_hard_mismatch_px"])
    assert anchor.empirical_output["winner_frontier_feasible_at_zero_pose_zero_rate"] is False
    assert anchor.empirical_output["operating_point_verdict"] == "WRONG_OPERATING_POINT_WALL_CHARACTERIZATION"
    assert anchor.empirical_output["full_archive_or_contest_score_claim"] is False
    frontier_anchor = equation.empirical_anchors[1]
    assert frontier_anchor.source_artifact == SOURCE_FRONTIER_MAGNITUDE
    assert frontier_anchor.empirical_output["exact_production_blocker"] == (
        "NO_COMPLETE_N600_ARCHIVE_WITHIN_TOTAL_SCORE_BYTE_CAP"
    )
    bank = frontier_anchor.empirical_output["bank"]
    assert bank["derived_action_interval_from_rounded_components"]["label"] == (
        "DERIVED_INTERVAL_FROM_ROUNDED_COMPONENTS"
    )
    assert bank["derived_strict_total_archive_cap"]["interval_bytes"] == [216_221, 216_225]
    assert bank["derived_strict_total_archive_cap"]["guaranteed_safe_cap_bytes"] == 216_221
    assert frontier_anchor.empirical_output["pose_clean_trade_cells_control"]["point_id"] == "precision_drop1"
    assert frontier_anchor.empirical_output["matched_receiver_control"]["archive_bytes"] == 7_898_534
    assert frontier_anchor.empirical_output["matched_receiver_treatment"]["archive_bytes"] == 6_728_570
    assert frontier_anchor.empirical_output["matched_receiver_treatment"]["derived_n600_action"] == pytest.approx(
        224.29719362434912,
        abs=1e-12,
    )
    assert frontier_anchor.empirical_output["pointer_moved"] is False


def test_frontier_magnitude_chart_recomputes_every_projection() -> None:
    chart = validate_frontier_magnitude_chart()
    assert chart["n600_trade_cells_launch_eligible"] is False
    assert chart["selection"]["exact_receiver_frontier_magnitude_control"] == ("v10_rung_e_exact_two_plane_n48_local")
    assert chart["selection"]["direct_rgb_frontier_magnitude_pose_clean_control"] == ("precision_drop1")
    assert chart["selection"]["matched_receiver_positive_band_control"] == ("banked_n12_scorer_plane_precision_drop1")
    assert len(chart["trade_cells_curve"]["points"]) == 10
    assert len(chart["exact_archive_rows"]) == 4
    assert len(chart["source_receipts"]) == 8
    assert chart["exact_production_gap"]["blocker"] == "NO_COMPLETE_N600_ARCHIVE_WITHIN_TOTAL_SCORE_BYTE_CAP"
    assert chart["exact_production_gap"]["secondary_blocker"] == "MISSING_ARBITRARY_NUMERATOR_PLANE_CODEC"
    bank = next(row for row in chart["exact_archive_rows"] if row["point_id"].startswith("c1_solved"))
    assert "exact_action" not in bank
    assert "strict_bytes_to_beat_pointer_at_measured_distortion" not in bank
    assert bank["derived_action_interval_from_rounded_components"]["lower"].startswith("272.73427665248019432934")
    assert bank["derived_action_interval_from_rounded_components"]["upper"].startswith("272.73427921927025974944")
    assert bank["derived_strict_total_archive_cap"]["scope"] == ("total archive.zip bytes, not a predictor-only budget")


def test_frontier_chart_refuses_nonexistent_local_source_receipt(tmp_path: Path) -> None:
    payload = json.loads(Path(SOURCE_FRONTIER_MAGNITUDE).read_text())
    payload["source_receipts"][3]["path"] = ".omx/research/nonexistent-fake-receipt.json"
    payload["source_receipts"][3]["sha256"] = "0" * 64
    chart_path = tmp_path / "fake-source-chart.json"
    chart_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source artifact is absent"):
        validate_frontier_magnitude_chart(chart_path)


def test_frontier_chart_refuses_banked_v3_content_mismatch(tmp_path: Path) -> None:
    payload = json.loads(Path(SOURCE_FRONTIER_MAGNITUDE).read_text())
    control = next(
        row for row in payload["exact_archive_rows"] if row["point_id"] == "banked_n12_exact_receiver_control"
    )
    control["archive_bytes"] += 1
    chart_path = tmp_path / "banked-drift-chart.json"
    chart_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="drifted from banked v3 arm control"):
        validate_frontier_magnitude_chart(chart_path)


def test_canonical_equation_registry_query_roundtrip(tmp_path: Path) -> None:
    from tac.canonical_equations.registry import query_equations

    registry = tmp_path / "canonical_equations.jsonl"
    equation = populate_einstein_kolmogorov_crux_action_rate_contract_v1(
        path=registry,
        lock_path=tmp_path / "canonical_equations.lock",
        agent="pytest",
        subagent_id="einstein-kolmogorov-registry-test",
    )
    loaded = query_equations(path=registry)
    assert [item.equation_id for item in loaded] == [EQUATION_ID]
    assert loaded[0].provenance.source_sha256 == equation.provenance.source_sha256
    assert loaded[0].empirical_anchors[0].empirical_output == (equation.empirical_anchors[0].empirical_output)
    assert loaded[0].empirical_anchors[1].empirical_output == (equation.empirical_anchors[1].empirical_output)
