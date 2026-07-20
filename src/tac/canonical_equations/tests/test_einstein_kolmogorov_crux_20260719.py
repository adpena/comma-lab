# SPDX-License-Identifier: MIT
"""Focused checks for the Einstein--Kolmogorov action/rate contract."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import tac.canonical_equations.einstein_kolmogorov_crux_20260719 as crux_equation
from tac.canonical_equations.einstein_kolmogorov_crux_20260719 import (
    EQUATION_ID,
    SOURCE_FRONTIER_MAGNITUDE,
    SOURCE_FRONTIER_MAGNITUDE_SHA256,
    SOURCE_MEASUREMENT,
    SOURCE_MEASUREMENT_SHA256,
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
from tac.canonical_equations.equation import VERIFIED_VIA_EMPIRICAL_ANCHOR


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
    with pytest.raises(ValueError, match="source-receipt set drifted"):
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


def test_frontier_chart_refuses_self_consistent_forged_rung_e_values(tmp_path: Path) -> None:
    payload = json.loads(Path(SOURCE_FRONTIER_MAGNITUDE).read_text())
    row = next(
        row for row in payload["exact_archive_rows"] if row["point_id"] == "v10_rung_e_exact_two_plane_n48_local"
    )
    row["d_seg"] = row["d_seg"] * 2
    row["seg_term"] = 100 * row["d_seg"]
    row["derived_n600_action_at_unchanged_mean_distortion"] = contest_action(
        d_seg=row["d_seg"],
        d_pose=row["d_pose"],
        archive_bytes=row["derived_n600_linear_archive_bytes"],
    )
    row["strict_bytes_to_beat_pointer_at_measured_distortion"] = maximum_byte_budget(
        target_action=payload["pointer"]["score"],
        d_seg=row["d_seg"],
        d_pose=row["d_pose"],
    )
    chart_path = tmp_path / "forged-rung-e-chart.json"
    chart_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="rung-E chart row drifted from source receipt fields: d_seg"):
        validate_frontier_magnitude_chart(chart_path)


def test_frontier_chart_refuses_self_consistent_forged_precision_drop1_values(tmp_path: Path) -> None:
    payload = json.loads(Path(SOURCE_FRONTIER_MAGNITUDE).read_text())
    row = next(row for row in payload["trade_cells_curve"]["points"] if row["point_id"] == "precision_drop1")
    row["d_pose"] = row["d_pose"] * 2
    row["derived_action_on_declared_payload_scope"] = contest_action(
        d_seg=row["d_seg"],
        d_pose=row["d_pose"],
        archive_bytes=row["derived_n600_payload_bytes"],
    )
    chart_path = tmp_path / "forged-precision-drop1-chart.json"
    chart_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=r"trade-cells chart row precision_drop1 drifted.*d_pose"):
        validate_frontier_magnitude_chart(chart_path)


def test_frontier_chart_refuses_arbitrary_replacement_inverse_row(tmp_path: Path) -> None:
    payload = json.loads(Path(SOURCE_FRONTIER_MAGNITUDE).read_text())
    payload["exact_inverse_nonarchive_rows"][1] = {
        "point_id": "arbitrary_inverse_replacement",
        "pair_count": 24,
        "d_seg": 0.0,
        "d_pose": 0.0,
    }
    chart_path = tmp_path / "replacement-inverse-chart.json"
    chart_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="exact inverse nonarchive chart row set drifted"):
        validate_frontier_magnitude_chart(chart_path)


def test_frontier_chart_refuses_extra_exact_archive_row(tmp_path: Path) -> None:
    payload = json.loads(Path(SOURCE_FRONTIER_MAGNITUDE).read_text())
    extra = dict(payload["exact_archive_rows"][1])
    extra["point_id"] = "unreceipted_extra_exact_archive"
    payload["exact_archive_rows"].append(extra)
    chart_path = tmp_path / "extra-exact-archive-chart.json"
    chart_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="exact archive chart row set drifted"):
        validate_frontier_magnitude_chart(chart_path)


def _unmount_absolute_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original_resolver = crux_equation._resolve_source_path

    def resolve_with_unmounted_absolute_sources(value: str) -> Path:
        if Path(value).is_absolute():
            return tmp_path / "unmounted" / Path(value).name
        return original_resolver(value)

    monkeypatch.setattr(crux_equation, "_resolve_source_path", resolve_with_unmounted_absolute_sources)


def test_frontier_chart_retains_frozen_absolute_declarations_when_sources_are_unmounted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _unmount_absolute_sources(monkeypatch, tmp_path)
    chart = validate_frontier_magnitude_chart()
    assert len(chart["source_receipts"]) == 8
    assert {row["point_id"] for row in chart["exact_inverse_nonarchive_rows"]} == {
        "factor2_exact_lattice_frame1_n600",
        "joint_zero_band_n24",
    }


def test_unmounted_absolute_sources_refuse_self_consistent_forged_c1_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _unmount_absolute_sources(monkeypatch, tmp_path)
    payload = json.loads(Path(SOURCE_FRONTIER_MAGNITUDE).read_text())
    bank = next(
        row for row in payload["exact_archive_rows"] if row["point_id"] == "c1_solved_distortion_n600_contest_cpu"
    )
    bank["d_seg"] *= 2
    lower_seg, upper_seg = crux_equation._rounded_component_interval(bank["d_seg"], decimal_places=8)
    lower_pose, upper_pose = crux_equation._rounded_component_interval(bank["d_pose"], decimal_places=8)
    bank["rounded_component_intervals"]["d_seg"]["lower"] = str(lower_seg)
    bank["rounded_component_intervals"]["d_seg"]["upper"] = str(upper_seg)
    bank["derived_seg_term_from_reported_center"] = 100 * bank["d_seg"]
    bank["derived_action_point_estimate_from_reported_centers"] = contest_action(
        d_seg=bank["d_seg"],
        d_pose=bank["d_pose"],
        archive_bytes=bank["archive_bytes"],
    )
    bank["derived_action_interval_from_rounded_components"]["lower"] = str(
        crux_equation._decimal_action(
            d_seg=lower_seg,
            d_pose=lower_pose,
            archive_bytes=bank["archive_bytes"],
        )
    )
    bank["derived_action_interval_from_rounded_components"]["upper"] = str(
        crux_equation._decimal_action(
            d_seg=upper_seg,
            d_pose=upper_pose,
            archive_bytes=bank["archive_bytes"],
        )
    )
    cap_min = crux_equation._decimal_strict_byte_budget(
        target=crux_equation.Decimal(str(payload["pointer"]["score"])),
        d_seg=upper_seg,
        d_pose=upper_pose,
    )
    cap_max = crux_equation._decimal_strict_byte_budget(
        target=crux_equation.Decimal(str(payload["pointer"]["score"])),
        d_seg=lower_seg,
        d_pose=lower_pose,
    )
    cap = bank["derived_strict_total_archive_cap"]
    cap["point_estimate_from_reported_centers"] = maximum_byte_budget(
        target_action=payload["pointer"]["score"],
        d_seg=bank["d_seg"],
        d_pose=bank["d_pose"],
    )
    cap["interval_bytes"] = [cap_min, cap_max]
    cap["guaranteed_safe_cap_bytes"] = cap_min
    chart_path = tmp_path / "unmounted-forged-c1-chart.json"
    chart_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="C1 chart row drifted from source receipt fields: d_seg"):
        validate_frontier_magnitude_chart(chart_path)


def test_unmounted_absolute_sources_refuse_forged_factor2_lattice_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _unmount_absolute_sources(monkeypatch, tmp_path)
    payload = json.loads(Path(SOURCE_FRONTIER_MAGNITUDE).read_text())
    row = next(
        row
        for row in payload["exact_inverse_nonarchive_rows"]
        if row["point_id"] == "factor2_exact_lattice_frame1_n600"
    )
    row["d_seg"] *= 2
    chart_path = tmp_path / "unmounted-forged-lattice-chart.json"
    chart_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="exact-lattice inverse chart row drifted from source receipt fields: d_seg"):
        validate_frontier_magnitude_chart(chart_path)


def test_canonical_equation_registry_query_roundtrip(tmp_path: Path) -> None:
    from tac.canonical_equations.registry import query_equations, query_equations_by_producer

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
    assert [anchor.empirical_verification_status for anchor in loaded[0].empirical_anchors] == [
        VERIFIED_VIA_EMPIRICAL_ANCHOR,
        VERIFIED_VIA_EMPIRICAL_ANCHOR,
    ]
    assert [item.equation_id for item in query_equations_by_producer(SOURCE_MEASUREMENT, path=registry)] == [
        EQUATION_ID
    ]
    assert [item.equation_id for item in query_equations_by_producer(SOURCE_FRONTIER_MAGNITUDE, path=registry)] == [
        EQUATION_ID
    ]


def test_canonical_producer_files_match_frozen_authority_hashes() -> None:
    assert crux_equation._sha256_file(Path(SOURCE_MEASUREMENT)) == SOURCE_MEASUREMENT_SHA256
    assert crux_equation._sha256_file(Path(SOURCE_FRONTIER_MAGNITUDE)) == SOURCE_FRONTIER_MAGNITUDE_SHA256


def test_builder_provenance_is_cwd_independent_and_keeps_stable_source_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    equation = build_einstein_kolmogorov_crux_action_rate_contract_v1()

    assert equation.provenance.source_path == SOURCE_MEASUREMENT
    assert equation.provenance.source_sha256 == SOURCE_MEASUREMENT_SHA256
    assert equation.empirical_anchors[0].provenance.source_sha256 == SOURCE_MEASUREMENT_SHA256
    assert equation.empirical_anchors[1].provenance.source_path == SOURCE_FRONTIER_MAGNITUDE
    assert equation.empirical_anchors[1].provenance.source_sha256 == SOURCE_FRONTIER_MAGNITUDE_SHA256
    assert equation.domain_of_validity["anchor_measurement_sha256"] == SOURCE_MEASUREMENT_SHA256
    assert equation.domain_of_validity["frontier_magnitude_chart_sha256"] == SOURCE_FRONTIER_MAGNITUDE_SHA256
    assert equation.canonical_producers == (SOURCE_MEASUREMENT, SOURCE_FRONTIER_MAGNITUDE)


@pytest.mark.parametrize("producer", ["measurement", "frontier"])
def test_builder_refuses_byte_identical_canonical_producer_aliases(
    producer: str,
    tmp_path: Path,
) -> None:
    measurement_path: str | Path = SOURCE_MEASUREMENT
    frontier_path: str | Path = SOURCE_FRONTIER_MAGNITUDE
    if producer == "measurement":
        alias = tmp_path / "measurement-alias.json"
        alias.write_bytes(Path(SOURCE_MEASUREMENT).read_bytes())
        measurement_path = alias
    else:
        alias = tmp_path / "frontier-alias.json"
        alias.write_bytes(Path(SOURCE_FRONTIER_MAGNITUDE).read_bytes())
        frontier_path = alias

    with pytest.raises(ValueError, match="canonical producer path must resolve to"):
        build_einstein_kolmogorov_crux_action_rate_contract_v1(
            measurement_path=measurement_path,
            frontier_chart_path=frontier_path,
        )


def test_frontier_chart_refuses_authority_semantic_relabeling_when_ssd_sources_are_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _unmount_absolute_sources(monkeypatch, tmp_path)
    payload = json.loads(Path(SOURCE_FRONTIER_MAGNITUDE).read_text())
    c1 = next(
        row for row in payload["exact_archive_rows"] if row["point_id"] == "c1_solved_distortion_n600_contest_cpu"
    )
    rung_e = next(
        row for row in payload["exact_archive_rows"] if row["point_id"] == "v10_rung_e_exact_two_plane_n48_local"
    )
    c1["measurement_axis"] = "[contest-CUDA] (relabelled while SSD absent)"
    c1["verdict"] = "PROMOTION_AUTHORIZED"
    rung_e["measurement_axis"] = "[contest-CUDA] PROMOTION-GRADE"
    rung_e["verdict"] = "FRONTIER_PROMOTION_AUTHORIZED"
    payload["selection"]["exact_receiver_frontier_magnitude_control"] = "invented_unreceipted_row"
    chart_path = tmp_path / "authority-relabelled-chart.json"
    chart_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="immutable authority hash drifted"):
        validate_frontier_magnitude_chart(chart_path)


def test_measurement_refuses_unvalidated_authority_metadata_mutation(tmp_path: Path) -> None:
    payload = json.loads(Path(SOURCE_MEASUREMENT).read_text())
    payload["axis"] = "[contest-CUDA] PROMOTION-GRADE"
    measurement_path = tmp_path / "authority-relabelled-measurement.json"
    measurement_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="immutable authority hash drifted"):
        crux_equation._load_scoped_measurement(measurement_path)
