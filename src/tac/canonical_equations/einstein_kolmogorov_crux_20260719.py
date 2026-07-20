# SPDX-License-Identifier: MIT
"""Canonical score/rate law for the scoped Einstein--Kolmogorov n24 probe.

The pure arithmetic remains usable without filesystem state.  The explicit
``build_*``/``populate_*`` surface is the triality equation leg: it reads the
frozen aggregate measurement, binds its exact bytes through canonical
``Provenance``, and registers a research-only empirical anchor.  No score or
frontier authority is created by registration.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "einstein_kolmogorov_crux_action_rate_contract_v1"
RESEARCH_ONLY_AXIS = "[macOS-CPU advisory]"
RATE_DENOMINATOR_BYTES = 37_545_489
RATE_WEIGHT = 25
SEGMENTATION_WEIGHT = 100.0
POSE_RADICAND_WEIGHT = 10.0
SOURCE_MEASUREMENT = ".omx/research/einstein_kolmogorov_crux_measurement_20260719.json"
SOURCE_FRONTIER_MAGNITUDE = ".omx/research/einstein_kolmogorov_frontier_magnitude_chart_20260720.json"


class InfeasibleByteBudgetError(ValueError):
    """The requested target is already exceeded before any counted bytes."""


def _nonnegative_real(value: float | int, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _nonnegative_bytes(value: int, field: str = "archive_bytes") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def contest_action(*, d_seg: float | int, d_pose: float | int, archive_bytes: int) -> float:
    """Evaluate ``100*d_seg + sqrt(10*d_pose) + 25*bytes/37_545_489`` exactly in form."""
    measured_seg = _nonnegative_real(d_seg, "d_seg")
    measured_pose = _nonnegative_real(d_pose, "d_pose")
    bytes_used = _nonnegative_bytes(archive_bytes)
    return (
        SEGMENTATION_WEIGHT * measured_seg
        + math.sqrt(POSE_RADICAND_WEIGHT * measured_pose)
        + RATE_WEIGHT * bytes_used / RATE_DENOMINATOR_BYTES
    )


def frontier_feasible_at_zero_pose_and_rate(*, d_seg: float | int, target_action: float | int) -> bool:
    """Necessary frontier gate using the Seg term alone.

    ``False`` is a hard impossibility result: non-negative Pose and rate terms
    cannot rescue the row. ``True`` is only necessary, never sufficient.
    """

    measured_seg = _nonnegative_real(d_seg, "d_seg")
    target = _nonnegative_real(target_action, "target_action")
    return SEGMENTATION_WEIGHT * measured_seg < target


def _project_integral_population_bytes(*, mean_bytes_per_pair: float | int, pair_count: int) -> int:
    """Project a measured mean byte count only when the product is integral."""

    mean_bytes = _nonnegative_real(mean_bytes_per_pair, "mean_bytes_per_pair")
    population = _nonnegative_bytes(pair_count, "pair_count")
    if population == 0:
        raise ValueError("pair_count must be positive")
    projected = mean_bytes * population
    rounded = round(projected)
    if not math.isclose(projected, rounded, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("mean byte projection is not an integral population byte count")
    return int(rounded)


def inclusive_maximum_byte_budget(*, target_action: float | int, d_seg: float | int, d_pose: float | int) -> int:
    """Return the greatest byte count whose action is at most ``target_action``."""

    target = _nonnegative_real(target_action, "target_action")
    measured_seg = _nonnegative_real(d_seg, "d_seg")
    measured_pose = _nonnegative_real(d_pose, "d_pose")
    non_rate = SEGMENTATION_WEIGHT * measured_seg + math.sqrt(POSE_RADICAND_WEIGHT * measured_pose)
    slack = target - non_rate
    if slack < 0.0:
        raise InfeasibleByteBudgetError("target action is infeasible before the rate term")
    candidate = math.floor(slack * RATE_DENOMINATOR_BYTES / RATE_WEIGHT)
    while candidate >= 0 and contest_action(d_seg=measured_seg, d_pose=measured_pose, archive_bytes=candidate) > target:
        candidate -= 1
    while contest_action(d_seg=measured_seg, d_pose=measured_pose, archive_bytes=candidate + 1) <= target:
        candidate += 1
    return candidate


def maximum_byte_budget(*, target_action: float | int, d_seg: float | int, d_pose: float | int) -> int:
    """Return the greatest integral byte count that *strictly beats* a target.

    The strict integer ceiling is ``ceil(x) - 1``, not ``floor(x)`` at an
    equality boundary. The final comparisons use :func:`contest_action` so
    binary floating-point roundoff cannot silently admit an equal-score byte.
    """

    target = _nonnegative_real(target_action, "target_action")
    measured_seg = _nonnegative_real(d_seg, "d_seg")
    measured_pose = _nonnegative_real(d_pose, "d_pose")
    non_rate = SEGMENTATION_WEIGHT * measured_seg + math.sqrt(POSE_RADICAND_WEIGHT * measured_pose)
    slack = target - non_rate
    if slack <= 0.0:
        raise InfeasibleByteBudgetError("target action cannot be strictly beaten before the rate term")
    candidate = math.ceil(slack * RATE_DENOMINATOR_BYTES / RATE_WEIGHT) - 1
    while (
        candidate >= 0 and contest_action(d_seg=measured_seg, d_pose=measured_pose, archive_bytes=candidate) >= target
    ):
        candidate -= 1
    while contest_action(d_seg=measured_seg, d_pose=measured_pose, archive_bytes=candidate + 1) < target:
        candidate += 1
    if candidate < 0:
        raise InfeasibleByteBudgetError("target action cannot be strictly beaten with any non-negative byte count")
    return candidate


def fixed_byte_palette_delta(
    *,
    before_d_seg: float | int,
    before_d_pose: float | int,
    after_d_seg: float | int,
    after_d_pose: float | int,
    before_bytes: int,
    after_bytes: int,
) -> float:
    """Return the action delta for a zero-rate palette substitution.

    Refuses unequal byte counts so a rate change cannot be accidentally labelled
    as palette-only actuation.
    """
    before = _nonnegative_bytes(before_bytes, "before_bytes")
    after = _nonnegative_bytes(after_bytes, "after_bytes")
    if before != after:
        raise ValueError("fixed-byte palette actuation requires identical packet bytes")
    return contest_action(d_seg=after_d_seg, d_pose=after_d_pose, archive_bytes=after) - contest_action(
        d_seg=before_d_seg, d_pose=before_d_pose, archive_bytes=before
    )


@dataclass(frozen=True)
class MeasuredHardRReceipt:
    """A caller-supplied, scope-checked hard-R measurement row."""

    receipt_id: str
    verdict_scope: str
    d_seg: float
    d_pose: float
    archive_bytes: int
    axis: Literal["[macOS-CPU advisory]"] = RESEARCH_ONLY_AXIS
    research_only: Literal[True] = True

    def __post_init__(self) -> None:
        if not self.receipt_id.strip() or not self.verdict_scope.strip():
            raise ValueError("receipt_id and verdict_scope must be non-empty")
        if self.axis != RESEARCH_ONLY_AXIS or self.research_only is not True:
            raise ValueError("this equation accepts only explicit research-only macOS advisory receipts")
        _nonnegative_real(self.d_seg, "d_seg")
        _nonnegative_real(self.d_pose, "d_pose")
        _nonnegative_bytes(self.archive_bytes)


@dataclass(frozen=True)
class DerivationEdge:
    source: str
    target: str
    relation: Literal["MEASURED_HARD_R_INPUT", "DERIVES", "SCOPES"]


@dataclass(frozen=True)
class ResearchOnlyDecision:
    """Non-promotable action/budget decision derived from one supplied receipt."""

    equation_id: str
    axis: Literal["[macOS-CPU advisory]"]
    verdict_scope: str
    measured_action: float
    maximum_bytes_at_target: int
    research_only: Literal[True]
    promotion_eligible: Literal[False]
    derivation_edges: tuple[DerivationEdge, ...]


def derive_research_only_decision(*, receipt: MeasuredHardRReceipt, target_action: float | int) -> ResearchOnlyDecision:
    """Compose ``measured hard-R receipt -> equation -> research-only decision``."""
    target = _nonnegative_real(target_action, "target_action")
    budget = maximum_byte_budget(target_action=target, d_seg=receipt.d_seg, d_pose=receipt.d_pose)
    decision_id = f"research_only_decision:{receipt.receipt_id}"
    return ResearchOnlyDecision(
        equation_id=EQUATION_ID,
        axis=receipt.axis,
        verdict_scope=receipt.verdict_scope,
        measured_action=contest_action(d_seg=receipt.d_seg, d_pose=receipt.d_pose, archive_bytes=receipt.archive_bytes),
        maximum_bytes_at_target=budget,
        research_only=True,
        promotion_eligible=False,
        derivation_edges=(
            DerivationEdge(receipt.receipt_id, EQUATION_ID, "MEASURED_HARD_R_INPUT"),
            DerivationEdge(EQUATION_ID, decision_id, "DERIVES"),
            DerivationEdge(receipt.verdict_scope, decision_id, "SCOPES"),
        ),
    )


def _load_scoped_measurement(path: str | Path) -> tuple[dict, dict, dict]:
    """Load the immutable n24 aggregate and return payload/source/winner rows."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "einstein_kolmogorov_crux_measurement.v2":
        raise ValueError("unexpected Einstein--Kolmogorov measurement schema")
    if payload.get("research_only") is not True or payload.get("score_claim") is not False:
        raise ValueError("canonical anchor requires an explicit research-only non-score receipt")
    scope = str(payload.get("verdict_scope", ""))
    if "n24" not in scope or "no contest-axis score" not in scope:
        raise ValueError("measurement verdict scope must retain n24 and non-contest boundaries")
    rows = payload.get("tournament")
    if not isinstance(rows, list):
        raise ValueError("measurement tournament must be a list")
    by_arm = {row.get("arm"): row for row in rows if isinstance(row, dict)}
    source = by_arm.get("source_per_pair_means")
    winner = by_arm.get("dspsa32_then_coordinate12")
    if not isinstance(source, dict) or not isinstance(winner, dict):
        raise ValueError("measurement must contain source and scoped winner rows")
    for row_name, row in (("source", source), ("winner", winner)):
        if not isinstance(row.get("hard_mismatch_px"), int):
            raise ValueError(f"{row_name} row lacks integral hard mismatch custody")
        if not isinstance(row.get("candidate_bytes"), int):
            raise ValueError(f"{row_name} row lacks integral byte custody")
        _nonnegative_real(row.get("d_seg"), f"{row_name}.d_seg")
        _nonnegative_bytes(row["candidate_bytes"], f"{row_name}.candidate_bytes")
    if winner["candidate_bytes"] != source["candidate_bytes"]:
        raise ValueError("fixed-label palette anchor requires identical packet bytes")
    if winner["hard_mismatch_px"] >= source["hard_mismatch_px"]:
        raise ValueError("measured scoped winner must strictly improve the in-run source control")
    correction = payload.get("operating_point_correction")
    if not isinstance(correction, dict):
        raise ValueError("measurement lacks the operating-point correction")
    if correction.get("verdict") != "WRONG_OPERATING_POINT_WALL_CHARACTERIZATION":
        raise ValueError("measurement operating-point verdict is not fail-closed")
    target_action = _nonnegative_real(correction.get("target_action"), "operating_point.target_action")
    measured_feasible = frontier_feasible_at_zero_pose_and_rate(
        d_seg=winner["d_seg"],
        target_action=target_action,
    )
    if measured_feasible or correction.get("frontier_feasible_even_at_zero_pose_zero_bytes") is not False:
        raise ValueError("measurement operating point must fail the Seg-only frontier necessity gate")
    if correction.get("n600_explicit_target_launch_eligible") is not False:
        raise ValueError("infeasible explicit-target operating point must not authorize n600 scaling")
    return payload, source, winner


def validate_frontier_magnitude_chart(
    path: str | Path = SOURCE_FRONTIER_MAGNITUDE,
) -> dict:
    """Validate the exact-solver frontier chart and every score-law projection.

    The chart intentionally mixes separately labelled evidence axes. Validation
    checks arithmetic and scope; it does not promote local rows to contest
    authority or turn linear byte projections into archives.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "einstein_kolmogorov_frontier_magnitude_chart.v1":
        raise ValueError("unexpected frontier-magnitude chart schema")
    if (
        payload.get("research_only") is not True
        or payload.get("score_claim") is not False
        or payload.get("promotion_eligible") is not False
    ):
        raise ValueError("frontier-magnitude chart must remain research-only and non-promotable")
    pointer = payload.get("pointer")
    if not isinstance(pointer, dict) or pointer.get("moved") is not False:
        raise ValueError("frontier-magnitude chart must preserve the pointer")
    target = _nonnegative_real(pointer.get("score"), "pointer.score")

    archive_rows = payload.get("exact_archive_rows")
    if not isinstance(archive_rows, list):
        raise ValueError("frontier-magnitude chart lacks exact archive rows")
    by_id = {row.get("point_id"): row for row in archive_rows if isinstance(row, dict)}
    bank = by_id.get("c1_solved_distortion_n600_contest_cpu")
    rung_e = by_id.get("v10_rung_e_exact_two_plane_n48_local")
    banked_control = by_id.get("banked_n12_exact_receiver_control")
    banked_treatment = by_id.get("banked_n12_scorer_plane_precision_drop1")
    if not all(isinstance(row, dict) for row in (bank, rung_e, banked_control, banked_treatment)):
        raise ValueError("frontier-magnitude chart lacks an exact receiver control/treatment row")
    bank_action = contest_action(
        d_seg=bank.get("d_seg"),
        d_pose=bank.get("d_pose"),
        archive_bytes=bank.get("archive_bytes"),
    )
    if not math.isclose(bank_action, bank.get("exact_action"), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("bank exact-action arithmetic drift")
    if maximum_byte_budget(target_action=target, d_seg=bank["d_seg"], d_pose=bank["d_pose"]) != bank.get(
        "strict_bytes_to_beat_pointer_at_measured_distortion"
    ):
        raise ValueError("bank strict byte cap drift")
    rung_projected_bytes = _nonnegative_bytes(
        rung_e.get("derived_n600_linear_archive_bytes"), "rung_e.derived_n600_linear_archive_bytes"
    )
    rung_projected_action = contest_action(
        d_seg=rung_e.get("d_seg"),
        d_pose=rung_e.get("d_pose"),
        archive_bytes=rung_projected_bytes,
    )
    if not math.isclose(
        rung_projected_action,
        rung_e.get("derived_n600_action_at_unchanged_mean_distortion"),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("rung-E projected action arithmetic drift")
    if maximum_byte_budget(target_action=target, d_seg=rung_e["d_seg"], d_pose=rung_e["d_pose"]) != rung_e.get(
        "strict_bytes_to_beat_pointer_at_measured_distortion"
    ):
        raise ValueError("rung-E strict byte cap drift")
    for row, frontier_relevant in ((banked_control, True), (banked_treatment, False)):
        pair_count = _nonnegative_bytes(row.get("pair_count"), f"{row.get('point_id')}.pair_count")
        if pair_count == 0:
            raise ValueError("banked receiver row pair count must be positive")
        archive_bytes = _nonnegative_bytes(row.get("archive_bytes"), f"{row.get('point_id')}.archive_bytes")
        projected_bytes = (archive_bytes * 600 + pair_count - 1) // pair_count
        if projected_bytes != row.get("derived_n600_linear_archive_bytes"):
            raise ValueError(f"{row.get('point_id')} projected archive-byte arithmetic drift")
        action = contest_action(
            d_seg=row.get("d_seg"),
            d_pose=row.get("d_pose"),
            archive_bytes=projected_bytes,
        )
        if not math.isclose(
            action,
            row.get("derived_n600_action_at_unchanged_mean_distortion"),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{row.get('point_id')} projected action arithmetic drift")
        if row.get("frontier_relevant_distortion") is not frontier_relevant:
            raise ValueError(f"{row.get('point_id')} frontier-distortion gate drift")
        if frontier_relevant:
            cap = maximum_byte_budget(target_action=target, d_seg=row["d_seg"], d_pose=row["d_pose"])
            if cap != row.get("strict_bytes_to_beat_pointer_at_measured_distortion"):
                raise ValueError(f"{row.get('point_id')} strict byte cap drift")
        elif (
            row.get("strict_bytes_to_beat_pointer_at_measured_distortion") != -1
            or contest_action(d_seg=row["d_seg"], d_pose=row["d_pose"], archive_bytes=0) < target
        ):
            raise ValueError(f"{row.get('point_id')} infeasible distortion classification drift")
    treatment_delta = banked_treatment.get("treatment_minus_control")
    if not isinstance(treatment_delta, dict) or (
        treatment_delta.get("archive_bytes") != banked_treatment["archive_bytes"] - banked_control["archive_bytes"]
        or not math.isclose(
            treatment_delta.get("d_seg"),
            banked_treatment["d_seg"] - banked_control["d_seg"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or not math.isclose(
            treatment_delta.get("d_pose"),
            banked_treatment["d_pose"] - banked_control["d_pose"],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("banked receiver matched A/B delta drift")

    trade = payload.get("trade_cells_curve")
    points = trade.get("points") if isinstance(trade, dict) else None
    if not isinstance(points, list) or len(points) != 10:
        raise ValueError("frontier-magnitude chart requires all ten settled trade-cells points")
    trade_by_id = {point.get("point_id"): point for point in points if isinstance(point, dict)}
    for point in points:
        if not isinstance(point, dict):
            raise ValueError("trade-cells points must be mappings")
        projected_bytes = _project_integral_population_bytes(
            mean_bytes_per_pair=point.get("measured_mean_payload_bytes_per_pair"),
            pair_count=600,
        )
        if projected_bytes != point.get("derived_n600_payload_bytes"):
            raise ValueError(f"{point.get('point_id')} projected byte arithmetic drift")
        projected_action = contest_action(
            d_seg=point.get("d_seg"),
            d_pose=point.get("d_pose"),
            archive_bytes=projected_bytes,
        )
        if not math.isclose(
            projected_action,
            point.get("derived_action_on_declared_payload_scope"),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{point.get('point_id')} projected action arithmetic drift")
        feasible = frontier_feasible_at_zero_pose_and_rate(d_seg=point.get("d_seg"), target_action=target)
        if feasible is not point.get("seg_only_frontier_necessary_gate"):
            raise ValueError(f"{point.get('point_id')} Seg-only gate drift")
    lead = trade_by_id.get("precision_drop1")
    if not isinstance(lead, dict) or lead.get("pose_violation_count") != 0:
        raise ValueError("frontier-magnitude trade-cells lead must retain the pose-clean control")

    gap = payload.get("exact_production_gap")
    if (
        not isinstance(gap, dict)
        or gap.get("blocker") != "NO_COMPACT_PREDICTOR_DESCRIPTION_IN_216_TO_244_KB_BOX"
        or gap.get("secondary_blocker") != "MISSING_ARBITRARY_NUMERATOR_PLANE_CODEC"
    ):
        raise ValueError("frontier-magnitude chart must fail closed on compactness and the remaining ABI gap")
    if payload.get("n600_trade_cells_launch_eligible") is not False:
        raise ValueError("frontier-magnitude chart cannot authorize an n600 trade-cells launch")
    return payload


def build_einstein_kolmogorov_crux_action_rate_contract_v1(
    *,
    measurement_path: str | Path = SOURCE_MEASUREMENT,
    frontier_chart_path: str | Path = SOURCE_FRONTIER_MAGNITUDE,
) -> CanonicalEquation:
    """Build the hash-bound, research-only canonical equation and n24 anchor."""

    payload, source, winner = _load_scoped_measurement(measurement_path)
    frontier_chart = validate_frontier_magnitude_chart(frontier_chart_path)
    correction = payload["operating_point_correction"]
    measurement_path_str = str(measurement_path)
    measured_utc = str(payload["utc"])
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=measurement_path,
        reactivation_criteria=(
            "Research-only n24 fixed-label palette evidence. Reactivate for promotion only "
            "after a complete n600 archive is byte-closed and scored on a contest axis."
        ),
        measurement_axis=RESEARCH_ONLY_AXIS,
        hardware_substrate="macos_arm64",
        captured_at_utc=measured_utc,
    )
    frontier_utc = str(frontier_chart["written_at_utc"])
    frontier_provenance = build_provenance_for_research_sidecar(
        sidecar_path=frontier_chart_path,
        reactivation_criteria=(
            "Cross-axis exact-solver reuse and projected rate controls only. Reactivate "
            "for promotion after a complete n600 trade-cells archive is byte-closed and "
            "scored on a contest axis."
        ),
        measurement_axis="[cross-axis exact receipts; axes remain separate]",
        hardware_substrate="mixed; see frontier chart row custody",
        captured_at_utc=frontier_utc,
    )
    anchor = EmpiricalAnchor(
        anchor_id="einstein_kolmogorov_n24_fixed_label_palette_20260719",
        measurement_utc=measured_utc,
        inputs={
            "scope": payload["verdict_scope"],
            "source_candidate_sha256": source["candidate_sha256"],
            "source_packet_bytes": source["candidate_bytes"],
            "pair_count": payload["inputs"]["pair_count"],
            "scorer_pixels": payload["inputs"]["scorer_pixels"],
        },
        predicted_output={
            "fixed_byte_palette_winner_strictly_improves_source": True,
            "winner_frontier_feasible_at_zero_pose_zero_rate": False,
            "full_archive_or_contest_score_claim": False,
        },
        empirical_output={
            "source_hard_mismatch_px": source["hard_mismatch_px"],
            "source_d_seg": source["d_seg"],
            "winner_hard_mismatch_px": winner["hard_mismatch_px"],
            "winner_d_seg": winner["d_seg"],
            "winner_candidate_bytes": winner["candidate_bytes"],
            "winner_candidate_sha256": winner["candidate_sha256"],
            "winner_seg_term": SEGMENTATION_WEIGHT * winner["d_seg"],
            "target_action": correction["target_action"],
            "winner_frontier_feasible_at_zero_pose_zero_rate": False,
            "operating_point_verdict": correction["verdict"],
            "full_archive_or_contest_score_claim": False,
        },
        residual=0.0,
        source_artifact=measurement_path_str,
        measurement_method=(
            "PDW1 encode/decode/re-encode, factor-2 uint8 realization certificate, "
            "and singleton frozen CPU-Torch SegNet over all 24 packet pairs"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    frontier_rows = {row["point_id"]: row for row in frontier_chart["exact_archive_rows"]}
    trade_rows = {row["point_id"]: row for row in frontier_chart["trade_cells_curve"]["points"]}
    frontier_anchor = EmpiricalAnchor(
        anchor_id="einstein_kolmogorov_exact_solver_frontier_magnitude_20260720",
        measurement_utc=frontier_utc,
        inputs={
            "pointer": frontier_chart["pointer"],
            "exact_solver_receipt_count": len(frontier_chart["source_receipts"]),
            "trade_cells_point_count": len(trade_rows),
            "axes_are_not_promoted_or_inferred_equivalent": True,
        },
        predicted_output={
            "frontier_magnitude_exact_receiver_control_exists": True,
            "matched_receiver_positive_band_control_exists": True,
            "frontier_magnitude_pose_clean_trade_cells_control_exists": True,
            "complete_n600_in_box_archive_exists": False,
            "n600_trade_cells_launch_eligible": False,
        },
        empirical_output={
            "bank": {
                "archive_bytes": frontier_rows["c1_solved_distortion_n600_contest_cpu"]["archive_bytes"],
                "d_seg": frontier_rows["c1_solved_distortion_n600_contest_cpu"]["d_seg"],
                "d_pose": frontier_rows["c1_solved_distortion_n600_contest_cpu"]["d_pose"],
                "exact_action": frontier_rows["c1_solved_distortion_n600_contest_cpu"]["exact_action"],
            },
            "exact_receiver_control": {
                "pair_count": frontier_rows["v10_rung_e_exact_two_plane_n48_local"]["pair_count"],
                "archive_bytes": frontier_rows["v10_rung_e_exact_two_plane_n48_local"]["archive_bytes"],
                "d_seg": frontier_rows["v10_rung_e_exact_two_plane_n48_local"]["d_seg"],
                "d_pose": frontier_rows["v10_rung_e_exact_two_plane_n48_local"]["d_pose"],
            },
            "matched_receiver_control": {
                "pair_count": frontier_rows["banked_n12_exact_receiver_control"]["pair_count"],
                "archive_bytes": frontier_rows["banked_n12_exact_receiver_control"]["archive_bytes"],
                "d_seg": frontier_rows["banked_n12_exact_receiver_control"]["d_seg"],
                "d_pose": frontier_rows["banked_n12_exact_receiver_control"]["d_pose"],
            },
            "matched_receiver_treatment": {
                "point_id": "banked_n12_scorer_plane_precision_drop1",
                "archive_bytes": frontier_rows["banked_n12_scorer_plane_precision_drop1"]["archive_bytes"],
                "d_seg": frontier_rows["banked_n12_scorer_plane_precision_drop1"]["d_seg"],
                "d_pose": frontier_rows["banked_n12_scorer_plane_precision_drop1"]["d_pose"],
                "derived_n600_action": frontier_rows["banked_n12_scorer_plane_precision_drop1"][
                    "derived_n600_action_at_unchanged_mean_distortion"
                ],
            },
            "pose_clean_trade_cells_control": {
                "point_id": "precision_drop1",
                "d_seg": trade_rows["precision_drop1"]["d_seg"],
                "d_pose": trade_rows["precision_drop1"]["d_pose"],
                "derived_n600_payload_bytes": trade_rows["precision_drop1"]["derived_n600_payload_bytes"],
                "derived_action_on_declared_payload_scope": trade_rows["precision_drop1"][
                    "derived_action_on_declared_payload_scope"
                ],
            },
            "exact_production_blocker": frontier_chart["exact_production_gap"]["blocker"],
            "pointer_moved": False,
        },
        residual=0.0,
        source_artifact=str(frontier_chart_path),
        measurement_method=(
            "Read-only SHA revalidation, matched n12 production receiver A/B, and exact "
            "score-law composition of the banked C1, Rung-E, zero-band joint-solve, "
            "exact-lattice, and trade-cells receipts"
        ),
        provenance=frontier_provenance,
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Einstein--Kolmogorov scoped action/rate contract",
        one_line_summary=(
            "Exact contest action arithmetic with a hash-bound research-only n24 "
            "fixed-label wall plus matched exact-receiver frontier-magnitude anchor."
        ),
        latex_form=(
            r"S=100D_{seg}+\sqrt{10D_{pose}}+25B/37545489,\quad "
            r"B^{<}_{max}=\left\lceil(S_t-100D_{seg}-\sqrt{10D_{pose}})37545489/25\right\rceil-1"
        ),
        python_callable_module_path=("tac.canonical_equations.einstein_kolmogorov_crux_20260719:contest_action"),
        domain_of_validity={
            "action_contract": "frozen comma contest score arithmetic",
            "empirical_anchor_scope": payload["verdict_scope"],
            "anchor_measurement_sha256": provenance.source_sha256,
            "research_only": True,
            "promotion_eligible": False,
            "full_archive_claim": False,
            "operating_point_verdict": correction["verdict"],
            "n600_explicit_target_launch_eligible": False,
            "frontier_magnitude_chart_sha256": frontier_provenance.source_sha256,
            "frontier_magnitude_exact_production_blocker": frontier_chart["exact_production_gap"]["blocker"],
            "n600_trade_cells_launch_eligible": False,
        },
        units_in={
            "d_seg": "fraction",
            "d_pose": "mean squared error",
            "archive_bytes": "bytes",
        },
        units_out={"action": "score units", "maximum_byte_budget": "bytes"},
        empirical_anchors=(anchor, frontier_anchor),
        predicted_vs_empirical_residual={
            "fixed_byte_palette_no_regression": 0.0,
            "frontier_magnitude_chart_arithmetic": 0.0,
        },
        last_calibration_utc=frontier_utc,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.probe_einstein_kolmogorov_crux",
            "tac.optimization.einstein_kolmogorov_crux",
        ),
        canonical_producers=(measurement_path_str, str(frontier_chart_path)),
        provenance=provenance,
    )


def populate_einstein_kolmogorov_crux_action_rate_contract_v1(
    *,
    measurement_path: str | Path = SOURCE_MEASUREMENT,
    frontier_chart_path: str | Path = SOURCE_FRONTIER_MAGNITUDE,
    path=None,
    lock_path=None,
    agent=None,
    subagent_id=None,
) -> CanonicalEquation:
    """Append the scoped law through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_einstein_kolmogorov_crux_action_rate_contract_v1(
        measurement_path=measurement_path,
        frontier_chart_path=frontier_chart_path,
    )
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="n24 fixed-label wall plus exact-receiver frontier-magnitude A/B; research-only and non-promotable",
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "SOURCE_FRONTIER_MAGNITUDE",
    "SOURCE_MEASUREMENT",
    "InfeasibleByteBudgetError",
    "MeasuredHardRReceipt",
    "ResearchOnlyDecision",
    "build_einstein_kolmogorov_crux_action_rate_contract_v1",
    "contest_action",
    "derive_research_only_decision",
    "fixed_byte_palette_delta",
    "frontier_feasible_at_zero_pose_and_rate",
    "inclusive_maximum_byte_budget",
    "maximum_byte_budget",
    "populate_einstein_kolmogorov_crux_action_rate_contract_v1",
    "validate_frontier_magnitude_chart",
]
