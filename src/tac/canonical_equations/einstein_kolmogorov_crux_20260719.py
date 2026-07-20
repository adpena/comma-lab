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


def maximum_byte_budget(*, target_action: float | int, d_seg: float | int, d_pose: float | int) -> int:
    """Return the greatest integral byte count satisfying the target action.

    Raises rather than returning a misleading negative budget when the measured
    non-rate terms already exceed ``target_action``.
    """
    target = _nonnegative_real(target_action, "target_action")
    measured_seg = _nonnegative_real(d_seg, "d_seg")
    measured_pose = _nonnegative_real(d_pose, "d_pose")
    non_rate = SEGMENTATION_WEIGHT * measured_seg + math.sqrt(POSE_RADICAND_WEIGHT * measured_pose)
    slack = target - non_rate
    if slack < 0.0:
        raise InfeasibleByteBudgetError("target action is infeasible before the rate term")
    return math.floor(slack * RATE_DENOMINATOR_BYTES / RATE_WEIGHT)


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
    return payload, source, winner


def build_einstein_kolmogorov_crux_action_rate_contract_v1(
    *, measurement_path: str | Path = SOURCE_MEASUREMENT
) -> CanonicalEquation:
    """Build the hash-bound, research-only canonical equation and n24 anchor."""

    payload, source, winner = _load_scoped_measurement(measurement_path)
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
            "full_archive_or_contest_score_claim": False,
        },
        empirical_output={
            "source_hard_mismatch_px": source["hard_mismatch_px"],
            "source_d_seg": source["d_seg"],
            "winner_hard_mismatch_px": winner["hard_mismatch_px"],
            "winner_d_seg": winner["d_seg"],
            "winner_candidate_bytes": winner["candidate_bytes"],
            "winner_candidate_sha256": winner["candidate_sha256"],
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
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Einstein--Kolmogorov scoped action/rate contract",
        one_line_summary=(
            "Exact contest action arithmetic with a hash-bound research-only n24 "
            "fixed-label palette anchor and explicit non-promotion scope."
        ),
        latex_form=(
            r"S=100D_{seg}+\sqrt{10D_{pose}}+25B/37545489,\quad "
            r"B_{max}=\left\lfloor(S_t-100D_{seg}-\sqrt{10D_{pose}})37545489/25\right\rfloor"
        ),
        python_callable_module_path=("tac.canonical_equations.einstein_kolmogorov_crux_20260719:contest_action"),
        domain_of_validity={
            "action_contract": "frozen comma contest score arithmetic",
            "empirical_anchor_scope": payload["verdict_scope"],
            "anchor_measurement_sha256": provenance.source_sha256,
            "research_only": True,
            "promotion_eligible": False,
            "full_archive_claim": False,
        },
        units_in={
            "d_seg": "fraction",
            "d_pose": "mean squared error",
            "archive_bytes": "bytes",
        },
        units_out={"action": "score units", "maximum_byte_budget": "bytes"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"fixed_byte_palette_no_regression": 0.0},
        last_calibration_utc=measured_utc,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.probe_einstein_kolmogorov_crux",
            "tac.optimization.einstein_kolmogorov_crux",
        ),
        canonical_producers=(measurement_path_str,),
        provenance=provenance,
    )


def populate_einstein_kolmogorov_crux_action_rate_contract_v1(
    *,
    measurement_path: str | Path = SOURCE_MEASUREMENT,
    path=None,
    lock_path=None,
    agent=None,
    subagent_id=None,
) -> CanonicalEquation:
    """Append the scoped law through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_einstein_kolmogorov_crux_action_rate_contract_v1(measurement_path=measurement_path)
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="n24 fixed-label palette probe; research-only and non-promotable",
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "SOURCE_MEASUREMENT",
    "InfeasibleByteBudgetError",
    "MeasuredHardRReceipt",
    "ResearchOnlyDecision",
    "build_einstein_kolmogorov_crux_action_rate_contract_v1",
    "contest_action",
    "derive_research_only_decision",
    "fixed_byte_palette_delta",
    "maximum_byte_budget",
    "populate_einstein_kolmogorov_crux_action_rate_contract_v1",
]
