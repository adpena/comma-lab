# SPDX-License-Identifier: MIT
"""Canonical fail-closed law for PF2 measurement-assignment eligibility."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "ddm_pf2_bucket_assignment_join_eligibility_v1"
REPO = Path(__file__).resolve().parents[3]
RECEIPT = REPO / (
    ".omx/research/ddm_ms5_pf2_bucket_assignment_20260724T044736Z/ddm_ms5_pf2_bucket_assignment_receipt.json"
)


def pf2_bucket_assignment_join_eligible(
    *,
    pair_membership_set_equal: bool,
    receiver_actuator_ids: Sequence[str],
    direction_ids: Sequence[str],
) -> bool:
    """Return whether one PF2 row identifies a real scorer measurement input."""

    if not isinstance(pair_membership_set_equal, bool):
        raise ValueError("pair membership equality must be an exact boolean")
    if (
        isinstance(receiver_actuator_ids, (str, bytes))
        or isinstance(direction_ids, (str, bytes))
        or not isinstance(receiver_actuator_ids, Sequence)
        or not isinstance(direction_ids, Sequence)
    ):
        raise ValueError("actuator and direction IDs must be sequences")
    if any(not isinstance(value, str) or not value for value in receiver_actuator_ids):
        raise ValueError("receiver actuator IDs must be nonempty strings")
    if any(not isinstance(value, str) or not value for value in direction_ids):
        raise ValueError("direction IDs must be nonempty strings")
    return bool(
        pair_membership_set_equal
        and receiver_actuator_ids
        and direction_ids
        and len(set(receiver_actuator_ids)) == len(receiver_actuator_ids)
        and len(set(direction_ids)) == len(direction_ids)
    )


def build_ddm_pf2_bucket_assignment_join_eligibility_v1(
    *,
    source_receipt: Path = RECEIPT,
) -> CanonicalEquation:
    """Build the apparatus law from the MS5 SHA-bound receipt."""

    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "Produce a measured causal join from each PF2 key to receiver-effective "
            "J2/G2G actuator IDs and G2F-compatible signed directions."
        ),
        measurement_axis="[macOS-CPU frozen-scorer advisory]",
        hardware_substrate="darwin_arm64_cpu_numpy",
        captured_at_utc="2026-07-24T05:04:00Z",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="DDM PF2 bucket assignment foreign-key eligibility",
        one_line_summary=(
            "A PF2 bucket can drive scorer measurement only when raw-event "
            "membership and receiver actuator/direction foreign keys all close."
        ),
        latex_form=(
            r"A_b=\mathbf{1}[E_b=E_b\vert_{P_b}]"
            r"\mathbf{1}[\lvert U_b\rvert>0]"
            r"\mathbf{1}[\lvert D_b\rvert>0]"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_pf2_bucket_assignment_20260724:pf2_bucket_assignment_join_eligible"
        ),
        domain_of_validity={
            "vehicle": "current b8c81edec2 PF2 construction lineage",
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
            "current_membership_closure": "1200/1200",
            "current_foreign_key_closure": "0/1200",
            "verdict_scope": "INSTANCE_CURRENT_PF2_CONSTRUCTION_LINEAGE",
            "scientific_validator": ("tac.optimization.ddm_pf2_bucket_assignment:validate_assignment_table"),
        },
        units_in={
            "pair_membership_set_equal": "boolean",
            "receiver_actuator_ids": "stable_id_sequence",
            "direction_ids": "signed_direction_id_sequence",
        },
        units_out={"measurement_assignment_eligible": "boolean"},
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-07-24T05:04:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.produce_ddm_ms4_metric_custody",
            "tac.optimization.ddm_metric_producers:audit_pf2_bucket_assignments",
        ),
        canonical_producers=("tools.assign_ddm_ms5_pf2_buckets",),
        provenance=provenance,
    )


def populate_ddm_pf2_bucket_assignment_join_eligibility(
    *,
    source_receipt: Path = RECEIPT,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Register the MS5 law through the append-only registry callable."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_pf2_bucket_assignment_join_eligibility_v1(source_receipt=source_receipt)
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "MS5 membership 1200/1200; actuator/direction join 0/1200; "
            "producer rerun held; score_claim=false; MAIN review required"
        ),
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "build_ddm_pf2_bucket_assignment_join_eligibility_v1",
    "pf2_bucket_assignment_join_eligible",
    "populate_ddm_pf2_bucket_assignment_join_eligibility",
]
