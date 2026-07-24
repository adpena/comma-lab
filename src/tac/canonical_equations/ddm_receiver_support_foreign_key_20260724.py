# SPDX-License-Identifier: MIT
"""Canonical causal law for the measured PF2 receiver foreign key."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "ddm_receiver_support_pf2_causal_intersection_v1"
REPO = Path(__file__).resolve().parents[3]
RECEIPT = REPO / (
    ".omx/research/ddm_ms6_receiver_support_measurement_20260724T052034Z/"
    "ddm_ms6_receiver_support_measurement_receipt.json"
)


def receiver_support_pf2_causal_intersection(
    *,
    pf2_event_ids: Sequence[int],
    changed_argmax_event_ids: Sequence[int],
) -> tuple[int, ...]:
    """Return the exact raw PF2 events causally changed by one receiver probe."""

    if isinstance(pf2_event_ids, (str, bytes)) or isinstance(
        changed_argmax_event_ids,
        (str, bytes),
    ):
        raise ValueError("event IDs must be integer sequences")

    def normalized(values: Sequence[int], label: str) -> tuple[int, ...]:
        if not isinstance(values, Sequence) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError(f"{label} event IDs must be nonnegative integers")
        result = tuple(values)
        if result != tuple(sorted(set(result))):
            raise ValueError(f"{label} event IDs must be sorted and unique")
        return result

    pf2 = normalized(pf2_event_ids, "PF2")
    changed = normalized(changed_argmax_event_ids, "changed argmax")
    return tuple(sorted(set(pf2).intersection(changed)))


def build_ddm_receiver_support_pf2_causal_intersection_v1(
    *,
    source_receipt: Path = RECEIPT,
) -> CanonicalEquation:
    """Build the receiver/PF2 causal-intersection law from measured custody."""

    provenance = build_provenance_for_research_sidecar(
        source_receipt,
        reactivation_criteria=(
            "Resume the per-actuator sweep from preserved SSD checkpoints; "
            "rebuild only after any bound input SHA changes."
        ),
        measurement_axis="[macOS-CPU frozen-scorer advisory]",
        hardware_substrate="darwin_arm64_cpu_torch_batch32",
        captured_at_utc="2026-07-24T06:25:00Z",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="DDM receiver-support PF2 causal intersection",
        one_line_summary=(
            "A signed actuator owns a PF2 event only when its receiver-realized "
            "SegNet argmax changes at that exact raw-event address."
        ),
        latex_form=(
            r"J_{b,u,s}=\{e\in E_b:"
            r"\operatorname{argmax}F(R(G(\theta+s q_u)))_e"
            r"\ne\operatorname{argmax}F(R(G(\theta)))_e\}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_receiver_support_foreign_key_20260724:"
            "receiver_support_pf2_causal_intersection"
        ),
        domain_of_validity={
            "vehicle": "SHA-bound V19C endpoint one-quantum receiver sweep",
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
            "verdict_scope": "INSTANCE_V19C_ENDPOINT_ONE_QUANTUM_SWEEP",
            "empty_intersection_semantics": "MEASURED_EMPTY_NOT_OMITTED",
            "infeasible_quantum_semantics": "EXPLICIT_BLOCKER_NOT_ZERO",
            "scientific_validator": (
                "tac.optimization.ddm_pf2_bucket_assignment:"
                "build_measured_assignment_table"
            ),
        },
        units_in={
            "pf2_event_ids": "global_row_major_event_id_sequence",
            "changed_argmax_event_ids": "global_row_major_event_id_sequence",
        },
        units_out={"causal_intersection": "global_row_major_event_id_sequence"},
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-07-24T06:25:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.measure_ddm_ms6_receiver_support",
            "tools.produce_ddm_ms4_metric_custody",
        ),
        canonical_producers=("tools.measure_ddm_ms6_receiver_support",),
        provenance=provenance,
    )


def populate_ddm_receiver_support_pf2_causal_intersection(
    *,
    source_receipt: Path = RECEIPT,
    path: Path | None = None,
    lock_path: Path | None = None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Register the MS6 law through the append-only registry callable."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_receiver_support_pf2_causal_intersection_v1(
        source_receipt=source_receipt,
    )
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "Receiver/composite-R/SegNet causal intersection with v2 "
            "checkpoint-local scorer custody; partial coverage remains "
            "non-promoting; score_claim=false"
        ),
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "build_ddm_receiver_support_pf2_causal_intersection_v1",
    "populate_ddm_receiver_support_pf2_causal_intersection",
    "receiver_support_pf2_causal_intersection",
]
