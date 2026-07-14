# SPDX-License-Identifier: MIT
"""Canonical equation: measured witness reverse-waterfill allocation (#336/#157).

This is a DERIVED allocation law, not an empirical score claim.  Each tensor's
discrete rate-distortion points must first be MEASURED through the byte-closed
receiver and frozen scorers.  The selected joint allocation remains advisory
until it is replayed as one archive because Brotli rate and scorer effects are
not separable across simultaneous tensor changes.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "witness_measured_reverse_waterfill_v1"
_MEMO = ".omx/research/witness_sensitivity_bitalloc_336_20260713.md"
_AXIS = "[macOS-CPU advisory; exact LVLS1 NumPy receiver; real-GT n600]"


def build_witness_measured_reverse_waterfill_v1() -> CanonicalEquation:
    """Build the #157 KKT law specialized only by #336's measured witness table."""
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Measured witness reverse-waterfill over byte-closed tensor RD curves",
        one_line_summary=(
            "Convexify each measured tensor RD curve, accept coarsening segments while "
            "their marginal distortion per byte is below 25/37,545,489, then jointly replay."
        ),
        latex_form=(
            r"D_{tq}=100(d^{tq}_{seg}-d^0_{seg})+"
            r"\sqrt{10d^{tq}_{pose}}-\sqrt{10d^0_{pose}},\quad "
            r"q_t^*=\arg\min_q\{D_{tq}+\lambda(B_{tq}-B_0)\},\quad "
            r"\lambda^*=25/37{,}545{,}489"
        ),
        python_callable_module_path=(
            "tac.witness_sensitivity_bitalloc:solve_measured_reverse_waterfill"
        ),
        domain_of_validity={
            "rate_points": "actual archive ZIP bytes for one changed witness tensor",
            "distortion_points": "real-GT n600, exact shipped LVLS1 NumPy receiver, frozen CPU scorers",
            "precision_grid": "int8,int7,int6,int5,int4,int3,int2",
            "separability": "allocation only; combined archive replay is mandatory",
            "score_authority": "NONE until byte-closed upstream exact evaluation",
            "verdict_scope": "INSTANCE x FORMULATION: frozen #406 V9 ep150 EMA-best witness and post-hoc scalar quantization",
        },
        units_in={
            "archive_bytes": "bytes",
            "d_seg": "fraction_argmax_disagreement",
            "d_pose": "official_pose_MSE",
        },
        units_out={
            "nbits": "integer_bits_per_tensor",
            "predicted_delta_S": "contest_score_units_advisory",
        },
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-07-13T04:22:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools.probe_witness_sensitivity_bitalloc",
            "tac.losses.variable_level_codec",
        ),
        canonical_producers=(
            "tac.witness_sensitivity_bitalloc.solve_measured_reverse_waterfill",
            "tac.losses.variable_level_waterfill_allocator.solve_waterfill_allocation",
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "new frozen witness checkpoint, new precision grid, or new byte-closed n600 response table"
            ),
            measurement_axis=_AXIS,
            hardware_substrate="m5_max_cpu",
        ),
    )


def populate_witness_measured_reverse_waterfill_equation(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Idempotently append the derived law to the canonical equation registry."""
    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_witness_measured_reverse_waterfill_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="task_336_reuses_task_157_measured_kkt_reverse_waterfill",
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "build_witness_measured_reverse_waterfill_v1",
    "populate_witness_measured_reverse_waterfill_equation",
]
