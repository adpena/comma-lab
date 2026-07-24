# SPDX-License-Identifier: MIT
"""Canonical equation for the DDM score-quotient functional family.

This is a contract equation, not an empirical score row.  The distortion
arguments remain future receiver-closed frozen-scorer outputs; the rate is the
exact byte count emitted by the deterministic real-coder packet compiler.
"""

from __future__ import annotations

from pathlib import Path

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
)
from tac.optimization.ddm_score_quotient_functional_contract import (
    ScoreQuotientObjectiveV1,
    score_quotient_functional_objective,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "ddm_score_quotient_functional_v1"
DIRECTIVE_PATH = (
    ".omx/research/"
    "ddm_is1_directive7_score_quotient_functional_family_20260724.md"
)


def ddm_score_quotient_functional(
    d_seg: float,
    d_pose: float,
    receipt,
) -> ScoreQuotientObjectiveV1:
    """Evaluate exact contest S using a real-coder packet receipt."""

    return score_quotient_functional_objective(d_seg, d_pose, receipt)


def build_ddm_score_quotient_functional_v1(
    *,
    provenance_root: str | Path = ".",
) -> CanonicalEquation:
    """Build the design-only canonical equation with fail-closed provenance."""

    sidecar = Path(provenance_root) / DIRECTIVE_PATH
    provenance = build_provenance_for_research_sidecar(
        sidecar_path=sidecar,
        reactivation_criteria=(
            "receiver-closed n600 frozen SegNet/PoseNet fit with exact archive bytes, "
            "DM1 25-row real coder prices, and exact contest-CPU/CUDA replay"
        ),
        measurement_axis="[macOS-CPU frozen-scorer advisory]",
        hardware_substrate="macos_arm64_cpu",
        captured_at_utc="2026-07-24T12:54:13Z",
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="DDM family-(d) score-quotient real-coder functional",
        one_line_summary=(
            "Minimize exact contest S over scorer-plane function parameters, "
            "temporal latents, 25 external placements, and at-risk exceptions."
        ),
        latex_form=(
            r"S(q)=100d_{\rm seg}(\mathcal{R}D(q))"
            r"+\sqrt{10d_{\rm pose}(\mathcal{R}D(q))}"
            r"+25\,B_{\rm counted}(q)/37545489"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_score_quotient_functional_20260724:"
            "ddm_score_quotient_functional"
        ),
        domain_of_validity={
            "included": [
                "typed ddm_score_quotient_functional_contract.v1 receipts",
                "real deterministic coder bytes",
                "scorer-plane RGB plus six pose-target statistics",
                "DM1-owned 25-row values/prices supplied as external typed records",
            ],
            "excluded": [
                "camera-RGB or human-fidelity objectives",
                "proxy entropy in place of emitted bytes",
                "unrealized scorer claims",
                "frontier mutation or promotion",
            ],
            "axis": "[macOS-CPU frozen-scorer advisory]",
            "score_claim": False,
            "frontier_pointer": "0.1910828242 [contest-CPU] UNMOVED",
            "current_verdict": "INCOMPLETE",
            "missing_stream": "FIT_RESULT_RECEIVER_CLOSED_V14_OR_BETTER",
        },
        units_in={
            "d_seg": "fraction_of_SegNet_argmax_pixels",
            "d_pose": "mean_squared_error_over_6_PoseNet_coordinates",
            "receipt.total_counted_bytes": "bytes",
        },
        units_out={"score": "contest_score_units"},
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-07-24T12:54:13Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.optimization.ddm_score_quotient_functional_contract:"
            "build_ddm_event_continuation_v1_fit_request",
            "DDMEventContinuationV1",
        ),
        canonical_producers=(
            "tac.optimization.ddm_score_quotient_functional_contract:"
            "compile_score_quotient_packet",
            "tac.optimization.ddm_score_quotient_functional_contract:"
            "receive_score_quotient_packet",
            "DM1.external_25_row_real_coder_price_records",
        ),
        provenance=provenance,
    )


def populate_ddm_score_quotient_functional_v1(
    *,
    path=None,
    lock_path=None,
    provenance_root: str | Path = ".",
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append the EQUATIONS leg through the locked canonical registry helper."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_ddm_score_quotient_functional_v1(
        provenance_root=provenance_root
    )
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "DC1 family-(d) real-coder objective contract; design-only, "
            "INCOMPLETE pending receiver-closed v14-or-better fit; "
            "FEED-DDM-DC1-SCORE-QUOTIENT-20260724"
        ),
    )
    return equation


__all__ = [
    "DIRECTIVE_PATH",
    "EQUATION_ID",
    "build_ddm_score_quotient_functional_v1",
    "ddm_score_quotient_functional",
    "populate_ddm_score_quotient_functional_v1",
]
