# SPDX-License-Identifier: MIT
"""Canonical chain-rule identity for replacing a frozen SegNet backward graph.

This law is exact only when the supplied input costate is the real teacher
costate at the current rendered frame.  A student or cache is therefore a
provider hypothesis, not an authority claim; the typed scorer-gradient policy
owns its periodic teacher verification and fail-closed fallback.
"""
from __future__ import annotations

from tac.canonical_equations.equation import RECALIBRATE_NEVER_AUTO, CanonicalEquation
from tac.provenance.builders import build_provenance_for_research_sidecar

SEGNET_COSTATE_INJECTION_EQUATION_ID = "segnet_input_costate_injection_v1"
_MEMO = ".omx/research/frozen_segnet_necessity_optimality_alternatives_20260712.md"


def build_segnet_input_costate_injection_v1() -> CanonicalEquation:
    """Build the analytic triality leg; empirical provider residuals are separate."""

    return CanonicalEquation(
        equation_id=SEGNET_COSTATE_INJECTION_EQUATION_ID,
        name="Frozen-SegNet input-costate injection chain rule",
        one_line_summary=(
            "A detached exact input costate reproduces the teacher parameter gradient as "
            "J_x(theta)^T lambda without retaining the teacher graph."
        ),
        latex_form=(
            r"L_{inj}(\theta)=\langle\operatorname{stopgrad}(\hat\lambda),x(\theta)\rangle,\quad "
            r"\nabla_\theta L_{inj}=J_x(\theta)^\top\hat\lambda;\ "
            r"\hat\lambda=\nabla_xL_{teacher}\Rightarrow\nabla_\theta L_{inj}="
            r"\nabla_\theta L_{teacher}"
        ),
        python_callable_module_path=(
            "tac.boundary_math.segnet_gradient_replacement:costate_injection_loss_numpy"
        ),
        domain_of_validity={
            "included": (
                "differentiable renderer x(theta)",
                "costate shape exactly equals rendered-frame shape",
                "costate is finite and stop-gradient detached",
                "global provider-vs-teacher input-costate agreement passes explicit thresholds",
                "optional annulus agreement is additional and never replaces global agreement",
            ),
            "exact_identity_condition": (
                "lambda_hat equals d(real frozen-teacher relaxation)/dx at the current frame"
            ),
            "approximate_provider_policy": (
                "teacher and provider anchor costates are evaluated on the same content-hashed "
                "frame; the current injection costate is separately bound to the current frame "
                "and step; at a teacher refresh, anchor and current frame hashes must be equal "
                "and the anchor/current provider costate bytes must be identical; periodic "
                "global metrics plus a provenance-bound one-step teacher-loss regret gate are "
                "required; any failure selects full_teacher"
            ),
            "objective_context_binding": (
                "SHA-256 over scorer, preprocess, receiver R, GT targets, pair index, loss name "
                "and parameters, and stage name and parameters must match policy, anchor, current "
                "provider evaluation, and teacher step check; the context is rehashed on every "
                "decision and compared with the compile-time fingerprint"
            ),
            "custody_binding": (
                "provider checkpoint/cache SHA-256 is verified at compile and each teacher "
                "refresh; a device/inode/size/mtime/ctime stat fingerprint is checked between "
                "refreshes; provider evidence carries the same custody SHA-256"
            ),
            "excluded": (
                "forward/logit agreement used as a substitute for input-gradient agreement",
                "mask-only agreement with a globally wrong costate",
                "cross-frame, cross-objective, cross-provider, or replayed-step evidence",
                "short-horizon endpoint/cadence evidence described as non-refresh gradient agreement",
                "exact argmax as a differentiable training loss",
                "score or frontier claims without upstream/evaluate.py on exact archive bytes",
            ),
            "authority": "analytic identity plus local behavioral receipt; score_claim=false",
        },
        units_in={
            "theta": "renderer_parameter_units",
            "x": "rendered_frame_units",
            "lambda_hat": "teacher_loss_per_rendered_frame_unit",
            "objective_context_fingerprint": "sha256",
            "anchor_frame_sha256": "sha256",
            "provider_custody_sha256": "sha256",
        },
        units_out={"gradient": "teacher_loss_per_renderer_parameter_unit"},
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-07-12T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_NEVER_AUTO,
        canonical_consumers=(
            "tac.witness_dsl.scorer_gradient_policy",
            "tools.probe_segnet_costate_injection",
        ),
        canonical_producers=("tools.probe_segnet_costate_injection",),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "promote a provider only after governed on-trajectory teacher-gradient, "
                "short-horizon, wall-time, and full-P=600 exact-scorer A/B evidence"
            ),
            measurement_axis="[derived chain-rule; local research receipt]",
            hardware_substrate="framework-portable",
        ),
    )


def populate_segnet_input_costate_injection_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Idempotently append the equation through the canonical registry writer."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_segnet_input_costate_injection_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="frozen_segnet_gradient_replacement_contract_20260712",
    )
    return equation


__all__ = [
    "SEGNET_COSTATE_INJECTION_EQUATION_ID",
    "build_segnet_input_costate_injection_v1",
    "populate_segnet_input_costate_injection_v1",
]
