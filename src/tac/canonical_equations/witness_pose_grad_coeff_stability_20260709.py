# SPDX-License-Identifier: MIT
"""Canonical equation: #146 WITNESS POSE-GRADIENT-COEFFICIENT <-> POSE-EPS STABILITY LAW (AMBER unblock).

The score-domain pose loss term mirrors the contest score's ``sqrt(10 * d_pose)`` (HNeRV parity LESSON 6),
implemented as ``sqrt(10*p + eps)`` with ``p = d_pose >= 0``. Its gradient w.r.t. ``p`` is

    d/dp sqrt(10*p + eps) = 10 / (2*sqrt(10*p + eps)) = 5 / sqrt(10*p + eps),

which is monotone-DECREASING in ``p`` and therefore MAXIMAL at ``p = 0``:

    coeff_max(eps) = 5 / sqrt(eps).                                   (the blowup)
    eps_floor(C)   = (5 / C)^2.                                       (the inverse: bound coeff <= C)

At ``eps = 1e-8`` (the value the AMBER deep-unroll DIAGNOSIS saw) this coefficient is ``5e4`` — a
per-pair (batch=1) update on an EASY (near-zero-pose) pair injects a 5e4-scaled gradient with NO
averaging, which (x w_seg=100, x no grad clip) drives the OPTIMIZER-DIVERGENCE DEAD-SATURATION LOCK:
the sigmoid rails, the gradient collapses to 0, the AdamW step -> 0, and the loss FREEZES at EXACTLY
``1070.0802`` (finite, 6-dp identical across steps). The incumbent default (``eps = 1e-2``) bounds the
coefficient to ``50`` — a 1000x reduction — which is why the live #205 run does not collapse.

This law is the invertible identity that CONNECTS the two diagnosed levers (grad clipping + the sqrt-eps
blowup): specifying a STABILITY BUDGET ``coeff_max = C`` is the SAME as raising ``eps`` to ``(5/C)^2``.
It is the analytical basis for the ``--pose-grad-coeff-max`` reparametrization and the ``amber`` preset
(``coeff_max = 25`` => ``eps_floor = 4e-2``, tighter than the incumbent for the batch=1 arm).

TWO PROVABLE DEGENERACIES (VERIFIED_VIA_SOURCE_INSPECTION — the byte-identity contract):
``pose_grad_coeff_max <= 0`` => the effective eps is the incoming ``pose_eps`` UNCHANGED (no override,
byte-identical); and the bound can only RAISE eps (``max(pose_eps, floor)``), never lower it, so it can
never destabilise an already-safer config.

means != ends: the collapse cure it formalizes is BUILT + unit-proven ($0, on a synthetic reproduction
of the 5e4 coefficient), but whether the AMBER preset UN-COLLAPSES the deep-unroll arm at n600 is a
SEPARATE operator-GO heavy launch (un-collapse A/B OWED). Pointer 0.19110 UNMOVED.

DSL leg: ``tac.witness_dsl.curriculum_dsl.WitnessStability``. Mechanism: ``tac.witness_stability``.
Consumer: ``experiments/train_levelset_witness_realized_through_R_mlx.py`` (post-parse resolution).
DAG: ``.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`` (FEED-collapsefix / FEED-amber-unblock).
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "witness_pose_grad_coeff_stability_v1"

_UTC = "2026-07-09T00:00:00Z"
_PREDICTED = "[predicted]"
_ADVISORY = "[macOS-CPU advisory]"
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"

# The exact law anchors (MEASURED coefficient values from the closed-form derivation; the collapse
# signature 1070.0802 is the DAG FEED-amber-unblock diagnosis).
_SCORE_POSE_CONST = 10.0
_COEFF_AT_1EM8 = 5.0e4      # 5/sqrt(1e-8) = the blowup the diagnosis saw
_COEFF_AT_1EM2 = 50.0       # 5/sqrt(1e-2) = the incumbent (tamed)
_COEFF_AMBER = 25.0         # amber preset coeff-max => eps_floor 4e-2
_FROZEN_LOSS = 1070.0802    # the finite dead-saturation lock value (optimizer divergence)


def build_witness_pose_grad_coeff_stability_v1() -> CanonicalEquation:
    """Build the #146 pose-gradient-coefficient <-> pose-eps stability law with its honest-tier anchors.

    Anchor 1 = the closed-form derivation + the two byte-identity degeneracies (source-verified, exact).
    Anchor 2 = the collapse SIGNATURE (eps 1e-8 -> coeff 5e4 -> loss frozen 1070.0802) from the DAG
    diagnosis, with the incumbent eps 1e-2 -> coeff 50 as the tamed contrast; un-collapse-at-n600 OWED."""

    anchor_derivation = EmpiricalAnchor(
        anchor_id="pose_grad_coeff_eps_law_derivation_source_verified_20260709",
        measurement_utc=_UTC,
        inputs={
            "term": "sqrt(10*d_pose + eps)",
            "mechanism": "tac.witness_stability.pose_eps_floor_for_coeff_max / max_pose_grad_coeff",
            "tests": "src/tac/tests/test_witness_stability.py",
        },
        predicted_output={
            "coeff_max_of_eps": "5/sqrt(eps)",
            "eps_floor_of_coeff_max": "(5/C)^2",
            "degeneracy": "coeff_max<=0 => effective pose_eps == pose_eps (byte-identical); "
                          "floor can only RAISE eps (max(pose_eps, floor))",
        },
        empirical_output={
            "coeff_at_1e-8": _COEFF_AT_1EM8,
            "coeff_at_1e-2": _COEFF_AT_1EM2,
            "eps_floor_at_C50": 1e-2,
            "eps_floor_at_C25": 4e-2,
        },
        residual=0.0,
        source_artifact="src/tac/witness_stability.py",
        measurement_method="closed_form_derivation_source_inspection",
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
        provenance=build_provenance_for_predicted(
            model_id="witness_pose_grad_coeff_stability.derivation",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="closed_form",
        ),
    )
    anchor_collapse = EmpiricalAnchor(
        anchor_id="pose_grad_coeff_blowup_collapse_signature_1070_0802_20260709",
        measurement_utc=_UTC,
        inputs={
            "regime": "deep-unroll / per-pair (batch=1) witness training",
            "pose_eps_at_collapse": 1e-8, "w_seg": 100, "grad_clip": 0.0, "lr": 2e-3,
            "diagnosis": "DAG FEED-amber-unblock (#146/#211); do NOT re-diagnose",
        },
        predicted_output={
            "coeff_max": _COEFF_AT_1EM8,
            "mechanism": "sigmoid rails -> grad->0 -> AdamW step->0 -> loss FREEZES (finite, not NaN)",
        },
        empirical_output={
            "frozen_loss": _FROZEN_LOSS,
            "spike": "57 -> 1070",
            "tamed_incumbent": {"pose_eps": 1e-2, "coeff_max": _COEFF_AT_1EM2, "grad_clip": 1.0},
            "un_collapse_at_n600": "OWED (operator-GO heavy launch; means != ends)",
        },
        residual=0.0,
        source_artifact=_DAG,
        measurement_method="deep_unroll_witness_training_collapse_observation",
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_DAG,
            reactivation_criteria="AMBER un-collapse A/B: fresh deep-unroll n600 run with "
                                  "--stability-preset amber -> does d_seg keep descending (no 1070 lock)?",
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("#146 witness pose-gradient-coefficient <-> pose-eps stability law: the score-domain pose "
              "term sqrt(10*d_pose+eps) has gradient coefficient 5/sqrt(10*d_pose+eps) (max 5/sqrt(eps) "
              "at d_pose=0); bound it to <= C by raising eps to (5/C)^2 — the AMBER deep-unroll cure"),
        one_line_summary=(
            "Pose grad coeff = 5/sqrt(10*d_pose+eps) (max 5/sqrt(eps)); eps_floor(C)=(5/C)^2; eps 1e-8"
            "->5e4->loss frozen 1070.0802; incumbent 1e-2->coeff 50 tames; un-collapse A/B owed."
        ),
        latex_form=(
            r"\frac{\partial}{\partial p}\sqrt{a p + \epsilon} = \frac{a}{2\sqrt{a p+\epsilon}}"
            r"\ \le\ \frac{a}{2\sqrt{\epsilon}} =: C_{\max},\quad "
            r"\epsilon_{\mathrm{floor}}(C) = \left(\tfrac{a}{2C}\right)^2,\ a=10"
        ),
        python_callable_module_path=(
            "tac.witness_stability:pose_eps_floor_for_coeff_max"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness", "coord_inr_seg_witness"],
            "regime": "score_domain_pose_loss (sqrt(10*d_pose+eps)); batch=1 deep-unroll divergence-prone",
            "lever": ("tac.witness_dsl.curriculum_dsl.WitnessStability "
                      "(--stability-preset/--pose-grad-coeff-max/--grad-clip)"),
            "measurement_axis": ["predicted", "macOS-CPU advisory"],
            "note": ("the DERIVATION is exact (source-verified); whether the AMBER preset UN-COLLAPSES "
                     "the deep-unroll arm at n600 is a SEPARATE operator-GO heavy launch (OWED)"),
        },
        units_in={"coeff_max": "pose_loss_grad_coefficient", "score_pose_const": "dimensionless"},
        units_out={"pose_eps_floor": "sqrt_argument_epsilon"},
        empirical_anchors=(anchor_derivation, anchor_collapse),
        predicted_vs_empirical_residual={
            "coeff_eps_law_source_verified": 0.0,
            "collapse_signature_diagnosis": 0.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl",       # DSL leg: WitnessStability factory
            "experiments/train_levelset_witness_realized_through_R_mlx.py",  # the wire-in consumer
        ),
        canonical_producers=(
            "tac.witness_stability",
        ),
        provenance=build_provenance_for_predicted(
            model_id="witness_pose_grad_coeff_stability.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu",
        ),
    )


def populate_witness_pose_grad_coeff_stability_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the #146 pose-grad-coeff stability law (mirrors
    ``populate_dseg_aware_fourier_taper_equation``; latest-row-wins). EQUATIONS leg of the AMBER
    collapse-fix; DSL leg = ``curriculum_dsl.WitnessStability``; mechanism = ``tac.witness_stability``."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_witness_pose_grad_coeff_stability_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="witness_pose_grad_coeff_stability_20260709 (equations leg of the #146/#211 AMBER "
              "deep-unroll collapse-fix; DSL leg = WitnessStability; un-collapse A/B operator-GO-owed)",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "build_witness_pose_grad_coeff_stability_v1",
    "populate_witness_pose_grad_coeff_stability_equation",
]
