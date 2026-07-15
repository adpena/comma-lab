# SPDX-License-Identifier: MIT
"""Canonical equation: AutoClip percentile grad-clip threshold (the #B-4 clip cure).

THE MEASURED POISON (audit §A-1, ``.omx/research/v9_missing_signal_constants_audit_20260715.md``):
the sealed ``--grad-clip 0.5`` is SATURATED on the live V9·CGauge config — C0
(``levelset_n600_witness_20260715T095030Z``) ``grad_clip_activation`` rows ep1-39 show global
``frac_clipped=1.0`` at EVERY accum step, ``norm_mean≈5.9-6.2`` (max 17.5) vs threshold 0.5.
A saturated clip re-parametrizes descent into NORMALIZED flow with effective step
``lr·c/‖g‖ ≈ lr/12`` — the LR cosine no longer controls the descent clock, and every
loss-WEIGHT sweep verdict taken under saturation conflates direction with magnitude
(the weights only set the direction mixture when the norm is pinned).

THE LAW (AutoClip, Seetharaman-Wichern-Venkataramani-Le Roux, arXiv:2007.14469):

    clip_t(step) = percentile_p( {‖g‖_1..step} ∩ window w )

observe-then-threshold per optimizer step; warmup falls back to the fixed incumbent clip.
The threshold ADAPTS to the realized norm distribution so clipping suppresses only
spikes. ZClip (arXiv:2504.02507) is the z-score sibling; the principled successor is the
#500 categorical-Fisher trust region (``categorical_fisher_natural_trust_region``) which
subsumes clipping — this law is the CHEAP static-free approximation on the way there.

Mechanism: ``tac.witness_control.adaptive_grad_clip`` (ring-buffer state machine,
resume-safe under ``__acl_``). DSL leg: ``curriculum_dsl.AdaptiveGradClip``
(``--grad-clip-mode autoclip``; default ``fixed`` => byte-identical incumbent path).

means != ends: the descent-speed EFFECT is ASSUMED_AWAITING_VERIFICATION until the
bounded n24 A/B lands (gradient-quality + no-flicker admission bar per the 2026-07-15
relaxed-identity directive), and NO promotion claim exists without a byte-closed exact
n600 row. Pointer UNMOVED.
"""
from __future__ import annotations

import numpy as np

from tac.canonical_equations.equation import (
    ASSUMED_AWAITING_VERIFICATION,
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "autoclip_percentile_threshold_v1"

_UTC = "2026-07-15T00:00:00Z"
_ADVISORY = "[macOS-MLX research-signal]"
_PREDICTED = "[predicted]"
_AUDIT = ".omx/research/v9_missing_signal_constants_audit_20260715.md"

# --- the C0 saturation anchor numbers (MEASURED, quoted from the audit §A-1) ---------
C0_FRAC_CLIPPED = 1.0          # every accum step, ep1-39
C0_NORM_MEAN_LO = 5.9          # gnorm running mean band
C0_NORM_MEAN_HI = 6.2
C0_NORM_MAX = 17.5
SEALED_CLIP = 0.5              # the saturated constant
EFFECTIVE_STEP_SUPPRESSION = 12.0  # lr_effective ≈ lr / 12 at gnorm≈6, clip 0.5


def autoclip_threshold(history: np.ndarray, percentile: float = 10.0) -> float:
    """The law's pure math: percentile_p of the observed gradient-norm history.

    Deterministic (numpy linear interpolation, fp64). The trainer state machine
    (``tac.witness_control.adaptive_grad_clip``) adds the ring window + warmup
    fallback around this exact call."""
    h = np.asarray(history, np.float64)
    if h.size == 0:
        raise ValueError("autoclip_threshold requires a non-empty history")
    if not (0.0 < float(percentile) < 100.0):
        raise ValueError(f"percentile must be in (0,100), got {percentile!r}")
    return float(np.percentile(h, float(percentile)))


def build_autoclip_percentile_threshold_v1() -> CanonicalEquation:
    """Build the AutoClip percentile clip law with its honest-tier anchors."""

    anchor_saturation = EmpiricalAnchor(
        anchor_id="c0_grad_clip_saturation_frac1_ep1_39_20260715",
        measurement_utc=_UTC,
        inputs={
            "run": "experiments/results/levelset_n600_witness_20260715T095030Z",
            "telemetry": "grad_clip_activation rows ep1-39 (task-408 Q1 producer)",
            "sealed_clip": SEALED_CLIP,
        },
        predicted_output={
            "claim": ("a clip threshold far below the typical norm saturates: frac_clipped→1, "
                      "effective step = lr·c/‖g‖ (normalized flow; LR schedule decoupled)"),
        },
        empirical_output={
            "frac_clipped": C0_FRAC_CLIPPED,
            "norm_mean_band": [C0_NORM_MEAN_LO, C0_NORM_MEAN_HI],
            "norm_max": C0_NORM_MAX,
            "effective_step_suppression_x": EFFECTIVE_STEP_SUPPRESSION,
            "verdict_scope": ("FORMULATION-level for the live config family — every v9_cgauge_* "
                              "shares --grad-clip 0.5; [macOS-MLX advisory], NON-PROMOTABLE"),
        },
        residual=0.0,
        source_artifact=_AUDIT,
        measurement_method="C0 grad_clip_activation telemetry read against the sealed constant",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_AUDIT,
            reactivation_criteria=("re-measure saturation whenever the loss-term stack or the "
                                   "norm scale changes (stage boundaries: l7 / Muon / finisher)"),
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    anchor_effect_owed = EmpiricalAnchor(
        anchor_id="autoclip_descent_speed_effect_n24_ab_owed_20260715",
        measurement_utc=_UTC,
        inputs={
            "arm_a": "--grad-clip 0.5 (sealed control)",
            "arm_b": "--grad-clip-mode autoclip (percentile 10, window 1000, warmup 10)",
            "admission_bar": ("gradient-quality + no-flicker telemetry; d_seg descent per "
                              "WALL-CLOCK (joint objective), not per epoch"),
        },
        predicted_output={
            "hypothesis": ("un-pinning the effective LR (measured 12× suppression) speeds "
                           "epochs-to-target; frac_clipped drops off 1.0 to ~p (the percentile)"),
        },
        empirical_output={
            "status": ("OWED — the bounded n24 A/B is the admission measurement; NO score claim; "
                       "means != ends; pointer UNMOVED"),
        },
        residual=0.0,
        source_artifact=_AUDIT,
        measurement_method="bounded n24 governed A/B (launch pending at registration time)",
        empirical_verification_status=ASSUMED_AWAITING_VERIFICATION,
        provenance=build_provenance_for_predicted(
            model_id="autoclip_percentile_threshold.descent_speed_effect",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("AutoClip percentile grad-clip threshold: clip_t = percentile_p(‖g‖ history, window w) "
              "with fixed-clip warmup — the cure for the measured saturated --grad-clip 0.5"),
        one_line_summary=(
            "Adaptive clip at the p-th percentile of the observed gradient-norm history (AutoClip "
            "2007.14469); cures the C0-measured frac_clipped=1.0 saturation (effective step lr/12)."
        ),
        latex_form=(
            r"c_t=\mathrm{P}_p\left(\{\|g_s\|\}_{s=t-w+1}^{t}\right),\quad "
            r"g'_t=g_t\min\!\left(1,\,c_t/\|g_t\|\right)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.autoclip_percentile_grad_clip_20260715:autoclip_threshold"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness", "v9_cgauge_*"],
            "lever": ("tac.witness_dsl.curriculum_dsl.AdaptiveGradClip "
                      "(--grad-clip-mode autoclip / -percentile / -window / -warmup-steps)"),
            "measurement_axis": ["macOS-MLX research-signal", "predicted"],
            "note": ("training-path lever under the 2026-07-15 relaxed-identity directive "
                     "(drift OK if gradient quality + no flicker); decode/verdict paths untouched. "
                     "Successor law: #500 categorical-Fisher trust region (subsumes clipping)."),
        },
        units_in={"history": "gradient_l2_norms", "percentile": "percent", "window": "steps"},
        units_out={"clip_t": "gradient_l2_norm"},
        empirical_anchors=(anchor_saturation, anchor_effect_owed),
        predicted_vs_empirical_residual={
            "c0_saturation_measured": 0.0,
            "descent_speed_effect_owed_n24_ab": 0.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl",  # DSL leg: AdaptiveGradClip factory
            "experiments/train_levelset_witness_realized_through_R_mlx.py",  # the clip-site consumer
        ),
        canonical_producers=(
            "tac.witness_control.adaptive_grad_clip",
        ),
        provenance=build_provenance_for_predicted(
            model_id="autoclip_percentile_threshold.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )


def populate_autoclip_percentile_threshold_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the AutoClip clip law (latest-row-wins).

    EQUATIONS leg of the #B-4 grad-clip cure; DSL leg = ``curriculum_dsl.AdaptiveGradClip``;
    mechanism = ``tac.witness_control.adaptive_grad_clip``."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_autoclip_percentile_threshold_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="autoclip_percentile_grad_clip_20260715 (equations leg of the #B-4 grad-clip cure; "
              "DSL leg = AdaptiveGradClip; n24 A/B owed)",
    )
    return eq


__all__ = [
    "C0_FRAC_CLIPPED",
    "C0_NORM_MAX",
    "C0_NORM_MEAN_HI",
    "C0_NORM_MEAN_LO",
    "EFFECTIVE_STEP_SUPPRESSION",
    "EQUATION_ID",
    "SEALED_CLIP",
    "autoclip_threshold",
    "build_autoclip_percentile_threshold_v1",
    "populate_autoclip_percentile_threshold_equation",
]
