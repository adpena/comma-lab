# SPDX-License-Identifier: MIT
"""Canonical equation: class-prior LOGIT ADJUSTMENT law (Menon et al. 2021, arXiv 2007.07314) —
DAG FEED-07b lever #3's BUILD half (#218), the textbook ZERO-BYTE rare-class cure.

For a long-tailed K-class target with class priors ``pi_c``, the Fisher-consistent training loss
for the BALANCED error is the logit-adjusted CE: shift each logit by the log-prior INSIDE the loss,

    L_LA(y, f) = -log softmax_y( f + tau * log(pi) )        (train-time form)

equivalently, at decode time, ``argmax_c( f_c - tau * log(pi_c) )`` (the decode-time form; the two
are the SAME estimator — Menon et al. Thm 1). Rare classes (Lane pi~0.0059, Movable pi~0.0124) get
strongly negative ``log pi_c``, so under-predicting them costs more training gradient, at EXACTLY
ZERO archive bytes (a loss-shape, not a weight/table). Under the frozen-scorer witness setup the
adjustment is applied to the FROZEN SegNet logits inside the seg loss only — the deployed/rendered
argmax path reads RAW logits (the witness WEIGHTS absorb the pressure; byte-identity boundary
documented at the trainer's ``_LogitAdjustSegAdapter``).

Both forms are prior-floor-clamped (``max(pi_c, floor)``) so an absent class never yields
``log 0 = -inf``. A GLOBAL constant added to every logit changes neither softmax-CE nor any margin
(top1-top2) form, so the un-centered train-time offsets here and the mean-centered decode-time
offsets in ``tac.boundary_math.laguerre_logit_offset.menon_logit_adjustment_offsets`` (the #218
facet-3 margin-TARGET sister, sign-flipped: ``b = -tau*log(pi)``, zero-sum) are loss-equivalent
up to that constant.

Anchors (honest tiers): the measured n600 GT class priors (the probe input to the law) are
VERIFIED; the witness-side d_seg delta of the loss-time adjustment is ASSUMED_AWAITING_VERIFICATION
until the #218 A/B arm lands.

DSL leg: ``tac.witness_dsl.curriculum_dsl.LogitAdjust`` (emits the real trainer flag
``--logit-adjust-loss-tau``). Trainer leg: ``experiments/
train_levelset_witness_realized_through_R_mlx.py`` (``_logit_adjust_offsets_np`` delegates to
:func:`logit_adjust_offsets` below — the equations leg IS the callable).

means != ends: advisory anchors, NON-PROMOTABLE; pointer 0.19110 UNMOVED.
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

EQUATION_ID = "logit_adjustment_class_prior_law_v1"

_UTC = "2026-07-07T00:00:00Z"
_ADVISORY = "[macOS-MLX research-signal]"
_PREDICTED = "[predicted]"
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"

# --- MEASURED constants (probe-measured GT class-area priors, n600 cached L*; canonical comma10k
# class order [Road0, Lane1, Undrivable2, Movable3, MyCar4] — NEVER the forbidden luma-sort) -------
MEASURED_GT_CLASS_PRIORS_N600 = (0.232, 0.0059, 0.495, 0.0124, 0.254)

#: prior floor: an absent class is clamped here so log(prior) stays finite (fail-closed vs -inf).
PRIOR_FLOOR = 1e-8


def logit_adjust_offsets(
    class_priors: np.ndarray, tau: float, *, floor: float = PRIOR_FLOOR
) -> np.ndarray:
    """TRAIN-time logit-adjustment offsets ``offset_c = tau * log(max(prior_c, floor))``.

    ``class_priors``: ``(K,)`` non-negative class frequencies (normalized here if not already).
    Add the returned ``(K,)`` float32 vector to the logits INSIDE the training loss (Menon et al.
    logit-adjusted CE; inside a margin form it is the additive per-class margin generalization).
    Sign convention: this is the ``+tau*log(pi)`` TRAIN-time form; the DECODE-time sister (the
    facet-3 ``menon_logit_adjustment_offsets``) is ``-tau*log(pi)`` mean-centered. Fail-closed on
    malformed priors / non-finite tau."""
    p = np.asarray(class_priors, np.float64).reshape(-1)
    if p.shape[0] < 2:
        raise ValueError(f"class_priors must be (K,) with K>=2, got shape {p.shape}")
    if (p < 0).any() or not np.isfinite(p).all():
        raise ValueError("class_priors must be finite and non-negative")
    total = float(p.sum())
    if total <= 0.0:
        raise ValueError("class_priors sum must be > 0")
    if not np.isfinite(float(tau)):
        raise ValueError(f"tau must be finite, got {tau!r}")
    if not (float(floor) > 0.0):
        raise ValueError(f"floor must be > 0, got {floor!r}")
    pi = np.maximum(p / total, float(floor))
    return (float(tau) * np.log(pi)).astype(np.float32)


def build_logit_adjustment_class_prior_law_v1() -> CanonicalEquation:
    """Build the class-prior logit-adjustment law (the #218 loss-time build's equations leg)."""

    anchor_priors_measured = EmpiricalAnchor(
        anchor_id="gt_class_area_priors_n600_probe_20260707",
        measurement_utc=_UTC,
        inputs={
            "source": "cached GT L* argmax (frozen CPU-torch SegNet), n600 class-area fractions",
            "class_order": "[Road0, Lane1, Undrivable2, Movable3, MyCar4] (canonical comma10k; "
                           "NEVER the forbidden luma-sort)",
        },
        predicted_output={"long_tail": "Lane+Movable are ~1.8% of pixels => rare-class starvation"},
        empirical_output={
            "priors": list(MEASURED_GT_CLASS_PRIORS_N600),
            "log_priors": [round(float(x), 4) for x in
                           np.log(np.asarray(MEASURED_GT_CLASS_PRIORS_N600))],
            "verdict": ("Lane log-prior -5.13 / Movable -4.39 vs Road -1.46 => the adjustment "
                        "shifts loss pressure onto exactly the two measured un-born island classes "
                        "(FEED-07c: lane 83.9% / movable 93.1% un-born)"),
        },
        residual=0.0,
        source_artifact=_DAG,
        measurement_method="GT class-area bincount over the cached n600 L* (theta-independent probe)",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_DAG,
            reactivation_criteria="re-derive on any GT-cache / pair-set change (priors are per-set)",
            measurement_axis=_ADVISORY,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    anchor_dseg_delta_owed = EmpiricalAnchor(
        anchor_id="logit_adjust_loss_dseg_delta_owed_218_ab_20260707",
        measurement_utc=_UTC,
        inputs={
            "literature": "Menon et al. 2021 (arXiv 2007.07314) logit-adjusted CE, Fisher-consistent "
                          "for balanced error; zero-byte loss-shape",
            "trainer_route": "--logit-adjust-loss-tau (serial base_loss adapter wrap; "
                             "micro-batch>1 fails closed)",
            "boundary": "training-LOSS surface only; deployed argmax/verdict/byte-close read RAW logits",
        },
        predicted_output={
            "claim": "rare-class (Lane/Movable) island birth pressure at 0 bytes => d_seg drop on "
                     "the FEED-07c measured un-born island mass (63.9% of flips)",
        },
        empirical_output={
            "status": "OWED — the #218 loss-time A/B arm is the owed anchor",
        },
        residual=0.0,
        source_artifact=_DAG,
        measurement_method="derivation only (no loss-time-adjusted run yet; #218 A/B owed)",
        empirical_verification_status=ASSUMED_AWAITING_VERIFICATION,
        provenance=build_provenance_for_predicted(
            model_id="logit_adjustment_class_prior_law.dseg_delta",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("Class-prior logit adjustment (Menon et al. 2021): train-time offset "
              "tau*log(prior_c) on the seg-loss logits — Fisher-consistent zero-byte rare-class "
              "cure; measured n600 priors [0.232, 0.0059, 0.495, 0.0124, 0.254]"),
        one_line_summary=(
            "logits_c += tau*log(prior_c) inside the seg loss shifts gradient onto the rare "
            "(Lane/Movable) classes at zero archive bytes; deployed argmax reads raw logits."
        ),
        latex_form=(
            r"\mathcal{L}_{\mathrm{LA}}(y,f)=-\log\,\mathrm{softmax}_y\!\big(f+\tau\log\pi\big),"
            r"\quad \pi_c=\max(\mathrm{prior}_c,\ \epsilon_{\mathrm{floor}})"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.logit_adjustment_class_prior_20260707:logit_adjust_offsets"
        ),
        domain_of_validity={
            "vehicle": ["softmax_of_sdf_levelset_witness"],
            "lever": ("tac.witness_dsl.curriculum_dsl.LogitAdjust (--logit-adjust-loss-tau); "
                      "SISTER: the #218 facet-3 margin-TARGET pair --logit-adjust-per-class + "
                      "--logit-adjust-tau (decode-form offsets via "
                      "tac.boundary_math.laguerre_logit_offset.menon_logit_adjustment_offsets) — "
                      "different surface, the two compose"),
            "measurement_axis": ["macOS-MLX research-signal", "predicted"],
            "note": ("priors are PER-PAIR-SET (recompute from the loaded gt cache at run time; the "
                     "n600 constants above are the measured anchor); micro-batch>1 not routed "
                     "(fails closed at the trainer)"),
        },
        units_in={"class_priors": "fraction_of_pixels", "tau": "dimensionless"},
        units_out={"offsets": "logits"},
        empirical_anchors=(anchor_priors_measured, anchor_dseg_delta_owed),
        predicted_vs_empirical_residual={
            # the rare-class prior mass the adjustment targets (Lane+Movable fraction, measured).
            "rare_class_prior_mass_n600": float(
                MEASURED_GT_CLASS_PRIORS_N600[1] + MEASURED_GT_CLASS_PRIORS_N600[3]),
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "experiments/train_levelset_witness_realized_through_R_mlx.py",
            "tac.witness_dsl.curriculum_dsl",
        ),
        canonical_producers=(
            "experiments/train_levelset_witness_realized_through_R_mlx.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="logit_adjustment_class_prior_law.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="apple_m5_max_cpu_mlx",
        ),
    )


def populate_logit_adjustment_class_prior_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the logit-adjustment class-prior law into the
    canonical registry (latest-row-wins query semantics). Equations leg of DAG FEED-07b lever #3's
    BUILD half; DSL leg = ``LogitAdjust``; trainer leg = ``--logit-adjust-loss-tau``."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_logit_adjustment_class_prior_law_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="logit_adjustment_class_prior_20260707 (equations leg of FEED-07b lever #3 BUILD; "
              "DSL leg = LogitAdjust; trainer leg = --logit-adjust-loss-tau; #218 A/B = owed anchor)",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "MEASURED_GT_CLASS_PRIORS_N600",
    "PRIOR_FLOOR",
    "build_logit_adjustment_class_prior_law_v1",
    "logit_adjust_offsets",
    "populate_logit_adjustment_class_prior_equation",
]
