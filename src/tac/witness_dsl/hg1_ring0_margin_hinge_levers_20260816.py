# SPDX-License-Identifier: MIT
"""HG1 TR1 consumers: the signed ring-0 margin hinge, at a DERIVED target.

The signed margin hinge is ALREADY BUILT in the TR1 renderer trainer:
``make_loss_fn``'s ``margin_hinge`` form is ``relu(margin_target - signed)`` with
``signed = logit[GT] - max_{c != GT} logit[c]`` -- exactly the quantity ddm_rn1
identified as the one a decoder cannot have and training can.  These factories do NOT
add a term.  They give the DSL custody of the three trainer flags that steer it, which
``lever_registry.completeness()`` reported as UNMAPPED on the live vehicle, and they
replace the trainer's hand-typed ``--margin-target 1.0`` default with the value the
measured R-noise floor implies.

Why the default target is wrong (MEASURED, ddm_hg1 n=96 seeded-random,
``[macOS-CPU advisory]``): the hinge's active support is the set ``m < margin_target``.
At the trainer default 1.0 that is 1.2295% of the frame, of which only 5,448 pixels are
actual flips -- **97.7% of the hinge's gradient lands on pixels that are already
correct**.  The margin field is small ONLY near the decision boundary, so the hinge is
already 99.56% ring-0-concentrated at a small target and needs no support mask; what it
needs is a target that stops pulling safe pixels.

``m_safe = headroom * delta_R`` is that target.  ``delta_R = 0.019590163230895963`` is
the MEASURED p95 uint8-induced margin perturbation over the annulus, so a pixel parked
below it can flip back under the round trip the scorer actually applies.  The trainer
default sits 25.5x above it.  The value is resolved LIVE from the artifact through the
registered ``margin_band_satisficing_threshold_v1`` law -- never hardcoded here.

Pose safety is a SEPARATE, composable leg (``lever_hg1_q3_constrained_seg_grad``).  It is
not folded into the hinge factory, because the exact-kernel guarantee it rests on holds
only PRE-quantization (#532: uint8 rounding measured 62.74 against 1.7e-13), and the one
built Q3-constrained solve returned ``Q3_FIRST_ROUTE_NOT_CLEARED_FORMULATION_SCOPE`` with
a d_pose ratio of 1.0424 -- NOT flat.  Composing it is a decision a launch must make and
measure, not a default this module may grant.

Every factory is opt-in: not composing one leaves the trainer's own default
(``--seg-form-start ce``), so an uncomposed program is byte-identical.  ``score_claim`` is
False on both -- these are training-force levers, advisory until byte-closed.
"""

from __future__ import annotations

from pathlib import Path

from tac.witness_dsl.curriculum_dsl import Lever

TRAINER_RELPATH = "experiments/train_tr1_partition_renderer_mlx.py"

#: The artifact the hinge target is resolved from (MEASURED; never a literal here).
DELTA_R_ARTIFACT = "reports/delta_R_noise_floor.json"

#: R-survival multiplier shared with the sister ``MarginBandSatisficing`` lever, so the
#: two forces cannot disagree about what "safe" means on the same annulus.
DEFAULT_HEADROOM = 2.0

#: The trainer's own hand-typed default, kept ONLY so the notes can price the delta.
TRAINER_DEFAULT_MARGIN_TARGET = 1.0

#: ddm_hg1 n=96 seeded-random, [macOS-CPU advisory]. Fraction of the hinge's active
#: support that is an already-correct pixel, at the trainer default target.
WASTED_GRADIENT_SHARE_AT_TRAINER_DEFAULT = 0.9765


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def lever_hg1_ring0_margin_hinge(
    headroom: float = DEFAULT_HEADROOM,
    margin_weighted: bool = True,
    delta_r_artifact: str | Path = DELTA_R_ARTIFACT,
) -> Lever:
    """Signed ring-0 margin hinge at the R-survival target, with the annulus reweight.

    Emits the ALREADY-BUILT ``margin_hinge`` seg form plus the target the measured
    R-noise floor implies.  ``margin_weighted`` routes the canonical inverse-GT-margin
    per-pixel reweight; ``margin_hinge`` is in the trainer's
    ``MARGIN_WEIGHTED_HONORING_SEG_FORMS``, so the trainer's own
    ``assert_margin_weighted_loss_is_honored`` guard passes rather than refusing.

    STRUCTURAL LIMIT, stated because a launch must plan around it: ``margin_hinge`` is a
    START-ONLY form.  ``reachable_seg_forms`` gives an outgoing transition to ``ce``
    alone (ce -> tau_softplus at the knee), so a run cannot SCHEDULE the hinge as a
    finishing stage -- it occupies the form from epoch 0.  A from-scratch hinge run
    therefore has no CE trunk, and must be judged at the seg asymptote against a matched
    ``--seg-form-start ce`` control on the same seed and schedule, never against a warm
    incumbent.

    Falsifier: the hinge is INERT if its active support is ~0, and it is not a hinge at
    all -- it is a global margin push that will fight rate -- if its active support
    approaches the whole frame.  ddm_hg1 measured both edges; the target resolved here
    sits between them.
    """
    if not (float(headroom) > 0.0):
        raise ValueError(
            f"lever_hg1_ring0_margin_hinge: headroom must be > 0, got {headroom!r} -- a "
            "non-positive multiplier would place the hinge target at or below the "
            "MEASURED R-noise floor, where a corrected pixel cannot survive the round trip."
        )

    from tac.canonical_equations.margin_band_satisficing_threshold_20260712 import (
        margin_safe_lawref,
        resolve_margin_band_threshold,
    )

    resolved = resolve_margin_band_threshold(
        headroom=float(headroom), artifact_path=delta_r_artifact, repo_root=_repo_root()
    )
    if not (resolved.m_safe < TRAINER_DEFAULT_MARGIN_TARGET):
        raise ValueError(
            f"lever_hg1_ring0_margin_hinge: resolved target {resolved.m_safe} is not below "
            f"the trainer default {TRAINER_DEFAULT_MARGIN_TARGET}; this lever exists to LOWER "
            "the target onto the measured R-survival floor, so a resolution at or above the "
            "default means the artifact or the headroom is wrong and the lever would be inert."
        )

    overrides: dict = {
        "--seg-form-start": "margin_hinge",
        "--margin-target": resolved.m_safe,
    }
    if margin_weighted:
        overrides["--margin-weighted-loss"] = "on"

    return Lever(
        name="hg1_ring0_margin_hinge",
        overrides=overrides,
        notes=(
            "HG1/RN1 signed ring-0 margin hinge on TR1: the ALREADY-BUILT margin_hinge form "
            "relu(margin_target - (logit[GT] - max_{c!=GT} logit)), retargeted from the "
            f"trainer's hand-typed {TRAINER_DEFAULT_MARGIN_TARGET} to m_safe="
            f"{resolved.m_safe} DERIVED-LIVE = headroom({resolved.headroom}) * "
            f"delta_R({resolved.delta_r}) MEASURED uint8 margin-perturbation p95. At the "
            f"trainer default {WASTED_GRADIENT_SHARE_AT_TRAINER_DEFAULT:.1%} of the hinge's "
            "gradient lands on already-correct pixels (ddm_hg1 n=96 advisory). No support "
            "mask is emitted: the hinge is already 99.56% ring-0-concentrated at a small "
            "target, because the margin field is small only at the boundary. START-ONLY form "
            "(no knee transition into it) => needs a matched ce control at the seg asymptote. "
            "score_claim=False; advisory until byte-closed."
        ),
        lawrefs={
            "--margin-target": margin_safe_lawref(
                headroom=float(headroom), artifact_path=delta_r_artifact
            )
        },
        constant_manifest={
            "--margin-target": {
                **resolved.lawref_manifest,
                "single_value_owner": "margin_band_satisficing_threshold_v1",
            }
        },
        runtime_receipt_schemas={
            "hg1_ring0_margin_hinge_init": (
                "ON-only trainer telemetry: resolved margin_target, seg form label, and the "
                "per-epoch hinge active-pixel fraction (the inert/global-push falsifier)"
            ),
        },
        policy_contracts={
            "score_claim": False,
            "default_off_byte_identity": True,
            "adds_new_loss_term": False,
            "start_only_seg_form": True,
        },
    )


def lever_hg1_q3_constrained_seg_grad() -> Lever:
    """Project the seg gradient entering rendered frame_1 onto the yuv6 pose-null Q3.

    Composable pose-safety leg for the hinge.  The trainer flag is args-only and default
    ``off``; ON, the SEG loss gradient is projected blockwise through sq1's exact 6x12
    float projector ``P = I - pinv(A) A``, leaving forward pixels unchanged and the JD1
    pose path on the unwrapped render loss.

    SCOPE, stated because the charter this serves overstated it: the exact-kernel
    guarantee ("a Q3-confined perturbation costs EXACTLY zero d_pose, any amplitude, any
    base") is EXACT only PRE-quantization.  #532 measured uint8 rounding breaking it at
    62.74 against 1.7e-13, and ddm_q31's built Q3-constrained solve returned
    ``Q3_FIRST_ROUTE_NOT_CLEARED_FORMULATION_SCOPE`` at n32 with a d_pose mean ratio of
    1.0424 -- not flat.  So this lever REDUCES expected pose damage; it does not
    guarantee zero.  A launch composing it still owes a measured pose leg.

    Falsifier: composing this changes d_pose by more than the run's own pose noise floor
    in the harmful direction, or it measurably suppresses the seg descent the hinge is
    there to produce (Q3 is a strict subspace, so a seg force with support only in Q4 is
    annihilated -- ddm_js4 measured the #837 seg-reachable overlap SURVIVING, which is
    why this is worth composing, but that was n32 on another module family).
    """
    return Lever(
        name="hg1_q3_constrained_seg_grad",
        overrides={"--seg-grad-q3-project": "on"},
        notes=(
            "HG1/PG1/#889 pose-safety leg for the ring-0 margin hinge: project the SEG "
            "gradient entering rendered frame_1 blockwise onto the frame_1 yuv6-null "
            "subspace Q3 (sq1's exact P = I - pinv(A) A, rank 6 of 12 per 2x2 block). "
            "Forward pixels unchanged; JD1 pose path untouched. The exact-kernel claim is "
            "PRE-quantization only (#532 uint8 breaks it 62.74 vs 1.7e-13; ddm_q31 n32 "
            "measured d_pose ratio 1.0424, NOT flat), so this reduces pose damage rather "
            "than nulling it and a composing launch still owes a measured pose leg. "
            "Args-only, no resumable state, default off is byte-identical. score_claim=False."
        ),
        constant_manifest={
            "--seg-grad-q3-project": {
                "value": "on",
                "rung": "MEASURED-EXACT-KERNEL-PRE-QUANTIZATION",
                "provenance": (
                    "ddm_bo1 / ddm_control_surface_exact_quartering: PoseNet's stem folds to "
                    "affine in eval mode, so A delta = 0 implies the pose input is bit-identical "
                    "(measured 5.684e-14 at the scorer input vs 4.855 for a same-norm generic "
                    "control). Q3 dim = 6 * 49152 = 294912. Holds pre-quantization only."
                ),
            },
        },
        runtime_receipt_schemas={
            "hg1_q3_projection_init": (
                "ON-only trainer telemetry: projector rank check and the per-epoch share of "
                "seg-gradient energy annihilated by P (the inert-projection falsifier)"
            ),
        },
        policy_contracts={
            "score_claim": False,
            "default_off_byte_identity": True,
            "guarantees_zero_pose_damage": False,
        },
    )
