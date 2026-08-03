# SPDX-License-Identifier: MIT
"""fh1 adapted-force levers (task #805) — DESIGNED-state DSL stubs for burn-4.

The DSL-holds-every-designed-lever law: the v8/v9/v10 forces ADAPTED to the TR1
vehicle by ``.omx/research/ddm_fh1_forces_harvest_20260731.md`` exist here as
default-OFF, never-fired stub factories from the moment of DESIGN, so the
config-orphan class cannot recur.  This module follows the ph3_s10 precedent
verbatim (``ph3_s10_frontloaded_levers_20260731``): it is deliberately SEPARATE
from the actively-edited surfaces (``spec_tr1_renderer_20260728`` + the TR1
trainer, owned by ddm_tp1 / the burn-4 owner this session) to avoid a
mid-flight collision; fold-and-delete each stub at real-implementation landing.

None of these stubs is consumed by any compiled program; the override flags are
DESIGN CUSTODY (the flags do not exist on the trainer yet — building them is
the named owner's job, gated by the fh1 memo's falsifiers).  Firing one through
a validated program would fail closed (never-invent-flags), which is the point.

Pointer honesty: 0.1910828242 [contest-CPU] UNMOVED; every motivating number is
[macOS-CPU advisory] and lives with full provenance in the fh1 memo rows.
"""

from __future__ import annotations

from tac.witness_dsl.curriculum_dsl import Lever

# The trainer these levers were DESIGNED FOR (ddm_lr2, 2026-08-03). Without this line the
# registry's per-module resolution silently defaulted every factory below to the RETIRED
# levelset trainer, so no TR1-scoped query could surface them — they were graded against a
# vehicle they were never written for. The module docstring above states the target verbatim
# ("the v8/v9/v10 forces ADAPTED to the TR1 vehicle"); this declaration makes that binding
# machine-readable instead of a comment. It does NOT make the stubs fireable: the flags below
# exist on NEITHER trainer (MEASURED). It makes their debt correctly ATTRIBUTED to TR1.
TRAINER_RELPATH = "experiments/train_tr1_partition_renderer_mlx.py"


def TieLocusEdgeWeighted(window: int = 0) -> Lever:  # stub — burn-4 owner real impl supersedes
    """fh1 R1+R4: tie-locus sub-pixel displacement x per-class-PAIR edge weight.

    v9 Force 3 (+Force 4 folded) adapted to TR1's through-R realized logits:
    L_tie on genuine inter-class straddles with W[c,c'] raced over
    {uniform, TR1-measured flip-mass, Young's-sigma}.  Pool: PLACEMENT.
    """

    return Lever(
        "fh1_tie_locus_edge_weighted",
        overrides={"--tie-locus-edge-weight": True},
        epochs_delta=window,
        notes=(
            "DESIGNED-STUB (never-fired): providers (t_ref, straddle pairs) are "
            "theta-independent gt-cache numpy = cross-vehicle valid; engage post-knee; "
            "falsifier = mean|t_wit-t_ref| flat OR no d_seg gain vs margin-weight-only "
            "at matched compute. fh1 memo R1+R4."
        ),
    )


def MarginSatisficeCap(window: int = 0) -> Lever:  # stub — burn-4 owner real impl supersedes
    """fh1 R3: #459-corrected satisficing CAP as a 4th _live_margin_weight allocator.

    w = 1[m < m_safe] * base allocator, m_safe = headroom * delta_R with delta_R
    re-measured through TR1's R (v9 anchor 0.0196 is R/scorer-side; UU-1 caveat).
    A CAP, never a finisher stage (sc2 s3.1 binding).  Pool: PLACEMENT budget + RATE.
    """

    return Lever(
        "fh1_margin_satisfice_cap",
        overrides={"--margin-weight-fn-satisfice-cap": True},
        epochs_delta=window,
        notes=(
            "DESIGNED-STUB (never-fired): mean-1 + stop-grad invariants preserved; "
            "raced vs incumbent inverse allocator; falsifier = frac_px_below_msafe "
            "not shrinking OR endpoint d_seg worse (cap starving late descent). fh1 memo R3."
        ),
    )


def XiAdvectedTokenBase(window: int = 0) -> Lever:  # stub — QA90-gated; real impl supersedes
    """fh1 R2: v9 Force-1 temporal screw adapted to token space.

    Advect the shared base by the per-pair GT screw homography on ground/dynamic
    cells before adding deltas (identity elsewhere); deltas stop carrying
    ego-motion-predictable structure.  Pools: FLICKER + RATE (smevr delta bytes).
    GATED on the QA90 $0 temporal-coherence read of the delta stream.
    """

    return Lever(
        "fh1_xi_advected_token_base",
        overrides={"--token-temporal-mode-xi-advected": True},
        epochs_delta=window,
        notes=(
            "DESIGNED-STUB (never-fired): stop-grad GT xi (confound-safe arm only; "
            "carrier_live is DEAD on the seg-only staging per L68); decode side must "
            "mirror the warp (byte-close plan required before fire); falsifier = QA90 "
            "shows no coherent advected delta structure, or raced arm worse on "
            "(d_seg, smevr bytes) jointly. fh1 memo R2."
        ),
    )


def BirthPlateauKneeConjunct(window: int = 0) -> Lever:  # stub — rides P2 gate key + tp1 port
    """fh1 R8: #430 coherent-cascade order as an event CONJUNCT on the knee.

    CE->tau knee (and at_knee quant engage, and KD anneal-to-CE) require
    birth-plateau AND loss-knee (lane betti0_realized slope -> 0 above-nucleus).
    Motivation MEASURED: births +8.75/gate STILL RISING at the ep399 endpoint.
    """

    return Lever(
        "fh1_birth_plateau_knee_conjunct",
        overrides={"--knee-requires-birth-plateau": True},
        epochs_delta=window,
        notes=(
            "DESIGNED-STUB (never-fired): pure event-graph edit, no new loss term; "
            "falsifier = conjunct arm reaches same endpoint births with worse d_seg "
            "at matched compute. Rides the P2 typed birth_completion key. fh1 memo R8."
        ),
    )


def ErfBirthContextCoadapt(window: int = 0) -> Lever:  # stub — SPECULATIVE (post-window-1 rung)
    """fh1 R14 / F-E: ERF birth-context co-adaptation (fresh area-Lagrange term).

    Transient upweight of the seg loss on the ERF-neighborhood annulus
    (r ~ r50 ~ 85px) of components flipping erased->realized within the last
    gate window, so context co-adapts WITH the birth (QA92 collateral physics:
    the cross-term dS^2/d(component)d(context) is large at birth sites).
    """

    return Lever(
        "fh1_erf_birth_context_coadapt",
        overrides={"--erf-birth-context-weight": True},
        epochs_delta=window,
        notes=(
            "DESIGNED-STUB (never-fired, SPECULATIVE-DERIVED): genuinely absent from "
            "v8/v9/v10 AND the config (its physics was measured 07-31, QA92 6c23883be6); "
            "falsifier = no improvement in birth survival / settling epochs at matched "
            "compute. Not a burn-4 arm; a rung behind window-1 birth telemetry. fh1 memo R14."
        ),
    )


FH1_ADAPTED_FORCE_STUB_LEVERS = (
    TieLocusEdgeWeighted,
    MarginSatisficeCap,
    XiAdvectedTokenBase,
    BirthPlateauKneeConjunct,
    ErfBirthContextCoadapt,
)
