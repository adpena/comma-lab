# SPDX-License-Identifier: MIT
"""ax1 all-axes derived levers — DESIGNED-state DSL stubs (ddm_ax1, task #789).

The DSL-holds-every-designed-lever law: the four levers DERIVED by the ax1
all-axes pass (.omx/research/ddm_ax1_all_axes_derivation_20260730.md) exist here
as default-OFF, never-fired stub factories from the moment of DESIGN, so the
config-orphan class cannot recur.  The burn-3 composition arm supersedes each
stub with its real implementation in the trainer/DSL surface it names —
fold-and-delete each stub at real-implementation landing.  Deliberately SEPARATE
from actively-edited modules (the ph3_s10 anti-collision pattern).

Derivation homes (memo section per lever): §2a margin-coupled token quant ·
§4a delta group-sparsity (ξ-informed weight field per §5) · §4b frame_0
carried-ξ warp · §10 Pool-A joint-race constraint.  NOT duplicated here:
QA84 rowband (BUILT, lever_token_rowband) · QA75/QA80/QA81 (built stubs in
ph3_s10_frontloaded_levers_20260731).

Pointer honesty: 0.1910828242 [contest-CPU] UNMOVED; every number in the notes
is [macOS-CPU advisory] with provenance in the memo's cited rows.
"""

from __future__ import annotations

from tac.witness_dsl.curriculum_dsl import Lever


def Ax1MarginCoupledTokenQuant(window: int = 100) -> Lever:  # stub — burn-3 real impl supersedes
    """ax1 §2a: per-cell token precision allocated by aggregated flip-distance mass.

    The loss is margin-weighted but the quantizer spends noise uniformly; the
    flip-distance law d=|m|/||dw|| says quant noise should concentrate where
    margins are wide.  Pool-A member (competes with rowband + shrinkage for the
    same counted bytes — joint race only, never stack-claimed).
    """

    return Lever(
        "ax1_margin_coupled_token_quant",
        overrides={"--token-quant-margin-coupling": True},
        epochs_delta=window,
        notes=(
            "DESIGNED-STUB (never-fired): derived ax1 §2a from the SegNet-fractal flip-distance "
            "law; falsifier = matched-SMEVR-byte A/B no d_seg win => instance-close. Pool A."
        ),
    )


def Ax1DeltaGroupSparsity(window: int = 100) -> Lever:  # stub — burn-3 real impl supersedes
    """ax1 §4a: group-shrinkage on per-pair token deltas (ξ-informed weight field, §5).

    op1 P2 measured 98.806% image-stationary flip mass, but training has NO
    delta-shrinkage force — SMEVR exploits stationarity only at coding time.
    Shrink deltas at the SOURCE (relaxed in the movable band / dash corridor per
    the ego-motion prior).  Ordering constraint (§7): engage only AFTER the
    base-stability event.  Pool-A member.
    """

    return Lever(
        "ax1_delta_group_sparsity",
        overrides={"--token-delta-group-sparsity": True},
        epochs_delta=window,
        notes=(
            "DESIGNED-STUB (never-fired): derived ax1 §4a/§5 from op1's 98.806% image-stationarity; "
            "engage-after-base-stability event per §7; falsifier = d_seg cost at matched bytes. Pool A."
        ),
    )


def Ax1Frame0CarriedWarp(window: int = 100) -> Lever:  # stub — burn-3 real impl supersedes
    """ax1 §4b: frame_0 = receiver-side carried-ξ warp of rendered frame_1 (replaces zeros stub).

    frame_0 is structurally seg-free — pure pose surface; the warp is rule-118
    FREE receiver code, ZERO new tokens, zero seg risk.  Gives the terminal 6-eq
    pose solve a physical photometric base instead of zeros.  Distinct from QA39
    (token-CODING inter-prediction); this is frame_0 OUTPUT synthesis.  Pool C.
    """

    return Lever(
        "ax1_frame0_carried_warp",
        overrides={"--frame0-carried-warp": True},
        epochs_delta=window,
        notes=(
            "DESIGNED-STUB (never-fired): derived ax1 §4b/§5/§6; falsifier = worse d_pose than the "
            "zeros base through the terminal solve => instance-close. Pool C (band-lemma orthogonal)."
        ),
    )


def Ax1PoolAJointRace(window: int = 0) -> Lever:  # stub — burn-3 race harness supersedes
    """ax1 §9/§10: the Pool-A CONSTRAINT as a lever — rowband x quant-coupling x shrinkage race jointly.

    Non-additive-pools law: the three token-byte levers enter ONE allocation race
    at matched SMEVR bytes (extends qa84_grammar_race_programs); per-lever
    stack-claims are structurally refused.  v19b's +0.0805 synergy precedent is
    the measured reason the race is joint.
    """

    return Lever(
        "ax1_pool_a_joint_race",
        overrides={"--pool-a-joint-race": True},
        epochs_delta=window,
        notes=(
            "DESIGNED-STUB (never-fired): op-routable 4 — the one named build gap; consumer = "
            "burn-3 config window; extends qa84_grammar_race_programs with quant+shrinkage arms."
        ),
    )


AX1_DERIVED_STUB_LEVERS = (
    Ax1MarginCoupledTokenQuant,
    Ax1DeltaGroupSparsity,
    Ax1Frame0CarriedWarp,
    Ax1PoolAJointRace,
)
