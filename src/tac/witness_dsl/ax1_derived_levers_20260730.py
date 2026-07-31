# SPDX-License-Identifier: MIT
"""ax1 all-axes derived levers — DESIGNED-state DSL stubs (ddm_ax1, task #789).

The DSL-holds-every-designed-lever law: the levers DERIVED by the ax1 all-axes pass
(.omx/research/ddm_ax1_all_axes_derivation_20260730.md) exist here as default-OFF,
never-fired stub factories from the moment of DESIGN, so the config-orphan class cannot
recur.  The burn-3 composition arm supersedes each stub with its real implementation in
the trainer/DSL surface it names — fold-and-delete each stub at real-implementation
landing.  Deliberately SEPARATE from actively-edited modules (the ph3_s10 anti-collision
pattern).

FOLD-AND-DELETE LOG (ddm_pa1b, task #793 — the ph3 pattern dw1 used for QA75):
  * ``Ax1MarginCoupledTokenQuant`` (§2a) — SUPERSEDED by the REAL factory
    ``spec_tr1_renderer_20260728.lever_token_quant_margin_coupling`` (+ trainer flags
    ``--token-quant-margin-coupling`` / ``--token-quant-coupling-field`` / ``-min-levels``;
    logic in ``ax1_pool_a_levers_20260730``).  Stub deleted.
  * ``Ax1DeltaGroupSparsity`` (§4a/§5) — SUPERSEDED by
    ``spec_tr1_renderer_20260728.lever_delta_group_sparsity`` (+ trainer flags
    ``--token-delta-group-sparsity`` / ``--delta-sparsity-weight`` / ``-engage`` /
    ``-weight-field``).  Stub deleted.
  * ``Ax1PoolAJointRace`` (§9/§10) — SUPERSEDED by the REAL race harness
    ``spec_tr1_burn2_20260731.pool_a_race_programs`` + ``ax1_pool_a_race_20260730``
    (theorem-2 enumeration + matched-bytes seal + hull-curvature analyzer).  Stub deleted.

Remaining DESIGN-stub: ``Ax1Frame0CarriedWarp`` (§4b, Pool C — orthogonal to Pool-A by the
band lemma; NOT ddm_pa1b's scope, still design-only pending its own realization arm).

Pointer honesty: 0.1910828242 [contest-CPU] UNMOVED; every number in the notes is
[macOS-CPU advisory] with provenance in the memo's cited rows.
"""

from __future__ import annotations

from tac.witness_dsl.curriculum_dsl import Lever


def Ax1Frame0CarriedWarp(window: int = 100) -> Lever:  # stub — Pool C, own arm supersedes
    """ax1 §4b: frame_0 = receiver-side carried-ξ warp of rendered frame_1 (replaces zeros stub).

    frame_0 is structurally seg-free — pure pose surface; the warp is rule-118 FREE receiver
    code, ZERO new tokens, zero seg risk.  Gives the terminal 6-eq pose solve a physical
    photometric base instead of zeros.  Distinct from QA39 (token-CODING inter-prediction); this
    is frame_0 OUTPUT synthesis.  Pool C (band-lemma orthogonal to the Pool-A byte race).
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


AX1_DERIVED_STUB_LEVERS = (Ax1Frame0CarriedWarp,)
