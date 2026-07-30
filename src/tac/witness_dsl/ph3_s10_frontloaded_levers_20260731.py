# SPDX-License-Identifier: MIT
"""ph3 §10 front-loaded seg/photometric levers — DESIGNED-state DSL stubs (10th convocation).

The DSL-holds-every-designed-lever law: the three §10 levers exist here as
default-OFF, never-fired stub factories from the moment of DESIGN, so the
config-orphan class cannot recur.  The ddm_bc1 arm (or burn-2) supersedes each
stub with its real implementation in ``spec_tr1_renderer_20260728`` — this
module is deliberately SEPARATE from that actively-edited surface to avoid a
mid-flight collision; fold-and-delete each stub at real-implementation landing.

Ledger custody: QA75 (routing amendment) / QA80 / QA81 in
``.omx/research/ddm_deferral_queue_ledger_20260729.md``; design memo ph3 §10
(``.omx/research/ddm_ph3_realization_hybrid_adaptive_convocation_20260731.md``,
commit 09f98b2c0b).  Burn-2 headline order: QA75 > QA81 > QA80.

Pointer honesty: 0.1910828242 [contest-CPU] UNMOVED; every number in the notes
is [macOS-CPU advisory] and lives with full provenance in the cited rows.
"""

from __future__ import annotations

from tac.witness_dsl.curriculum_dsl import Lever

# ph3 §10.1 Qa75SolveFrameDistill was the DESIGNED-STUB; its real implementation LANDED in
# ddm_dw1 (2026-07-30) and superseded it per this module's own fold-and-delete note. The live
# lever is ``tac.witness_dsl.spec_tr1_renderer_20260728.lever_solve_frame_distill`` (raced forms
# kd_logits|margin_field|argmax_ce; teacher = the b2b SegNet field; the trainer consumes
# --distill-* flags). The QA80/QA81 stubs below remain DESIGNED-state pending their real landings.


def Qa80MarginBoundedPhotometric(window: int = 100) -> Lever:  # stub — bc1/burn-2 real impl supersedes
    """QA80 (ph3 §10.2): photometric-consistency term confined to the margin-slack budget.

    Per-pixel amplitude bound below the flip distance d=|m|/||dw|| (SegNet-fractal
    law) makes the term PROVABLY seg-flip-free (pp1 band lemma); carries
    pose-legible luma structure from birth (dissolves the L68 photometric wall
    pre-emptively).  Margin-slack region only — PoseNet-visible, NOT resize-null.
    """

    return Lever(
        "qa80_margin_bounded_photometric",
        overrides={"--photometric-margin-budget-weight": True},
        epochs_delta=window,
        notes=(
            "DESIGNED-STUB (never-fired): band lemma x flip-distance law => zero seg cost by "
            "construction; consumer = burn frames' pose-recoverability + QA77 delta "
            "instrument. Ledger QA80; burn-2 headline #3."
        ),
    )


def Qa81LaneCarrierComposite(window: int = 100) -> Lever:  # stub — bc1/burn-2 real impl supersedes
    """QA81 (ph3 §10.3): composite the cb1 Lane band carrier into the render in-training.

    Lane is 38.7% of flips and renderer-REACH-limited (69.5x over its exact
    floor): the renderer trains on the RESIDUAL after the explicit 1-2KB carrier
    paints the thin structure it measurably cannot.  Per-class
    predict-then-innovate (v8 doctrine inside the burn).
    """

    return Lever(
        "qa81_lane_carrier_composite",
        overrides={"--lane-carrier-composite": True},
        epochs_delta=window,
        notes=(
            "DESIGNED-STUB (never-fired): consumes cb1's byte-closed Lane band carrier rows; "
            "ledger QA81; burn-2 headline #2."
        ),
    )


PH3_S10_STUB_LEVERS = (
    # Qa75SolveFrameDistill FOLDED-AND-DELETED (ddm_dw1 2026-07-30): real impl is
    # spec_tr1_renderer_20260728.lever_solve_frame_distill (raced forms; b2b field teacher).
    Qa80MarginBoundedPhotometric,
    Qa81LaneCarrierComposite,
)
