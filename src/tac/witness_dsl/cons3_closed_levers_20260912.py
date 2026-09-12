# SPDX-License-Identifier: MIT
"""DSL Lever records for the six levers the late 2026-09-11 arms decided (ddm_cons3).

WHY THIS MODULE EXISTS.  The DSL-holds-every-designed-lever law: a lever is not "built" until it is
a ``Lever`` factory, and a lever with a verdict that no factory holds is orphaned signal the
activation ledger can never nag, rank or retire.  cons1 landed the four levers of the day's first
half; these are the six the day's second half decided, plus one that it deliberately did NOT decide.

WHAT THESE FLAGS ARE, STATED SO NOBODY IS MISLED.  **None of the flags below exists on either
trainer** -- they act on the CODER, the ENCODER, the CARRIER and the RENDERER, surfaces no witness
trainer argparse owns, which is exactly why ``lever_registry.completeness().unmapped`` (a
TRAINER-FLAG surface) never listed them and never could.  These are DESIGN-STATE records in the ax1
stub tradition: they carry the verdict and its scope, and they must NOT be compiled into a launch.

THE SIX, with the verdict each carries:

  * ``Pc3Cap1PredictorRefit`` -- LANDED as pointer move 45 (-160 B, S 0.1371383667388406): an
    exhaustive search over the closed legal predictor schema at BIT-IDENTICAL codes (51,581 ->
    50,270 Rice bits), cold n600 decode byte-identical to move 44 across all 3,662,409,600 bytes.
  * ``Ntb2HpacFrameQuadRounding`` -- FALSIFIED at exact bytes (+58 B, archive 179,417 B, ΔS
    +3.8620e-05) and it LOCATES the turn: step 4 buys 483 B more model saving and gives 789 B more
    back on the tail than step 2, so the joint flips sign BETWEEN step 2 and step 4.  Step 2 is the
    winner and the rounding family is closed at this resolution -- coarser does not pay.
  * ``Hpr1ConvADilationRung`` -- **NO VERDICT: designed, costed, never fired.**  It is registered
    UNFIRED on purpose, so the duty-to-measure queue carries it instead of a memo paragraph.  It is
    the cheapest open rung on the coder (one receiver edit, no shape bit).
  * ``Hpr1QRecalibrationRung`` -- CLOSED NEGATIVE, and 51x harder after the refit than before it:
    the held-out recalibration gain is -9,439.97 B on the refit prior against -183.53 B on the old
    one, because the refit moved q CLOSER to its optimum (mean stated confidence 0.99802722 against
    a 0.99802843 argmax-correct rate: a calibration error of -1.21e-06).
  * ``Tmx1TailMixerRefit`` -- CLOSED NEGATIVE at exact bytes: +20 B on the live base, +22 B on move
    47, on a bar of -30.04 B.  Six exact encodes, two bases, four held-out folds, no gain anywhere.
  * ``Rbf1PostRenderBoundaryTreatment`` -- CLOSED at formulation scope on two n600 channels: the
    free post-render pixel actuators pay a QUADRATIC pose tax (~6.6e-3 * tau^2) that symmetric
    application does not cure; and they touch 2,287,200 token-edge pixels at n600 to reach
    the 12,196 that are wrong -- a 187.5:1 collateral exposure.

Pointer honesty: ddm_cons3 measured nothing and moved nothing.  The frontier is
``composition S 0.13638261682704697 @ 179,111 B [contest-CUDA T4 n600] (move 48)``, and every
number here is quoted from the arm that measured it.  Laws: ``hpr1_counted_section_refit_debt_v1``
(now n = 2 with tmx1's negative anchor), ``refit_rungs_ranked_by_conditioned_mass_v1``,
``pc3_rung_price_expires_at_pointer_move_v1``, ``rbf1_post_render_pose_amplitude_tax_v1``.
"""

from __future__ import annotations

from tac.witness_dsl.curriculum_dsl import Lever

# Declared so the registry attributes this module's debt to the vehicle we ship rather than
# silently defaulting it to the retired levelset trainer. Building any of these levers is NOT
# implied: their flags exist on NEITHER trainer (MEASURED), and all six act on the coder, the
# encoder, the carrier or the renderer, which no trainer argparse owns.
TRAINER_RELPATH = "experiments/train_tr1_partition_renderer_mlx.py"


def Pc3Cap1PredictorRefit(window: int = 100) -> Lever:  # DESIGN-STATE record
    """pc3: refit the pose carrier's CAP1 predictor at bit-identical codes.

    LANDED as pointer move 45: archive 180,406 -> 180,246 B (-160 B), S 0.1371383667388406
    [contest-CUDA T4 n600], commit 01f2b66ad. An exhaustive search over the closed legal schema
    moved 51,581 -> 50,270 Rice bits without changing a single decoded code, so the cold n600
    public decode is byte-identical to move 44's retained raw across all 3,662,409,600 bytes and
    d_seg / d_pose are move 44's by construction. Superseded as a rung only by its own price:
    pc3 measured that rung prices EXPIRE at a pointer move (per-dim 54 -> 75 B, global 808 ->
    914 B across this very move).
    """
    return Lever(
        "pc3_cap1_predictor_refit",
        overrides={"--pose-carrier-predictor-refit": True},
        epochs_delta=window,
        notes=(
            "DESIGN-STATE (carrier surface; flag exists on NEITHER trainer): LANDED as pointer "
            "move 45 at bit-identical codes. Law: pc3_rung_price_expires_at_pointer_move_v1. "
            "Do not compile into a launch."
        ),
    )


def Ntb2HpacFrameQuadRounding(window: int = 100) -> Lever:  # DESIGN-STATE record
    """ntb2's step-4 sibling of the frame-embedding rounding, fired by hpr1 on the refit prior.

    FALSIFIED at exact bytes and USEFUL anyway: 3,543 of 4,800 values moved, hpac 11,146 B
    (-1,116) against step 2's 11,629 B (-633), stream 119,685 B (+1,174) against 118,896 (+385),
    archive 179,417 B (+58) against 179,111 (-248), ΔS +3.8620e-05. ntb2 wrote that frame_quad
    "measures where the trade turns" and never ran it; it turns BETWEEN step 2 and step 4, and
    step 2 is the winner. The rounding family is closed at this resolution: coarser does not pay.
    """
    return Lever(
        "ntb2_hpac_frame_quad_rounding",
        overrides={"--hpac-frame-embedding-rounding-step": 4},
        epochs_delta=window,
        notes=(
            "DESIGN-STATE (encoder surface; flag exists on NEITHER trainer): FALSIFIED at exact "
            "bytes (+58 B) and it LOCATES the turn between step 2 and step 4. Sister of "
            "Ntb2HpacFrameEvenRounding (cons1), which LANDED as move 46 and again as move 48's "
            "composition. Do not compile into a launch."
        ),
    )


def Hpr1ConvADilationRung(window: int = 100) -> Lever:  # DESIGN-STATE record
    """hpr1 R6: dilate the HPAC prior's ``conv_a`` cone. DESIGNED, COSTED, NEVER FIRED.

    Registered UNFIRED deliberately. This is the one lever in this module with NO verdict, and
    that is the state the activation ledger should carry: it stays in never_fired() and in
    duty_to_measure() until someone prices it. hpr1 costed it as the CHEAPEST unfired rung --
    conv_a reaches its taps through geometry-general code, so a spatial rung needs ONE receiver
    edit (set model.conv_a.dilation in cpr1/inflate.py) and ships no shape bit; conv_a alone is
    50.5 % of the prior's 20,416 values. The pad-3 routine in cpr1/hpac_integer_sparse._conv_a is
    DEAD in the shipped path, which is what makes the single edit sufficient.

    What it must be measured against: the RETRAINED control (move 47/48's prior through the same
    rail at Delta 0), never an older pointer -- hpr1's R1 confound is exactly that mistake, and
    R1 measured +812 B against its own control.
    """
    return Lever(
        "hpr1_conv_a_cone_dilation",
        overrides={"--hpac-conv-a-dilation": 2},
        epochs_delta=window,
        notes=(
            "DESIGN-STATE (coder surface; flag exists on NEITHER trainer): DESIGNED and COSTED, "
            "NEVER FIRED -- it stays in the duty-to-measure queue on purpose. Price it against "
            "the retrained control, not an older pointer. Do not compile into a launch."
        ),
    )


def Hpr1QRecalibrationRung(window: int = 100) -> Lever:  # DESIGN-STATE record
    """hpr1: re-solve the tail coder's q (KT back-off recalibration) on the refit prior.

    CLOSED NEGATIVE. Held-out recalibration gain -9,439.97 B on the refit prior against -183.53 B
    on the old one -- 51x HARDER after the refit, because the refit moved q closer to its optimum,
    not away from it. The mechanism is one number: mean stated confidence 0.99802722 against a
    mean argmax-correct rate of 0.99802843, a calibration error of -1.21e-06. The mixer is
    calibrated to six decimal places on the refit prior.

    Instrument: 117,964,800 symbols, 64 bins, two frame-parity folds of 58,982,400 each, coded
    under the other's KT table with back-off (mxo3's own cross_bits, imported), with control47
    reproducing move 47's archive at 179,359 B, Delta 0.
    """
    return Lever(
        "hpr1_q_recalibration_on_refit_prior",
        overrides={"--hpac-q-recalibrate": True},
        epochs_delta=window,
        notes=(
            "DESIGN-STATE (coder surface; flag exists on NEITHER trainer): CLOSED NEGATIVE on the "
            "refit prior (-9,439.97 B held-out, 51x harder than on the old prior). A refit that "
            "improves calibration makes its recalibration rung WORSE, not better. Do not compile "
            "into a launch."
        ),
    )


def Tmx1TailMixerRefit(window: int = 100) -> Lever:  # DESIGN-STATE record
    """tmx1: refit the tail coder's 40 counted int8 mixer weights on the current field.

    CLOSED NEGATIVE at exact bytes. Six 600-frame encodes with the real coder: +203 and +22 B on
    move 47, +155 and +20 B on move 48, against a fire bar of -30.04 B. The full-sample fits gain
    -90.2 +- 25.4 B and -98.2 +- 23.9 B in sample at |t| > 3.5 and lose about +55 B on every one
    of four held-out folds; the field takes back 112 B and 118 B of the fitted gain.

    The rider's 60 B length is FIXED by the shipped schema, so Delta model = 0 and Delta B is the
    tail alone -- which is why this is the refit-debt law's NEGATIVE instance rather than a
    capacity rung. Not closed by this: a section with more fitted capacity (the `semantic`
    member's 29,862 B is the queued candidate), or a different coder geometry.
    """
    return Lever(
        "tmx1_tail_mixer_refit",
        overrides={"--tail-mixer-refit": True},
        epochs_delta=window,
        notes=(
            "DESIGN-STATE (coder surface; flag exists on NEITHER trainer): CLOSED NEGATIVE, DO "
            "NOT FIRE (+20 B best of four fits). Laws: hpr1_counted_section_refit_debt_v1 (this "
            "is its n=2 negative anchor) + refit_rungs_ranked_by_conditioned_mass_v1. Do not "
            "compile into a launch."
        ),
    )


def Rbf1PostRenderBoundaryTreatment(window: int = 100) -> Lever:  # DESIGN-STATE record
    """rbf1: free post-render boundary treatments on the rendered frames (guided / ssaa / sdf).

    CLOSED at formulation scope on two n600 channels. Every treatment is a free pixel edit that
    costs zero archive bytes and still loses, because a post-render edit pays a QUADRATIC pose tax
    -- Delta S_pose ~= 6.6e-3 * tau^2, constant to 5 % over tau = 1..8 at n = 24 -- that symmetric
    application does not cure. The seg side is worse than the pose side suggests: the treatments touch 2,287,200
    token-edge pixels at n600 to reach the 12,196 that are wrong -- a 187.5:1 collateral
    exposure.

    Not closed by this: an actuator that changes the RENDER rather than its output. The render is
    already pose-optimal for its own field, which is the positive statement hiding inside this
    negative.
    """
    return Lever(
        "rbf1_post_render_boundary_treatment",
        overrides={"--post-render-boundary-treatment": "sdf"},
        epochs_delta=window,
        notes=(
            "DESIGN-STATE (renderer-output surface; flag exists on NEITHER trainer): CLOSED at "
            "formulation scope on two n600 channels. Law: rbf1_post_render_pose_amplitude_tax_v1 "
            "(amplitude clause, n=24; the f^1.16 support clause is EXCLUDED and still owed). Do "
            "not compile into a launch."
        ),
    )


CONS3_CLOSED_DESIGN_LEVERS = (
    Pc3Cap1PredictorRefit,
    Ntb2HpacFrameQuadRounding,
    Hpr1QRecalibrationRung,
    Tmx1TailMixerRefit,
    Rbf1PostRenderBoundaryTreatment,
)

#: Registered with a factory but deliberately WITHOUT a verdict: it must stay in the
#: duty-to-measure queue until a real encode prices it. Kept separate from the tuple above so a
#: reader cannot mistake "held by the DSL" for "decided".
CONS3_UNFIRED_DESIGN_LEVERS = (Hpr1ConvADilationRung,)
