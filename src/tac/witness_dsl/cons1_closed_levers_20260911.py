# SPDX-License-Identifier: MIT
"""DSL Lever records for the four levers DESIGNED on 2026-09-11 that reached a verdict (ddm_cons1).

WHY THIS MODULE EXISTS.  The DSL-holds-every-designed-lever law: a lever is not "built" until it
is a ``Lever`` factory, and a lever with a verdict that no factory holds is orphaned signal the
activation ledger can never nag, rank or retire.  Four levers were designed and decided in one day
and none of them had a factory.

WHAT THESE FLAGS ARE, STATED SO NOBODY IS MISLED.  **None of the four flags below exists on either
trainer** (MEASURED: they are absent from the levelset trainer's argparse and from the TR1
renderer's).  Three of the four are not trainer levers at all -- they act on the CODER, the
CARRIER and the RECEIVER, surfaces the witness trainer does not own -- which is exactly why
``lever_registry.completeness().unmapped`` (a TRAINER-FLAG surface, 80 flags at the time of
writing) never listed them and never could.  These are DESIGN-STATE records in the ax1 stub
tradition: they carry the verdict and its scope so the activation ledger stops calling them
never-fired orphans, and they must NOT be compiled into a launch.  Each is superseded by its real
implementation on its own surface, or by nothing, because it is closed.

THE FOUR, with the verdict each carries:

  * ``Obx2LatticeHead`` -- MEASURED, then CLOSED with its object.  The lattice head itself PASSED
    MAIN's attribution rule (seg gain 0.001364974976 against a 0.0008921983506 threshold, 1.53x
    the bar) and bought 6.4x on Pose for 575 B, capturing 53.1 % of the render floor available to
    it.  The OBJECT was refused: 89.83 % of its seg error is partition-level and the render floor
    alone is 5.77x the seg ceiling.  Mechanism validated, object refused.
  * ``Hpr1ShapeRung`` -- FALSIFIED at INSTANCE scope.  One receptive-field shape rung
    (``conv_past`` dilation 2) measured +812 B against its own control at exact bytes.  The shape
    FAMILY is not closed and un-recoverability is not claimed; the rung is SUPERSEDED, which is a
    different exit.  Its own CONTROL became pointer move 47 (-887 B).
  * ``Ntb2HpacFrameEvenRounding`` -- LANDED as pointer move 46 (-245 B), then SUPERSEDED by
    hpr1's retrain on the same object: the two edits do not add.
  * ``Pc3PerDimensionRung`` -- CLOSED at FORMULATION scope by the REALIZED measurement (the bound
    could not reach this granularity): thirteen rungs realized from the shipped point, not one
    nets below zero, cheapest 4.4x short of the mean gain it needs.

Pointer honesty: ddm_cons1 measured nothing and moved nothing.  Every number here is quoted from
the arm that measured it, and each is registered as a canonical equation with its anchors --
``obx2_render_floor_admission_gate_v1``, ``obx2_pose_vs_scorer_plane_rmse_v1``,
``hpr1_counted_section_refit_debt_v1``, ``pc3_rung_price_expires_at_pointer_move_v1``.
"""

from __future__ import annotations

from tac.witness_dsl.curriculum_dsl import Lever

# Declared so the registry attributes this module's debt to the vehicle we ship rather than
# silently defaulting it to the retired levelset trainer. Building any of these levers is NOT
# implied: their flags exist on NEITHER trainer (MEASURED), and three of the four act on the
# coder/carrier/receiver, which no trainer argparse owns.
TRAINER_RELPATH = "experiments/train_tr1_partition_renderer_mlx.py"


def Obx2LatticeHead(window: int = 100) -> Lever:  # DESIGN-STATE record
    """obx2 design A: an edge-local implicit correction head on a coefficient lattice.

    MEASURED then CLOSED WITH ITS OBJECT (formulation scope). The head passed attribution
    (1.53x the bar) and bought 6.4x on Pose for 575 B, capturing 53.1% of the render floor.
    The object was refused: render floor 0.00231 = 5.77x the seg ceiling on its own.
    Reactivates only on a different GENERATOR FORM whose measured render floor is below 4.0e-4
    (pre-registered: 3.1990253844713356e-4 at pointer-like d_pose) AND whose error is SMOOTH --
    independent noise does 9.40x the Pose damage at matched scorer-plane RMSE.
    """
    return Lever(
        "obx2_lattice_head",
        overrides={"--obx2-lattice-head": True},
        epochs_delta=window,
        notes=(
            "DESIGN-STATE (flag exists on NEITHER trainer): MEASURED, then CLOSED at formulation "
            "scope with its object. Laws: obx2_render_floor_admission_gate_v1 + "
            "obx2_pose_vs_scorer_plane_rmse_v1. Do not compile into a launch."
        ),
    )


def Hpr1ShapeRung(window: int = 100) -> Lever:  # DESIGN-STATE record
    """hpr1: a receptive-field SHAPE rung on the HPAC prior (conv_past dilation 2).

    FALSIFIED at INSTANCE scope: +812 B against its own control at exact bytes; ΔS +5.41e-4.
    The shape FAMILY is not closed and un-recoverability is not claimed -- the rung is
    SUPERSEDED. Its control became pointer move 47 (-887 B, receiver unchanged).
    """
    return Lever(
        "hpr1_shape_rung_conv_past_dil2",
        overrides={"--hpac-conv-past-dilation": 2},
        epochs_delta=window,
        notes=(
            "DESIGN-STATE (coder surface; flag exists on NEITHER trainer): FALSIFIED at INSTANCE "
            "scope, superseded by its own retrain control. Law: hpr1_counted_section_refit_debt_v1. "
            "Do not compile into a launch."
        ),
    )


def Ntb2HpacFrameEvenRounding(window: int = 100) -> Lever:  # DESIGN-STATE record
    """ntb2: output-lossless even rounding of the HPAC frame-embedding prior.

    LANDED as pointer move 46 (-245 B, S 0.1369752312953257), then SUPERSEDED by hpr1's retrain
    of the same object -- the two edits do not add. Lever 2 (renderer precision cuts) closed
    NEGATIVE on every door; four doors were named as NOT opened (quantization-AWARE refit;
    per-row mixed depth; row pruning through MODE_ROW_PRUNE; GROWING the prior).
    """
    return Lever(
        "ntb2_hpac_frame_even_rounding",
        overrides={"--hpac-frame-embedding-even-rounding": True},
        epochs_delta=window,
        notes=(
            "DESIGN-STATE (encoder surface; flag exists on NEITHER trainer): LANDED move 46, then "
            "SUPERSEDED by hpr1's retrain on the same object. Do not compile into a launch."
        ),
    )


def Pc3PerDimensionRung(window: int = 100) -> Lever:  # DESIGN-STATE record
    """pc3: a per-dimension halving of the pose carrier's coefficient lattice pitch.

    CLOSED at FORMULATION scope by the REALIZED measurement, which reached a granularity the
    n600 ceiling bound could not: thirteen rungs realized from the shipped point, not one nets
    below zero, the cheapest 4.4x short of the mean gain it needs. Its price is also a moving
    target -- 54 B on move 44, 75 B on move 45 (pc3's own predictor refit).
    Not closed by this: a different carrier FORMAT, a rank change, or a different solver basin.
    """
    return Lever(
        "pc3_per_dimension_lattice_div2",
        overrides={"--pose-carrier-dim-pitch-div2": True},
        epochs_delta=window,
        notes=(
            "DESIGN-STATE (carrier surface; flag exists on NEITHER trainer): CLOSED at formulation "
            "scope. Law: pc3_rung_price_expires_at_pointer_move_v1. Do not compile into a launch."
        ),
    )


CONS1_CLOSED_DESIGN_LEVERS = (
    Obx2LatticeHead,
    Hpr1ShapeRung,
    Ntb2HpacFrameEvenRounding,
    Pc3PerDimensionRung,
)
