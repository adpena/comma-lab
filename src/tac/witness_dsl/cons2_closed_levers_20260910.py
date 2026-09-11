# SPDX-License-Identifier: MIT
"""DSL Lever records for the eight 2026-09-10 levers that reached a verdict (ddm_cons2).

WHY THIS MODULE EXISTS.  The DSL-holds-every-designed-lever law: a lever with a verdict that no
``Lever`` factory holds is orphaned signal -- the activation ledger cannot nag it, rank it, or
retire it, so the next arm rediscovers it.  Eight levers were designed and decided on 2026-09-10
and none of them had a factory.  Seven of the eight are NEGATIVE, which is exactly the class that
gets rediscovered, because a negative leaves no artifact anyone trips over.

WHAT THESE FLAGS ARE, STATED SO NOBODY IS MISLED.  **None of the eight flags below exists on either
trainer** (MEASURED: absent from the levelset trainer's argparse and from the TR1 renderer's).  Most
are not trainer levers at all -- they act on the GENERATOR, the CODER, the RECEIVER and the ENCODER,
surfaces no trainer argparse owns, which is why ``lever_registry.completeness().unmapped`` (a
TRAINER-FLAG surface) never listed them and never could.  These are DESIGN-STATE records in the ax1
stub tradition, carrying each verdict and its scope.  They must NOT be compiled into a launch.

THE COMMON DOOR.  Four of them (gdc1-gdc4) were measured against ONE pre-derived gate: at move 43's
held distortion the strict archive cap is 154,507 B, the live rate demand 25,959 B, and after the
119,969 B replaceable token-tail envelope the generator-plus-correction door is **94,010 B**.  Every
generator family missed it, and by margins that are not close:

    family                          best full-n600 total   over the 94,010 B door
    gdc1 ordered scanline (K=6)          329,122 B              235,112 B  (3.50x)
    gdc3 anisotropic key-row ribbon      453,163 B              359,153 B  (4.82x)
    gdc2 categorical Cool-Chic K=8       540,681 B              446,671 B  (5.75x)
    gdc4 run-native endpoint gen.        (screens falsified)               (2.50x best possible)

The other four measured the tail from four directions and all landed in the same place: the
receiver-visible tail on this object is closed at the measured level.

Pointer honesty: ddm_cons2 measured nothing and moved nothing.  Every number here is quoted from the
arm that measured it.  Laws: ``lane_surprise_atlas_oracle_ladder_v1`` (ls1/ls2),
``lane_boundary_context_map_bound_v1`` (tc2-tc4), ``boundary_segment_recode_price_v1`` (bnd2/bnd3).
"""

from __future__ import annotations

from tac.witness_dsl.curriculum_dsl import Lever

# Declared so the registry attributes this module's debt to the vehicle we ship rather than
# silently defaulting it to the retired levelset trainer. Building any of these levers is NOT
# implied: their flags exist on NEITHER trainer (MEASURED), and most act on the
# generator/coder/receiver/encoder, which no trainer argparse owns.
TRAINER_RELPATH = "experiments/train_tr1_partition_renderer_mlx.py"

# MEASURED (gdc1): the door every generator family was priced against, re-derived at move 43.
#
# The demand here (25,959 B) and ls1/ls2's (25,899 B) are NOT a contradiction and should not be
# harmonized: the strict sub-0.12 archive cap is 154,507 B in both, and the demand is just
# archive - cap at whichever move is live. Move 43 shipped 180,466 B (180,466 - 154,507 = 25,959);
# move 44 shipped 180,406 B (180,406 - 154,507 = 25,899). A demand figure is only ever true of one
# pointer move, which is the same expiry the rung-price law records on another surface.
GENERATOR_DOOR_BYTES = 94_010
STRICT_ARCHIVE_CAP_BYTES = 154_507
RATE_DEMAND_BYTES = 25_959
MOVE43_ARCHIVE_BYTES = 180_466
MOVE44_ARCHIVE_BYTES = 180_406


def Gdc1OrderedScanlineProgram(window: int = 100) -> Lever:  # DESIGN-STATE record
    """gdc1: a no-training ordered-scanline partition program with a coded residual.

    FORMULATION-NO-GO. Best full-n600 point K=6: 223,494 B program + 105,628 B real coded
    residual = 329,122 B, which is 235,112 B over the 94,010 B door. The arm's lasting positive
    is the DOOR itself -- it re-derived the stale 137,986 B cap to a strict 154,507 B archive cap
    and a 25,959 B rate demand, which every later generator family was then priced against.
    Reactivates only for a program whose FIELD DESCRIPTION intercept is smaller, not for a
    better residual coder on this one.
    """
    return Lever(
        "gdc1_ordered_scanline_program",
        overrides={"--gdc1-ordered-scanline-k": 6},
        epochs_delta=window,
        notes=(
            "DESIGN-STATE (generator surface; flag exists on NEITHER trainer): FORMULATION-NO-GO, "
            "329,122 B vs the 94,010 B door. Do not compile into a launch."
        ),
    )


def Gdc2CategoricalCoolChicDistill(window: int = 100) -> Lever:  # DESIGN-STATE record
    """gdc2: categorical Cool-Chic distillation of the retained K=8 scanline teacher.

    FORMULATION-NO-GO -- the worst of the four doors. Best construction packet + R_exact =
    540,681 B against the 94,010 B gate, 5.75x over. Scope: the GDC1 governed Cool-Chic form at
    the specified latent budget; a different latent budget or a non-categorical head is not
    closed by this row.
    """
    return Lever(
        "gdc2_categorical_coolchic_k8_distill",
        overrides={"--gdc2-coolchic-latent-budget": "governed_v1"},
        epochs_delta=window,
        notes=(
            "DESIGN-STATE (generator surface; flag exists on NEITHER trainer): FORMULATION-NO-GO, "
            "540,681 B = 5.75x the door. Do not compile into a launch."
        ),
    )


def Gdc3AnisotropicKeyRowRibbon(window: int = 100) -> Lever:  # DESIGN-STATE record
    """gdc3: a fixed anisotropic key-row ribbon with zero-order-hold reconstruction.

    FORMULATION-NO-GO, and instructive about WHY. Best of four full-n600 schedules (uniform2):
    227,551 B packet + 225,612 B exact residual = 453,163 B, 4.820370x the door; the PACKET ALONE
    is 2.4205x the whole door. The falsifier did exactly what its geometry claimed -- 100 % of its
    errors were boundary-adjacent, the cheapest measured residual class -- and the field
    description intercept dominated anyway. That is the transferable finding: on this object the
    intercept, not the error geometry, is what closes a generator.
    """
    return Lever(
        "gdc3_anisotropic_keyrow_ribbon",
        overrides={"--gdc3-keyrow-schedule": "uniform2"},
        epochs_delta=window,
        notes=(
            "DESIGN-STATE (generator surface; flag exists on NEITHER trainer): FORMULATION-NO-GO, "
            "453,163 B = 4.82x the door; packet alone 2.42x. Do not compile into a launch."
        ),
    )


def Gdc4RunNativeEndpointGenerator(window: int = 100) -> Lever:  # DESIGN-STATE record
    """gdc4: a learned run-native endpoint generator, refused BEFORE its burn by its own arithmetic.

    FORMULATION-NO-GO. All three pre-registered early-stop screens are falsified on the real n600
    field -- packet <= 60,000 B, mismatches <= 65,000, and >= 90 % of errors long-and-boundary-
    adjacent -- and the family's BEST POSSIBLE member misses the 94,010 B door by 2.50x. The
    refusal is the point: a burn that its own screens forbid is not a measurement anyone owes.
    Correction it carries: gdc4 reported COMPUTED code lengths, never serialized prices.
    """
    return Lever(
        "gdc4_run_native_endpoint_generator",
        overrides={"--gdc4-run-native-endpoints": True},
        epochs_delta=window,
        notes=(
            "DESIGN-STATE (generator surface; flag exists on NEITHER trainer): FORMULATION-NO-GO, "
            "refused pre-burn; best possible member 2.50x over the door. Do not compile into a launch."
        ),
    )


def Ls1LaneConditionedSurpriseAtlas(window: int = 100) -> Lever:  # DESIGN-STATE record
    """ls1/ls2: a Lane-conditioned context correction over the shipped token field.

    CLOSED at the MEASURED level for every receiver-visible information set. The best oracle a
    decoder could actually compute falls 8,364.784 B short of the 25,899 B demand WITH ITS TABLES
    FREE; the charged full-resolution realisation returns 385.550480 B, 1.49 % of the demand. The
    only rung that comes near (1,035.362 B short) uses previous-row geometry the receiver does not
    have. ls2 refused its own fire on TIMING, not on gain -- a native receiver with a measured
    decode wall-clock inside the 1260 s gate is the one leg that reopens it.
    Law: lane_surprise_atlas_oracle_ladder_v1.
    """
    return Lever(
        "ls1_lane_conditioned_surprise_correction",
        overrides={"--ls2-lane-probability-correction": "joint_distance_plus_phase"},
        epochs_delta=window,
        notes=(
            "DESIGN-STATE (coder surface; flag exists on NEITHER trainer): CLOSED at the measured "
            "level; 386 B realized vs a 25,899 B demand. Law: lane_surprise_atlas_oracle_ladder_v1. "
            "Do not compile into a launch."
        ),
    )


def Mxo1FreeDecodeTimeContextMixer(window: int = 100) -> Lever:  # DESIGN-STATE record
    """mxo1: free decode-time ONLINE context mixing layered over the shipped prior.

    CLOSED BY THRESHOLD, not by failure. The best of three original post-shipping-prior learners
    saved 368 realized RC64 bytes on the full n600 stream -- real bytes, and 1.42 % of the
    25,899 B demand, under the charter's own 3,000 B receiver-delta fire threshold and slightly
    under ls2's 386 B linear screen. Two independent families of online corrector therefore land
    within 5 % of each other at ~1.4 % of the demand, which is the useful cross-check: the ceiling
    is the OBJECT's, not any one learner's.
    """
    return Lever(
        "mxo1_online_context_mixer",
        overrides={"--mxo1-online-mixer": True},
        epochs_delta=window,
        notes=(
            "DESIGN-STATE (receiver surface; flag exists on NEITHER trainer): CLOSED by threshold, "
            "368 B realized vs a 3,000 B trigger. Do not compile into a launch."
        ),
    )


def Tc2LaneBoundaryContextMap(window: int = 100) -> Lever:  # DESIGN-STATE record
    """tc2-tc4: a lane-boundary DISTANCE context map for the shipped tail mixer.

    CLOSED, with a correction that travelled further than the lever. The map buys 5,480 B with the
    GT edge as ORACLE (of a ~30 KB Lane->Road attribution) and 78 B with a real causal predictor:
    the boundary tokens' surprise at cell precision is near-inherent given every context tried.
    tc4's shipped form ALSO timed out at 1800 s on T4, which is the sister lesson -- a receiver
    change needs a measured decode wall-clock. The correction: tc2's "5.5 KB" is ORACLE GAIN, never
    a map's serialized cost. Law: lane_boundary_context_map_bound_v1.
    """
    return Lever(
        "tc2_lane_boundary_context_map",
        overrides={"--tc2-lane-distance-context": True},
        epochs_delta=window,
        notes=(
            "DESIGN-STATE (coder surface; flag exists on NEITHER trainer): CLOSED -- 5,480 B oracle, "
            "78 B causal; tc4's form timed out at 1800 s on T4. Law: "
            "lane_boundary_context_map_bound_v1. Do not compile into a launch."
        ),
    )


def Bnd2BoundarySegmentRecode(window: int = 100) -> Lever:  # DESIGN-STATE record
    """bnd2/bnd3: a boundary-SEGMENT recode of the shipped field's mispredicted tokens.

    CLOSED NEGATIVE on every real n600 variant: +1,416 B at best against a 119,784 B envelope,
    because the median boundary segment is ONE CELL, so addresses cost more than the tokens they
    address. Masking the 235,044 mispredicted tokens GROWS the stream by +9,532 B -- the direct
    demonstration that an attribution is not a detachable payload. bnd3 decomposed the address
    term and reached the same wall from the other side.
    Law: boundary_segment_recode_price_v1.
    """
    return Lever(
        "bnd2_boundary_segment_recode",
        overrides={"--bnd2-segment-recode": True},
        epochs_delta=window,
        notes=(
            "DESIGN-STATE (encoder surface; flag exists on NEITHER trainer): CLOSED NEGATIVE, "
            "+1,416 B best; masking GROWS the stream +9,532 B. Law: "
            "boundary_segment_recode_price_v1. Do not compile into a launch."
        ),
    )


CONS2_CLOSED_DESIGN_LEVERS = (
    Gdc1OrderedScanlineProgram,
    Gdc2CategoricalCoolChicDistill,
    Gdc3AnisotropicKeyRowRibbon,
    Gdc4RunNativeEndpointGenerator,
    Ls1LaneConditionedSurpriseAtlas,
    Mxo1FreeDecodeTimeContextMixer,
    Tc2LaneBoundaryContextMap,
    Bnd2BoundarySegmentRecode,
)
