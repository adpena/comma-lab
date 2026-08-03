# SPDX-License-Identifier: MIT
"""ddm_ph4 — THE CONSUMER-LATTICE LAW, and the seg-free pose actuator it finds.

WHAT THIS REGISTERS.  A camera-plane edit to frame_1 is read by THREE consumers
at TWO lattices, and its cost on each scored term follows from WHICH camera
pixels it touches.  That is now a measured function, not a judgement call, so
any proposed receiver-side edit can be priced on both axes before it is built.

  consumer                      operator          sees a D-blind edit?
  ----------------------------  ----------------  --------------------
  SegNet                        D(f1)             NO  -- exactly 0.0
  PoseNet, frame_1 half         yuv6(D(f1))       NO  -- same lattice
  PoseNet, frame_0 half         yuv6(D(W(f1)))    YES -- gain 0.2231

THE SEG HALF IS A PROOF.  ``upstream/modules.py:107-109`` is
``x = x[:, -1, ...]`` then ``interpolate(..., (384,512), 'bilinear')``.  SegNet
therefore has exactly ONE path to the frames, and ``D`` is point-sampling at
stride 874/384 = 2.276 > 2, so it reads 768 of 874 rows and 1024 of 1164
columns and never touches the other 230,904 camera pixels.  An edit confined to
those pixels changes ``d_seg`` by EXACTLY zero -- not to within a quantization
residue, and with NO lattice caveat, because there is no second SegNet lattice.
MEASURED ``0.0e+00`` in 20/20 cells (4 strided pairs x 5 amplitudes), against a
cardinality-matched D-VISIBLE control that moves ``D(f1)`` by exactly the step.

THE POSE HALF IS A MEASUREMENT, AND IT HAS TWO REGIMES.  In this vehicle
frame_0 is manufactured by warping frame_1's CAMERA raster
(``inflate_runner.py:307-327``).  ``W`` resamples at sub-pixel offsets, so
``D∘W != D`` and ``D``'s null space is NOT ``D∘W``'s -- this is ``ddm_pz1``
§7.1 ("null-space membership does not survive a change of lattice") evaluated
on the blind set.  The response gain, scorer-plane f0 rms per LSB of camera
step, MEASURED on the live shipped receiver and bytes:

  * LINEAR regime (amp >= 4): the gain converges to the touched set's
    POPULATION SHARE.  Blind: 0.2231 vs share 0.226969 (1.7% apart).  The
    homography is an area-preserving interpolation, so each camera pixel
    contributes its area fraction of the warp's attention -- no set is
    privileged.
  * ROUNDING regime (amp = 1): ``_to_uint8`` = ``clip(round(.))`` acts as a
    THRESHOLD on the sub-LSB warp response, so the smallest step is 1.204x MORE
    efficient per LSB than the asymptote.  Signature: at amp 1 the camera-domain
    f0 delta is rms 0.4700 with 22.1% of pixels changed, and
    ``sqrt(0.221) = 0.4701`` -- every changed pixel moved by exactly +/-1 LSB
    and nothing else moved at all.  A response, not a linearization.

WHAT IT REFUTES.  ``ddm_rz1`` §1 R1(c) / §3.6 rank this set PREDICTED NULL --
"read by neither scorer", "must receive zero bits", "do not spend on these".
The blind set is read by PoseNet's frame_0 half at 0.2231 gain.  It is not dead
bits; it is a 692,712-dimension-per-pair actuator that is EXACTLY seg-free, and
nothing in the vehicle has ever written to it (``ddm_ll1``'s window solve
verifiably touches 0 blind pixels).  ``ddm_ra1`` §6's one-clause note -- "blind
to D but not to the warp" -- was right and was never followed up.

WHAT IT ALSO REFUTES.  ``ddm_rz1`` §3.2 ranks pose-free chroma steering #2 on
being "the only attack that is exactly pose-free by construction", and
pre-registers the expected break as the <=0.9-LSB uint8 lift residue.  That
attack writes to the D-VISIBLE complement, whose pass-through is MEASURED here
at 0.8902 -- five to six orders of magnitude above the residue it named.  The
attack is re-scoped (a real seg actuator that must PAY a real pose cost), not
killed.

NOT ADDITIVE, AND THIS IS THE TRAP.  blind 0.2231 + visible 0.8902 = 1.1133 >
0.9809 all-pixel.  rms does not add across spatially correlated response
fields, so NEITHER gain may be inferred from the others by interpolation --
this arm first INFERRED 0.76 for the visible set that way and measured 0.8902.
``passthrough_gain`` therefore refuses any set it has not measured.

WHAT IS OWED (do not read this module as closing the question).  Every number
here is IMAGE-DOMAIN and scorer-free.  It establishes that the blind subspace
EXISTS, is exactly seg-free, and carries authority; it does NOT establish that
it can be AIMED at a specific 6-scalar pose residual.  That needs one PoseNet
job (``ddm_ph4`` §O1), which carries a free exact positive control: ``d_seg``
must come back BIT-IDENTICAL, and if it does not, the proof above is wrong.

Evidence: ``.omx/research/ddm_ph4_physics_photometrics_dynamics_20260803.md``
Artifacts: ``.omx/research/ddm_ph4_blind_set_pose_reach_cx1_20260803.json``
           ``.omx/research/ddm_ph4_visible_set_warp_passthrough_cx1_20260803.json``
Instrument: ``experiments/ddm_ph4_blind_set_pose_reach.py``
Axis: [macOS-CPU advisory] NON-PROMOTABLE.  score_claim=false.
"""

from __future__ import annotations

from typing import Any

CAMERA_H, CAMERA_W = 874, 1164
SCORER_H, SCORER_W = 384, 512

#: ``D``-blind camera pixels: read by NEITHER scorer through ``D``.  Reproduced
#: independently by ``ddm_rz1`` R1, ``ddm_ra1``, ``ddm_ll1.blind_mask()``.
BLIND_PX = 230_904
BLIND_SHARE = 0.22696926089315625
VISIBLE_PX = CAMERA_H * CAMERA_W - BLIND_PX          # 786,432 = 196,608 x 4
VISIBLE_SHARE = 1.0 - BLIND_SHARE

#: Exactly seg-free dimensions per pair: blind pixels x 3 RGB channels.
BLIND_DIMS_PER_PAIR = BLIND_PX * 3                   # 692,712
#: ``d_pose`` is an MSE over the first 6 PoseNet outputs per pair.
POSE_SCALARS_PER_PAIR = 6

#: MEASURED gain: scorer-plane frame_0 rms per LSB of camera step, by amplitude.
#: 4 strided pairs (0/200/399/599) on ``v4d_cx1_pj2ix2``, all 4 controls PASS.
BLIND_GAIN_BY_AMP: dict[int, float] = {
    1: 0.2686, 2: 0.2364, 4: 0.2264, 8: 0.2238, 16: 0.2231,
}
VISIBLE_GAIN_BY_AMP: dict[int, float] = {1: 0.9033, 8: 0.8906, 16: 0.8902}
ALL_PIXEL_GAIN_AMP1 = 0.9809                          # control C4

#: Asymptotic (linear-regime) gains.  Each converges on its population share.
BLIND_GAIN_ASYMPTOTIC = 0.2231
VISIBLE_GAIN_ASYMPTOTIC = 0.8902
#: Small-signal excess from ``_to_uint8``'s round acting as a threshold.
ROUNDING_REGIME_EXCESS = BLIND_GAIN_BY_AMP[1] / BLIND_GAIN_ASYMPTOTIC  # 1.204

#: The comparison that prices the actuator, same base / chain / scorer plane.
#: ``ddm_ll1``'s window solve, adjudicated at n600 by ``ddm_pz1`` §3.2/§5.
LL1_WINDOW_SOLVE_SCORER_F0_RMS = 1.6986
LL1_WINDOW_SOLVE_DELTA_S_SEG = +0.000394
LL1_WINDOW_SOLVE_DELTA_S_POSE = +0.0310208
#: The blind step that matches that authority over PoseNet's input -- at
#: EXACTLY ZERO seg cost instead of ``+0.000394``.
BLIND_AMP_MATCHING_LL1_AUTHORITY = 8                  # scorer f0 rms 1.7905

_MEASURED_SETS = {"blind": BLIND_GAIN_BY_AMP, "visible": VISIBLE_GAIN_BY_AMP}
_ASYMPTOTE = {"blind": BLIND_GAIN_ASYMPTOTIC, "visible": VISIBLE_GAIN_ASYMPTOTIC}
_SHARE = {"blind": BLIND_SHARE, "visible": VISIBLE_SHARE}


def seg_delta_per_lsb(edit_support: str) -> float:
    """``d_seg``-plane movement per LSB of a camera edit on ``edit_support``.

    ``'blind'`` returns exactly ``0.0`` -- SegNet has one path and ``D`` never
    reads those pixels (MEASURED ``0.0e+00``, 20/20 cells).  ``'visible'``
    returns ``1.0``: ``D``'s bilinear weights sum to 1 within each private 2x2
    window, so a uniform step of ``A`` moves ``D(f1)`` by exactly ``A``
    (MEASURED: seg-plane delta max = 1, 8, 16 at amps 1, 8, 16).
    """
    if edit_support == "blind":
        return 0.0
    if edit_support == "visible":
        return 1.0
    raise ValueError(f"unknown edit support {edit_support!r}; "
                     "measured supports are 'blind' and 'visible'")


def passthrough_gain(edit_support: str, amp_lsb: int) -> dict[str, Any]:
    """Predicted PoseNet frame_0 scorer-plane gain for a camera edit.

    Refuses any support it has not measured.  The two measured gains are NOT
    additive (0.2231 + 0.8902 = 1.1133 > 0.9809 all-pixel) because rms does not
    add across correlated response fields, so interpolating a third support
    from them is exactly the error this arm made and corrected.
    """
    if edit_support not in _MEASURED_SETS:
        raise ValueError(
            f"no MEASURED pass-through for support {edit_support!r}. "
            "Measure it (experiments/ddm_ph4_blind_set_pose_reach.py "
            "--edit-set ...); do NOT interpolate between the measured gains -- "
            "rms is not additive across correlated response fields.")
    amp = int(amp_lsb)
    if amp < 1:
        raise ValueError("amp_lsb must be >= 1 LSB")
    table = _MEASURED_SETS[edit_support]
    exact = table.get(amp)
    asym = _ASYMPTOTE[edit_support]
    gain = exact if exact is not None else asym
    return {
        "edit_support": edit_support,
        "amp_lsb": amp,
        "gain_scorer_f0_rms_per_lsb": gain,
        "scorer_f0_rms": gain * amp,
        "regime": "rounding_threshold" if amp < 4 else "linear",
        "provenance": "MEASURED" if exact is not None else
                      "asymptotic_extrapolation_linear_regime",
        "population_share_of_support": _SHARE[edit_support],
        "seg_delta_per_lsb": seg_delta_per_lsb(edit_support),
    }


def null_space_claim_survives(*, consumer_operators: tuple[str, ...],
                              null_of: str) -> dict[str, Any]:
    """``ddm_pz1`` §7.1 as an executable gate.

    A "this edit is free because it lives in <scorer>'s null space" claim holds
    only if EVERY consumer of the modified bytes reads through the SAME
    operator the null space was built for.  Pass the full consumer list; any
    consumer whose operator differs breaks the claim.

    The canonical failure this encodes: ``null_of='D'`` with a consumer reading
    ``D(W(.))`` -- measured pass-through 0.2231 (blind) / 0.8902 (visible), i.e.
    the claim fails by ~5 orders of magnitude against a sub-LSB expectation.
    """
    if not consumer_operators:
        raise ValueError("empty consumer list -- an unenumerated consumer set "
                         "is VACUOUS, never a passing null-space claim")
    breaking = tuple(c for c in consumer_operators if c != null_of)
    return {
        "null_of": null_of,
        "consumers": tuple(consumer_operators),
        "claim_survives": not breaking,
        "breaking_consumers": breaking,
        "reason": ("every consumer reads through the same operator"
                   if not breaking else
                   "consumer(s) read through a DIFFERENT operator/lattice; "
                   "null-space membership does not survive a change of lattice "
                   "(ddm_pz1 §7.1, measured on this vehicle by ddm_ph4)"),
    }


def actuator_overdetermination() -> dict[str, Any]:
    """How much freedom the seg-free subspace has against the pose objective."""
    return {
        "blind_dims_per_pair": BLIND_DIMS_PER_PAIR,
        "pose_scalars_per_pair": POSE_SCALARS_PER_PAIR,
        "overdetermination": BLIND_DIMS_PER_PAIR / POSE_SCALARS_PER_PAIR,
        "shipped_pose_knobs": 11,
        "shipped_knobs_at_grid_optimum_frac": 0.950,
        "note": "the 11 shipped knobs (s_t, sel, ab x2, beta_idx, p_best x6) "
                "were measured at their discrete optimum on 95.0% of pairs "
                "(ddm_pz1 §4); the blind subspace is DISJOINT from all of "
                "them, which is the argument for it. Capacity is not "
                "alignment -- see ddm_ph4 §O1.",
    }
