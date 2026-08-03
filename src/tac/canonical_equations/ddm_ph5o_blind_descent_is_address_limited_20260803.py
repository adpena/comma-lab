# SPDX-License-Identifier: MIT
"""ddm_ph5o — the blind subspace CAN be aimed, and the descent is ADDRESS-LIMITED.

WHAT THIS REGISTERS.  ``ddm_ph4`` proved the 230,904 ``D``-blind camera pixels
are an exactly seg-free actuator with 692,712 dimensions per pair, and left
O1 open: *capacity is not alignment*.  O1 is now measured, and the answer has
two halves that point opposite ways:

  * **ALIGNMENT: YES.**  The blind subspace contains a real descent direction
    for ``d_pose``.  Measured on the live ``pu2`` base through the shipped
    receiver and the frozen CPU-torch PoseNet, the descent/ascent split at
    matched support is strongly ASYMMETRIC and in the gradient's favour --
    which a single-sign sweep could not have established.
  * **RATE: NO.**  The descent direction is a handful of ISOLATED PIXELS
    chosen by a scorer-derived saliency.  Its address IS its information, and
    the address is per-pair private (measured pairwise Jaccard 0.0056).  The
    byte cost of naming the pixels exceeds the pose gain they buy by roughly
    an order of magnitude at every support size measured.

THE STRUCTURAL REASON THE CHEAP FORM CANNOT WORK.  A rank-k correction over a
GENERIC deterministically-generated basis is free in ``inflate.py`` (rule 118)
and costs only k coefficients per pair.  Measured with a rank-6 separable-DCT
basis restricted to the blind mask: **100% of pairs solved to the all-zero
integer coefficient vector.**  That is not a solver failure -- it is an inner
product.  The useful direction is a ~12-pixel spike; a smooth low-frequency
basis is maximally delocalised; their overlap is negligible.  **The cheapness
of a generic basis and the localisation of the descent are the same property
with opposite signs.**

WHY IT IS NOT A TRUST-REGION ARTEFACT (the discriminator that had to be run).
A full-field +-1 LSB sign step -- the smallest amplitude the uint8 camera
raster can express, applied to all 230,904 blind pixels at once -- RAISES
``d_pose`` by 4.26e-01, which is 153x the entire first-order drop the measured
gradient promises (``sum|g| = 2.784e-03``) and 4.5x the base ``d_pose`` itself.
A linearisation whose promised drop exceeds the whole objective is outside its
trust region, so "misaligned" was NOT established by that step.  Sweeping the
SUPPORT at fixed +-1 LSB quantum -- the only remaining free parameter once the
quantum is pinned -- recovers the descent window and settles the question.

WHAT IT REFUTES.  My own first pass, twice.  (1) I gated the forward-warp
parity control at bit-identity and it fired: the re-associated bilinear sum
agrees to 1.5 ULP at magnitude 255, not to the bit; the leg that is actually
load-bearing (the uint8 frame the receiver emits) IS bit-identical at 0.0.
(2) My first solver reported the BASE value whenever nothing improved, making
"stepped and got worse" and "never stepped at all" the same symbol -- memory
``m50`` inside my own instrument.  Both are recorded here rather than smoothed,
because a registry that hides its author's near-misses teaches nothing.

WHAT IT DOES NOT CLAIM.  This prices the actuator with the shipped pose knobs
held FIXED, which is the realised receiver chain (the archive carries those
knobs and the correction is applied to frame_1 before the warp).  A JOINT
re-solve of the 11 pose knobs together with the blind field is not measured
here and could differ; it would not change the addressing arithmetic, which is
combinatorial rather than statistical.

axis: [macOS-CPU advisory] NON-PROMOTABLE.  score_claim=false.  Pointer UNMOVED.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "ADDRESS_BITS_PER_PIXEL",
    "BLIND_PX",
    "DESCENT_BY_SUPPORT_N600",
    "GENERIC_BASIS_RANK6_ZERO_COEFF_FRACTION",
    "MEASURED_SUPPORT_JACCARD",
    "SHARED_ADDRESS_JACCARD_FLOOR",
    "break_even_d_pose_cut",
    "descent_support_byte_cost",
    "generic_basis_can_carry_descent",
    "joint_delta_s_of_support",
    "shared_address_pricing_is_admissible",
]

#: ``ddm_ph4`` C1 / ``ddm_rz1`` R1, re-verified by ``ddm_ph5o`` C1.
BLIND_PX = 230904

#: The decoder cannot run PoseNet (CLAUDE.md "no scorers at inflate time"), so a
#: correction that names individual blind pixels must carry BOTH the address and
#: its sign.  ``log2(230904) = 17.817`` bits, plus 1 sign bit.  This is a
#: COMBINATORIAL cost, not an estimate.
ADDRESS_BITS_PER_PIXEL = math.log2(BLIND_PX) + 1.0

#: MEASURED, n600, live ``pu2`` base (archive sha ``c72ef357``), through the
#: shipped receiver + frozen CPU-torch PoseNet.  Maps support size (number of
#: blind pixels moved by exactly one LSB along ``-sign(g)``) to the relative
#: change in mean ``d_pose``.  Negative is an improvement.
#: Populated by ``experiments/ddm_ph5o_blind_pose_sparsity.py``.
DESCENT_BY_SUPPORT_N600: dict[int, float] = {
    1: -0.002704,
    2: -0.004911,
    5: -0.008112,
    12: -0.012181,
    27: -0.006525,
    61: +0.046591,
}

#: The descent WINDOW closes: the curve turns back up between 12 and 27 and is
#: positive by 61.  The optimum sits at a support of order ten pixels out of
#: 230,904 -- which is exactly why the address dominates the price.
BEST_FIXED_SUPPORT_N600 = 12
#: Per-pair ORACLE support selection (the upper bound on this actuator, and not
#: itself shippable without paying for the selection): d_pose -5.9177%, 92.7% of
#: pairs improved, mean support 13.68 px/pair, 19,514 B, joint +0.009260.
ORACLE_RELATIVE_D_POSE_N600 = -0.059177
ORACLE_MEAN_SUPPORT_PX = 13.68
#: MINIMUM measured byte-cut factor over every configuration tried: the oracle
#: needs 3.5x cheaper addressing, the best FIXED support needs 5.6x.
MIN_BYTE_CUT_FACTOR_NEEDED = 3.5

#: MEASURED sign asymmetry at matched support -- the positive control on the
#: AIMING itself.  If the response were purely second order the two signs would
#: be symmetric and the gradient would carry no information.  They are not.
ASCENT_BY_SUPPORT_N600: dict[int, float] = {
    1: +0.002542,
    2: +0.005510,
    5: +0.014166,
    12: +0.035889,
    27: +0.086051,
    61: +0.224014,
}

#: MEASURED mean pairwise Jaccard overlap of the top-12 descent supports across
#: 16 strided pairs.  ~0 means the support is PER-PAIR PRIVATE: 180 distinct
#: pixels of a possible 192, only 10 chosen by more than one pair, and a global
#: consensus set covers just 12.5% of any pair's own support.  This REFUTES the
#: one escape that would have changed the arithmetic by ~19x (ship the address
#: table once, ship only per-pair signs).
MEASURED_SUPPORT_JACCARD = 0.0056

#: The overlap a shared-address claim must MEASURE before it may be priced.
SHARED_ADDRESS_JACCARD_FLOOR = 0.5

#: MEASURED: fraction of pairs whose rank-6 generic separable-DCT solve landed
#: on the all-zero INTEGER coefficient vector (1 B/coefficient quantisation).
GENERIC_BASIS_RANK6_ZERO_COEFF_FRACTION = 1.0

#: The live operating point every Delta-S below is named against -- a Delta-S
#: without its baseline is unanchored and baselines move (memory ``m46``).
#: ``ddm_pu2`` report.txt, n600: d_seg 0.00431179, d_pose 0.00154519,
#: 353,805 B.  Gap decomposition against the PR130 bar 0.1721413 computed by
#: ``gap_decomposition_against_floor_20260802``.
LIVE_POSE_CONTRIBUTION = 0.1243057
LIVE_TOTAL_GAP = 0.6189276
LIVE_POSE_GAP = 0.1090381
RATE_DENOMINATOR_BYTES = 37545489
N_PAIRS = 600


def descent_support_byte_cost(n_support: int, *, n_pairs: int = N_PAIRS,
                              shared_addresses: bool = False,
                              measured_jaccard: float | None = None) -> float:
    """Bytes an explicit blind-pixel correction of ``n_support`` px/pair costs.

    ``shared_addresses=True`` prices the cheap variant (one address table for
    the whole video, per-pair signs only) and REFUSES unless a measured overlap
    at or above :data:`SHARED_ADDRESS_JACCARD_FLOOR` is supplied.  The escape is
    not available on this vehicle -- measured 0.0056 -- and this function makes
    that un-assumable rather than merely written down.
    """
    if isinstance(n_support, bool) or not isinstance(n_support, int):
        raise TypeError("n_support must be an int")
    if n_support < 0:
        raise ValueError("n_support must be non-negative")
    if n_support > BLIND_PX:
        raise ValueError(
            f"n_support {n_support} exceeds the blind set ({BLIND_PX} px)")
    if isinstance(n_pairs, bool) or not isinstance(n_pairs, int) or n_pairs <= 0:
        raise ValueError("n_pairs must be a positive int")
    if not shared_addresses:
        return n_support * ADDRESS_BITS_PER_PIXEL * n_pairs / 8.0
    if measured_jaccard is None:
        raise ValueError(
            "shared_addresses=True requires measured_jaccard: the shared-table "
            "price may not be assumed.  MEASURED on this vehicle: "
            f"{MEASURED_SUPPORT_JACCARD} (top-12 supports, 16 strided pairs), "
            "which is ~0 -- the support is per-pair private.")
    if not math.isfinite(measured_jaccard) or not 0.0 <= measured_jaccard <= 1.0:
        raise ValueError("measured_jaccard must be a finite overlap in [0,1]")
    if measured_jaccard < SHARED_ADDRESS_JACCARD_FLOOR:
        raise ValueError(
            f"measured support overlap {measured_jaccard} is below the "
            f"{SHARED_ADDRESS_JACCARD_FLOOR} floor; a shared address table "
            "cannot reproduce per-pair supports that do not coincide")
    addresses = n_support * math.log2(BLIND_PX) / 8.0
    signs = n_support * n_pairs / 8.0
    return addresses + signs


def break_even_d_pose_cut(bytes_added: float, *,
                          pose_contribution: float = LIVE_POSE_CONTRIBUTION,
                          rate_denominator: int = RATE_DENOMINATOR_BYTES
                          ) -> float:
    """Relative ``d_pose`` cut a correction must deliver to pay for its bytes.

    ``S`` carries pose as ``sqrt(10*d_pose)``, so a relative cut ``r`` moves the
    pose term by ``pose*(sqrt(1-r) - 1)``.  Setting that equal and opposite to
    the rate cost gives the requirement.  Returns a POSITIVE fraction.
    """
    if not math.isfinite(bytes_added) or bytes_added < 0:
        raise ValueError("bytes_added must be finite and non-negative")
    if not math.isfinite(pose_contribution) or pose_contribution <= 0.0:
        raise ValueError("pose_contribution must be finite and positive")
    if rate_denominator <= 0:
        raise ValueError("rate_denominator must be positive")
    ds_rate = 25.0 * bytes_added / rate_denominator
    if ds_rate >= pose_contribution:
        raise ValueError(
            "the rate cost exceeds the entire pose contribution; no cut to "
            "d_pose can pay for it")
    return 1.0 - ((pose_contribution - ds_rate) / pose_contribution) ** 2


def joint_delta_s_of_support(n_support: int, *,
                             relative_d_pose_change: float | None = None,
                             shared_addresses: bool = False,
                             measured_jaccard: float | None = None
                             ) -> dict[str, Any]:
    """Full joint Delta-S for an explicit blind correction at ``n_support``.

    Uses the MEASURED n600 descent when ``relative_d_pose_change`` is omitted
    and refuses supports it has not measured, so the curve cannot be silently
    extrapolated past its evidence.
    """
    if relative_d_pose_change is None:
        if n_support not in DESCENT_BY_SUPPORT_N600:
            raise ValueError(
                f"support {n_support} was not measured; measured supports are "
                f"{sorted(DESCENT_BY_SUPPORT_N600)}.  Supply "
                "relative_d_pose_change explicitly to price an unmeasured one, "
                "and label it as such.")
        relative_d_pose_change = DESCENT_BY_SUPPORT_N600[n_support]
    if not math.isfinite(relative_d_pose_change) or relative_d_pose_change < -1.0:
        raise ValueError(
            "relative_d_pose_change must be finite and >= -1.0 (a -1.0 cut is "
            "d_pose driven to exactly zero; anything below is not a quantity)")
    b_tot = descent_support_byte_cost(
        n_support, shared_addresses=shared_addresses,
        measured_jaccard=measured_jaccard)
    ds_pose = LIVE_POSE_CONTRIBUTION * (
        math.sqrt(max(1.0 + relative_d_pose_change, 0.0)) - 1.0)
    ds_rate = 25.0 * b_tot / RATE_DENOMINATOR_BYTES
    joint = ds_pose + ds_rate
    return {
        "support": n_support,
        "total_bytes": b_tot,
        "delta_S_pose": ds_pose,
        "delta_S_rate": ds_rate,
        "delta_S_joint": joint,
        "pct_of_total_gap": 100.0 * joint / LIVE_TOTAL_GAP,
        "byte_cut_factor_needed": (ds_rate / -ds_pose) if ds_pose < 0 else None,
        "pays_for_itself": bool(joint < 0.0),
        "baseline": "ddm_pu2 n600, archive sha c72ef357, 353,805 B",
    }


def generic_basis_can_carry_descent(*, rank: int, basis_family: str,
                                    measured_zero_coeff_fraction: float | None
                                    = None) -> dict[str, Any]:
    """Gate a "a free generic basis carries the correction" claim.

    The rate-attractive form of this actuator is a rank-k correction over a
    deterministically generated basis, which is free in ``inflate.py``.  It was
    MEASURED and it is empty: a rank-6 separable-DCT basis on the blind mask
    solved to the all-zero integer coefficient vector on 100% of pairs.  The
    mechanism generalises to any smooth low-frequency family -- the descent is a
    ~12-pixel spike and smooth bases are delocalised -- so this refuses the
    claim for smooth families absent a fresh measurement.
    """
    smooth = {"dct", "cosine", "fourier", "polynomial", "spherical_harmonic",
              "low_frequency_separable"}
    frac = measured_zero_coeff_fraction
    if basis_family.lower() in smooth and frac is None:
        frac = GENERIC_BASIS_RANK6_ZERO_COEFF_FRACTION
    if frac is None:
        raise ValueError(
            f"basis family {basis_family!r} has no measurement on this "
            "vehicle; supply measured_zero_coeff_fraction from a real solve "
            "rather than asserting representability")
    if not math.isfinite(frac) or not 0.0 <= frac <= 1.0:
        raise ValueError("measured_zero_coeff_fraction must be a fraction in [0,1]")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive int")
    return {
        "rank": rank,
        "basis_family": basis_family,
        "measured_zero_coeff_fraction": frac,
        "carries_descent": bool(frac < 1.0),
        "note": ("the blind descent direction is a ~12-pixel spike; a smooth "
                 "delocalised basis has negligible overlap with it, which is "
                 "why the free form is empty and the paid form is expensive"),
    }


def shared_address_pricing_is_admissible(measured_jaccard: float) -> bool:
    """Whether a shared blind-address table may be priced on this vehicle."""
    if not math.isfinite(measured_jaccard) or not 0.0 <= measured_jaccard <= 1.0:
        raise ValueError("measured_jaccard must be an overlap in [0,1]")
    return bool(measured_jaccard >= SHARED_ADDRESS_JACCARD_FLOOR)
