# SPDX-License-Identifier: MIT
"""Canonical equation: rate-credit pose affordance — how much d_pose degradation a
returned-byte credit pays for, and why the bar is SUPERLINEAR in the bytes returned.

DERIVED 2026-08-16 (closed form, exact; no measurement required to state it). The
contest score is

    S = 100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes / 37_545_489

A candidate that DELETES or SHRINKS a payload section returns dB bytes and degrades
d_pose. It is admissible iff the pose term grows by less than the rate credit:

    R(dB)      = 25 * dB / 37_545_489                       (rate credit, S units)
    POSE       = sqrt(10 * d_pose_base)                     (base pose term, S units)
    admissible <=> sqrt(10 * d_pose_new) - POSE < R(dB)
               <=> d_pose_new / d_pose_base < ((POSE + R(dB)) / POSE) ** 2

The last line is the usable form: a single dimensionless RATIO BAR, computable before
any decode, that says exactly how many times worse d_pose is allowed to get.

The structural fact this equation exists to record
-------------------------------------------------
The bar is ``(1 + R/POSE)**2`` — QUADRATIC in the credit, because the pose term is a
SQUARE ROOT of the scored quantity. Returning twice the bytes does not buy twice the
pose tolerance; it buys roughly four times. On the live frontier this is not a small
effect: deleting the carrier outright (22,161 B) tolerates a 7.72x pose degradation,
while a conservative rank-11 shave (1,514 B) tolerates only 1.26x. The cheapest rung is
also by far the most FORGIVING rung.

That inverts the ordinary shave-conservatively instinct, and it is why an
``alpha = 0`` (section-deleted) probe belongs at the FRONT of any rate-credit ladder,
not the end: it is the cheapest single decode, it upper-bounds the whole
d_pose(fidelity) curve, and it sits under the loosest bar. A ladder that measures only
interior rungs answers the affordability question in the regime where it is hardest to
satisfy.

What this equation does NOT license
-----------------------------------
It says nothing about WHERE on the d_pose(fidelity) curve a given rung lands — that is
a measurement, and it is the binding unknown. Reconstruction-error ladders (greedy
keep set -> exhaustive keep set -> rotated subspace) are optimal in EUCLIDEAN field MSE;
the scored quantity is a PoseNet readout. Ranking rungs by field MSE and then applying
this bar is a LEVEL error (S-geometry pullback, CLAUDE.md P0 triple, task #974): the bar
is exact, the ordering fed into it is not.

Axis: closed-form DERIVED (exact arithmetic). The d_pose values it is applied to are
whatever axis their own measurement carries; this equation never upgrades them.
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

EQUATION_ID = "carrier_rate_credit_pose_affordance_v1"
AXIS = "[DERIVED closed form] exact score arithmetic; carries no measurement authority"
CHARTER = ".omx/research/charters/ddm_ra2_carrier_rank_pose_calibration.md"

#: upstream/evaluate.py:63 rate denominator (the 2026-08-16 value; Catalog #812 records
#: that evaluate.py sums rglob('*') over videos/ rather than hardcoding this constant, so
#: a videos/ change moves it — never treat it as immortal).
RATE_DENOMINATOR_BYTES = 37_545_489

#: hv1 ep0634 frontier, contest-CUDA T4 n600, archive sha 80d9c8c6...
FRONTIER_ARCHIVE_SHA256 = (
    "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
)
FRONTIER_S = 0.15959729295498598
FRONTIER_ARCHIVE_BYTES = 182_759
FRONTIER_POSE_TERM = 0.0082945765
FRONTIER_D_POSE = FRONTIER_POSE_TERM**2 / 10.0
#: The real coded length of the carrier.br stream on that archive (MEASURED, not the
#: 22,278 B CPR1 rebaseline that an earlier pass used).
CARRIER_STREAM_BYTES = 22_161
GAP_TO_TARGET = 0.0095973  # FRONTIER_S - 0.15


def rate_credit(returned_bytes: int, *, denominator: int = RATE_DENOMINATOR_BYTES) -> float:
    """Rate-term credit, in S units, for returning ``returned_bytes`` to the archive.

    NEGATIVE ``returned_bytes`` (a byte-ADDING edit) is legal and meaningful: the credit
    goes negative, the ratio bar drops below 1, and the edit is admissible only if it
    IMPROVES d_pose. That is the correct arithmetic, not an error.
    """
    if denominator <= 0:
        raise ValueError("rate denominator must be positive")
    return 25.0 * returned_bytes / denominator

def pose_affordance_ratio(
    returned_bytes: int,
    *,
    pose_term: float = FRONTIER_POSE_TERM,
    denominator: int = RATE_DENOMINATOR_BYTES,
) -> float:
    """Max ``d_pose_new / d_pose_base`` that the credit still pays for (exclusive bound).

    Quadratic in the credit: ``(1 + R/POSE)**2``.
    """
    if pose_term <= 0.0:
        raise ValueError("pose_term must be positive; a zero base pose term has no ratio bar")
    return ((pose_term + rate_credit(returned_bytes, denominator=denominator)) / pose_term) ** 2


def is_affordable(
    returned_bytes: int,
    d_pose_new: float,
    *,
    d_pose_base: float = FRONTIER_D_POSE,
    pose_term: float | None = None,
    denominator: int = RATE_DENOMINATOR_BYTES,
) -> bool:
    """Exact admission test. Ignores d_seg — valid only when the change is seg-invisible.

    On the hv1 receiver the carrier renders frame_0 only and SegNet reads ``x[:, -1]``
    (frame_1), so a carrier-only edit is seg-invisible BY CONSTRUCTION. Any other
    section must add its own d_seg term before using this test.

    ``pose_term`` and ``d_pose_base`` describe the SAME base and must agree —
    ``pose_term == sqrt(10*d_pose_base)``. Passing an inconsistent pair silently
    computes a bar for one base and applies it to another; that is the
    silently-wrong-instrument class, so it REFUSES instead.
    """
    if d_pose_base <= 0.0:
        raise ValueError("d_pose_base must be positive; a zero base has no ratio")
    derived = (10.0 * d_pose_base) ** 0.5
    if pose_term is None:
        pose_term = derived
    elif abs(pose_term - derived) > 1e-9 * max(derived, 1.0):
        raise ValueError(
            f"pose_term {pose_term!r} disagrees with sqrt(10*d_pose_base) = {derived!r}; "
            "they must describe the same base — refusing to apply one base's bar to another"
        )
    return d_pose_new / d_pose_base < pose_affordance_ratio(
        returned_bytes, pose_term=pose_term, denominator=denominator
    )


#: DERIVED rungs on the live frontier's carrier stream. keep_bytes -> (returned, credit,
#: ratio bar, credit as a fraction of the remaining sub-0.15 gap). Coded lengths are
#: MEASURED through the shipped CPR1 codec + Brotli cell by
#: experiments/ddm_ra1b_exhaustive_keepset_refit.py; the arithmetic on them is exact.
DERIVED_RUNGS = {
    "alpha0_carrier_deleted": (0, 22_161, 0.014756100, 7.722874, 1.538),
    "rank1_exhaustive": (1_867, 20_294, 0.013512942, 6.912323, 1.408),
    "rank4_exhaustive": (7_499, 14_662, 0.009762824, 4.739385, 1.017),
    "rank4_greedy": (7_576, 14_585, 0.009711553, 4.712510, 1.012),
    "rank6_exhaustive": (11_218, 10_943, 0.007286495, 3.528630, 0.759),
    "rank11_exhaustive": (20_647, 1_514, 0.001008110, 1.257849, 0.105),
}


#: MEASURED 2026-08-16 (ra2c, n600 upstream/evaluate.py, archive held byte-identical):
#: deleting the carrier entirely (alpha=0) raised d_pose from 1.4747e-4 to 51.67767334.
#: With the trivial alpha=1 anchor (ratio 1, proven by a 600/600 byte-identity control) this
#: fixes the one free constant of the damage law `ratio - 1 = K * frac_err ** p`.
#: p = 2 is the principled exponent: d_pose is a quadratic form in the pose residual.
CARRIER_ALPHA0_DPOSE = 51.67767334
CARRIER_BASE_DPOSE_ADVISORY = 0.00014747
CARRIER_DAMAGE_K = CARRIER_ALPHA0_DPOSE / CARRIER_BASE_DPOSE_ADVISORY - 1.0

#: MEASURED 2026-08-16 (exact SVD of the shipped objects in archive 80d9c8c6...):
#: label -> (condition number, rank-4 relative error, rank-11 relative error).
#: Every spectrum is flat -- the 12-dim carrier is effectively full rank. By Eckart-Young the
#: truncated SVD is the OPTIMAL rank-r approximation in Frobenius norm, so these are LOWER
#: BOUNDS on any rank-r approximation error in that norm, not estimates of one method.
CARRIER_SPECTRA = {
    "basis_12x2304": (3.61, 0.52081, 0.14714),
    "coeff_600x12": (5.40, 0.49203, 0.11224),
    "rendered_field_600x2304": (17.32, 0.25145, 0.04231),
}


def carrier_error_tolerance(
    returned_bytes: int,
    *,
    pose_term: float = FRONTIER_POSE_TERM,
    exponent: float = 2.0,
    denominator: int = RATE_DENOMINATOR_BYTES,
) -> float:
    """Max fractional carrier error an approximation may carry and still be affordable.

    Inverts the affordance bar through the measured damage law: solve
    `K * err ** p = bar - 1` for err, where `bar = (1 + R/POSE) ** 2`.

    A LARGER exponent is MORE forgiving at small error, so `exponent=2` (the principled
    quadratic) yields an UPPER BOUND on tolerance for any law with p <= 2.
    """
    bar = pose_affordance_ratio(returned_bytes, pose_term=pose_term, denominator=denominator)
    if bar <= 1.0:
        return 0.0
    return float((bar - 1.0) / CARRIER_DAMAGE_K) ** (1.0 / exponent)


def build_carrier_rate_credit_pose_affordance_v1() -> CanonicalEquation:
    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=FRONTIER_ARCHIVE_SHA256,
        source_path=CHARTER,
        captured_at_utc="2026-08-16T16:30:00Z",
    )
    anchors = (
        EmpiricalAnchor(
            anchor_id="carrier_affordance_bar_is_quadratic_in_credit_20260816",
            measurement_utc="2026-08-16T16:30:00Z",
            inputs={
                "frontier": {
                    "S": FRONTIER_S,
                    "archive_bytes": FRONTIER_ARCHIVE_BYTES,
                    "archive_sha256": FRONTIER_ARCHIVE_SHA256,
                    "pose_term": FRONTIER_POSE_TERM,
                    "d_pose": FRONTIER_D_POSE,
                    "axis": "[contest-CUDA T4 n600]",
                },
                "carrier_stream_bytes": CARRIER_STREAM_BYTES,
                "coded_lengths_source": (
                    "experiments/ddm_ra1b_exhaustive_keepset_refit.py — shipped CPR1 "
                    "codec + shipped Brotli cell, every payload round-trip-asserted "
                    "through the receiver's modular cumsum"
                ),
            },
            predicted_output={
                "closed_form": "d_pose_new/d_pose_base < ((POSE + 25*dB/D)/POSE)**2",
                "structural_claim": (
                    "the bar is QUADRATIC in the returned bytes, so the cheapest rung is "
                    "also the most forgiving — alpha=0 belongs at the FRONT of a "
                    "rate-credit ladder, not the end"
                ),
            },
            empirical_output={
                "rungs": DERIVED_RUNGS,
                "headline": (
                    "deleting the carrier outright returns 153.8% of the remaining "
                    "sub-0.15 gap in rate credit and tolerates a 7.72x d_pose "
                    "degradation; a conservative rank-11 shave returns 10.5% and "
                    "tolerates only 1.26x"
                ),
                "verdict_scope": (
                    "the ARITHMETIC is exact and general; the RUNG TABLE is instance-"
                    "scoped to the hv1 ep0634 carrier stream and its measured coded "
                    "lengths"
                ),
                "does_not_license": (
                    "any claim about WHERE a rung lands on the d_pose(fidelity) curve. "
                    "That is unmeasured. Ranking rungs by Euclidean field MSE and then "
                    "applying this bar is an S-geometry LEVEL error (task #974): the bar "
                    "is exact, the ordering fed into it is not."
                ),
            },
            residual=0.0,
            source_artifact=CHARTER,
            measurement_method=(
                "closed-form derivation from upstream/evaluate.py:63 (rate) and the "
                "sqrt(10*d_pose) pose term; coded lengths MEASURED through the shipped "
                "codec; no scorer forward involved"
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_SOURCE_INSPECTION",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "Rate-credit pose affordance: returned bytes buy a QUADRATIC d_pose "
            "tolerance, so the cheapest rung is the most forgiving"
        ),
        one_line_summary=(
            "Seg-invisible edit returning dB bytes: admissible iff d_pose ratio < "
            "(1 + R/POSE)**2, R = 25*dB/D. Quadratic in the credit, so the cheapest "
            "rung is the most forgiving."
        ),
        latex_form=(
            r"R(\Delta B)=\frac{25\,\Delta B}{D},\quad P=\sqrt{10\,d_{pose}^{base}},\quad "
            r"\frac{d_{pose}^{new}}{d_{pose}^{base}}<\left(1+\frac{R}{P}\right)^{2}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.carrier_rate_credit_pose_affordance_20260816"
            ":pose_affordance_ratio"
        ),
        domain_of_validity={
            "axis": AXIS,
            "research_only": False,
            "applies_to": (
                "any candidate that returns archive bytes while degrading only d_pose. "
                "The seg-invisibility premise must be established separately — on the "
                "hv1 receiver it holds by construction for carrier-only edits (carrier "
                "renders frame_0; SegNet reads frame_1)."
            ),
            "does_not_apply_to": (
                "edits that also move d_seg (add the 100*Delta_d_seg term), or any claim "
                "about the shape of d_pose(fidelity)"
            ),
        },
        units_in={
            "dB": "bytes returned to the archive",
            "d_pose_base, d_pose_new": "PoseNet MSE (the scored quantity, not a proxy)",
        },
        units_out={
            "R": "S units (score points)",
            "ratio bar": "dimensionless",
        },
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"registration": 0.0},
        last_calibration_utc="2026-08-16T16:30:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "ddm_ra2 carrier rank/pose calibration — the admission test for every rung",
            "any rate-credit ladder on any payload section: order the ladder so the "
            "LARGEST-credit rung is measured first, because it sits under the loosest "
            "bar and bounds the whole curve in one decode",
            "mp2 / rfo2 rate-route rungs — same test, different section",
        ),
        canonical_producers=(
            "experiments/ddm_ra1b_exhaustive_keepset_refit.py (coded lengths)",
            "experiments/ddm_ra2a_carrier_fidelity_pose_ladder.py (the d_pose measurement "
            "this bar is applied to)",
        ),
        provenance=provenance,
    )


def populate_carrier_rate_credit_pose_affordance_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (EQUATIONS leg of the ra2a landing)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_carrier_rate_credit_pose_affordance_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id
    )
    return eq


__all__ = [
    "AXIS",
    "CARRIER_ALPHA0_DPOSE",
    "CARRIER_BASE_DPOSE_ADVISORY",
    "CARRIER_DAMAGE_K",
    "CARRIER_SPECTRA",
    "CARRIER_STREAM_BYTES",
    "carrier_error_tolerance",
    "DERIVED_RUNGS",
    "EQUATION_ID",
    "FRONTIER_POSE_TERM",
    "RATE_DENOMINATOR_BYTES",
    "build_carrier_rate_credit_pose_affordance_v1",
    "is_affordable",
    "populate_carrier_rate_credit_pose_affordance_equation",
    "pose_affordance_ratio",
    "rate_credit",
]
