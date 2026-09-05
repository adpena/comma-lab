# SPDX-License-Identifier: MIT
"""Canonical equation: the pose carrier's BASIS payload is a rate-fidelity exchange
governed by the quantiser STEP, not by the bit depth -- and basis payload bytes reach
``archive.zip`` at very nearly 1:1.

MEASURED 2026-09-05 (``ddm_pc1``) on the cl2 frontier body, archive sha ``08ec8533...``,
179,982 B, through the receiver's own container: every byte count below comes from
building a real ``archive.zip`` with ``ddm_up3.build_archive`` and stat-ing it, under a
container-identity control that reproduces the shipped bytes bit for bit.

The geometry
------------
``cpr1/carrier_codec`` stores the carrier basis as ``12 x 3 x 24 x 32 = 27,648`` symbols
on a 5-bit zigzag alphabet, canonical-Huffman coded (12,277 B on this body), then
re-coded by the RR5 adaptive-arithmetic rider (12,046 B).  ``cpr1/inflate.py:245-246``
multiplies the decoded codes by a per-atom ``basis_scales`` float, and
``normalized_basis`` then centres and RMS-normalises **per atom**.

Consequence 1 -- the per-atom scale CANCELS (in exact arithmetic; see the caveat)
--------------------------------------------------------------------------------
A positive per-atom scale is divided straight back out by the per-atom RMS.  MEASURED:
re-rendering the basis from the raw codes alone reproduces the shipped normalised basis
to ``1.9073e-06`` max abs difference on an RMS-1 field (float32 epsilon).  All twelve
shipped scales are positive.

**CAVEAT, measured after an overclaim was caught.**  An earlier draft of this module said
the 48-byte ``basis_scales`` field "reaches nothing".  That is false at the render: the
receiver rounds twice, and the 1.9e-06 residue crosses rounding boundaries.  Over 24
strided pairs only 10 of 24 rendered ``frame_0`` images are bit-identical when the scales
are dropped; 14 differ by +-1 uint8.  So the field is RECOVERABLE but not free by
construction -- removing it needs a measured d_pose like any other change.  What the
cancellation DOES license, and the only thing this module rests on, is the next
consequence: the per-atom quantiser STEP is free, because a re-quantised atom is
re-normalised by its own RMS whatever step produced it.

Consequence 2 -- the quantiser STEP is therefore free, and it is the real lever
-------------------------------------------------------------------------------
Because the scale cancels, re-quantising an atom with ANY positive step is legal at zero
cost.  So at a fixed alphabet width the step selects a point on a rate-fidelity curve,
and the two ends are far apart.  MEASURED at 4 bits: a global ``round(codes/2)`` (the
obvious first pass) gives per-atom cosine 0.899-0.983 and an 8,196 B payload, while a
per-atom searched step gives 0.991-1.000 and an 11,927 B payload -- **3,731 B apart at
the same bit depth**.  At 3 bits the ends are 3,524 B apart.

The searched step lands near 0.94 at 4 bits: *clip the tails, do not rescale*.  The
shipped code distribution is concentrated (per-atom std 1.44-3.77 against a +-15 range),
so clipping preserves both fidelity and entropy while rescaling destroys both.  Three of
the twelve atoms already span only [-7, +7] and survive a 4-bit alphabet with NO loss.

**A bit-depth number alone does not price a basis idea.**  Any basis-precision work must
report the step and sweep it against Delta S, not pick a depth.

Consequence 3 -- payload bytes pass through to the archive at ~1:1
------------------------------------------------------------------
The carrier stream is ``brotli(q=9, lgwin=16)`` over a body whose two payloads are
already entropy-coded, so brotli removes only 247 B (1.1%) of the 22,278 B body.
MEASURED across a 12 KB range of basis payload sizes, the archive delta is
``PASS_THROUGH`` times the payload delta with ``PASS_THROUGH`` in [0.981, 1.121].  A
basis idea can therefore be priced from its payload size alone, before any archive is
built -- with the spread carried, never dropped.

What this equation does NOT license
------------------------------------
Nothing about d_pose.  Fidelity here is a per-atom cosine in the rendered field, and the
scored quantity is a PoseNet readout; ranking basis variants by cosine and then claiming
a d_pose is the S-geometry level error.  The one d_pose anchor recorded below is a
REFUSAL measured through the full re-solve, not a curve.

Axis: bytes ``[exact local byte arithmetic, receiver-verified]``; the single d_pose
anchor is ``[macOS-CPU advisory, cpu_torch fp32, DALI GT]``.  No score claim.
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

EQUATION_ID = "pose_carrier_basis_rate_fidelity_exchange_v1"
AXIS = (
    "[exact local byte arithmetic, receiver-verified parse-back] for every byte count; "
    "the single d_pose anchor is [macOS-CPU advisory, cpu_torch fp32, n600, DALI GT]"
)
CHARTER = (
    ".omx/research/charters/"
    "ddm_pc1_pose_carrier_efficiency_basis_precision_and_generated_basis_20260905.md"
)
MEMO = ".omx/research/ddm_pc1_pose_carrier_efficiency_20260905.md"

#: upstream/evaluate.py:63 rate denominator.
RATE_DENOMINATOR_BYTES = 37_545_489

#: The cl2 frontier body this equation was measured on.
FRONTIER_ARCHIVE_SHA256 = (
    "08ec85333d13d71344b4482cf261e3b2d508725e49f3ca05971265a81498ad4e"
)
FRONTIER_ARCHIVE_BYTES = 179_982
FRONTIER_S_T4 = 0.14781744131049854
#: n600 d_pose of this body, cpu_torch fp32 / DALI GT (the base every delta is against).
#: The contest-CUDA T4 row for the same body reads 6.14e-06 -- 0.1% apart, a control.
BASE_D_POSE_CPU_TORCH = 6.134076407345324e-06

#: Carrier geometry, MEASURED from decode_compact_carrier on this body.
CARRIER_DIM = 12
BASIS_PLANES = 3
CARRIER_H, CARRIER_W = 24, 32
BASIS_SYMBOLS = CARRIER_DIM * BASIS_PLANES * CARRIER_H * CARRIER_W  # 27,648
SHIPPED_BASIS_PAYLOAD_BYTES = 12_277  # canonical Huffman
SHIPPED_BASIS_PAYLOAD_BYTES_ARITH = 12_046  # after the RR5 rider
SHIPPED_RICE_PAYLOAD_BYTES = 9_830
CARRIER_BODY_BYTES = 22_278
CARRIER_STREAM_BYTES = 22_031  # what the archive actually stores
#: basis_scales: 12 f32 whose per-atom effect cancels in normalized_basis to 1.9073e-06,
#: but NOT to zero at the render (10/24 pairs bit-identical, 14 differ by +-1 uint8).
#: Recoverable bytes, not free ones -- the name says "nearly" for a reason.
NEARLY_DEAD_BASIS_SCALE_BYTES = 48
DEAD_BASIS_SCALE_BYTES = NEARLY_DEAD_BASIS_SCALE_BYTES  # legacy alias, same value

#: label -> (basis payload bytes, archive bytes, min per-atom cosine, max per-atom cosine)
#: Every archive byte count is a real build through the identity-controlled container.
MEASURED_BASIS_RUNGS: dict[str, tuple[int, int, float, float]] = {
    "shipped_5bit": (12_277, 179_982, 1.0, 1.0),
    "4bit_step_searched": (11_927, 179_590, 0.991, 1.000),
    "3bit_step_searched": (9_087, 176_778, 0.970, 0.992),
    "4bit_global_halve": (8_196, 175_911, 0.899, 0.983),
    "3bit_global_quarter": (5_563, 172_869, 0.715, 0.949),
    "generated_no_stored_basis": (0, 167_939, 0.0, 0.0),
}

#: MEASURED pass-through of basis payload bytes into archive.zip, over the rungs above.
#: The generated rung also switches the RR5 rider off (it has nothing to code), which is
#: why its ratio sits at the low end; the band is carried, never averaged away.
PASS_THROUGH_MIN = 0.981
PASS_THROUGH_MAX = 1.121
PASS_THROUGH_TYPICAL = 1.03


def archive_delta_from_basis_payload_bytes(
    payload_bytes: int,
    *,
    shipped_payload_bytes: int = SHIPPED_BASIS_PAYLOAD_BYTES,
    pass_through: float = PASS_THROUGH_TYPICAL,
) -> float:
    """DERIVED archive-byte delta for a basis payload of ``payload_bytes``.

    Negative means the archive shrinks.  This is a PREDICTION with a measured band
    (``PASS_THROUGH_MIN``/``MAX``): use it to rank basis ideas before building, never to
    report a byte count.  A reported byte count must come from a real archive build.
    """
    if payload_bytes < 0:
        raise ValueError("a basis payload cannot be negative")
    return pass_through * (payload_bytes - shipped_payload_bytes)


def archive_delta_band(payload_bytes: int) -> tuple[float, float]:
    """The MEASURED band around :func:`archive_delta_from_basis_payload_bytes`."""
    low = archive_delta_from_basis_payload_bytes(
        payload_bytes, pass_through=PASS_THROUGH_MAX
    )
    high = archive_delta_from_basis_payload_bytes(
        payload_bytes, pass_through=PASS_THROUGH_MIN
    )
    return (min(low, high), max(low, high))


def rate_credit(returned_bytes: int, *, denominator: int = RATE_DENOMINATOR_BYTES) -> float:
    """Score credit, in S units, for returning ``returned_bytes`` to the archive."""
    if denominator <= 0:
        raise ValueError("rate denominator must be positive")
    return 25.0 * returned_bytes / denominator


def break_even_d_pose(d_pose_base: float, delta_score_rate: float) -> float:
    """The largest d_pose that a rate change still pays for.

    ``Delta S = delta_score_rate + sqrt(10 d_new) - sqrt(10 d_base) <= 0`` gives

        sqrt(10 d_new) <= sqrt(10 d_base) - delta_score_rate

    which is the whole story for BOTH signs, and the sign is the point.  A rate
    SAVING (``delta_score_rate < 0``) raises the bar above ``d_pose_base``: the
    edit may cost pose.  A byte-ADDING edit (``delta_score_rate > 0``) lowers it
    below ``d_pose_base``: the edit must IMPROVE pose to be admissible.  At
    exactly zero the bar is ``d_pose_base`` itself.

    Returns ``0.0`` when the added bytes cost more than the entire pose term, so
    that no non-negative d_pose can pay for them.  An earlier draft returned
    ``inf`` for the non-saving case; that reads as an unlimited pose budget and
    is exactly backwards, so it is gone.
    """
    if d_pose_base <= 0.0:
        raise ValueError("d_pose_base must be positive")
    leg = (10.0 * d_pose_base) ** 0.5 - delta_score_rate
    if leg <= 0.0:
        return 0.0
    return leg * leg / 10.0


def basis_break_even_d_pose(
    payload_bytes: int,
    *,
    d_pose_base: float = BASE_D_POSE_CPU_TORCH,
    pass_through: float = PASS_THROUGH_TYPICAL,
) -> float:
    """End-to-end: a basis payload size -> the d_pose it may cost and still pay.

    The composition this equation exists to make cheap: payload bytes -> archive bytes
    (measured pass-through) -> rate credit -> break-even d_pose.  Any additional cap
    (for example ft1's same-object pose ceiling) binds SEPARATELY and is not applied
    here; a caller that has one must take the minimum itself.
    """
    delta_bytes = archive_delta_from_basis_payload_bytes(
        payload_bytes, pass_through=pass_through
    )
    return break_even_d_pose(d_pose_base, rate_credit(round(delta_bytes)))


def build_pose_carrier_basis_rate_fidelity_exchange_v1() -> CanonicalEquation:
    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=FRONTIER_ARCHIVE_SHA256,
        source_path=MEMO,
        captured_at_utc="2026-09-05T18:00:00Z",
    )
    anchors = (
        EmpiricalAnchor(
            anchor_id="basis_quantiser_step_is_the_rate_fidelity_lever_20260905",
            measurement_utc="2026-09-05T17:35:00Z",
            inputs={
                "body": {
                    "archive_sha256": FRONTIER_ARCHIVE_SHA256,
                    "archive_bytes": FRONTIER_ARCHIVE_BYTES,
                    "S_t4": FRONTIER_S_T4,
                    "axis": "[contest-CUDA T4 n600]",
                },
                "carrier_stream_bytes": CARRIER_STREAM_BYTES,
                "carrier_body_bytes": CARRIER_BODY_BYTES,
                "basis_symbols": BASIS_SYMBOLS,
                "method": (
                    "every archive byte count is a real build through "
                    "ddm_up3.build_archive under a container-identity control that "
                    "reproduces the shipped bytes bit for bit; the basis payload is "
                    "encoded by the receiver's own runtime/rr5_arith_basis coders"
                ),
            },
            predicted_output={
                "charter_derived_rate_saving_4bit": -2_440,
                "charter_derived_rate_saving_3bit": -4_880,
                "charter_derived_rate_saving_generated": -12_200,
            },
            empirical_output={
                "rungs_payload_archive_cosmin_cosmax": MEASURED_BASIS_RUNGS,
                "headline": (
                    "at a FIXED 4-bit alphabet the quantiser step alone moves the basis "
                    "payload by 3,731 B (8,196 vs 11,927) and per-atom cosine by "
                    "0.899-0.983 vs 0.991-1.000; the bit depth does not price a basis, "
                    "the step does"
                ),
                "nearly_dead_bytes": (
                    f"basis_scales is {DEAD_BASIS_SCALE_BYTES} B whose per-atom effect "
                    "cancels in normalized_basis to 1.9073e-06 on an RMS-1 field (all "
                    "twelve scales positive) -- but NOT to zero at the render: over 24 "
                    "strided pairs only 10 of 24 frame_0 images are bit-identical when "
                    "the scales are dropped, 14 differ by +-1 uint8. Recoverable bytes, "
                    "NOT free by construction. An earlier draft claimed otherwise and "
                    "was falsified by this test."
                ),
                "pass_through_band": [PASS_THROUGH_MIN, PASS_THROUGH_MAX],
                "verdict_scope": (
                    "FAMILY for the exchange itself (it follows from the per-atom scale "
                    "cancelling, which is a property of normalized_basis); INSTANCE for "
                    "the rung table, which is this body's basis and this container"
                ),
                "does_not_license": (
                    "any d_pose claim. Fidelity here is a per-atom cosine in the "
                    "rendered field; the scored quantity is a PoseNet readout, and "
                    "ordering variants by cosine then asserting a d_pose is the "
                    "S-geometry level error."
                ),
            },
            residual=0.0,
            source_artifact=MEMO,
            measurement_method=(
                "RX1 container parsed with the receiver's own runtime/residual_archive; "
                "basis re-encoded with runtime/rr5_arith_basis; archives rebuilt with "
                "ddm_up3.build_archive and stat-ed; no scorer forward involved"
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_SOURCE_INSPECTION",
        ),
        EmpiricalAnchor(
            anchor_id="generated_dct_basis_refused_on_a_population_lower_bound_20260905",
            measurement_utc="2026-09-05T17:58:00Z",
            inputs={
                "generated_bases_tried": (
                    "lowest-total-degree separable 2-D DCT at 24x32, three generic "
                    "colour placements: achromatic (luma), per-plane (planar), and "
                    "generic-opponent"
                ),
                "explained_energy_of_the_600_realized_fields": {
                    "luma": 0.02491, "planar": 0.01550, "opponent": 0.01550,
                },
                "solver": (
                    "ddm_jg5.refine_pair verbatim, 40 outer rounds, 400 GN iterations, "
                    "+-2 polish, warm-started from the least-squares field projection"
                ),
                "pairs_measured": 20,
                "sampling": "strided (0, 30, ... 570), never a contiguous prefix",
                "rate_saving_if_it_worked": -12_043,
                "break_even_d_pose": 2.5125e-05,
                "ft1_same_object_pose_ceiling": 1.694e-05,
            },
            predicted_output={
                "charter_prediction_d_pose": "1e-5 to 5e-5, break-even at 2.3e-5",
            },
            empirical_output={
                "per_pair_final_d_pose_min_median_max": [6.132, 20.5045, 117.077],
                "all_pairs_stopped_at": "no_improving_step (a physical stop, not a budget)",
                "n600_mean_lower_bound_from_the_20_measured_pairs": 0.998631,
                "exceeds_ft1_ceiling_by": 58_952,
                "exceeds_break_even_by": 39_748,
                "why_20_pairs_settle_it": (
                    "d_pose is a mean of 600 NON-NEGATIVE per-pair values, so the n600 "
                    "mean is bounded below by the measured pairs' sum over 600. That "
                    "bound holds for the whole population and no unmeasured pair can "
                    "reduce it: this is a completed argument, not a subset verdict."
                ),
                "verdict_scope": (
                    "FAMILY -- generated separable-DCT bases at rank 12 on this "
                    "vehicle's carrier. NOT closed: non-DCT generated families "
                    "(Zernike, wavelet, steerable, Gabor), higher-rank generated bases, "
                    "and generated-plus-small-stored-correction hybrids."
                ),
            },
            residual=0.998631 - 2.5125e-05,
            source_artifact=MEMO,
            measurement_method=(
                "full jg5 re-solve per pair on the generated basis, exact non-STE "
                "receiver render, frozen CPU PoseNet, DALI GT targets"
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "Pose-carrier basis: the quantiser STEP is the rate-fidelity lever, not the "
            "bit depth; basis payload bytes reach archive.zip at ~1:1"
        ),
        one_line_summary=(
            "The per-atom basis scale cancels, so the quantiser step is free: at a fixed "
            "4-bit alphabet the step alone moves the payload 3,731 B. Payload bytes reach "
            "archive.zip at 1.03x (band 0.981-1.121)."
        ),
        latex_form=(
            r"\Delta B_{archive} \approx \kappa\,(b - b_0),\ \kappa\in[0.981,1.121];\quad "
            r"d_{pose}^{even} = \tfrac{1}{10}\left(\sqrt{10 d_{pose}^{base}} "
            r"- \tfrac{25\,\Delta B_{archive}}{D}\right)^{2}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.pose_carrier_basis_rate_fidelity_exchange_20260905"
            ":basis_break_even_d_pose"
        ),
        domain_of_validity={
            "axis": AXIS,
            "research_only": False,
            "applies_to": (
                "the CPR1 pose carrier's basis payload on any body whose receiver "
                "normalises the basis per atom (centre + RMS), which is what makes the "
                "per-atom scale cancel and the step free"
            ),
            "does_not_apply_to": (
                "the coefficient payload (a different coder with a different lever), "
                "any d_pose prediction, or a receiver that consumes the basis without "
                "per-atom normalisation"
            ),
        },
        units_in={
            "payload_bytes": "bytes of the stored basis payload",
            "d_pose_base": "PoseNet MSE (the scored quantity)",
        },
        units_out={
            "archive delta": "bytes",
            "break-even d_pose": "PoseNet MSE",
        },
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"registration": 0.0},
        last_calibration_utc="2026-09-05T18:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "ddm_pc1 pose-carrier efficiency -- prices every basis variant before it is "
            "built, and refuses bit-depth-only framings",
            "any future basis-precision or generated-basis arm: report the STEP and "
            "sweep it against Delta S, never a bit depth alone",
            "carrier_rate_credit_pose_affordance_v1 -- this equation supplies the byte "
            "delta that that one's affordance bar consumes",
        ),
        canonical_producers=(
            "experiments/ddm_pc1_pose_carrier_efficiency.py (modes probe, rate, "
            "coverage, solve, measure, price)",
        ),
        provenance=provenance,
    )


def populate_pose_carrier_basis_rate_fidelity_exchange_equation(
    *, path=None, lock_path=None, agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (EQUATIONS leg of the ddm_pc1 landing)."""
    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_pose_carrier_basis_rate_fidelity_exchange_v1()
    register_canonical_equation(
        equation, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id
    )
    return equation


__all__ = [
    "AXIS",
    "BASE_D_POSE_CPU_TORCH",
    "BASIS_SYMBOLS",
    "CARRIER_STREAM_BYTES",
    "DEAD_BASIS_SCALE_BYTES",
    "EQUATION_ID",
    "MEASURED_BASIS_RUNGS",
    "NEARLY_DEAD_BASIS_SCALE_BYTES",
    "PASS_THROUGH_MAX",
    "PASS_THROUGH_MIN",
    "PASS_THROUGH_TYPICAL",
    "SHIPPED_BASIS_PAYLOAD_BYTES",
    "archive_delta_band",
    "archive_delta_from_basis_payload_bytes",
    "basis_break_even_d_pose",
    "break_even_d_pose",
    "build_pose_carrier_basis_rate_fidelity_exchange_v1",
    "populate_pose_carrier_basis_rate_fidelity_exchange_equation",
    "rate_credit",
]
