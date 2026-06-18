# SPDX-License-Identifier: MIT
"""Capacity-RD score-aware-QAT desk model — the $0 decision gate for the pivot.

The approved campaign pivot (operator 2026-06-18: "higher capacity + score-aware
FP-shrink QAT to cut its rate") asks ONE question that must be answered on paper
BEFORE any GPU/MPS training is burned:

    Over decoder capacity p, minimise
        S(p) = 100·d_seg(p) + sqrt(10·d_pose) + 25·B_QAT(p)/B0
    where B_QAT(p) is the QAT-shrunk byte budget (~half native via int4/int5 on the
    d_seg-blind weights), NOT the native byte budget B(p).

Native capacity scaling FORFEITS rate headroom: bytes grow ~quadratically in base_ch
(decoder params ~136·C^2) while d_seg drops slowly, so on the NATIVE Pareto curve no
capacity is sub-0.15 (bc20 floor 0.118, bc28 0.156, bc36 0.204 — measured/estimated
2026-06-16). The pivot's whole premise is that QAT shifts B(p) DOWN, so the
S-optimal capacity moves UP (lower d_seg) at the same byte budget.

THIS MODULE IS A DESK MODEL, NOT A MEASUREMENT. Every number it emits is
``[advisory]`` and NON-PROMOTABLE. It composes:

  * MEASURED anchors: the two endpoints of the real (capacity, d_seg, bytes) Pareto
    curve — base_ch20 small basis and the pr110 frontier — plus the MEASURED
    post-int8-brotli byte-shrink ratios from ``reports/fp_shrink_ptq_bc20_n600.json``.
  * The EXACT ``decoder_param_count`` formula (``tac.torch_vehicle.configurable_taper_decoder``)
    for native byte modelling at intermediate capacities.
  * The QAT distortion-hold ASSUMPTION from the design memo (the break-even contract:
    QAT holds d_seg within +delta of the float floor while cutting bytes ~40-47%).

The output is the S_QAT(p) table + the argmin capacity + a STOP/PROCEED gate verdict.
If no capacity's QAT-shrunk S is promising (< the configured proceed threshold), the
caller STOPS and reports — do not burn days training a vehicle the math already kills.

NO score claim. NO promotion. The exact pointer stays pointer-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from tac.torch_vehicle.configurable_taper_decoder import (
    decoder_param_count,
    vendored_taper,
)

# Contest constants.
RATE_DENOM = 37_545_489  # B0
SEG_WEIGHT = 100.0
POSE_WEIGHT_INNER = 10.0  # sqrt(10 * d_pose)


def score(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    """The exact contest score S = 100*d_seg + sqrt(10*d_pose) + 25*bytes/B0."""

    return SEG_WEIGHT * d_seg + math.sqrt(POSE_WEIGHT_INNER * d_pose) + 25.0 * archive_bytes / RATE_DENOM


def rate_term(archive_bytes: int) -> float:
    return 25.0 * archive_bytes / RATE_DENOM


def pose_term(d_pose: float) -> float:

    return math.sqrt(POSE_WEIGHT_INNER * d_pose)


# ---------------------------------------------------------------------------
# MEASURED anchors (the two endpoints of the real capacity↔d_seg↔byte Pareto).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasuredAnchor:
    """A real (capacity, d_seg, d_pose, bytes) point. ``base_ch`` may be None for the
    frontier (which is an entropy-recode of a frontier-class HNeRV, not a clean base_ch)."""

    label: str
    base_ch: int | None
    decoder_params: int | None
    d_seg: float
    d_pose: float
    archive_bytes: int
    provenance: str


# base_ch20 small basis — the 600-pair plain-CE MPS basin, full 600-pair exact eval
# (reports/fp_shrink_ptq_bc20_n600.json fp32 row). Local CPU ~= contest-CPU (G3).
ANCHOR_BC20 = MeasuredAnchor(
    label="base_ch20_small_basis",
    base_ch=20,
    decoder_params=decoder_param_count(vendored_taper(20)),
    d_seg=0.002600919948890805,
    d_pose=0.00034168662969022987,
    archive_bytes=89_136,
    provenance="reports/fp_shrink_ptq_bc20_n600.json fp32 row (600-pair exact, [contest-CPU advisory])",
)


# Frontier — lane_pr110_payload_entropy_recode, the contest-CPU frontier pointer.
# S=0.19110, bytes=177169 (canonical_frontier_pointer.json). Component split from the
# 2026-06-16 grand symposium (verified vs upstream/evaluate.py): d_seg=0.00056, with
# the residual distortion as pose. We BACK OUT d_pose from S so the anchor is internally
# consistent with its measured S (no invented pose number).
def _frontier_anchor() -> MeasuredAnchor:

    s_total = 0.19109982419209975
    bytes_ = 177_169
    d_seg = 0.00056  # symposium component analysis (vs upstream/evaluate.py)
    residual_pose_term = s_total - rate_term(bytes_) - SEG_WEIGHT * d_seg
    # pose_term = sqrt(10*d_pose) -> d_pose = pose_term^2 / 10
    d_pose = max(0.0, residual_pose_term) ** 2 / POSE_WEIGHT_INNER
    return MeasuredAnchor(
        label="frontier_pr110_entropy_recode",
        base_ch=None,
        decoder_params=None,
        d_seg=d_seg,
        d_pose=d_pose,
        archive_bytes=bytes_,
        provenance=(
            "canonical_frontier_pointer.json (S=0.19110, 177169 B, [contest-CPU]); "
            "d_seg=0.00056 from 2026-06-16 grand symposium component analysis; "
            "d_pose backed out from S for internal consistency"
        ),
    )


ANCHOR_FRONTIER = _frontier_anchor()


# ---------------------------------------------------------------------------
# MEASURED QAT byte-shrink ratios (post-int8-brotli archive bytes, bc20).
# From reports/fp_shrink_ptq_bc20_n600.json (real vendored build_archive re-encode +
# real brotli). The PTQ DISTORTION collapses (that's why we QAT), but the BYTE ratio is
# the same grid QAT will store on, so the byte saving carries over to QAT.
# ---------------------------------------------------------------------------

# bit-width -> (archive_bytes, fraction of int8 bytes). int8 is the codec baseline.
MEASURED_BYTE_SHRINK_BC20 = {
    8: (89_136, 1.0),
    7: (78_750, 78_750 / 89_136),  # -11.6%
    6: (68_400, 68_400 / 89_136),  # -23.3%
    5: (57_475, 57_475 / 89_136),  # -35.5%
    4: (46_590, 46_590 / 89_136),  # -47.7%
}


def qat_byte_fraction(nbits: int) -> float:
    """Fraction of the int8 archive bytes a uniform int-N grid yields (measured bc20).
    This is the BYTE lever; QAT preserves the byte saving because it stores on the same
    grid. For a SCORE-AWARE mixed grid (some tensors int8, some int4) the realised
    fraction is between the two endpoints; ``qat_byte_fraction_mixed`` models that."""
    if nbits not in MEASURED_BYTE_SHRINK_BC20:
        raise ValueError(f"no measured byte-shrink ratio for nbits={nbits}; have {sorted(MEASURED_BYTE_SHRINK_BC20)}")
    return MEASURED_BYTE_SHRINK_BC20[nbits][1]


def qat_byte_fraction_mixed(frac_low_precision: float, low_nbits: int = 4) -> float:
    """Score-aware mixed-precision byte fraction: a fraction ``frac_low_precision`` of the
    (d_seg-blind) weights go to ``low_nbits``, the rest stay int8. Linear blend of the
    measured int8 (1.0) and low_nbits byte fractions — an APPROXIMATION (brotli is not
    perfectly linear, but the measured endpoints bound it). ``frac_low_precision=1.0``
    reduces to the uniform low_nbits fraction; 0.0 is int8."""
    if not 0.0 <= frac_low_precision <= 1.0:
        raise ValueError("frac_low_precision must be in [0,1]")
    low = qat_byte_fraction(low_nbits)
    return (1.0 - frac_low_precision) * 1.0 + frac_low_precision * low


# ---------------------------------------------------------------------------
# Frontier (pr110) archive section split — MEASURED from the on-disk frontier archive.
# QAT targets the DECODER section only (latents/sidecar/selector are kept verbatim).
# ---------------------------------------------------------------------------

# Measured 2026-06-18 by parsing the FP11 + CTXR grammar of the contest-CPU frontier
# archive (experiments/results/pr110_payload_entropy_recode_20260610/submission_dir/archive.zip).
FRONTIER_ARCHIVE_BYTES = 177_169
FRONTIER_DECODER_SECTION_BYTES = 161_104  # 90.9% — the QAT-attackable share
FRONTIER_LATENT_SECTION_BYTES = 15_070  #  8.5% — kept verbatim
FRONTIER_SIDECAR_BYTES = 607
FRONTIER_OTHER_BYTES = (
    FRONTIER_ARCHIVE_BYTES - FRONTIER_DECODER_SECTION_BYTES - FRONTIER_LATENT_SECTION_BYTES - FRONTIER_SIDECAR_BYTES
)  # selector + dqs1 + header (kept verbatim)


@dataclass(frozen=True)
class FrontierQatRow:
    qat_nbits: int
    frac_low_precision: float
    decoder_section_bytes: int  # QAT-shrunk decoder section
    qat_archive_bytes: int  # whole archive after shrinking only the decoder section
    d_seg_hold_delta: float
    qat_S_perfect_hold: float  # S if QAT holds d_seg at the float floor exactly
    qat_S_with_spill: float  # S if QAT spills +delta d_seg (the break-even budget)
    decoder_shrink_fraction: float


def frontier_qat_rows(
    *,
    nbits_options: tuple[int, ...] = (4, 5, 6),
    frac_low_options: tuple[float, ...] = (0.70, 1.0),
    d_seg_hold_delta: float = 0.0003,
) -> list[FrontierQatRow]:
    """Model QAT-shrinking the FRONTIER decoder section. The decoder is already
    range-coded near the entropy floor of its FP11 grid; a SPARSER int-N grid lowers the
    entropy further, and the bc20 MEASURED whole-archive int8->int-N ratios are the best
    available transfer estimate for the decoder-section shrink (the decoder dominates both
    archives, so the whole-archive ratio ~= the decoder-section ratio). FLAGGED as a
    transfer assumption: the frontier weights are at ~FP11 (denser than int8), so going to
    int4 is a BIGGER precision drop than bc20's int8->int4 — the d_seg-hold burden is
    HEAVIER here; only a real QAT measurement closes it.

    Only the decoder section shrinks; latents + sidecar + selector + dqs1 are kept verbatim
    (lossless, distortion-neutral)."""
    rows: list[FrontierQatRow] = []
    fixed = FRONTIER_LATENT_SECTION_BYTES + FRONTIER_SIDECAR_BYTES + FRONTIER_OTHER_BYTES
    for nbits in nbits_options:
        for frac in frac_low_options:
            dec_frac = qat_byte_fraction_mixed(frac, low_nbits=nbits)
            dec_bytes = round(FRONTIER_DECODER_SECTION_BYTES * dec_frac)
            arch_bytes = dec_bytes + fixed
            s_perfect = score(ANCHOR_FRONTIER.d_seg, ANCHOR_FRONTIER.d_pose, arch_bytes)
            s_spill = score(
                ANCHOR_FRONTIER.d_seg + d_seg_hold_delta,
                ANCHOR_FRONTIER.d_pose,
                arch_bytes,
            )
            rows.append(
                FrontierQatRow(
                    qat_nbits=nbits,
                    frac_low_precision=frac,
                    decoder_section_bytes=dec_bytes,
                    qat_archive_bytes=arch_bytes,
                    d_seg_hold_delta=d_seg_hold_delta,
                    qat_S_perfect_hold=s_perfect,
                    qat_S_with_spill=s_spill,
                    decoder_shrink_fraction=dec_frac,
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Native byte model B(p) and d_seg(p) interpolation across capacity.
# ---------------------------------------------------------------------------


def native_archive_bytes(base_ch: int, latent_dim: int = 28) -> int:
    """Estimate native int8-brotli archive bytes for ``base_ch`` from the EXACT decoder
    param count, calibrated to the bc20 MEASURED anchor (89136 B at 83356 params) plus a
    bc-independent latent/header constant. Uses the same calibration the 2026-06-16 memo
    used: archive ~= 0.89 B/decoder-param + ~15 KB const. We FIT both constants to the two
    measured anchors (bc20 + frontier param-equivalent) where possible; here we use the
    memo's calibration directly (bc20 anchor exact, slope from the memo table)."""
    dec = decoder_param_count(vendored_taper(base_ch), latent_dim=latent_dim)
    # Calibrate: bc20 dec=83356 -> 89136 archive. Memo slope ~0.89 B/param + ~15KB const.
    # Solve const from the bc20 anchor at slope 0.89: 89136 = 0.89*83356 + const.
    slope = 0.89
    const = ANCHOR_BC20.archive_bytes - slope * ANCHOR_BC20.decoder_params
    return round(slope * dec + const)


def dseg_at_capacity(base_ch: int) -> tuple[float, str]:
    """d_seg(p) at ``base_ch`` — MEASURED where we have it, else a HONEST interpolation
    between the two measured anchors on the (decoder_params, d_seg) curve.

    MEASURED: bc20 -> 0.0026 (clean basin). The frontier reaches d_seg 0.00056 at a
    frontier-class param count (pr110's underlying HNeRV is ~228K decoder params, base_ch
    ~36-class). We model d_seg(p) as a power law in decoder_params fit to the TWO measured
    endpoints: d_seg = A * params^(-gamma). This is the ONLY honest extrapolation we have
    and it is flagged as MODELLED, not measured. Returns (d_seg, evidence_tag)."""

    dec = decoder_param_count(vendored_taper(base_ch))
    if base_ch == ANCHOR_BC20.base_ch:
        return ANCHOR_BC20.d_seg, "MEASURED(bc20 basin)"
    # Two-point power-law fit on (decoder_params, d_seg): bc20 anchor + frontier anchor.
    # Frontier underlying decoder ~ base_ch36-class (228958 params per the memo table).
    p1, s1 = float(ANCHOR_BC20.decoder_params), ANCHOR_BC20.d_seg
    p2 = float(decoder_param_count(vendored_taper(36)))  # frontier param-class proxy
    s2 = ANCHOR_FRONTIER.d_seg
    gamma = -(math.log(s2 / s1)) / (math.log(p2 / p1))
    A = s1 / (p1 ** (-gamma))
    d_seg = A * dec ** (-gamma)
    return d_seg, f"MODELLED(power-law A={A:.4g} gamma={gamma:.3f} on bc20+frontier endpoints)"


# ---------------------------------------------------------------------------
# The desk model.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapacityRow:
    base_ch: int
    decoder_params: int
    d_seg: float
    d_seg_evidence: str
    d_pose: float
    native_bytes: int
    native_S: float
    qat_nbits: int
    qat_frac_low_precision: float
    qat_bytes: int
    qat_S: float
    qat_d_seg_hold_delta: float  # the +delta to d_seg the QAT-hold assumption budgets


@dataclass(frozen=True)
class DeskCalcResult:
    rows: list[CapacityRow]
    argmin_native: CapacityRow
    argmin_qat: CapacityRow
    bc20_native_S: float
    frontier_S: float
    proceed_threshold: float
    proceed: bool
    chosen: CapacityRow | None
    notes: list[str] = field(default_factory=list)


def run_desk_calc(
    *,
    base_chs: tuple[int, ...] = (20, 24, 28, 32, 36),
    d_pose_hold: float = 0.00034168662969022987,
    qat_nbits: int = 4,
    qat_frac_low_precision: float = 0.70,
    qat_d_seg_hold_delta: float = 0.0003,
    proceed_threshold: float = 0.30,
) -> DeskCalcResult:
    """Compute S_native(p) and S_QAT(p) across capacities and emit the PROCEED/STOP gate.

    Args:
      base_chs: capacities to sweep.
      d_pose_hold: the pose the design says we HOLD on the trunk (basin value). Held
        constant across capacities (pose is decoupled from capacity, per the design).
      qat_nbits: the low-precision target for the d_seg-blind weights (4 or 5).
      qat_frac_low_precision: fraction of weights pushed to low precision (score-aware:
        the d_seg-blind ~70% to int4, the d_seg-critical ~30% kept int8). 0.70 is the
        design-memo target ("coarsen the d_seg-blind/stem-Nyquist weights to int4").
      qat_d_seg_hold_delta: the d_seg the QAT-hold assumption ADDS to the float floor
        (the break-even contract: "hold d_seg within +0.0003 of the float floor").
      proceed_threshold: STOP if the best S_QAT >= this. The pivot needs a vehicle whose
        QAT-shrunk S is promising; 0.30 is a generous "worth training" bar (bc20 native
        is 0.378; sub-0.15 is the target).

    Returns a DeskCalcResult with the full table + the gate verdict.
    """
    rows: list[CapacityRow] = []
    for bc in base_chs:
        dec = decoder_param_count(vendored_taper(bc))
        d_seg_float, ev = dseg_at_capacity(bc)
        native_bytes = native_archive_bytes(bc)
        native_S = score(d_seg_float, d_pose_hold, native_bytes)

        # QAT: bytes shrink by the score-aware mixed fraction; d_seg holds within +delta.
        qat_frac = qat_byte_fraction_mixed(qat_frac_low_precision, low_nbits=qat_nbits)
        qat_bytes = round(native_bytes * qat_frac)
        d_seg_qat = d_seg_float + qat_d_seg_hold_delta
        qat_S = score(d_seg_qat, d_pose_hold, qat_bytes)

        rows.append(
            CapacityRow(
                base_ch=bc,
                decoder_params=dec,
                d_seg=d_seg_float,
                d_seg_evidence=ev,
                d_pose=d_pose_hold,
                native_bytes=native_bytes,
                native_S=native_S,
                qat_nbits=qat_nbits,
                qat_frac_low_precision=qat_frac_low_precision,
                qat_bytes=qat_bytes,
                qat_S=qat_S,
                qat_d_seg_hold_delta=qat_d_seg_hold_delta,
            )
        )

    argmin_native = min(rows, key=lambda r: r.native_S)
    argmin_qat = min(rows, key=lambda r: r.qat_S)
    bc20_native_S = next(r.native_S for r in rows if r.base_ch == 20)
    proceed = argmin_qat.qat_S < proceed_threshold
    chosen = argmin_qat if proceed else None

    notes = [
        "ALL numbers [advisory] NON-PROMOTABLE; exact pointer stays pointer-only (0.19110).",
        f"d_seg(p) is MEASURED at bc20 ({ANCHOR_BC20.d_seg:.4g}) and MODELLED elsewhere via a "
        "two-point power law on the bc20+frontier endpoints — the single honest extrapolation.",
        "QAT byte fractions are MEASURED (bc20 post-int8-brotli, fp_shrink_ptq_bc20_n600.json); "
        "the score-aware mixed fraction LINEAR-BLENDS the measured int8/int4 endpoints (approx).",
        "QAT distortion-hold is an ASSUMPTION (the break-even contract): real QAT must be MEASURED. "
        "If QAT cannot hold d_seg within +delta of the float floor at the chosen grid, this table is void.",
        "Pose held constant on the trunk (basin d_pose) across capacities per the design (pose decoupled).",
    ]

    return DeskCalcResult(
        rows=rows,
        argmin_native=argmin_native,
        argmin_qat=argmin_qat,
        bc20_native_S=bc20_native_S,
        frontier_S=0.19109982419209975,
        proceed_threshold=proceed_threshold,
        proceed=proceed,
        chosen=chosen,
        notes=notes,
    )
