#!/usr/bin/env python3
"""ddm_tac1 — the two-axis composition table for the dx2 object.

DERIVATION ONLY. This script measures nothing. It performs exact arithmetic over
MEASURED inputs, each cited to a receipt in ``AXES`` / ``CUTS`` below, and emits the
per-pair iso-0.12 surface for every unordered pair of the dx2 object's eight score axes.

Every fraction it prints is a REQUIREMENT, not an achieved value. No arm has achieved
any of them. See the memo's NOT-CLAIMED section.

Exchange rate is CITED from ``.omx/research/ddm_tx1_toolbox_crosswalk_20260819.md`` §0
(25/37_545_489 S/B) and is NOT re-derived here.

Usage:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_tac1_two_axis_composition.py \
        --out-dir /Volumes/APDataStore/pact/ddm_tac1_two_axis_composition
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import itertools
import json
import math
import pathlib
import sys
from typing import Callable

# --------------------------------------------------------------------------------------
# MEASURED constants. Each carries its receipt. None of these is re-derived here.
# --------------------------------------------------------------------------------------

# CITED, not re-derived: ddm_tx1_toolbox_crosswalk_20260819.md section 0.
EXCHANGE_NUM = 25
EXCHANGE_DEN = 37_545_489
RATE_S_PER_BYTE = EXCHANGE_NUM / EXCHANGE_DEN  # 6.658589531221714e-07

# dx2 authority row [contest-CUDA T4, n600], archive sha
# 976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674.
DX2_ARCHIVE_BYTES = 180_368
DX2_D_SEG = 0.00020139
DX2_D_POSE = 0.00000637
DX2_S_PUBLISHED = 0.14821987563243377

TARGET_S = 0.12

# ms9 exact field replay (n600, 117,964,800 px): manufactured / representation split.
MS9_TOTAL_FLIPS = 23_757
MS9_MANUFACTURED_FLIPS = 21_493
MS9_REPRESENTATION_SURVIVED_FLIPS = 2_264
# mst1 stage split: share of the 21,493 manufactured errors born at the native render.
MST1_NATIVE_RENDER_FLIPS = 16_917

SEG_PIXELS = 117_964_800


def s_rate(archive_bytes: float) -> float:
    """Rate term of S for an archive of ``archive_bytes`` bytes."""
    return EXCHANGE_NUM * archive_bytes / EXCHANGE_DEN


def s_pose(d_pose: float) -> float:
    """Pose term of S. sqrt(10 * d_pose) -- nonlinear, handled exactly."""
    return math.sqrt(10.0 * d_pose)


def s_seg(d_seg: float) -> float:
    """Seg term of S."""
    return 100.0 * d_seg


def recompute_s(archive_bytes: float, d_seg: float, d_pose: float) -> float:
    """S recomputed FROM COMPONENTS. Never from a rounded display."""
    return s_rate(archive_bytes) + s_seg(d_seg) + s_pose(d_pose)


# --------------------------------------------------------------------------------------
# The axis inventory.
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class Axis:
    """One score axis of the dx2 object.

    ``max_saving_s`` is the ARITHMETIC ceiling: the S this axis would return if it were
    driven to its idealised limit (a byte pool to zero bytes; a distortion term to zero).
    It is deliberately the MOST GENEROUS bound, so an ``IMPOSSIBLE`` verdict computed
    against it is robust. It is NOT an achievable value and no arm has achieved it.
    """

    key: str
    label: str
    kind: str  # "rate" | "distortion"
    nominal: str
    max_saving_s: float
    saving_fn: Callable[[float], float]
    measured_status: str
    owning_memo: str
    nominal_bytes: int | None = None
    # Structural ceiling: the part of the axis a MEASURED row says is actually available.
    structural_ceiling_s: float | None = None
    structural_note: str = ""

    def saving(self, fraction: float) -> float:
        return self.saving_fn(fraction)


def _linear(max_s: float) -> Callable[[float], float]:
    def fn(fraction: float) -> float:
        return fraction * max_s

    return fn


def _pose_saving(fraction: float) -> float:
    """Pose saving is CONCAVE-DEFICIENT: removing fraction f of d_pose returns only
    P0 * (1 - sqrt(1 - f)) of S, which is strictly LESS than f * P0 for f < 1.
    Handled exactly; never linearised."""
    p0 = s_pose(DX2_D_POSE)
    return p0 - math.sqrt(10.0 * DX2_D_POSE * (1.0 - fraction))


def build_axes() -> list[Axis]:
    manufactured_frac = MS9_MANUFACTURED_FLIPS / MS9_TOTAL_FLIPS
    native_frac = MST1_NATIVE_RENDER_FLIPS / MS9_TOTAL_FLIPS
    seg_total_s = s_seg(DX2_D_SEG)
    pose_total_s = s_pose(DX2_D_POSE)

    rows: list[Axis] = []

    def rate_axis(key, label, nbytes, status, memo, ceiling_bytes=None, note=""):
        max_s = nbytes * RATE_S_PER_BYTE
        rows.append(
            Axis(
                key=key,
                label=label,
                kind="rate",
                nominal=f"{nbytes:,} B",
                max_saving_s=max_s,
                saving_fn=_linear(max_s),
                measured_status=status,
                owning_memo=memo,
                nominal_bytes=nbytes,
                structural_ceiling_s=(
                    None if ceiling_bytes is None else ceiling_bytes * RATE_S_PER_BYTE
                ),
                structural_note=note,
            )
        )

    rate_axis(
        "tokens",
        "RC64 token stream (the transmitted label field)",
        113_777,
        "SHARP local optimum: 5 concordant arms measured every tested direction negative; "
        "ld1's lossy Lane rungs make the archive BIGGER (+196..+1,528 B)",
        "ar1b (census) / oe1+ld1+ae1+ni1+wj1 (sharp-optimum law)",
        note="no measured rung returns a single byte on this field under the frozen model",
    )
    rate_axis(
        "renderer",
        "semantic renderer packet (SM3R)",
        30_856,
        "3 retained rungs: w72 -10,879 B REFUSED 46.3x; svd_r32 -5,191 B dead; "
        "film_flat_w96 -1,078 B UNMEASURED",
        "ar1b (census) / rj1 (rungs) / w72 (measured refusal)",
    )
    rate_axis(
        "carrier",
        "CAP1 carrier + frame-0 selector",
        22_010,
        "ap1 measured 3 coarsening depths: 2,742 / 5,875 / 9,035 B credit, ALL net-positive; "
        "SegNet-inert at every depth (0 flips in every class); pose-load-bearing",
        "ar1b (census) / ap1 (measured purchase table)",
    )
    rate_axis(
        "hpac",
        "HPAC / IHS1 probability model for the tokens",
        13_515,
        "part of the sharp optimum on this field; ap1 coarsening 1,912 / 3,908 / 6,026 B "
        "credit, all net-positive at 81,587x / 32,967x / 21,772x the going rate",
        "ar1b (census) / ap1 / oe1+ae1 (model-direction closures)",
    )
    rate_axis(
        "framing",
        "ZIP + RX1 structural framing",
        114,
        "receiver-required structure; ap1 held it exact and assigned it no coarsening credit",
        "ar1b (census) / ap1",
        ceiling_bytes=0,
        note="MEASURED-MOVABLE CEILING 0 B: receiver-required framing, not a parameter group",
    )
    rate_axis(
        "residual",
        "fixed compact residual table (fp16 scale + 125 signed six-bit codes)",
        96,
        "ap1 coarsened it at 3 lattice levels: 0 B ZIP credit at EVERY level while causing "
        "catastrophic distortion (d_pose +144.8 at level 1)",
        "ar1b (census) / ap1",
        ceiling_bytes=0,
        note="MEASURED-MOVABLE CEILING 0 B: three measured levels each returned 0 archive bytes",
    )

    seg = Axis(
        key="seg",
        label="SegNet distortion term (100 * d_seg)",
        kind="distortion",
        nominal=f"d_seg {DX2_D_SEG} -> {seg_total_s:.6f} S",
        max_saving_s=seg_total_s,
        saving_fn=_linear(seg_total_s),
        measured_status=(
            f"{manufactured_frac * 100:.4f}% manufactured after a correct transmitted label "
            f"(ms9, {MS9_MANUFACTURED_FLIPS:,}/{MS9_TOTAL_FLIPS:,}); of the whole error "
            f"{native_frac * 100:.4f}% is born at the native render (mst1)"
        ),
        owning_memo="ms9 (manufactured split) / mst1 (stage split)",
        structural_ceiling_s=manufactured_frac * seg_total_s,
        structural_note=(
            f"STRUCTURAL CEILING {manufactured_frac * seg_total_s:.8f} S: the remaining "
            f"{100 - manufactured_frac * 100:.4f}% ({MS9_REPRESENTATION_SURVIVED_FLIPS:,} flips) "
            "is representation error carried by the transmitted labels -- it is removable only "
            "through the TOKEN axis, not by any downstream repair"
        ),
    )
    rows.append(seg)

    pose = Axis(
        key="pose",
        label="PoseNet distortion term (sqrt(10 * d_pose))",
        kind="distortion",
        nominal=f"d_pose {DX2_D_POSE} -> {pose_total_s:.7f} S",
        max_saving_s=pose_total_s,
        saving_fn=_pose_saving,
        measured_status=(
            "no measured reducing mechanism on this object; qs5 in-compile Schur compensation "
            "is proven on ITS object only. Saving is CONCAVE-DEFICIENT in the fraction removed."
        ),
        owning_memo="dx2 authority row / sy2 (compensation scope) / w72 (renderer carries pose)",
        structural_note=(
            "removing fraction f of d_pose returns only P0*(1-sqrt(1-f)) S; at f=0.50 that is "
            f"{(_pose_saving(0.5) / pose_total_s) * 100:.2f}% of the term, not 50%"
        ),
    )
    rows.append(pose)

    return rows


# --------------------------------------------------------------------------------------
# The pair surface.
# --------------------------------------------------------------------------------------


def solve_equal_fraction(ax_a: Axis, ax_b: Axis, gap: float) -> float | None:
    """Smallest t in (0,1] with saving_A(t) + saving_B(t) >= gap.

    Because both saving functions are monotone non-decreasing, the minimum of
    max(alpha, beta) over the feasible set is attained at alpha = beta = t. So t is the
    single number 'the fraction each axis must give'.
    Returns None when the pair cannot reach the gap at t = 1.
    """
    if ax_a.saving(1.0) + ax_b.saving(1.0) <= gap:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ax_a.saving(mid) + ax_b.saving(mid) >= gap:
            hi = mid
        else:
            lo = mid
    return hi


def solve_corner(ax_solo: Axis, ax_perfect: Axis, gap: float) -> float | None:
    """Fraction of ``ax_solo`` required when ``ax_perfect`` is driven to its full limit."""
    residual = gap - ax_perfect.saving(1.0)
    if residual <= 0.0:
        return 0.0
    if ax_solo.saving(1.0) < residual:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ax_solo.saving(mid) >= residual:
            hi = mid
        else:
            lo = mid
    return hi


# --------------------------------------------------------------------------------------
# The measured feasibility cuts, one per pair. Each names its memo and its number.
# `cut` is one of: IMPOSSIBLE | CONTRADICTED | TENSION | OPEN.
# `sy2` is one of: STACK | OBJECT-CHANGE | DEGENERATE (a leg with no actuator inside the
# pair, i.e. a disguised triple) -- per ddm_sy2 commit fe2ba12dc2: "a closed leg survives
# only when another leg first changes the object it prices."
# --------------------------------------------------------------------------------------

CUTS: dict[str, dict[str, str]] = {
    "tokensxhpac": {
        "cut": "OPEN",
        "sy2": "OBJECT-CHANGE",
        "sy2_basis": (
            "HPAC is the probability model FOR the token field; changing the field changes "
            "the object the model was fitted to. ap1 explicitly excludes this diagonal from "
            "its closure: 'it lies outside AP1's isolation rule'."
        ),
        "evidence": (
            "jf1 (committed memo, ep2 of 60): mandatory positive control FAILS by +7,554 B "
            "(epoch-2 refit stream 121,331 B vs shipped 113,777 B on the UNMODIFIED field). "
            "Best diagonal rung k060000 = 130,007 B model+stream, +2,715 B over the 127,292 B "
            "bar -- it CONSUMES 6.406021% of the demand instead of supplying it. All four "
            "rungs lose. Scope-reduced negative, terminal fit pending."
        ),
        "cheapest_deciding_measurement": (
            "Harvest jf1's sealed epoch-60 null + diagonal, pack each real model+stream, and "
            "compare ABSOLUTE bytes to the shipped 127,292 B bar (not to the same-epoch null). "
            "Cost: marginal $0 -- already fired and retained. This is sy2's rank-1 fire row."
        ),
    },
    "tokensxseg": {
        "cut": "TENSION",
        "sy2": "DEGENERATE",
        "sy2_basis": (
            "seg's only actuators are the token field and the renderer; inside this pair the "
            "seg leg's actuator IS the token leg. One actuator read on two score terms, not "
            "two axes."
        ),
        "evidence": (
            "ld1: six correct-Lane-to-Road field edits under the frozen model -- edits that "
            "DEGRADE seg -- every one made the archive BIGGER (+196/+279/+824/+1,528/+598/"
            "+21 B). The demanded corner (smaller AND more accurate) is the opposite end of a "
            "measured monotone. Structural: tokens can repair only the 9.5298% "
            "representation-survived seg (2,264 flips, 0.00191922 S); the other 90.4702% is "
            "manufactured downstream of already-correct labels (ms9) and is unreachable here."
        ),
        "cheapest_deciding_measurement": (
            "None cheap: the pair is one actuator. The deciding row is the token leg's, which "
            "is the sharp-optimum question, already answered by five concordant arms."
        ),
    },
    "rendererxseg": {
        "cut": "TENSION",
        "sy2": "OBJECT-CHANGE",
        "sy2_basis": (
            "the renderer MAKES the frames SegNet argmaxes, so moving it changes the object "
            "seg is priced on. Passes sy2 -- but the seg leg's actuator is the SAME renderer "
            "(mst1: 78.7093% of manufactured error is born at the native render), so the pair "
            "demands one object be simultaneously smaller and more accurate."
        ),
        "evidence": (
            "w72 MEASURED: shedding 10,879 B = 35.2573% of the renderer multiplied d_seg by "
            "116.8x (0.00020139 -> 0.02351655) and d_pose by 303,989x; S 46.3x worse. That is "
            "a SMALLER shed than the pair demands, in the WRONG direction. rj1's wider W64 "
            "rung refused 3.51x with d_pose 97.70% of the damage."
        ),
        "cheapest_deciding_measurement": (
            "The unmeasured film_amortized_flat_w96 rung (rj1, -1,078 B, archive 179,290 B, "
            "sha 34855e3c...): one n600 row on already-retained bytes. It cannot be the route "
            "(2.5435% of demand at ZERO distortion) but it is the only renderer rung whose "
            "distortion sign is unknown, and it prices the small-shed end of the tension curve."
        ),
    },
    "rendererxpose": {
        "cut": "TENSION",
        "sy2": "DEGENERATE",
        "sy2_basis": (
            "pose's actuator inside this pair IS the renderer (w72 s3, three independent "
            "confirmations: PoseNet scores the rendered FRAMES). One actuator read twice."
        ),
        "evidence": (
            "Clears only at the far corner and by 461 B: it demands 99.88% of the renderer's "
            "bytes AND 99.88% of its d_pose. w72 MEASURED d_pose x303,989 at 35.2573% shed."
        ),
        "cheapest_deciding_measurement": (
            "Same film_amortized_flat_w96 n600 row as rendererxseg -- one measurement decides "
            "both, because both are the same actuator."
        ),
    },
    "rendererxcarrier": {
        "cut": "TENSION",
        "sy2": "OBJECT-CHANGE",
        "sy2_basis": (
            "sy2 names it: moving the renderer changes the residual coordinate system the "
            "carrier is fitted to ('renderer-relative shared latent')."
        ),
        "evidence": (
            "Demands 80.17% of the combined 52,866 B pool. Both legs measured refusing far "
            "below that: renderer at 35.2573% -> 46.3x (w72); carrier at 41.05% -> 652.9x the "
            "going rate (ap1 carrier_l3). The pair has NO distortion leg, yet BOTH measured "
            "refusals are paid in distortion. sy2's own arithmetic: even the smallest renderer "
            "rung (W96, -1,078 B) needs the carrier <= 9,204 B just to TIE dx2, and sub-0.12 "
            "at W96's components needs archive ~124,102 B = -55,188 B, more than the whole pool."
        ),
        "cheapest_deciding_measurement": (
            "sy2's rank-2 row: one complete W96 archive after a fresh carrier solve and "
            "in-compile Schur compensation, then one full-component n600 row on those exact "
            "bytes. Not cheap -- one resumable joint materialisation plus one n600 row."
        ),
    },
    "carrierxseg": {
        "cut": "CONTRADICTED",
        "sy2": "STACK",
        "sy2_basis": (
            "MEASURED stack: ap1 recorded 0 flips in EVERY one of the five GT classes at ALL "
            "three carrier coarsening depths. The carrier does not change the object seg is "
            "priced on, so seg's measured closure transfers unchanged."
        ),
        "evidence": (
            "ap1: carrier is SegNet-inert (0/0 per class x 3 depths) and the seg leg therefore "
            "has no actuator inside the pair -- a disguised triple. The carrier's cheapest "
            "measured rung is 167.8x over the going rate at 12.46% of the pool, against 81.10% "
            "demanded."
        ),
        "cheapest_deciding_measurement": "none warranted; the stack verdict is measured.",
    },
    "rendererxhpac": {
        "cut": "CONTRADICTED",
        "sy2": "STACK",
        "sy2_basis": (
            "MEASURED stack: w72's inflate receipt records decoded_token_sha256 "
            "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb -- BIT-IDENTICAL "
            "to dx2's field. A renderer move does not change the object HPAC prices."
        ),
        "evidence": (
            "Demands 95.52% of BOTH pools. Measured refusals at 35.2573% (renderer, 46.3x) and "
            "14.15% (hpac_l1, 81,587x the going rate)."
        ),
        "cheapest_deciding_measurement": "none warranted; the stack verdict is measured.",
    },
    "tokensxrenderer": {
        "cut": "CONTRADICTED",
        "sy2": "STACK",
        "sy2_basis": (
            "MEASURED stack: w72's decoded token field is bit-identical to dx2's, so the "
            "renderer does not change the token object; and the token field does not price the "
            "renderer's bytes. Two independently-priced pools on one unchanged object."
        ),
        "evidence": (
            "Token leg has no measured rung returning a single byte (five concordant arms; "
            "ld1's rungs make the archive BIGGER). Renderer leg's best measured rung refused "
            "46.3x at 35.2573%, against 29.30% demanded of each."
        ),
        "cheapest_deciding_measurement": "none warranted; both legs measured refusing.",
    },
    "tokensxcarrier": {
        "cut": "CONTRADICTED",
        "sy2": "STACK",
        "sy2_basis": (
            "At the measured scope the carrier is SegNet-inert (ap1) and the token field does "
            "not price carrier bytes: two independently-priced pools. NAMED UNMEASURED ESCAPE: "
            "a field change alters the render, so a RE-SOLVED carrier is priced on a changed "
            "object -- no row on this object prices that."
        ),
        "evidence": (
            "Token leg: no mechanism (five arms). Carrier leg at 31.21% demanded; nearest "
            "measured rung (41.05%) cost 652.9x the going rate."
        ),
        "cheapest_deciding_measurement": (
            "folded into rendererxcarrier's carrier re-solve row; no independent cheap row."
        ),
    },
    "tokensxpose": {
        "cut": "CONTRADICTED",
        "sy2": "DEGENERATE",
        "sy2_basis": (
            "pose has no actuator inside this pair. Its measured actuators are the renderer "
            "(w72), the carrier and the residual (ap1). w72 isolated renderer->pose with the "
            "tokens held bit-identical; that gives the token axis no pose actuator."
        ),
        "evidence": "token leg: no measured rung returns a byte (five concordant arms).",
        "cheapest_deciding_measurement": "none warranted.",
    },
    "tokensxframing": {
        "cut": "CONTRADICTED",
        "sy2": "STACK",
        "sy2_basis": "two independently-priced pools; framing is receiver-required structure.",
        "evidence": (
            "framing's MEASURED-MOVABLE CEILING is 0 B (ar1b/ap1: receiver-required structure, "
            "held exact, assigned no coarsening credit), so the pair degenerates to tokens "
            "alone -- a SINGLE axis with no measured mechanism."
        ),
        "cheapest_deciding_measurement": "none warranted.",
    },
    "tokensxresidual": {
        "cut": "CONTRADICTED",
        "sy2": "STACK",
        "sy2_basis": "two independently-priced pools.",
        "evidence": (
            "residual's MEASURED-MOVABLE CEILING is 0 B: ap1 coarsened it at three lattice "
            "levels and every level returned 0 archive bytes while causing catastrophic "
            "distortion (d_pose +144.8 at level 1). The pair degenerates to tokens alone."
        ),
        "cheapest_deciding_measurement": "none warranted.",
    },
    "hpacxseg": {
        "cut": "IMPOSSIBLE",
        "sy2": "DEGENERATE",
        "sy2_basis": (
            "HPAC has no non-destructive seg actuator: coarsening it breaks exact token "
            "reconstruction (ap1 hpac_l1: d_seg +0.6707, d_pose +135.5). seg's actuators are "
            "tokens and renderer, neither of which is in this pair."
        ),
        "evidence": (
            "IMPOSSIBLE at the MEASURED structural seg ceiling: 0.00899908 (hpac) + 0.01821979 "
            "(manufactured seg, 90.4702% of the authority d_seg) = 0.02721887 S < 0.02821988 S "
            "gap, short by 0.00100100 S = 1,503.3 B. Feasible only by borrowing the 9.5298% "
            "representation-survived seg, which lives in the transmitted labels -- a third axis."
        ),
        "cheapest_deciding_measurement": "none; arithmetically closed at the measured ceiling.",
    },
}


# Which actuator moves which score term. Every cell is MEASURED or explicitly not.
READOUT_MATRIX = {
    "tokens": {
        "rate": "YES 113,777 B (ar1b census)",
        "seg": "YES but only the 9.5298% representation-survived part (2,264 flips, "
        "0.00191922 S) -- ms9",
        "pose": "NOT ISOLATED by any row on this object",
    },
    "renderer": {
        "rate": "YES 30,856 B (ar1b census)",
        "seg": "YES, dominant: 78.7093% of manufactured error born at the native render (mst1)",
        "pose": "YES, dominant: w72 measured d_pose x303,989 with tokens bit-identical",
    },
    "carrier": {
        "rate": "YES 22,010 B (ar1b census)",
        "seg": "NO -- MEASURED ZERO: ap1 recorded 0 flips in all 5 classes at all 3 depths",
        "pose": "YES: ap1 d_pose +0.00987 / +0.05466 / +1.54927 at the three depths",
    },
    "hpac": {
        "rate": "YES 13,515 B (ar1b census)",
        "seg": "ONLY DESTRUCTIVELY: coarsening breaks exact token reconstruction "
        "(ap1 hpac_l1 d_seg +0.6707)",
        "pose": "ONLY DESTRUCTIVELY (ap1 hpac_l1 d_pose +135.49)",
    },
    "framing": {
        "rate": "NO -- receiver-required structure, movable ceiling 0 B",
        "seg": "n/a",
        "pose": "n/a",
    },
    "residual": {
        "rate": "NO -- MEASURED 0 B ZIP credit at all 3 ap1 levels",
        "seg": "ONLY DESTRUCTIVELY (ap1 residual_l1 d_seg +0.4066)",
        "pose": "ONLY DESTRUCTIVELY (ap1 residual_l1 d_pose +144.84)",
    },
}


def fb1_corner_cross_checks(gap: float) -> dict[str, object]:
    """Independently reproduce fb1 section 5 (commit 9c137a91ed) from my own constants.

    This is the charter's required independent cross-check: if my arithmetic is right it
    must land on fb1's published table without having copied it.
    """
    manufactured_frac = MS9_MANUFACTURED_FLIPS / MS9_TOTAL_FLIPS
    renderer_bytes = 30_856
    rows = []
    for f in (0.00, 0.25, 0.50, 0.75, 1.00):
        d_seg_after = DX2_D_SEG * (1.0 - manufactured_frac * f)
        s_at_full_archive = recompute_s(DX2_ARCHIVE_BYTES, d_seg_after, DX2_D_POSE)
        bytes_needed = (s_at_full_archive - TARGET_S) / RATE_S_PER_BYTE
        rows.append(
            {
                "f_manufactured_seg_removed": f,
                "d_seg_after": d_seg_after,
                "renderer_bytes_also_required": bytes_needed,
                "pct_of_whole_renderer": 100.0 * bytes_needed / renderer_bytes,
                "exceeds_renderer": bytes_needed > renderer_bytes,
            }
        )

    # fb1's "adversarial best case": half the renderer AND all manufactured seg.
    half = renderer_bytes / 2.0
    d_seg_repaired = DX2_D_SEG * (1.0 - manufactured_frac)
    s_corner = recompute_s(DX2_ARCHIVE_BYTES - half, d_seg_repaired, DX2_D_POSE)
    return {
        "fb1_s5_ladder": rows,
        "fb1_adversarial_corner": {
            "renderer_bytes_shed": half,
            "archive_after": DX2_ARCHIVE_BYTES - half,
            "d_seg_after": d_seg_repaired,
            "S_recomputed": s_corner,
            "S_published_by_fb1": 0.119727,
            "margin_below_0_12_S": TARGET_S - s_corner,
            "margin_below_0_12_byte_equiv": (TARGET_S - s_corner) / RATE_S_PER_BYTE,
            "margin_published_by_fb1_B": 410,
        },
        "min_f_for_renderer_only_close": _min_f_for_renderer_close(renderer_bytes, gap),
        "min_f_published_by_fb1": 0.4212,
    }


def _min_f_for_renderer_close(renderer_bytes: int, gap: float) -> float:
    """Smallest manufactured-seg repair fraction f at which shedding the ENTIRE renderer
    reaches sub-0.12. fb1 publishes 0.4212."""
    manufactured_frac = MS9_MANUFACTURED_FLIPS / MS9_TOTAL_FLIPS
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        d_seg_after = DX2_D_SEG * (1.0 - manufactured_frac * mid)
        s_after = recompute_s(DX2_ARCHIVE_BYTES - renderer_bytes, d_seg_after, DX2_D_POSE)
        if s_after < TARGET_S:
            hi = mid
        else:
            lo = mid
    return hi


def _largest_archive_under_target(
    s_of: Callable[[int], float], seed: float
) -> int:
    """Largest integer archive size whose S is STRICTLY below TARGET_S.

    Walks in both directions from ``seed`` so a floor that lands either side of the true
    boundary still returns the exact maximum. The contest inequality is strict, so a size
    whose S equals TARGET_S exactly does not qualify.
    """
    n = int(math.floor(seed))
    while n > 0 and s_of(n) >= TARGET_S:
        n -= 1
    while s_of(n + 1) < TARGET_S:
        n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- gate: reproduce the published S from components before deriving anything ----
    s_recomputed = recompute_s(DX2_ARCHIVE_BYTES, DX2_D_SEG, DX2_D_POSE)
    if abs(s_recomputed - DX2_S_PUBLISHED) > 1e-15:
        print(
            f"REFUSE: recomputed S {s_recomputed!r} != published {DX2_S_PUBLISHED!r}",
            file=sys.stderr,
        )
        return 2
    gap = s_recomputed - TARGET_S

    # Independent cross-check 1: the fixed-distortion byte demand must reproduce 42,382 B
    # and the strict archive ceiling must reproduce 137,986 B (tl1 / fb1 / ar1b agree).
    distortion_s = s_seg(DX2_D_SEG) + s_pose(DX2_D_POSE)
    max_archive_fixed_distortion = _largest_archive_under_target(
        lambda n: recompute_s(n, DX2_D_SEG, DX2_D_POSE),
        (TARGET_S - distortion_s) * EXCHANGE_DEN / EXCHANGE_NUM,
    )
    demand_bytes = DX2_ARCHIVE_BYTES - max_archive_fixed_distortion

    # Independent cross-check 2: the ZERO-distortion residual demand must reproduce 150 B.
    max_archive_zero_distortion = _largest_archive_under_target(
        s_rate, TARGET_S * EXCHANGE_DEN / EXCHANGE_NUM
    )
    zero_distortion_demand = DX2_ARCHIVE_BYTES - max_archive_zero_distortion

    axes = build_axes()
    by_key = {a.key: a for a in axes}

    # Independent cross-check 3: the residue map must sum to the archive with zero remainder.
    census_sum = sum(
        a.nominal_bytes for a in axes if a.kind == "rate"
    )
    census_remainder = DX2_ARCHIVE_BYTES - census_sum

    # Independent cross-check 4: tokens + hpac must reproduce jf1's 127,292 B bar.
    jf1_bar = by_key["tokens"].nominal_bytes + by_key["hpac"].nominal_bytes

    checks = {
        "s_recomputed_from_components": s_recomputed,
        "s_published": DX2_S_PUBLISHED,
        "s_match_exact": s_recomputed == DX2_S_PUBLISHED,
        "distortion_term_s": distortion_s,
        "gap_to_0_12_s": gap,
        "max_archive_at_fixed_distortion_B": max_archive_fixed_distortion,
        "fixed_distortion_demand_B": demand_bytes,
        "fixed_distortion_demand_B_expected_fb1_tl1_ar1b": 42_382,
        "max_archive_at_zero_distortion_B": max_archive_zero_distortion,
        "zero_distortion_demand_B": zero_distortion_demand,
        "zero_distortion_demand_B_expected_tl1": 150,
        "residue_census_sum_B": census_sum,
        "residue_census_remainder_B": census_remainder,
        "tokens_plus_hpac_B": jf1_bar,
        "tokens_plus_hpac_B_expected_jf1_bar": 127_292,
        "bytes_bought_per_0_001_S_distortion": 0.001 / RATE_S_PER_BYTE,
        # Cross-check 6: ms9's exact flip numerator must reproduce the authority d_seg to
        # the 8 decimals the authority row publishes.
        "d_seg_from_ms9_numerator": MS9_TOTAL_FLIPS / SEG_PIXELS,
        "d_seg_authority_row": DX2_D_SEG,
        "d_seg_numerator_rounds_to_authority": round(MS9_TOTAL_FLIPS / SEG_PIXELS, 8)
        == DX2_D_SEG,
        "ms9_split_sums": (
            MS9_MANUFACTURED_FLIPS + MS9_REPRESENTATION_SURVIVED_FLIPS == MS9_TOTAL_FLIPS
        ),
        "mst1_native_render_share_of_all_seg_error": MST1_NATIVE_RENDER_FLIPS
        / MS9_TOTAL_FLIPS,
    }

    rows = []
    for ax_a, ax_b in itertools.combinations(axes, 2):
        max_joint = ax_a.saving(1.0) + ax_b.saving(1.0)
        feasible = max_joint > gap
        t = solve_equal_fraction(ax_a, ax_b, gap)
        corner_a = solve_corner(ax_a, ax_b, gap)  # A's demand when B is perfected
        corner_b = solve_corner(ax_b, ax_a, gap)

        # Same surface computed against the MEASURED structural ceilings where a receipt
        # bounds the axis below its arithmetic limit.
        ceil_a = (
            ax_a.structural_ceiling_s
            if ax_a.structural_ceiling_s is not None
            else ax_a.max_saving_s
        )
        ceil_b = (
            ax_b.structural_ceiling_s
            if ax_b.structural_ceiling_s is not None
            else ax_b.max_saving_s
        )
        max_joint_structural = ceil_a + ceil_b
        feasible_structural = max_joint_structural > gap

        pair_key = f"{ax_a.key}x{ax_b.key}"
        cut = CUTS.get(
            pair_key,
            {
                "cut": "IMPOSSIBLE",
                "sy2": "n/a -- arithmetically closed before classification",
                "sy2_basis": "",
                "evidence": (
                    "Both axes driven to their idealised limits still fall short of the "
                    "0.028219875632433777 S gap. Arithmetic over the ar1b census "
                    "(zero remainder) and the dx2 authority components."
                ),
                "cheapest_deciding_measurement": "none; arithmetically closed.",
            },
        )
        if feasible and pair_key not in CUTS:
            raise AssertionError(
                f"feasible pair {pair_key} has no declared cut -- refuse to emit an "
                "unclassified feasible row"
            )
        if not feasible and pair_key in CUTS and CUTS[pair_key]["cut"] != "IMPOSSIBLE":
            raise AssertionError(
                f"infeasible pair {pair_key} declared {CUTS[pair_key]['cut']}"
            )
        # An IMPOSSIBLE cut on a pair that IS arithmetically feasible is only admissible
        # when a MEASURED structural ceiling closes it. Otherwise the cut is unearned.
        if cut["cut"] == "IMPOSSIBLE" and feasible and feasible_structural:
            raise AssertionError(
                f"{pair_key} declared IMPOSSIBLE but is feasible at both the arithmetic "
                "and the measured structural ceilings"
            )

        rows.append(
            {
                "pair": pair_key,
                "axis_a": ax_a.key,
                "axis_b": ax_b.key,
                "axis_a_max_saving_S": ax_a.max_saving_s,
                "axis_b_max_saving_S": ax_b.max_saving_s,
                "axis_a_max_saving_byte_equiv": ax_a.max_saving_s / RATE_S_PER_BYTE,
                "axis_b_max_saving_byte_equiv": ax_b.max_saving_s / RATE_S_PER_BYTE,
                "max_joint_saving_S": max_joint,
                "max_joint_saving_byte_equiv": max_joint / RATE_S_PER_BYTE,
                "gap_S": gap,
                "gap_byte_equiv": gap / RATE_S_PER_BYTE,
                "surplus_at_both_perfected_S": max_joint - gap,
                "surplus_at_both_perfected_byte_equiv": (max_joint - gap)
                / RATE_S_PER_BYTE,
                "arithmetically_feasible": feasible,
                "equal_fraction_demand": t,
                "demand_on_a_when_b_perfected": corner_a,
                "demand_on_b_when_a_perfected": corner_b,
                "under_50pct_of_each": (t is not None and t < 0.5),
                "max_joint_saving_S_at_structural_ceilings": max_joint_structural,
                "feasible_at_structural_ceilings": feasible_structural,
                # ---- both currencies, per the tl1 convention ----
                "currency_bytes_at_fixed_distortion": demand_bytes,
                "currency_distortion_S_at_fixed_bytes": gap,
                "demand_on_a_equal_fraction_byte_equiv": (
                    None if t is None else ax_a.saving(t) / RATE_S_PER_BYTE
                ),
                "demand_on_b_equal_fraction_byte_equiv": (
                    None if t is None else ax_b.saving(t) / RATE_S_PER_BYTE
                ),
                # ---- the measured cut ----
                "cut": cut["cut"],
                "sy2_classification": cut["sy2"],
                "sy2_basis": cut["sy2_basis"],
                "cut_evidence": cut["evidence"],
                "cheapest_deciding_measurement": cut["cheapest_deciding_measurement"],
            }
        )

    payload = {
        "schema": "ddm_tac1_two_axis_composition.v1",
        "arm": "ddm_tac1",
        "type": "DERIVED",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "verdict_scope": "INSTANCE:DX2_OBJECT_PAIRWISE_ISO_0_12_SURFACE",
        "object": {
            "archive_bytes": DX2_ARCHIVE_BYTES,
            "archive_sha256": (
                "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"
            ),
            "d_seg": DX2_D_SEG,
            "d_pose": DX2_D_POSE,
            "S_recomputed_from_components": s_recomputed,
            "axis_label": "[contest-CUDA T4, n600]",
        },
        "exchange_rate_S_per_byte": RATE_S_PER_BYTE,
        "exchange_rate_source": "ddm_tx1_toolbox_crosswalk_20260819.md section 0 (CITED)",
        "cross_checks": checks,
        "independent_cross_check_vs_fb1_section5": fb1_corner_cross_checks(gap),
        "actuator_readout_matrix": READOUT_MATRIX,
        "cut_counts": {
            c: sum(1 for r in rows if r["cut"] == c)
            for c in ("IMPOSSIBLE", "CONTRADICTED", "TENSION", "OPEN")
        },
        "sy2_counts": {
            c: sum(1 for r in rows if r["sy2_classification"] == c)
            for c in ("STACK", "OBJECT-CHANGE", "DEGENERATE")
        },
        "prior_law_prediction": {
            "statement": (
                "most pairs are STACKS or in measured tension, and at most 2 pairs survive "
                "as genuinely OPEN"
            ),
            "open_count": sum(1 for r in rows if r["cut"] == "OPEN"),
            "falsifier": (
                ">=4 pairs survive all cuts as open AND at least one demands under 50% of "
                "each of its two axes"
            ),
            "falsifier_clause_1_open_ge_4": sum(1 for r in rows if r["cut"] == "OPEN") >= 4,
            "falsifier_clause_2_any_open_under_50pct": any(
                r["cut"] == "OPEN" and r["under_50pct_of_each"] for r in rows
            ),
        },
        "axes": [
            {
                "key": a.key,
                "label": a.label,
                "kind": a.kind,
                "nominal": a.nominal,
                "nominal_bytes": a.nominal_bytes,
                "arithmetic_max_saving_S": a.max_saving_s,
                "arithmetic_max_saving_byte_equiv": a.max_saving_s / RATE_S_PER_BYTE,
                "pct_of_gap_at_arithmetic_max": 100.0 * a.max_saving_s / gap,
                "structural_ceiling_S": a.structural_ceiling_s,
                "structural_note": a.structural_note,
                "measured_status": a.measured_status,
                "owning_memo": a.owning_memo,
            }
            for a in axes
        ],
        "pairs": rows,
        "NOT_CLAIMED": (
            "Every fraction here is a REQUIREMENT computed from measured inputs, not an "
            "achieved value. No arm has achieved any of them. Axis maxima are idealised "
            "limits (a byte pool at zero bytes; a distortion term at zero) and several are "
            "physically degenerate. This table prices nothing's achievability and moves no "
            "pointer."
        ),
    }

    json_path = out_dir / "ddm_tac1_pair_table.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = out_dir / "ddm_tac1_pair_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {}
    for path in (json_path, csv_path):
        data = path.read_bytes()
        manifest[path.name] = {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    manifest_path = out_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(checks, indent=2, sort_keys=True))
    print("\n--- axes ---")
    for a in payload["axes"]:
        print(
            f"{a['key']:<10} {a['kind']:<11} max_saving_S={a['arithmetic_max_saving_S']:.8f} "
            f"({a['arithmetic_max_saving_byte_equiv']:>10,.1f} B-equiv, "
            f"{a['pct_of_gap_at_arithmetic_max']:>7.2f}% of gap)"
        )
    print("\n--- fb1 s5 independent cross-check ---")
    print(json.dumps(payload["independent_cross_check_vs_fb1_section5"], indent=2))
    print("\n--- pairs (arithmetically feasible first) ---")
    for r in sorted(rows, key=lambda r: (-r["max_joint_saving_S"],)):
        t = r["equal_fraction_demand"]
        tstr = f"{t * 100:6.2f}%" if t is not None else "  n/a "
        print(
            f"{r['pair']:<20} joint_max_S={r['max_joint_saving_S']:.8f} "
            f"demand_each={tstr} "
            f"struct_feas={str(r['feasible_at_structural_ceilings']):<5} "
            f"{r['cut']:<13} {r['sy2_classification']}"
        )
    print("\n--- counts ---")
    print(json.dumps(payload["cut_counts"], indent=2, sort_keys=True))
    print(json.dumps(payload["sy2_counts"], indent=2, sort_keys=True))
    print(json.dumps(payload["prior_law_prediction"], indent=2, sort_keys=True))
    print(f"\nwrote {json_path} / {csv_path} / {manifest_path}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
