# SPDX-License-Identifier: MIT
"""Necessity §9 calibration law — DP-ε (generator, seed) → REAL d_seg through the scorer chain.

The necessity solver (``realization_necessity_preimage_per_stratum_v1``) gives the RATE floor by
inverting the flattened factorization per Morse-Smale stratum. This equation is its d_seg-verified
sibling: it *realizes* the witness from the (generator, seed) design and measures d_seg through the
REAL frozen SegNet argmax on the bit-exact cached GT (n600), then reads off the min-S operating point.

Two MEASURED facts (2026-07-16, ``[macOS-CPU advisory]``):

  1. DP-ε geometry AMPLIFIES d_seg through SegNet (~14x): the argmax turns a tiny geometric
     simplification error into many flips, so d_seg rises monotonically with ε. ⟹ ε* = 0 (lossless
     partition) is the min-d_seg geometry; spending rate on coarser geometry buys NEGATIVE ΔS.
  2. The static one-time hood-texture seed (frame-0 MyCar interior, 16x-downsampled, ~1.76 KB) is the
     single decisive waterfill buy: at ε=0 it cuts d_seg 0.04538 → 0.01328 (-71%) for +1759 B,
     ΔS ≈ -3.21 (ΔS/byte ≈ -1.8e-3). A flip is worth ~1.3 bytes; the hood cures ~48k flips.

The min-S operating point (waterfill knee) is ε=0-lossless + static hood-tex: S = 1.613
(d_seg 0.01328, rate 0.158) — 8.4x ABOVE the 0.19108 pointer. FORMULATION-scoped verdict: the
palette+geometry CELL-interior generator does NOT realize d_seg (photometric wall, L68, restated at the
cell surface); the cell generator must be the JOINT-trained render (the witness), not flat-palette fill.
The necessity frame stays the RATE-floor + per-stratum-carrier authority (edges cheap+realizable; the
static hood-tex is a genuine folded near-free carrier), NOT itself the d_seg vehicle.

``predicted_S`` is the callable: given a measured d_seg, seed bytes, and the banked pose contribution,
return S under the contest score. NOT a score claim — the frozen CPU-torch argmax is advisory until the
byte-closed ``upstream/evaluate.py`` CPU/CUDA row (lens 8).
"""
from __future__ import annotations

from dataclasses import dataclass

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

EQUATION_ID = "necessity_generator_seed_dseg_calibration_v1"
AXIS = "[macOS-CPU advisory] frozen CPU-torch fp32; bit-exact cached n600 GT"
TOOL = "tools/necessity_dseg_calibration.py"
ART_DIR = "experiments/results/necessity_dseg_calibration_20260715"
SUMMARY_SHA256 = "ef4b835f86ac77dd64db96b50c56b699412ed0b55cff1f30d6488469d2dcf972"

N_RATE = 37_545_489  # contest rate denominator
D_POSE_BANKED = 0.001610  # witness R1 dxi through byte-close (L68)

# ---- MEASURED constants (n600, [macOS-CPU advisory]; artifacts summary.json) -----------------
DSEG_PALETTE_EPS0 = 0.04538490          # ε=0 lossless partition + median palette, value-wall
DSEG_PALETTE_HOOD_EPS0 = 0.01327947     # + static ds16 hood-tex seed (the min-S d_seg)
HOOD_SEED_BYTES = 1759                  # brotli-q11 of the frame-0 MyCar-bbox ds16 crop
SEED_BYTES_LOSSLESS_ADJ = 228764        # ε=0 shared-edge-adjusted contour seed
SEED_BYTES_EPS1_ADJ = 143552            # ε=1 seed (= necessity K-ladder edges floor, K/H=0.47)
DXI_BYTES_BANKED = 7200                 # R1 pose dxi section
PALETTE_BYTES = 15                      # 5 class medians x 3 bytes
MIN_S_OPERATING_POINT = 1.6131          # ε=0-lossless + hood-tex (waterfill knee)
POINTER = 0.19108                       # submittable frontier (UNMOVED)
SEGNET_DP_AMPLIFICATION = 14.0          # d_seg-real Δ / d_seg-geo Δ at ε=0.5 (argmax amplification)
HOOD_DSEG_REDUCTION_FRAC = 0.7075       # (0.04538-0.01328)/0.04538


def predicted_S(dseg_real: float, seed_bytes_counted: int,
                pose_contribution: float = (10.0 * D_POSE_BANKED) ** 0.5) -> float:
    """Contest score for a realized (generator, seed) witness: 100·d_seg + √(10·d_pose) + 25·bytes/N.

    ``seed_bytes_counted`` = every COUNTED byte in archive.zip (contour seed + palette + hood-tex +
    banked dxi); the generator is FREE (rule 118). NOT a score claim — advisory until byte-closed eval.
    """
    if not (dseg_real >= 0.0):
        raise ValueError(f"dseg_real must be >= 0, got {dseg_real!r}")
    if seed_bytes_counted < 0:
        raise ValueError(f"seed_bytes_counted must be >= 0, got {seed_bytes_counted!r}")
    return 100.0 * float(dseg_real) + float(pose_contribution) + 25.0 * seed_bytes_counted / N_RATE


def hood_tex_delta_s_per_byte() -> float:
    """ΔS/byte of the decisive static hood-tex waterfill buy (MEASURED, ε=0). Negative = a buy."""
    d_dseg = DSEG_PALETTE_EPS0 - DSEG_PALETTE_HOOD_EPS0
    delta_s = -100.0 * d_dseg + 25.0 * HOOD_SEED_BYTES / N_RATE
    return delta_s / HOOD_SEED_BYTES


@dataclass(frozen=True)
class MinSOperatingPoint:
    dseg_real: float
    seed_bytes_counted: int
    predicted_S: float
    ratio_over_pointer: float
    sub_pointer: bool
    formulation_verdict: str


def min_s_operating_point() -> MinSOperatingPoint:
    """The measured waterfill knee: ε=0-lossless + static hood-tex. NOT sub-0.19108."""
    seed = SEED_BYTES_LOSSLESS_ADJ + PALETTE_BYTES + HOOD_SEED_BYTES + DXI_BYTES_BANKED
    s = predicted_S(DSEG_PALETTE_HOOD_EPS0, seed)
    return MinSOperatingPoint(
        dseg_real=DSEG_PALETTE_HOOD_EPS0,
        seed_bytes_counted=seed,
        predicted_S=s,
        ratio_over_pointer=s / POINTER,
        sub_pointer=s < POINTER,
        formulation_verdict=(
            "palette+geometry CELL-interior generator does NOT realize d_seg (photometric wall L68 "
            "at the cell surface); cell generator must be the JOINT-trained render (witness), not "
            "flat-palette fill; necessity frame is the RATE-floor + per-stratum-carrier authority; "
            "static hood-tex is a genuine folded near-free carrier (-71% value-wall / 1.76KB); ε*=0"
        ),
    )


def build_necessity_generator_seed_dseg_calibration_v1() -> CanonicalEquation:
    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=SUMMARY_SHA256,
        source_path=f"{ART_DIR}/summary.json",
        captured_at_utc="2026-07-16T05:20:00Z",
    )
    anchors = (
        EmpiricalAnchor(
            anchor_id="necessity_cal_min_s_operating_point_20260716",
            measurement_utc="2026-07-16T05:20:00Z",
            inputs={"frames": "n600 (ALL scored pairs) cached lstars",
                    "arm": "ε=0 lossless partition + static ds16 hood-tex, median palette",
                    "chain": "uint8 -> float -> bilinear (384,512) -> frozen SegNet argmax"},
            predicted_output={"sub_pointer": False,
                              "reason": "flat-palette cartoon not argmax-faithful in cell interiors"},
            empirical_output={"dseg_real": DSEG_PALETTE_HOOD_EPS0,
                              "predicted_S": MIN_S_OPERATING_POINT,
                              "ratio_over_pointer": MIN_S_OPERATING_POINT / POINTER,
                              "residual_edges_frac": 0.52, "residual_movable_interior_frac": 0.24},
            residual=0.0,
            source_artifact=f"{ART_DIR}/summary.json",
            measurement_method=(
                "realize (generator, seed) witness; per-pixel argmax disagreement vs cached lstars, "
                "mean over n600; S = 100·d_seg + √(10·d_pose_banked) + 25·bytes/N"
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
        EmpiricalAnchor(
            anchor_id="necessity_cal_hood_tex_waterfill_buy_20260716",
            measurement_utc="2026-07-16T05:20:00Z",
            inputs={"seed": "frame-0 MyCar-bbox ds16 crop, brotli-q11 (1759 B)",
                    "compare": "ε=0 palette with vs without the static hood-tex seed"},
            predicted_output={"decisive_waterfill_buy": True,
                              "reason": "necessity frame: Road-MyCar -> static one-time seed"},
            empirical_output={"dseg_before": DSEG_PALETTE_EPS0, "dseg_after": DSEG_PALETTE_HOOD_EPS0,
                              "reduction_frac": HOOD_DSEG_REDUCTION_FRAC,
                              "delta_s_per_byte": hood_tex_delta_s_per_byte()},
            residual=0.0,
            source_artifact=f"{ART_DIR}/summary.json",
            measurement_method="paired n600 arms rows_eps0.0.jsonl vs rows_eps0.0_hoodtex.jsonl",
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
        EmpiricalAnchor(
            anchor_id="necessity_cal_dp_eps_amplifies_dseg_20260716",
            measurement_utc="2026-07-16T05:20:00Z",
            inputs={"eps_sweep": [0.0, 0.5, 1.0, 2.0], "geometry": "Douglas-Peucker contour tol"},
            predicted_output={"eps_star": 0.0,
                              "reason": "SegNet argmax amplifies geometric simplification error"},
            empirical_output={"dseg_by_eps": {"0.0": 0.04538, "0.5": 0.05099,
                                              "1.0": 0.07637, "2.0": 0.08938},
                              "segnet_amplification": SEGNET_DP_AMPLIFICATION},
            residual=0.0,
            source_artifact=f"{ART_DIR}/summary.json",
            measurement_method="d_seg-real vs d_seg-geo ratio across the ε sweep (n600)",
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Necessity (generator, seed) → real d_seg calibration; min-S operating point",
        one_line_summary=(
            "min-S knee = ε=0-lossless + static hood-tex: S=1.613 (d_seg 0.0133, rate 0.158), 8.4x "
            "ABOVE 0.19108; palette CELL generator insufficient (photometric wall); ε*=0; hood-tex "
            "-71%/1.76KB decisive carrier"
        ),
        latex_form=(
            r"S = 100\,d_{seg}^{\mathrm{real}} + \sqrt{10\,d_{pose}} + 25\,\tfrac{|\mathrm{seed}|}{N},"
            r"\quad d_{seg}^{\mathrm{real}} = \Pr_x[\arg\max N_{seg}(G(\mathrm{seed}))(x) \ne \ell^*(x)]"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.necessity_generator_seed_dseg_calibration_20260715:"
            "min_s_operating_point"
        ),
        domain_of_validity={
            "network": "frozen contest SegNet, exact SegnetWrapper preprocess (raw 0-255)",
            "target": "cached n600 GT argmax partition (bit-exact vs the real forward)",
            "vehicle": "palette + DP-contour geometry generator (NOT the trained witness render)",
            "verdict_scope": "FORMULATION — the palette+geometry CELL-interior generator, at optimal "
                             "form (median palette, ε swept incl lossless, hood folded, variants "
                             "rejected); NOT a paradigm kill of necessity/projection/Kolmogorov",
            "axis": AXIS,
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={"dseg_real": "argmax disagreement fraction", "seed_bytes": "bytes",
                  "pose_contribution": "score units"},
        units_out={"predicted_S": "contest score units"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"sub_pointer": 0.0, "eps_star_is_0": 0.0},
        last_calibration_utc="2026-07-16T05:20:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "V9·CGauge archive builder (fold the static hood-tex seed; ε*=0 for stored geometry)",
            "p0_UNIFICATION_projection_preimage_SUPREME_20260715 (crown §9 calibration)",
            "realization_necessity_preimage_per_stratum_v1 (d_seg-verified sibling of the rate floor)",
        ),
        canonical_producers=(TOOL, f"{ART_DIR}/summary.json"),
        provenance=provenance,
    )


def populate_necessity_generator_seed_dseg_calibration_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (EQUATIONS leg of FEED-necessity-cal)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_necessity_generator_seed_dseg_calibration_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "DSEG_PALETTE_EPS0",
    "DSEG_PALETTE_HOOD_EPS0",
    "HOOD_SEED_BYTES",
    "MIN_S_OPERATING_POINT",
    "POINTER",
    "SEGNET_DP_AMPLIFICATION",
    "predicted_S",
    "hood_tex_delta_s_per_byte",
    "MinSOperatingPoint",
    "min_s_operating_point",
    "build_necessity_generator_seed_dseg_calibration_v1",
    "populate_necessity_generator_seed_dseg_calibration_equation",
]
