# SPDX-License-Identifier: MIT
"""Canonical equation: RGB/chroma NECESSITY per class-pair boundary through the FROZEN scorer (n600).

Operator 2026-07-15: "We might need RGB at boundaries regardless. Or some boundaries. Deep math and
geometry and frozen contest information space should reveal." MEASURED (n600, frozen CPU-torch SegNet +
PoseNet, the exact upstream modules.py forward, $0):

  chroma-necessity(pair a,b) := P[argmax flips at the (a,b) boundary | BT.601 desat], with the color
  factorization fixed by frame_utils.rgb_to_yuv6: Y = k.rgb, k = (0.299, 0.587, 0.114);
  ker(U,V) = span{(1,1,1)}; ker(Y) = the chroma 2-plane {delta : k.delta = 0}. The margin Jacobian g
  splits ORTHOGONALLY: S_lu = |g.k_hat|, S_ch = ||g - (g.k_hat)k_hat||; chroma distance-to-flip =
  margin/S_ch (0-255 units).

THE LAW (three parts, all MEASURED — memo .omx/research/rgb_at_boundaries_derivation_20260715.md):
  (1) REGION-CONSISTENCY DOMINANCE: chroma-only-at-annulus (grey context) is WORSE than no chroma at all
      (d_seg-equiv 0.006293 vs 0.005384) — SegNet's stride-2 region reading makes the per-class palette
      chroma LOAD-BEARING everywhere; per-pixel chroma is a boundary finisher ON TOP of it, never a
      replacement. The naive "RGB only at boundaries" is REFUTED.
  (2) PAIR-STRUCTURED NECESSITY: desat flip fraction at the pair boundary — Undriv|Movable 0.363 >
      Road|Movable 0.303 > Road|Undriv 0.242 > Road|MyCar 0.220 (half context-driven: annulus-scoped
      0.100) > Road|Lane 0.082 (largest absolute mass, 106k flips) — Movable (car) edges are the most
      chroma-decided; NO major boundary is chroma-free.
  (3) SENSITIVITY SPLIT: chroma gradient energy 7-12% per pair in the boundary band BUT 17-33% of
      boundary pixels are chroma-DOMINANT (S_ch > S_lu); 6-9% flip within an 8-LSB chroma move; chroma
      survives the real R at gain 0.98-1.00 (0.5-4 LSB) — every chroma-flippable pixel is reachable
      through the byte-closed decode.

Consumer: the c2-battery rung-0 lever SegChromaBoundary (LEVER-4c, 0-byte, never-fired) — this law is its
AIM (where the wins land) + CEILING (annulus chroma worth 0.002972 d_seg-equiv; worth != gain, S5-N10).
Pose constraint: wrong annulus chroma costs Delta d_pose ~4.26e-4 mean (n600) — pose-safety comes from
CORRECTNESS (match GT), not invisibility; the 2x2 zero-sum pose-null projection remains available.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "rgb_chroma_necessity_per_boundary_pair_v1"

_UTC = "2026-07-15T00:00:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_PREDICTED = "[predicted]"
_ROWS = "experiments/results/rgb_at_boundaries_chroma_jacobian_20260715/rows.jsonl"
_SUMMARY = "experiments/results/rgb_at_boundaries_chroma_jacobian_20260715/summary.json"

# --- the n600 measured payload (load-bearing, quoted from summary.json) -----------------------------
DSEG_EQUIV_DESAT_FULL = 0.005384       # all chroma removed = 6.2x the whole d_seg need (0.00087)
DSEG_EQUIV_DESAT_ANNULUS = 0.002972    # annulus-local chroma worth (the rung-0 lever ceiling)
DSEG_EQUIV_KEEP_ANNULUS = 0.006293     # chroma ONLY at annulus (grey context) — WORSE than desat_full
DPOSE_DELTA_DESAT_ANNULUS = 4.26e-4    # wrong annulus chroma pose risk (mean, n600)
DPOSE_DELTA_KEEP_ANNULUS = 9.57e-3     # grey-context pose cost (regional chroma is pose-visible)
CHROMA_R_TRANSFER_GAIN_MIN = 0.98      # annulus chroma through the real R (0.5-4 LSB)
FLIP_DESAT_FULL_BY_PAIR = {
    "Undrivable|Movable": 0.363,
    "Road|Movable": 0.303,
    "Road|Undrivable": 0.242,
    "Road|MyCar": 0.220,
    "Road|Lane": 0.082,
    "Lane|Undrivable": 0.292,
    "Lane|MyCar": 0.089,
}


def build_rgb_chroma_necessity_per_boundary_pair_v1() -> CanonicalEquation:
    """Build the RGB-at-boundaries chroma-necessity law with its n600 measured anchor."""

    anchor_n600 = EmpiricalAnchor(
        anchor_id="rgb_chroma_necessity_desat_ablations_and_jacobian_n600_20260715",
        measurement_utc=_UTC,
        inputs={
            "tool": "tools/rgb_at_boundaries_chroma_jacobian_n600.py",
            "gt": "experiments/results/mlx_fleet_gt_cache/gt_n600.npz (exact frozen-scorer cache)",
            "scorers": "frozen CPU-torch SegNet+PoseNet via the exact upstream modules.py forward",
            "ablations": "desat_full / desat_annulus(margin<1) / keep_annulus, BT.601 Y-replicate",
            "no_fake_gate": "baseline forward reproduces the cached exact L* px-exact (frame-0 gate)",
        },
        predicted_output={
            "hypothesis": ("chroma necessity is pair-structured; 'RGB only at boundaries' sufficiency "
                           "unknown before measurement"),
        },
        empirical_output={
            "d_seg_equiv_desat_full": DSEG_EQUIV_DESAT_FULL,
            "d_seg_equiv_desat_annulus": DSEG_EQUIV_DESAT_ANNULUS,
            "d_seg_equiv_keep_annulus": DSEG_EQUIV_KEEP_ANNULUS,
            "keep_annulus_worse_than_desat_full": True,
            "flip_desat_full_by_pair": FLIP_DESAT_FULL_BY_PAIR,
            "boundary_px_chroma_dominant_frac_range": [0.17, 0.33],
            "grad_chroma_energy_frac_range": [0.07, 0.12],
            "frac_boundary_px_chroma_flippable_8lsb": [0.06, 0.09],
            "chroma_R_transfer_gain": [CHROMA_R_TRANSFER_GAIN_MIN, 1.00],
            "d_pose_delta_desat_annulus_mean": DPOSE_DELTA_DESAT_ANNULUS,
            "d_pose_delta_keep_annulus_mean": DPOSE_DELTA_KEEP_ANNULUS,
            "verdict": ("region-consistent palette chroma is TRUNK (load-bearing everywhere); per-pixel "
                        "chroma is a pair-aimed annulus FINISHER; Movable edges most chroma-decided; "
                        "refines #508 'finisher-only' (too coarse at the region scale)"),
        },
        residual=0.0,
        source_artifact=_SUMMARY,
        measurement_method=("n600 exact desat ablations (batched frozen SegNet forwards) + margin-Jacobian "
                            "BT.601 orthogonal split (1 bwd/frame) + PoseNet pair forwards + real-R chroma "
                            "transfer check; rows " + _ROWS),
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_SUMMARY,
            reactivation_criteria=("re-measure if the annulus band definition (margin<1), the GT cache, or "
                                   "the frozen scorer checkpoints change; the rung-0 A/B realized gain "
                                   "supersedes the worth bound when measured"),
            measurement_axis=_ADVISORY,
            hardware_substrate="m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("RGB/chroma necessity per class-pair boundary through the frozen scorer: region-consistent "
              "palette chroma is trunk-level (keep_annulus 0.006293 > desat_full 0.005384); per-pixel "
              "chroma is a pair-structured annulus finisher (Movable edges 0.30-0.36 > horizon 0.24 > "
              "hood 0.22 > lane 0.08); chroma survives R at gain 0.98-1.00"),
        one_line_summary=(
            "n600 frozen-scorer law: palette chroma = trunk; per-pixel chroma = pair-aimed annulus "
            "finisher worth <=0.0030 d_seg; 17-33% of boundary px chroma-dominant; R not a barrier."
        ),
        latex_form=(
            r"S_{ch}(p)=\lVert g(p)-(g(p)\cdot\hat k)\hat k\rVert,\; S_{lu}(p)=|g(p)\cdot\hat k|,\;"
            r"\hat k\propto(0.299,0.587,0.114);\quad d_{flip}^{ch}(p)=m(p)/S_{ch}(p);\quad"
            r"\mathrm{necessity}(a,b)=\Pr[\mathrm{argmax\ flip}\mid\mathrm{desat},\partial_{ab}]"
        ),
        python_callable_module_path=(
            "tools.rgb_at_boundaries_chroma_jacobian_n600:summarize"
        ),
        domain_of_validity={
            "vehicle": ["frozen contest SegNet/PoseNet (scorer-side law, witness-independent)"],
            "measurement_axis": ["macOS-CPU advisory"],
            "annulus": "GT top1-top2 margin < 1 at (384,512)",
            "note": ("ablation WORTHS (GT-chroma vs desat), not achieved witness moves; the witness palette "
                     "sits between grey and GT so realized rung-0 gain <= the annulus worth (S5-N10). "
                     "Pair rows with <=5 supporting frames (Movable|MyCar, Lane|Movable) carry no verdict."),
        },
        units_in={"g": "d(margin)/d(rgb) at the (384,512) scorer grid, 0-255 RGB units",
                  "margin": "segnet_top1_top2_logit_gap", "desat": "BT.601 Y-replicate"},
        units_out={"necessity": "flip probability at the pair boundary", "worth": "d_seg-equivalent"},
        empirical_anchors=(anchor_n600,),
        predicted_vs_empirical_residual={
            "n276_aggregate_consistency": 0.0,  # confirms the n96 aggregate DOF (78.8/21.2 order) per-pair
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl",  # SegChromaBoundary rung-0 aim + ceiling
            ".omx/research/rgb_at_boundaries_derivation_20260715.md",
        ),
        canonical_producers=(
            "tools/rgb_at_boundaries_chroma_jacobian_n600.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="rgb_chroma_necessity_per_boundary_pair.v1",
            inputs_sha256="0" * 64,
            measurement_axis=_PREDICTED,
            hardware_substrate="m5_max_cpu",
        ),
    )


def populate_rgb_chroma_necessity_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (latest-row-wins), mirroring the LEVER-4c pattern."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_rgb_chroma_necessity_per_boundary_pair_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes=("rgb_at_boundaries_20260715 (equations leg of the #515 RGB-at-boundaries derivation; "
               "refines #508 finisher-only scope; aims the never-fired rung-0 SegChromaBoundary)"),
    )
    return eq


__all__ = [
    "CHROMA_R_TRANSFER_GAIN_MIN",
    "DPOSE_DELTA_DESAT_ANNULUS",
    "DPOSE_DELTA_KEEP_ANNULUS",
    "DSEG_EQUIV_DESAT_ANNULUS",
    "DSEG_EQUIV_DESAT_FULL",
    "DSEG_EQUIV_KEEP_ANNULUS",
    "EQUATION_ID",
    "FLIP_DESAT_FULL_BY_PAIR",
    "build_rgb_chroma_necessity_per_boundary_pair_v1",
    "populate_rgb_chroma_necessity_equation",
]
