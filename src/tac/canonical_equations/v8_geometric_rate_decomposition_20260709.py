# SPDX-License-Identifier: MIT
"""Canonical equation: v8 GEOMETRIC RATE DECOMPOSITION LAW (whole-scene per-edge, MEASURED).

The v8 (edge-centric per-class carrier) rate thesis, MEASURED through REAL bit-exact coders on the
n600 GT argmax labels. The law: the whole-scene partition rate is the sum over inter-class edges
(each counted once) of a PER-EDGE decomposition into a parametric-GENERATOR dominant stream plus a
residual sidecar for the uncovered boundary px:

    S_edge   = S_gen(dominant, ~k coeffs/frame)  +  S_resid(uncovered px)          (per-edge)
    S_rate   = sum_edges  S_edge                                                    (whole-scene, de-shared)

where S = 25 * bytes / 37_545_489 is the contest rate term. The DOMINANT stream stores a parsimonious
parametric generator per class (horizon deg-3 poly + ego-xi; Road/Lane centerline poly + halfwidth via
Wave-F #234; Movable bbox sparse sites with Hungarian slot-track; hood static model) rather than the
boundary bitmap; the RESIDUAL sidecar losslessly carries the uncovered boundary px (car concavities,
faint/occluded lane fragments, horizon secondary arcs) via generic sparse-coord temporal/spatial-delta
+ brotli q11.

THE MEASURED VERDICT (n600, real coders, bit-exact roundtrips; NO b/px proxy, NO projection):
  * bitmap whole-scene                = 0.339  S   (de-shared, brotli)
  * geometric DOMINANT-only           = 0.061  S   -> 5.5x BELOW bitmap, 1.9x BELOW the 0.118 frontier
  * geometric COMPLETE (+ residual)   = 0.140  S   -> 2.4x below bitmap, ~1.2x ABOVE the 0.118 frontier

The v8 rate thesis is CONFIRMED on the DOMINANT structure (geometry << bitmap << frontier). Lossless
completion with TODAY'S generic residual coder does NOT by itself beat the frontier rate: the whole
0.079-S gap (0.140 - 0.061) is the residual sidecar, whose scattered short fragments sit near their
coordinate entropy (a chain-code residual buys only -13%).

PER-EDGE rows (bitmap -> dominant -> complete, MEASURED):
  Road/Lane                  0.204  -> 0.0275 (72.5% cover, 7.4x) -> 0.0695
  Road/Undriv (horizon)      0.047  -> 0.0032 (poly+xi, 14.6x)    -> 0.0221
  Movable (Road/Mov+Undriv/Mov) 0.0532 -> 0.00344 (bbox 70% cover) -> 0.0209
  Road/MyCar (hood)          0.028  -> 0.0202 (static, complete)   -> 0.0202
  Lane/* (3 rows)            0.007  -> 0.007  (already tiny)       -> 0.007

TWO MEASURED-headroom levers (un-exploited; move 0.140 toward 0.061) — refinement notes, NOT built:
  (1) DE-SHARE double-count: the horizon residual's "secondary arcs" ARE objects breaking the horizon
      = Movable px ALREADY carried by the Movable sites; attributing them to the Movable carrier shrinks
      the horizon residual for free (currently INFLATES, not deflates, the complete number).
  (2) CURVE-RELATIVE residual coder (not built): code the signed OFFSET-from-generator per residual px
      (small ints) instead of absolute coords; the generic 0.4-0.6 B/px is an UPPER BOUND, not the floor.

means != ends: this is a MEASURED RATE-half characterization on the DECODE side, NOT a converged-witness
score. The pointer moves only through a byte-closed upstream/evaluate.py exact-eval row; the d_seg half
(#205) remains the true blocker. Snapshot-scope (n600 GT argmax labels at this cache), advisory-class,
promotion_eligible=false; NOT a converged-witness fact. Pointer 0.19110 UNMOVED, #205 untouched.

DSL leg: N/A (a MEASUREMENT — no trainer lever / launch / curriculum change). EQUATIONS leg = this module.
DAG: `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` (FEED-v8-rollup / FEED-v8-roadlane).
Memos: `.omx/research/v8_movable_residual_rollup_20260709.md` + `.omx/research/v8_roadlane_geometric_rate_20260709.md`.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "v8_geometric_rate_decomposition_v1"

_UTC = "2026-07-09T00:00:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"
_MEMO_ROLLUP = ".omx/research/v8_movable_residual_rollup_20260709.md"
_MEMO_ROADLANE = ".omx/research/v8_roadlane_geometric_rate_20260709.md"
_MEMO_KIT = ".omx/research/residual_kit_deshare_curverel_build_20260709.md"

# Whole-scene MEASURED S numbers (n600 GT argmax, real coders, bit-exact).
_S_BITMAP = 0.339
_S_DOMINANT = 0.061
_S_COMPLETE = 0.140
_S_FRONTIER_RATE = 0.118  # current pointer 0.19110 rate term (25*bytes/N)


def build_v8_geometric_rate_decomposition_v1() -> CanonicalEquation:
    """Build the v8 geometric rate decomposition law with its two MEASURED whole-scene anchors.

    Anchor 1 = the Road/Lane edge (the biggest owed row, 59% of the bitmap budget) measured through
    the REAL LBND2 lane coder + Wave-F #234 fit machinery (dominant 0.0275 / complete 0.0695).
    Anchor 2 = the whole-scene roll-up (all 5 edges once): bitmap 0.339 -> dominant 0.061 -> complete
    0.140, with the Movable sparse-site carrier + shared residual sidecars, each bit-exact."""

    anchor_roadlane = EmpiricalAnchor(
        anchor_id="v8_roadlane_geometric_rate_dominant_measured_20260709",
        measurement_utc=_UTC,
        inputs={
            "edge": "Road<->Lane (comma10k class Road0/Lane1)",
            "labels": "experiments/results/mlx_fleet_gt_cache/gt_n600.npz['lstars'] (600x384x512 int64 argmax)",
            "fit_machinery": (
                "tac.boundary_math.lane_sdf_component."
                "{cluster_lane_lines,fit_lane_line,rasterize_lane_band} (Wave-F #234)"
            ),
            "coder": (
                "tac.boundary_math.analytic_lane_render_band.serialize_lane_band_rd_tracked "
                "(LBND2: fp16-quantize -> zigzag temporal-delta -> Hungarian coherent-slot xi-track "
                "-> brotli q11; roundtrips bit-exact through the UNCHANGED inflate decode)"
            ),
        },
        predicted_output={
            "law": "S_edge = S_gen(dominant) + S_resid(uncovered px)",
            "generator": "centerline deg-3 poly + deg-1 halfwidth per instance (~5 lines x 11 coeffs)",
        },
        empirical_output={
            "s_bitmap_deshared": 0.204,
            "s_dominant_lossless": 0.0275,   # 7.4x < bitmap
            "boundary_cover_pct": 72.5,
            "fit_residual_px": 1.00,
            "s_residual_sidecar": 0.04203,   # uncovered 26.6% / 228 px-frame
            "s_complete": 0.0695,            # 0.0275 + 0.04203
            "reduction_dominant_x": 7.4,
            "note": (
                "reduction is PRIMARILY parsimony (~5 lines x 11 coeffs vs full bitmap); lanes are NOT "
                "ego-frozen (all coeffs move 55-82% nonzero) so the xi ego-warp lever is WEAKER than the "
                "rigid horizon; horizon-class 14.6x transfer does NOT hold (FEED-v8-lane-xi MEASURED NO-GO)"
            ),
        },
        residual=0.0,
        source_artifact=_MEMO_ROADLANE,
        measurement_method="n600_argmax_real_lane_fit_and_lbnd2_coder_bit_exact_roundtrip",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO_ROADLANE,
            reactivation_criteria=(
                "byte-closed upstream/evaluate.py exact-eval anchor on a v8 archive that ships the "
                "Road/Lane centerline generator + residual sidecar (converts advisory -> contest rate)"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )

    anchor_rollup = EmpiricalAnchor(
        anchor_id="v8_wholescene_rate_rollup_movable_residual_measured_20260709",
        measurement_utc=_UTC,
        inputs={
            "scope": "all 5 whole-scene inter-class edges, each counted once (de-shared)",
            "labels": "gt_n600.npz['lstars'] (comma10k order Road0/Lane1/Undriv2/Movable3/MyCar4)",
            "movable_carrier": (
                "Hungarian slot-track -> temporal-delta bbox ints -> zigzag -> brotli q11 (bit-exact); "
                "ONE sparse-site carrier de-shares Road<->Movable AND Undriv<->Movable"
            ),
            "residual_coder": (
                "shared generic: uncovered boundary px -> row-major flat-index -> temporal/spatial delta "
                "-> zigzag -> brotli q11 (bit-exact all 3 edges)"
            ),
        },
        predicted_output={
            "law": "S_rate = sum_edges [ S_gen(dominant) + S_resid(uncovered) ]",
            "thesis": "geometry << bitmap; dominant beats the frontier rate term",
        },
        empirical_output={
            "per_edge_bitmap_dominant_complete": {
                "Road/Lane": [0.204, 0.0275, 0.0695],
                "Road/Undriv_horizon": [0.047, 0.0032, 0.0221],   # horizon deg-3 poly + xi = 14.6x; P4-R3 PIN: the code-emitted dominant = 0.00277 (4167 B, horizon_poly_xi_byte_cost); 0.0032 here is an amortization-method delta (+14%, 1.1% of the 0.0411 gap, IMMATERIAL — 0.00277 is even cheaper -> the rate argument is STRONGER); anchor value preserved append-only (Catalog #110/#113), pin annotated not mutated
                "Movable_Road+Undriv": [0.0532, 0.00344, 0.0209],  # bbox 70% cover, 15.4x vs bitmap
                "Road/MyCar_hood": [0.028, 0.0202, 0.0202],        # static model, complete (no residual)
                "Lane/*_3rows": [0.007, 0.007, 0.007],             # already tiny, untouched
            },
            "whole_scene": {
                "bitmap": _S_BITMAP,
                "geometric_dominant_only": _S_DOMINANT,   # 5.5x < bitmap, 1.9x < frontier
                "geometric_complete_lossless": _S_COMPLETE,  # 2.4x < bitmap, ~1.2x > frontier; P4/P5b CORRECTED-triple: 0.140 - de-share 0.00440 (anchor v8_residual_deshare_dedup_measured_20260709, dilate=2 footprint proxy; footprint-sweep MEASURED band [0.000@dil0, 0.0069@dil3] -> NOT footprint-robust) - triple-point 0.00102 = COMPLETE 0.135 (residual enemy 0.074); shippable = WASH-with-frontier, sub-0.118 win P-C-gated (r* RANGE [0.061,0.135])
                "frontier_rate_term": _S_FRONTIER_RATE,
            },
            "verdict": (
                "v8 rate thesis CONFIRMED on dominant structure (0.061 = 5.5x < bitmap 0.339, 1.9x < "
                "frontier 0.118); lossless COMPLETE (0.140) ties/slightly over the frontier with today's "
                "generic residual coder -> the whole 0.079 gap is the residual sidecar"
            ),
            "headroom_levers_unbuilt": {
                "de_share_double_count": (
                    "horizon residual secondary arcs ARE Movable px already carried; attributing to the "
                    "Movable carrier shrinks the horizon residual free (currently INFLATES complete)"
                ),
                "curve_relative_residual_coder": (
                    "code signed OFFSET-from-generator (small ints) not absolute coords; generic "
                    "0.4-0.6 B/px is an UPPER BOUND -> #1 open real-coder lever for rate completion"
                ),
            },
        },
        residual=0.0,
        source_artifact=_MEMO_ROLLUP,
        measurement_method="n600_argmax_real_sparse_site_and_residual_coders_bit_exact_roundtrip",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO_ROLLUP,
            reactivation_criteria=(
                "byte-closed exact-eval on a v8 archive shipping all 5 edge carriers + residual sidecars; "
                "OR building either headroom lever (de-share / curve-relative residual) and re-measuring"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )

    # ---- BUILD-#386 residual-kit anchors: de-share CONFIRMED (quantified) + curve-relative REFUTED ----
    anchor_deshare = EmpiricalAnchor(
        anchor_id="v8_residual_deshare_dedup_measured_20260709",
        measurement_utc=_UTC,
        inputs={
            "probe": "Lever-1 Movable-first de-share + general pairwise archive-dedup audit (n600)",
            "labels": "gt_n600.npz['lstars'] (self-detected Road0/Lane1/Undriv2/Movable3/MyCar4)",
            "module": "tac.boundary_math.movable_deshare.{measure_deshare_magnitude,pairwise_dedup_audit}",
            "coder": "tac.boundary_math.curve_relative_offset_coder.encode_absolute_2d (bit-exact)",
            "footprint": "Movable region dilate=2 (proxy for what the Movable bbox carrier holds)",
        },
        predicted_output={
            "law": "S_edge_deshared = S_edge - S_movable_overlap (EXACT set-partition, monotone, FREE)",
            "principle": "no archive byte reconstructible from another section + free generator "
                         "(operator no-duplicate-data binding 2026-07-09)",
        },
        empirical_output={
            "deshare_per_edge_amortized": {
                "horizon_Road_Undriv": {"attributed_px": 12592, "frac": 0.096, "bytes": 3917, "S": 0.00261},
                "lane_Road_Lane": {"attributed_px": 7009, "frac": 0.008, "bytes": 2689, "S": 0.00179},
            },
            "deshare_total_S": 0.00440,  # amortized-conservative; ~10.7% of the 0.0411 gap to sub-0.15
            "deshare_range_note": (
                "amortized (edge avg B/px) = CONSERVATIVE lower est; the overlap coded ALONE (secondary "
                "arcs are scattered) = horizon 9143 B / lane 6375 B -> honest range S 0.0044-0.0104"
            ),
            "pairwise_dedup_audit_top": {
                "hood<->road_mycar_sep": {"shared_px": 303229, "S": 0.02801, "verdict": (
                    "NOT a live double-count: the Road<->MyCar separatrix IS the hood silhouette -> its "
                    "UNIQUE geometric home is the hood carrier G4. v8's fold (Road/MyCar into hood, no "
                    "separate edge row) is VALIDATED; SPLITTING them would double-count 0.028")},
                "horizon_resid<->lane_resid": {"shared_px": 1430, "S": 0.00102, "verdict": (
                    "NEW small double-count at Road/Undriv/Lane triple-points -> assign each px to ONE "
                    "edge (free deflation ~0.001); found by the operator's general dedup sweep")},
            },
            "geometric_homes": (
                "horizon secondary arcs -> Movable silhouette G3; lane-near-car fragments -> Movable G3; "
                "Road<->MyCar boundary -> hood G4; horizon∩lane triple-points -> priority-assign one edge"
            ),
        },
        residual=0.0,
        source_artifact=_MEMO_KIT,
        measurement_method="n600_argmax_movable_first_deshare_plus_pairwise_dedup_bit_exact_coder",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO_KIT,
            reactivation_criteria=(
                "byte-closed exact-eval on a v8 archive that applies the Movable-first attribution + "
                "triple-point priority partition before coding residual sidecars"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )

    anchor_curve_rel_negative = EmpiricalAnchor(
        anchor_id="v8_curve_relative_offset_coder_NEGATIVE_20260709",
        measurement_utc=_UTC,
        inputs={
            "probe": "Lever-2 curve-relative signed-offset (s,n) coder vs absolute-2-D baseline (n600)",
            "labels": "gt_n600.npz['lstars']; residual measured on the DE-SHARED set (levers compose)",
            "module": "tac.boundary_math.curve_relative_offset_coder.measure_curve_relative (bit-exact both)",
        },
        predicted_output={
            "conjecture_S1": "2-D->1-D dim reduction: code n(s) (small signed offset) not absolute coords "
                             "-> ~4-5x residual shrink -> complete 0.140 toward ~0.077-0.083",
        },
        empirical_output={
            "horizon_deshared": {"on_curve": 0.99, "abs_msmax": 4.0, "H_n_bits": 2.73,
                                 "bytes_abs": 37753, "bytes_curve_rel": 38222, "ratio": 0.988},
            "lane_deshared": {"on_curve": 0.60, "n_mean_px": 96.4, "H_n_bits": 7.47,
                              "bytes_abs": 315790, "bytes_curve_rel": 352260, "ratio": 0.896},
            "verdict": (
                "REFUTED for this coder: curve-relative does NOT beat the absolute-2-D baseline on either "
                "edge. HORIZON: the offset IS small (|n|mean 4.0, H 2.73b confirming S1's geometry) but the "
                "absolute flat-index delta is ALREADY ~1/px for a near-horizontal arc -> no dimensional "
                "reduction available (0.988x). LANE: 40% of residual is OFF-curve and |n|mean=96px -> the "
                "Road/Lane residual is GENUINELY far-from-generator (S1's binding uncertainty RESOLVED on "
                "the FAR side) -> curve-relative cannot help (0.896x). The conjectured ~5x does NOT occur."
            ),
            "verdict_scope": "FORMULATION (this axis-aligned (s,n) offset coder on this residual set); "
                             "NOT PARADIGM (the shearlet dimensional-reduction idea is not tested to floor)",
            "reformulation_queue": [
                "improve the LANE GENERATOR to cut the 40% off-curve residual (the lever is the "
                "generator's coverage, not the residual coder) — the residual is far because the analytic "
                "band misses whole lane lines",
                "true Euclidean-normal offset (not axis-aligned) — unlikely to help horizon (already "
                "optimal), may tighten lane if generator improves",
                "joint 2-D (s,n) context model / learned entropy over the offset field",
            ],
            "consequence_for_P8_rate_row": (
                "Lever-2 does NOT close the 0.079 enemy. P8 residual stays ~complete-lossless: "
                "0.140 - de-share ~0.004 - triple-point ~0.001 ~= 0.135 (still ~1.15x the 0.118 frontier). "
                "v8's rate WIN is the DOMINANT-only 0.061 (thesis-confirmed); COMPLETE is a wash-with-frontier."
            ),
        },
        residual=0.0,
        source_artifact=_MEMO_KIT,
        measurement_method="n600_argmax_curve_relative_vs_absolute_2d_bit_exact_both",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO_KIT,
            reactivation_criteria=(
                "re-measure curve-relative AFTER improving the lane generator's coverage (reduce the 40% "
                "off-curve residual); OR a true-normal / learned-entropy reformulation lands"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )

    # ---- BUILD-#386 inc-1a harness anchor: the analytic-composite MASK-d_seg floor (harness smoke) ----
    anchor_inc1a_analytic = EmpiricalAnchor(
        anchor_id="v8_inc1a_analytic_composite_mask_dseg_floor_20260709",
        measurement_utc=_UTC,
        inputs={
            "probe": "inc-1a harness $0 smoke: composite argmax from ANALYTIC generators only "
                     "(openpilot lane band + horizon poly + hood clamp; NO trained fields), n600",
            "labels": "gt_n600.npz['lstars']",
            "module": "tac.inc1a_harness.{composite_assembler,mask_dseg_meter,analytic_smoke}",
            "bc_mode": "no_offset (safe default)",
        },
        predicted_output={
            "role": "harness positive-control + the FLOOR row any trained v8 Stage-A arm must beat; "
                    "NOT the 1a decoupling verdict (that needs trained arms + matched control)",
        },
        empirical_output={
            "analytic_composite_mask_dseg": 0.100403,  # [analytic-generators, no-trained-fields]
            "per_class": {
                "undrivable": {"dseg": 0.162, "flip_share": 0.797,
                               "note": "F4 single-valued-horizon lateral under-coverage, now MEASURED"},
                "movable": {"dseg": 1.0, "note": "unmodeled in analytic smoke -> folds to Road (expected)"},
                "mycar_hood": {"dseg": 0.0039, "note": "hood clamp near-perfect"},
            },
            "runtime_s": 29,
        },
        residual=0.0,
        source_artifact=".omx/research/inc1a_harness_build_20260709.md",
        measurement_method="n600_analytic_composite_argmax_vs_lstar_mask_dseg_deterministic",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=".omx/research/inc1a_harness_build_20260709.md",
            reactivation_criteria=(
                "the trained Stage-A decoupled arm + matched-compute control run through the same "
                "harness (governed EVENT); delta_mask floor from recess R7 replaces the delta_R proxy"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )

    # ---- inc-1a KILL-GATE RULING anchor (the pre-#385 $0 decoupling screen, RUN 2026-07-10) ----
    # The gate was RUN. Its honest ruling: the A/B is NOT $0-rulable — the matched-compute
    # shared-head CONTROL arm is a TRAINED artifact (governed EVENT + owed-9 lateral carrier
    # BUILD), so evaluate_kill REFUSES rather than fabricate one (NO-FAKE). What DID land at $0:
    # the analytic decoupled-arm floor reproduced EXACTLY (0.100403, n600), harness end-to-end
    # validation, and a 3-replicate determinism confirmation (spread=0 -> operative floor = R7).
    anchor_inc1a_killgate = EmpiricalAnchor(
        anchor_id="v8_inc1a_killgate_ruling_analytic_20260710",
        measurement_utc="2026-07-10T00:00:00Z",
        inputs={
            "probe": "inc-1a KILL-GATE RUN (pre-#385): attempt the decoupled-vs-matched-control A/B "
                     "at the analytic-generator level, n600, deterministic, $0/no-GPU",
            "labels": "gt_n600.npz['lstars'] (600x384x512 int64 argmax)",
            "config": "witness_autoconfig.derive_crucible_v8_inc1a_config (validate==[], unknown_count==0, 18 fields)",
            "evaluator": "tac.inc1a_harness.decoupling_screen.evaluate_kill (the harness OWNS the verdict)",
            "n_replicates": 3,
            "bc_mode": "no_offset",
        },
        predicted_output={
            "sealed_design": "A/B decoupled per-class fields vs MATCHED-COMPUTE shared-head control; "
                             "improvement > delta_mask -> CONFIRMED (proceed 1b)",
            "contingency": "SPEC_v8.1 §11 owed-9 + operator clause: run at the analytic level if the "
                           "trained arms / lateral carrier do not exist; do NOT fake a trained arm",
        },
        empirical_output={
            "verdict": "REFUSED-arm-missing-or-toy",
            "why_refused": "the matched-compute shared-head CONTROL arm is a TRAINED artifact (governed "
                           "EVENT + owed-9 lateral 3-curve carrier BUILD); NOT fabricated (NO-FAKE) -> the "
                           "decoupling A/B is structurally NOT $0-rulable",
            "decoupled_arm_analytic_mask_dseg_n600": 0.10040343390570747,  # reproduces the 0.100403 precedent EXACTLY
            "control_arm": "NOT RUN (does not exist)",
            "seed_spread_3rep": 0.0,
            "seed_spread_note": "deterministic generators -> spread=0 is a DETERMINISM confirmation, NOT a "
                                "measured training-seed variance (that instrument is N/A at the analytic level)",
            "operative_delta_mask": 3.46e-06,  # = R7 frame-sampling floor (seed component N/A)
            "per_class_mask_dseg": {"Road": 0.0212, "Lane": 0.3617, "Undrivable": 0.1616,
                                    "Movable": 1.0, "MyCar": 0.0039},
            "per_frame_std": 0.03991,
            "constraint_carved": (
                "the v8 kill-gate is NOT a $0 pre-training screen: no CONFIRMED/KILLED ruling is reachable "
                "at $0; choosing v8 at #385 commits to owed-9 (lateral 3-curve carrier BUILD) + a governed "
                "decoupled/control training EVENT before any A/B ruling exists. Banked at $0: the analytic "
                "decoupled FLOOR 0.100403 (any trained Stage-A arm must beat it) + harness validation"
            ),
        },
        residual=0.0,
        source_artifact=".omx/research/inc1a_killgate_run_20260710.md",
        measurement_method="n600_analytic_composite_argmax_vs_lstar_deterministic_evaluate_kill_refused",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=".omx/research/inc1a_killgate_run_20260710.md",
            reactivation_criteria=(
                "the governed EVENT lands the trained decoupled Stage-A arm + matched-compute shared-head "
                "control through the SAME harness; evaluate_kill then rules CONFIRMED/KILLED/INCONCLUSIVE "
                "with the operative delta_mask = max(R7 3.46e-6, in-run >=3-seed spread)"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "v8 geometric rate decomposition: whole-scene partition rate S_rate = sum_edges "
            "[S_gen(dominant ~k coeffs) + S_resid(uncovered px)]; MEASURED n600 bitmap 0.339 -> "
            "geometric-dominant 0.061 (5.5x < bitmap, 1.9x < the 0.118 frontier) -> lossless-complete "
            "0.140 (residual-coder-bound)"
        ),
        one_line_summary=(
            "Per-edge rate = parametric generator (dominant) + residual sidecar; MEASURED bitmap 0.339 "
            "-> dominant 0.061 (1.9x < frontier) -> complete 0.140 (residual-bound); pointer UNMOVED."
        ),
        latex_form=(
            r"S_{\mathrm{rate}} = \sum_{e \in \mathrm{edges}} \Big[ S_{\mathrm{gen}}^{(e)}"
            r"(\text{dominant},\,k\text{ coeffs}) + S_{\mathrm{resid}}^{(e)}(\text{uncovered px}) \Big],"
            r"\quad S = \tfrac{25\,\mathrm{bytes}}{37\,545\,489}"
        ),
        # Reproduction / producer callable: the real LBND2 lane coder that measured the dominant Road/Lane
        # stream (importable dotted path; this is a MEASUREMENT law, not a single closed-form derive fn).
        python_callable_module_path=(
            "tac.boundary_math.analytic_lane_render_band:serialize_lane_band_rd_tracked"
        ),
        domain_of_validity={
            "vehicle": ["v8_edge_centric_per_class_carrier"],
            "labels": "n600 GT argmax (gt_n600.npz['lstars']); real coders, bit-exact roundtrips",
            "scope": (
                "SNAPSHOT-scope RATE-half characterization on the DECODE side at THIS label cache — "
                "NOT a converged-witness fact and NOT a byte-closed exact-eval score"
            ),
            "measurement_axis": ["macOS-CPU advisory"],
            "promotion_eligible": False,
            "note": (
                "means != ends: dominant 0.061 beats the frontier RATE term, but the pointer moves only "
                "through a byte-closed upstream/evaluate.py exact-eval row; the d_seg half (#205) is the "
                "true blocker. Lossless COMPLETE (0.140) does NOT beat the frontier with today's residual "
                "coder — the 0.079 gap is the residual sidecar (de-share + curve-relative levers unbuilt)"
            ),
        },
        units_in={"edge_boundary_px": "count", "generator_coeffs": "count_per_frame"},
        units_out={"s_rate_term": "contest_rate_score_25bytes_over_N"},
        empirical_anchors=(anchor_roadlane, anchor_rollup, anchor_deshare, anchor_curve_rel_negative,
                           anchor_inc1a_analytic, anchor_inc1a_killgate),
        predicted_vs_empirical_residual={
            "roadlane_dominant_measured": 0.0,
            "wholescene_rollup_measured": 0.0,
            "deshare_lever1_confirmed_measured": 0.0,
            "curve_relative_lever2_refuted_measured": 0.0,
            "inc1a_killgate_ruling_analytic_measured": 0.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            ".omx/research/SPEC_v8_perclass_decomposition_20260708.md",   # SPEC_v8 carrier-table
            ".omx/research/v8_increment1_design_draft_20260709.md",        # the v8 increment-1 draft
            "task_359_v8_perclass_decomposition",                          # task #359 (gated on v7.5)
        ),
        canonical_producers=(
            "tac.boundary_math.analytic_lane_render_band",  # LBND2 lane coder (Road/Lane dominant)
            "tac.boundary_math.lane_sdf_component",          # Wave-F #234 lane fit machinery
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO_ROLLUP,
            reactivation_criteria=(
                "byte-closed upstream/evaluate.py exact-eval on a v8 archive shipping the per-edge "
                "generators + residual sidecars (advisory -> contest rate authority)"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )


def populate_v8_geometric_rate_decomposition_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the v8 geometric rate decomposition law (mirrors
    ``populate_witness_pose_grad_coeff_stability_equation``; latest-row-wins). EQUATIONS leg of the
    FEED-v8-rollup measurement; DSL leg = N/A (a measurement, no trainer lever / launch / curriculum
    change)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_v8_geometric_rate_decomposition_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="v8_geometric_rate_decomposition_20260709 (equations leg of FEED-v8-rollup; MEASURED "
              "whole-scene per-edge geometric rate; DSL leg N/A = measurement; advisory NON-PROMOTABLE)",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "build_v8_geometric_rate_decomposition_v1",
    "populate_v8_geometric_rate_decomposition_equation",
]
