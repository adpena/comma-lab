# v8 residual-kit BUILD — de-share (Lever-1) + curve-relative δ(s) coder (Lever-2) — MEASURED n600

**Date:** 2026-07-09 · **Task:** BUILD #386 ("build all unbuilt") · **Axis:** `[macOS-CPU advisory ·
NON-PROMOTABLE]` · pointer contest-CPU **0.19110 UNMOVED** · #205 STOPPED/untouched · MPS/GPU untouched
(pure numpy/scipy/brotli on the label cache). `$0`, read-only on `gt_n600.npz['lstars']` (600×384×512
int64 argmax). Every number MEASURED through a REAL bit-exact coder; NO b/px proxy, NO projection.

## Answer-first (P8-brief-ready)

**The 0.079 residual enemy (= 192% of the 0.0411 gap to sub-0.15) is attacked by two levers, now BUILT
+ MEASURED at n600. The honest verdict: Lever-1 pays a real but modest ≈0.0044 S; Lever-2 does NOT pay.**

- **Lever-1 DE-SHARE (Movable-first attribution): CONFIRMED, S ≈ 0.0044 (range 0.0044–0.0104).** 12,592
  horizon-residual px (9.6%) + 7,009 lane-residual px (0.8%) are Movable double-count → attributing them
  to the Movable carrier deflates the complete number for FREE. ≈10.7% of the remaining gap.
- **Lever-2 CURVE-RELATIVE δ(s) coder: REFUTED for this coder, MEASURED.** Does NOT beat the absolute-2-D
  baseline on either edge (horizon 0.99×, lane 0.90×). The conjectured ~5× does not occur. verdict_scope
  FORMULATION (not paradigm); reformulation queue below.
- **Pairwise dedup audit (operator no-duplicate-data binding):** the biggest "overlap" — hood ↔ Road/MyCar
  separatrix (0.028) — is NOT a live double-count but a **structural VALIDATION** that v8 correctly gives
  Road↔MyCar a single geometric home (the hood carrier G4); a NEW small horizon∩lane triple-point
  double-count (0.001) was found.
- **Net for the P8 rate row:** COMPLETE-lossless **0.140 − de-share ~0.004 − triple-point ~0.001 ≈ 0.135**
  (still ~1.15× the 0.118 frontier). Lever-2 does NOT get it toward 0.061. **v8's rate WIN stays the
  DOMINANT-only 0.061 (thesis-confirmed); COMPLETE is a wash-with-frontier.** This is the decisive empirical
  answer S5(adversary)/S1 flagged — the residual-coder-optimism trap is de-risked with a MEASURED no.

Pointer moves only through a byte-closed `upstream/evaluate.py` n600 exact row; this is MEANS. The d_seg
half (#205) remains the true blocker.

## STORES CONSULTED (recall, not re-derive)
- `docs/operating_manual_craft_handoff.md` (§3 blast-radius, §4 re-derive-from-primary, §5 label MEASURED,
  §6 attack-own-conclusion, §8.1 honest-negative-is-a-deliverable).
- `t5_crucible3/SYNTHESIS_DRAFT_v8_20260709.md` (T2 rate policy: TRIPLE row; §E.R1/R2 the two recess probes)
  · `position_S1_deepmath_20260709.md` §4 (R(D) framing, de-share EXACT-monotone, curve-relative
  DOUBLY-CONDITIONAL, binding uncertainty "Road/Lane may be far-from-generator AND d_seg-valuable")
  · `position_S6_structureblind_20260709.md` §2 (residual chart, de-sharing partition, geometric homes).
- `v8_movable_residual_rollup_20260709.md` (the 0.061/0.140 ledger, the two headroom levers named
  un-exploited; the shared residual coder pattern) · `v8_roadlane_geometric_rate_20260709.md` (LBND2 lane
  coder) · registered eq `v8_geometric_rate_decomposition_v1`.
- `operator_no_duplicate_data_archive_geometry_first_20260709.md` (binding; de-share = first instance of a
  general archive-dedup audit; every byte names one geometric home).
- Primary artifacts RE-DERIVED (§4): `road_undriv_bulk_field.py` (`_horizon_profile` + deg-3 polyfit reused
  verbatim for the horizon residual), `analytic_lane_render_band.py` (`build_analytic_lane_band_prior`
  coverage for the lane residual + generator curves), `margin_conditional_residual.py` (the #226 residual
  coder pattern the baseline mirrors).

## What was built (modules, default-OFF, byte-close-integrated only behind future flags)

1. **`src/tac/boundary_math/movable_deshare.py`** — Lever-1 + the general dedup audit.
   - `detect_seg_roles` (self-detects Road/Undriv via the proven helper + Lane/Movable/MyCar by
     area/centroid signature — NO hardcoded argmax order, per CLAUDE.md canonical rule; verified against
     comma10k order on the real cache).
   - `separatrix_mask`, `movable_footprint` (dilate=2), `horizon_residual_idx` (deg-3 poly miss),
     `lane_residual_idx` (analytic-band coverage<0.5) — deterministic numpy primitives.
   - `deshare_partition` — Movable-first attribution with a PARTITION GUARANTEE (kept ⊔ attributed =
     input, disjoint; no px double-counted or lost).
   - `measure_deshare_magnitude` (Probe 1) + `pairwise_dedup_audit` (Probe 1b, all C(5,2)=10 ledger-row
     pairs).
2. **`src/tac/boundary_math/curve_relative_offset_coder.py`** — Lever-2 + the absolute-2-D baseline.
   - `encode_absolute_2d`/`decode_absolute_2d` — the generic 2-D flat-index residual coder (the ~0.4–0.6
     B/px baseline the curve-relative coder must beat), single-stream brotli q11, bit-exact.
   - `GeneratorCurve` + `curve_from_column_function` (horizon) + `curves_from_coverage_mask` (lanes;
     explicit monotone segmentation resolves the multi-valued JUNCTION case — S1's flagged degradation).
   - `chart_transform`/`reconstruct_from_chart` — residual px → `(seg_id, s, n)` + off-support
     `exceptions` (the honest lossless top-up). Bit-exact by INTEGER construction (`other = center[s] + n`).
   - `encode_curve_relative`/`decode_curve_relative` — bit-exact; `delta_s_spectrum` (offset entropy +
     Haar N-term); `measure_curve_relative` (Probe 2, with a self-verifying decode(encode)==input assert).
3. **`tools/measure_v8_residual_kit_deshare_curverel.py`** — the $0 driver (runs both probes on n600,
   emits `.omx/research/residual_kit_measured_20260709.json`).
4. **Tests (16/16):** `tests/test_movable_deshare.py` (partition guarantee, footprint, determinism, role
   self-detect on real cache, measurement + audit smoke) · `tests/test_curve_relative_offset_coder.py`
   (absolute + curve-relative roundtrip bit-exactness, col-param + row-param + junction multi-segment +
   off-support exceptions, spectrum/N-term sanity, savings on a synthetic smooth signal).

## ROUNDTRIP PROOF (NO-FAKE — the coder reproduces the boundary it claims)
Both coders assert `decode(encode(x)) == x` bit-for-bit; the n600 driver reports
`curve_relative_bit_exact=True` and `absolute_bit_exact=True` for BOTH edges on all 600 frames. The
curve-relative bytes INCLUDE the off-support absolute exceptions (the lossless top-up). Junction /
off-support / empty edge cases are covered by dedicated tests.

## MEASURED tables (n600) — see the DAG FEED-residualkit for the formatted versions
- **Probe 1 de-share:** horizon 12,592 px / 3,917 B / S 0.00261 · lane 7,009 px / 2,689 B / S 0.00179 ·
  TOTAL 6,606 B / **S 0.00440** (amortized-conservative; overlap-coded-alone ceiling S 0.0104).
- **Probe 1b dedup audit (top pairs):** hood↔Road/MyCar-sep 303,229 px / S 0.028 (STRUCTURAL VALIDATION,
  not a live double-count) · horizon↔movable 12,592 px (the de-share) · lane↔movable 7,009 px (the
  de-share) · horizon↔lane 1,430 px / S 0.001 (NEW triple-point double-count).
- **Probe 2 curve-relative (de-shared):** horizon on-curve 0.99, |n|mean 4.0, H 2.73 b, ratio **0.99×** ·
  lane on-curve 0.60, |n|mean 96.4, H 7.47 b, ratio **0.90×**. Both bit-exact.

## GEOMETRIC HOMES (operator binding — every byte names ONE home)
- horizon secondary-arc residual px → **Movable silhouette (G3)** (cars breaking the horizon).
- lane-fragment-near-car residual px → **Movable silhouette (G3)**.
- Road↔MyCar boundary px → **hood carrier (G4)** (the separatrix IS the hood silhouette; fold VALIDATED).
- horizon∩lane triple-point px → **priority-assigned to ONE edge** (the smaller-residual edge).
A residual px that cannot name one of these homes is suspect; the dedup sweep found none unaccounted.

## Adversarial self-review (§6, before commit)
1. **Did the levers get a fair shot?** Curve-relative measured on the DE-SHARED residual (Lever-1 removes
   the far Movable arcs FIRST — the levers compose); single-stream brotli (not per-field, which would
   penalize it); offset axis chosen per curve (col for horizon, row for lanes). The horizon offset IS
   small (H 2.73 b) exactly as S1 predicts — the loss is that absolute is ALREADY near-optimal there, not
   that the chart is bad. The lane loss is because the residual is genuinely far (measured, not assumed).
   The negative is robust.
2. **Is the de-share magnitude honest?** Reported as AMORTIZED (edge avg B/px × attributed px) —
   conservative — with the coded-diff diagnostic flagged non-monotone (delta-gap effect) and the
   overlap-coded-alone ceiling given as the range top. Free-ness depends on the Movable carrier's actual
   coverage; approximated by the dilate=2 footprint (documented caveat).
3. **Class fix vs instance fix?** De-share generalized to the full pairwise dedup audit (operator binding)
   — the class, not just Movable↔horizon.
4. **Would the tests pass if the coder were broken?** No — the roundtrip tests mutate the pixel set and
   assert exact recovery; a broken chart/coder fails them (verified by construction).

## Triality legs
- **DAG:** `FEED-residualkit` appended to `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **Equations:** 2 anchors APPENDED to `v8_geometric_rate_decomposition_v1`
  (`v8_residual_deshare_dedup_measured_20260709` CONFIRMED · `v8_curve_relative_offset_coder_NEGATIVE_20260709`
  REFUTED), both `VERIFIED_VIA_EMPIRICAL_ANCHOR`, registered idempotently.
- **DSL:** **N/A** — this is a MEASUREMENT + two default-OFF coder modules; no trainer/byte-close flag
  exists yet (grep of `src/tac/witness_dsl/` confirms no deshare/curve-relative surface). A `Lever` factory
  is owed ONLY when byte-close integration lands behind a flag (future work); building one now would be a
  premature orphan.

## Reformulation queue (Lever-2 negative — verdict_scope FORMULATION, NOT paradigm)
1. **Improve the LANE GENERATOR coverage** to cut the 40% off-curve lane residual — the lever is the
   generator's coverage (the analytic band misses whole lane lines), NOT the residual coder. This is the
   highest-value next move for the Road/Lane 0.042 chunk (53% of the enemy).
2. True Euclidean-normal offset (not axis-aligned) — unlikely to help horizon (already optimal), may
   tighten lane only if the generator improves first.
3. Joint 2-D / learned-entropy context model over the offset field.

`[triality: DAG=FEED-residualkit · equations=2 anchors on v8_geometric_rate_decomposition_v1 · DSL=N/A]`
· pointer 0.19110 UNMOVED · #205 untouched · MEANS not ends.
