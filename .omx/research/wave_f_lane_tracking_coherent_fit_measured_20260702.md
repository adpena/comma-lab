# Wave-F Stage-2b — CORRESPONDENCE-FIRST lane tracking + coherent batch denoise: BUILT + MEASURED (n600)

- **Date:** 2026-07-02
- **Status:** BUILT + MEASURED + tested (58/58) + decode-consistent (bit-exact through the REAL inflate
  subprocess) + default-off 7/7. Advisory / build-only. **Pointer contest-CPU 0.19110 UNMOVED** (moves
  only via a byte-closed `upstream/evaluate.py` n600 exact row). Every rate row is `[macOS-CPU advisory]`
  MEASURED brotli byte-count. NOT a score claim.
- **Research authority:** `lane_coeff_tracking_denoising_optimal_survey_20260702.md` (correspondence-first,
  RTS/l1-trend, RPCA). **Builds on:** the LANDED LBND2 codec + the ego-predictive NEGATIVE + moving-average
  POSITIVE (`wave_f_lane_band_rd_code_LANDED_stage1_measured` + `wave_f_unified_xi_build_measured`).
- **Result JSON:** `.omx/research/wave_f_lane_tracking_rate_n600_RESULT.json`.
- **Commits:** `2816b597b` (core codec + module) + this landing (decoupled-architecture refactor + tests +
  tool + memo).

---

## TL;DR (the honest headline)

The research survey DERIVED that CORRESPONDENCE-FIRST (fix the slot-swaps losslessly) would recover a large
fraction of the moving-average gain and land rate **~0.007–0.012**, because "44% of the temporal-delta L1
mass sits in the top-5% jumps = slot-swaps." **The n600 measurement FALSIFIES that expectation.** Built the
full correspondence-first pipeline (Hungarian tracking + bounded-K coherent slotting + Kalman-RTS batch
smoother + l1-trend + RPCA), all bit-exact + decode-consistent, and MEASURED it byte-closed at n600:

- **Correspondence IS lossless** (verified: the tracked/coherent-slot dequant lines == the sort dequant
  lines as a per-pair set; geometric RMS 0.021 m = quantization only) — but the rate win is **TINY: 0.5%**
  (`coherent_slot_none` 0.02750 vs LBND2 baseline 0.02765, ratio **0.995**), NOT the derived big win.
- **The moving-average REMAINS the rate winner** (`sort_MA_win25` **0.01489**, win15 0.01608) — no
  correspondence + batch-denoise variant beats it (best `coherent_slot_median` 0.01760).
- **Persistent-track packing is WORSE** (`persistent_none` 0.03001, ratio **1.085**) — a COLUMN EXPLOSION
  (K=21 tracks vs the sort's 6 slots over the 600-frame drive) whose birth/death spikes + sparse-column
  overhead exceed the swap savings.

**The corrected crux (MEASURED):** the dominant temporal-delta mass is **per-frame fit JITTER** (far-field
c2/c3 curvature outliers of the independent per-frame SegNet-mask polyfit; measured nearest-match p99 ~2.6 m
at 40 m), **NOT lateral slot-swaps.** The contest clip is highway-parallel: lanes rarely cross, and the
coeffs evolve smoothly with **no sharp lane-change edges**, so (a) the swap component correspondence can
remove is small, and (b) there are no edges for edge-preserving (l1-trend/Potts) denoise to exploit over the
moving-average. The moving-average (median) is already near the rate-distortion frontier for THIS data.

This is a clean **implementation-level negative** on the "correspondence-first beats moving-average on rate"
thesis (per NO-FAKE: negatives are deep-math signal, reported honestly, not faked into a win). The paradigm
(coherence-first) is intact and the machinery is real, tested, and reusable — but at n600 it does not lower
the lane rate below the moving-average.

---

## MEASURED n600 bake-off (real `gt_n600.npz` lstars, byte-closed, `[macOS-CPU advisory]`)

| variant | brotli B | rate_term | ratio vs LBND2 | geom RMS (m) | note |
|---|---:|---:|---:|---:|---|
| **sort_MA_win25** | 22,367 | **0.01489** | 0.539 | 6.07 | incumbent moving-average (median) — RATE WINNER |
| sort_MA_win15 | 24,149 | 0.01608 | 0.582 | 5.98 | incumbent |
| sort_MA_win9 | 26,260 | 0.01749 | 0.632 | 5.58 | incumbent |
| coherent_slot_median | 26,436 | 0.01760 | 0.637 | 7.02 | corr + clean-track median |
| coherent_slot_rts | 27,249 | 0.01814 | 0.656 | 9.50 | corr + Kalman-RTS (MMSE) |
| coherent_slot_l1trend | 36,087 | 0.02403 | 0.869 | 8.22 | corr + l1-trend (edge-preserving) |
| coherent_slot_rpca | 39,360 | 0.02621 | 0.948 | **2.96** | corr + RPCA (lowest distortion) |
| **coherent_slot_none** | 41,303 | 0.02750 | **0.995** | 0.021 | **lossless correspondence — 0.5% win** |
| LBND2_sort_baseline | 41,526 | 0.02765 | 1.000 | 0.000 | Stage-1 |
| persistent_none | 45,065 | 0.03001 | 1.085 | 0.021 | persistent tracks — COLUMN EXPLOSION |

- **The one genuine positive:** `coherent_slot_none` = a **LOSSLESS 0.5% rate refinement** (bounded-K
  Hungarian slotting minimizing the temporal delta instead of the lateral sort). Free, deterministic,
  ships-as-LBND2. Small, honest.
- geom RMS is a far-field-amplified diagnostic (c3 diff × 60³ m); useful only for RELATIVE comparison. RPCA
  (low-rank) preserves the trajectory best (2.96) but compresses worst; the moving-average trades a modest
  RMS (~6) for the lowest rate.

## Why the survey's premise missed (the deep-math correction, one principle)

The survey's load-bearing claim — "a slot-swap is an index-permutation discontinuity that defeats every
temporal model; correspondence removes it losslessly for a big win" — is CORRECT as a principle but
**empirically minor for this data**: the measured top-5% jump mass is dominated by **fit-jitter** (each pair
is an INDEPENDENT SegNet-mask polyfit; the far-field curvature is noisy), which correspondence CANNOT remove
(it is a genuine per-frame bad fit, not a re-labeling). The two failure modes the survey attributed to
swaps are actually:
1. **fit-jitter** → removed by DENOISING (the moving-average's −42%), not by correspondence.
2. **lane-count re-indexing** (a lane appears/leaves → the lateral sort shifts every slot index) → the ONLY
   part correspondence removes → measured ~0.5%.
The bounded-K coherent slotting captures (2) losslessly; nothing cheap captures more, because (1) is
information (real per-frame fit noise) that only a lossy denoiser removes, and the moving-average is already
near the R-D frontier on this edge-free highway data.

**Persistent-tracks vs bounded-K (the measured architecture lesson):** persistent tracks (one column per
physical lane over its lifetime) give clean per-lane series but K grows with lane turnover (n600 K=21) → a
column explosion whose birth spikes + sparse-column overhead cost MORE than the sort's 6-slot reuse. The
correct realization is **bounded-K coherent slotting** (K = max-concurrent, like the sort; Hungarian-assign
lines to slots to minimize the delta; reuse a freed slot for a new lane). The batch denoise still runs in
the CLEAN persistent-track space then re-packs compact (the decoupling), so denoising is within-lane (not
smeared across swaps like the sort+moving-average) — but on edge-free data that advantage does not beat the
moving-average on rate.

## What was built (all committed, tested, decode-consistent)

- **`src/tac/boundary_math/lane_track_and_smooth.py`** (new, reusable estimators):
  `track_lane_slots` (Tier-A Hungarian persistent tracker; near-field-lateral association robust to
  curvature jitter + gap revival) · **`coherent_slot_pack`** (bounded-K coherent slotting = the lossless
  rate winner among correspondence variants) · `track_and_batch_denoise_lines` (the CLEAN decoupled denoise
  source transform) · `_rts_local_linear_trend` (Kalman-RTS fixed-interval MMSE smoother) ·
  `_l1trend_1d` (Kim-Koh-Boyd edge-preserving, deterministic ADMM + scipy banded Cholesky) · `_median_1d` ·
  `_rpca_pcp` (Principal Component Pursuit inexact ALM, guarded) · `coherent_denoise_track_matrix` (method
  dispatch) · `top_pct_jump_mass` (the swap-signature diagnostic).
- **`analytic_lane_render_band.py`:** refactored `serialize_lane_band_rd` via `_serialize_matrix_lbnd2`
  (sort path BYTE-IDENTICAL) + `serialize_lane_band_rd_tracked` (pack_mode ∈ {coherent_slot, persistent,
  sort}; ships LBND2 bytes) + `roundtrip_lines_through_rd_tracked` + `lane_band_tracking_rate_report`
  (the gate-order MEASURED bake-off + geom RMS) + `_dequant_lines_multiset_key` (lossless invariant) +
  `_induced_lateral_rms_vs_raw`.
- **`tools/wave_f_lane_tracking_rate_n600.py`** — the $0 CPU no-scorer n600 bake-off tool.
- **Tests:** `src/tac/tests/test_lane_tracking_coherent_fit.py` (25) + `experiments/tests/
  test_levelset_lane_band_decode_consistency.py` +3 Group-C subprocess bit-exact gates (tracked LBND2 ==
  numpy oracle through the REAL inflate, `max_abs_uint8_diff == 0`). Existing LBND2/ego suites unbroken.

## Acceptance gates (all HONORED)

1. **Decode-consistency bit-exact** — ✅ the SHIPPED inflate.py band render (coherent_slot & persistent,
   with/without denoise) == the numpy-fp32 oracle, bit-for-bit (`max_abs_uint8_diff == 0`), through the REAL
   inflate subprocess. Ships as STANDARD LBND2 bytes — the UNCHANGED `_lane_parse_rd` decodes it (ZERO new
   inflate code; provenance keys in the header `rd` block are inert to the decode).
2. **LOSSLESS invariant** — ✅ the correspondence-only dequant lines == the sort dequant lines as a per-pair
   quantized multiset (geom RMS 0.021 m = quantization only). Correspondence changes the slot INDEX, never a
   lane's coeffs.
3. **Default-off byte-identical 7/7** — ✅ untouched (`wire_in_224_byte_identical_smoke.py` PASS).
4. **rule-118** — ✅ tracking + batch denoise are FREE (offline generic algorithm; the decoder renders
   whatever coeffs each slot holds); COUNTED = the (coherent) LBND2 quantized temporal-delta coeff stream +
   presence bitmap (SAME KIND as LBND2). NO GT mask, NO scorer weights, NO supercombo weights ship.
5. **MEASURED rate @ n600** — ✅ real byte-closed serialization, per variant + geom RMS.
6. **Determinism + host-portable** — ✅ decode is the numpy-fp64 LBND2 path (zero mlx); the tracker
   (stable-argsort LAP tie-break), RTS, l1-trend ADMM, RPCA ALM are all fixed-iteration deterministic.

## Honest verdict + what's deferred

**Correspondence-first (bounded-K coherent slotting) is a REAL but SMALL (0.5%) LOSSLESS rate refinement; it
does NOT beat the moving-average on rate at n600, and persistent-track denoise is dominated by the column
explosion.** The survey's DERIVED 0.007–0.012 is FALSIFIED for this data — the delta mass is fit-JITTER, not
slot-swaps, and the edge-free highway clip gives the edge-preserving denoisers no advantage over the
moving-average. The machinery is real, tested, decode-consistent, and reusable.

**Deferred / next (named):**
1. **The #205 through-R d_seg leg is the REAL gate** (per ANTI-SIGNAL-LOSS measurement-first): the value of
   the correspondence + edge-preserving denoise is LOWER GEOMETRIC DISTORTION at matched rate + WITHIN-lane
   (non-smeared) denoising — whether that NETS lower exact d_seg vs the moving-average's blur is a scorer
   run, gated. Ship `coherent_slot_none` (lossless 0.5%) as the safe default band source; A/B the
   coherent+RPCA (lowest-distortion) vs sort+MA in the trained-in run.
2. **Task-λ denoise** (`coherent_denoise_track_matrix(per_dim_lambda=...)` seam, landed): set the l1-trend λ
   per-coeff by the MEASURED margin-saliency `∂d_seg/∂coeff` at the KKT point — needs the scorer through R
   (out of scope). This is where edge-preserving could beat the moving-average ON THE TASK metric even
   though it does not on the geometric metric.
3. **Global min-cost-flow (Tier B) + supercombo prior + pooled-PMF range coder** — documented seams; low EV
   given the measured smallness of the correspondence gain. Do NOT over-invest before the #205 leg.

## Wire-in (6-hook, research_only)
1. Sensitivity-map: the per-variant Δrate + geom-RMS rows → `tac.sensitivity_map` (rate + distortion axes);
   the `per_dim_lambda` task-λ seam IS a sensitivity-map consumer. 2. Pareto: the smoothing method/λ and the
   pack_mode are rate↔distortion Pareto knobs (#205 measures the d_seg leg). 3. Bit-allocator: the batch
   denoise is a source-denoise pre-pass to the LBND2 allocator. 4. Cathedral autopilot: N/A (research_only;
   feeds the existing byte-close LBND2 path; no new archive artifact). 5. Continual-learning: this memo +
   `wave_f_lane_tracking_rate_n600_RESULT.json` are the anchors. 6. Probe-disambiguator: the bake-off IS the
   disambiguator (correspondence-only vs +denoise vs moving-average, resolved by measured n600 bytes + RMS).

**Council mission-contribution:** `frontier_breaking` attempt tempered by an HONEST measured negative on the
rate thesis + a genuine (small) lossless win + reusable estimators for the #205 leg. All MEANS; the END is
the #205 byte-closed exact row. Pointer 0.19110 UNMOVED.

## Sisters
`lane_coeff_tracking_denoising_optimal_survey` (the research this MEASURES + partially FALSIFIES) ·
`wave_f_lane_band_rd_code_LANDED_stage1_measured` · `wave_f_unified_xi_build_measured` (ego NEGATIVE +
moving-average POSITIVE) · `analytic_lane_band_primary_authority_decomposition` ·
`project_contest_is_indirect_rate_distortion_task_space_coding` ·
`not-pessimistic-first-results-adversarial-deepmath-oss-against-negatives`.
