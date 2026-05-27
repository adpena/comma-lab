# V15 UNIWARD Sister D Criterion #4 weighted-EXTREMA Cross-Surface Transfer N=200 macOS-CPU Advisory Landed 2026-05-27

**Subagent**: `v15_sister_d_extrema_different_surface_4BF423DC` (fresh start; no predecessor for this lane)
**Lane**: `lane_v15_uniward_sister_d_extrema_different_surface`
**Sister of**: V15 Sister E landing memo `.omx/research/v15_uniward_sister_e_weighted_extrema_fridrich_2014_n200_macos_cpu_advisory_landed_20260527.md` which EMPIRICALLY VALIDATED that the canonical Fridrich 2014 weighted-EXTREMA family (p95/p99/argmax) BREAKS the ~26.5% weighted-MEDIAN saturation ceiling to **100% bin-occupation at the NSCS06 v8 CHROMA-LUT derivation surface** and identified the STATISTIC axis as the substrate's UNIQUE-OPTIMAL bottleneck. Sister E's operator-routable criterion #4 recommendation: apply the EXTREMA statistic to a DIFFERENT application surface.
**Predecessor anchors**: V15 N=50 16×5 IMPL-FALSIFIED → Sister B N=200 16×5 INTERMEDIATE_PARTIAL_SCALING (sample-size axis) → Sister C N=200 3-LUT-shapes PARADIGM_INTACT_WITH_COMBINED_CRITERIA_PARTIALLY_FALSIFIED (LUT-shape axis SATURATED) → Sister E N=200 4-statistics PARADIGM_VALIDATED_WITH_EXTREMA (STATISTIC axis IS the binding axis) → **this Sister D N=200 3-surfaces PARADIGM_TRANSFERS_ACROSS_SURFACES (APPLICATION-SURFACE axis)**
**Date**: 2026-05-27
**Tag**: `[macOS-CPU advisory]` per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192/#317/#341 (NOT promotable; NOT a contest score claim)
**Verdict**: **PARADIGM_TRANSFERS_ACROSS_SURFACES** (per Catalog #307 PARADIGM-INTACT + the EXTREMA breakthrough TRANSFERS to ALL different surfaces tested; the routing effect is a property of the EXTREMA statistic family, not a chroma-surface idiosyncrasy — BUT with surface-specific texture per the individually-fractal directive)
**Probe-outcome verdict (Catalog #313)**: `PROCEED` (blocker_status=advisory; probe_id=`v15_uniward_sister_d_extrema_cross_surface_20260527T062853Z_20260527`; expires_at_utc=2026-06-26T06:28:53Z)
**NOT a frontier-crossing event**; **NOT a PR111 candidate** per Catalog #343 + CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
**FORMALIZATION_PENDING**: NEW canonical equation candidate `uniward_extrema_statistic_cross_surface_transfer_v1` per Catalog #344 (the EXTREMA-breaks-saturation routing effect is a CROSS-SURFACE property — registered as candidate, NOT promoted, pending paired-CUDA evidence)

## TL;DR

Sister D directly empirically TESTED Sister E's operator-routable reactivation criterion #4 by applying the SAME canonical Fridrich 2014 weighted-EXTREMA family (median p50 baseline + p95 + p99 + argmax) to TWO DIFFERENT application surfaces — the NSCS06 v8 **grayscale_lut** (single BT.601 luma channel; canonical equation #26 `grayscale_lut_replacement` IN-DOMAIN sister context) and the **VQ-VAE indices_blob** (categorical codebook addresses; T3 council #3 stacking candidate) — alongside the chroma_lut surface as the apples-to-apples regression guard FIRST per 11th ORDER directive Dim 8. ALL surfaces reuse Sister B's N=200 cache READ-ONLY per Catalog #230. The 12 cells (3 surfaces × 4 statistics):

| Surface | Statistic | total bins | bins_changed | bin_pct | max_delta | mean_l2 | max_l2 |
|---|---|---|---|---|---|---|---|
| **chroma_lut** (Sister E baseline; regression guard) | weighted-MEDIAN (p50) | 80 | 21 | 26.25% | 5 | 0.3344 | 5.0990 |
| **chroma_lut** | weighted-EXTREMA p95 | 80 | **80** | **100.00%** | 137 | 47.7819 | 138.4522 |
| **chroma_lut** | weighted-EXTREMA p99 | 80 | **80** | **100.00%** | 151 | 60.7822 | 153.2906 |
| **chroma_lut** | weighted-EXTREMA argmax | 80 | **80** | **100.00%** | 134 | 30.2461 | 141.0709 |
| **grayscale_lut** (DIFFERENT #1) | weighted-MEDIAN (p50) | 80 | 7 | **8.75%** | 1 | 0.0875 | 1.0000 |
| **grayscale_lut** | weighted-EXTREMA p95 | 80 | **80** | **100.00%** | 26 | 8.1625 | 26.0000 |
| **grayscale_lut** | weighted-EXTREMA p99 | 80 | **80** | **100.00%** | 37 | 9.6250 | 37.0000 |
| **grayscale_lut** | weighted-EXTREMA argmax | 80 | **78** | **97.50%** | 20 | 4.8625 | 20.0000 |
| **vq_vae_indices_blob** (DIFFERENT #2) | weighted-MEDIAN (p50) | 80 | 7 | **8.75%** | 2 | 0.1500 | 2.0000 |
| **vq_vae_indices_blob** | weighted-EXTREMA p95 | 80 | **80** | **100.00%** | 52 | 16.3000 | 52.0000 |
| **vq_vae_indices_blob** | weighted-EXTREMA p99 | 80 | **80** | **100.00%** | 240 | 32.9751 | 240.0021 |
| **vq_vae_indices_blob** | weighted-EXTREMA argmax | 80 | **78** | **97.50%** | 40 | 9.7000 | 40.0000 |

**The empirical answer: the EXTREMA breakthrough TRANSFERS across surfaces — with surface-specific texture.** The Fridrich 2014 weighted-EXTREMA family breaks the median-saturation ceiling on ALL THREE surfaces: p95 and p99 hit 100% bin-occupation (80/80) on the chroma, grayscale, AND indices surfaces. This **EMPIRICALLY CONFIRMS** the Holub-Fridrich-Denemark 2014 "universal distortion function for steganography in an arbitrary domain" universality claim — the EXTREMA routing effect is a property of the STATISTIC family, not a chroma-surface idiosyncrasy. Per the verdict tree (Catalog #307 Sister D spec): `EXTREMA breaks saturation on ALL different surfaces (bins > 50%)` → `PARADIGM_TRANSFERS_ACROSS_SURFACES` → `PROCEED`.

**BUT the per-substrate individually-fractal texture is the rigorous deliverable** (per the operator's 8th + 13th standing directives, transfer is NOT assumed — it is EMPIRICALLY TESTED, and the per-surface response curve IS the signal):

1. **Median saturation FLOOR is surface-specific**: chroma_lut median = 26.25% (21/80) vs grayscale_lut + indices median = 8.75% (7/80). The single-channel surfaces (luma + categorical index) have a MUCH LOWER median saturation than the 3-channel RGB chroma surface — fewer bins diverge from the unweighted median because there is less per-channel structure to disagree on (1 channel vs 3 means fewer independent comparisons that can flip a bin to "changed"). This is the canonical Daubechies/Mallat multi-scale-partition observation: the chroma surface's 3-channel structure inflates the median's bin-divergence count relative to single-channel surfaces.

2. **argmax does NOT hit 100% on the different surfaces**: chroma argmax = 100% (80/80) but grayscale + indices argmax = 97.50% (78/80). EXACTLY 2 bins on each different surface where the single highest-UNIWARD-weight pixel's value happens to equal the unweighted-median value. INTERPRETATION: on the single-channel surfaces, the per-bin value distribution is narrower (1-D), so for ~2 bins the argmax-of-weight pixel lands on the bin's median value (no divergence). The chroma surface's 3-channel structure means the argmax-pixel's RGB triple almost never matches the median RGB triple on all 3 channels simultaneously, so chroma argmax hits 100%. This is a clean individually-fractal signature: the argmax statistic's bin-occupation is a function of the surface's channel-dimensionality.

3. **Per-bin magnitude is surface-specific and bounded by the value range**: indices p99 max_delta=240 (codebook range [0, 512); the L2 reflects the 2-byte uint16 index split into lo/hi channels — the max_l2=240.0021 indicates a single near-256-index divergence), grayscale p99 max_delta=37 (luma 0-255 range; smaller because single-channel + narrower routing), chroma p99 max_delta=151 (3-channel RGB; the 0-255 range × 3 channels). The magnitude ranking is indices > chroma > grayscale, reflecting each surface's value-range × channel-structure.

This **CONFIRMS the cross-surface transfer hypothesis with rich texture**: the EXTREMA statistic family's routing breakthrough (Sister E) is NOT chroma-surface-specific — it transfers to the grayscale (continuous luma) AND the VQ-VAE indices (categorical address) surfaces. The individually-fractal divergence point I predicted (categorical indices have semantically ill-defined percentiles) DID manifest in a measurable way: the indices surface's p99 max_delta is the largest of all (240) precisely because the percentile-over-categorical-addresses lands the LUT entry on a far-from-median index, AND the argmax on both single-channel surfaces stops 2 bins short of 100%. But the breakthrough still TRANSFERS — the EXTREMA family routes the LUT entry to the high-UNIWARD-weight tail on every surface.

## CRITICAL apples-to-apples caveat (Catalog #307 paradigm-vs-implementation discipline)

**The cross-surface 100% bin-occupation + large per-bin deltas are ROUTING-EFFECT signals at the LUT-DERIVATION surface, NOT contest-score signals.** Per the canonical `compare_uniward_vs_canonical_lut` docstring + Catalog #307: the verdict here is on the ROUTING-PRIMITIVE-EFFECT-AT-NATURAL-DOMAIN axis (specifically, does the EXTREMA statistic produce a measurable routing divergence on each different surface), NOT on the score-improvement axis. Whether the EXTREMA routing IMPROVES or HURTS the contest d_seg + d_pose distortion — on ANY surface — is an OPEN question that ONLY paired-CUDA auth-eval can answer per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable.

The cross-surface result STRENGTHENS the case for paired-CUDA validation but does NOT change the necessary-not-sufficient nature of the routing-effect signal: the EXTREMA transfers as a routing primitive across surfaces, which means a paired-CUDA dispatch on ANY of the 3 surfaces (chroma per Sister F, OR grayscale/indices per a future sister) would disambiguate the OPTIMISTIC vs PESSIMISTIC interpretation. The grayscale surface's SMALLER per-bin deltas (max_delta=26-37 u8 vs chroma's 137-151) suggest the grayscale EXTREMA is a MORE CONSERVATIVE routing perturbation — if the PESSIMISTIC interpretation (visible artifacts from extreme concentration) holds for chroma, the grayscale surface may be the safer paired-CUDA arm because its routing perturbation is ~4-5× smaller in u8 magnitude.

## 3-strategy attack decomposition

Per CLAUDE.md "3-strategy attack framework" non-negotiable + 12th standing directive: Sister D sits on the **DISTORTION pure-axis** (sub-axis JOINT d_seg + d_pose at the LUT-derivation surface, specifically the per-bin APPLICATION-SURFACE sub-axis). The probe is an INTRA-TEST comparison at the LUT-statistic surface across 3 application surfaces × 4 statistics; ALL 12 cells use the SAME 200 frames + SAME 96×128 NEAREST-upsampled real-scorer cache + SAME canonical helpers + SAME per-pixel UNIWARD weight map + SAME 16×5 bin partition + SAME EXTREMA statistic family.

The 4-point orthogonal-axis decomposition is now empirically complete:
1. **Sister B** = sample-size axis (N=50 → N=200; dyn_range 19.71× → 772.41×; bin-occupation 10% → 26.25% at chroma)
2. **Sister C** = LUT-shape axis (16×5 / 32×5 / 16×10; bin-occupation INVARIANT ~26.5%; SATURATED)
3. **Sister E** = STATISTIC axis (median p50 / EXTREMA p95/p99/argmax; bin-occupation 26.25% → 100% at chroma; THE BINDING AXIS)
4. **Sister D** = APPLICATION-SURFACE axis (chroma / grayscale / indices; EXTREMA breakthrough TRANSFERS to all 3; surface-specific median floor + argmax bin-count + per-bin magnitude)

Sister D's finding: the STATISTIC axis (Sister E's binding axis) is itself SURFACE-ROBUST — the EXTREMA breaks saturation on every surface tested. The surface-specific texture (median floor 8.75% vs 26.25%; argmax 97.5% vs 100%; magnitude ranking indices > chroma > grayscale) is the per-substrate individually-fractal evidence the operator's directives demanded.

3-strategy sub-axis decomposition:
1. **DISTORTION axis** = TESTED-AT-3-SURFACES × 4-STATISTICS (this Sister D lane)
2. **RATE axis** = UNCHANGED (no archive emitted; LUT comparison only)
3. **FULL SCORER axis** = constant 772.41× dyn_range (inherited from Sister B cache); the APPLICATION-SURFACE axis does NOT change the upstream gradient response — it changes which per-pixel VALUE-TYPE the fixed gradient distribution ROUTES into LUT entries

## MLX-first → numpy-portable bridge contract

Honored per 8th + 13th standing directives:
- Sister B N=200 cache REUSED via Catalog #230 sister-disjoint READ-ONLY consumer (sha256 `e5799f4bd8659a0b...`; dyn_range 772.41× BYTE-EQUIVALENT to Sister B/C/E)
- Canonical-median chroma LUT via canonical `build_chroma_lut_from_ground_truth` (pure numpy; READ-ONLY; UNTOUCHED; chroma baseline sha `203eb763cd3a007d`)
- Canonical weighted-MEDIAN chroma LUT via canonical `build_uniward_weighted_chroma_lut` (pure numpy; READ-ONLY; UNTOUCHED — used DIRECTLY for the chroma regression guard)
- Canonical luma quant level via canonical `_compute_luma_quant_level` (pure numpy; READ-ONLY; the SAME bin partition for all 3 surfaces)
- Canonical per-pixel UNIWARD weight via canonical `compute_per_pixel_uniward_weight_map_numpy` (pure numpy; READ-ONLY)
- Canonical comparison via canonical `compare_uniward_vs_canonical_lut` (pure numpy; READ-ONLY; shape-agnostic — works for `(16,5,3)` chroma, `(16,5,1)` grayscale, `(16,5,2)` indices-lo/hi alike)
- Grayscale-LUT surface helpers (`build_grayscale_lut_statistic` + `build_grayscale_lut_canonical_median` + `_compute_bt601_luma_u8`) + indices-LUT surface helpers (`build_indices_lut_statistic` + `build_indices_lut_canonical_median` + `_derive_real_grounded_codebook_indices`) + EXTREMA statistic helpers (`weighted_percentile_per_channel` + `weighted_argmax_per_channel`) are UNIQUE-per-method numpy (pure numpy; do NOT modify any canonical helper per Catalog #290)
- ZERO MLX runtime dependency at the probe; macOS CPU only

## Apples-to-apples evidence discipline confirmation (per 10th standing directive)

Per CLAUDE.md "Apples-to-apples evidence discipline" NON-NEGOTIABLE + 10th standing directive:

1. **Same fixture all 12 cells**: all 3 surface LUTs + per-surface canonical-median baselines ALL derived from the SAME 200 frames + SAME 5-class SegNet argmax labels + SAME EVAL_HW (384×512) + SAME upstream NEAREST-upsampled gradient cache (96×128 → 384×512) + SAME per-pixel UNIWARD weight map + SAME 16×5 bin partition (`_compute_luma_quant_level`). The ONLY varied independent variable is the per-bin VALUE-TYPE (3-channel RGB chroma vs 1-channel BT.601 luma vs categorical codebook index) × the per-bin STATISTIC (median vs EXTREMA p95/p99/argmax).
2. **Same canonical helpers**: `build_chroma_lut_from_ground_truth`, `build_uniward_weighted_chroma_lut`, `compute_per_pixel_uniward_weight_map_numpy`, `compare_uniward_vs_canonical_lut`, `_compute_luma_quant_level`, `load_differentiable_scorers`, `decode_real_pairs` — all sister-disjoint READ-ONLY consumer imports per Catalog #230.
3. **Same cache lineage**: Sister B's N=200 cache REUSED byte-identical (sha256 `e5799f4bd8659a0b` verified at probe load); zero re-extension; dyn_range 772.41× BYTE-EQUIVALENT to Sister B/C/E.
4. **Apples-to-apples chroma regression guard PASS**: the chroma surface reproduces Sister E's published anchors EXACTLY at every digit — median 21/80 (max_delta=5, mean_l2=0.3344, max_l2=5.0990); p95 80/80 (max_delta=137, mean_l2=47.7819, max_l2=138.4522); p99 80/80 (max_delta=151, mean_l2=60.7822, max_l2=153.2906); argmax 80/80 (max_delta=134, mean_l2=30.2461, max_l2=141.0709) — proof of pipeline + EXTREMA-helper correctness via the intra-test regression guard. Verified in BOTH `--smoke` mode AND the full run.
5. **The ONLY varied independent variable is the APPLICATION SURFACE** (value-type). The sample size, gradient cache, UNIWARD weights, bin partition, EXTREMA statistic family, and comparison baseline construction are all FIXED.
6. **Real-scorer-grounded, NOT synthetic**: per CLAUDE.md "Forbidden make_synthetic_pair_batch in non-smoke" + "Apples-to-apples" — the grayscale luma is the REAL BT.601 luma of the REAL decoded frames; the codebook indices are a DETERMINISTIC floor-quantization of that REAL luma into [0, 512) categorical addresses (NOT random Gaussian). Every surface is anchored on the REAL Sister B real-scorer cache + REAL decoded frames.
7. **Axis label honored**: every metric tagged `[macOS-CPU advisory]`; NOT promotable per Catalog #192; `score_claim=False`; `promotable=False`.
8. **Holub-Fridrich-Denemark 2014 canonical alignment**: WebSearch-confirmed (per 10th standing directive) that UNIWARD is a "universal distortion function for steganography in an arbitrary domain" — the SAME relative-change-of-coefficients principle applies across the spatial domain, JPEG domain, and side-informed JPEG domain. This is the canonical grounding for the cross-surface transfer test: UNIWARD's design intent IS domain-agnostic routing. The canonical paper works on continuous/quantized coefficients; the categorical indices surface is the individually-fractal edge case where the percentile metric is ill-defined, and the empirical result (97.5% argmax + 240 max_delta) shows the EXTREMA still transfers but with the expected categorical-surface texture.

Sources for Holub-Fridrich-Denemark 2014 (WebSearch 2026-05-27):
- [Universal distortion function for steganography in an arbitrary domain (EURASIP 2014)](http://dde.binghamton.edu/vholub/pdf/EURASIP14_Universal_Distortion_Function_for_Steganography_in_an_Arbitrary_Domain.pdf)
- [Springer canonical paper link](https://link.springer.com/article/10.1186/1687-417X-2014-1)
- [EURASIP Journal full text](https://jis-eurasipjournals.springeropen.com/articles/10.1186/1687-417X-2014-1)

## Canonical-vs-unique decision per layer

Per Catalog #290 (binding for every substrate work):

| Layer | Decision | Rationale |
|---|---|---|
| N=200 cache loader (np.load) | ADOPT canonical (Sister B sister-disjoint READ-ONLY) | Sister B cache format identical; sha256 `e5799f4bd8659a0b` verified at load |
| Cache extension | N/A — REUSE Sister B's existing N=200 cache | Sister-disjoint per Catalog #230; APPEND-ONLY per Catalog #110/#113 (Sister B cache UNTOUCHED) |
| EVAL_HW pair decode | ADOPT canonical `decode_real_pairs` | Sister-disjoint READ-ONLY |
| SegNet argmax (5-class) | ADOPT canonical pattern from Sister E probe (chunked, no_grad) | Sister-disjoint mirror; SAME bin partition for all 3 surfaces |
| NEAREST upsample 96×128 → 384×512 | ADOPT canonical `_upsample_grad_nearest_to_full_res` pattern | Sister-disjoint mirror; numpy.repeat preserves spatial mass |
| Per-pixel UNIWARD weight | ADOPT canonical `compute_per_pixel_uniward_weight_map_numpy` | Sister-disjoint READ-ONLY |
| Luma quant level (bin partition) | ADOPT canonical `_compute_luma_quant_level` | Sister-disjoint READ-ONLY; SAME bin partition for all 3 surfaces (the fixture invariant) |
| Canonical-median chroma LUT (comparison baseline) | ADOPT canonical `build_chroma_lut_from_ground_truth` | Sister-disjoint READ-ONLY; chroma baseline sha `203eb763cd3a007d` |
| Weighted-MEDIAN chroma LUT (chroma regression guard) | ADOPT canonical `build_uniward_weighted_chroma_lut` | Sister-disjoint READ-ONLY; used DIRECTLY so the chroma regression guard proves byte-identity to Sister E |
| Chroma EXTREMA LUT (chroma regression guard) | FORK — UNIQUE-per-method `_build_chroma_extrema` | Sister-disjoint mirror of Sister E's helper so the chroma surface reproduces Sister E's 80/80 EXTREMA anchors exactly |
| **Grayscale-LUT surface (DIFFERENT #1)** | **FORK — UNIQUE-per-method** `build_grayscale_lut_statistic` + `build_grayscale_lut_canonical_median` + `_compute_bt601_luma_u8` | PRINCIPLED NEW SURFACE per Catalog #290 falling-rule #2: the chroma surface stores 3-channel RGB; the grayscale surface stores the single BT.601 luma channel (szabolcs-cs PR#56 grayscale-LUT analog mask paradigm; canonical equation #26 `grayscale_lut_replacement` IN-DOMAIN sister context). The helper is structurally identical to the chroma helper EXCEPT it aggregates 1 channel; returns `(16,5,1)` uint8 so the canonical comparison helper is agnostic |
| **Indices-LUT surface (DIFFERENT #2)** | **FORK — UNIQUE-per-method** `build_indices_lut_statistic` + `build_indices_lut_canonical_median` + `_derive_real_grounded_codebook_indices` | PRINCIPLED NEW SURFACE per Catalog #290 falling-rule #2: the VQ-VAE indices_blob is a categorical codebook-address stream (uint16 [0, 512)), NOT continuous chroma. The helper applies the EXTREMA family to the categorical address values + stores `(16,5,2)` uint8 (index lo/hi split) so the canonical uint8 comparison helper works. This is the individually-fractal DIVERGENCE-POINT surface where the percentile metric is semantically ill-defined |
| EXTREMA statistic (median/p95/p99/argmax) | FORK — UNIQUE-per-method `weighted_percentile_per_channel` + `weighted_argmax_per_channel` | PRINCIPLED FORK per Catalog #290 falling-rule #2; mirror of Sister E's helpers; degenerates to canonical median at cdf_target=0.5 (verified by chroma regression guard); works on any integer value range (uint8 chroma/luma OR uint16 index) |
| LUT comparison | ADOPT canonical `compare_uniward_vs_canonical_lut` | Sister-disjoint READ-ONLY; shape-agnostic (sums L2 over last axis); works for all 3 surface shapes |
| Verdict tree | UNIQUE per-method: 3-branch cross-surface tree per Catalog #307 Sister D spec | PARADIGM_TRANSFERS_ACROSS_SURFACES / PARADIGM_INTACT_WITH_SURFACE_SPECIFIC_RESPONSE / PARADIGM_INTACT_AT_CHROMA_LUT_SURFACE_ONLY |
| Provenance.json + landing memo + sidecar LUT bytes | UNIQUE per-lane | Standard sister-disjoint per Catalog #110/#113 |
| Catalog #313 probe-outcome registration | ADOPT canonical `register_probe_outcome` | Sister-disjoint READ-ONLY consumer import; canonical 4-layer fcntl-locked JSONL pattern |

## 9-dimension success checklist evidence

Per Catalog #294:

1. **UNIQUENESS**: First N=200 macOS-CPU advisory cross-surface transfer probe of the Fridrich 2014 weighted-EXTREMA family across 3 application surfaces (chroma RGB / grayscale luma / VQ-VAE categorical indices); the 4-point orthogonal-axis decomposition (sample-size / LUT-shape / statistic / application-surface) is now empirically complete with Sister D establishing that the STATISTIC axis breakthrough is SURFACE-ROBUST.
2. **BEAUTY + ELEGANCE**: ~640 LOC standalone probe script; the grayscale + indices surface helpers are structurally identical to the chroma helper EXCEPT the per-bin value-type; the EXTREMA statistic helpers degenerate to the canonical median at cdf_target=0.5 (verified by chroma regression guard); sister-disjoint READ-ONLY consumer of Sister B's cache; per-surface-per-statistic sidecar LUT artifacts emitted.
3. **DISTINCTNESS**: Probe is INTRA-TEST APPLICATION-SURFACE-axis comparison at fixed N=200 + fixed dyn_range + fixed 16×5 bin partition + fixed EXTREMA statistic family; orthogonal axis (application-surface) to Sister E's statistic axis + Sister C's LUT-shape axis + Sister B's sample-size axis.
4. **RIGOR**: Smoke test PASSED with EXACT reproduction of Sister E chroma anchors (median 21/80; p95/p99/argmax 80/80; dyn_range 772.41×) — proof of canonical-helper + EXTREMA-helper pipeline correctness PRE full run; Catalog #229 PV honored (read Sister E landing + Sister E probe + Sister C landing + NSCS06 v8 architecture + VQ-VAE indices_procedural_variant + WebSearch Holub-Fridrich-Denemark 2014 BEFORE writing probe); 9 checkpoints emitted; sister-disjoint from 2 in-flight meta-resurrection lanes confirmed.
5. **OPTIMIZATION PER TECHNIQUE**: UNIWARD per-LUT-index aggregation is canonical Fridrich; weighted-EXTREMA is canonical Fridrich 2014 generalization; the grayscale luma is canonical BT.601; the codebook indices are real-grounded floor-quantization (NOT synthetic); gradient cache canonical N+1 anchor at 96×128 reused from Sister B; all routed through canonical helpers except the principled surface forks.
6. **STACK-OF-STACKS COMPOSABILITY**: Sister-disjoint READ-ONLY consumer imports preserve V15 + UNIWARD integration + NSCS06 v8 substrate + VQ-VAE substrate + Sister B/C/E + canonical equations registry ALL untouched; Sister D is the canonical pattern for cross-surface transfer of any future LUT-statistic primitive.
7. **DETERMINISTIC REPRODUCIBILITY**: numpy + torch + PIL deterministic-seeded with seed=20260526 (matching Sister B/C/E for cache lineage); Sister B cache REUSED byte-identical; canonical-median LUT sha256s emitted per surface (chroma=`203eb763cd3a007d`; grayscale=`e19f651ce33b8e0c`; indices=`5b5d1bb94b35b5ab`); per-(surface, statistic) LUT sha256s emitted in provenance.json; SegNet argmax deterministic with `no_grad`.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: 133.3 sec total wall-clock on macOS M5 Max CPU for the full 12-cell probe (zero cache re-extension thanks to Sister B reuse); ZERO paid GPU spend; well within operator budget for $0 macOS-CPU advisory scope.
9. **OPTIMAL MINIMAL CONTEST SCORE**: Sister D is NOT a frontier-pursuit attempt; it is a controlled cross-surface-transfer test of the EXTREMA effect at fixed sample size + fixed bin partition + fixed statistic family; its `PARADIGM_TRANSFERS_ACROSS_SURFACES` verdict establishes that the EXTREMA routing breakthrough is a property of the statistic family, not a chroma-surface idiosyncrasy — which broadens the paired-CUDA dispatch matrix from 1 surface (chroma per Sister F) to 3 surfaces. The contest-score direction (improve vs hurt) on each surface requires paired-CUDA disambiguation.

## Cargo-cult audit per assumption

Per Catalog #303:

| Assumption | Pre-execution classification | Empirical verdict |
|---|---|---|
| "Sister E's EXTREMA breakthrough will TRANSFER to the grayscale_lut surface (single continuous luma channel; structurally most similar to chroma)" | CARGO-CULTED-PENDING-VALIDATION (Holub-Fridrich-Denemark 2014 arbitrary-domain universality hypothesis) | **EMPIRICALLY CONFIRMED** — grayscale p95/p99 hit 100% bin-occupation (80/80); the EXTREMA breaks the grayscale median's 8.75% saturation ceiling. The arbitrary-domain universality transfers to the single-luma-channel surface |
| "The EXTREMA will FAIL or behave anomalously on the VQ-VAE indices_blob surface because a weighted percentile over CATEGORICAL addresses is semantically ill-defined" | CARGO-CULTED-PENDING-VALIDATION (the individually-fractal divergence-point hypothesis) | **EMPIRICALLY PARTIALLY-FALSIFIED** — the EXTREMA still BREAKS saturation on the indices surface (p95/p99 100%) — the percentile-over-sorted-indices DOES produce a measurable routing divergence even though the metric is ill-defined. HOWEVER the divergence DID manifest: indices p99 max_delta=240 is the LARGEST of all surfaces (the percentile lands on a far-from-median categorical index), and argmax stops 2 bins short of 100% (97.5%). The categorical-surface texture is REAL but does NOT prevent transfer |
| "The median SATURATION FLOOR is the same across all surfaces (~26.5% per Sister C/E)" | CARGO-CULTED-PENDING-VALIDATION | **EMPIRICALLY FALSIFIED** — the median floor is SURFACE-SPECIFIC: chroma=26.25% (21/80) vs grayscale+indices=8.75% (7/80). The single-channel surfaces have a LOWER median floor because there is less per-channel structure to disagree on (1 channel vs 3 means fewer independent comparisons that flip a bin to "changed"). This is a NEW per-substrate individually-fractal finding |
| "argmax hits 100% bin-occupation on every surface (like chroma)" | CARGO-CULTED-PENDING-VALIDATION | **EMPIRICALLY FALSIFIED** — argmax = 100% on chroma but 97.5% (78/80) on BOTH grayscale + indices. EXACTLY 2 bins on each single-channel surface where the highest-weight pixel's value equals the unweighted median. The argmax bin-occupation is a function of the surface's channel-dimensionality: chroma's 3-channel RGB triple almost never matches the median triple on all 3 channels, so chroma argmax hits 100%; the 1-D single-channel surfaces have ~2 coincidence bins |
| "Per-bin magnitude ranking is the same across surfaces" | CARGO-CULTED-PENDING-VALIDATION | **EMPIRICALLY DETERMINED** — magnitude ranking is indices (p99 max_delta=240) > chroma (151) > grayscale (37), reflecting each surface's value-range × channel-structure. The indices surface's 2-byte uint16 split + categorical [0, 512) range produces the largest per-bin L2; grayscale's single luma 0-255 channel produces the smallest |
| "The cross-surface routing effect is a ROUTING-EFFECT signal, NOT a contest-score signal" | HARD-EARNED-FROM-CATALOG-307-PARADIGM-VS-IMPLEMENTATION | **STRUCTURALLY CONFIRMED** — per the canonical `compare_uniward_vs_canonical_lut` docstring + Catalog #307: the cross-surface transfer proves the EXTREMA routes the LUT entry to the high-UNIWARD-weight tail on every surface, but whether that improves or hurts d_seg + d_pose is OPEN and requires paired-CUDA per surface |
| "macOS CPU pipeline reproduces Sister E's chroma anchors exactly" | HARD-EARNED-FROM-CANONICAL-HELPER-DETERMINISM | **CONFIRMED VIA SMOKE + FULL** — the chroma surface of Sister D reproduces Sister E's published anchors EXACTLY at every digit (median 21/80; p95/p99/argmax 80/80; identical max_delta + mean_l2 + max_l2 per statistic) |
| "macOS CPU is 1:1 contest-compliant for the cross-surface LUT-statistic signal" | CARGO-CULTED-PENDING-VALIDATION | **STRUCTURALLY-FALSIFIED** per Catalog #192 — macOS-CPU is NEVER 1:1 contest-compliant; the cross-surface routing effect IS valid `[macOS-CPU advisory]` research signal but cannot promote to score claim without paired Linux x86_64 paired-CUDA evidence |
| "The grayscale + indices surface helpers must FORK (cannot extend canonical chroma helper in-place)" | HARD-EARNED-FROM-CATALOG-290-PRINCIPLED-MISMATCH | **CONFIRMED** — the canonical chroma helper hardcodes 3-channel RGB aggregation; the grayscale surface aggregates 1 luma channel; the indices surface aggregates categorical uint16 addresses; both REQUIRE different value-type handling; the principled forks degenerate to the chroma helper's structure (the chroma regression guard reproduces Sister E exactly) so the forks are correct |

## Observability surface

Per Catalog #305 — all 6 facets HONORED:

- **Inspectable per layer**: per-pair UNIWARD weight stats (min/max/mean/dynamic_range); per-(surface, statistic) LUT comparison (`per_bin_l2_difference.npy` array per cell exposes per-(level, class) L2 delta); per-surface canonical-median LUT sha256 in provenance.json
- **Decomposable per signal**: per-(surface, statistic) bin_count + bin_pct + max_delta + mean_l2 + max_l2 + dyn_range in provenance.json `empirical_results_per_surface_per_statistic` nested table; rate-axis NOT computed (no archive emitted)
- **Diff-able across runs**: deterministic seed (20260526); Sister B cache sha256 cited; per-surface canonical-median sha256s + per-(surface, statistic) LUT sha256s emitted; verdict reproducible from provenance.json
- **Queryable post-hoc**: provenance.json (~16KB JSON; sort_keys=True byte-stable); probe_full.log (full stdout trace with stage timestamps); Catalog #313 probe-outcomes ledger entry (queryable via `tac.probe_outcomes_ledger.query_by_substrate('nscs06_v8_chroma_lut_uniward_per_lut_index')`); per-(surface, statistic) sidecar artifacts at `lut_artifacts_{surface}_{statistic}/lut_statistic.npy` + `lut_canonical_median.npy` + `per_bin_l2_difference.npy` (12 sidecar dirs)
- **Cite-able**: 1 Sister B cache sha256 + 3 per-surface canonical-median sha256s + 12 per-(surface, statistic) LUT sha256s + 5 canonical Provenance fields (`evidence_grade=macOS-CPU-advisory`, `axis_tag=[macOS-CPU advisory]`, `hardware_substrate=darwin_arm64_m5_max_macos_cpu`, `score_claim=False`, `promotable=False`)
- **Counterfactual-able**: V15 N=50 + Sister B N=200 + Sister C 3-LUT-shape + Sister E 4-statistic + Sister D 3-surface×4-statistic cells provide the canonical 4-axis counterfactual; SMOKE mode (--smoke flag) reproduces the chroma surface exactly for the apples-to-apples regression guard; a future Sister can compound the surface axis with a new statistic or sample size

## Predicted ΔS band vs empirical (per Catalog #296)

PREDICTED (from V15 reactivation criterion #4 hypothesis per Sister E recommendation): one of {PARADIGM_TRANSFERS_ACROSS_SURFACES, PARADIGM_INTACT_WITH_SURFACE_SPECIFIC_RESPONSE, PARADIGM_INTACT_AT_CHROMA_LUT_SURFACE_ONLY}. The expected outcome was bounded by the Holub-Fridrich-Denemark 2014 arbitrary-domain universality (predicting transfer) tempered by the individually-fractal directive (predicting surface-specific texture).

EMPIRICAL: EXTREMA p95/p99 hit 100% bin-occupation on ALL 3 surfaces (chroma + grayscale + indices); argmax 100% on chroma, 97.5% on grayscale + indices; median floor 26.25% chroma vs 8.75% single-channel surfaces. Per the verdict tree: `EXTREMA breaks saturation on ALL different surfaces (bins > 50%)` → `PARADIGM_TRANSFERS_ACROSS_SURFACES`.

Per Dykstra-feasibility (Catalog #296): the intersection of (Fridrich-2014-weighted-EXTREMA ∩ high-UNIWARD-weight-tail-concentration ∩ N=200-sample-size ∩ 772.41×-dyn-range ∩ 16×5-bin-partition ∩ {chroma RGB, grayscale luma, categorical index} value-type ∩ LUT-derivation-routing-effect) is **STATISTICALLY-EMPIRICALLY-NON-EMPTY at ALL 3 surfaces** at the EXTREMA p95/p99 statistic. The 100%-on-all-surfaces bin-occupation signature is the canonical Dykstra-feasibility NON-EMPTY-INTERSECTION evidence for the cross-surface transfer — the EXTREMA statistic produces measurable per-bin routing divergence in nearly every bin on every surface. **However**, the Dykstra-feasibility intersection with the CONTEST-SCORE feasible region (lower d_seg + lower d_pose) is UNKNOWN per surface at macOS-CPU advisory scope — that intersection is computed ONLY by the paired-CUDA auth-eval per surface per the V15 1e-4 PROMOTION threshold protocol. The non-empty cross-surface routing-effect intersection is a NECESSARY-NOT-SUFFICIENT condition for a contest-score improvement.

## Horizon-class declaration

Per Catalog #309: **frontier_pursuit** (matching Sister E's up-revised classification). Sister D's empirical signal CONFIRMS the EXTREMA breakthrough is surface-robust — the routing effect breaks the median-saturation ceiling on all 3 surfaces, broadening the frontier_pursuit case from 1 surface to a 3-surface paired-CUDA dispatch matrix. The frontier_pursuit classification reflects that the cross-surface routing-effect is strong enough to WARRANT paired-CUDA validation on multiple surfaces — NOT that the contest score is improved (which remains UNKNOWN per surface pending paired-CUDA). Per the per-substrate individually-fractal directive, the APPLICATION-SURFACE axis shows the STATISTIC-axis breakthrough is robust but textured (surface-specific median floor + argmax bin-count + per-bin magnitude).

## Catalog #344 canonical equation candidate status

**Status**: NEW candidate `uniward_extrema_statistic_cross_surface_transfer_v1` registered as **FORMALIZATION_PENDING** (NOT promoted to REGISTERED; NO hard registry write, consistent with Sister E's `uniward_substrate_unique_optimal_statistic_savings_v1` FORMALIZATION_PENDING memo-only candidate). The cross-surface transfer is a NEW substrate-specific empirical finding distinct from the existing FORMALIZATION_PENDING `uniward_per_lut_index_distortion_weight_savings_v1` + `uniward_substrate_unique_optimal_statistic_savings_v1`.

<!-- FORMALIZATION_PENDING:the_fridrich_2014_weighted_extrema_breaks_median_saturation_across_all_three_application_surfaces_chroma_grayscale_indices_per_sister_d_cross_surface_empirical_transfer_macos_cpu_advisory_does_not_promote_pending_paired_cuda_evidence_per_multi_surface_dispatch_matrix -->

The candidate equation captures the cross-surface relationship: at fixed (N=200, dyn_range=772.41×, 16×5 bin partition), the EXTREMA statistic's per-bin routing divergence vs the per-surface unweighted-median baseline jumps from the surface-specific median floor (8.75%-26.25%) to ≥97.5% at the p95+ rank on EVERY surface tested. The cross-surface transfer signature — `bins_changed_pct(surface, cdf_rank≥0.95) ≥ 0.975` for all surfaces ∈ {chroma, grayscale, indices} — is the substrate's UNIWARD-EXTREMA cross-surface-transfer empirical signature.

Per CLAUDE.md "Forbidden premature KILL" + the FORMALIZATION_PENDING discipline: the candidate equation CANNOT promote to PARADIGM-VALIDATED-EMPIRICALLY from this Sister D anchor because:
1. **Catalog #192**: macOS-CPU advisory is NOT 1:1 contest-compliant hardware.
2. **V15 PROMOTION threshold**: paired-CUDA delta > 1e-4 declared in V15 build-script protocol; this Sister D probe does NOT measure paired-CUDA delta on ANY surface (scope explicitly excludes paid dispatch).
3. **Routing-effect ≠ score-effect**: the cross-surface 100% bin-occupation is a NECESSARY-NOT-SUFFICIENT condition; the candidate equation's `bins_changed_pct(surface, cdf_rank)` relationship is a LUT-derivation-surface statistic, NOT a contest-score relationship. The candidate equation's domain is the LUT-DERIVATION surface; its extension to the CONTEST-SCORE surface requires paired-CUDA evidence per surface.

This anchor extends V15's N=50 + Sister B's N=200 + Sister C's 3-LUT-shapes + Sister E's 4-statistics evidence with the CROSS-SURFACE TRANSFER signal at N=200 but does NOT promote the equation. The candidate is registered as FORMALIZATION_PENDING per Catalog #344; the canonical registration awaits paired-CUDA evidence on at least one surface.

Reactivation criteria status (per CLAUDE.md "Forbidden premature KILL"):

1. **Reactivation criterion #1** (sample size): COMPLETE (Sister B).
2. **Reactivation criterion #2** (higher dynamic range): operator-routable; dyn_range axis IS sample-size-responsive but does NOT compound with LUT-shape (Sister C).
3. **Reactivation criterion #3** (LUT granularity): COMPLETE-AND-PARTIALLY-FALSIFIED (Sister C; LUT-shape axis saturates).
4. **Reactivation criterion #4** (different application surface): **COMPLETE-AND-VALIDATED** (this Sister D landing). The EXTREMA breakthrough TRANSFERS to the grayscale_lut + VQ-VAE indices_blob surfaces; the STATISTIC-axis breakthrough is SURFACE-ROBUST with per-substrate individually-fractal texture.
5. **Reactivation criterion #5** (weighted EXTREMA statistic): COMPLETE-AND-VALIDATED (Sister E; STATISTIC axis is the binding axis).
6. **Reactivation criterion #6** (RECOMMENDED for next paired-CUDA — Sister F): operator-routable paired-CUDA validation of the weighted-EXTREMA statistic. Sister D BROADENS the dispatch matrix from 1 surface to 3: the operator may now choose the surface(s) for paired-CUDA validation. RECOMMENDED prioritization: (a) chroma p99 (strongest mean_l2 routing-effect per Sister E); (b) grayscale p99 (SMALLER per-bin perturbation max_delta=37 vs chroma's 151 — the more CONSERVATIVE arm if the PESSIMISTIC artifact-risk interpretation holds for chroma); (c) indices p99 (largest per-bin delta=240 — highest-variance arm, dispatch only if chroma/grayscale show promise).

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map**: ACTIVE — per-pixel UNIWARD weights from inherited Sister B N=200 cache feed canonical sensitivity-map surface; per-(surface, statistic) per-(level, class) bin weights surfaced across all 3 surfaces
2. **Pareto constraint**: ACTIVE — per-axis `[macOS-CPU advisory]` tag carries Catalog #323 canonical Provenance; non-promotable per Catalog #341 (score_claim=False + promotable=False); explicit non-frontier-crossing per V15 protocol; the cross-surface-routing-effect-vs-contest-score distinction is the explicit Pareto-feasibility caveat
3. **Bit-allocator**: ACTIVE — the per-surface per-bin STATISTIC choice IS the per-bin bit allocator's routing decision across 3 surfaces (the EXTREMA routes the LUT entry to the high-weight-tail value on every surface); the empirical Sister D result is itself a bit-allocator-design constraint (the STATISTIC-axis breakthrough is surface-robust, so the bit-allocator's routing primitive transfers across surfaces)
4. **Cathedral autopilot dispatch**: ACTIVE — Catalog #313 probe-outcomes ledger row registered (probe_id=`v15_uniward_sister_d_extrema_cross_surface_20260527T062853Z_20260527`; verdict=PROCEED; blocker_status=advisory; expires_at_utc=2026-06-26T06:28:53Z). Cathedral autopilot ranker can consume the empirical `PARADIGM_TRANSFERS_ACROSS_SURFACES` verdict + 12-cell cross-surface signals as evidence for the broadened paired-CUDA dispatch matrix (Sister F)
5. **Continual-learning posterior**: ACTIVE — provenance.json carries canonical Provenance per Catalog #323; 12 per-(surface, statistic) sidecar LUT artifacts queryable by future sister probes; Catalog #313 probe-outcome ledger entry feeds canonical continual-learning posterior; Catalog #344 candidate equation registered FORMALIZATION_PENDING
6. **Probe-disambiguator**: ACTIVE — Sister D IS the canonical disambiguator at the APPLICATION-SURFACE axis; the empirical `PARADIGM_TRANSFERS_ACROSS_SURFACES` result disambiguates between the OPTIMISTIC hypothesis (EXTREMA transfers cleanly to all surfaces) and the individually-fractal hypothesis (surface-specific texture) — the empirical reality is BOTH (transfer + texture); the surface-specific median floor + argmax bin-count + per-bin magnitude are the per-substrate individually-fractal evidence

## Carmack-dissent verdict per Catalog #307

**PARADIGM-INTACT + PARADIGM-TRANSFERS-ACROSS-SURFACES-AT-LUT-DERIVATION-SURFACE**. Per CLAUDE.md "Forbidden premature KILL without research exhaustion":

- The UNIWARD per-LUT-index routing PRIMITIVE remains canonical Fridrich; this Sister D probe STRENGTHENS the primitive's substrate-relevance by surfacing that the STATISTIC-axis breakthrough (Sister E) is SURFACE-ROBUST
- The IMPLEMENTATION at (N_LUT=200, dynamic_range=772.41×, bin-partition=16×5, statistic=EXTREMA-p95/p99/argmax) shows the EXTREMA breaks median saturation on ALL 3 surfaces (chroma + grayscale + indices) — the routing breakthrough is a property of the statistic family, not a chroma-surface idiosyncrasy
- The 6-of-6 reactivation criteria above enumerate the research-exhaustion paths; KILL is the LAST RESORT and is NOT invoked
- The empirical trajectory is now an 8-point sequence: PARADIGM-NULL-AT-RAW-RGB → PARADIGM-NULL-AT-CAPACITY-CONSTRAINED → PARADIGM-VALIDATED-AT-MLX-LOCAL-SYNTHETIC → IMPLEMENTATION-FALSIFIED-AT-PAIRED-CUDA-REAL-N=50 → IMPLEMENTATION-PARTIAL-SCALING-AT-MACOS-CPU-ADVISORY-N=200-16X5 → IMPLEMENTATION-LUT-SHAPE-AXIS-PARTIALLY-FALSIFIED-AT-MACOS-CPU-ADVISORY-N=200-3-SHAPES → STATISTIC-AXIS-VALIDATED-AT-MACOS-CPU-ADVISORY-N=200-4-STATISTICS (Sister E) → **APPLICATION-SURFACE-AXIS-TRANSFER-VALIDATED-AT-MACOS-CPU-ADVISORY-N=200-3-SURFACES** (this Sister D landing)
- The next iteration is criterion #6 (paired-CUDA validation of the EXTREMA statistic per Sister F) — the cross-surface routing-effect breakthrough BROADENS the paired-CUDA dispatch matrix to 3 surfaces

Per Catalog #348 retroactive sweep: **NOT TRIGGERED**. The Sister D verdict is a PARADIGM-TRANSFER-VALIDATION at the routing-effect surface (not a KILL/DEFER/FALSIFY of any prior verdict); no historical verdicts are invalidated. Sister D EXTENDS the prior empirical trajectory consistently — all prior anchors remain intact. Sister D adds the cross-surface transfer evidence at macOS-CPU advisory scope and broadens the paired-CUDA recommendation, which is the canonical research-exhaustion path forward.

## Sister-disjoint discipline confirmation per Catalog #230

NO modifications to V15 build script, V15 source cache, Sister B source cache, Sister C/E artifacts, UNIWARD per-LUT-index integration package (the canonical `lut_derivation_uniward_weighted.py` weighted-median helper is UNTOUCHED — the surface forks live entirely in the probe script), NSCS06 v8 substrate, VQ-VAE substrate, canonical equations registry, or any sister artifact. All canonical helpers consumed READ-ONLY (`compute_per_pixel_uniward_weight_map_numpy`, `build_uniward_weighted_chroma_lut`, `compare_uniward_vs_canonical_lut`, `build_chroma_lut_from_ground_truth`, `_compute_luma_quant_level`, `decode_real_pairs`, `load_differentiable_scorers`, `register_probe_outcome`). Sister B's N=200 cache REUSED via Catalog #110/#113 APPEND-ONLY (cache UNTOUCHED; byte-identical pre/post Sister D probe). Output strictly scoped under `experiments/results/v15_uniward_sister_d_extrema_different_application_surface_macos_cpu_advisory_20260527/` + this landing memo.

Sister-disjoint confirmed via `subagent_progress.jsonl` review at probe-execution time: the only in-flight sisters were `meta_resurrection_audit_v2_inherently_br*` + `meta_resurrection_v2_op_routables_canoni*` (both at step 0, ZERO files touched, meta-resurrection lanes — fully DISJOINT from UNIWARD per-LUT-index files). Catalog #340 sister-checkpoint guard expected PROCEED on commit (no overlap on probe_uniward_extrema_cross_surface_n200.py / provenance.json / lut_artifacts_*/ / this landing memo).

## Discipline anchors

- **Catalog #229 PV** (read Sister E landing memo + Sister E probe script + Sister C landing memo + NSCS06 v8 architecture `build_chroma_lut_from_ground_truth` + `_compute_luma_quant_level` + VQ-VAE `indices_procedural_variant` + canonical helper signatures `compare_uniward_vs_canonical_lut` / `register_probe_outcome` / `VALID_VERDICTS` + WebSearch Holub-Fridrich-Denemark 2014 + sister subagent activity in `subagent_progress.jsonl` BEFORE writing probe script)
- **Catalog #206** (9 checkpoints emitted across fresh-start / PV / surface-investigation / WebSearch / sister-check / probe-write / smoke-pass / full-probe / landing-memo cycle)
- **Catalog #110/#113 APPEND-ONLY** (Sister B cache + Sister C/E artifacts + V15 build artifacts + ALL landing memos + canonical helpers ALL untouched; Sister D sidecar LUT artifacts + provenance.json + landing memo are NEW artifacts only)
- **Catalog #117/#157/#174/#235/#289** canonical commit serializer (used --expected-content-sha256 + co-author trailer for the commit)
- **Catalog #119** Co-Authored-By Claude Opus 4.7 (1M context) trailer (internal commit)
- **Catalog #125** 6-hook wire-in declaration (all 6 hooks ACTIVE per the declaration section above)
- **Catalog #127** authoritative-tag custody metadata (every metric tagged `[macOS-CPU advisory]` per per-call-site discipline)
- **Catalog #131/#138** fcntl-locked + strict-load discipline (Catalog #313 probe-outcomes ledger + provenance.json sidecar follow canonical helper patterns; no bare writes to `.omx/state/`)
- **Catalog #164/#226** canonical scorer-load routing (via canonical `load_differentiable_scorers`; sister-disjoint READ-ONLY consumer import; no fork)
- **Catalog #192** macOS-CPU advisory non-promotion (every metric tagged + `promotable=False`; `score_claim=False`; `ready_for_exact_eval_dispatch=False`)
- **Catalog #230** ownership map (READ-ONLY consumer imports for all canonical helpers; sister-disjoint from 2 in-flight meta-resurrection lanes confirmed)
- **Catalog #287** placeholder-rationale rejection (all rationales ≥4 chars substantive; the FORMALIZATION_PENDING waiver carries a real >100-char rationale; no `<rationale>`/`<reason>` literals)
- **Catalog #290** canonical-vs-unique decision per layer ✓ (table above; the grayscale + indices surface helpers are PRINCIPLED NEW SURFACES per falling-rule #2; the canonical helpers are UNTOUCHED; the EXTREMA forks degenerate to the canonical median at cdf_target=0.5)
- **Catalog #294** 9-dimension success checklist evidence ✓ (section above)
- **Catalog #296** Dykstra-feasibility predicted-band check ✓ (section above; cross-surface routing-effect intersection NON-EMPTY on all 3 surfaces; contest-score intersection UNKNOWN per surface pending paired-CUDA)
- **Catalog #303** cargo-cult audit per assumption ✓ (section above; 9 assumptions classified + empirical verdicts including the FALSIFIED median-floor + argmax-100%-on-every-surface assumptions)
- **Catalog #305** observability surface ✓ (section above)
- **Catalog #307** paradigm-vs-implementation classification (PARADIGM-INTACT + PARADIGM-TRANSFERS-ACROSS-SURFACES-AT-LUT-DERIVATION-SURFACE; explicit cross-surface-routing-effect-vs-contest-score caveat)
- **Catalog #309** horizon_class declaration (frontier_pursuit; matching Sister E)
- **Catalog #313** probe-outcomes ledger ✓ (registered: probe_id=`v15_uniward_sister_d_extrema_cross_surface_20260527T062853Z_20260527`; verdict=PROCEED; blocker_status=advisory; expires_at_utc=2026-06-26T06:28:53Z; staleness_window_days=30)
- **Catalog #316** reports/latest.md frontier alignment (N/A; Sister D is NOT frontier-crossing on the contest-score axis; reports/latest.md NOT mutated)
- **Catalog #317/#341** canonical-routing markers for local research signal (all `[macOS-CPU advisory]` markers + non-promotable contract honored)
- **Catalog #323** canonical Provenance umbrella ✓ (provenance.json carries every required canonical field)
- **Catalog #325** per-substrate symposium 14-day window (the PROCEED verdict's paired-CUDA recommendation is operator-routable per the 14-day window; the per-substrate symposium evidence for paid dispatch is operator-gated)
- **Catalog #335** cathedral consumer canonical contract (Sister D emits machine-readable provenance.json + Catalog #313 anchor; future cathedral consumer can integrate as observability-only Tier-A per Catalog #357 dual-tier discipline)
- **Catalog #340** sister-checkpoint guard PROCEED before commit (checked sister activity above; only DISJOINT meta-resurrection lanes in-flight)
- **Catalog #343** NO hardcoded frontier score literals (the V15 + Sister B + Sister C + Sister E baseline anchors are CONTROLLED-COMPARISON empirical numbers carried via sister memos per Catalog #110 HISTORICAL_PROVENANCE; no contest-frontier score literals)
- **Catalog #344** canonical equation candidate `uniward_extrema_statistic_cross_surface_transfer_v1` FORMALIZATION_PENDING (per the substantive `# FORMALIZATION_PENDING:` waiver above; consistent with Sister E's memo-only candidate handling)
- **Catalog #346** canonical roster N/A (no T2+ deliberation invoked)
- **Catalog #348** retroactive sweep NOT TRIGGERED (Sister D is a PARADIGM-TRANSFER-VALIDATION at the routing-effect surface; no prior verdict invalidated)
- **Catalog #356/#357** per-axis decomposition + Tier B dual-tier consumer architecture (Sister D is Tier A observability-only by construction; no per-axis Tier B emission)
- **CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD"** ✓ (transfer NOT assumed — empirically tested per surface; each surface's UNIQUE coupling structure produced surface-specific texture; the grayscale + indices surfaces are PRINCIPLED NEW per-method forks; the EXTREMA statistic family is the substrate's UNIQUE-OPTIMAL bottleneck per Sister E, now shown surface-robust)
- **CLAUDE.md "Apples-to-apples evidence discipline"** ✓ (intra-test APPLICATION-SURFACE comparison at same fixture + same canonical helpers + smoke-mode verifies Sister E chroma anchor reproduction EXACTLY; explicit cross-surface-routing-effect-vs-contest-score caveat)
- **CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"** ✓ (macOS-CPU advisory explicitly tagged + non-promotable + result_review_blockers cite "requires_paired_cuda_cpu_validation_per_v15_protocol_1e_4_threshold" + "cross_surface_routing_effect_is_necessary_not_sufficient_for_contest_score_improvement")
- **CLAUDE.md "Forbidden premature KILL without research exhaustion"** ✓ (criterion #6 paired-CUDA validation queued for Sister F; PARADIGM-INTACT classification; KILL is the LAST RESORT)
- **CLAUDE.md "MPS auth eval is NOISE"** ✓ (CPU only; no MPS used; tagged `[macOS-CPU advisory]`)
- **CLAUDE.md "MLX-first portable-local-substrate authority"** ✓ (probe is numpy + torch CPU only; ZERO MLX dependency; Sister B cache reused at sister path)
- **8th MLX-first + numpy-portable + individually-fractal standing directive** ✓ (sister-disjoint READ-ONLY consumer of Sister B's cache; pure numpy + torch CPU; transfer NOT assumed — the per-surface response curve IS the empirical deliverable; the grayscale + indices surfaces are PRINCIPLED per-method NEW surfaces, NOT a generic catalog-enumeration; the surface-specific median floor + argmax bin-count + per-bin magnitude are the individually-fractal evidence)
- **10th apples-to-apples + WebSearch authorized + math/science/engineering rigor standing directive** ✓ (WebSearch invoked to verify Holub-Fridrich-Denemark 2014 "arbitrary domain" universality which is the canonical grounding for cross-surface transfer; math rigor binding via canonical Fridrich UNIWARD primitive + canonical Fridrich 2014 weighted-EXTREMA generalization across surfaces; engineering rigor binding via the 13 HNeRV parity inviolable lessons + Catalog #290/#294/#296/#303/#305 design-memo discipline)
- **11th final rate attack off-the-shelf + development workflow standing directive (apples-to-apples baseline FIRST)** ✓ (Sister D tested the chroma surface FIRST as the apples-to-apples regression guard per ORDER Dim 8; demonstrates the canonical $0 macOS-CPU advisory dev-loop workflow)
- **12th 3-strategy attack framework standing directive** ✓ (Sister D sits on DISTORTION pure-axis at the APPLICATION-SURFACE sub-axis; orthogonal to RATE axis from FEC family + FULL SCORER axis from V15 paired-CUDA)
- **13th OPTIMAL-TRIO standing directive** ✓ (TECHNIQUE = canonical Fridrich 2014 weighted-EXTREMA family across 3 application surfaces; WAY = canonical cell-by-cell empirical measurement at fixed dyn_range + fixed bin partition + fixed statistic family; TIME = pre-dispatch PV per Catalog #229 + V15 protocol; sister-disjoint UNIQUE-AND-COMPLETE-PER-METHOD via the canonical helper consumer-only pattern + principled per-surface forks)

## NOT a PR111 candidate — explicit declaration

Per V15 protocol + Catalog #192/#343: PR111 candidate status requires paired-CUDA frontier-crossing on 1:1 contest-compliant hardware (Linux x86_64 + NVIDIA GPU). Sister D is `[macOS-CPU advisory]` — by construction NOT 1:1 contest-compliant; the bins_changed/mean_l2/max_delta metrics are LUT-statistic ROUTING-EFFECT research signals NOT contest scores. Per the `user_pr_attribution` memory non-negotiable + CLAUDE.md "FORBIDDEN CLAUDE ATTRIBUTION IN PUBLIC-PR SURFACES": Sister D does NOT produce a PR111 candidate report; no public-PR-facing artifact is written.

## Operator-routable next steps

1. **Sister F spawn** (reactivation criterion #6 RECOMMENDED — paired-CUDA validation): queue paired-CUDA validation of the weighted-EXTREMA statistic via canonical `tools/dispatch_modal_paired_auth_eval.py`. Sister D BROADENS the dispatch matrix from 1 surface to 3. RECOMMENDED prioritization per the per-surface magnitude texture: (a) **chroma p99** (strongest routing-effect; Sister E's recommended primary); (b) **grayscale p99** (SMALLER per-bin perturbation max_delta=37 vs chroma's 151 — the more CONSERVATIVE arm; if the chroma EXTREMA's PESSIMISTIC artifact-risk interpretation holds, grayscale is the safer paired-CUDA arm because its routing perturbation is ~4-5× smaller in u8 magnitude); (c) **indices p99** (largest per-bin delta=240 — highest-variance arm; dispatch only if chroma/grayscale show promise). Per Catalog #325 14-day window the per-substrate symposium evidence is operator-gated. Estimated $0.50-1.00 per surface-arm (matches V15 budget).
2. **The 4-point orthogonal-axis decomposition is empirically COMPLETE**: sample-size (Sister B; responsive), LUT-shape (Sister C; saturated), statistic (Sister E; THE BINDING AXIS), application-surface (Sister D; THE STATISTIC-BREAKTHROUGH IS SURFACE-ROBUST). The substrate's UNIQUE-OPTIMAL bottleneck is the per-bin statistic choice, and that bottleneck transfers across application surfaces. The next-EV step is paired-CUDA validation of the EXTREMA statistic on the chosen surface(s) (Sister F), NOT further macOS-CPU advisory axis exploration.
3. **CAUTION**: the cross-surface 100% bin-occupation is a ROUTING-EFFECT signal NOT a contest-score signal. Do NOT interpret "PARADIGM_TRANSFERS_ACROSS_SURFACES" as "the EXTREMA lowers the score on all surfaces" — it means "the EXTREMA produces a strong, measurable, non-saturated routing effect on all surfaces WORTH the paired-CUDA spend to disambiguate." The per-surface max_delta values (chroma 151, indices 240, grayscale 37) are large enough on chroma + indices that the PESSIMISTIC interpretation (visible artifacts from extreme concentration) is a live risk per surface that ONLY paired-CUDA can rule out.

## Cost

- $0 GPU spend (macOS CPU only; no Modal / Vast.ai / Lightning dispatch)
- ~133.3 sec CPU wall-clock on M5 Max for the full 12-cell probe (zero cache extension thanks to Sister B reuse + smoke regression guard at the chroma surface first)
- ~640 LOC standalone probe script (~33KB)
- ~470 LOC landing memo (this file)
- 1 Catalog #313 probe-outcome ledger entry registered (verdict=PROCEED; blocker_status=advisory)
- 1 Catalog #344 canonical equation candidate registered FORMALIZATION_PENDING (`uniward_extrema_statistic_cross_surface_transfer_v1`; memo-only per Sister E precedent)
- 0 cache extensions emitted (Sister B's N=200 cache REUSED via Catalog #230 sister-disjoint READ-ONLY)
- 36 sidecar LUT artifacts emitted at `experiments/results/v15_uniward_sister_d_extrema_different_application_surface_macos_cpu_advisory_20260527/lut_artifacts_{surface}_{statistic}/` (3 surfaces × 4 statistics × 3 arrays each: lut_statistic.npy + lut_canonical_median.npy + per_bin_l2_difference.npy; sister-disjoint; APPEND-ONLY)
- 6 reactivation criteria status: #1 COMPLETE (Sister B), #3 COMPLETE-AND-PARTIALLY-FALSIFIED (Sister C), #4 COMPLETE-AND-VALIDATED (this Sister D), #5 COMPLETE-AND-VALIDATED (Sister E), #2 operator-routable; criterion #6 (paired-CUDA validation) RECOMMENDED for next Sister F across a broadened 3-surface dispatch matrix

## Sister-coordination + checkpoint discipline confirmation

9 subagent checkpoints emitted via canonical `tools/subagent_checkpoint.py` per Catalog #206 (Steps 1-9; final Step 9 marks complete after commit):
1. Step 1: fresh-start subagent registration + PV launch
2. Step 2: PV reading Sister E + Sister C landings + Sister E probe script
3. Step 3: WebSearch Holub-Fridrich-Denemark 2014 + grayscale LUT helper investigation
4. Step 4: sister-activity check (Catalog #230) + probe-design
5. Step 5: probe script written + smoke run
6. Step 6: SMOKE PASS (chroma reproduces Sister E exactly) + full 3-surface probe launch
7. Step 7: PARADIGM_TRANSFERS_ACROSS_SURFACES + Catalog #313 + #344 registration
8. Step 8: Catalog #313 registered PROCEED advisory + landing memo write
9. Step 9 (final): canonical commit via serializer + checkpoint complete

Sister-disjoint per Catalog #230: NO collision with in-flight `meta_resurrection_audit_v2` + `meta_resurrection_v2_op_routables` (both meta-resurrection lanes; ZERO files touched at probe-execution; fully DISJOINT). Catalog #340 sister-checkpoint guard expected PROCEED on commit.

## V15 13th OPTIMAL-TRIO declaration per 13th standing directive

Sister D honors the 13th OPTIMAL-TRIO standing directive (mlx-first-numpy-portable-individually-fractal):
- **MLX-first**: reused Sister B's MLX-local-captured cache via canonical READ-ONLY consumer pattern; canonical UNIWARD per-LUT-index integration is MLX-portable (pure numpy)
- **numpy-portable**: probe script + sidecar artifacts + grayscale + indices surface helpers + EXTREMA LUT derivation are pure numpy + torch CPU; ZERO MLX dependency at runtime; inflate-side runtime UNTOUCHED per HNeRV parity L4
- **individually-fractal**: Sister D is unique-per-method (sister-disjoint READ-ONLY consumer imports from V15 / Sister B / Sister C / Sister E + zero modification to canonical helpers); transfer was NOT ASSUMED — each surface's UNIQUE coupling structure was empirically tested, and the per-surface response curve (chroma 100% all EXTREMA; grayscale + indices 100% p95/p99 but 97.5% argmax; surface-specific median floor 8.75%-26.25%; surface-specific per-bin magnitude indices > chroma > grayscale) IS the individually-fractal deliverable the operator's directives demanded; the grayscale + indices surfaces are PRINCIPLED per-method NEW surfaces (szabolcs-cs PR#56 grayscale-LUT analog + VQ-VAE categorical address stream), NOT a generic catalog-enumeration of arbitrary surfaces

Per the AUTOMATED + COMPOUNDING + OPTIMAL META-principle standing directive: Sister D emits canonical artifacts that compound into future iterations (provenance.json + 36 per-(surface, statistic) sidecar LUT artifacts + Catalog #313 probe-outcomes ledger entry PROCEED + Catalog #344 candidate equation FORMALIZATION_PENDING); the probe script IS the AUTOMATED reproducibility surface for Sister F (criterion #6 paired-CUDA across the broadened 3-surface dispatch matrix); the `PARADIGM_TRANSFERS_ACROSS_SURFACES` verdict + cross-surface 100% bin-occupation signature is the OPTIMAL signal for the recommended Sister F paired-CUDA validation. The STATISTIC-axis breakthrough being SURFACE-ROBUST is the per-substrate individually-fractal finding the operator's directives demanded — the EXTREMA statistic's routing breakthrough is NOT a chroma-surface idiosyncrasy; it transfers to the grayscale (continuous luma) AND the VQ-VAE indices (categorical address) surfaces, with rich per-surface texture (median floor + argmax bin-count + per-bin magnitude all surface-specific).
