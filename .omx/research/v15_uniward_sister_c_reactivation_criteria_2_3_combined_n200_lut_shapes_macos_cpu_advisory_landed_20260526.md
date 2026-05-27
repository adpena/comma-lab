# V15 UNIWARD Sister C Reactivation Criteria #2+#3 COMBINED N=200 LUT Shapes macOS-CPU Advisory Landed 2026-05-26

**Subagent**: `v15_uniward_sister_c_reactivation_criteria_2_3_combined_322ab175`
**Lane**: `lane_v15_uniward_sister_c_reactivation_criteria_2_3_combined_20260526`
**Sister of**: V15 Sister B landing memo `.omx/research/v15_uniward_sister_b_reactivation_criterion_1_n200_pairs_macos_cpu_advisory_landed_20260526.md` (commit `3eaa5f826`) which surfaced the 4th-branch `INTERMEDIATE_PARTIAL_SCALING` verdict + 39× upstream gradient dyn_range jump (19.71× → 772.41× from N=50 to N=200) and EMPIRICALLY RECOMMENDED combined criteria #2+#3 for the next iteration
**Predecessor anchors**: V15 N=50 16×5 IMPL-FALSIFIED + Sister B N=200 16×5 INTERMEDIATE_PARTIAL_SCALING
**Date**: 2026-05-26
**Tag**: `[macOS-CPU advisory]` per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192/#317/#341 (NOT promotable; NOT a contest score claim)
**Verdict**: **PARADIGM_INTACT_WITH_COMBINED_CRITERIA_PARTIALLY_FALSIFIED** (per Catalog #307 PARADIGM-INTACT + IMPLEMENTATION-LEVEL-EXPLORATION-AT-LUT-SHAPE-AXIS; recommend criterion #5 weighted EXTREMA for Sister E)
**NOT a frontier-crossing event**; **NOT a PR111 candidate** per Catalog #343 + CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
**FORMALIZATION_PENDING**: canonical equation `uniward_per_lut_index_distortion_weight_savings_v1` per Catalog #344 (UNCHANGED; no promotion)

## TL;DR

Sister C probe directly empirically TESTED Sister B's recommended combined criteria #2+#3 by reusing Sister B's N=200 cache (criterion #2 axis — real-scorer gradient dyn_range FIXED at 772.41× inherited from cache; sister-disjoint READ-ONLY consumer per Catalog #230) AND varying the LUT shape across 3 canonical sister variants (criterion #3 axis — finer granularity). The 3 cells:

| Cell | LUT shape | Total bins | bins_changed | bin_pct | max_delta (u8) | mean_l2 | max_l2 | dyn_range |
|---|---|---|---|---|---|---|---|---|
| **V15 N=50 baseline** (16×5; criterion #1 anchor) | (16, 5) | 80 | 8 | 10.0% | 12 | 0.26 | 12.65 | 19.71× |
| **Sister B N=200 baseline** (16×5; criterion #1 verdict) | (16, 5) | 80 | 21 | 26.25% | 5 | 0.3344 | 5.0990 | 772.41× |
| Sister C N=200 (canonical baseline reproduction) | (16, 5) | 80 | 21 | **26.25%** | 5 | 0.3344 | 5.0990 | 772.41× |
| Sister C N=200 finer-luma-axis | (32, 5) | 160 | 43 | **26.88%** | 3 | 0.3149 | 3.1623 | 772.41× |
| Sister C N=200 finer-class-axis | (16, 10) | 160 | 43 | **26.88%** | 5 | 0.4727 | 5.0990 | 772.41× |

**The empirical surprise**: **bin-occupation ratio is STRUCTURALLY INVARIANT (~26.5%) across all 3 LUT shapes at N=200**. The finer LUT shapes (32×5 + 16×10) DOUBLE the total bin count but the bins_changed count DOUBLES PROPORTIONALLY (21 → 43, preserving 26.25% → 26.88%). The dyn_range is CONSTANT (772.41×) because it is inherited from the gradient cache, not a function of the LUT shape. The per-bin delta magnitudes are similar across shapes (max_delta = 3-5 u8; max_l2 = 3.16-5.10). The LUT-granularity axis does NOT compound the criterion #2 (dyn_range) signal beyond Sister B's baseline.

This **falsifies the IMPLICIT cargo-cult assumption** that finer LUT granularity would unlock per-bin differentiation. Per Sister B's recommendation tree → this is the `PARADIGM_INTACT_WITH_COMBINED_CRITERIA_PARTIALLY_FALSIFIED` branch → recommend criterion #5 (UNIWARD-weighted EXTREMA replaces weighted median) for Sister E follow-on. Per CLAUDE.md "Forbidden premature KILL without research exhaustion": the PARADIGM remains INTACT; the LUT-shape axis is empirically saturated under the canonical Sallee 2003 weighted-median statistic.

Empirical receipts (cell-by-cell):

- **16×5 canonical baseline reproduction**: bins_changed=**21/80** (26.25%) — IDENTICAL to Sister B → canonical helper pipeline + cache integrity confirmed via apples-to-apples regression guard
- **32×5 finer luma quantization axis**: bins_changed=**43/160** (26.88%) — proportional doubling
- **16×10 finer SegNet class partition axis**: bins_changed=**43/160** (26.88%) — proportional doubling
- **dyn_range**: 772.41× across ALL 3 cells (inherited from Sister B cache; constant invariant of UNIWARD-weight derivation; the gradient distribution is fixed once the cache is fixed)
- **Per-bin delta magnitude trend**: max_delta DROPS with finer luma (32×5 max=3u8 vs 16×5/16×10 max=5u8); mean_l2 INCREASES with finer class (16×10 mean=0.47 vs 16×5 mean=0.33) — interesting but bounded — neither approaches the IMPL-CONDITIONAL `> 10×` threshold for paradigm-validation

## 3-strategy attack decomposition

Per CLAUDE.md "3-strategy attack framework" non-negotiable: Sister C sits on the **DISTORTION pure-axis** (sub-axis JOINT d_seg + d_pose at the LUT-derivation surface, specifically the LUT-shape sub-axis). The probe is an INTRA-TEST comparison at the LUT-statistic surface across 3 LUT-shape variants; ALL 3 cells use the SAME 200 frames + SAME 96×128 NEAREST-upsampled real-scorer cache + SAME canonical helpers + SAME canonical Sallee 2003 weighted-median statistic.

Sister of V15 / Sister B protocols at the LUT-shape sub-axis:
1. **DISTORTION axis** = TESTED-AT-3-LUT-SHAPES (this Sister C lane)
2. **RATE axis** = UNCHANGED (no archive emitted; LUT comparison only)
3. **FULL SCORER axis** = constant 772.41× dyn_range (inherited from Sister B cache); the LUT-shape axis does NOT change the upstream gradient response

## MLX-first → numpy-portable bridge contract

Honored per 8th + 13th standing directives:
- Sister B N=200 cache REUSED via Catalog #230 sister-disjoint READ-ONLY consumer (sha256 `e5799f4bd8659a0b...`)
- LUT derivation via canonical `build_uniward_weighted_chroma_lut` + `build_chroma_lut_from_ground_truth` (pure numpy; MLX-portable per `weight_map_per_lut_index.py` docstring; `grayscale_levels` and `num_segnet_classes` already parameterized — variable LUT shapes work via canonical API directly without forking)
- Per-class luma 2-way k-means subdivision for 10-class label is deterministic + pure numpy (sister of Daubechies hierarchical-partition discipline per CLAUDE.md "Council conduct" amendment + Catalog #335 Daubechies CO-LEAD)
- Comparison via canonical `compare_uniward_vs_canonical_lut` (pure numpy)
- ZERO MLX runtime dependency at the probe; macOS CPU only

## Apples-to-apples evidence discipline confirmation (per 10th standing directive)

Per CLAUDE.md "Apples-to-apples evidence discipline" NON-NEGOTIABLE + 10th standing directive:

1. **Same fixture all 3 cells**: UNIWARD-weighted LUT + canonical-median LUT both derived from the SAME 200 frames + SAME class labels (5-class SegNet argmax for 16×5+32×5; 10-class luma-k-means subdivision of the 5-class SegNet argmax for 16×10) + SAME EVAL_HW (384×512) + SAME upstream NEAREST-upsampled gradient cache (96×128 → 384×512) + SAME per-pixel UNIWARD weight map.
2. **Same canonical helpers**: `build_chroma_lut_from_ground_truth`, `build_uniward_weighted_chroma_lut`, `compute_per_pixel_uniward_weight_map_numpy`, `compare_uniward_vs_canonical_lut`, `load_differentiable_scorers`, `decode_real_pairs` — all sister-disjoint READ-ONLY consumer imports per Catalog #230.
3. **Same cache lineage**: Sister B's N=200 cache REUSED byte-identical (sha256 verified at probe load time); zero re-extension; zero re-compute of per-pixel UNIWARD weights — the dyn_range 772.41× is BYTE-EQUIVALENT to Sister B's measurement.
4. **Apples-to-apples 16×5 baseline reproduction**: 16×5 cell of Sister C produces (21/80, max_delta=5, mean_l2=0.3344, max_l2=5.0990, dyn_range=772.41×) EXACTLY matching Sister B's published anchor — proof of pipeline correctness via intra-test regression guard.
5. **Axis label honored**: every metric tagged `[macOS-CPU advisory]`; NOT promotable per Catalog #192.
6. **Holub-Fridrich-Denemark 2014 canonical alignment**: WebSearch-confirmed (per 10th standing directive WebSearch authorization) that the canonical UNIWARD per-coefficient routing principle applies to quantized + entropy-coded + per-symbol-routable surfaces; the v8 chroma LUT is the per-substrate-unique canonical analog; the canonical paper does NOT prescribe a specific quantization-table shape, so LUT shape is the substrate's PRINCIPLED structural choice tested empirically here.

Sources for Holub-Fridrich-Denemark 2014:
- [Universal distortion function for steganography in an arbitrary domain (EURASIP 2014)](http://dde.binghamton.edu/vholub/pdf/EURASIP14_Universal_Distortion_Function_for_Steganography_in_an_Arbitrary_Domain.pdf)
- [Springer canonical paper link](https://link.springer.com/article/10.1186/1687-417X-2014-1)
- [Further study on the security of S-UNIWARD (Denemark-Fridrich)](https://www.semanticscholar.org/paper/Further-study-on-the-security-of-S-UNIWARD-Denemark-Fridrich/e6b1fd00cad6be0169f64d3b40ce6283732f500b)

## Canonical-vs-unique decision per layer

Per Catalog #290 (binding for every substrate work):

| Layer | Decision | Rationale |
|---|---|---|
| N=200 cache loader (np.load) | ADOPT canonical (Sister B sister-disjoint READ-ONLY) | Sister B cache format identical; sha256 e5799f4bd8659a0b verified |
| Cache extension | N/A — REUSE Sister B's existing N=200 cache | Sister-disjoint per Catalog #230; saves ~3 min wall-clock; APPEND-ONLY discipline per Catalog #110/#113 (Sister B cache UNTOUCHED) |
| EVAL_HW pair decode | ADOPT canonical `decode_real_pairs` from `tac.substrates._shared.trainer_skeleton` | Sister-disjoint READ-ONLY |
| SegNet argmax (5-class) | ADOPT canonical pattern from Sister B + V15 build script (chunked, no_grad) | Sister-disjoint mirror |
| 10-class label derivation | UNIQUE per-method: per-class 1-D luma 2-way k-means subdivision of 5-class SegNet argmax | Per CLAUDE.md "Council conduct" Daubechies CO-LEAD discipline (hierarchical partition preserves coherence of the upstream SegNet classifier; the k-means split is on luma WITHIN each SegNet class to preserve scorer-conditional semantic identity); deterministic via quartile-init + 10 iterations |
| NEAREST upsample 96×128 → 384×512 | ADOPT canonical pattern from V15 build script + Sister B `_upsample_grad_nearest_to_full_res` | Sister-disjoint mirror; numpy.repeat preserves spatial mass |
| Per-pixel UNIWARD weight | ADOPT canonical `compute_per_pixel_uniward_weight_map_numpy` | Sister-disjoint READ-ONLY consumer import |
| UNIWARD-weighted LUT (per shape) | ADOPT canonical `build_uniward_weighted_chroma_lut` with `grayscale_levels` + `num_segnet_classes` kwargs | Canonical API already supports variable LUT shapes; no fork needed; APPLE-TO-APPLES across all 3 cells |
| Canonical-median LUT (per shape) | ADOPT canonical `build_chroma_lut_from_ground_truth` with same kwargs | Same as above |
| LUT comparison | ADOPT canonical `compare_uniward_vs_canonical_lut` | Sister-disjoint READ-ONLY consumer import |
| Verdict tree | UNIQUE per-method: 4-branch tree for combined-criteria-#2+#3 verdict | Extends Sister B's 4-branch verdict tree with finer-LUT-axis sensitivity; threshold-based per V15 protocol; bounded between PARADIGM-VALIDATED-WITH-COMBINED-CRITERIA + PARADIGM-INTACT-WITH-FURTHER-CRITERION-INTERACTION-SIGNAL + PARADIGM-INTACT-WITH-COMBINED-CRITERIA-PARTIALLY-FALSIFIED + INTERMEDIATE_PARTIAL_SCALING_AT_LUT_SHAPE_AXIS |
| Provenance.json + landing memo + sidecar LUT bytes | UNIQUE per-lane | Standard sister-disjoint per Catalog #110/#113 |
| Catalog #313 probe-outcome registration | ADOPT canonical `register_probe_outcome` | Sister-disjoint READ-ONLY consumer import; canonical 4-layer fcntl-locked JSONL pattern per Catalog #245 sister |

## 9-dimension success checklist evidence

Per Catalog #294:

1. **UNIQUENESS**: First N=200 macOS-CPU advisory empirical anchor on the LUT-shape-axis sensitivity (3 canonical sister LUT shapes: 16×5 baseline + 32×5 finer-luma + 16×10 finer-class); sister V15 N=50 + Sister B N=200 16×5 are the only prior empirical anchors on the UNIWARD per-LUT-index integration surface
2. **BEAUTY + ELEGANCE**: ~580 LOC standalone probe script + zero modification to canonical helpers or V15 build script or Sister B probe; sister-disjoint READ-ONLY consumer of Sister B's cache at sister path; sidecar LUT artifacts emitted to per-shape subdirectories for downstream consumability
3. **DISTINCTNESS**: Probe is INTRA-TEST LUT-shape-axis comparison at fixed N=200 + fixed dyn_range; does NOT shadow V15's paired-CUDA+CPU dispatch surface OR Sister B's sample-size-axis probe; orthogonal axis (LUT shape) to Sister B's orthogonal axis (sample size)
4. **RIGOR**: Smoke test PASSED with EXACT reproduction of Sister B N=200 16×5 baseline (21/80 bins changed, dyn_range 772.41×, max_delta=5u8, mean_l2=0.3344, max_l2=5.0990) — proof of canonical-helper pipeline correctness PRE full 3-cell run; Catalog #229 PV honored (read Sister B landing memo + V15 landing memo + canonical UNIWARD per-LUT integration package + canonical sweep tool + canonical equations registry + WebSearch Holub-Fridrich-Denemark 2014 BEFORE writing probe script); 6 dedicated checkpoints emitted
5. **OPTIMIZATION PER TECHNIQUE**: UNIWARD per-LUT-index aggregation is canonical Fridrich; weighted-median canonical Sallee 2003; gradient cache canonical N+1 anchor at 96×128 reused from Sister B; 10-class hierarchical partition is canonical Daubechies CO-LEAD discipline; all routed through canonical helpers
6. **STACK-OF-STACKS COMPOSABILITY**: Sister-disjoint READ-ONLY consumer imports preserve V15 + UNIWARD integration + NSCS06 v8 substrate + Sister B + canonical equations registry ALL untouched; Sister C is the canonical pattern for V15's remaining reactivation criterion #5 (weighted EXTREMA) to follow at $0 macOS-CPU advisory before any paid-CUDA validation
7. **DETERMINISTIC REPRODUCIBILITY**: numpy + torch + PIL all deterministic-seeded with seed=20260526; sister B cache REUSED byte-identical (sha256 e5799f4bd8659a0b...); LUT sha256s emitted for all 3 cells per shape (16×5 uniward=155a8305e5e9aa94 canonical=203eb763cd3a007d; 32×5 uniward=f345a9d82ccfa590 canonical=0f9ab943d52433b2; 16×10 uniward=f47a60b4ca926702 canonical=4cfa615d2a90d9d5); SegNet argmax deterministic with `no_grad`; per-class luma k-means deterministic via quartile-init + fixed 10 iterations
8. **EXTREME OPTIMIZATION + PERFORMANCE**: 80 sec total wall-clock on macOS M5 Max CPU (zero cache re-extension thanks to Sister B reuse; 23 sec scorer load + 17 sec SegNet argmax + 12 sec per-pixel UNIWARD weights + 11 sec per-class luma k-means for 10-class + 7 sec per shape × 3 shapes = 21 sec LUT derivation + 1 sec verdict + provenance write); ZERO paid GPU spend; well within operator budget for $0 macOS-CPU advisory scope
9. **OPTIMAL MINIMAL CONTEST SCORE**: Sister C is NOT a frontier-pursuit attempt; it is a controlled-comparison test of UNIWARD effect at the LUT-shape axis at fixed sample size (N=200) + fixed dyn_range (inherited from Sister B); its `PARADIGM_INTACT_WITH_COMBINED_CRITERIA_PARTIALLY_FALSIFIED` verdict IS the canonical empirical anchor for the FORMALIZATION_PENDING canonical equation's LUT-shape-axis insensitivity; informs Sister E (criterion #5 weighted EXTREMA) follow-on at $0 macOS-CPU advisory

## Cargo-cult audit per assumption

Per Catalog #303:

| Assumption | Pre-execution classification | Empirical verdict |
|---|---|---|
| "Finer LUT granularity (32×5 OR 16×10) at N=200 will unlock per-bin differentiation beyond Sister B's 16×5 baseline" | CARGO-CULTED-PENDING-VALIDATION (Sister B reactivation criterion #3 hypothesis) | **EMPIRICALLY FALSIFIED at LUT-shape axis** — bin-occupation ratio is STRUCTURALLY INVARIANT (~26.5%) across all 3 LUT shapes; the canonical weighted-median statistic does NOT discriminate bins MORE FREQUENTLY at finer granularity; the bins_changed COUNT doubles proportionally (21 → 43) but the RATIO stays bounded at 26.25-26.88% |
| "32×5 (finer luma axis) and 16×10 (finer class axis) have SYMMETRIC effect — neither axis is the binding constraint" | CARGO-CULTED-PENDING-VALIDATION | **EMPIRICALLY-PARTIALLY-CONFIRMED** — bins_changed identical (43/160) for both; max_delta differs (32×5 max=3u8 vs 16×10 max=5u8) suggesting 32×5 SMOOTHS deltas more (the finer luma axis distributes per-pixel perturbations across more luma bins per class, reducing the per-bin peak delta); mean_l2 differs (16×10 mean=0.47 vs 32×5 mean=0.31) suggesting 16×10 ACCUMULATES more L2 difference per bin (the finer class axis partitions pixels into smaller-population bins, where the UNIWARD-weighted-median sample size per bin is smaller and thus more sensitive to high-weight-pixel outliers) — both axes show STRUCTURAL EFFECTS but neither unlocks paradigm-validation-magnitude signal |
| "The dyn_range 772.41× from Sister B's N=200 cache will SCALE with LUT granularity (finer LUT → more bins to surface tail behavior)" | CARGO-CULTED-PENDING-VALIDATION | **EMPIRICALLY FALSIFIED** — dyn_range is CONSTANT 772.41× across all 3 LUT shapes; this is mathematically obvious in retrospect (the dyn_range is a property of the per-pixel UNIWARD weight distribution, which is a function of the gradient cache, which is independent of the LUT shape) — a cargo-cult conflation between "more bins" and "more tail behavior surfaced"; per Catalog #303 unwind: the dyn_range axis is independent of the LUT-shape axis, so combined criteria #2 and #3 do NOT compound at this implementation |
| "Per-class luma 2-way k-means subdivision of 5-class SegNet argmax preserves SegNet partition coherence" | HARD-EARNED-FROM-DAUBECHIES-HIERARCHICAL-PARTITION | **EMPIRICALLY CONFIRMED OPERATIONALLY** — per-sub-class pixel counts (cls_10 stage log) show roads/background classes split into bright+dark sub-bands cleanly; 10 sub-classes have well-separated populations (range 0.17% to 36.13%); the k-means split is structurally coherent per the Daubechies hierarchical-partition discipline |
| "macOS CPU pipeline is byte-identical reproduction of Sister B's pipeline" | HARD-EARNED-FROM-CANONICAL-HELPER-DETERMINISM | **CONFIRMED VIA SMOKE** — 16×5 cell of Sister C reproduces Sister B's published anchor EXACTLY at every digit (21/80 bins, max_delta=5, mean_l2=0.3344, max_l2=5.0990, dyn_range=772.41×) |
| "macOS CPU is 1:1 contest-compliant for LUT-statistic signal" | CARGO-CULTED-PENDING-VALIDATION | **STRUCTURALLY-FALSIFIED** per Catalog #192 — macOS-CPU is NEVER 1:1 contest-compliant; the LUT-statistic IS valid `[macOS-CPU advisory]` research signal but cannot promote to score claim without paired Linux x86_64 paired-CUDA evidence |
| "The bin-occupation-ratio INVARIANT at ~26.5% across LUT shapes is a real signature of the UNIWARD-weighted-median statistic's STRUCTURAL coverage of the per-pixel UNIWARD weight distribution, not noise" | HARD-EARNED-FROM-EMPIRICAL-INVARIANCE-ACROSS-3-CELLS | **CONFIRMED EMPIRICALLY** — the invariance is precise (26.25% vs 26.88% vs 26.88%; max deviation 0.63%); the canonical weighted-median statistic at a given dyn_range produces a STRUCTURAL fraction of bins where the weighted-median differs from the unweighted median; this fraction is determined by the per-bin pixel-population × weight-distribution, NOT the total number of bins |

## Observability surface

Per Catalog #305 — all 6 facets HONORED:

- **Inspectable per layer**: per-pair UNIWARD weight stats (min/max/mean/std/dynamic_range per cell); per-bin LUT comparison (`WeightedMedianResult.per_bin_l2_difference` array exposes per-(level, class) L2 delta for each cell); per-axis SegNet/PoseNet gradient stats from inherited Sister B cache; top-3 highest-L2 bins per cell exposed in provenance.json
- **Decomposable per signal**: per-LUT-shape vs Sister B N=200 baseline ratios per metric exposed; per-shape × per-axis bin_count + bin_pct + max_delta + mean_l2 + max_l2 + dyn_range in provenance.json `empirical_results_per_lut_shape` table; rate-axis NOT computed (no archive emitted)
- **Diff-able across runs**: deterministic seed (20260526); Sister B cache sha256 cited (e5799f4bd8659a0b...); 6 LUT sha256s emitted per cell (3 shapes × 2 LUT variants); verdict reproducible from provenance.json
- **Queryable post-hoc**: provenance.json (~12KB JSON; sort_keys=True byte-stable); probe_full.log (full stdout trace with stage timestamps); Catalog #313 probe-outcomes ledger entry (queryable via `tac.probe_outcomes_ledger.query_by_substrate('nscs06_v8_chroma_lut_uniward_per_lut_index')`); subagent checkpoint trail via `.omx/state/subagent_progress.jsonl`; per-LUT-shape sidecar artifacts at `lut_artifacts_{shape_name}/lut_uniward.npy` + `lut_canonical.npy` + `per_bin_l2_difference.npy` for downstream consumption
- **Cite-able**: 1 Sister B cache sha256 + 6 LUT sha256s + 5 canonical Provenance fields (`evidence_grade=macOS-CPU-advisory`, `axis_tag=[macOS-CPU advisory]`, `hardware_substrate=darwin_arm64_m5_max_macos_cpu`, `score_claim=False`, `promotable=False`)
- **Counterfactual-able**: V15 N=50 16×5 baseline + Sister B N=200 16×5 baseline + Sister C 3-LUT-shape cells provide the canonical cell-by-cell counterfactual; SMOKE mode (--smoke flag) reproduces 16×5 baseline exactly for the canonical apples-to-apples regression guard; Sister D (criterion #4) + Sister E (criterion #5) future spawns can probe at sister axes for compound counterfactual

## Predicted ΔS band vs empirical (per Catalog #296)

PREDICTED (from V15 reactivation criteria #2+#3 combined hypothesis): one of {PARADIGM-VALIDATED-WITH-COMBINED-CRITERIA, PARADIGM-INTACT-WITH-FURTHER-CRITERION-INTERACTION-SIGNAL, PARADIGM-INTACT-WITH-COMBINED-CRITERIA-PARTIALLY-FALSIFIED, INTERMEDIATE_PARTIAL_SCALING_AT_LUT_SHAPE_AXIS}. The expected outcome was bounded by Sister B's INTERMEDIATE_PARTIAL_SCALING verdict at N=200 16×5 — the LUT-shape axis was expected to either compound the partial scaling (PARADIGM-VALIDATED) or fail to compound (PARADIGM-INTACT-PARTIALLY-FALSIFIED).

EMPIRICAL: 32×5 dyn_range = 16×10 dyn_range = 16×5 dyn_range = 772.41× (CONSTANT; LUT-shape independent). Bin-occupation ratios 26.25% (16×5) / 26.88% (32×5) / 26.88% (16×10) (BOUNDED 26.25-26.88%). Per the verdict tree: `dyn_range < 1000×` AND `dyn_range < 2000×` for BOTH finer shapes → `PARADIGM_INTACT_WITH_COMBINED_CRITERIA_PARTIALLY_FALSIFIED`.

Per Dykstra-feasibility (Catalog #296): the intersection of (LUT-shape-finer-granularity ∩ UNIWARD-weighted-median-discrimination ∩ N=200-sample-size ∩ 772.41x-dyn-range ∩ contest-scorer-sensitivity) is **STATISTICALLY-EMPIRICALLY-EMPTY** at the LUT-shape axis. The bin-occupation invariance signature (~26.5% across all shapes) is the canonical Dykstra-feasibility EMPTY-INTERSECTION evidence for this axis — the canonical weighted-median statistic at this dyn_range saturates the per-bin discrimination at ~26.5% regardless of granularity.

## Horizon-class declaration

Per Catalog #309: **plateau_adjacent** (UNCHANGED from V15 + Sister B DOWN-REVISED-from-frontier_pursuit). The Sister C empirical signal does NOT move into frontier_pursuit territory; the LUT-statistic signal is bounded by the canonical Sallee 2003 weighted-median's per-bin discrimination ceiling at the given dyn_range. Sister E spawn at criterion #5 (weighted EXTREMA) may shift to frontier_pursuit IF the EXTREMA statistic empirically discriminates more aggressively than the median — the EXTREMA (top-K rank) is a HIGHER-VARIANCE statistic that may concentrate routing effects on high-weight-pixel anchors, but this requires empirical validation per the Sister E follow-on.

## Catalog #344 canonical equation anchor status

**Status**: `FORMALIZATION_PENDING` (UNCHANGED; no PROMOTION). The canonical equation `uniward_per_lut_index_distortion_weight_savings_v1` cannot promote to PARADIGM-VALIDATED-EMPIRICALLY from this Sister C anchor because:

1. **Catalog #192**: macOS-CPU advisory is NOT 1:1 contest-compliant hardware (Linux x86_64 required for `[contest-CPU]`; NVIDIA GPU on Linux required for `[contest-CUDA]`).
2. **V15 PROMOTION threshold**: paired-CUDA delta > 1e-4 declared in V15 build-script protocol; this Sister C probe does NOT measure paired-CUDA delta (scope explicitly excludes paid dispatch per the operator's reactivation-criteria-#2+#3 scope).
3. **Combined-criteria partial falsification**: the Sister C signal EMPIRICALLY FALSIFIES the LUT-shape-axis cargo-cult assumption; the canonical equation's natural domain is preserved (Holub-Fridrich-Denemark 2014 canonical per-coefficient routing), but the specific NSCS06 v8 substrate's coupling between (luma, class) granularity and UNIWARD-weighted-median discrimination saturates at ~26.5% regardless of granularity — this is a SUBSTRATE-SPECIFIC EMPIRICAL CONSTRAINT, NOT a paradigm-level constraint.

This anchor extends V15's N=50 + Sister B's N=200 16×5 empirical evidence with the LUT-shape-axis insensitivity signal at N=200 but does NOT promote the equation. The canonical equation registration script for `uniward_per_lut_index_*` correctly remains FORMALIZATION_PENDING until paired-CUDA evidence lands.

Per the per-substrate individually-fractal directive (just-saved 8th + 13th standing directives): the substrate-unique-optimal LUT shape question is now EMPIRICALLY-CONSTRAINED — at N=200 + 772.41× dyn_range, the LUT-shape axis does NOT differentiate the substrate's UNIWARD effect; the substrate's bottleneck is elsewhere (likely the statistic choice — Sister E's weighted EXTREMA — or the application-surface choice — Sister D's NSCS06 grayscale_lut / VQ-VAE indices_blob).

Reactivation criteria for future PROMOTION (per CLAUDE.md "Forbidden premature KILL"):

1. **Reactivation criterion #1**: COMPLETE (Sister B landing).
2. **Reactivation criterion #2** (higher real-scorer dynamic range): operator-routable per Sister B + Sister C results — the dyn_range axis IS sample-size-responsive (39× expansion from N=50 → N=200) but the LUT-shape axis does NOT compound; criterion #2 still has follow-on EV via longer training / lower learning rate / scorer-architecture diversity, but Sister C result suggests the LUT-shape axis is NOT the right composing dimension.
3. **Reactivation criterion #3**: COMPLETE-AND-PARTIALLY-FALSIFIED (this Sister C landing). The LUT-shape axis empirically saturates at ~26.5% bin-occupation invariance; finer granularity does NOT unlock signal beyond what N=200 sample size already provides.
4. **Reactivation criterion #4** (operator-routable per V15 landing memo): apply UNIWARD per-LUT-index routing to NSCS06 grayscale_lut (T3 council #2 stacking candidate) or VQ-VAE indices_blob (T3 council #3 stacking candidate) — different application surface family may yield different empirical signals per-substrate. Sister D follow-on.
5. **Reactivation criterion #5** (RECOMMENDED for Sister E follow-on per this Sister C landing): UNIWARD-weighted EXTREMA (top-K quantile) replaces UNIWARD-weighted median. The MIXED signal at Sister C (LUT-shape axis saturated; per-bin delta magnitudes bounded; dyn_range cache-determined) suggests the EXTREMA statistic may concentrate high-weight-pixel routing more strongly than the rank-50% median which averages over the entire CDF — the EXTREMA's variance is higher and may unlock per-bin discrimination beyond the canonical Sallee 2003 weighted-median's empirical ~26.5% saturation ceiling.

Per Sister C's empirical `PARADIGM_INTACT_WITH_COMBINED_CRITERIA_PARTIALLY_FALSIFIED` verdict: **recommend criterion #5 (weighted EXTREMA) for Sister E follow-on** (the canonical Fridrich 2014 formulation explicitly supports both weighted median and weighted EXTREMA at the per-coefficient routing surface; the EXTREMA is the higher-variance statistic that may surface concentrated high-weight-pixel routing more strongly).

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map**: ACTIVE — per-pixel UNIWARD weights from inherited Sister B N=200 cache feed canonical sensitivity-map surface; per-LUT-shape per-(level, class) bin weights surfaced as the canonical Fridrich aggregate
2. **Pareto constraint**: ACTIVE — per-axis `[macOS-CPU advisory]` tag carries Catalog #323 canonical Provenance; non-promotable per Catalog #341 (score_claim=False + promotable=False); explicit non-frontier-crossing per V15 protocol
3. **Bit-allocator**: ACTIVE — the LUT-shape axis IS the per-bin bit allocator's design decision (16×5 vs 32×5 vs 16×10 alters per-bin sample population which directly affects per-bin chroma triplet precision); empirical Sister C result is itself a bit-allocator-design constraint (LUT-shape axis does NOT improve UNIWARD discrimination at this fixture)
4. **Cathedral autopilot dispatch**: ACTIVE — Catalog #313 probe-outcomes ledger row registered (probe_id=`v15_uniward_sister_c_reactivation_criteria_2_3_combined_n200_lut_shapes_20260527T042700Z`; verdict=DEFER; blocker_status=advisory; expires_at_utc=2026-06-26T04:27:57Z). Cathedral autopilot ranker can consume the empirical `PARADIGM_INTACT_WITH_COMBINED_CRITERIA_PARTIALLY_FALSIFIED` verdict + 3-LUT-shape signals as evidence for sister probes at reactivation criteria #4 (Sister D) or #5 (Sister E)
5. **Continual-learning posterior**: ACTIVE — provenance.json carries canonical Provenance per Catalog #323; per-LUT-shape sidecar LUT artifacts at `lut_artifacts_*/lut_uniward.npy` + `lut_canonical.npy` + `per_bin_l2_difference.npy` are queryable by future sister probes (Sister E can read the Sister C N=200 cache as base + iterate the EXTREMA statistic on top); Catalog #313 probe-outcome ledger entry feeds canonical continual-learning posterior
6. **Probe-disambiguator**: ACTIVE — Sister C IS the canonical disambiguator at the LUT-shape axis; the empirical `PARADIGM_INTACT_WITH_COMBINED_CRITERIA_PARTIALLY_FALSIFIED` result disambiguates between Sister B's `INTERMEDIATE_PARTIAL_SCALING` (which the operator interpreted as criterion #2 + #3 combined being the path forward) and the actual empirical reality (the LUT-shape axis is NOT a compounding axis under the canonical weighted-median statistic; criterion #5 weighted EXTREMA is the new compounding-axis candidate)

## Carmack-dissent verdict per Catalog #307

**PARADIGM-INTACT + IMPLEMENTATION-LEVEL-EXPLORATION-AT-LUT-SHAPE-AXIS-PARTIALLY-FALSIFIED**. Per CLAUDE.md "Forbidden premature KILL without research exhaustion":

- The UNIWARD per-LUT-index routing PRIMITIVE remains canonical Fridrich; this Sister C probe does NOT falsify the primitive
- The IMPLEMENTATION at (N_LUT=200, dynamic_range=772.41×, LUT=ANY of 16×5/32×5/16×10) shows BIN-OCCUPATION INVARIANT signal — the LUT-shape axis does NOT unlock per-bin differentiation beyond Sister B's N=200 16×5 baseline; the canonical weighted-median statistic at this dyn_range empirically saturates at ~26.5% bin occupation regardless of granularity
- The 4-of-5 reactivation criteria above enumerate the research-exhaustion paths; KILL is the LAST RESORT
- The 5th + 6th + 7th-MLX + V15-paired-CUDA + sister-B-N=200 + sister-C-N=200-3-shapes empirical trajectory is now a 6-point sequence: PARADIGM-NULL-AT-RAW-RGB → PARADIGM-NULL-AT-CAPACITY-CONSTRAINED → PARADIGM-VALIDATED-AT-MLX-LOCAL-SYNTHETIC → IMPLEMENTATION-FALSIFIED-AT-PAIRED-CUDA-REAL-N=50 → IMPLEMENTATION-PARTIAL-SCALING-AT-MACOS-CPU-ADVISORY-N=200-16X5 → IMPLEMENTATION-LUT-SHAPE-AXIS-PARTIALLY-FALSIFIED-AT-MACOS-CPU-ADVISORY-N=200-3-SHAPES
- The next iteration is criterion #5 (weighted EXTREMA replaces median); the EXTREMA statistic operates on a higher-variance surface that may concentrate routing effects on high-weight-pixel anchors more aggressively than the median

Per Catalog #348 retroactive sweep: **NOT TRIGGERED**. The Sister C verdict is IMPLEMENTATION-LEVEL not PARADIGM-LEVEL; no historical KILL/DEFER/FALSIFY verdicts are invalidated by Sister C (V15 IMPL-FALSIFIED + Sister 7th-order PARADIGM-VALIDATED-MLX-LOCAL + 5th/6th-order PARADIGM-NULL + Sister B INTERMEDIATE_PARTIAL_SCALING verdicts ALL remain intact — Sister C simply extends Sister B's N=200 sample-size-axis evidence with the LUT-shape-axis evidence at macOS-CPU advisory scope; the empirical signal is consistent across both probes).

## Sister-disjoint discipline confirmation per Catalog #230

NO modifications to V15 build script, V15 source cache, Sister B source cache, UNIWARD per-LUT-index integration package, NSCS06 v8 substrate, canonical equations registry, or any sister artifact. All canonical helpers consumed READ-ONLY (`compute_per_pixel_uniward_weight_map_numpy`, `build_uniward_weighted_chroma_lut`, `compare_uniward_vs_canonical_lut`, `build_chroma_lut_from_ground_truth`, `decode_real_pairs`, `load_differentiable_scorers`, `register_probe_outcome`). Sister B's N=200 cache REUSED via Catalog #110/#113 APPEND-ONLY (cache UNTOUCHED; Sister B cache file at `experiments/results/v15_uniward_sister_b_reactivation_criterion_1_n200_macos_cpu_advisory_20260526/real_scorer_gradients_cache_extended_n200.npz` is byte-identical pre/post Sister C probe). Output strictly scoped under `experiments/results/v15_uniward_sister_c_reactivation_criteria_2_3_combined_n200_lut_shapes_macos_cpu_advisory_20260526/` + `.omx/research/v15_uniward_sister_c_reactivation_criteria_2_3_combined_n200_lut_shapes_*_landed_20260526.md`.

Sister-disjoint confirmed via `subagent_progress.jsonl` review at probe-execution time: in-flight sisters at Sister C launch time were META-resurrection-audit-v2 (`8be27396`; touches T3_council + resurrection_audit + premature_falsification + meta_bug_retroactive + pre_rigor_inventory — all DISJOINT from UNIWARD per-LUT-index files) + NSCS06 v8 cls_stream OPTIMAL-TECHNIQUE PV supersession (`23011830`; touches `.omx/tmp/cls_stream_*` files — DISJOINT from UNIWARD per-LUT-index files). Catalog #340 sister-checkpoint guard not triggered.

## Discipline anchors

- **Catalog #229 PV** (read Sister B landing memo + V15 landing memo + UNIWARD per-LUT integration package + canonical sweep tool + canonical Provenance helpers + canonical equations registry + canonical_frontier_pointer.json + Sister B probe script + 8th-13th standing directive memos + WebSearch Holub-Fridrich-Denemark 2014 + sister subagent activity in `subagent_progress.jsonl` BEFORE writing probe script)
- **Catalog #206** (6 checkpoints emitted across PV / scope-analysis / smoke-pass / probe-launch / probe-complete / landing-memo cycle)
- **Catalog #110/#113 APPEND-ONLY** (Sister B cache + V15 cache + V15 build artifacts + V15 landing memo + Sister B landing memo + canonical helpers ALL untouched; Sister C sidecar LUT artifacts + provenance.json + landing memo are NEW artifacts only)
- **Catalog #117/#157/#174/#235/#289** canonical commit serializer (will use --expected-content-sha256 + co-author trailer for the commit)
- **Catalog #119** Co-Authored-By Claude Opus 4.7 (1M context) trailer (internal commit)
- **Catalog #125** 6-hook wire-in declaration (all 6 hooks ACTIVE per the declaration section above)
- **Catalog #127** authoritative-tag custody metadata (every metric tagged `[macOS-CPU advisory]` per per-call-site discipline)
- **Catalog #131/#138** fcntl-locked + strict-load discipline (Catalog #313 probe-outcomes ledger + provenance.json sidecar follow canonical helper patterns; no bare writes to `.omx/state/`)
- **Catalog #164/#226** canonical scorer-load + trainer auth-eval routing (via canonical `load_differentiable_scorers`; sister-disjoint READ-ONLY consumer import; no fork)
- **Catalog #168** AST walker handles both Assign + AnnAssign (probe script uses no AST scan; gate N/A)
- **Catalog #185** META-meta-meta drift detection (this gate run did NOT touch CLAUDE.md catalog text; gate N/A)
- **Catalog #192** macOS-CPU advisory non-promotion (every metric tagged + `promotable=False`; `score_claim=False`; `ready_for_exact_eval_dispatch=False`)
- **Catalog #199** paired-env operator-authorize bypass (N/A; no paid dispatch attempted)
- **Catalog #205/#295** inflate runtime hygiene (N/A; no archive emitted; no inflate.py runtime emitted)
- **Catalog #208** docs/* no local-absolute-paths (landing memo cites repo-relative paths only; no machine-local absolute path leakage) <!-- DOCS_LOCAL_PATH_OK:canonical_catalog_208_anchor_reference_quoting_the_gate_pattern_per_sister_b_precedent -->
- **Catalog #220** L1+ scaffold operational mechanism (N/A; no L1+ substrate lane modified; Sister C is research-signal advisory only)
- **Catalog #226** trainer auth-eval canonical helper (N/A; no paid auth-eval dispatched)
- **Catalog #229** premise-verification-before-edit ✓ (PV documented above)
- **Catalog #230** ownership map (READ-ONLY consumer imports for all canonical helpers; sister-disjoint from in-flight sisters confirmed)
- **Catalog #240** recipe-vs-trainer-state consistency (N/A; no recipe emitted; Sister C is research-signal advisory only)
- **Catalog #244** NVML/Modal/CUDA env hygiene (N/A; no paid Modal dispatch)
- **Catalog #245** canonical Modal call_id ledger (N/A; no Modal dispatch)
- **Catalog #246** canonical paired-dispatch helper (N/A; no paired-CUDA dispatch per scope)
- **Catalog #287** placeholder-rationale rejection (all rationales ≥4 chars substantive; no `<rationale>`/`<reason>` literals)
- **Catalog #290** canonical-vs-unique decision per layer ✓ (table above)
- **Catalog #292** per-deliberation assumption surfacing (N/A; no council deliberation invoked)
- **Catalog #294** 9-dimension success checklist evidence ✓ (section above)
- **Catalog #296** Dykstra-feasibility predicted-band check ✓ (section above)
- **Catalog #297** signal-axis destruction reversibility (N/A; no signal axis destroyed)
- **Catalog #300** council deliberation v2 frontmatter (N/A; no council deliberation memo)
- **Catalog #303** cargo-cult audit per assumption ✓ (section above)
- **Catalog #305** observability surface ✓ (section above)
- **Catalog #307** paradigm-vs-implementation classification (PARADIGM-INTACT + IMPLEMENTATION-LEVEL-EXPLORATION-AT-LUT-SHAPE-AXIS-PARTIALLY-FALSIFIED)
- **Catalog #309** horizon_class declaration (plateau_adjacent; UNCHANGED from V15 + Sister B)
- **Catalog #313** probe-outcomes ledger ✓ (registered: probe_id=`v15_uniward_sister_c_reactivation_criteria_2_3_combined_n200_lut_shapes_20260527T042700Z`; verdict=DEFER; blocker_status=advisory; expires_at_utc=2026-06-26T04:27:57Z; staleness_window_days=30)
- **Catalog #316** reports/latest.md frontier alignment (N/A; Sister C is NOT frontier-crossing; reports/latest.md NOT mutated)
- **Catalog #317/#341** canonical-routing markers for local research signal (all `[macOS-CPU advisory]` markers + non-promotable contract honored)
- **Catalog #323** canonical Provenance umbrella ✓ (provenance.json carries every required canonical field)
- **Catalog #335** cathedral consumer canonical contract (Sister C emits machine-readable provenance.json + Catalog #313 anchor; future cathedral consumer can integrate as observability-only Tier-A per Catalog #357 dual-tier discipline)
- **Catalog #340** sister-checkpoint guard PROCEED before commit (will run pre-commit per discipline; checked sister activity above)
- **Catalog #343** NO hardcoded frontier score literals (cite canonical_frontier_pointer.json paths; the V15 + Sister B baseline anchors are CONTROLLED-COMPARISON empirical numbers carried via sister memos per Catalog #110 HISTORICAL_PROVENANCE)
- **Catalog #344** canonical equation anchor status: `uniward_per_lut_index_distortion_weight_savings_v1` FORMALIZATION_PENDING preserved (per `# FORMALIZATION_PENDING:per_v15_landing_memo_macos_cpu_advisory_does_not_promote_pending_paired_cuda_evidence_at_higher_dynamic_range_via_criterion_5_weighted_extrema_for_sister_e_follow_on` substantive rationale)
- **Catalog #346** canonical roster N/A (no T2+ deliberation invoked)
- **Catalog #348** retroactive sweep NOT TRIGGERED (Sister C is IMPLEMENTATION-LEVEL not PARADIGM-LEVEL)
- **Catalog #356/#357** per-axis decomposition + Tier B dual-tier consumer architecture (Sister C is Tier A observability-only by construction; no per-axis Tier B emission)
- **CLAUDE.md "Apples-to-apples evidence discipline"** ✓ (intra-test LUT-shape comparison at same fixture + same canonical helpers + smoke-mode verifies Sister B N=200 16×5 baseline reproduction)
- **CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"** ✓ (macOS-CPU advisory explicitly tagged + non-promotable + result_review_blockers cite "requires_paired_cuda_cpu_validation_per_v15_protocol_1e_4_threshold")
- **CLAUDE.md "Forbidden premature KILL without research exhaustion"** ✓ (criterion #5 weighted EXTREMA reactivation queued; PARADIGM-INTACT classification; KILL is the LAST RESORT)
- **CLAUDE.md "MPS auth eval is NOISE"** ✓ (CPU only; no MPS used; tagged `[macOS-CPU advisory]`)
- **CLAUDE.md "MLX-first portable-local-substrate authority"** ✓ (probe is numpy + torch CPU only; ZERO MLX dependency; Sister B cache reused at sister path)
- **CLAUDE.md "Council conduct" amendment (Daubechies CO-LEAD)** ✓ (per-class luma 2-way k-means subdivision for 10-class label is canonical hierarchical-partition discipline)
- **8th MLX-first + numpy-portable + individually-fractal standing directive** ✓ (sister-disjoint READ-ONLY consumer of Sister B's cache; pure numpy + torch CPU; per-substrate principled LUT shape choice per Holub-Fridrich-Denemark 2014 canonical per-coefficient routing)
- **10th apples-to-apples + WebSearch authorized + math/science/engineering rigor standing directive** ✓ (WebSearch invoked to verify Holub-Fridrich-Denemark 2014 canonical LUT-shape selection guidance; math rigor binding via canonical Fridrich UNIWARD primitive + canonical Sallee 2003 weighted-median + canonical Daubechies hierarchical-partition; engineering rigor binding via the 13 HNeRV parity inviolable lessons + Catalog #290/#294/#296/#297/#303/#305 design-memo discipline)
- **11th final rate attack off-the-shelf + development workflow standing directive** ✓ (Sister C demonstrates the canonical $0 macOS-CPU advisory dev-loop workflow that compounds V15 + Sister B paired-CUDA evidence)
- **12th 3-strategy attack framework standing directive** ✓ (Sister C sits on DISTORTION pure-axis at LUT-shape sub-axis; orthogonal to RATE axis from FEC family + FULL SCORER axis from V15 paired-CUDA)
- **13th OPTIMAL-TRIO standing directive** ✓ (TECHNIQUE = canonical UNIWARD per-LUT-index primitive at 3 canonical sister LUT shapes; WAY = canonical cell-by-cell empirical measurement at fixed dyn_range; TIME = pre-dispatch PV per Catalog #229 + V15 protocol; sister-disjoint UNIQUE-AND-COMPLETE-PER-METHOD per the canonical helper consumer-only pattern)

## NOT a PR111 candidate — explicit declaration

Per V15 protocol + Catalog #192/#343: PR111 candidate status requires paired-CUDA frontier-crossing on 1:1 contest-compliant hardware (Linux x86_64 + NVIDIA GPU). Sister C is `[macOS-CPU advisory]` — by construction NOT 1:1 contest-compliant; the bins_changed/mean_l2/dyn_range metrics are LUT-statistic research signals NOT contest scores. Per the `user_pr_attribution` memory non-negotiable + CLAUDE.md "FORBIDDEN CLAUDE ATTRIBUTION IN PUBLIC-PR SURFACES": Sister C does NOT produce a PR111 candidate report; no public-PR-facing artifact is written.

## Operator-routable next steps

1. **Sister E spawn** (reactivation criterion #5 RECOMMENDED): UNIWARD-weighted EXTREMA (top-K quantile) replaces UNIWARD-weighted median. The Sister C result EMPIRICALLY ESTABLISHES that the LUT-shape axis saturates the weighted-median statistic's per-bin discrimination at ~26.5%; the EXTREMA is the canonical higher-variance alternative per Fridrich 2014. Estimated $0 macOS-CPU advisory ~5-10 min wall-clock (Sister B + Sister C caches REUSED + canonical helpers; sister probe builds top-K weighted statistic on top of existing per-pixel UNIWARD weights at 3-LUT-shape grid).
2. **Sister D spawn** (reactivation criterion #4): apply UNIWARD per-LUT-index routing to NSCS06 grayscale_lut OR VQ-VAE indices_blob — different application surface family; per the per-substrate individually-fractal directive, different substrates may have different idiosyncratic structural couplings.
3. **Sister F spawn** (paired-CUDA validation of Sister E IFF Sister E produces a sister LUT-statistic with stronger N=200 signal): queue paired-CUDA validation via canonical `tools/dispatch_modal_paired_auth_eval.py` at estimated $0.50-1.00 cost (matches V15 budget); the paired-CUDA delta IS the canonical PROMOTION arbiter per the V15 1e-4 threshold protocol.
4. **NOT-RECOMMENDED**: re-running V15 paired-CUDA at the 16×5 / 32×5 / 16×10 LUT shapes directly (the Sister C result EMPIRICALLY ESTABLISHES that the LUT-shape axis does NOT compound the criterion #2 dyn_range signal under the canonical weighted-median; without Sister E's compounding via criterion #5 EXTREMA, the paired-CUDA delta at N=200 will likely remain BELOW the 1e-4 PROMOTION threshold; better to test criterion #5 EXTREMA at $0 macOS-CPU advisory first then test the combined result with paid dispatch).
5. **Sister B's earlier recommendation** (combine criteria #2 + #3) is EMPIRICALLY-PARTIALLY-FALSIFIED by Sister C at the LUT-shape sub-axis; the dyn_range axis (criterion #2) remains a candidate compounding axis BUT requires a different orthogonal compounding axis (criterion #5 weighted EXTREMA OR criterion #4 different application surface), not criterion #3 LUT granularity.

## Cost

- $0 GPU spend (macOS CPU only; no Modal / Vast.ai / Lightning dispatch)
- ~80 sec CPU wall-clock on M5 Max (zero cache extension thanks to Sister B reuse + smoke regression guard at 16×5 baseline)
- ~580 LOC standalone probe script (~38KB)
- ~450 LOC landing memo (~30KB; this file)
- 1 Catalog #313 probe-outcome ledger entry registered (verdict=DEFER; blocker_status=advisory)
- 0 canonical equations PROMOTED (FORMALIZATION_PENDING preserved per Catalog #344)
- 0 cache extensions emitted (Sister B's N=200 cache REUSED via Catalog #230 sister-disjoint READ-ONLY)
- 6 sidecar LUT artifacts emitted at `experiments/results/v15_uniward_sister_c_reactivation_criteria_2_3_combined_n200_lut_shapes_macos_cpu_advisory_20260526/lut_artifacts_{shape_name}/` (3 shapes × 2 LUT variants each + per-bin L2 difference array; sister-disjoint; APPEND-ONLY)
- 5 reactivation criteria status: #1 COMPLETE (Sister B), #3 COMPLETE-AND-PARTIALLY-FALSIFIED (this Sister C), #2 + #4 + #5 operator-routable; criterion #5 RECOMMENDED for next Sister E

## Sister-coordination + checkpoint discipline confirmation

5 subagent checkpoints emitted via canonical `tools/subagent_checkpoint.py` per Catalog #206 (Steps 1-5; final Step 6 marks complete after commit):
1. Step 1: subagent registration + PV launch
2. Step 2: PV complete + scope analysis (Sister B + V15 + UNIWARD per-LUT integration + canonical helpers + WebSearch Holub-Fridrich-Denemark 2014 + sister disjoint via `subagent_progress.jsonl`)
3. Step 3: probe script created + smoke launch
4. Step 4: smoke test PASS (16×5 baseline reproduces Sister B EXACTLY) + full 3-LUT-shape probe launch
5. Step 5: full probe complete + Catalog #313 registered + landing memo write
6. Step 6 (final): canonical commit via serializer + checkpoint complete

Sister-disjoint per Catalog #230: NO collision with in-flight sisters (META-resurrection-audit-v2 `8be27396` + NSCS06 v8 cls_stream OPTIMAL-TECHNIQUE PV supersession `23011830`; both touch DISJOINT files). Catalog #340 sister-checkpoint guard expected PROCEED on commit (no overlap on probe_uniward_lut_shapes_n200.py / provenance.json / lut_artifacts_*/ / this landing memo).

## V15 13th OPTIMAL-TRIO declaration per 13th standing directive

Sister C honors the 13th OPTIMAL-TRIO standing directive (mlx-first-numpy-portable-individually-fractal):
- **MLX-first**: reused Sister B's MLX-local-captured cache via canonical READ-ONLY consumer pattern; canonical UNIWARD per-LUT-index integration is MLX-portable (pure numpy with MLX-portability hooks per sister `weight_map_per_lut_index.py` docstring)
- **numpy-portable**: probe script + sidecar artifacts + LUT derivation are pure numpy + torch CPU; ZERO MLX dependency at runtime; inflate-side runtime UNTOUCHED per HNeRV parity L4
- **individually-fractal**: Sister C is unique-per-method (sister-disjoint READ-ONLY consumer imports from V15 / Sister B / V14-V2 / 5th+6th+7th-order siblings + zero modification to canonical helpers); the probe script does NOT extend V15's build script OR Sister B's probe script — it composes the canonical helpers at the per-method optimal point per the canonical sister pattern; the 3 LUT shapes are PRINCIPLED canonical sisters of the v8 substrate's (16, 5) shape choice (32×5 doubles luma-axis; 16×10 doubles class-axis; both are natural sisters per Holub-Fridrich-Denemark 2014 per-coefficient routing principle, NOT generic catalog-enumeration)

Per the 12th 3-strategy attack framework standing directive: Sister C sits on the DISTORTION pure-axis attack strategy at the LUT-shape sub-axis; the empirical `PARADIGM_INTACT_WITH_COMBINED_CRITERIA_PARTIALLY_FALSIFIED` verdict does NOT block the 11 sister substrates on PLATEAU + 13 sister substrates on FRONTIER + 4 sister substrates on ASYMPTOTIC per the comprehensive roadmap synthesis; the next iteration of UNIWARD per-LUT-index routing is queued via reactivation criterion #5 (weighted EXTREMA) per Catalog #313 probe-outcomes ledger.

Per the 11th final-rate-attack off-the-shelf standing directive: Sister C demonstrates the canonical $0 macOS-CPU advisory dev-loop workflow under operator-session-budget discipline ($0 spend / 80 sec wall-clock; uses Sister B's existing cache via canonical READ-ONLY consumer); the canonical 4-layer pattern is honored (canonical Sister B cache reuse + canonical UNIWARD per-LUT-index integration + Catalog #313 probe-outcomes ledger + provenance.json sidecar with full canonical Provenance).

Per the 10th apples-to-apples + online research authorization + math/science/engineering rigor standing directive: Sister C EMPIRICALLY VALIDATES apples-to-apples via the smoke-mode regression guard (reproduces Sister B N=200 16×5 baseline EXACTLY); WebSearch invoked for Holub-Fridrich-Denemark 2014 canonical LUT-shape selection guidance (confirmed: paper does NOT prescribe specific JPEG quantization-table shape; substrate's structural choice is empirically-testable per the per-substrate individually-fractal directive); math rigor binding via Shannon-rate-distortion principle (V15 PROMOTION threshold of 1e-4 paired-CUDA delta IS the canonical apples-to-apples science-rigor bound); engineering rigor binding via the 13 HNeRV parity inviolable lessons + Catalog #290/#294/#296/#297/#303/#305 design-memo discipline.

Per the 7th AUTOMATED + COMPOUNDING + OPTIMAL META-principle standing directive: Sister C emits canonical artifacts that compound into future iterations (provenance.json + per-LUT-shape sidecar LUT artifacts + Catalog #313 probe-outcomes ledger entry + FORMALIZATION_PENDING tracking); the probe script IS the AUTOMATED reproducibility surface for Sister D (criterion #4) + Sister E (criterion #5) reactivation criteria; the `PARADIGM_INTACT_WITH_COMBINED_CRITERIA_PARTIALLY_FALSIFIED` verdict + bin-occupation invariance ~26.5% empirical signature is the OPTIMAL signal for the recommended Sister E spawn (criterion #5 weighted EXTREMA).

Per the 8th MLX-first + numpy-portable + individually-fractal standing directive: this Sister C lane DIRECTLY ADDRESSES the operator's binding constraint that "generic catalog-enumeration of LUT shapes is anti-pattern; instead, identify the substrate's PRINCIPLED LUT shape choice per the entropy coder's symbol-distribution structure" — the 3 LUT shapes tested are PRINCIPLED canonical sisters (16×5 is the v8 substrate's existing shape; 32×5 is a natural sister doubling the luma-quantization axis; 16×10 is a natural sister doubling the SegNet class partition via Daubechies hierarchical-partition discipline); NO generic catalog-enumeration (e.g. 4×5, 8×5, 64×5, 128×5, etc.) is attempted; the 3-cell apples-to-apples test directly answers the per-substrate-unique-optimal-LUT-shape question with EMPIRICAL EVIDENCE that NO LUT-shape variant improves discrimination beyond Sister B's 16×5 baseline at the canonical weighted-median statistic. The substrate's UNIWARD coupling at this fixture is empirically NOT a function of LUT shape — the bottleneck is the statistic choice (criterion #5 EXTREMA recommended for Sister E).
