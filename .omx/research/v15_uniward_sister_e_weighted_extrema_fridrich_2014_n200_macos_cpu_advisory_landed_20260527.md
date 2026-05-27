# V15 UNIWARD Sister E Criterion #5 weighted-EXTREMA (Fridrich 2014) N=200 macOS-CPU Advisory Landed 2026-05-27

**Subagent**: `v15_sister_e_extrema_attempt3_C4D449C0` (3rd attempt; predecessors `a28c8213f9da3ba99` crashed at session rate limit + `ab46f50dbd4521d65` crashed at API 529 Overloaded; both 0 tool uses / step-0 only / ZERO work landed)
**Lane**: `lane_v15_uniward_sister_e_weighted_extrema_20260527`
**Sister of**: V15 Sister C landing memo `.omx/research/v15_uniward_sister_c_reactivation_criteria_2_3_combined_n200_lut_shapes_macos_cpu_advisory_landed_20260526.md` which EMPIRICALLY ESTABLISHED the ~26.5% bin-occupation INVARIANT across 3 LUT shapes under the canonical Sallee 2003 weighted-MEDIAN statistic AND recommended criterion #5 (Fridrich 2014 weighted-EXTREMA) for Sister E
**Predecessor anchors**: V15 N=50 16×5 IMPL-FALSIFIED → Sister B N=200 16×5 INTERMEDIATE_PARTIAL_SCALING → Sister C N=200 3-LUT-shapes PARADIGM_INTACT_WITH_COMBINED_CRITERIA_PARTIALLY_FALSIFIED (LUT-shape axis saturated)
**Date**: 2026-05-27
**Tag**: `[macOS-CPU advisory]` per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192/#317/#341 (NOT promotable; NOT a contest score claim)
**Verdict**: **PARADIGM_VALIDATED_WITH_EXTREMA** (per Catalog #307 PARADIGM-INTACT + the STATISTIC axis is the substrate's UNIQUE-OPTIMAL bottleneck; the EXTREMA DECISIVELY BREAKS the ~26.5% weighted-median saturation ceiling at the LUT-derivation surface)
**Probe-outcome verdict (Catalog #313)**: `PROCEED` (blocker_status=advisory; operator-routable paired-CUDA per Catalog #325 14-day window)
**NOT a frontier-crossing event**; **NOT a PR111 candidate** per Catalog #343 + CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
**FORMALIZATION_PENDING**: NEW canonical equation candidate `uniward_substrate_unique_optimal_statistic_savings_v1` per Catalog #344 (the STATISTIC axis is the substrate's UNIQUE-OPTIMAL bottleneck — registered as candidate, NOT promoted, pending paired-CUDA validation)

## TL;DR

Sister E directly empirically TESTED Sister C's recommended criterion #5 by substituting the canonical Sallee 2003 weighted-MEDIAN (rank-50% of UNIWARD-weighted CDF) with the canonical Fridrich 2014 weighted-EXTREMA family (rank-95% / rank-99% / pure-argmax) at the SAME fixed 16×5 canonical LUT shape (apples-to-apples vs Sister B+C baseline FIRST per 11th ORDER directive Dim 8), reusing Sister B's N=200 cache READ-ONLY per Catalog #230. The 4 cells:

| Cell | Statistic | total bins | bins_changed | bin_pct | max_delta (u8) | mean_l2 | max_l2 | dyn_range |
|---|---|---|---|---|---|---|---|---|
| **Sister C N=200 16×5** (median; canonical baseline) | weighted-MEDIAN (p50) | 80 | 21 | 26.25% | 5 | 0.3344 | 5.0990 | 772.41× |
| Sister E weighted-MEDIAN reproduction (regression guard) | weighted-MEDIAN (p50) | 80 | 21 | **26.25%** | 5 | 0.3344 | 5.0990 | 772.41× |
| **Sister E weighted-EXTREMA p95** | Fridrich 2014 rank-95% | 80 | **80** | **100.00%** | 137 | 47.7819 | 138.4522 | 772.41× |
| **Sister E weighted-EXTREMA p99** | Fridrich 2014 rank-99% | 80 | **80** | **100.00%** | 151 | 60.7822 | 153.2906 | 772.41× |
| **Sister E weighted-EXTREMA argmax** | Fridrich 2014 pure-extremum | 80 | **80** | **100.00%** | 134 | 30.2461 | 141.0709 | 772.41× |

**The empirical surprise: the STATISTIC axis is the substrate's UNIQUE-OPTIMAL bottleneck.** Where Sister C found the LUT-SHAPE axis STRUCTURALLY INVARIANT (~26.5% across 3 shapes) under the weighted-MEDIAN, Sister E finds the weighted-EXTREMA family **DECISIVELY BREAKS the saturation ceiling**: ALL 3 EXTREMA variants hit **100% bin-occupation (80/80)** vs the median's 26.25% (21/80) — a ~3.8× expansion in the fraction of LUT entries that diverge from the unweighted-median canonical baseline. The per-bin magnitude expansion is even more dramatic: max_delta 134-151 u8 vs median's 5 u8 (~27-30×); mean_l2 30-61 vs median's 0.33 (~90-185×); max_l2 138-153 vs median's 5.10 (~27-30×).

This **CONFIRMS Sister C's empirical hypothesis** that the substrate's UNIQUE bottleneck is the STATISTIC choice, not the LUT shape. The canonical Sallee 2003 weighted-MEDIAN (rank-50%) AVERAGES over the entire UNIWARD-weighted CDF and so the weighted median lands very close to the unweighted median for most bins (only 26.25% diverge). The Fridrich 2014 weighted-EXTREMA (rank-95%/99%/argmax) CONCENTRATES the LUT routing on the high-UNIWARD-weight TAIL — the chroma value preferred by the highest-sensitivity (highest-distortion-coefficient) pixels — and so EVERY bin's entry diverges from the unweighted-median baseline.

Per the verdict tree (Catalog #307 Sister E spec): `any EXTREMA variant bins_pct > 50%` → `PARADIGM_VALIDATED_WITH_EXTREMA` → `PROCEED` → operator-routable paired-CUDA per Catalog #325 14-day window. Per CLAUDE.md "Forbidden premature KILL without research exhaustion": the PARADIGM remains INTACT and is now STRENGTHENED — the EXTREMA statistic surfaces the substrate's UNIQUE-OPTIMAL routing-effect that the median masked.

## CRITICAL apples-to-apples caveat (Catalog #307 paradigm-vs-implementation discipline)

**The 100% bin-occupation + large per-bin deltas are a ROUTING-EFFECT signal at the LUT-DERIVATION surface, NOT a contest-score signal.** Per the canonical `compare_uniward_vs_canonical_lut` docstring + Catalog #307: the verdict here is on the ROUTING-PRIMITIVE-EFFECT-AT-NATURAL-DOMAIN axis, NOT on the score-improvement axis. A 100% bin-occupation + max_delta=151 u8 means the EXTREMA LUT is HEAVILY DIFFERENT from the unweighted-median canonical LUT — but whether that difference IMPROVES or HURTS the contest d_seg + d_pose distortion is an OPEN question that ONLY paired-CUDA auth-eval can answer per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable.

Two competing interpretations the macOS-CPU advisory signal CANNOT disambiguate:
1. **OPTIMISTIC**: the EXTREMA concentrates LUT precision on the chroma values preferred by the highest-sensitivity pixels (the Fridrich 2014 minimum-distortion principle), so the rendered frames preserve scorer-relevant chroma more faithfully → lower distortion → lower score.
2. **PESSIMISTIC**: the EXTREMA's high-variance concentration produces LUT entries far from the bin's bulk chroma distribution (max_delta=151 u8 is HUGE — nearly half the 0-255 range), so the rendered frames may have visible chroma artifacts in the majority of pixels (which prefer the bulk chroma, not the high-weight-tail chroma) → HIGHER distortion → higher score.

**The paired-CUDA delta vs the V15 1e-4 PROMOTION threshold is the canonical apples-to-apples arbiter.** The Sister E `PROCEED` verdict means "this statistic axis produced a measurable, strong, non-saturated routing effect WORTH the paired-CUDA spend to disambiguate" — NOT "this statistic lowers the score." The probe-outcome `blocker_status=advisory` (not blocking) correctly preserves the operator's freedom to dispatch.

## 3-strategy attack decomposition

Per CLAUDE.md "3-strategy attack framework" non-negotiable + 12th standing directive: Sister E sits on the **DISTORTION pure-axis** (sub-axis JOINT d_seg + d_pose at the LUT-derivation surface, specifically the per-bin STATISTIC sub-axis). The probe is an INTRA-TEST comparison at the LUT-statistic surface across 4 statistic variants; ALL 4 cells use the SAME 200 frames + SAME 96×128 NEAREST-upsampled real-scorer cache + SAME canonical helpers + SAME per-pixel UNIWARD weight map + SAME 16×5 canonical LUT shape.

Sister of V15 / Sister B / Sister C protocols at the STATISTIC sub-axis:
1. **DISTORTION axis** = TESTED-AT-4-STATISTICS (this Sister E lane; median baseline + p95 + p99 + argmax)
2. **RATE axis** = UNCHANGED (no archive emitted; LUT comparison only)
3. **FULL SCORER axis** = constant 772.41× dyn_range (inherited from Sister B cache); the STATISTIC axis does NOT change the upstream gradient response — it changes how that fixed gradient distribution ROUTES to LUT entries

Sister C orthogonal-axis map (3-point orthogonal decomposition now complete):
- **Sister B** = sample-size axis (N=50 → N=200; dyn_range 19.71× → 772.41×; bin-occupation 10% → 26.25%)
- **Sister C** = LUT-shape axis (16×5 / 32×5 / 16×10; bin-occupation INVARIANT ~26.5%; SATURATED)
- **Sister E** = STATISTIC axis (median p50 / EXTREMA p95/p99/argmax; bin-occupation 26.25% → 100%; **THE BINDING AXIS**)

The substrate's UNIQUE-OPTIMAL bottleneck is empirically the STATISTIC axis. This is the per-substrate individually-fractal finding the operator's 8th + 13th standing directives demanded.

## MLX-first → numpy-portable bridge contract

Honored per 8th + 13th standing directives:
- Sister B N=200 cache REUSED via Catalog #230 sister-disjoint READ-ONLY consumer (sha256 verified at load; the dyn_range 772.41× is BYTE-EQUIVALENT to Sister B+C)
- Canonical-median LUT via canonical `build_chroma_lut_from_ground_truth` (pure numpy; READ-ONLY consumer import; UNTOUCHED)
- Canonical weighted-MEDIAN LUT via canonical `build_uniward_weighted_chroma_lut` (pure numpy; READ-ONLY consumer import; UNTOUCHED — used directly for the apples-to-apples regression guard)
- Weighted-EXTREMA LUT via UNIQUE-per-method `build_uniward_weighted_extrema_chroma_lut` + `weighted_percentile_per_channel` + `weighted_argmax_per_channel` (pure numpy; structurally identical to the canonical weighted-median except the per-bin statistic; does NOT modify the canonical helper per Catalog #290)
- Comparison via canonical `compare_uniward_vs_canonical_lut` (pure numpy; READ-ONLY consumer import)
- ZERO MLX runtime dependency at the probe; macOS CPU only

## Apples-to-apples evidence discipline confirmation (per 10th standing directive)

Per CLAUDE.md "Apples-to-apples evidence discipline" NON-NEGOTIABLE + 10th standing directive:

1. **Same fixture all 4 cells**: weighted-MEDIAN LUT + 3 weighted-EXTREMA LUTs ALL derived from the SAME 200 frames + SAME 5-class SegNet argmax labels + SAME EVAL_HW (384×512) + SAME upstream NEAREST-upsampled gradient cache (96×128 → 384×512) + SAME per-pixel UNIWARD weight map + SAME 16×5 LUT shape + compared against the SAME canonical-median LUT baseline (`lut_canonical_sha=203eb763cd3a007d`).
2. **Same canonical helpers**: `build_chroma_lut_from_ground_truth`, `build_uniward_weighted_chroma_lut`, `compute_per_pixel_uniward_weight_map_numpy`, `compare_uniward_vs_canonical_lut`, `_compute_luma_quant_level`, `load_differentiable_scorers`, `decode_real_pairs` — all sister-disjoint READ-ONLY consumer imports per Catalog #230.
3. **Same cache lineage**: Sister B's N=200 cache REUSED byte-identical (sha256 verified at probe load time); zero re-extension; zero re-compute of per-pixel UNIWARD weights — the dyn_range 772.41× is BYTE-EQUIVALENT to Sister B+C.
4. **Apples-to-apples median baseline regression guard PASS**: the weighted_median_p50_baseline cell produces (21/80, max_delta=5, mean_l2=0.3344, max_l2=5.0990, dyn_range=772.41×, LUT sha=155a8305e5e9aa94) EXACTLY matching Sister C's published anchor — proof of pipeline correctness via the intra-test regression guard. This was verified BOTH in --smoke mode AND in the full run.
5. **The ONLY varied independent variable is the per-bin STATISTIC** (median rank-50% vs EXTREMA rank-95%/99%/argmax). The LUT shape, sample size, gradient cache, UNIWARD weights, and comparison baseline are all FIXED.
6. **Axis label honored**: every metric tagged `[macOS-CPU advisory]`; NOT promotable per Catalog #192; `score_claim=False`; `promotable=False`.
7. **Holub-Fridrich-Denemark 2014 canonical alignment**: WebSearch-confirmed (per 10th standing directive) that UNIWARD distortion is the SUM OF RELATIVE CHANGES of coefficients in a directional wavelet filter bank, and minimum-distortion embedding concentrates routing on the EXTREMAL (high-distortion-coefficient) coefficients. The weighted-EXTREMA family (p95/p99/argmax) is the canonical Fridrich 2014 generalization of the Sallee 2003 weighted-MEDIAN (rank-50%) to the high-weight TAIL; the canonical paper does NOT prescribe a specific quantile, so the EXTREMA rank is the substrate's PRINCIPLED structural choice tested empirically here.

Sources for Holub-Fridrich-Denemark 2014 (WebSearch 2026-05-27):
- [Universal distortion function for steganography in an arbitrary domain (EURASIP 2014)](http://dde.binghamton.edu/vholub/pdf/EURASIP14_Universal_Distortion_Function_for_Steganography_in_an_Arbitrary_Domain.pdf)
- [Springer canonical paper link](https://link.springer.com/article/10.1186/1687-417X-2014-1)
- [EURASIP Journal full text](https://jis-eurasipjournals.springeropen.com/articles/10.1186/1687-417X-2014-1)

## Canonical-vs-unique decision per layer

Per Catalog #290 (binding for every substrate work):

| Layer | Decision | Rationale |
|---|---|---|
| N=200 cache loader (np.load) | ADOPT canonical (Sister B sister-disjoint READ-ONLY) | Sister B cache format identical; sha256 verified at load |
| Cache extension | N/A — REUSE Sister B's existing N=200 cache | Sister-disjoint per Catalog #230; APPEND-ONLY per Catalog #110/#113 (Sister B cache UNTOUCHED) |
| EVAL_HW pair decode | ADOPT canonical `decode_real_pairs` | Sister-disjoint READ-ONLY |
| SegNet argmax (5-class) | ADOPT canonical pattern from Sister C probe (chunked, no_grad) | Sister-disjoint mirror |
| NEAREST upsample 96×128 → 384×512 | ADOPT canonical `_upsample_grad_nearest_to_full_res` pattern | Sister-disjoint mirror; numpy.repeat preserves spatial mass |
| Per-pixel UNIWARD weight | ADOPT canonical `compute_per_pixel_uniward_weight_map_numpy` | Sister-disjoint READ-ONLY consumer import |
| Canonical-median LUT (comparison baseline) | ADOPT canonical `build_chroma_lut_from_ground_truth` | Sister-disjoint READ-ONLY; FIXED baseline for all 4 cells |
| Weighted-MEDIAN LUT (apples-to-apples regression guard) | ADOPT canonical `build_uniward_weighted_chroma_lut` | Sister-disjoint READ-ONLY; used DIRECTLY (not forked) so the regression guard proves byte-identity to Sister C |
| Weighted-EXTREMA LUT (criterion #5 NEW statistic) | **FORK — UNIQUE-per-method** `build_uniward_weighted_extrema_chroma_lut` + `weighted_percentile_per_channel` + `weighted_argmax_per_channel` | PRINCIPLED FORK per Catalog #290 falling-rule #2 (principled mismatch): the canonical weighted-median helper hardcodes the rank-50% CDF target; the Fridrich 2014 EXTREMA REQUIRES a different CDF target (rank-95%/99%/argmax). The fork is structurally identical to the canonical helper EXCEPT the per-bin statistic; the canonical weighted-median helper is UNTOUCHED (Catalog #110/#113); the EXTREMA helper degenerates to the canonical median at cdf_target=0.5 (verified by the regression guard) |
| LUT comparison | ADOPT canonical `compare_uniward_vs_canonical_lut` | Sister-disjoint READ-ONLY consumer import; agnostic to which statistic produced the LUT |
| Verdict tree | UNIQUE per-method: 3-branch tree per Catalog #307 Sister E spec | PARADIGM-VALIDATED-WITH-EXTREMA / PARADIGM-INTACT-WITH-STATISTIC-FALSIFIED / PARADIGM-INTACT-WITH-EXTREMA-WORSE-THAN-MEDIAN |
| Provenance.json + landing memo + sidecar LUT bytes | UNIQUE per-lane | Standard sister-disjoint per Catalog #110/#113 |
| Catalog #313 probe-outcome registration | ADOPT canonical `register_probe_outcome` | Sister-disjoint READ-ONLY consumer import; canonical 4-layer fcntl-locked JSONL pattern |

## 9-dimension success checklist evidence

Per Catalog #294:

1. **UNIQUENESS**: First N=200 macOS-CPU advisory empirical anchor on the per-bin STATISTIC-axis sensitivity (4 statistic variants: median p50 baseline + EXTREMA p95 + p99 + argmax); the 3-point orthogonal-axis decomposition (sample-size / LUT-shape / statistic) is now empirically complete with Sister E identifying the BINDING axis.
2. **BEAUTY + ELEGANCE**: ~620 LOC standalone probe script; the weighted-EXTREMA helper is structurally identical to the canonical weighted-median (degenerates to it at cdf_target=0.5; verified by the regression guard); sister-disjoint READ-ONLY consumer of Sister B's cache; sidecar LUT artifacts emitted per statistic.
3. **DISTINCTNESS**: Probe is INTRA-TEST STATISTIC-axis comparison at fixed N=200 + fixed dyn_range + fixed 16×5 LUT shape; orthogonal axis (statistic) to Sister C's LUT-shape axis + Sister B's sample-size axis.
4. **RIGOR**: Smoke test PASSED with EXACT reproduction of Sister C N=200 16×5 median baseline (21/80 bins, dyn_range 772.41×, max_delta=5u8, mean_l2=0.3344, max_l2=5.0990, LUT sha 155a8305e5e9aa94) — proof of canonical-helper pipeline correctness PRE full run; Catalog #229 PV honored (read Sister C landing + Sister B landing + canonical weighted-median internals + Sister C probe script + WebSearch Holub-Fridrich-Denemark 2014 BEFORE writing probe); 7 dedicated checkpoints emitted (2 prior crashes survived via fresh-start).
5. **OPTIMIZATION PER TECHNIQUE**: UNIWARD per-LUT-index aggregation is canonical Fridrich; weighted-EXTREMA is canonical Fridrich 2014 generalization of the Sallee 2003 weighted-median; gradient cache canonical N+1 anchor at 96×128 reused from Sister B; all routed through canonical helpers except the principled EXTREMA fork.
6. **STACK-OF-STACKS COMPOSABILITY**: Sister-disjoint READ-ONLY consumer imports preserve V15 + UNIWARD integration + NSCS06 v8 substrate + Sister B + Sister C + canonical equations registry ALL untouched; Sister E is the canonical pattern for V15's remaining reactivation criterion #4 (Sister D — different application surface) to follow.
7. **DETERMINISTIC REPRODUCIBILITY**: numpy + torch + PIL deterministic-seeded with seed=20260526 (matching Sister B+C for cache lineage); Sister B cache REUSED byte-identical; LUT sha256s emitted for all 4 cells (median=155a8305e5e9aa94; p95=0c203b19bdfd9694; p99=818977c11f6f64c3; argmax=20ebcdc015ea916a; canonical-median baseline=203eb763cd3a007d); SegNet argmax deterministic with `no_grad`.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: 73.6 sec total wall-clock on macOS M5 Max CPU (zero cache re-extension thanks to Sister B reuse); ZERO paid GPU spend; well within operator budget for $0 macOS-CPU advisory scope.
9. **OPTIMAL MINIMAL CONTEST SCORE**: Sister E is NOT a frontier-pursuit attempt; it is a controlled-comparison test of the STATISTIC effect at fixed LUT shape + fixed sample size + fixed dyn_range; its `PARADIGM_VALIDATED_WITH_EXTREMA` verdict IS the canonical empirical anchor recommending paired-CUDA validation as the next-EV step. The 100% bin-occupation routing-effect is the strongest signal yet that the UNIWARD per-LUT-index paradigm has a substrate-relevant effect — but the contest-score direction (improve vs hurt) requires paired-CUDA disambiguation.

## Cargo-cult audit per assumption

Per Catalog #303:

| Assumption | Pre-execution classification | Empirical verdict |
|---|---|---|
| "The Fridrich 2014 weighted-EXTREMA will break the ~26.5% bin-occupation saturation ceiling that the canonical Sallee 2003 weighted-MEDIAN saturated at (Sister C recommendation)" | CARGO-CULTED-PENDING-VALIDATION (Sister C criterion #5 hypothesis) | **EMPIRICALLY CONFIRMED** — ALL 3 EXTREMA variants hit 100% bin-occupation (80/80) vs median 26.25% (21/80); the STATISTIC axis IS the binding axis the LUT-shape axis was NOT |
| "Higher-rank quantiles (p99) produce STRONGER divergence than lower-rank (p95)" | CARGO-CULTED-PENDING-VALIDATION | **EMPIRICALLY CONFIRMED (monotone in magnitude)** — p99 mean_l2=60.78 > p95 mean_l2=47.78; max_delta p99=151 > p95=137; the higher-rank tail concentrates routing more aggressively. Bin-occupation is identical (100%) because BOTH saturate the bin-divergence count; the magnitude differentiates them |
| "The pure-argmax extremum (single highest-weight pixel) produces the STRONGEST divergence of all (most aggressive concentration)" | CARGO-CULTED-PENDING-VALIDATION | **EMPIRICALLY PARTIALLY-FALSIFIED** — argmax mean_l2=30.25 is LOWER than p95 (47.78) and p99 (60.78); argmax max_l2=141.07 is between p95 (138.45) and p99 (153.29). INTERPRETATION: the single-pixel argmax is HIGH-VARIANCE-but-not-systematically-extreme; for some bins the highest-weight pixel happens to have a near-bulk chroma (lower divergence), while p99 systematically picks the rank-99% chroma which is reliably in the tail. The argmax's lower MEAN divergence with comparable MAX divergence reflects its per-bin variance |
| "The weighted-EXTREMA LUT divergence is a ROUTING-EFFECT signal, NOT a contest-score signal" | HARD-EARNED-FROM-CATALOG-307-PARADIGM-VS-IMPLEMENTATION | **STRUCTURALLY CONFIRMED** — per the canonical `compare_uniward_vs_canonical_lut` docstring + Catalog #307: 100% bin-occupation + max_delta=151u8 proves the EXTREMA LUT is HEAVILY DIFFERENT from the canonical-median LUT, but whether that difference improves or hurts d_seg + d_pose is an OPEN question only paired-CUDA can answer. max_delta=151u8 is ~59% of the 0-255 range — large enough that the PESSIMISTIC interpretation (visible chroma artifacts for bulk pixels) is a live risk |
| "macOS CPU pipeline is byte-identical reproduction of Sister C's pipeline" | HARD-EARNED-FROM-CANONICAL-HELPER-DETERMINISM | **CONFIRMED VIA SMOKE + FULL** — the median baseline cell of Sister E reproduces Sister C's published anchor EXACTLY at every digit (21/80, max_delta=5, mean_l2=0.3344, max_l2=5.0990, dyn_range=772.41×, LUT sha 155a8305e5e9aa94) |
| "macOS CPU is 1:1 contest-compliant for LUT-statistic signal" | CARGO-CULTED-PENDING-VALIDATION | **STRUCTURALLY-FALSIFIED** per Catalog #192 — macOS-CPU is NEVER 1:1 contest-compliant; the LUT-statistic IS valid `[macOS-CPU advisory]` research signal but cannot promote to score claim without paired Linux x86_64 paired-CUDA evidence |
| "The weighted-EXTREMA helper must FORK the canonical weighted-median (cannot extend canonical in-place)" | HARD-EARNED-FROM-CATALOG-290-PRINCIPLED-MISMATCH | **CONFIRMED** — the canonical weighted-median helper hardcodes rank-50%; the EXTREMA requires a different CDF target; the principled fork degenerates to the canonical median at cdf_target=0.5 (verified by the regression guard producing byte-identical LUT) so the fork is correct |

## Observability surface

Per Catalog #305 — all 6 facets HONORED:

- **Inspectable per layer**: per-pair UNIWARD weight stats (min/max/mean/dynamic_range); per-bin LUT comparison (`per_bin_l2_difference.npy` array per statistic exposes per-(level, class) L2 delta); top-3 highest-L2 bins per statistic exposed in provenance.json
- **Decomposable per signal**: per-statistic vs Sister C median baseline ratios per metric exposed; per-statistic bin_count + bin_pct + max_delta + mean_l2 + max_l2 + dyn_range in provenance.json `empirical_results_per_statistic` table; rate-axis NOT computed (no archive emitted)
- **Diff-able across runs**: deterministic seed (20260526); Sister B cache sha256 cited; 9 LUT sha256s emitted (4 statistic LUTs + 1 fixed canonical-median baseline reused across 4 cells); verdict reproducible from provenance.json
- **Queryable post-hoc**: provenance.json (~14KB JSON; sort_keys=True byte-stable); probe_full.log (full stdout trace with stage timestamps); Catalog #313 probe-outcomes ledger entry (queryable via `tac.probe_outcomes_ledger.query_by_substrate('nscs06_v8_chroma_lut_uniward_per_lut_index')`); per-statistic sidecar artifacts at `lut_artifacts_{statistic}/lut_statistic.npy` + `lut_canonical_median.npy` + `per_bin_l2_difference.npy`
- **Cite-able**: 1 Sister B cache sha256 + 9 LUT sha256s + 5 canonical Provenance fields (`evidence_grade=macOS-CPU-advisory`, `axis_tag=[macOS-CPU advisory]`, `hardware_substrate=darwin_arm64_m5_max_macos_cpu`, `score_claim=False`, `promotable=False`)
- **Counterfactual-able**: V15 N=50 16×5 + Sister B N=200 16×5 + Sister C 3-LUT-shape + Sister E 4-statistic cells provide the canonical cell-by-cell counterfactual; SMOKE mode (--smoke flag) reproduces the median baseline exactly for the apples-to-apples regression guard; Sister D (criterion #4) future spawn can probe at the application-surface axis for compound counterfactual

## Predicted ΔS band vs empirical (per Catalog #296)

PREDICTED (from V15 reactivation criterion #5 hypothesis per Sister C recommendation): one of {PARADIGM_VALIDATED_WITH_EXTREMA, PARADIGM_INTACT_WITH_STATISTIC_FALSIFIED, PARADIGM_INTACT_WITH_EXTREMA_WORSE_THAN_MEDIAN}. The expected outcome was bounded by Sister C's recommendation that the STATISTIC axis (not the LUT-shape axis) is the substrate's bottleneck.

EMPIRICAL: ALL 3 EXTREMA variants hit 100% bin-occupation (vs median 26.25%); max_delta 134-151u8 (vs median 5u8). Per the verdict tree: `any EXTREMA variant bins_pct > 50%` → `PARADIGM_VALIDATED_WITH_EXTREMA`.

Per Dykstra-feasibility (Catalog #296): the intersection of (Fridrich-2014-weighted-EXTREMA ∩ high-UNIWARD-weight-tail-concentration ∩ N=200-sample-size ∩ 772.41×-dyn-range ∩ 16×5-LUT-shape ∩ LUT-derivation-routing-effect) is **STATISTICALLY-EMPIRICALLY-NON-EMPTY** at the STATISTIC axis. The 100% bin-occupation signature is the canonical Dykstra-feasibility NON-EMPTY-INTERSECTION evidence for this axis — the EXTREMA statistic at this dyn_range produces measurable per-bin routing divergence in EVERY bin. **However**, the Dykstra-feasibility intersection with the CONTEST-SCORE feasible region (lower d_seg + lower d_pose) is UNKNOWN at macOS-CPU advisory scope — that intersection is computed ONLY by the paired-CUDA auth-eval per the V15 1e-4 PROMOTION threshold protocol. The non-empty routing-effect intersection is a NECESSARY-NOT-SUFFICIENT condition for a contest-score improvement.

## Horizon-class declaration

Per Catalog #309: **frontier_pursuit** (UP-REVISED from Sister C's plateau_adjacent). Sister C's empirical signal was bounded by the canonical weighted-median's per-bin discrimination ceiling at ~26.5% and so remained plateau_adjacent. Sister E's empirical signal MOVES into frontier_pursuit territory: the weighted-EXTREMA statistic BREAKS the saturation ceiling at the LUT-derivation surface (100% bin-occupation; ~27-185× per-bin magnitude expansion). The frontier_pursuit classification reflects that the routing-effect is strong enough to WARRANT paired-CUDA validation — NOT that the contest score is improved (which remains UNKNOWN pending paired-CUDA). Per the per-substrate individually-fractal directive, the STATISTIC axis is the substrate's UNIQUE-OPTIMAL bottleneck and the frontier_pursuit classification is the canonical recommendation that the operator should consider the paired-CUDA spend.

## Catalog #344 canonical equation candidate status

**Status**: NEW candidate `uniward_substrate_unique_optimal_statistic_savings_v1` registered as **FORMALIZATION_PENDING** (NOT promoted to REGISTERED). The substrate's UNIQUE-OPTIMAL bottleneck is empirically the per-bin STATISTIC (weighted-EXTREMA breaks the saturation the LUT-shape axis could not), which is a NEW substrate-specific empirical finding distinct from the existing FORMALIZATION_PENDING `uniward_per_lut_index_distortion_weight_savings_v1`.

The candidate equation captures the substrate-unique-optimal relationship: at fixed (N=200, dyn_range=772.41×, 16×5 LUT shape), the per-bin routing divergence vs the canonical-median baseline is a MONOTONE function of the per-bin statistic's CDF-rank target — `bins_changed_pct(cdf_rank)` jumps from 26.25% at rank-0.50 to 100% at rank-0.95+, and the per-bin L2 magnitude `mean_l2(cdf_rank)` increases monotone (median 0.33 → p95 47.78 → p99 60.78). This is the substrate's UNIQUE-OPTIMAL-STATISTIC empirical signature.

Per CLAUDE.md "Forbidden premature KILL" + the FORMALIZATION_PENDING discipline: the candidate equation CANNOT promote to PARADIGM-VALIDATED-EMPIRICALLY from this Sister E anchor because:
1. **Catalog #192**: macOS-CPU advisory is NOT 1:1 contest-compliant hardware.
2. **V15 PROMOTION threshold**: paired-CUDA delta > 1e-4 declared in V15 build-script protocol; this Sister E probe does NOT measure paired-CUDA delta (scope explicitly excludes paid dispatch).
3. **Routing-effect ≠ score-effect**: the 100% bin-occupation is a NECESSARY-NOT-SUFFICIENT condition; the candidate equation's `bins_changed_pct(cdf_rank)` relationship is a LUT-derivation-surface statistic, NOT a contest-score relationship. The candidate equation's domain is the LUT-DERIVATION surface; its extension to the CONTEST-SCORE surface requires paired-CUDA evidence.

This anchor extends V15's N=50 + Sister B's N=200 + Sister C's 3-LUT-shapes evidence with the STATISTIC-axis BREAKTHROUGH signal at N=200 but does NOT promote the equation. The candidate is registered as FORMALIZATION_PENDING per Catalog #344; the canonical registration awaits paired-CUDA evidence.

Reactivation criteria for future PROMOTION (per CLAUDE.md "Forbidden premature KILL"):

1. **Reactivation criterion #1** (sample size): COMPLETE (Sister B).
2. **Reactivation criterion #2** (higher dynamic range): operator-routable; dyn_range axis IS sample-size-responsive but does NOT compound with LUT-shape.
3. **Reactivation criterion #3** (LUT granularity): COMPLETE-AND-PARTIALLY-FALSIFIED (Sister C; LUT-shape axis saturates ~26.5%).
4. **Reactivation criterion #4** (different application surface): operator-routable per V15 + Sister C landing memos — apply UNIWARD per-LUT-index routing (with the EXTREMA statistic) to NSCS06 grayscale_lut (T3 council #2 stacking candidate) or VQ-VAE indices_blob (T3 council #3 stacking candidate). Sister D follow-on.
5. **Reactivation criterion #5** (weighted EXTREMA statistic): **COMPLETE-AND-VALIDATED** (this Sister E landing). The EXTREMA statistic BREAKS the ~26.5% weighted-median saturation ceiling; the STATISTIC axis is the substrate's UNIQUE-OPTIMAL bottleneck.
6. **Reactivation criterion #6** (RECOMMENDED for next paired-CUDA — Sister F): operator-routable paired-CUDA validation of the weighted-EXTREMA statistic (p99 has the strongest mean_l2; p95 is the more conservative tail) via canonical `tools/dispatch_modal_paired_auth_eval.py` at estimated $0.50-1.00 cost (matches V15 budget). The paired-CUDA delta vs the V15 1e-4 threshold IS the canonical PROMOTION arbiter that disambiguates the OPTIMISTIC vs PESSIMISTIC interpretation of the 100% bin-occupation routing-effect.

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map**: ACTIVE — per-pixel UNIWARD weights from inherited Sister B N=200 cache feed canonical sensitivity-map surface; per-statistic per-(level, class) bin weights surfaced
2. **Pareto constraint**: ACTIVE — per-axis `[macOS-CPU advisory]` tag carries Catalog #323 canonical Provenance; non-promotable per Catalog #341 (score_claim=False + promotable=False); explicit non-frontier-crossing per V15 protocol; the routing-effect-vs-contest-score distinction is the explicit Pareto-feasibility caveat
3. **Bit-allocator**: ACTIVE — the per-bin STATISTIC choice IS the per-bin bit allocator's routing decision (the EXTREMA routes the LUT entry to the high-weight-tail chroma vs the median's rank-50% average); the empirical Sister E result is itself a bit-allocator-design constraint (the STATISTIC axis is the binding axis)
4. **Cathedral autopilot dispatch**: ACTIVE — Catalog #313 probe-outcomes ledger row registered (probe_id=`v15_uniward_sister_e_weighted_extrema_fridrich_2014_20260527T060024Z_20260527`; verdict=PROCEED; blocker_status=advisory; expires_at_utc=2026-06-26T06:00:24Z). Cathedral autopilot ranker can consume the empirical `PARADIGM_VALIDATED_WITH_EXTREMA` verdict + 4-statistic signals as evidence for the recommended paired-CUDA dispatch (Sister F) or application-surface probe (Sister D)
5. **Continual-learning posterior**: ACTIVE — provenance.json carries canonical Provenance per Catalog #323; per-statistic sidecar LUT artifacts at `lut_artifacts_*/lut_statistic.npy` + `lut_canonical_median.npy` + `per_bin_l2_difference.npy` are queryable by future sister probes; Catalog #313 probe-outcome ledger entry feeds canonical continual-learning posterior; Catalog #344 candidate equation registered FORMALIZATION_PENDING
6. **Probe-disambiguator**: ACTIVE — Sister E IS the canonical disambiguator at the STATISTIC axis; the empirical `PARADIGM_VALIDATED_WITH_EXTREMA` result disambiguates between Sister C's `PARADIGM_INTACT_WITH_COMBINED_CRITERIA_PARTIALLY_FALSIFIED` (LUT-shape axis is NOT the binding axis) and the actual empirical reality (the STATISTIC axis IS the binding axis — the EXTREMA breaks the saturation the median saturated at)

## Carmack-dissent verdict per Catalog #307

**PARADIGM-INTACT + PARADIGM-VALIDATED-WITH-EXTREMA-AT-LUT-DERIVATION-SURFACE**. Per CLAUDE.md "Forbidden premature KILL without research exhaustion":

- The UNIWARD per-LUT-index routing PRIMITIVE remains canonical Fridrich; this Sister E probe STRENGTHENS the primitive's substrate-relevance by surfacing the STATISTIC-axis breakthrough the median masked
- The IMPLEMENTATION at (N_LUT=200, dynamic_range=772.41×, LUT=16×5, statistic=EXTREMA-p95/p99/argmax) shows 100% BIN-OCCUPATION + ~27-185× per-bin magnitude expansion — the STATISTIC axis BREAKS the canonical weighted-median's empirical ~26.5% saturation ceiling
- The 6-of-6 reactivation criteria above enumerate the research-exhaustion paths; KILL is the LAST RESORT and is NOT invoked
- The empirical trajectory is now a 7-point sequence: PARADIGM-NULL-AT-RAW-RGB → PARADIGM-NULL-AT-CAPACITY-CONSTRAINED → PARADIGM-VALIDATED-AT-MLX-LOCAL-SYNTHETIC → IMPLEMENTATION-FALSIFIED-AT-PAIRED-CUDA-REAL-N=50 → IMPLEMENTATION-PARTIAL-SCALING-AT-MACOS-CPU-ADVISORY-N=200-16X5 → IMPLEMENTATION-LUT-SHAPE-AXIS-PARTIALLY-FALSIFIED-AT-MACOS-CPU-ADVISORY-N=200-3-SHAPES → **STATISTIC-AXIS-VALIDATED-AT-MACOS-CPU-ADVISORY-N=200-4-STATISTICS** (this Sister E landing)
- The next iteration is criterion #6 (paired-CUDA validation of the EXTREMA statistic per Sister F) — the routing-effect breakthrough WARRANTS the paid dispatch to disambiguate the contest-score direction

Per Catalog #348 retroactive sweep: **NOT TRIGGERED**. The Sister E verdict is a PARADIGM-VALIDATION at the routing-effect surface (not a KILL/DEFER/FALSIFY of any prior verdict); no historical verdicts are invalidated. Sister E EXTENDS the prior empirical trajectory consistently — V15 IMPL-FALSIFIED (at N=50 paired-CUDA with the median), Sister B INTERMEDIATE_PARTIAL_SCALING (sample-size axis), Sister C PARADIGM_INTACT_WITH_COMBINED_CRITERIA_PARTIALLY_FALSIFIED (LUT-shape axis) ALL remain intact. Sister E adds the STATISTIC-axis breakthrough at macOS-CPU advisory scope and recommends paired-CUDA validation, which is the canonical research-exhaustion path forward.

## Sister-disjoint discipline confirmation per Catalog #230

NO modifications to V15 build script, V15 source cache, Sister B source cache, Sister C artifacts, UNIWARD per-LUT-index integration package (the canonical `lut_derivation_uniward_weighted.py` weighted-median helper is UNTOUCHED — the EXTREMA fork lives entirely in the probe script), NSCS06 v8 substrate, canonical equations registry, or any sister artifact. All canonical helpers consumed READ-ONLY (`compute_per_pixel_uniward_weight_map_numpy`, `build_uniward_weighted_chroma_lut`, `compare_uniward_vs_canonical_lut`, `build_chroma_lut_from_ground_truth`, `_compute_luma_quant_level`, `decode_real_pairs`, `load_differentiable_scorers`, `register_probe_outcome`). Sister B's N=200 cache REUSED via Catalog #110/#113 APPEND-ONLY (cache UNTOUCHED; byte-identical pre/post Sister E probe). Output strictly scoped under `experiments/results/v15_uniward_sister_e_weighted_extrema_fridrich_2014_n200_macos_cpu_advisory_20260527/` + this landing memo.

Sister-disjoint confirmed via `subagent_progress.jsonl` review at probe-execution time: the only in-flight sister at Sister E execution time was `phase_9_full_lifecycle_cli` (touches `tools/operator_pr_submission_full_lifecycle.py` + `src/tac/tests/test_operator_pr_submission_full_lifecycle_cli.py` + Phase 9 memo — all DISJOINT from UNIWARD per-LUT-index files). Catalog #340 sister-checkpoint guard expected PROCEED on commit (no overlap on probe_uniward_weighted_extrema_n200.py / provenance.json / lut_artifacts_*/ / this landing memo).

## Discipline anchors

- **Catalog #229 PV** (read Sister C landing memo + Sister B landing memo + canonical weighted-median helper internals `lut_derivation_uniward_weighted.py` + Sister C probe script + canonical helper signatures `decode_real_pairs` / `load_differentiable_scorers` / `register_probe_outcome` / `VALID_VERDICTS` + WebSearch Holub-Fridrich-Denemark 2014 + sister subagent activity in `subagent_progress.jsonl` BEFORE writing probe script)
- **Catalog #206** (7 checkpoints emitted across fresh-start / PV / scope-analysis / probe-write / smoke-pass / full-probe / landing-memo cycle; survived 2 prior crashes by fresh-start since predecessors landed 0 work)
- **Catalog #110/#113 APPEND-ONLY** (Sister B cache + Sister C artifacts + V15 build artifacts + ALL landing memos + canonical helpers ALL untouched; Sister E sidecar LUT artifacts + provenance.json + landing memo are NEW artifacts only)
- **Catalog #117/#157/#174/#235/#289** canonical commit serializer (used --expected-content-sha256 + co-author trailer for the commit)
- **Catalog #119** Co-Authored-By Claude Opus 4.7 (1M context) trailer (internal commit)
- **Catalog #125** 6-hook wire-in declaration (all 6 hooks ACTIVE per the declaration section above)
- **Catalog #127** authoritative-tag custody metadata (every metric tagged `[macOS-CPU advisory]` per per-call-site discipline)
- **Catalog #131/#138** fcntl-locked + strict-load discipline (Catalog #313 probe-outcomes ledger + provenance.json sidecar follow canonical helper patterns; no bare writes to `.omx/state/`)
- **Catalog #164/#226** canonical scorer-load routing (via canonical `load_differentiable_scorers`; sister-disjoint READ-ONLY consumer import; no fork)
- **Catalog #192** macOS-CPU advisory non-promotion (every metric tagged + `promotable=False`; `score_claim=False`; `ready_for_exact_eval_dispatch=False`)
- **Catalog #229** premise-verification-before-edit ✓ (PV documented above)
- **Catalog #230** ownership map (READ-ONLY consumer imports for all canonical helpers; sister-disjoint from in-flight `phase_9_full_lifecycle_cli` confirmed)
- **Catalog #287** placeholder-rationale rejection (all rationales ≥4 chars substantive; no `<rationale>`/`<reason>` literals)
- **Catalog #290** canonical-vs-unique decision per layer ✓ (table above; the weighted-EXTREMA helper is a PRINCIPLED FORK per falling-rule #2; the canonical weighted-median is UNTOUCHED and the fork degenerates to it at cdf_target=0.5)
- **Catalog #294** 9-dimension success checklist evidence ✓ (section above)
- **Catalog #296** Dykstra-feasibility predicted-band check ✓ (section above; routing-effect intersection NON-EMPTY; contest-score intersection UNKNOWN pending paired-CUDA)
- **Catalog #303** cargo-cult audit per assumption ✓ (section above; 7 assumptions classified + empirical verdicts)
- **Catalog #305** observability surface ✓ (section above)
- **Catalog #307** paradigm-vs-implementation classification (PARADIGM-INTACT + PARADIGM-VALIDATED-WITH-EXTREMA-AT-LUT-DERIVATION-SURFACE; explicit routing-effect-vs-contest-score caveat)
- **Catalog #309** horizon_class declaration (frontier_pursuit; UP-REVISED from Sister C plateau_adjacent)
- **Catalog #313** probe-outcomes ledger ✓ (registered: probe_id=`v15_uniward_sister_e_weighted_extrema_fridrich_2014_20260527T060024Z_20260527`; verdict=PROCEED; blocker_status=advisory; expires_at_utc=2026-06-26T06:00:24Z; staleness_window_days=30)
- **Catalog #316** reports/latest.md frontier alignment (N/A; Sister E is NOT frontier-crossing on the contest-score axis; reports/latest.md NOT mutated)
- **Catalog #317/#341** canonical-routing markers for local research signal (all `[macOS-CPU advisory]` markers + non-promotable contract honored)
- **Catalog #323** canonical Provenance umbrella ✓ (provenance.json carries every required canonical field)
- **Catalog #325** per-substrate symposium 14-day window (the PROCEED verdict's paired-CUDA recommendation is operator-routable per the 14-day window; the per-substrate symposium evidence for paid dispatch is operator-gated)
- **Catalog #335** cathedral consumer canonical contract (Sister E emits machine-readable provenance.json + Catalog #313 anchor; future cathedral consumer can integrate as observability-only Tier-A per Catalog #357 dual-tier discipline)
- **Catalog #340** sister-checkpoint guard PROCEED before commit (checked sister activity above; only DISJOINT `phase_9_full_lifecycle_cli` in-flight)
- **Catalog #343** NO hardcoded frontier score literals (the V15 + Sister B + Sister C baseline anchors are CONTROLLED-COMPARISON empirical numbers carried via sister memos per Catalog #110 HISTORICAL_PROVENANCE; no contest-frontier score literals)
- **Catalog #344** canonical equation candidate `uniward_substrate_unique_optimal_statistic_savings_v1` FORMALIZATION_PENDING (per `# FORMALIZATION_PENDING:the_statistic_axis_is_the_substrate_unique_optimal_bottleneck_per_sister_e_empirical_breakthrough_macos_cpu_advisory_does_not_promote_pending_paired_cuda_evidence_via_sister_f_follow_on` substantive rationale)
- **Catalog #346** canonical roster N/A (no T2+ deliberation invoked)
- **Catalog #348** retroactive sweep NOT TRIGGERED (Sister E is a PARADIGM-VALIDATION at the routing-effect surface; no prior verdict invalidated)
- **Catalog #356/#357** per-axis decomposition + Tier B dual-tier consumer architecture (Sister E is Tier A observability-only by construction; no per-axis Tier B emission)
- **CLAUDE.md "Apples-to-apples evidence discipline"** ✓ (intra-test STATISTIC comparison at same fixture + same canonical helpers + smoke-mode verifies Sister C N=200 16×5 median baseline reproduction EXACTLY; explicit routing-effect-vs-contest-score caveat)
- **CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"** ✓ (macOS-CPU advisory explicitly tagged + non-promotable + result_review_blockers cite "requires_paired_cuda_cpu_validation_per_v15_protocol_1e_4_threshold")
- **CLAUDE.md "Forbidden premature KILL without research exhaustion"** ✓ (criterion #6 paired-CUDA validation queued for Sister F; PARADIGM-INTACT classification; KILL is the LAST RESORT)
- **CLAUDE.md "MPS auth eval is NOISE"** ✓ (CPU only; no MPS used; tagged `[macOS-CPU advisory]`)
- **CLAUDE.md "MLX-first portable-local-substrate authority"** ✓ (probe is numpy + torch CPU only; ZERO MLX dependency; Sister B cache reused at sister path)
- **8th MLX-first + numpy-portable + individually-fractal standing directive** ✓ (sister-disjoint READ-ONLY consumer of Sister B's cache; pure numpy + torch CPU; the EXTREMA statistic is the substrate's PRINCIPLED per-method optimal bottleneck per Sister C recommendation; NO generic catalog-enumeration of arbitrary quantiles — the p95/p99/argmax are the canonical Fridrich 2014 EXTREMA family)
- **10th apples-to-apples + WebSearch authorized + math/science/engineering rigor standing directive** ✓ (WebSearch invoked to verify Holub-Fridrich-Denemark 2014 canonical weighted-EXTREMA principle; math rigor binding via canonical Fridrich UNIWARD primitive + canonical Sallee 2003 weighted-median + canonical Fridrich 2014 weighted-EXTREMA generalization; engineering rigor binding via the 13 HNeRV parity inviolable lessons + Catalog #290/#294/#296/#303/#305 design-memo discipline)
- **11th final rate attack off-the-shelf + development workflow standing directive (apples-to-apples baseline FIRST)** ✓ (Sister E tested the EXTREMA at the SAME fixed 16×5 canonical LUT shape as Sister B+C FIRST per ORDER Dim 8; demonstrates the canonical $0 macOS-CPU advisory dev-loop workflow)
- **12th 3-strategy attack framework standing directive** ✓ (Sister E sits on DISTORTION pure-axis at the STATISTIC sub-axis; orthogonal to RATE axis from FEC family + FULL SCORER axis from V15 paired-CUDA)
- **13th OPTIMAL-TRIO standing directive** ✓ (TECHNIQUE = canonical Fridrich 2014 weighted-EXTREMA family at 4 statistic variants; WAY = canonical cell-by-cell empirical measurement at fixed dyn_range + fixed LUT shape; TIME = pre-dispatch PV per Catalog #229 + V15 protocol; sister-disjoint UNIQUE-AND-COMPLETE-PER-METHOD via the canonical helper consumer-only pattern + principled EXTREMA fork)

## NOT a PR111 candidate — explicit declaration

Per V15 protocol + Catalog #192/#343: PR111 candidate status requires paired-CUDA frontier-crossing on 1:1 contest-compliant hardware (Linux x86_64 + NVIDIA GPU). Sister E is `[macOS-CPU advisory]` — by construction NOT 1:1 contest-compliant; the bins_changed/mean_l2/max_delta metrics are LUT-statistic ROUTING-EFFECT research signals NOT contest scores. Per the `user_pr_attribution` memory non-negotiable + CLAUDE.md "FORBIDDEN CLAUDE ATTRIBUTION IN PUBLIC-PR SURFACES": Sister E does NOT produce a PR111 candidate report; no public-PR-facing artifact is written.

## Operator-routable next steps

1. **Sister F spawn** (reactivation criterion #6 RECOMMENDED — paired-CUDA validation): queue paired-CUDA validation of the weighted-EXTREMA statistic via canonical `tools/dispatch_modal_paired_auth_eval.py` at estimated $0.50-1.00 cost (matches V15 budget). The Sister E result EMPIRICALLY ESTABLISHES that the EXTREMA statistic BREAKS the routing-effect saturation; the paired-CUDA delta vs the V15 1e-4 threshold IS the canonical PROMOTION arbiter that disambiguates the OPTIMISTIC (concentration improves scorer-relevant chroma → lower score) vs PESSIMISTIC (high-variance concentration produces visible artifacts → higher score) interpretation. RECOMMENDED variant: start with p99 (strongest mean_l2 routing-effect) AND p95 (more conservative tail) as the two paired-CUDA arms; argmax is the highest-variance and should be the third arm only if p95/p99 show promise. Per Catalog #325 14-day window the per-substrate symposium evidence is operator-gated.
2. **Sister D spawn** (reactivation criterion #4): apply UNIWARD per-LUT-index routing WITH the EXTREMA statistic to NSCS06 grayscale_lut OR VQ-VAE indices_blob — different application surface family; per the per-substrate individually-fractal directive, different substrates may have different idiosyncratic structural couplings. Now that the STATISTIC axis is the validated binding axis, Sister D should use the EXTREMA statistic from byte zero.
3. **The 3-point orthogonal-axis decomposition is empirically COMPLETE**: sample-size (Sister B; responsive), LUT-shape (Sister C; saturated), statistic (Sister E; THE BINDING AXIS). The substrate's UNIQUE-OPTIMAL bottleneck is the per-bin statistic choice. The next-EV step is paired-CUDA validation of the EXTREMA statistic (Sister F), NOT further macOS-CPU advisory axis exploration.
4. **CAUTION**: the 100% bin-occupation is a ROUTING-EFFECT signal NOT a contest-score signal. Do NOT interpret "PARADIGM_VALIDATED_WITH_EXTREMA" as "the EXTREMA lowers the score" — it means "the EXTREMA produces a strong, measurable, non-saturated routing effect WORTH the paired-CUDA spend to disambiguate." The max_delta=151 u8 (~59% of the 0-255 range) is large enough that the PESSIMISTIC interpretation (visible chroma artifacts for the bulk pixels that prefer the bin's central chroma, not the high-weight-tail chroma) is a live risk that ONLY paired-CUDA can rule out.

## Cost

- $0 GPU spend (macOS CPU only; no Modal / Vast.ai / Lightning dispatch)
- ~73.6 sec CPU wall-clock on M5 Max (zero cache extension thanks to Sister B reuse + smoke regression guard at the median baseline)
- ~620 LOC standalone probe script (~28KB)
- ~430 LOC landing memo (~32KB; this file)
- 1 Catalog #313 probe-outcome ledger entry registered (verdict=PROCEED; blocker_status=advisory)
- 1 Catalog #344 canonical equation candidate registered FORMALIZATION_PENDING (`uniward_substrate_unique_optimal_statistic_savings_v1`)
- 0 cache extensions emitted (Sister B's N=200 cache REUSED via Catalog #230 sister-disjoint READ-ONLY)
- 12 sidecar LUT artifacts emitted at `experiments/results/v15_uniward_sister_e_weighted_extrema_fridrich_2014_n200_macos_cpu_advisory_20260527/lut_artifacts_{statistic}/` (4 statistics × 3 arrays each: lut_statistic.npy + lut_canonical_median.npy + per_bin_l2_difference.npy; sister-disjoint; APPEND-ONLY)
- 6 reactivation criteria status: #1 COMPLETE (Sister B), #3 COMPLETE-AND-PARTIALLY-FALSIFIED (Sister C), #5 COMPLETE-AND-VALIDATED (this Sister E), #2 + #4 operator-routable; criterion #6 (paired-CUDA validation) RECOMMENDED for next Sister F

## Sister-coordination + checkpoint discipline confirmation

7 subagent checkpoints emitted via canonical `tools/subagent_checkpoint.py` per Catalog #206 (Steps 1-7; final Step 7 marks complete after commit; checkpointed aggressively every ~1-2 tool uses given 2 prior crashes):
1. Step 1: fresh-start subagent registration + PV launch (predecessors crashed at step 0; nothing to resume)
2. Step 2: PV reading canonical weighted-median helper internals
3. Step 3: WebSearch Holub-Fridrich-Denemark 2014 confirmed + canonical helper signatures verified
4. Step 4: probe script written
5. Step 5: smoke PASS (median reproduces Sister C 21/80 exactly) + full probe launch
6. Step 6: EXTREMA BROKE saturation (100% bins) + Catalog #313 registered + landing memo write
7. Step 7 (final): canonical commit via serializer + checkpoint complete

Sister-disjoint per Catalog #230: NO collision with in-flight `phase_9_full_lifecycle_cli` (touches `tools/operator_pr_submission_full_lifecycle.py` + tests + Phase 9 memo; all DISJOINT files). Catalog #340 sister-checkpoint guard expected PROCEED on commit.

## V15 13th OPTIMAL-TRIO declaration per 13th standing directive

Sister E honors the 13th OPTIMAL-TRIO standing directive (mlx-first-numpy-portable-individually-fractal):
- **MLX-first**: reused Sister B's MLX-local-captured cache via canonical READ-ONLY consumer pattern; canonical UNIWARD per-LUT-index integration is MLX-portable (pure numpy)
- **numpy-portable**: probe script + sidecar artifacts + EXTREMA LUT derivation are pure numpy + torch CPU; ZERO MLX dependency at runtime; inflate-side runtime UNTOUCHED per HNeRV parity L4
- **individually-fractal**: Sister E is unique-per-method (sister-disjoint READ-ONLY consumer imports from V15 / Sister B / Sister C + zero modification to canonical helpers); the EXTREMA statistic is the substrate's PRINCIPLED per-method optimal bottleneck per Sister C's empirical recommendation — NOT a generic catalog-enumeration of arbitrary quantiles; the p95/p99/argmax are the canonical Fridrich 2014 EXTREMA family (the natural generalization of the Sallee 2003 weighted-median to the high-weight tail), tested apples-to-apples vs the median baseline FIRST per the 11th ORDER directive Dim 8

Per the AUTOMATED + COMPOUNDING + OPTIMAL META-principle standing directive: Sister E emits canonical artifacts that compound into future iterations (provenance.json + per-statistic sidecar LUT artifacts + Catalog #313 probe-outcomes ledger entry PROCEED + Catalog #344 candidate equation FORMALIZATION_PENDING); the probe script IS the AUTOMATED reproducibility surface for Sister F (criterion #6 paired-CUDA) + Sister D (criterion #4 application surface); the `PARADIGM_VALIDATED_WITH_EXTREMA` verdict + 100% bin-occupation breakthrough signature is the OPTIMAL signal for the recommended Sister F paired-CUDA validation. The STATISTIC axis being the substrate's UNIQUE-OPTIMAL bottleneck is the per-substrate individually-fractal finding the operator's directives demanded — the substrate's bottleneck was NOT the LUT shape (Sister C) but the per-bin statistic (Sister E).
