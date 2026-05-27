# V15 UNIWARD Sister B Reactivation Criterion #1 N=200 Pairs macOS-CPU Advisory Landed 2026-05-26

**Subagent**: `v15_uniward_sister_b_C1990441`
**Lane**: `lane_v15_uniward_sister_b_reactivation_criterion_1_n200_macos_cpu_advisory_20260526`
**Sister of**: V15 paired CPU+CUDA landing memo `.omx/research/v15_uniward_7th_order_paired_cpu_cuda_on_nscs06_v8_chroma_lut_stacked_landed_20260526.md` (commit `877cb9f8c`)
**Predecessor anchor**: V15 N=50 IMPLEMENTATION-LEVEL FALSIFICATION at paired-CUDA + paired-CPU (CUDA delta=2.66e-6 / CPU delta=1.47e-7 / 8/80 bins changed / dyn_range=19.71x)
**Date**: 2026-05-26
**Tag**: `[macOS-CPU advisory]` per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192/#317/#341 (NOT promotable; NOT a contest score claim)
**Verdict**: **INTERMEDIATE_PARTIAL_SCALING** at N=200 vs N=50 baseline (per Catalog #307 PARADIGM-INTACT + IMPLEMENTATION-LEVEL-EXPLORATION; recommend reactivation criteria #2 AND #3 combined per V15 landing memo)
**NOT a frontier-crossing event**; **NOT a PR111 candidate** per Catalog #343 + CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" (macOS-CPU advisory is NOT 1:1 contest-compliant per Catalog #192)
**FORMALIZATION_PENDING**: canonical equation `uniward_per_lut_index_distortion_weight_savings_v1` per Catalog #344

## TL;DR

Sister B probe extended V15's N+1 cache from 50 → 200 pairs (4× larger) on macOS CPU at $0 spend, then re-derived UNIWARD-weighted vs canonical-median chroma LUT at N=200 to test V15 reactivation criterion #1 sample-size-dependence hypothesis. Result: **the LUT-level signal DOES scale with sample size, but with a COMPLEX pattern that the V15 verdict tree's three branches did not anticipate** — emitting the previously-undocumented `INTERMEDIATE_PARTIAL_SCALING` verdict that recommends reactivation criteria #2 (higher gradient dynamic range) AND #3 (finer LUT granularity) COMBINED for the next iteration.

Empirical receipts:

| Metric | N=50 (V15 baseline) | N=200 (sister B) | Ratio (N=200 / N=50) |
|---|---|---|---|
| bins_changed | 8/80 | **21/80** | **2.62×** ↑ |
| max_per_channel_delta_u8 | 12 | **5** | 0.42× ↓ |
| mean_per_bin_l2_diff | 0.26 | **0.3344** | 1.29× ↑ |
| max_per_bin_l2_diff | 12.65 | **5.0990** | 0.40× ↓ |
| UNIWARD weight dynamic_range | 19.71× | **772.41×** | **39.19×** ↑ |

**Key empirical surprise**: as N quadruples, the number of bins-touched-by-UNIWARD nearly TRIPLES (2.62×) AND the per-pixel UNIWARD weight dynamic range LEAPS by 39× (19.71x → 772.41x), BUT the maximum per-bin LUT delta SHRINKS by 2.5× (12 → 5 u8 / 12.65 → 5.0990 L2). This is the signature of an **estimator transitioning from sparse-undersampled to dense-noise-averaged**: with more samples, more bins are touched (the LUT becomes more populated), the gradient dynamic range surfaces previously-hidden tail behavior (more pixels are now anchoring extreme gradient magnitudes), AND the canonical-median estimator converges (so the UNIWARD-weighted vs canonical-median delta concentrates on smaller per-bin perturbations distributed across MORE bins).

Per V15 protocol: this LUT-statistic signal is **research-only** at this hardware substrate (macOS-CPU advisory per Catalog #192); the V15 PROMOTION threshold of 1e-4 paired-CUDA delta CANNOT be measured at $0 (paired-CUDA requires Modal/Vast.ai/Lightning paid dispatch; scope explicitly excludes paid dispatch per the operator's reactivation-criterion-#1 scope). Canonical equation #344 `uniward_per_lut_index_distortion_weight_savings_v1` remains FORMALIZATION_PENDING; this anchor extends V15's N=50 evidence with the N=200 LUT-level signal but does NOT PROMOTE the equation.

## 3-strategy attack decomposition

Per CLAUDE.md "3-strategy attack framework" non-negotiable: sister B sits on the **DISTORTION pure-axis** (sub-axis JOINT d_seg + d_pose at the LUT-derivation surface). The probe is an INTRA-TEST comparison at the LUT-statistic surface (UNIWARD-weighted vs canonical-median); both arms use the SAME 200 frames, SAME 96×128 NEAREST-upsampled real-scorer cache, SAME canonical helpers, SAME 16×5 LUT shape.

Sister of V15 protocol:
1. **DISTORTION axis** = TESTED-AT-LARGER-N (this sister-B lane)
2. **RATE axis** = UNCHANGED (no archive emitted; cache + LUT only)
3. **FULL SCORER axis** = the 39× dynamic-range jump at the upstream gradient surface is the key new signal; the LUT-level downstream signal compresses but the upstream signal expands

## MLX-first → numpy-portable bridge contract

Honored per 8th + 13th standing directives:
- N+1 cache extension via canonical `compute_real_scorer_gradient_per_pixel` (CPU torch; sister of V15 build script)
- LUT derivation via canonical `build_uniward_weighted_chroma_lut` + `build_chroma_lut_from_ground_truth` (pure numpy; MLX-portable per `weight_map_per_lut_index.py` docstring)
- Comparison via canonical `compare_uniward_vs_canonical_lut` (pure numpy)
- ZERO MLX runtime dependency at the probe; macOS CPU only

## Apples-to-apples evidence discipline confirmation (per 10th standing directive)

Per CLAUDE.md "Apples-to-apples evidence discipline" NON-NEGOTIABLE + 10th standing directive:

1. **Same fixture both arms**: UNIWARD-weighted LUT + canonical-median LUT both derived from the SAME 200 frames + SAME class labels + SAME EVAL_HW (384×512) at the same SegNet-argmax-derived class partition.
2. **Same canonical helpers**: `build_uniward_weighted_chroma_lut` + `build_chroma_lut_from_ground_truth` are both READ-ONLY sister-disjoint consumer imports per Catalog #230.
3. **Same cache lineage**: V15's first 50 pairs preserved byte-identical from V15 cache (verified at smoke run reproducing V15 N=50 baseline EXACTLY: bins_changed=8/80, max_delta=12, mean_l2=0.2599, max_l2=12.6491, dyn_range=19.71x).
4. **Apples-to-apples N=50 vs N=200**: both LUTs at each N use IDENTICAL pipeline; the ONLY varied independent variable is N.
5. **Axis label honored**: every metric tagged `[macOS-CPU advisory]`; NOT promotable per Catalog #192.

## Canonical-vs-unique decision per layer

Per Catalog #290 (binding for every substrate work):

| Layer | Decision | Rationale |
|---|---|---|
| Cache loader (np.load) | ADOPT canonical | V15 cache format identical |
| Cache extension scorer eval | ADOPT canonical `compute_real_scorer_gradient_per_pixel` | Sister-disjoint READ-ONLY from `tools/uniward_per_pixel_n_plus_1_real_scorer_anchored_sweep_20260526.py` |
| Cache concatenation | UNIQUE | Sister-disjoint APPEND-ONLY: V15 cache UNTOUCHED; sister cache emitted to sister path per Catalog #110/#113 |
| EVAL_HW pair decode | ADOPT canonical `decode_real_pairs` from `tac.substrates._shared.trainer_skeleton` | Sister-disjoint READ-ONLY |
| SegNet argmax | ADOPT canonical pattern from V15 build script (chunked, no_grad) | Sister-disjoint mirror |
| NEAREST upsample 96×128 → 384×512 | ADOPT canonical pattern from V15 build script `_upsample_grad_nearest_to_full_res` | Sister-disjoint mirror; numpy.repeat preserves spatial mass |
| Per-pixel UNIWARD weight | ADOPT canonical `compute_per_pixel_uniward_weight_map_numpy` | Sister-disjoint READ-ONLY consumer import |
| UNIWARD-weighted LUT | ADOPT canonical `build_uniward_weighted_chroma_lut` | Sister-disjoint READ-ONLY consumer import |
| Canonical-median LUT | ADOPT canonical `build_chroma_lut_from_ground_truth` | Sister-disjoint READ-ONLY consumer import |
| LUT comparison | ADOPT canonical `compare_uniward_vs_canonical_lut` | Sister-disjoint READ-ONLY consumer import |
| Verdict tree | UNIQUE | NEW logic extending V15 reactivation criterion #1 with 4-branch verdict tree (V15's 3-branch tree was incomplete) |
| Provenance.json + landing memo + sidecar cache | UNIQUE per-lane | Standard sister-disjoint per Catalog #110/#113 |

## 9-dimension success checklist evidence

Per Catalog #294:

1. **UNIQUENESS**: First N=200 macOS-CPU advisory empirical anchor on V15's UNIWARD per-LUT-index integration package; sister V15 N=50 paired-CUDA+CPU landing is the only prior empirical anchor at this surface
2. **BEAUTY + ELEGANCE**: 567 LOC standalone probe script + zero modification to V15 build script or canonical helpers; sister-disjoint append-only sidecar cache emission
3. **DISTINCTNESS**: Probe is INTRA-TEST LUT-statistic comparison at increased N; does NOT shadow V15's paired-CUDA+CPU dispatch surface; orthogonal axis to V15's archive-emission flow
4. **RIGOR**: Smoke test PASSED with EXACT reproduction of V15 N=50 baseline (bins_changed=8/80, max_delta=12, mean_l2=0.2599, max_l2=12.6491, dyn_range=19.71x) — proof of canonical-helper pipeline correctness PRE full N=200 run; Catalog #229 PV honored (read V15 landing memo + sister 7th-order memo + canonical helpers + canonical sweep tool + canonical frontier pointer BEFORE writing probe script); 5 dedicated checkpoints emitted
5. **OPTIMIZATION PER TECHNIQUE**: UNIWARD per-LUT-index aggregation is canonical Fridrich; weighted-median canonical Sallee 2003; gradient cache canonical N+1 anchor at 96×128; all routed through canonical helpers
6. **STACK-OF-STACKS COMPOSABILITY**: Sister-disjoint READ-ONLY consumer imports preserve V15 + UNIWARD integration + NSCS06 v8 substrate untouched; sister B probe IS the canonical pattern for V15's other 4 reactivation criteria (#2-#5) to follow at $0 macOS-CPU advisory before any paid-CUDA validation
7. **DETERMINISTIC REPRODUCIBILITY**: numpy + torch + PIL all deterministic-seeded with seed=20260526; sister cache byte-stable + reproducible; SegNet argmax deterministic with `no_grad`
8. **EXTREME OPTIMIZATION + PERFORMANCE**: 5 min 30 sec total wall-clock on macOS M5 Max CPU (~213s cache extension for 150 new pairs + ~75s LUT derivation + auth-eval ~40s); ZERO paid GPU spend; well within operator budget for $0 macOS-CPU advisory scope
9. **OPTIMAL MINIMAL CONTEST SCORE**: Sister B is NOT a frontier-pursuit attempt; it is a controlled-comparison test of UNIWARD effect at LARGER sample size; its result IS the canonical empirical anchor for the FORMALIZATION_PENDING canonical equation's N-axis dependence; informs sister C/D/E future spawns for reactivation criteria #2/#3/#4/#5 at $0 macOS-CPU advisory

## Cargo-cult audit per assumption

Per Catalog #303:

| Assumption | Pre-execution classification | Empirical verdict |
|---|---|---|
| "Larger N_LUT_DERIVATION (200 vs 50 pairs) will surface UNIWARD-weighted statistic differentiation at the LUT level" | CARGO-CULTED-PENDING-VALIDATION (V15 reactivation criterion #1 hypothesis) | **EMPIRICALLY-PARTIAL** — bins_changed nearly TRIPLES (2.62×) AND per-pixel dyn_range LEAPS 39× — strong scaling signal at the upstream gradient surface — BUT per-bin max delta SHRINKS 2.5× (12→5) indicating downstream LUT-statistic concentration smooths out at larger N |
| "V15's 3-branch verdict tree (IMPL-CONDITIONAL / IMPL-FALSIFIED-STABLE / FALSE-POSITIVE) exhaustively partitions the N-scaling outcome space" | CARGO-CULTED-FROM-V15-LANDING-MEMO | **EMPIRICALLY-FALSIFIED** — the empirical pattern lands in a 4th branch INTERMEDIATE_PARTIAL_SCALING that V15's tree did not anticipate: bins-touched scales above the 0.5-2.0 IMPL-FALSIFIED-STABLE range AND below the 10× IMPL-CONDITIONAL threshold |
| "First 50 pairs of N=200 extension produce IDENTICAL cache values to V15 cache (deterministic given seed + scorer state)" | HARD-EARNED-FROM-CANONICAL-HELPER-DETERMINISM | **CONFIRMED OPERATIONALLY** — smoke run reproduces V15 N=50 baseline EXACTLY at every digit (bins_changed=8/80, max_delta=12, mean_l2=0.2599 vs V15 0.26, max_l2=12.6491 vs V15 12.65, dyn_range=19.71x) |
| "NEAREST upsample 96×128 → 384×512 preserves spatial mass without smoothing artifacts" | HARD-EARNED-FROM-V15-BUILD-SCRIPT-MIRROR | UNCHANGED — same canonical pattern; same empirical signal |
| "macOS CPU `compute_real_scorer_gradient_per_pixel` produces gradients with IDENTICAL distribution to V15's cache (validates cache lineage)" | HARD-EARNED-FROM-CANONICAL-HELPER-PURITY | CONFIRMED VIA SMOKE — V15's first 50 pairs are byte-identical when reloaded; pairs 51-200 follow same compute path |
| "macOS CPU is 1:1 contest-compliant for LUT-statistic signal" | CARGO-CULTED-PENDING-VALIDATION | **STRUCTURALLY-FALSIFIED** per Catalog #192 — macOS-CPU is NEVER 1:1 contest-compliant; the LUT-statistic IS valid `[macOS-CPU advisory]` research signal but cannot promote to score claim without paired Linux x86_64 paired-CUDA evidence |
| "The 39× dynamic-range jump at the upstream gradient surface is statistical noise vs real signal" | CARGO-CULTED-PENDING-CROSS-VALIDATION | **STATISTICALLY-LIKELY-REAL** — the V15 cache had only 50 samples; tail-end magnitudes scale with sample-count exposure; 39× expansion at 4× sample-count is consistent with heavy-tailed gradient distribution (canonical for ReLU-deep networks like SegNet UNet + PoseNet FastViT) — but a sister test would re-extend to N=400 OR seed-variation to disambiguate |

## Observability surface

Per Catalog #305 — all 6 facets HONORED:

- **Inspectable per layer**: per-pair UNIWARD weight stats (min/max/mean/std/dynamic_range per pair); per-bin LUT comparison (`WeightedMedianResult.per_bin_l2_difference` array exposes per-(level, class) L2 delta); per-axis SegNet/PoseNet gradient stats from extended cache
- **Decomposable per signal**: N=200 vs N=50 ratios per metric exposed (bins_changed_ratio / mean_l2_ratio / max_l2_ratio / dyn_range_ratio); per-bin delta exposed for top-3 highest-L2 bins; rate-axis NOT computed (no archive emitted)
- **Diff-able across runs**: deterministic seed (20260526); sister cache sha256 emitted (extended_cache_path field); LUT sha256s emitted (both UNIWARD + canonical); verdict reproducible from provenance.json
- **Queryable post-hoc**: provenance.json (~3KB JSON; sort_keys=True byte-stable); probe_full.log (full stdout trace with stage timestamps); Catalog #313 probe-outcomes ledger entry (queryable via `tac.probe_outcomes_ledger.query_by_substrate('nscs06_v8_chroma_lut_uniward_per_lut_index')`); subagent checkpoint trail via `.omx/state/subagent_progress.jsonl`
- **Cite-able**: 1 sister cache sha256 + 2 LUT sha256s + 5 canonical Provenance fields (`evidence_grade=[macOS-CPU advisory]`, `axis_tag=[macOS-CPU advisory]`, `hardware_substrate=macos_arm64_m5_max_cpu`, `score_claim=False`, `promotable=False`)
- **Counterfactual-able**: N=50 baseline (V15) vs N=200 (sister B) IS the canonical counterfactual; SMOKE mode (--smoke flag) reproduces N=50 baseline exactly for the canonical apples-to-apples regression guard; sister C/D/E future spawns can probe reactivation criteria #2/#3/#4/#5 at sister N values for compound counterfactual

## Predicted ΔS band vs empirical (per Catalog #296)

PREDICTED (from V15 reactivation criterion #1 hypothesis): one of {IMPL_CONDITIONAL_SAMPLE_SIZE_DEPENDENT, IMPL_FALSIFIED_STABLE_AT_ARCHIVE_VARIANT, FALSE_POSITIVE_RISK}. The V15 verdict tree implicitly bounded the bins_changed_ratio outcome to be either `>= 10×` (IMPL-CONDITIONAL) OR `within [0.5, 2.0]×` (IMPL-FALSIFIED-STABLE) OR `< 0.5×` (FALSE-POSITIVE).

EMPIRICAL: bins_changed_ratio = 2.62× — **lands BETWEEN the IMPL-FALSIFIED-STABLE upper bound (2.0×) and the IMPL-CONDITIONAL lower bound (10×)**. The V15 verdict tree did NOT exhaustively partition the outcome space; this probe surfaces a 4th branch INTERMEDIATE_PARTIAL_SCALING (newly defined in this lane) that recommends reactivation criteria #2 AND #3 combined for the next iteration.

Per Dykstra-feasibility (Catalog #296): the intersection of (sample-size-dependence ∩ LUT-shape-granularity-sufficient ∩ real-scorer-dynamic-range-stable) is STATISTICALLY NON-EMPTY at the BINS-CHANGED axis (the count nearly triples) but STATISTICALLY MIXED at the PER-BIN-DELTA-MAGNITUDE axis (max delta shrinks 2.5×). The dyn_range axis is the key surprise: the 39× expansion suggests the real-scorer gradient distribution has heavy tails that the N=50 cache UNDER-sampled.

## Horizon-class declaration

Per Catalog #309: **plateau_adjacent** (UNCHANGED from V15 DOWN-REVISED-from-frontier_pursuit). The N=200 empirical signal does NOT move sister B into frontier_pursuit territory; the LUT-statistic signal scales partially but is structurally bounded by the 16×5 LUT shape (only 80 bins total can possibly change). Sister C/D/E spawns at criteria #2/#3 (higher dynamic range + finer LUT granularity) may shift to frontier_pursuit IF the empirical evidence supports it.

## Catalog #344 canonical equation anchor status

**Status**: `FORMALIZATION_PENDING` (UNCHANGED; no PROMOTION). The canonical equation `uniward_per_lut_index_distortion_weight_savings_v1` cannot promote to PARADIGM-VALIDATED-EMPIRICALLY from this sister B anchor because:

1. **Catalog #192**: macOS-CPU advisory is NOT 1:1 contest-compliant hardware (Linux x86_64 required for `[contest-CPU]`; NVIDIA GPU on Linux required for `[contest-CUDA]`).
2. **V15 PROMOTION threshold**: paired-CUDA delta > 1e-4 declared in V15 build-script protocol; this sister B probe does NOT measure paired-CUDA delta (scope explicitly excludes paid dispatch per operator's reactivation criterion #1 specification).
3. **Sample-size sufficiency**: the N=200 signal is real but PARTIAL; sister probes at criteria #2/#3 (higher dynamic range + finer LUT granularity) may surface stronger signal.

This anchor extends V15's N=50 empirical evidence with the N=200 LUT-level signal but does NOT promote the equation. The canonical equation registration script `tools/register_uniward_per_pixel_score_conditional_sensitivity_canonical_equation_20260526.py` for the sister `uniward_per_pixel_*` equation correctly refuses registration on NULL verdicts; the corresponding `uniward_per_lut_index_*` equation similarly remains FORMALIZATION_PENDING until paired-CUDA evidence lands.

Reactivation criteria for future PROMOTION (per CLAUDE.md "Forbidden premature KILL"):

1. **Reactivation criterion #1**: COMPLETE (this sister B landing).
2. **Reactivation criterion #2** (operator-routable): re-derive cache at higher real-scorer dynamic range (e.g., longer training / lower learning rate / scorer-architecture diversity); sister B's empirical 39× dyn_range jump shows the dynamic range axis IS responsive to sample-count exposure — criterion #2 may compound this signal.
3. **Reactivation criterion #3** (operator-routable): finer LUT granularity (extend v8 to 32 levels × 5 classes OR 16 levels × 10 classes); sister B's 21/80 bins-changed signal is bounded by 80 total bins; finer granularity may expose more bin-level differentiation.
4. **Reactivation criterion #4** (operator-routable per V15 landing memo): apply UNIWARD per-LUT-index routing to NSCS06 grayscale_lut (T3 council #2 stacking candidate) or VQ-VAE indices_blob (T3 council #3 stacking candidate); different application surface family may yield different empirical signals per-substrate.
5. **Reactivation criterion #5** (operator-routable): UNIWARD-weighted EXTREMA (top-K quantile) replaces UNIWARD-weighted median; sister B's MIXED signal (more bins, smaller per-bin delta) suggests the EXTREMA may surface concentrated high-weight-pixel routing more strongly.

Per sister B's empirical INTERMEDIATE_PARTIAL_SCALING verdict + the 39× upstream dyn_range expansion: **recommend reactivation criteria #2 AND #3 combined for the next iteration** (higher dynamic range from longer training compounds with finer LUT granularity to surface the partial-scaling signal in both directions).

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map**: ACTIVE — per-pixel UNIWARD weights from extended N=200 cache feed canonical sensitivity-map surface; emitted to provenance.json with full distribution stats
2. **Pareto constraint**: ACTIVE — per-axis `[macOS-CPU advisory]` tag carries Catalog #323 canonical Provenance; non-promotable per Catalog #341 (score_claim=False + promotable=False); explicit non-frontier-crossing per V15 protocol
3. **Bit-allocator**: N/A at this probe surface (no archive emission; LUT-statistic comparison only). The 16×5 LUT shape was hardcoded inherited from canonical v8 substrate; sister C probe per criterion #3 may extend bit-allocator surface
4. **Cathedral autopilot dispatch**: ACTIVE — Catalog #313 probe-outcomes ledger row registered (probe_id=`v15_uniward_sister_b_reactivation_criterion_1_n200_pairs_20260527T034044Z` + verdict=PARTIAL + blocker_status=advisory + expires_at_utc=2026-06-26T04:02:55Z). Cathedral autopilot ranker can consume the empirical INTERMEDIATE_PARTIAL_SCALING verdict + N=200 LUT signals as evidence for sister probes at reactivation criteria #2-#5
5. **Continual-learning posterior**: ACTIVE — provenance.json carries canonical Provenance per Catalog #323; sister cache sidecar at `experiments/results/v15_uniward_sister_b_reactivation_criterion_1_n200_macos_cpu_advisory_20260526/real_scorer_gradients_cache_extended_n200.npz` is queryable by future sister probes (criterion #2 cache extension can read this as base)
6. **Probe-disambiguator**: ACTIVE — sister B IS the canonical disambiguator at N=200 sample size; the empirical INTERMEDIATE_PARTIAL_SCALING result disambiguates the V15 verdict-tree's 3-branch partition and surfaces a 4th branch (partial scaling) that the V15 landing memo did not anticipate

## Carmack-dissent verdict per Catalog #307

**PARADIGM-INTACT + IMPLEMENTATION-LEVEL-EXPLORATION-AT-LARGER-N**. Per CLAUDE.md "Forbidden premature KILL without research exhaustion":

- The UNIWARD per-LUT-index routing PRIMITIVE remains canonical Fridrich; this sister B probe does NOT falsify the primitive
- The IMPLEMENTATION at (N_LUT=200, dynamic_range=772.41x, LUT=16×5) shows PARTIAL SCALING signal — neither fully validating nor falsifying the V15 N=50 implementation; the signal is real but bounded by LUT shape
- The 5 reactivation criteria above enumerate the research-exhaustion paths; KILL is the LAST RESORT
- The 5th + 6th + 7th-MLX + V15-paired-CUDA + sister-B-N=200 empirical trajectory is now a 5-point sequence: PARADIGM-NULL-AT-RAW-RGB → PARADIGM-NULL-AT-CAPACITY-CONSTRAINED → PARADIGM-VALIDATED-AT-MLX-LOCAL-SYNTHETIC → IMPLEMENTATION-FALSIFIED-AT-PAIRED-CUDA-REAL-N=50 → IMPLEMENTATION-PARTIAL-SCALING-AT-MACOS-CPU-ADVISORY-N=200

Per Catalog #348 retroactive sweep: **NOT TRIGGERED**. The sister B verdict is IMPLEMENTATION-LEVEL not PARADIGM-LEVEL; no historical KILL/DEFER/FALSIFY verdicts are invalidated by sister B (V15 IMPL-FALSIFIED + sister 7th-order PARADIGM-VALIDATED-MLX-LOCAL + 5th/6th-order PARADIGM-NULL verdicts ALL remain intact — sister B simply extends V15's N=50 empirical evidence with the N=200 LUT-level signal at macOS-CPU advisory scope).

## Sister-disjoint discipline confirmation per Catalog #230

NO modifications to V15 build script, V15 source cache, UNIWARD per-LUT-index integration package, NSCS06 v8 substrate, or any sister artifact. All canonical helpers consumed READ-ONLY (`compute_per_pixel_uniward_weight_map_numpy`, `build_uniward_weighted_chroma_lut`, `compare_uniward_vs_canonical_lut`, `build_chroma_lut_from_ground_truth`, `compute_real_scorer_gradient_per_pixel`, `load_differentiable_scorers`, `decode_real_pairs`). Output strictly scoped under `experiments/results/v15_uniward_sister_b_reactivation_criterion_1_n200_macos_cpu_advisory_20260526/` + `.omx/research/v15_uniward_sister_b_reactivation_criterion_1_n200_pairs_*_landed_20260526.md`.

Sister-disjoint confirmed via `subagent_progress.jsonl` review at probe-execution time: all in-flight sisters working on disjoint files (5D canvas populator / Phase 5+6 submission_packet linter+compliance / build-2-3 extended operators / Cascade C' Wave 7 trainer / V15 build script subagent had completed prior to my work). Catalog #340 sister-checkpoint guard not triggered.

## Discipline anchors

- **Catalog #229 PV** (read V15 landing memo + UNIWARD 7th-order landing memo + V15 build script + canonical sweep tool + canonical Provenance helpers + canonical equations registry + canonical_frontier_pointer.json + 10th apples-to-apples standing directive memo + sister subagent activity in `subagent_progress.jsonl` BEFORE writing probe script)
- **Catalog #206** (5 checkpoints emitted across PV / plan / smoke-pass / probe-launch / probe-complete-and-ledger-register cycle)
- **Catalog #110/#113 APPEND-ONLY** (V15 cache + V15 build artifacts + V15 landing memo + canonical helpers ALL untouched; sister B sidecar cache + provenance.json + landing memo are NEW artifacts only)
- **Catalog #117/#157/#174/#235/#289** canonical commit serializer (will use --expected-content-sha256 + co-author trailer for the commit)
- **Catalog #119** Co-Authored-By Claude Opus 4.7 (1M context) trailer (internal commit)
- **Catalog #125** 6-hook wire-in declaration (all 6 hooks ACTIVE or N/A with rationale per the declaration section above)
- **Catalog #127** authoritative-tag custody metadata (every metric tagged `[macOS-CPU advisory]` per per-call-site discipline)
- **Catalog #131/#138** fcntl-locked + strict-load discipline (Catalog #313 probe-outcomes ledger + provenance.json sidecar follow canonical helper patterns; no bare writes to `.omx/state/`)
- **Catalog #164/#226** canonical scorer-load + trainer auth-eval routing (via canonical `load_differentiable_scorers`; sister-disjoint READ-ONLY consumer import; no fork)
- **Catalog #168** AST walker handles both Assign + AnnAssign (probe script uses no AST scan; gate N/A)
- **Catalog #185** META-meta-meta drift detection (this gate run did NOT touch CLAUDE.md catalog text; gate N/A)
- **Catalog #192** macOS-CPU advisory non-promotion (every metric tagged + `promotable=False`; `score_claim=False`; `ready_for_exact_eval_dispatch=False`)
- **Catalog #199** paired-env operator-authorize bypass (N/A; no paid dispatch attempted)
- **Catalog #205/#295** inflate runtime hygiene (N/A; no archive emitted; no inflate.py runtime emitted)
- **Catalog #208** docs/* no local-absolute-paths (landing memo cites repo-relative paths only; no `/Users/adpena/` leakage)
- **Catalog #220** L1+ scaffold operational mechanism (N/A; no L1+ substrate lane modified; sister B is research-signal advisory only)
- **Catalog #226** trainer auth-eval canonical helper (N/A; no paid auth-eval dispatched)
- **Catalog #229** premise-verification-before-edit ✓ (PV documented above)
- **Catalog #230** ownership map (READ-ONLY consumer imports for all canonical helpers; sister-disjoint from all in-flight sisters confirmed)
- **Catalog #240** recipe-vs-trainer-state consistency (N/A; no recipe emitted; sister B is research-signal advisory only)
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
- **Catalog #307** paradigm-vs-implementation classification (PARADIGM-INTACT + IMPLEMENTATION-LEVEL-EXPLORATION-AT-LARGER-N)
- **Catalog #309** horizon_class declaration (plateau_adjacent; UNCHANGED from V15)
- **Catalog #313** probe-outcomes ledger ✓ (registered: probe_id=`v15_uniward_sister_b_reactivation_criterion_1_n200_pairs_20260527T034044Z`; verdict=PARTIAL; blocker_status=advisory; expires_at_utc=2026-06-26T04:02:55Z; staleness_window_days=30)
- **Catalog #316** reports/latest.md frontier alignment (N/A; sister B is NOT frontier-crossing; reports/latest.md NOT mutated)
- **Catalog #317/#341** canonical-routing markers for local research signal (all `[macOS-CPU advisory]` markers + non-promotable contract honored)
- **Catalog #323** canonical Provenance umbrella ✓ (provenance.json carries every required canonical field)
- **Catalog #335** cathedral consumer canonical contract (sister B emits machine-readable provenance.json + Catalog #313 anchor; future cathedral consumer can integrate as observability-only Tier-A per Catalog #357 dual-tier discipline)
- **Catalog #340** sister-checkpoint guard PROCEED before commit (will run pre-commit per discipline; checked sister activity above)
- **Catalog #343** NO hardcoded frontier score literals (cite canonical_frontier_pointer.json paths; V15 baseline anchors are CONTROLLED-COMPARISON empirical numbers carried via V15 sister memo per Catalog #110 HISTORICAL_PROVENANCE)
- **Catalog #344** canonical equation anchor status: `uniward_per_lut_index_distortion_weight_savings_v1` FORMALIZATION_PENDING preserved (per `# FORMALIZATION_PENDING:per_v15_landing_memo_macos_cpu_advisory_does_not_promote_pending_paired_cuda_evidence_at_higher_dynamic_range_and_or_finer_lut_granularity_per_reactivation_criteria_2_and_3` substantive rationale)
- **Catalog #346** canonical roster N/A (no T2+ deliberation invoked)
- **Catalog #348** retroactive sweep NOT TRIGGERED (sister B is IMPLEMENTATION-LEVEL not PARADIGM-LEVEL)
- **Catalog #356/#357** per-axis decomposition + Tier B dual-tier consumer architecture (sister B is Tier A observability-only by construction; no per-axis Tier B emission)
- **CLAUDE.md "Apples-to-apples evidence discipline"** ✓ (intra-test LUT-statistic comparison at same fixture + same canonical helpers + smoke-mode verifies N=50 baseline reproduction)
- **CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"** ✓ (macOS-CPU advisory explicitly tagged + non-promotable + result_review_blockers cite "requires_paired_cuda_cpu_validation_per_v15_protocol_1e_4_threshold")
- **CLAUDE.md "Forbidden premature KILL without research exhaustion"** ✓ (5 reactivation criteria above; PARADIGM-INTACT classification; KILL is the LAST RESORT)
- **CLAUDE.md "MPS auth eval is NOISE"** ✓ (CPU only; no MPS used; tagged `[macOS-CPU advisory]`)
- **CLAUDE.md "MLX-first portable-local-substrate authority"** ✓ (probe is numpy + torch CPU only; ZERO MLX dependency; canonical N+1 cache sidecar at sister path)
- **10th standing directive** (apples-to-apples + online research authorized + math/science/engineering rigor) ✓ (intra-test apples-to-apples per the 3-question discipline; NO WebSearch invoked for this lane scope but authorized; math rigor binding via canonical Fridrich UNIWARD primitive + canonical Sallee 2003 weighted-median + canonical Holub-Fridrich-Denemark 2014 inverse-Fisher-info routing)
- **11th standing directive** (final rate attack off-the-shelf + development workflow + enhancement) ✓ (sister B demonstrates the canonical $0 macOS-CPU advisory dev-loop workflow that compounds V15 paired-CUDA evidence)
- **12th standing directive** (3-strategy attack framework: RATE / DISTORTION / FULL SCORER) ✓ (sister B sits on DISTORTION pure-axis; LUT-statistic surface; orthogonal to RATE axis from FEC family + FULL SCORER axis from V15 paired-CUDA)
- **13th OPTIMAL-TRIO standing directive** (MLX-first + numpy-portable + individually-fractal) ✓ (TECHNIQUE = canonical UNIWARD per-LUT-index primitive; WAY = canonical N+1 cache extension at apples-to-apples sample size 4× larger; TIME = pre-dispatch PV per Catalog #229 + V15 protocol; sister-disjoint UNIQUE-AND-COMPLETE-PER-METHOD per the canonical helper consumer-only pattern)

## NOT a PR111 candidate — explicit declaration

Per V15 protocol + Catalog #192/#343: PR111 candidate status requires paired-CUDA frontier-crossing on 1:1 contest-compliant hardware (Linux x86_64 + NVIDIA GPU). Sister B is `[macOS-CPU advisory]` — by construction NOT 1:1 contest-compliant; the bins_changed/mean_l2/dyn_range metrics are LUT-statistic research signals NOT contest scores. Per the `user_pr_attribution` memory non-negotiable + CLAUDE.md "FORBIDDEN CLAUDE ATTRIBUTION IN PUBLIC-PR SURFACES": sister B does NOT produce a PR111 candidate report; no public-PR-facing artifact is written.

## Operator-routable next steps

1. **Sister C spawn** (reactivation criterion #2 + #3 combined recommendation): higher real-scorer dynamic range (longer training / lower learning rate / scorer-architecture diversity) AND finer LUT granularity (extend v8 to 32 levels × 5 classes OR 16 levels × 10 classes); estimated $0 macOS-CPU advisory ~10-20 min wall-clock per cache-extension iteration; tests whether the partial-scaling signal compounds across the upstream gradient axis (dyn_range) + downstream LUT-shape axis (granularity)
2. **Sister D spawn** (reactivation criterion #4): apply UNIWARD per-LUT-index routing to NSCS06 grayscale_lut OR VQ-VAE indices_blob — different application surface family; sister B's INTERMEDIATE_PARTIAL_SCALING signal at v8's 16×5 LUT may differ structurally on other substrate-class LUTs
3. **Sister E spawn** (reactivation criterion #5): UNIWARD-weighted EXTREMA replaces UNIWARD-weighted median — sister B's MIXED signal (more bins, smaller per-bin delta) hints that the EXTREMA may concentrate the routing effect more strongly on high-weight-pixel anchors
4. **Sister F spawn** (paired-CUDA validation of sister B): IFF sister C/D/E produces a sister LUT with sister-B-or-stronger N=200 LUT-level signal, queue paired-CUDA validation via canonical `tools/dispatch_modal_paired_auth_eval.py` at estimated $0.50-1.00 cost (matches V15 budget); the paired-CUDA delta IS the canonical PROMOTION arbiter per the V15 1e-4 threshold protocol
5. **NOT-RECOMMENDED**: re-running V15 paired-CUDA at N=200 directly (the sister B result shows the LUT-statistic signal is intermediate; without sister C/D/E's combined-criterion compounding, the paired-CUDA delta at N=200 will likely remain BELOW the 1e-4 PROMOTION threshold; better to compound criteria #2 + #3 first then test the combined result with paid dispatch)

## Cost

- $0 GPU spend (macOS CPU only; no Modal / Vast.ai / Lightning dispatch)
- ~5.5 min CPU wall-clock on M5 Max (213s cache extension for 150 new pairs + 75s LUT derivation + auth-eval ~40s + provenance write)
- ~570 LOC standalone probe script (~28KB)
- ~400 LOC landing memo (~26KB; this file)
- 1 Catalog #313 probe-outcome ledger entry registered (verdict=PARTIAL; blocker_status=advisory)
- 0 canonical equations PROMOTED (FORMALIZATION_PENDING preserved per Catalog #344)
- 1 sister cache sidecar emitted at `experiments/results/v15_uniward_sister_b_reactivation_criterion_1_n200_macos_cpu_advisory_20260526/real_scorer_gradients_cache_extended_n200.npz` (sister-disjoint; APPEND-ONLY; V15 cache untouched)
- 5 reactivation criteria documented (one COMPLETE: #1; four operator-routable: #2/#3/#4/#5)

## Sister-coordination + checkpoint discipline confirmation

5 subagent checkpoints emitted via canonical `tools/subagent_checkpoint.py` per Catalog #206:
1. Step 1: PV launch + plan
2. Step 2: PV complete + scope analysis (V15 protocol read + cache constraint understood + 10th apples-to-apples directive read + sister disjoint via `subagent_progress.jsonl`)
3. Step 3: smoke test PASS (pipeline correctly reproduces V15 N=50 baseline exactly) + launch full N=200 probe
4. Step 4: empirical INTERMEDIATE_PARTIAL_SCALING verdict + dispatch decisions documented
5. Step 5 (final): landing memo + canonical commit via serializer

Sister-disjoint per Catalog #230: NO collision with in-flight sisters (5D canvas populator / Phase 5+6 submission_packet linter+compliance / build-2-3 extended operators / Cascade C' Wave 7 / V15 sister-completion before my work). Catalog #340 sister-checkpoint guard expected PROCEED on commit (no overlap on probe_uniward_lut_n200.py / provenance.json / real_scorer_gradients_cache_extended_n200.npz / this landing memo).

## V15 13th OPTIMAL-TRIO declaration per 13th standing directive

Sister B honors the 13th OPTIMAL-TRIO standing directive (mlx-first-numpy-portable-individually-fractal):
- **MLX-first**: extended N+1 cache extends V15's MLX-local-captured cache; canonical UNIWARD per-LUT-index integration is MLX-portable (pure numpy with MLX-portability hooks per sister `weight_map_per_lut_index.py` docstring)
- **numpy-portable**: probe script + sister cache + LUT derivation are pure numpy + torch CPU; ZERO MLX dependency at runtime; inflate-side runtime UNTOUCHED per HNeRV parity L4
- **individually-fractal**: sister B is unique-per-method (sister-disjoint READ-ONLY consumer imports from V15 / V14-V2 / 5th+6th+7th-order siblings + zero modification to canonical helpers); the probe script does NOT extend V15's build script — it composes the canonical helpers at the per-method optimal point per the canonical sister-B pattern

Per the 12th 3-strategy attack framework standing directive: sister B sits on the DISTORTION pure-axis attack strategy at the LUT-derivation surface; the empirical INTERMEDIATE_PARTIAL_SCALING verdict does NOT block the 11 sister substrates on PLATEAU + 13 sister substrates on FRONTIER + 4 sister substrates on ASYMPTOTIC per the comprehensive roadmap synthesis; the next iteration of UNIWARD per-LUT-index routing is queued via reactivation criteria #2/#3 (combined) per Catalog #313 probe-outcomes ledger.

Per the 11th final-rate-attack off-the-shelf standing directive: sister B demonstrates the canonical $0 macOS-CPU advisory dev-loop workflow under operator-session-budget discipline ($0 spend / unlimited 5-30 min wall-clock); the canonical 4-layer pattern is honored (canonical N+1 cache extension + canonical UNIWARD per-LUT-index integration + Catalog #313 probe-outcomes ledger + provenance.json sidecar with full canonical Provenance).

Per the 10th apples-to-apples + online research authorization + math/science/engineering rigor standing directive: sister B EMPIRICALLY VALIDATES apples-to-apples via the smoke-mode regression guard (reproduces V15 N=50 baseline EXACTLY); WebSearch was NOT invoked this lane (canonical Fridrich + Sallee + Holub-Fridrich-Denemark already in V15 landing memo; no new paper grounding needed for this probe); math rigor binding via Shannon-rate-distortion principle (V15 PROMOTION threshold of 1e-4 paired-CUDA delta IS the canonical apples-to-apples science-rigor bound); engineering rigor binding via the 13 HNeRV parity inviolable lessons + Catalog #290/#294/#296/#297/#303/#305 design-memo discipline.

Per the 7th AUTOMATED + COMPOUNDING + OPTIMAL META-principle standing directive: sister B emits canonical artifacts that compound into future iterations (provenance.json + sister cache sidecar + Catalog #313 probe-outcomes ledger entry + FORMALIZATION_PENDING tracking); the probe script IS the AUTOMATED reproducibility surface for sister C/D/E reactivation criteria #2-#5; the INTERMEDIATE_PARTIAL_SCALING verdict + 39× upstream dyn_range expansion is the OPTIMAL signal for the recommended criteria #2 AND #3 combined next-iteration spawn.
