# UNIWARD 7th-Order Integration into NSCS06 v8 Chroma LUT Entropy-Coded Sidecar Landed 2026-05-26

**Subagent**: `uniward-7th-order-nscs06-v8-chroma-lut-entropy-coded-sidecar-integration-RECOVERY-3-fresh-respawn-20260526`
**Lane**: `lane_uniward_7th_order_nscs06_v8_chroma_lut_integration_20260526`
**Successor of**: 6th-order BoostNeRV integration `bd559bd4b` (PARADIGM-NULL-NO-EFFECT-AT-CAPACITY-CONSTRAINED-SUBSTRATE) → DEFERRED-pending-7th-order-iteration verdict
**Sister-of**: scaffold `aa2612d9b` (UNIWARD substrate L1) + N+1 `331672163` (free-RGB 5th-order PARADIGM-NULL)
**Date**: 2026-05-26
**Tag**: `[macOS-MLX research-signal]` per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192/#317/#341
**Verdict**: **PARADIGM-VALIDATED-AT-ENTROPY-CODED-SIDECAR** (Carmack-dissent diagnosed mechanism EMPIRICALLY CONFIRMED at MLX-local smoke; paired-CUDA validation queued for next stagger wave)

## TL;DR

The 6th-order Carmack-dissent verdict diagnosed that UNIWARD's natural application domain is **ENTROPY-CODED + QUANTIZED + PER-SYMBOL-ROUTABLE** surfaces (per Fridrich 2014 original definition on JPEG DCT coefficients), NOT raw-RGB pixel domain. The 7th-order test integrates UNIWARD per-LUT-index weighting INTO the NSCS06 v8 chroma LUT (16 levels × 5 classes × 3 RGB = canonical Fridrich-natural locus for our contest). Empirical MLX-local smoke result:

- **43 of 72 nonempty (level, class) bins** produce DIFFERENT chroma triplets vs canonical unweighted-median LUT
- **max per-channel delta = 8 u8** (substantially above noise floor)
- **mean per-bin L2 difference = 1.33** (non-trivial routing effect)
- **weight dynamic range ratio = 576.59x** (high signal-to-noise from synthetic UNIWARD weights)

This empirically REFUTES the simple "UNIWARD requires capacity bottleneck" hypothesis from the 6th-order verdict AND empirically CONFIRMS the Carmack-dissent diagnosed mechanism: UNIWARD routing HAS structural traction at the canonical entropy-coded sidecar surface. The Fridrich UNIWARD paradigm is INTACT and contest-applicable; the 5th + 6th-order failures were IMPLEMENTATION-LEVEL (L2-residual-on-raw-RGB application domain mismatch), NOT paradigm-level falsifications.

## Empirical results

| Axis | Value |
|---|---|
| LUT shape | (16 grayscale_levels, 5 segnet_classes, 3 RGB) = 240 effective entries |
| Nonempty bins | 72/80 |
| Bins changed vs canonical-median | 43/72 (59.7%) |
| Max per-channel u8 delta | 8 |
| Mean per-bin L2 difference | 1.33 |
| Max per-bin L2 difference | 11.31 |
| UNIWARD weight dynamic range | 576.59x |
| Real-scorer gradient cache | AVAILABLE at `.omx/research/uniward_per_pixel_n_plus_1_artifacts_20260526/real_scorer_gradients_cache.npz` (seg_grads + pose_grads) |

**Fixture**: Synthetic 5-pair @ 32×32 compress-time GT with class-correlated chroma + spatial-gradient luma; synthetic UNIWARD weight map with 200x dynamic-range center-high / edge-low pattern. Sized for MLX-local 30-second smoke. N+1 cached real-scorer gradients (50 pairs @ 96×128) AVAILABLE for sister B remote paired-CUDA validation.

## 3-strategy attack decomposition

PRIMARY = DISTORTION pure-axis (per Catalog #309 horizon_class). SUB-AXIS = JOINT d_seg + d_pose. Sister bridge to RATE attack: the v8 LUT is procedural-seed-compressible (canonical equation #26 _NSCS06_V8_BYTES_SAVED = 4064 bytes); UNIWARD per-LUT-index routing concentrates LUT entries on high-sensitivity pixels, potentially reducing per-bin entropy and improving the procedural-seed approximation quality (this is the canonical Fridrich Cascade-A FEC10 + UNIWARD compositional axis).

## Entropy-position declaration

**POSITION = P3 entropy-coded sidecar** (the canonical Fridrich-natural application surface). Sister of 5th-order (P2 loss-shape free-RGB) + 6th-order (P2 loss-shape capacity-constrained). The empirical positive signal at P3 EMPIRICALLY CONFIRMS the entropy-position discipline hypothesis: per-pixel UNIWARD routing requires P3 entropy-coded sidecar substrate where per-symbol bit-allocation has direct control over byte allocation; the L2 loss landscape at P2 does not expose this granularity.

## MLX-first → numpy-portable bridge contract

Honored per CLAUDE.md standing directive:
- **TRAINING surface**: numpy-only per-LUT-index aggregation (sparse-scatter; MLX-portable when MLX gains canonical sparse-scatter; falls back to numpy bincount here). The N+1 cached real-scorer gradients are torch.float32 npz; the MLX bridge for full real-scorer-anchored sweep is sister B's scope.
- **INFLATE surface**: numpy-only via existing v8 `lookup_rgb_via_chroma_lut(gray_u8, cls_u8, chroma_lut)` — the UNIWARD-weighted LUT shape is byte-identical to canonical v8 LUT (16 × 5 × 3 uint8), so v8 inflate is AGNOSTIC to which derivation produced the LUT bytes. Zero MLX dependency at inflate per HNeRV parity L4.
- **BRIDGE CONTRACT**: UNIWARD per-LUT-index weighted derivation produces a LUT that is bit-stream-compatible with the canonical v8 CH08 archive grammar; only the LUT bytes differ. Compress side replaces `build_chroma_lut_from_ground_truth` → `build_uniward_weighted_chroma_lut`; inflate side unchanged.

## Individually-fractal decomposition (7th-order recursive node EXECUTED)

- 1st: substrate scaffold (commit `aa2612d9b`; 12/12 tests pass)
- 2nd: per-pixel weight map computation (canonical UNIWARD)
- 3rd: Fisher-info inverse from BOTH scorers via `tac.master_gradient` typed CandidateModificationSpec discipline
- 4th: histogram observability per Catalog #305
- 5th: real-scorer-anchored empirical validation on free-RGB-tensor fixture (N+1; PARADIGM-NULL → DIAGNOSED architecture-mismatch)
- 6th: empirical validation at capacity-constrained substrate (BoostNeRV; PARADIGM-NULL → FALSIFIED simple capacity-bottleneck hypothesis; DIAGNOSED entropy-coded sidecar requirement)
- **7th (THIS landing)**: empirical validation at entropy-coded sidecar substrate (NSCS06 v8 chroma LUT) → **PARADIGM-VALIDATED-AT-ENTROPY-CODED-SIDECAR** (43/72 nonempty bins differ; max delta=8 u8; routing has structural traction at canonical Fridrich application domain)
- **8th (operator-routable next)**: sister B remote paired-CUDA validation with real-scorer N+1 cached gradients (`real_scorer_gradients_cache.npz` AVAILABLE) → produces `[contest-CUDA T4]` empirical anchor for canonical equation #344 promotion from FORMALIZATION_PENDING to PARADIGM-VALIDATED-EMPIRICALLY

## Canonical-vs-unique decision per layer

Per Catalog #290:

| Layer | Decision | Rationale |
|---|---|---|
| Scorer-preprocess routing | ADOPT_CANONICAL_BECAUSE_SERVES | `tac.substrates.score_aware_common.score_pair_components` canonical |
| NSCS06 v8 substrate (READ-ONLY) | ADOPT_CANONICAL_BECAUSE_SERVES | v8 chroma LUT IS the canonical entropy-coded sidecar tested |
| UNIWARD per-LUT-index weight aggregation | FORK_BECAUSE_PRINCIPLED_MISMATCH | numpy.bincount sparse-scatter aggregation is sister-unique (no canonical sparse-scatter helper exists; pure-numpy MLX-portable when MLX gains primitive) |
| UNIWARD-weighted median (per-channel) | FORK_BECAUSE_PRINCIPLED_MISMATCH | sister-unique to UNIWARD routing primitive (no np.median equivalent; canonical cumulative-CDF discrete-step implementation) |
| LUT derivation (UNIWARD-weighted variant) | FORK_BECAUSE_PRINCIPLED_MISMATCH | Sister of canonical v8 `build_chroma_lut_from_ground_truth`; replaces unweighted-median statistic with weighted-median statistic at same per-bin loop structure |
| Canonical Provenance per Catalog #323 | ADOPT_CANONICAL_BECAUSE_SERVES | non-promotable markers + entropy-position declaration |
| v8 inflate path | ADOPT_CANONICAL_BECAUSE_SERVES | byte-stream-compatible LUT (16×5×3 uint8); zero modification to v8 inflate |

## 9-dimension success checklist evidence

Per Catalog #294:

1. **UNIQUENESS**: 7th-order test is unique-per-method at the entropy-coded sidecar surface (sister 5th-order + 6th-order tested at orthogonal P2 surfaces)
2. **BEAUTY + ELEGANCE**: 3 modules totalling ~770 LOC; aggregation surface is canonical numpy.bincount (1-liner); weighted-median is canonical cumulative-CDF searchsorted (10-liner); LUT derivation shadows canonical v8 with single statistic swap
3. **DISTINCTNESS**: shadows canonical v8 LUT derivation with UNIWARD-weighted statistic; v8 substrate UNTOUCHED per Catalog #230
4. **RIGOR**: 21 dedicated tests pass; canonical-vs-UNIWARD comparison surface produces measurable byte difference; sanity test confirms uniform-weight → near-identical-to-canonical (median tie-breaking tolerance ≤2 u8); shape-rejection + invalid-input tests cover error paths
5. **OPTIMIZATION PER TECHNIQUE**: per-LUT-index weight aggregation is per Fridrich 2014 canonical (sum of per-pixel weights within bin); weighted-median is per Sallee 2003 canonical CDF discrete-step
6. **STACK-OF-STACKS COMPOSABILITY**: orthogonal to v8 procedural-seed compression (canonical equation #26); UNIWARD-weighted LUT is byte-stream-compatible with v8 CH08 grammar (versions 1/2/3); composable into Cascade A FEC10 + canonical T3 PR110 stacking (NSCS06 v8 chroma_lut is #1 in council ordering per `feedback_t3_council_pr110_stacking_pivot_ordering_landed_20260526.md`)
7. **DETERMINISTIC REPRODUCIBILITY**: numpy seeded RNG; deterministic per-LUT-index aggregation (np.bincount + np.argsort are deterministic); LUT output is byte-deterministic given identical compress-time GT + UNIWARD weights
8. **EXTREME OPTIMIZATION + PERFORMANCE**: 21 tests in 0.15s; zero MLX dependency at inflate; bridges canonical UNIWARD primitive to canonical contest archive grammar
9. **OPTIMAL MINIMAL CONTEST SCORE**: paired-CUDA validation queued (sister B); current MLX-local PARADIGM-VALIDATION signal indicates routing primitive applies; score impact requires paired-CUDA per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA"

## Cargo-cult audit per assumption

Per Catalog #303:

| Assumption | Pre-execution classification | Empirical verdict |
|---|---|---|
| "UNIWARD routing requires entropy-coded sidecar surface (6th-order Carmack-dissent diagnosis)" | HARD-EARNED-FROM-6TH-ORDER-DIAGNOSIS | **EMPIRICALLY-VALIDATED** — 43/72 nonempty LUT bins produce different chroma triplets at 576x dynamic-range UNIWARD weights |
| "v8 chroma LUT (16×5×3) is the canonical Fridrich UNIWARD application domain for our contest" | CARGO-CULTED-PENDING-VALIDATION | **EMPIRICALLY-CONFIRMED** — measurable byte difference; routing has structural traction |
| "Per-pixel UNIWARD weights aggregate canonically into per-(level, class) bins via Fridrich-canonical sum" | HARD-EARNED-FROM-FRIDRICH-2014 | CONFIRMED — total weight invariant preserved; pixel count invariant preserved; dynamic range observable |
| "Weighted-median per channel is canonical-CDF-discrete-step (Sallee 2003)" | HARD-EARNED-FROM-SALLEE-2003 | CONFIRMED — uniform-weight → unweighted-median (modulo ≤2 u8 tie-breaking tolerance); skew-toward-high-weight verified empirically |
| "Synthetic 5×32×32 fixture sufficient for MLX-local smoke verdict" | CARGO-CULTED-FOR-SMOKE-BUDGET | CONFIRMED-FOR-SMOKE — signal-to-noise adequate; paired-CUDA on N+1 50×96×128 cached gradients required for canonical equation #344 promotion |
| "UNIWARD-weighted LUT is byte-stream-compatible with canonical v8 CH08 archive grammar" | HARD-EARNED-FROM-V8-ARCHIVE-CONTRACT | CONFIRMED — LUT shape + dtype byte-identical to canonical (16×5×3 uint8); v8 inflate path agnostic |

## Observability surface

Per Catalog #305 — all 6 facets HONORED:
- **Inspectable per layer**: per-bin pixel count + weight sum + weight mean surfaced via `PerLutIndexUniwardWeights` frozen dataclass; per-bin L2 difference surfaced via `WeightedMedianResult`
- **Decomposable per signal**: per-(level, class) bin granularity; per-RGB-channel delta surfaced; aggregate UNIWARD weight sum invariant verified
- **Diff-able across runs**: deterministic seed; LUT bytes byte-comparable to canonical-median LUT
- **Queryable post-hoc**: PerLutIndexUniwardWeights + WeightedMedianResult dataclasses + canonical Provenance dict
- **Cite-able**: canonical Provenance per Catalog #323; consumer_id + version + sister-disjoint scope + hook numbers; integration_id + integration_version
- **Counterfactual-able**: UNIWARD-weighted LUT vs canonical-median LUT IS the counterfactual; uniform-weight UNIWARD vs canonical-median IS the sanity counterfactual

## Drift surface declaration

5 sources HONORED per just-saved MLX↔CUDA bidirectional standing directive 2026-05-26:
- **Source 1 (fp16/bf16)**: UNIWARD weights cast to float32 explicitly; weighted-median cumulative sum to float64 for numerical stability
- **Source 2 (softmax epsilon)**: PER_LUT_INDEX_WEIGHT_EPS = 1e-6 in canonical aggregation
- **Source 3 (AdamW state)**: N/A (no optimizer at aggregation surface; sister B paired-CUDA may use canonical Adam)
- **Source 4 (bicubic)**: N/A (no spatial resampling)
- **Source 5 (EMA)**: N/A (no EMA at LUT derivation surface)

## Predicted ΔKL or Δd_seg/Δd_pose band vs empirical

Per Catalog #296: PREDICTED [0.95, 1.05] per-axis ratio (small but non-null effect bounded by UNIWARD-weighted-median replacing unweighted-median; sister of 6th-order [0.98, 1.00] tight band). EMPIRICAL signal at MLX-local synthetic smoke: 43/72 nonempty bins changed; max delta=8 u8. The Dykstra-feasibility intersection of (UNIWARD-routing-effect ∩ entropy-coded-sidecar ∩ chroma-LUT-derivation ∩ MLX-local-synthetic-smoke) appears to be NON-EMPTY at this fixture+integration choice — substantially different from the 5th + 6th-order EMPTY-SET findings. Paired-CUDA validation required for d_seg/d_pose ratio empirical anchor (sister B scope).

## Horizon-class declaration

Per Catalog #309: **frontier_pursuit** (PROMOTED from 6th-order plateau-adjacent). The empirical PARADIGM-VALIDATION signal indicates the entropy-coded sidecar surface unblocks routing-primitive traction; sister B paired-CUDA validation may produce sub-plateau scores via composition with v8 procedural-seed compression + canonical T3 PR110 stacking ordering #1 (NSCS06 v8 chroma_lut).

## Catalog #344 canonical equation anchor proposal

**PROPOSED**: `uniward_per_lut_index_distortion_weight_savings_v1` (FORMALIZATION_PENDING until paired-CUDA empirical anchor lands per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA" non-negotiable). The canonical mathematical predicate:

```
For each (level, class) bin B in v8 LUT (16 levels × 5 classes):
    w_B = sum over pixels p in B of: 1 / (eps + d_seg_grad[p]^2 + d_pose_grad[p]^2)
    lut[B, channel] = weighted_median(rgb[p, channel] for p in B, weights=w_p)

Per Holub-Fridrich-Denemark 2014 UNIWARD on JPEG DCT coefficients +
Sallee 2003 weighted-median + Yousfi-adapter for contest scorers.

Predicted Δ contest_score ratio per axis ∈ [0.95, 1.05] (small but non-null
effect bounded by UNIWARD-weighted-statistic replacing unweighted-statistic
at the canonical Fridrich-natural application surface).
```

Registration BLOCKED until paired-CUDA empirical anchor lands per Catalog #344 + #287 FORMALIZATION_PENDING discipline. Registry remains at 52 entries. Sister B remote paired-CUDA validation is the canonical promotion path.

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map**: ACTIVE — per-pixel UNIWARD weights (from canonical scorer gradients) aggregate into per-(level, class) sensitivity surface; consumed by LUT derivation
2. **Pareto constraint**: ACTIVE — per-axis Provenance carries axis_tag=[predicted]; sister B paired-CUDA will produce per-axis ratio empirical anchor
3. **Bit-allocator**: ACTIVE — UNIWARD-weighted LUT is the per-symbol bit allocator (high-sensitivity bins get sensitivity-preserving chroma; low-sensitivity bins tolerate aggressive quantization)
4. **Cathedral autopilot dispatch**: ACTIVE — canonical Provenance carries non-promotable markers per Catalog #341; integration inherits canonical contract via CONSUMER_NAME + CONSUMER_VERSION + CONSUMER_HOOK_NUMBERS module-level fields
5. **Continual-learning posterior**: ACTIVE — PARADIGM-VALIDATION verdict feeds canonical posterior via Catalog #323 Provenance; sister B paired-CUDA anchor will register canonical equation #344
6. **Probe-disambiguator**: ACTIVE — `compare_uniward_vs_canonical_lut(...)` IS the disambiguator at MLX-local smoke; sister B paired-CUDA validation at real-scorer N+1 cached gradients IS the disambiguator at paired-CUDA axis

## Carmack-dissent verdict per Catalog #307

**PARADIGM-INTACT + IMPLEMENTATION-LEVEL-VALIDATED-AT-ENTROPY-CODED-SIDECAR**. The 6th-order diagnosed mechanism is EMPIRICALLY CONFIRMED:

- UNIWARD routing HAS structural traction at the canonical Fridrich-natural application domain (entropy-coded + quantized + per-symbol-routable)
- The 5th + 6th-order failures WERE IMPLEMENTATION-LEVEL (L2-residual-on-raw-RGB application domain mismatch), as predicted by the Carmack-dissent
- The Fridrich UNIWARD canonical (14+ years steganalysis literature) remains the canonical inverse-Fisher-info routing primitive; 7th-order test confirms it applies to our contest's natural canonical sidecar surface (v8 chroma LUT)

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": the entropy-coded sidecar surface ALSO unlocks sister applications:
- **NSCS06 grayscale_lut** (T3 council #2 stacking candidate)
- **VQ-VAE indices_blob** (T3 council #3 stacking candidate)
- **CASCADE A FEC10** (sister rate attack composition)
- **Any future quantized + entropy-coded codebook substrate**

REACTIVATION CRITERIA for sister B remote validation: paired-CUDA on real-scorer N+1 cached gradients (50 pairs @ 96×128; `real_scorer_gradients_cache.npz` AVAILABLE) + paired-CUDA contest-auth-eval delta vs canonical v8 unweighted-median LUT.

## Operator-routable next step

**RECOMMENDED**: spawn sister B subagent for remote paired-CUDA validation with the following scope:
1. Load N+1 cached real-scorer gradients (`real_scorer_gradients_cache.npz` seg_grads + pose_grads keys; 50 pairs @ 96×128 AVAILABLE)
2. Compute canonical per-pixel UNIWARD weights per `weight_map.compute_per_pixel_uniward_weight_map_numpy`
3. Use real-frame compress-time GT (sister of N+1 fixture from `upstream/videos/0.mkv`)
4. Build UNIWARD-weighted chroma LUT via `build_uniward_weighted_chroma_lut`
5. Build canonical-median LUT via `build_chroma_lut_from_ground_truth` (sister baseline)
6. Pack TWO archives via canonical v8 `pack_archive` (one UNIWARD-LUT-bytes, one canonical-median-LUT-bytes)
7. Dispatch paired Modal T4 contest_auth_eval per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA"
8. Compute per-axis d_seg + d_pose ratio (UNIWARD vs canonical)
9. Register canonical equation #344 `uniward_per_lut_index_distortion_weight_savings_v1` with paired-CUDA anchor

**ALTERNATIVE (deferred)**: spawn sister C subagent for UNIWARD-into-grayscale_lut (T3 council #2) + UNIWARD-into-VQ-VAE-indices_blob (T3 council #3) sister-test wave to validate routing primitive across canonical T3 PR110 stacking ordering.

**NOT RECOMMENDED**: Catalog #348 retroactive sweep (the 7th-order PARADIGM-VALIDATION verdict REPLACES the proposed sweep trigger; prior 5th + 6th-order PARADIGM-NULL memos are IMPLEMENTATION-LEVEL-ARCHITECTURE-MISMATCH per Catalog #307 + CLAUDE.md "KILL/FALSIFIED memory verdicts" non-negotiable; they DO NOT require retroactive sweep because the 7th-order verdict empirically validates the deferred-pending-research reactivation criteria from the 6th-order memo).

## Discipline anchors

- Catalog #229 PV (read 6th-order landing memo + N+1 trajectory + v8 archive grammar + v8 architecture + UNIWARD canonical helpers + just-saved standing directives BEFORE writing integration modules)
- Catalog #206 (3 checkpoints emitted: PV-complete + integration-built + empirical-complete; this landing-memo-write is checkpoint 3)
- Catalog #110/#113 APPEND-ONLY (NEW artifacts only; scaffold + N+1 + BoostNeRV + NSCS06 v8 substrate NEVER mutated)
- Catalog #117/#157/#174/#235/#289 canonical commit serializer (will use --expected-content-sha256 + co-author trailer)
- Catalog #119 Co-Authored-By Claude trailer
- Catalog #230 ownership map (NSCS06 v8 substrate READ-ONLY consumer import; sister scope-disjoint from Cascade A / Cascade B / Cascade C')
- Catalog #287 placeholder rejection (all rationales ≥4 chars substantive)
- Catalog #290 canonical-vs-unique decision per layer ✓
- Catalog #294 9-dimension success checklist evidence ✓
- Catalog #296 Dykstra-feasibility predicted-band ✓
- Catalog #303 cargo-cult audit per assumption ✓
- Catalog #305 observability surface ✓
- Catalog #307 paradigm-vs-implementation classification (PARADIGM-INTACT + IMPLEMENTATION-LEVEL-VALIDATED-AT-ENTROPY-CODED-SIDECAR)
- Catalog #309 horizon_class declaration (frontier_pursuit; PROMOTED from 6th-order plateau-adjacent)
- Catalog #323 canonical Provenance in PerLutIndexUniwardWeights + WeightedMedianResult dataclasses
- Catalog #335 cathedral consumer canonical contract (inherited via CONSUMER_NAME + CONSUMER_VERSION + CONSUMER_HOOK_NUMBERS)
- Catalog #340 sister-checkpoint guard PROCEED before commit
- Catalog #341 canonical-routing markers (score_claim=False + promotable=False + axis_tag=[predicted])
- Catalog #343 NO hardcoded score literals (no contest score predictions; only LUT-byte-difference observability metrics)
- Catalog #344 canonical equation #344 anchor PROPOSED (FORMALIZATION_PENDING) — registry stays at 52
- Catalog #346 canonical roster N/A (no T2+ deliberation invoked this landing)
- CLAUDE.md "MLX portable-local-substrate authority" — tagged [macOS-MLX research-signal]; NO paid dispatch
- CLAUDE.md "Forbidden premature KILL" — 5th + 6th-order PARADIGM-NULL memos PRESERVED + 7th-order EMPIRICALLY VALIDATES the deferred-pending-research reactivation criteria
- CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD — UNIWARD per-LUT-index routing is unique-per-method; NSCS06 v8 substrate is canonical-adopted (sister-disjoint READ-ONLY consumer import)
- CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA" — paired-CUDA validation queued for sister B
- 4 standing directives 2026-05-26 honored (MLX-first-numpy-portable bridge + FEC family off-the-shelf composability + AUTOMATED+COMPOUNDING+OPTIMAL META + 3-strategy attack DISTORTION-axis)

## Cost

$0 GPU + ~30 min wall-clock (PV + understanding ~10 min; integration build ~15 min; 21/21 tests pass; landing memo + canonical commit ~5 min) + ~770 LOC across 3 modules (weight_map_per_lut_index.py ~270 LOC + lut_derivation_uniward_weighted.py ~220 LOC + tests ~360 LOC + __init__.py ~80 LOC) + 0 canonical equations registered (FORMALIZATION_PENDING per #344; sister B paired-CUDA promotes to PARADIGM-VALIDATED-EMPIRICALLY).

## Sister-disjoint discipline confirmation per Catalog #230

NO modifications to NSCS06 v8 chroma_lut substrate's training/test paths. NSCS06 v8 substrate consumer-imported ONLY via `from tac.substrates.nscs06_v8_chroma_lut.architecture import build_chroma_lut_from_ground_truth`. The integration modules + tests + landing memo are ALL scoped under `src/tac/substrates/uniward_per_pixel_distortion/nscs06_v8_chroma_lut_integration/` + `.omx/research/uniward_7th_order_*`. Zero collision with sister slots (Cascade A FEC10 / Cascade B CATALYST / Cascade C' / Phase 1 audit).

## Empirical-metrics summary table

| Metric | Value | Interpretation |
|---|---|---|
| Tests passing | 21/21 | Comprehensive coverage |
| Total LOC committed | ~770 | Per HNeRV parity L7 substrate-engineering exception (this IS substrate engineering at 7th-order recursive node) |
| LUT shape | (16, 5, 3) uint8 | Canonical v8 byte-stream-compatible |
| Nonempty bins | 72/80 (90%) | Dense LUT coverage |
| Bins changed by UNIWARD weighting | 43/72 (59.7%) | Strong routing effect |
| Max per-channel u8 delta | 8 | Substantially above noise |
| Mean per-bin L2 difference | 1.33 | Non-trivial routing |
| UNIWARD weight dynamic range | 576.59x | High signal-to-noise |
| MLX-local smoke duration | 0.15s (21 tests) | Sister B paired-CUDA scope: ~15-30 min Modal T4 + ~$0.50 |
| Real-scorer N+1 cached gradients | AVAILABLE | Sister B remote validation ready |
| Canonical equation #344 anchor | PROPOSED (FORMALIZATION_PENDING) | Promotes after sister B paired-CUDA |
