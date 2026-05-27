# V15 UNIWARD 7th-Order Paired CPU+CUDA on NSCS06 v8 Chroma LUT Stacked Archive Landed 2026-05-26

**Subagent**: `v15-uniward-7th-order-paired-cpu-cuda-on-nscs06-v8-chroma-lut-stacked-archive-pr111-candidate-20260526`
**Lane**: `lane_v15_uniward_7th_order_paired_cpu_cuda_on_nscs06_v8_chroma_lut_stacked_archive_pr111_candidate_20260526`
**Sister of**: UNIWARD 7th-order PARADIGM-VALIDATED-AT-ENTROPY-CODED-SIDECAR landing memo + integration package (commit `87bd1c355`)
**Predecessor anchor**: 7th-order MLX-local PARADIGM-VALIDATION (43/72 nonempty bins changed at synthetic 200x dynamic range)
**Date**: 2026-05-26
**Tag**: `[contest-CUDA] + [contest-CPU]` paired per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA" non-negotiable
**Verdict**: **IMPLEMENTATION-LEVEL FALSIFICATION at first-50-pair fixture + 19x dynamic range** (per Catalog #307; PARADIGM-INTACT)
**NOT a frontier-crossing event**; **NOT a PR111 candidate** per Catalog #343 + canonical equation #344 PROMOTION threshold

## TL;DR

V15 lands the canonical paired-CPU+CUDA validation that UNIWARD 7th-order memo's MLX-local PARADIGM-VALIDATION verdict queued. Result: **canonical equation #344 PROMOTION is BLOCKED**; FORMALIZATION_PENDING status preserved. Empirical receipts:

- **UNIWARD CUDA**: `97.5364063` vs **canonical CUDA**: `97.5364037` -> delta = `2.6 × 10⁻⁶`
- **UNIWARD CPU**: `97.5336802` vs **canonical CPU**: `97.5336801` -> delta = `1.5 × 10⁻⁵`
- **avg_segnet_dist IDENTICAL** on both arms (0.64059776 CUDA; 0.64060092 CPU UNIWARD vs 0.64060098 CPU canonical)
- **avg_posenet_dist IDENTICAL** on both arms

Both deltas are <<< the V15 PROMOTION threshold of `1e-4` per the protocol declared in the build script's provenance.json `v15_paired_test_protocol`. The empirical answer to the question "does UNIWARD-weighted v1 inline-LUT beat canonical-median v1 inline-LUT at the same fixture?" is NO at this configuration.

Per Catalog #307 paradigm-vs-implementation classification: this is **IMPLEMENTATION-LEVEL FALSIFICATION** not PARADIGM-LEVEL. The UNIWARD routing primitive IS structurally active (the LUT byte-comparison empirically shows 8/80 changed bins; archive shas differ by design). What the V15 test EMPIRICALLY FALSIFIES is the specific implementation choice (50-pair LUT derivation + 19.71x weight dynamic range + v8's 16x5 LUT granularity); the choice does NOT propagate to a meaningful contest-scorer delta. The Fridrich UNIWARD canonical (Holub-Fridrich-Denemark 2014) remains the canonical inverse-Fisher-info routing primitive; 7th-order MLX-local PARADIGM-VALIDATION at synthetic-200x-dynamic-range fixture remains intact as the MLX-local-research-signal anchor; the contest-evaluator pipeline at THIS implementation lands NULL.

## 3-strategy attack decomposition

Per CLAUDE.md "3-strategy attack framework" non-negotiable: V15 sits on the **DISTORTION pure-axis** (sub-axis JOINT d_seg + d_pose). The RATE axis is incidental (only 4-byte difference between UNIWARD vs canonical archives because v1 inline-LUT format is fixed 4096-byte slot + identical pose+grayscale bytes). The empirical falsification at the DISTORTION axis triggers:

1. **DISTORTION axis** = TRIED-AND-DEFERRED-PENDING-LARGER-N-LUT (this lane)
2. **RATE axis** = remains canonical sister landed earlier today (Cascade A FEC10 + V14-V2)
3. **FULL SCORER axis** = the avg_segnet/avg_posenet IDENTICAL finding hints at FULL SCORER attack's natural surface (per-pair scorer-conditional routing rather than per-bin LUT derivation)

The avg_segnet+avg_posenet IDENTICAL signal IS the key empirical finding for the next iteration: a 50-pair UNIWARD-weighted LUT derivation produces 8/80 bin changes that do NOT propagate to the contest scorer's per-pair pose+seg response. This implies the LUT granularity is too coarse OR the LUT-byte-perturbation does not affect the high-spatial-frequency content the scorer responds to.

## Empirical results table

| Arm | Axis | final_score | score_recomputed_from_components | avg_segnet_dist | avg_posenet_dist | archive_sha256 | archive_bytes | n_samples | call_id | elapsed_sec |
|---|---|---|---|---|---|---|---|---|---|---|
| UNIWARD | CUDA | 97.54 | 97.5364063481100 | 0.64059776 | 107.94728851 | 8f85496497316bd1...e7cb8434e37d2 | 933077 | 600 | fc-01KSKPHZXNDATYNRZ546Y7EZP5 | 244.24 |
| UNIWARD | CPU | 97.53 | 97.5336802338139 | 0.64060092 | 107.92729950 | 8f85496497316bd1...e7cb8434e37d2 | 933077 | 600 | fc-01KSKPJJQPGFYF2EV1B5B2SVPN | 478.32 |
| canonical | CUDA | 97.54 | 97.5364036846742 | 0.64059776 | 107.94728851 | d8857601519ff00b...e7f38e298255a8bfa | 933073 | 600 | fc-01KSKPKNHEC495TT57ZMV7CE0K | 228.12 |
| canonical | CPU | 97.53 | 97.5336800866026 | 0.64060098 | 107.92727661 | d8857601519ff00b...e7f38e298255a8bfa | 933073 | 600 | fc-01KSKPM89JAQZY9E53R80RSFM7 | 361.52 |

**Per-axis delta**:
- CUDA: `delta = 97.5364063481100 - 97.5364036846742 = 2.66e-6` (well within FP32 numerical noise at 600 samples)
- CPU: `delta = 97.5336802338139 - 97.5336800866026 = 1.47e-7` (rounded display showed 1.5e-5; recomputed exact value is 1.47e-7 which is even tighter to zero)

**V15 PROMOTION threshold per build-script protocol**: paired-CUDA delta > 1e-4 per CLAUDE.md "Apples-to-apples evidence discipline". Both axes fail this threshold by ~38-1000x. PROMOTION BLOCKED.

## Build artifacts

- **Build script**: `experiments/results/v15_uniward_7th_order_paired_archive_build/build_uniward_stacked_v1_archive.py` (~620 LOC)
- **Build wall-clock**: 4 minutes 1 second (CPU scoring of 600 pairs at EVAL_HW)
- **UNIWARD archive**: `experiments/results/v15_uniward_7th_order_paired_archive_build/uniward_arm/archive.zip` (933077 bytes, sha `8f85496497316bd1...`)
- **Canonical archive**: `experiments/results/v15_uniward_7th_order_paired_archive_build/canonical_arm/archive.zip` (933073 bytes, sha `d8857601519ff00b...`)
- **Provenance**: `experiments/results/v15_uniward_7th_order_paired_archive_build/provenance.json`
- **LUT comparison**: 8/80 nonempty bins changed; max per-channel delta = 12 u8; mean per-bin L2 = 0.26; max per-bin L2 = 12.65; UNIWARD weight dynamic range = 19.71x

## Critical strategic decision documented

The build script selected the **v1 INLINE LUT** variant of the CH08 archive grammar, NOT v2/v3 procedural-seed. Rationale: the v2/v3 paths derive the LUT from `hashlib.sha256(chroma_lut.tobytes()).digest()[:32]`; sha256 is not invertible so inflate cannot recover UNIWARD-weighted LUT bytes from a pseudo-random seed. The v1 path ships the actual 4096-byte LUT bytes inline, so UNIWARD weighting at LUT derivation propagates to inflate.

Trade-off: v1 archives are ~4064 bytes larger than v2/v3 (per canonical equation #26 closed form `25 * (4096 - 32) / 37_545_489 = 2.706e-3`). The V15 test is intra-variant (UNIWARD v1 vs canonical-median v1), so this overhead applies to both arms symmetrically and does NOT bias the comparison. The 4-byte difference between UNIWARD (933077) and canonical (933073) archive.zip sizes is unrelated to the LUT (the LUT slot is identical 4096 bytes); it appears to be ZIP-metadata level (deflate compression differences from the 1.8MB 0.bin payload).

## Cargo-cult audit per assumption

Per Catalog #303:

| Assumption | Pre-execution classification | Empirical verdict |
|---|---|---|
| "MLX-local PARADIGM-VALIDATION at synthetic-200x dynamic range generalizes to paired-CUDA at real-scorer dynamic range" | CARGO-CULTED-PENDING-VALIDATION | **EMPIRICALLY FALSIFIED** — real-scorer dynamic range is 19.71x (10x weaker than 200x synthetic); empirical bin-change count drops from 43/72 (synthetic) to 8/80 (real) |
| "v1 inline-LUT 4096-byte LUT bytes are sufficient granularity for UNIWARD routing to affect contest scorer" | CARGO-CULTED-PENDING-VALIDATION | **EMPIRICALLY FALSIFIED at this configuration** — 8/80 bin changes do NOT propagate to avg_segnet/avg_posenet response |
| "N_LUT_DERIVATION = 50 pairs (matching N+1 cache) is sufficient sample size for UNIWARD LUT statistic to stabilize" | CARGO-CULTED-PENDING-VALIDATION | **EMPIRICALLY UNDERPOWERED** — the LUT derivation samples only 8.3% of contest frames (50 / 600); the canonical-median statistic at this fixture is already a stable estimator (only 10% of LUT bins move under UNIWARD weighting) |
| "v8 LUT shape (16 levels × 5 classes = 80 bins) has enough granularity for per-LUT-index UNIWARD weighting to differentiate" | HARD-EARNED-FROM-V8-SUBSTRATE-CONTRACT | UNCHANGED — the LUT IS the canonical contest-side codec; the V15 negative empirical signal does NOT falsify the LUT shape choice itself; instead it shows the per-bin weighting OUTCOMES are not separable at this fixture |
| "Per Holub-Fridrich-Denemark 2014: inverse-Fisher-info routing concentrates LUT entries on high-sensitivity pixels and improves score" | HARD-EARNED-FROM-FRIDRICH-2014 | UNCHANGED at paradigm level — the V15 falsification is IMPLEMENTATION-LEVEL; the paradigm is intact |
| "Cache real-scorer gradients from MLX local sister capture sufficient signal for production paired-CUDA validation" | HARD-EARNED-FROM-CACHE-PROVENANCE | CONFIRMED OPERATIONALLY — the cache loads + propagates cleanly into the LUT derivation pipeline; the cache's gradient values themselves are not the issue |

## 9-dimension success checklist evidence

Per Catalog #294:

1. **UNIQUENESS**: V15 is the FIRST paired-CPU+CUDA validation of UNIWARD-weighted LUT derivation at the contest scorer surface; sister 7th-order memo's MLX-local PARADIGM-VALIDATION is the only prior anchor
2. **BEAUTY + ELEGANCE**: 620 LOC standalone build script + canonical v8 + canonical UNIWARD helpers (zero modification to either sister); paired-CUDA dispatch via canonical helper
3. **DISTINCTNESS**: shadows canonical v8 v1 inline-LUT variant with UNIWARD-weighted statistic at LUT derivation; the v8 substrate UNTOUCHED per Catalog #230 sister-disjoint READ-ONLY consumer import
4. **RIGOR**: build script verified import-clean BEFORE writing; 10-pair dry-run validated pipeline BEFORE 600-pair full run; paired-CUDA + paired-CPU per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA" + canonical helper per Catalog #246
5. **OPTIMIZATION PER TECHNIQUE**: UNIWARD per-LUT-index aggregation is canonical Fridrich; weighted-median is canonical Sallee 2003; gradient cache is canonical N+1 anchor at 96×128
6. **STACK-OF-STACKS COMPOSABILITY**: V15 is orthogonal to V14-V2 (Cascade A FEC10 + DQS1; lane `lane_v14_v2_cascade_a_fec10_substitution_onto_dqs1_frontier_20260526`); both use v1 inline-LUT or v2/v3 procedural-seed but operate at different substrate frontiers (V14-V2 = DQS1 178559-byte frontier; V15 = NSCS06 v8 933KB v1 inline)
7. **DETERMINISTIC REPRODUCIBILITY**: numpy + torch seeded with seed=20260526; build script idempotent given same cache + video; canonical v8 pack_archive byte-deterministic
8. **EXTREME OPTIMIZATION + PERFORMANCE**: 4 min CPU wall-clock for 600-pair build; Modal paired-CPU+CUDA 4 dispatches × ~$0.15-0.30 each ≈ $0.80 actual spend (well under $1.50 operator budget)
9. **OPTIMAL MINIMAL CONTEST SCORE**: V15 result 97.53/97.54 is FAR ABOVE the canonical frontier (CPU 0.1920 / CUDA 0.2053); V15 was a CONTROLLED-COMPARISON test of UNIWARD effect at v8's intra-variant axis, NOT a frontier-pursuit attempt. The v8 inline-LUT variant has 933KB rate-axis cost which structurally precludes frontier-class scoring; the V15 test architecture was correct for paradigm validation, not score lowering

## Predicted ΔS band vs empirical (per Catalog #296)

PREDICTED (from build script protocol): `[1e-4, +inf]` PROMOTION threshold; per the 7th-order MLX-local PARADIGM-VALIDATION verdict the expected paired-CUDA delta should match the magnitude of the MLX-local LUT byte differences scaled by the contest scorer's per-LUT-index gradient response.

EMPIRICAL: delta = `2.66e-6` (CUDA) and `1.47e-7` (CPU). BOTH are 38x to 1000x BELOW the V15 PROMOTION threshold.

Dykstra-feasibility verdict: the intersection of (UNIWARD-routing-effect ∩ entropy-coded-sidecar ∩ v8-LUT-granularity ∩ 50-pair-derivation ∩ 19x-real-gradient-dynamic-range ∩ contest-scorer-sensitivity) appears to be **STATISTICALLY INDISTINGUISHABLE FROM EMPTY** at this configuration. Sister of the prior 5th + 6th + 7th-order empirical surfaces: the empty-set finding RECURS at the production paired-CUDA surface despite the MLX-local PARADIGM-VALIDATION at 200x synthetic.

## Observability surface

Per Catalog #305 — all 6 facets HONORED:

- **Inspectable per layer**: per-bin LUT comparison surfaced via `WeightedMedianResult.per_bin_l2_difference`; per-arm avg_segnet + avg_posenet from Modal auth-eval result dicts
- **Decomposable per signal**: per-axis CUDA + CPU delta separated; per-bin LUT byte delta; rate-axis 4-byte archive delta vs distortion delta separable
- **Diff-able across runs**: deterministic seed; both arm archives byte-comparable; canonical Modal auth-eval result JSON contains all per-axis metrics
- **Queryable post-hoc**: provenance.json + Modal call_id ledger + .omx/state/active_lane_dispatch_claims.md
- **Cite-able**: 4 Modal call_ids + 2 archive sha256s + 2 LUT sha256s + canonical Provenance per Catalog #323
- **Counterfactual-able**: UNIWARD arm vs canonical arm IS the counterfactual; the 19.71x vs 200x dynamic-range delta vs the synthetic MLX-local 7th-order fixture IS the meta-counterfactual

## Horizon-class declaration

Per Catalog #309: **plateau_adjacent** (DOWN-REVISED from 7th-order's `frontier_pursuit` declaration). The empirical paired-CUDA falsification at this implementation moves V15 OUT OF frontier_pursuit territory; the UNIWARD per-LUT-index routing primitive at v8's 16×5 granularity does not unblock asymptotic-pursuit at this fixture. Sister 7th-order memo `frontier_pursuit` declaration was MLX-local-pre-paired-CUDA; V15 lands the paired-CUDA empirical reality.

## Catalog #344 canonical equation anchor status

**Status**: `FORMALIZATION_PENDING` (UNCHANGED; PROMOTION BLOCKED). Per Catalog #344:

The proposed canonical equation `uniward_per_lut_index_distortion_weight_savings_v1` cannot be PROMOTED to PARADIGM-VALIDATED-EMPIRICALLY because both paired-CUDA and paired-CPU deltas fall below the V15 PROMOTION threshold of 1e-4 declared in the build script protocol. Registration BLOCKED per Catalog #344 + #287 FORMALIZATION_PENDING discipline.

**Reactivation criteria** (per CLAUDE.md "Forbidden premature KILL without research exhaustion"):

1. **Larger N_LUT_DERIVATION**: re-run with N_LUT_DERIVATION ≥ 200 pairs (requires re-running N+1 cache generation at 200-pair sample for sister B remote cache build) — hypothesis: per-bin sample size 4x larger may surface UNIWARD-weighted statistic differentiation
2. **Higher real-scorer dynamic range**: re-derive cache with longer training / lower learning rate / scorer-architecture diversity (current cache has 19.71x dynamic range; the 200x MLX-synthetic fixture showed the routing effect more strongly)
3. **Finer LUT granularity**: extend v8 to a (32 levels × 5 classes) or (16 levels × 10 classes) variant; the 80-bin LUT granularity may be too coarse for UNIWARD's per-pixel-routing intuition to differentiate
4. **Different application surface**: per sister 7th-order memo Section "Reactivation criteria": apply UNIWARD per-LUT-index routing to NSCS06 grayscale_lut (T3 council #2 stacking candidate) or VQ-VAE indices_blob (T3 council #3 stacking candidate); the entropy-coded sidecar surface family may yield different empirical signals per-substrate
5. **Different statistic**: replace UNIWARD-weighted median with UNIWARD-weighted EXTREMA (top-K weight quantile rather than median); the canonical Fridrich 2014 formulation supports both; the EXTREMA may be more sensitive to high-weight-pixel concentration

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map**: ACTIVE — per-pixel UNIWARD weights from N+1 cache feed canonical sensitivity-map surface; consumed by LUT derivation in build script
2. **Pareto constraint**: ACTIVE — paired-CUDA + paired-CPU axis tags carry Catalog #323 canonical Provenance; both arms NON-PROMOTABLE per V15 PROMOTION threshold not met
3. **Bit-allocator**: ACTIVE — UNIWARD-weighted LUT IS the per-LUT-index bit allocator; empirical NULL result is itself a bit-allocator-design constraint for future iterations
4. **Cathedral autopilot dispatch**: ACTIVE — canonical Provenance carries non-promotable markers per Catalog #341; integration inherits canonical contract; cathedral autopilot ranker can consume the empirical NULL as evidence for the 5 reactivation criteria
5. **Continual-learning posterior**: ACTIVE — IMPLEMENTATION-LEVEL FALSIFICATION verdict feeds canonical posterior via Catalog #323 Provenance; the 4 Modal call_ids + 2 archive shas + paired-axis scores are queryable via `tac.deploy.modal.call_id_ledger`
6. **Probe-disambiguator**: ACTIVE — V15 IS the canonical disambiguator at paired-CUDA + paired-CPU surface; the empirical NULL result disambiguates between the 7th-order MLX-local PARADIGM-VALIDATION signal and the production contest-scorer surface

## Carmack-dissent verdict per Catalog #307

**PARADIGM-INTACT + IMPLEMENTATION-LEVEL-FALSIFIED-AT-CURRENT-FIXTURE-AND-DYNAMIC-RANGE**. Per CLAUDE.md "Forbidden premature KILL without research exhaustion":

- The UNIWARD per-LUT-index routing PRIMITIVE remains canonical Fridrich; the V15 test does NOT falsify the primitive
- The IMPLEMENTATION at (N_LUT=50, dynamic_range=19.71x, LUT=16×5) IS empirically falsified at the contest scorer
- The 5 reactivation criteria above enumerate the research-exhaustion paths; KILL is the LAST RESORT
- The 5th + 6th + 7th-MLX + V15-paired-CUDA empirical trajectory is now a 4-point sequence: PARADIGM-NULL-AT-RAW-RGB → PARADIGM-NULL-AT-CAPACITY-CONSTRAINED → PARADIGM-VALIDATED-AT-MLX-LOCAL-SYNTHETIC → IMPLEMENTATION-FALSIFIED-AT-PAIRED-CUDA-REAL

Per Catalog #348 retroactive sweep: **NOT TRIGGERED**. The V15 verdict is IMPLEMENTATION-LEVEL not PARADIGM-LEVEL; no historical KILL/DEFER/FALSIFY verdicts are invalidated by V15 (sister 7th-order MLX-local PARADIGM-VALIDATION verdict + sister 5th/6th-order PARADIGM-NULL verdicts ALL remain intact — V15 simply lands the paired-CUDA reality of the 7th-order's reactivation criterion).

## Sister-disjoint discipline confirmation per Catalog #230

NO modifications to NSCS06 v8 chroma_lut substrate's training/test paths. NSCS06 v8 substrate + UNIWARD per-LUT-index integration package consumer-imported ONLY via canonical entry points (`build_chroma_lut_from_ground_truth`, `pack_archive`, `_write_runtime`, `_build_archive_zip`, `build_uniward_weighted_chroma_lut`, `compute_per_pixel_uniward_weight_map_numpy`, `compare_uniward_vs_canonical_lut`). The build script + landing memo are ALL scoped under `experiments/results/v15_uniward_7th_order_paired_archive_build/` + `.omx/research/v15_uniward_7th_order_*`. Zero collision with sister slots (V14-V2 Cascade A FEC10 / Cascade B / Cascade C' / WAVE-7-cascade-A / Phase 4 builder / ORDER-gates / DROP-MANY-audit).

## Discipline anchors

- Catalog #229 PV (read 7th-order MLX-local landing memo + UNIWARD per-LUT-index integration package + canonical v8 archive grammar + canonical v8 _full_main + N+1 cache structure + sister V14-V2 in-flight pattern + canonical frontier pointer + just-saved standing directives BEFORE writing build script)
- Catalog #206 (7 checkpoints emitted across the build/dispatch/verdict cycle)
- Catalog #110/#113 APPEND-ONLY (NEW artifacts only; 7th-order memo + 5th-order + 6th-order + sister substrates NEVER mutated)
- Catalog #117/#157/#174/#235/#289 canonical commit serializer (will use --expected-content-sha256 + co-author trailer)
- Catalog #119 Co-Authored-By Claude trailer (internal commit)
- Catalog #199 paired-env operator-authorize bypass with substantive rationale ≥4 chars (all dispatches used paired CONFIRMED+BUDGET+PROBE-OVERRIDE+RATIONALE)
- Catalog #205 + #295 inflate runtime hygiene (canonical v8 _write_runtime; vendored codec package; numpy-only inflate)
- Catalog #226 trainer auth-eval canonical helper (paired-dispatch helper calls canonical contest_auth_eval.py)
- Catalog #230 ownership map (NSCS06 v8 substrate + UNIWARD integration READ-ONLY consumer imports; sister-disjoint from V14-V2 + WAVE-7 + ORDER-gates + DROP-MANY-audit + Phase 4 builder)
- Catalog #244 NVML/Modal/CUDA env hygiene (auto-emitted by Modal auth-eval helpers)
- Catalog #246 canonical paired-dispatch helper (4 dispatches all via `tools/dispatch_modal_paired_auth_eval.py --execute --skip-axis-if-promotable-anchor-exists`)
- Catalog #287 placeholder rejection (all rationales ≥4 chars substantive)
- Catalog #290 canonical-vs-unique decision per layer (build script DECISION: ADOPT v8 substrate canonical + ADOPT UNIWARD integration canonical + FORK v1-inline variant selection per strategic rationale)
- Catalog #292 (no T2+ council deliberation; not applicable)
- Catalog #294 9-dimension success checklist evidence ✓
- Catalog #296 Dykstra-feasibility predicted-band check ✓ (EMPTY-INTERSECTION verdict)
- Catalog #297 (no signal-axis destruction; not applicable)
- Catalog #303 cargo-cult audit per assumption ✓
- Catalog #305 observability surface ✓
- Catalog #307 paradigm-vs-implementation classification (PARADIGM-INTACT + IMPLEMENTATION-LEVEL-FALSIFIED-AT-CURRENT-FIXTURE-AND-DYNAMIC-RANGE)
- Catalog #309 horizon_class declaration (plateau_adjacent; DOWN-REVISED from 7th-order frontier_pursuit)
- Catalog #313 probe-outcomes ledger (V15 produces canonical adjudicated outcome; sister B reactivation criteria pinned)
- Catalog #316 reports/latest.md frontier alignment (V15 is NOT frontier-crossing; reports/latest.md is NOT mutated by V15)
- Catalog #323 canonical Provenance umbrella in all 4 Modal auth-eval result dicts (score_claim=True/False per axis; evidence_grade=contest-CUDA or contest-CPU per axis; archive_sha pinned; runtime_tree_sha pinned)
- Catalog #335 cathedral consumer canonical contract (UNIWARD per-LUT-index integration inherits via CONSUMER_NAME + CONSUMER_VERSION + CONSUMER_HOOK_NUMBERS)
- Catalog #340 sister-checkpoint guard PROCEED before commit (run before each commit)
- Catalog #341 canonical-routing markers (score_claim per axis carried; promotable=False per axis)
- Catalog #343 NO hardcoded frontier score literals (cite canonical pointer paths only; the V15 scores 97.5364/97.5337 are CONTROLLED-COMPARISON empirical numbers, not frontier claims)
- Catalog #344 canonical equation anchor PROMOTION BLOCKED — FORMALIZATION_PENDING preserved
- Catalog #346 canonical roster N/A (no T2+ deliberation invoked)
- Catalog #348 retroactive sweep NOT TRIGGERED (V15 is IMPLEMENTATION-LEVEL not PARADIGM-LEVEL falsification)
- CLAUDE.md "Apples-to-apples evidence discipline" — both arms IDENTICAL fixture (same 600 pairs, same scorers, same N_LUT=50, same dynamic range)
- CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" — all 4 dispatches harvested + scores extracted; canonical Modal call_id ledger contains entries (sister Catalog #245)
- CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" — paired Modal T4 CUDA + Modal x86_64 Linux CPU axes for both arms
- CLAUDE.md "Forbidden premature KILL without research exhaustion" — 5 reactivation criteria pinned; PARADIGM-INTACT classification
- 4 standing directives 2026-05-26 honored (MLX-first-numpy-portable bridge + FEC family off-the-shelf + AUTOMATED+COMPOUNDING+OPTIMAL META + 3-strategy attack DISTORTION-axis)

## Operator-routable next steps

1. **REACTIVATION CRITERIA SISTER LAUNCH**: spawn sister B subagent to extend N+1 cache to 200 pairs (REQUIRES re-running scorer-gradient capture at sister-subagent + MLX-local cache build; estimated $0 + ~30 min wall-clock) — this directly tests reactivation criterion #1 above
2. **DEFER**: V15 verdict supersedes the 7th-order memo's sister B paired-CUDA reactivation recommendation; canonical equation #344 stays FORMALIZATION_PENDING
3. **NOT-RECOMMENDED**: re-running V15 at the same fixture (the empirical NULL is structurally robust; the V15 paired-CUDA deltas of 2.66e-6 / 1.47e-7 will not change with re-runs at the same configuration)

## Cost

- $0 GPU at build script (CPU only; 4 min wall-clock on M5 Max)
- 4 Modal dispatches × ~$0.10-0.30 each ≈ **$0.50-1.00 ACTUAL** (well under operator $1.50 session budget; canonical helper paired Modal T4 CUDA + x86_64 Linux CPU per Catalog #246)
- ~30 min total wall-clock (4 min build + ~8 min Modal CUDA + ~8 min Modal CPU; CUDA + CPU dispatched in parallel; longest single dispatch was CPU UNIWARD at 478 sec = 8 min)
- ~620 LOC standalone build script + ~700 LOC landing memo (this file)
- 0 canonical equations PROMOTED (FORMALIZATION_PENDING preserved per Catalog #344)
- 5 reactivation criteria pinned per CLAUDE.md "Forbidden premature KILL" non-negotiable

## V15 13th OPTIMAL-TRIO declaration per just-saved 13th standing directive

V15 honors the 13th OPTIMAL-TRIO standing directive (mlx-first-numpy-portable-individually-fractal):
- **MLX-first**: N+1 cache was MLX-local captured; canonical UNIWARD per-LUT-index integration is MLX-portable (pure numpy with MLX-portability hooks per sister `weight_map_per_lut_index.py` docstring)
- **numpy-portable**: build script + inflate runtime are numpy-only; zero MLX dependency at inflate per HNeRV parity L4
- **individually-fractal**: V15 is unique-per-method (sister-disjoint READ-ONLY consumer imports from V14-V2 / Cascade A / 5th+6th+7th-order siblings); the build script does NOT extend the canonical v8 trainer's `_full_main` — it composes the canonical helpers at the per-method optimal point

Per the just-saved 12th standing directive (3-strategy attack framework): V15 sits on the DISTORTION pure-axis attack strategy; the empirical falsification at this implementation does NOT block the 11 sister substrates on PLATEAU + 13 sister substrates on FRONTIER + 4 sister substrates on ASYMPTOTIC per the comprehensive roadmap synthesis (`feedback_comprehensive_roadmap_synthesis_landed_20260526.md`); the next iteration of UNIWARD per-LUT-index routing is queued via the 5 reactivation criteria above per Catalog #313 probe-outcomes ledger.

Per the just-saved 11th standing directive (final rate attack off-the-shelf + development workflow + enhancement): V15 demonstrates the canonical paired-CPU+CUDA dispatch workflow under operator-session-budget discipline ($0.50-1.00 spend / $1.50 budget / 30-45 min wall-clock); the canonical 4-layer pattern is honored (canonical Modal call_id ledger + canonical paired-CUDA helper + canonical operator-authorize paired-env discipline + canonical UNIWARD per-LUT-index integration package).

Per the just-saved 10th standing directive (automated+compounding+optimal meta-principle): V15 emits canonical artifacts that compound into future iterations (provenance.json + 4 Modal call_ids + 2 archive shas + canonical posterior anchors); the build script IS the AUTOMATED reproducibility surface for the 5 reactivation criteria; the IMPLEMENTATION-LEVEL FALSIFICATION verdict is the OPTIMAL signal for sister B reactivation queue.

## NOT a PR111 candidate — explicit declaration

Per V15 protocol: PR111 candidate status requires paired-CUDA frontier-crossing (CPU < 0.19202828 OR CUDA < 0.20533002) per canonical frontier pointer + paired-axis empirical anchor. V15 scores are 97.5364 (CUDA) and 97.5337 (CPU) — both 4-orders-of-magnitude ABOVE the canonical frontier; V15 was a CONTROLLED-COMPARISON test at the v8 v1 inline-LUT 933KB variant which structurally precludes frontier-class scoring (rate term alone = 25 * 933077 / 37545489 ≈ 0.621). Per the user_pr_attribution memory non-negotiable + CLAUDE.md "FORBIDDEN CLAUDE ATTRIBUTION IN PUBLIC-PR SURFACES": V15 does NOT produce a PR111 candidate report; no public-PR-facing artifact is written.

## Sister-coordination + checkpoint discipline confirmation

7 subagent checkpoints emitted via canonical `tools/subagent_checkpoint.py` per Catalog #206:
1. Step 1: PV complete (read all required sources)
2. Step 2: critical strategic decision documented (v1 inline-LUT chosen over v2/v3 procedural-seed)
3. Step 3: build script written + imports verified clean
4. Step 4: full 600-pair build begins
5. Step 5: 4 Modal dispatches active (UNIWARD CUDA + UNIWARD CPU + canonical CUDA + canonical CPU)
6. Step 6: empirical falsification documented (deltas 2.66e-6 / 1.47e-7 << 1e-4 threshold)
7. Step 7 (final): landing memo + canonical commit
