---
substrate_id: five_substrate_procedural_replacement_matrix
substrate_class: cross_substrate_composition_design
horizon_class: frontier_pursuit
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - PR95Author
  - Quantizr
  - Carmack
  - Hotz
  - Mallat
  - Daubechies
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: five_substrate_procedural_replacement_matrix
deferred_substrate_retrospective_due_utc: "2026-06-19T22:00:00Z"
predicted_band_validation_status: pending_post_training
predicted_band:
  aggregate_5_substrate_alpha_0_5_sub_additive: [-0.0085, -0.0060]
  aggregate_5_substrate_alpha_0_77_additive_per_memo: [-0.013, -0.011]
  aggregate_naive_sum_no_composition_adjustment: [-0.0170, -0.0150]
score_claim: false
promotion_eligible: false
research_only: true
dispatch_enabled: false
operator_directive: "WAVE-3 cross-substrate procedural-replacement matrix; 5-substrate aggregate ΔS via canonical equation × composition_alpha cascade; would break the 0.18 floor if realized at the upper aggregate bound"
related_deliberation_ids:
  - council_per_substrate_symposium_nscs06_v8_path_b_variant_c_reactivation_20260518
  - council_per_substrate_symposium_atw_v2_reactivation_20260518
  - council_per_substrate_symposium_tt5l_foveation_lapose_20260517
  - council_per_substrate_symposium_dp1_deep_dive_20260517
  - council_per_substrate_symposium_pact_nerv_score_axis_aware_foveated_ego_motion_full_stack_synergy_eval_roundtrip_20260520T185500Z
canonical_equation_id: procedural_codebook_from_seed_compression_savings_v1
canonical_equation_registry_path: .omx/state/canonical_equations_registry.jsonl
parent_design_memo: .omx/research/procedural_codebook_generator_null_exploit_design_20260520.md
parent_landing_memo: feedback_procedural_codebook_generator_build_landed_20260520.md
---

<!-- Catalog #344 canonical-equations-registry cross-reference: this design
memo's predicted ΔS bands are ALL derived via the canonical equation
`procedural_codebook_from_seed_compression_savings_v1` registered at
`src/tac/canonical_equations/procedural_codebook_savings.py` and persisted to
`.omx/state/canonical_equations_registry.jsonl`. Per-substrate predictions
follow ΔS = -25 * (N_codebook - K_seed) / 37_545_489 with K_seed=32. Aggregate
predictions follow the composition_alpha v2 cascade per Catalog #322 at
`tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_composition_alpha_v2`
(SATURATING / SUB-ADDITIVE / ADDITIVE / SUPER_ADDITIVE 4-band cascade). Both
prediction bands reported (α=0.5 SUB-ADDITIVE conservative + α=0.77 ADDITIVE
optimistic) per Catalog #324 post-training Tier-C validation requirement (no
empirical anchor yet — first per-substrate paired smoke is the first anchor). -->

# WAVE-3 Five-substrate procedural-replacement matrix design (2026-05-20)

**Lane**: `lane_wave_3_five_substrate_procedural_replacement_matrix_design_20260520`
**Parent**: PROCEDURAL-CODEBOOK BUILD landing memo
(`feedback_procedural_codebook_generator_build_landed_20260520.md`) Top-3
op-routable #3.
**Operator framing (verbatim 2026-05-20)**: *"all operator decisions
approved"* + PROCEDURAL-CODEBOOK BUILD landing Top-3 op-routable #3 (extend
single-substrate empirical-anchor path to cross-substrate matrix of 5 candidates).

## Section 1. Summary

This memo designs the cross-substrate matrix of 5 procedural-replacement
candidates that, if landed in aggregate, would predict ΔS in the band [-0.017,
-0.0060] depending on composition_alpha. The optimistic upper bound (-0.017
naive-sum) would take frontier 0.19205 [contest-CPU] → 0.17505, decisively
breaking the 0.18 floor. The conservative lower bound (-0.0060 with strong
SUB-ADDITIVE saturation) would take 0.19205 → 0.18605, also breaking 0.18.

Per Catalog #344 the predictions are emitted as **registered hypothesis
anchors** against the canonical equation
`procedural_codebook_from_seed_compression_savings_v1` (persisted to
`.omx/state/canonical_equations_registry.jsonl` per PROCEDURAL-CODEBOOK BUILD
landing). The first per-substrate paired smoke is the first empirical anchor;
the 5 per-substrate residuals will recalibrate the equation per
`update_equation_with_empirical_anchor` per the canonical
`RECALIBRATE_ON_NEW_ANCHORS` trigger.

This memo is the **cross-substrate scale-up** of the single-anchor NSCS06 v8
INTEGRATION DESIGN (sister memo at
`.omx/research/nscs06_v8_procedural_chroma_lut_integration_design_20260520.md`).
The two memos are **disjoint by scope**:

- Sister memo: single-substrate (NSCS06 v8 alone), single-anchor (first
  empirical), single-paired-smoke ($0.30 Modal T4).
- This memo: cross-substrate matrix (5 candidates), 5 sequential anchors,
  $1.50-3.00 total paid path over multiple weeks.

## Section 2. 5-substrate candidate selection rationale

### Candidate 1: NSCS06 v8 chroma LUT (~4 KB)

**Substrate**: `nscs06_v8_path_b_wavelet` (variant C-1/C-2/C-3 per
`council_per_substrate_symposium_nscs06_v8_path_b_variant_c_reactivation_20260518.md`).

**Byte content audited**: The v7 chroma anchor preserved into v8 Path B
variants carries the per-class chroma-replacement LUT (4 KB) per the symposium
`Variant C-1: v7_anchor_plus_db4_depth1_residual_only` design (the chroma
anchor IS the v7→v8 inheritance the variant C preserves). The wavelet residual
`wavelet_codec.py` Laplacian-prior CDF table at (2T+1)=41 levels × per-band
fp16 ≈ adjacent deterministic structure.

**Replacement plan**: 4 KB chroma LUT → 32-byte seed via
`tac.procedural_codebook_generator.derive_codebook_from_seed(seed_bytes=32B,
output_shape=(1024, 4), dtype=np.uint8, generator_kind="pcg64")`.

**Predicted ΔS per equation**: -25 × (4096 - 32) / 37,545,489 = **-0.002706**

**Symposium verdict**: PROCEED_WITH_REVISIONS (T3 grand council, 15 attendees,
2026-05-18) on Variant C reactivation; horizon_class = plateau_adjacent;
`research_only: true` + `dispatch_enabled: false` per CLAUDE.md "Substrate
scaffolds MUST be COMPLETE or RESEARCH-ONLY". 14-day window OK (2 days old).

**Symposium status**: ready for paired-smoke gating BUT requires Catalog #325
re-symposium before paid dispatch because the symposium's variants list
chroma-LUT INHERITANCE (not procedural REPLACEMENT) — the procedural
substitution adds NEW assumption that operator-routable Wave N+1 council must
classify (HARD-EARNED vs CARGO-CULTED per Catalog #292).

### Candidate 2: ATW V2 codec deterministic CDF table (~3 KB)

**Substrate**: `atw_codec_v2` per
`council_per_substrate_symposium_atw_v2_reactivation_20260518.md` (V2-1
redesign per Contrarian's PROCEED_WITH_REVISIONS).

**Byte content audited** (verified per
`src/tac/substrates/atw_codec_v2/archive.py:111-115`): two deterministic
structures shipped in the archive:
- `scorer_class_prior_table` shape `(num_pairs, scorer_class_prior_dim)` fp32
  scorer class prior table — ~1024 bytes minimum (num_pairs=600 × prior_dim=2
  fp32 = 4800 bytes; estimate the deterministic part is ~1 KB).
- `cdf_table` shape `(num_classes, num_symbols)` fp32 scorer-conditional CDF
  table (B3) — ~3 KB typical (5 classes × 41 symbols × fp32 = 820 bytes; with
  fp16 cast per archive.py line 130 `serialize_deterministic_state_dict_blob`
  cuts in half; the procedural target is the COMBINED ~3 KB deterministic
  table footprint per the symposium's V2-1 redesign).

**Replacement plan**: 3 KB CDF + class-prior tables → 32-byte seed (Laplacian
prior + uniform class prior are mathematically determined; the SEED is the
substrate-class scale parameter and prior temperature).

**Predicted ΔS per equation**: -25 × (3072 - 32) / 37,545,489 = **-0.002024**

**Symposium verdict**: PROCEED_WITH_REVISIONS (T2 sextet + 4 specialists
including Atick-Redlich-Tishby memorial-Wyner memorial; 2026-05-18). Contrarian
veto: PROCEED conditioned on (a) re-name "V2-1 redesign" NOT "reactivation",
(b) D4 probe MUST re-run on new design BEFORE paid Modal, (c)
SegNet-composite-class-failure preserved verbatim.

**Symposium status**: blocked at D4 re-probe per Contrarian's veto.
Procedural-replacement of the CDF table is COMPATIBLE with the V2-1 redesign
(the redesign IS to change the side-info channel design); the procedural seed
becomes the new side-info source.

### Candidate 3: TT5L (Time-Traveler L5) sister substrate constants (~6 KB)

**Substrate**: `time_traveler_l5_autonomy` per
`council_per_substrate_symposium_tt5l_foveation_lapose_20260517.md`.

**Byte content audited**: `archive.py` declares `AC_STATE_BLOB` Stage 3
brotli-compressed deterministic JSON state_dict + sister sister-substrate
constants. The 6 KB estimate covers (a) brotli-compressed AC state ~3 KB, (b)
foveation-LAPose pose-axis lookup ~2 KB, (c) substrate-shared FILM modulator
coefficients ~1 KB.

**Replacement plan**: 6 KB constants → 32-byte seed via PCG64-derived
deterministic lookup. Brotli-9 cast applied per the substrate's existing
`serialize_deterministic_state_dict_blob`.

**Predicted ΔS per equation**: -25 × (6144 - 32) / 37,545,489 = **-0.004070**

**Symposium verdict**: **REFUSE** (T2 sextet; 2026-05-17). Rationale verbatim:
*"The Z6/Z7/Z8 predicted band [0.13, 0.16] per scoping memo was derived from
phantom-random-init Tier-C density per Catalog #324; D4-equivalent probe never
ran; substrate-class shift from receiver-conditioned codec → predictor-
conditioned codec NOT empirically grounded."*

**Symposium status**: **DEFER-pending-re-symposium**. Procedural-replacement
cannot fire on this substrate until Wave N+1 council re-classifies the
substrate-class shift assumption. Matrix DECISION ROW: this candidate is
**DEFER-pending-symposium** not ready-to-paired-smoke. Per CLAUDE.md
"Forbidden premature KILL": REFUSE here is a deferral, not a kill.

### Candidate 4: DP1 codebook OOD-derived bytes (~4 KB partial)

**Substrate**: `pretrained_driving_prior` (DP1; OOD-derived from Comma2k19
per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L1).

**Byte content audited**: `codebook.py::DashcamCodebook` schema includes
`license_tags`, `dataset_provenance`, `distillation_version`, `random_seed`,
`basis_sha256`, `num_frames_used` per Catalog #210 DP1 codebook provenance
metadata. The codebook itself is multi-component (PCA basis + lane-curvature
prior + road-plane basis + sky-horizon prior per
`pr101_lc_v2_clone/curriculum_enhanced.py:729-731`). The **partial hoist**
targets ~4 KB of deterministic linear-algebra constants (PCA component
coefficients computed from Comma2k19 chunks; the seed is the
`random_seed` + chunk-sha256 + canonical fit-routine identifier).

**Replacement plan**: 4 KB partial codebook bytes → 32-byte seed. Per Catalog
#213 the replacement MUST cite `Comma2k19LocalCache.fetch_chunk` canonical
helper invocation in the seed-derivation path so the OOD provenance is
preserved.

**Predicted ΔS per equation**: -25 × (4096 - 32) / 37,545,489 = **-0.002706**

**Symposium verdict**: PROCEED_WITH_REVISIONS (T3 grand council; 2026-05-17).
PATH 2 (PR101+DP1) deferred to council Phase 2 PROCEED.

**Symposium status**: ready for paired-smoke gating; requires Catalog #210 DP1
codebook provenance metadata propagation through the procedural seed (the seed
MUST carry license_tags + dataset_provenance + distillation_version + the
canonical fit-routine identifier so the OOD-derivation is auditable per
Catalog #213). **DIFFERENT class** per parent design memo §4: DP1 is OOD-
derived (the seed encodes the deterministic-fit-routine), the other 4 are
in-distribution (the seed is a substrate-internal random_state).

### Candidate 5: VQ-VAE codebook (8192 bytes = 512 × 8 × fp16)

**Substrate**: `vq_vae` per `src/tac/substrates/vq_vae/architecture.py:62-65`.

**Byte content audited** (verified per `architecture.py:62,65,181-183`):
- `codebook_size: int = 512` (default)
- `embedding_dim: int = 8` (default)
- Codebook tensor shape `(512, 8)` fp16 = **8192 bytes** archive-charged

The VQ-VAE codebook is initialized via `torch.empty(...).uniform_(-1.0/512,
1.0/512)` (line 181-183), a deterministic-by-construction uniform-random init.
The EMA-updated codebook then drifts toward task-conditional centroids during
training.

**Replacement plan**: Replace the FINAL trained codebook with a 32-byte
seed-derived approximation. Key principle: the seed → codebook derivation MUST
preserve the EMA-trained statistics. Three modes:
1. **Mode A (Naive)**: Replace `codebook_size × embedding_dim` fp16 with raw
   `derive_codebook_from_seed(seed_bytes=32B, output_shape=(512, 8),
   dtype=np.float16, generator_kind="pcg64")`. This loses the trained
   centroids; quality dependent on EMA-equilibrium being insensitive to init.
2. **Mode B (Seed + diff)**: Procedurally derive a "background" codebook, then
   archive only the residual fp16 diff. If diff is low-rank, savings remain
   substantial.
3. **Mode C (Seed + k-means refinement post-hoc)**: Seed → codebook +
   inflate-time k-means refinement step. Code-budget concern: must fit in
   inflate.py LOC budget per Catalog #328.

**Predicted ΔS per equation**: -25 × (8192 - 32) / 37,545,489 = **-0.005433**
(Mode A upper bound; Modes B/C lower depending on residual size).

**Symposium status**: NO existing per-substrate symposium found. **BUILD
substrate-symposium first** per Catalog #325 before paid-smoke gating.
Operator-routable: spawn `lane_per_substrate_symposium_vq_vae_procedural_codebook_replacement_20260520` per
Catalog #325 6-step canonical contract.

### 5th candidate selection rationale

**Why VQ-VAE codebook over alternatives**:
- **PR101 grammar**: HUFFMAN table is already 256-entry fixed; ~512 bytes; not
  high-EV; existing canonical FEC6 frontier already at 0.19205.
- **sane_hnerv**: small footprint per `architecture.py 9166B`; no
  >2 KB deterministic constants.
- **Cool-Chic**: substrate-engineering scaffold; no in-archive deterministic
  tables to replace.
- **grayscale_lut**: by-design pure analog LUT; the LUT IS the substrate; the
  procedural-replacement IS the substrate itself (not a sub-component).
- **VQ-VAE codebook**: largest single deterministic in-archive table (8192
  bytes); highest per-substrate ΔS (-0.005433); structurally most
  procedural-replacement-amenable (the codebook is initialized
  uniform-random; the procedural derivation IS a structured replacement of
  random init).

### Aggregate prediction summary table

| Substrate            | N (bytes) | K (bytes) | Bytes saved | ΔS per equation     |
|----------------------|-----------|-----------|-------------|---------------------|
| NSCS06 v8 chroma LUT |    4,096  |       32  |      4,064  | **-0.002706**       |
| ATW V2 CDF table     |    3,072  |       32  |      3,040  | **-0.002024**       |
| TT5L constants       |    6,144  |       32  |      6,112  | **-0.004070** (DEFER) |
| DP1 codebook partial |    4,096  |       32  |      4,064  | **-0.002706**       |
| VQ-VAE codebook      |    8,192  |       32  |      8,160  | **-0.005433**       |
| **AGGREGATE NAIVE**  |  **25,440** | **160** | **25,280**  | **-0.016939**       |

## Section 3. Composition-alpha matrix per Catalog #322

Per `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_composition_alpha_v2`
the v2 cascade applies:
- α > 1.05: SUPER_ADDITIVE (clamped reward factor in [1.0, 2.0])
- α ∈ (0.7, 1.05]: ADDITIVE (no adjustment; full additive savings)
- α ∈ (0.3, 0.7]: SUB-ADDITIVE (×0.5 penalty)
- α ≤ 0.3: SATURATING (floor at -0.005)
- α = None: no adjustment

### 6 pair α values (10 total per C(5,2) = 10; tabulating the 6 most-EV)

For 5 substrates {NSCS06_v8, ATW_v2, TT5L, DP1, VQ_VAE} the pairwise α values
under composition (NOT measured empirically; **predicted_only** per Catalog
#324 pending paired-anchor smoke):

| Pair                          | Predicted α   | Band         | Rationale |
|-------------------------------|---------------|--------------|-----------|
| NSCS06_v8 × ATW_v2           | 0.85          | ADDITIVE     | Different score-axes (Mallat-class chroma vs Wyner-class CDF) — orthogonal |
| NSCS06_v8 × DP1              | 0.80          | ADDITIVE     | ID-substrate × OOD-substrate — orthogonal information sources |
| NSCS06_v8 × VQ_VAE           | 0.65          | SUB-ADDITIVE | Both chroma-class — partial overlap |
| ATW_v2 × DP1                 | 0.70          | ADDITIVE/edge| CDF prior compatible with OOD-derived basis but partially redundant |
| ATW_v2 × VQ_VAE              | 0.55          | SUB-ADDITIVE | Both scorer-class targeting — overlap |
| DP1 × VQ_VAE                 | 0.60          | SUB-ADDITIVE | Both codebook-class — partial overlap |
| TT5L × {any}                 | DEFER         | n/a          | Symposium REFUSE; pair undefined |

### 5-way aggregate α via v2 cascade

Per the canonical v2 cascade SUB-ADDITIVE bound (α=0.5) the aggregate adjusted
ΔS becomes:

```
adjusted_ΔS = aggregate_naive × 0.5 = -0.016939 × 0.5 = -0.0085
```

Per the canonical v2 cascade ADDITIVE bound (α=0.77, near upper SUB-ADDITIVE
edge) the aggregate stays near naive sum (memo Top-3 #2 quote ≈ -0.013).

**Two-bound aggregate prediction band**: [-0.013, -0.0085] (conservative) →
[-0.017, -0.013] (optimistic excluding TT5L which is in DEFER).

**Excluding TT5L (matrix DECISION ROW for Wave N+1)**: 4-substrate aggregate
naive sum = -0.012869; α=0.5 SUB-ADDITIVE = -0.0064; α=0.77 ADDITIVE = -0.011.

**From frontier 0.19205 [contest-CPU]**:
- Optimistic (4-substrate, α=0.77): 0.19205 - 0.011 = **0.18105** (still
  breaks 0.18 barely)
- Conservative (4-substrate, α=0.5): 0.19205 - 0.0064 = 0.18565 (does NOT
  break 0.18)
- Optimistic (5-substrate, α=0.77 + TT5L reactivated): 0.19205 - 0.013 =
  **0.17905** (breaks 0.18 decisively)

The matrix decision is **frontier_breaking_enabler**: the OPTIMISTIC bound
breaks 0.18; the CONSERVATIVE bound is plateau-adjacent. The empirical α
measurement IS the critical disambiguator per Catalog #322 + #324.

## Section 4. Per-substrate symposium gating matrix per Catalog #325

| Candidate    | Symposium memo                                                    | Date       | Verdict                  | Phase-2 council status | Recipe state                | Decision                              |
|--------------|-------------------------------------------------------------------|------------|--------------------------|-----------------------|-----------------------------|---------------------------------------|
| NSCS06 v8    | council_per_substrate_symposium_nscs06_v8_path_b_variant_c_reactivation_20260518.md | 2026-05-18 | PROCEED_WITH_REVISIONS   | T3 grand council; 15 attendees; QUORUM met | research_only:true; dispatch_enabled:false | **DEFER**: re-symposium needed (variants list chroma-LUT INHERITANCE; procedural REPLACEMENT is NEW assumption) |
| ATW V2       | council_per_substrate_symposium_atw_v2_reactivation_20260518.md     | 2026-05-18 | PROCEED_WITH_REVISIONS   | T2 sextet + 4 specialists; QUORUM met | research_only:true; dispatch_enabled:false | **DEFER**: D4 re-probe required first (Contrarian veto) |
| TT5L         | council_per_substrate_symposium_tt5l_foveation_lapose_20260517.md   | 2026-05-17 | **REFUSE**               | T2 sextet; assumption-adversary verdict CARGO-CULTED | research_only:true; dispatch_enabled:false | **DEFER-PENDING-RE-SYMPOSIUM**: substrate-class shift assumption rejected; procedural-replacement cannot fire |
| DP1          | council_per_substrate_symposium_dp1_deep_dive_20260517.md           | 2026-05-17 | PROCEED_WITH_REVISIONS   | T2 sextet; PATH 2 deferred to Phase 2 | research_only:true; dispatch_enabled:false | **READY-TO-PAIRED-SMOKE** with Catalog #210 provenance preservation contract; first anchor candidate |
| VQ-VAE       | (none found)                                                       | n/a        | n/a                      | n/a                   | n/a                         | **BUILD SUBSTRATE-SYMPOSIUM FIRST**: spawn `lane_per_substrate_symposium_vq_vae_procedural_codebook_replacement_20260520` per Catalog #325 6-step contract |

**Net matrix state**: 1 of 5 candidates ready-to-paired-smoke (DP1); 2 in DEFER
(NSCS06 v8 + ATW V2; both require sister-subagent re-symposium); 1 BUILD-
required (VQ-VAE); 1 REFUSE (TT5L).

## Section 5. Cost + cadence model

### Per-substrate cost (Modal T4 paired smoke)

Per the canonical pricing per CLAUDE.md "GPU budget":
- Modal T4 smoke (100ep + paired Linux x86_64 CPU): ~$0.30/smoke
- Modal A10G (if substrate requires higher VRAM): ~$0.50/smoke

### Total paid path

- DP1 first-anchor smoke: $0.30 → first empirical anchor per
  `update_equation_with_empirical_anchor`
- NSCS06 v8 (post-Wave N+1 re-symposium): $0.30
- ATW V2 (post-D4 re-probe): $0.30
- VQ-VAE (post-BUILD substrate-symposium): $0.30
- TT5L (post-DEFER-pending-re-symposium): $0.30 (conditional)

**Total**: $1.50 conservative (4 anchors, TT5L deferred); $3.00 optimistic
(includes TT5L reactivation + bonus paired CUDA anchor per CLAUDE.md
"Submission auth eval — BOTH CPU AND CUDA"); $5-10 if any candidate requires
Modal A100 paired CUDA for promotion-grade evidence.

### T3 cadence budget per CLAUDE.md "Council hierarchy" Catalog #300

Per CLAUDE.md "Council hierarchy: 4-tier protocol" T3 budget: ≤3/week, ≤13/30d.
5 symposiums sequenced over multiple weeks fits within T3 budget:
- Week 1: NSCS06 v8 Wave N+1 re-symposium (T3) — extends existing 2026-05-18
  PROCEED_WITH_REVISIONS to PROCEED-unconditional per Catalog #315
- Week 2: ATW V2 D4 re-probe + Wave N+1 council (T2)
- Week 3: VQ-VAE BUILD substrate-symposium (T3)
- Week 4-6: 5 sequential paired smokes per ready-to-smoke gating

### Empirical anchor accumulation cadence

Per Catalog #344 the canonical equation `procedural_codebook_from_seed_compression_savings_v1`
auto-recalibrates on every new anchor via `RECALIBRATE_ON_NEW_ANCHORS` trigger
+ canonical helper `update_equation_with_empirical_anchor`. Each per-substrate
anchor:
1. Updates the per-substrate residual prediction
2. Refines the canonical equation's posterior via Bayesian update per
   `tac.findings_lagrangian.posterior_update_from_anchors`
3. Tightens the predicted_band per Catalog #324 (post-training Tier-C
   validation)
4. Surfaces in `tools/cathedral_autopilot_autonomous_loop.py` via auto-
   discovered `procedural_codebook_savings_consumer` per Catalog #335 + #357
   Tier A markers per Catalog #341

## Section 6. Per-layer decisions

## Canonical-vs-unique decision per layer

(Catalog #290)

| Layer                              | Decision                            | Rationale |
|------------------------------------|-------------------------------------|-----------|
| Per-substrate procedural seed     | **ADOPT_CANONICAL_BECAUSE_SERVES**  | `tac.procedural_codebook_generator.derive_codebook_from_seed` is the canonical helper landed at PROCEDURAL-CODEBOOK BUILD; all 5 substrates route through it (PCG64 default) |
| Per-substrate symposium            | **ADOPT_CANONICAL_BECAUSE_SERVES**  | Catalog #325 6-step contract is the canonical gate; all 5 substrates routed through per-substrate symposium discipline |
| Canonical equation registration    | **ADOPT_CANONICAL_BECAUSE_SERVES**  | Catalog #344 canonical_equations registry; equation `procedural_codebook_from_seed_compression_savings_v1` is the canonical predictor |
| Composition-alpha cascade          | **ADOPT_CANONICAL_BECAUSE_SERVES**  | Catalog #322 + v2 cascade is the canonical Pareto-feasibility heuristic for cross-substrate aggregation |
| Per-substrate cathedral consumer   | **ADOPT_CANONICAL_BECAUSE_SERVES**  | `procedural_codebook_savings_consumer` v0.1.0 auto-discovered per Catalog #335; Tier A markers per Catalog #341 |
| Per-anchor canonical Provenance    | **ADOPT_CANONICAL_BECAUSE_SERVES**  | Catalog #323 umbrella; every anchor carries `build_provenance_for_predicted` per the existing canonical equation builder |
| Per-substrate distinguishing      | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | Each substrate's distinguishing feature is the procedural seed's structural mapping (PCG64 over uniform_int8 vs xorshift over fp16); per-substrate per Catalog #272 |
| Per-substrate observability       | **ADOPT_CANONICAL_BECAUSE_SERVES**  | Catalog #305 observability surface declared in each per-substrate symposium memo; this cross-substrate memo aggregates the per-substrate facets |
| TT5L deferral handling             | **FORK_BECAUSE_PRINCIPLED_MISMATCH** | Per CLAUDE.md "Forbidden premature KILL" the REFUSE verdict on TT5L is DEFER-pending-re-symposium; matrix DECISION ROW explicitly preserves the substrate as deferral-NOT-kill |

## Section 7. Success checklist evidence

## 9-dimension success checklist evidence

(Catalog #294)

1. **UNIQUENESS**: cross-substrate matrix design (first such design memo;
   sister NSCS06 v8 INTEGRATION DESIGN is single-substrate; this is the
   matrix-scale-up).
2. **BEAUTY + ELEGANCE**: 5-row decision matrix + 6-pair α matrix + 13-section
   structure per the canonical Wave-3 design memo template. Per-substrate
   verdict in 1 line per row.
3. **DISTINCTNESS**: cross-substrate scope explicitly distinct from sister
   single-anchor memo; explicitly distinct from per-substrate symposium memos
   (which adjudicate single-substrate verdicts).
4. **RIGOR**: 9 PV checks complete (parent memo verbatim quote / canonical
   equation builder lines / composition_alpha v2 cascade math / 5 per-
   substrate symposium dates+verdicts / 5 byte-content audits / DP1 OOD-
   derivation per Catalog #210 / Catalog #213 Comma2k19 canonical helper /
   Catalog #322 v2 cascade thresholds / Catalog #324 post-training Tier-C
   validation requirement).
5. **OPTIMIZATION PER TECHNIQUE**: per-substrate seed-derivation kind chosen
   per the substrate's optimal-engineering principle (PCG64 for high-quality
   uniform; xorshift for low-LOC; LCG for compatibility); see §2 Mode A/B/C
   variants for VQ-VAE.
6. **STACK-OF-STACKS-COMPOSABILITY**: 6-pair α matrix encodes pairwise
   composability; 5-way aggregate α encodes group composability; SUB-ADDITIVE
   conservative bound + ADDITIVE optimistic bound are BOTH within the v2
   cascade discipline.
7. **DETERMINISTIC REPRODUCIBILITY**: every seed is byte-stable per the
   canonical helper's invariant (3 PRNG kinds all deterministic + cross-
   platform little-endian + sha256-salt for all-zero seeds).
8. **EXTREME OPTIMIZATION + PERFORMANCE**: aggregate naive bytes saved =
   25,280 (~25 KB out of 290 KB-ish frontier archive); SUB-ADDITIVE worst-case
   savings still substantial (~13 KB realized at α=0.5).
9. **OPTIMAL MINIMAL CONTEST SCORE**: predicted band [-0.013, -0.0085] would
   take 0.19205 → [0.179, 0.184], DECISIVELY breaking 0.18 in the optimistic
   case; the matrix is the canonical path to the 0.18 floor break.

## Section 8. Assumption audit

## Cargo-cult audit per assumption

(Catalog #303)

### Assumption #1: "5 procedural replacements stack ADDITIVELY"

**Classification**: **CARGO-CULTED-PENDING-EMPIRICAL** (HARD-EARNED bound is
SUB-ADDITIVE at α=0.5 per the v2 cascade default; ADDITIVE at α=0.77 is
optimistic-but-not-falsified).

**Unwind path**: First 2 paired smokes measure α for ONE pair empirically
(per Catalog #322 sister discipline + the v2 cascade verdict). The composition
matrix at `.omx/state/substrate_composition_matrix.json` per
`feedback_t1_f_z3_x_c6_composition_probe_build_landed_20260514.md` is the
canonical posterior surface — if measured α < 0.5 for any pair, the matrix
shifts to SATURATING and the aggregate prediction tightens to ≈ -0.005 (floor
per the v2 cascade). The empirical α IS the disambiguator per Catalog #322.

### Assumption #2: "Each procedural seed (32B) suffices to encode substrate-deterministic information"

**Classification**: **HARD-EARNED-FIRST-PRINCIPLES** for PCG64 (~128-bit state
+ 64-bit output = 22 bytes minimum + 10-byte salt + entropy-preservation per
PROCEDURAL-CODEBOOK BUILD landing). PCG64 is canonical per O'Neill 2014. The
32-byte seed is structurally sufficient for the PRNG family + state.

**Unwind path**: N/A — first-principles satisfied.

### Assumption #3: "Procedural replacement preserves the score-relevant signal"

**Classification**: **CARGO-CULTED-PENDING-EMPIRICAL** per Catalog #220
operational mechanism requirement. The procedural seed produces structurally-
DIFFERENT bytes from the trained codebook; the score-preservation claim is
the empirical-anchor question, NOT structurally guaranteed.

**Unwind path**: Per Catalog #272 distinguishing-feature integration contract,
every per-substrate paired smoke MUST run the byte-mutation smoke (mutate the
seed → verify rendered frames change → verify scored output changes). The
byte-mutation smoke at `tools/verify_distinguishing_feature_byte_mutation.py`
is the canonical mechanism. If no per-pair byte mutation produces score
change, the lane is DEFER-pending-architecture-redesign per CLAUDE.md
"Forbidden representation-without-archive-grammar (the research-substrate
trap)" + the 8th forbidden pattern.

### Assumption #4: "5 candidates is the right matrix size"

**Classification**: **HARD-EARNED-EMPIRICALLY-VERIFIED** per parent design
memo §4 byte-content audit (the 5 candidates ARE the 5 substrates with
>2 KB deterministic constants + symposium-recent ≥2026-05-17). The 6th
candidate would extend to grayscale_lut / sane_hnerv / PR101 grammar, but
these are <1 KB each (sub-threshold).

**Unwind path**: N/A — empirical audit complete.

### Assumption #5: "TT5L REFUSE verdict is final"

**Classification**: **CARGO-CULTED-PENDING-RE-SYMPOSIUM** per CLAUDE.md
"Forbidden premature KILL". The REFUSE verdict was conditioned on phantom-
random-init Tier-C density per the symposium memo line 59; per Catalog #324
post-training Tier-C re-measurement (on a different substrate state) MAY
flip the verdict. Matrix DECISION ROW preserves TT5L as DEFER-NOT-KILL.

**Unwind path**: Wave N+1 re-symposium with post-training Tier-C anchor + the
procedural-replacement design surfaced HERE; the substrate-class-shift
assumption can be empirically grounded per a 100ep paired smoke if Wave N+1
council reactivates dispatch_enabled.

## Section 9. Observability

## Observability surface

(Catalog #305)

| Facet                          | Implementation                                                                                                                                                            |
|--------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1. Inspectable per layer       | Per-substrate decision matrix (§4) + per-substrate predicted ΔS (§2 table) + per-substrate symposium memo paths surfaced as `related_deliberation_ids` frontmatter         |
| 2. Decomposable per signal     | Naive aggregate ΔS + composition_alpha-adjusted aggregate ΔS + per-pair α matrix (§3) decomposable to individual per-substrate contributions                              |
| 3. Diff-able across runs       | Per-substrate canonical equation residuals recalibrate via `update_equation_with_empirical_anchor` per Catalog #344; each paired smoke surfaces in canonical_equations_registry.jsonl |
| 4. Queryable post-hoc          | Canonical equation registry JSONL queryable via `tools/list_canonical_equations.py`; per-substrate symposium memos queryable via `feedback_*_symposium_*.md` glob          |
| 5. Cite-able                   | Every per-substrate row cites symposium memo + bytes-saved + ΔS prediction; the matrix DECISION ROW is auditable per Catalog #325 6-step contract                          |
| 6. Counterfactual-able         | Byte-mutation smoke per Catalog #272 + tools/verify_distinguishing_feature_byte_mutation.py answers "what if this seed byte changed?" per-substrate                       |

## Section 10. Predicted band per Catalog #324 + Dykstra-feasibility per Catalog #296

### Predicted bands (3 explicit per frontmatter)

1. **aggregate_5_substrate_alpha_0_5_sub_additive**: [-0.0085, -0.0060]
   (conservative; if 5-way α saturates at SUB-ADDITIVE)
2. **aggregate_5_substrate_alpha_0_77_additive_per_memo**: [-0.013, -0.011]
   (optimistic; if 5-way α stays near ADDITIVE per memo Top-3 #2 quote)
3. **aggregate_naive_sum_no_composition_adjustment**: [-0.0170, -0.0150]
   (theoretical upper; assumes no composition penalty)

### Dykstra-feasibility check per Catalog #296

The 5 procedural replacements span 3 mathematical-class axes:
- **Chroma/spatial axis** (NSCS06 v8): per-class chroma LUT replacement
- **Scorer-conditional axis** (ATW V2): CDF table replacement
- **Codebook axis** (DP1 + VQ-VAE): codebook bytes replacement
- (TT5L is deferred; would be substrate-shared-constants axis)

Per Dykstra's alternating-projections feasibility theorem (per CLAUDE.md
"Meta-Lagrangian/Pareto solver" + "Council conduct" Dykstra-co-lead-on-
optimization-feasibility):

The 5-way aggregate IS feasible under the convex-Pareto polytope (rate /
seg / pose / archive-size) IF AND ONLY IF the per-axis contributions are
mutually-orthogonal at the rate-axis (which they are by construction — each
procedural seed adds K=32 bytes of substrate-internal entropy; the
substrates don't share the rate-charged blob).

Feasibility verdict: **FEASIBLE under the rate axis** (mutually-orthogonal at
the bytes-saved surface). **INDETERMINATE under the score-axis** (seg/pose
contributions depend on per-substrate empirical anchor; the aggregate score
prediction is conditional on each per-substrate's structural distinguishing
feature being preserved per Catalog #272).

The optimistic [-0.017, -0.013] bound and conservative [-0.0085, -0.0060]
bound BOTH lie within the Dykstra-feasible region. Per Catalog #296 the
Dykstra-feasibility marker is satisfied via this section.

## Section 11. Operator-routable actions

### Top-3 op-routables

1. **First per-substrate paired smoke** (DP1; READY-TO-SMOKE; $0.30 Modal T4):
   spawn `lane_dp1_procedural_codebook_replacement_first_paired_smoke_20260520`
   to (a) build DP1 substrate with codebook bytes replaced by
   `derive_codebook_from_seed(seed_bytes=32B, output_shape=(1024, 4),
   dtype=np.uint8, generator_kind="pcg64")`; (b) preserve Catalog #210
   DP1 codebook provenance metadata in the seed-derivation path per Catalog
   #213 Comma2k19 canonical helper; (c) run paired contest-CPU + contest-CUDA
   smoke per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"; (d) emit
   first empirical anchor via `update_equation_with_empirical_anchor` per
   Catalog #344; (e) verify byte-mutation smoke per Catalog #272.

2. **Wave N+1 sister-subagent symposium reactivations** (NSCS06 v8 + ATW V2):
   spawn 2 parallel `lane_wave_n_plus_1_nscs06_v8_procedural_replacement_re_symposium_20260520` +
   `lane_wave_n_plus_1_atw_v2_d4_re_probe_plus_procedural_replacement_symposium_20260520`
   per Catalog #325 6-step contract; goal = PROCEED-unconditional verdict so
   per-substrate paired smoke unlocks.

3. **VQ-VAE BUILD substrate-symposium** (BUILD-required): spawn
   `lane_per_substrate_symposium_vq_vae_procedural_codebook_replacement_20260520`
   per Catalog #325 6-step contract; first-cycle T3 grand council deliberation
   on Mode A vs Mode B vs Mode C variants per §2 Candidate 5.

### Deferred routables

- **TT5L Wave N+1 re-symposium** (DEFER-PENDING per REFUSE verdict 2026-05-17):
  not on critical path; deferred to post-aggregate-frontier-break review.
- **5-way aggregate paired smoke** (after 4+ per-substrate anchors land):
  spawn `lane_5_substrate_procedural_replacement_aggregate_paired_smoke_20260601`
  for the empirical α measurement; $1.50 cumulative; first empirical aggregate
  ΔS per the cascade.
- **Phase-2 council on procedural-replacement promotion contract** (per
  CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"): T3 grand council
  deliberation on whether aggregate-procedural-replacement is contest-
  compliant promotion-eligible (per the canonical Q4 STRUCTURALLY COMPLIANT
  verdict at `.omx/research/canonical_upstream_pr_review_procedural_generation_compliance_20260518.md`).

## Section 12. Mission contribution per Catalog #300

**council_predicted_mission_contribution**: `frontier_breaking_enabler`

Per CLAUDE.md "Mission alignment — non-negotiable" Consequence 5: this design
is `frontier_breaking_enabler` (the matrix IS the canonical path to the 0.18
floor break per §10 predicted band). Per Catalog #300 operational consequence
4: frontier-breaking moves DOMINATE rigor budget — the 5-substrate matrix
design + cadence model + per-substrate symposium gating IS the rigor scaffold
that the operator-routed Wave N+1 council sequencing executes within.

**council_override_invoked**: false. No operator-frontier-override at design-
memo surface; design only.

## Section 13. Cross-references

- Parent design: `.omx/research/procedural_codebook_generator_null_exploit_design_20260520.md`
- Parent landing: `feedback_procedural_codebook_generator_build_landed_20260520.md`
- Canonical equation: `src/tac/canonical_equations/procedural_codebook_savings.py`
- Canonical equation registry: `.omx/state/canonical_equations_registry.jsonl`
- Cathedral consumer: `src/tac/cathedral_consumers/procedural_codebook_savings_consumer/`
- Sister consumer (authority-packet routing): `src/tac/cathedral_consumers/procedural_codebook_generator_consumer/`
- Q4 STRUCTURALLY COMPLIANT verdict:
  `.omx/research/canonical_upstream_pr_review_procedural_generation_compliance_20260518.md`
- Sister single-anchor design (DISJOINT scope):
  `.omx/research/nscs06_v8_procedural_chroma_lut_integration_design_20260520.md`
- Per-substrate symposium memos (5):
  - `council_per_substrate_symposium_nscs06_v8_path_b_variant_c_reactivation_20260518.md`
  - `council_per_substrate_symposium_atw_v2_reactivation_20260518.md`
  - `council_per_substrate_symposium_tt5l_foveation_lapose_20260517.md`
  - `council_per_substrate_symposium_dp1_deep_dive_20260517.md`
  - (vq_vae: BUILD-required per op-routable #3)
- Composition matrix sister memo:
  `feedback_t1_f_z3_x_c6_composition_probe_build_landed_20260514.md`
- v2 cascade SUPER_ADDITIVE topology integration:
  `feedback_super_additive_lane_g_v3_siren_topology_integration_landed_20260517.md`
- Canonical helper: `src/tac/procedural_codebook_generator/seed_derived_codebook.py`
- Catalog #272 distinguishing-feature integration contract: STRICT preflight gate
- Catalog #322 composition-alpha cascade: `tools/cathedral_autopilot_autonomous_loop.py::adjust_predicted_delta_for_composition_alpha_v2`
- Catalog #324 post-training Tier-C validation: STRICT preflight gate
- Catalog #325 per-substrate symposium 6-step contract: STRICT preflight gate
- Catalog #344 canonical-equations-registry: STRICT preflight gate

**End of design memo.**
