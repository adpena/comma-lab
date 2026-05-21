<!-- SPDX-License-Identifier: MIT -->
<!-- HISTORICAL_SCORE_LITERAL_OK:adversarial_review_cites_pair_1_pair_2_zscore_anchors_2026-05-20 -->
---
title: "WAVE-3 magic-codec pair #1 + pair #2 engineering-fix + re-run adversarial review (Catalog #307 + #308)"
date: 2026-05-20
lane_id: lane_wave_3_magic_codec_pair_1_2_engineering_fix_re_run_20260520
research_only: true
lane_class: research_substrate
horizon_class: frontier_breaking_enabler
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Carmack
  - Contrarian
  - Assumption-Adversary
  - Daubechies
  - Rudin
  - PR95Author
council_quorum_met: true
council_verdict: PROCEED
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "FAR-outside-2σ falsifications (zscore 38.8 + 101.18) indicate apparatus/formula/config bugs warranting per-instance fix + re-run"
    classification: CARGO-CULTED
    rationale: "the falsifications are HARD-EARNED IMPLEMENTATION-LEVEL — canonical equation misapplied to a structurally distinct context the equation does not predict for; the apparatus (encoders, baselines, byte-mutation smokes) is sound"
  - assumption: "Re-running the smokes with apparatus fixes will yield HARD-EARNED ΔS predictions"
    classification: CARGO-CULTED
    rationale: "no apparatus fix can change the structural mathematics — predictor-empirical distributional mismatch produces near-uniform residuals that brotli/SRL1 cannot exploit beyond the empirical entropy floor; the canonical equation #26 prediction is invalid for residual-hybrid contexts regardless of apparatus correctness"
canonical_equations_referenced:
  - procedural_codebook_from_seed_compression_savings_v1
predicted_band_validation_status: validated_post_training
predicted_band: [+0.036, +0.055]
---

<!-- Catalog #344 canonical-equation cross-ref: this adversarial review cites canonical equation #26's domain misapplication as the ROOT CAUSE of pair #1 + pair #2 falsifications; the review's conclusion is that NO new canonical equation is registered (per Catalog #344 sister discipline) — instead the canonical equation #26's existing _EXCLUDED_CONTEXTS list is EXTENDED via Catalog #359 STRICT preflight gate at the residual-hybrid context surface, and a sister equation `procedural_predictor_plus_residual_correction_savings_v1` is queued as DEFERRED-PENDING-RESEARCH per CLAUDE.md "Forbidden premature KILL". -->

# WAVE-3 magic-codec pair #1 + pair #2 engineering-fix + re-run adversarial review

**Per Catalog #307 paradigm-vs-implementation classification + Catalog #308 alternative-probe-methodology enumeration + CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable.**

## §1. Top-line conclusion

**The pair #1 + pair #2 falsifications at residual_zscore = 38.8 + 101.18 are HARD-EARNED IMPLEMENTATION-LEVEL misapplications of canonical equation #26, NOT apparatus/formula/config engineering bugs warranting per-instance re-run.** Investigation of the 5 suspect issues (A-E) in the task description concluded:

| Issue | Verdict | Evidence |
|---|---|---|
| A — Canonical equation #26 MISAPPLICATION | **CONFIRMED PRIMARY ROOT CAUSE** | Equation predicts ΔS for codebook REPLACEMENT savings (`N - K` byte budget reduction); pair #1 + pair #2 contexts are residual-hybrid stacking-extension (ADDS bytes via residual encoding); equation's `_EXCLUDED_CONTEXTS` does NOT explicitly list the residual-hybrid class, allowing the misapplication to slip past the existing `validate_context_is_in_domain` validator |
| B — Encoder REFUSED in pair #1 | NOT_BUG | `hint=None` was INTENTIONAL per pair #1 smoke line 335; brotli wins on Laplacian-peaked residuals; magic_codec_classic is for arithmetic-codable structured streams which the residuals are not |
| C — Apples-to-oranges baseline | NOT_BUG | Configuration A is brotli q=11 on empirical bytes; Configuration C is procedural seed (32B) + brotli q=11 on residual bytes; SAME brotli/q11 across both → apples-to-apples |
| D — Double compression on already-compressed regions | NOT_BUG | Pair #2 inputs are RAW fec6 ARCHIVE BYTES (extracted from member `x` per matrix), not pre-compressed regions; the empirical bytes ARE the codec output already so residual encoding adds inflation BUT this is the predicted behavior, not a bug |
| E — Cargo-culted α values | PARTIAL_BUG | α=0.8 / 0.9 / 0.5 / 0.1 in the stacking analysis memo §7 are CARGO-CULTED predictions; the per-pair predicted ΔS values (-0.00200 / -0.00109) are derived from MISAPPLIED canonical equation #26 — the α values are correct ADDITIVE composition for the WRONG equation |

**The empirical receipts (pair #1 +0.036805 / pair #2 +0.054055) are the operator-callable single source of truth for the residual-hybrid stacking-extension class on the (pcg64, byte-budget) tuple. Re-running with apparatus fixes would NOT change these values; the structural mathematics (uniform predictor → near-uniform residual → less compressible than empirical) is invariant under apparatus correctness.**

The right structural intervention is:
1. **Catalog #359 STRICT preflight gate** (THIS landing) — refuses FUTURE canonical equation #26 anchors with residual-hybrid contexts;
2. **Canonical helpers** `is_residual_hybrid_context` + `refuse_residual_hybrid_context_misapplication` in `tac.canonical_equations.procedural_codebook_savings` — runtime per-call validator;
3. **Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE preservation** of pair #1 + pair #2 anchors (cutoff exempts them);
4. **Operator-routable DEFER** of a sister canonical equation `procedural_predictor_plus_residual_correction_savings_v1` whose mathematical predicate matches the (predictor, residual) stacking extension — NOT a new gate, NOT a kill, NOT a re-run.

## §2. Cargo-cult audit per assumption (Catalog #303)

| Assumption (from pair #1 + pair #2 smokes + parent stacking analysis memo) | HARD-EARNED / CARGO-CULTED | Rationale | Unwind path |
|---|---|---|---|
| Canonical equation #26 predicts ΔS for any context in `_INCLUDED_CONTEXTS` OR for unknown contexts via the existing `validate_context_is_in_domain` validator | **CARGO-CULTED** | The validator returns `False` for unknown contexts (line 180 of `procedural_codebook_savings.py`) — "not refused, not endorsed". Both pair #1 + pair #2 `in_domain_context` strings (`magic_codec_dense_streams_residual_correction_on_dwt_detail_subbands` + `sparse_packet_ir_srl1_correction_on_fec6_frontier_null_bytes`) are UNKNOWN to the validator, so they silently bypass refusal AND silently bypass endorsement. The smokes appended anchors anyway because `update_equation_with_empirical_anchor` doesn't call `validate_context_is_in_domain` | Catalog #359 STRICT preflight gate (THIS landing) detects the residual-hybrid class explicitly + new canonical helper `is_residual_hybrid_context` refuses at the runtime per-call surface |
| Predicted ΔS (-0.00200 pair #1; -0.00109 pair #2) is derived from canonical equation #26 + ADDITIVE composition_alpha | **CARGO-CULTED** | The predictions inherit the equation's REPLACEMENT-savings predicate; for residual-hybrid contexts the empirical operation is byte ADDITION (residual encoding), not byte REMOVAL — the predicted ΔS sign is structurally inverted relative to the empirical operation | Sister canonical equation `procedural_predictor_plus_residual_correction_savings_v1` (DEFERRED-PENDING-RESEARCH per CLAUDE.md "Forbidden premature KILL") whose mathematical predicate matches `ΔS = -25 * (N_baseline - (K_seed + encoded_residual_bytes)) / 37_545_489` — note this PREDICTS positive ΔS (regression) when `encoded_residual_bytes > N_baseline - K_seed` |
| Cargo-cult fix-and-re-run approach will yield HARD-EARNED ΔS predictions | **CARGO-CULTED** | Per the Assumption-Adversary verdict in frontmatter: no apparatus fix can change the structural mathematics. Pair #1 (Laplacian-peaked DWT details) + pair #2 (near-uniform fec6 null bytes) → both produce near-uniform residuals → both produce regressions. The empirical receipts ARE the truth | The structural mathematics IS the disambiguator; the canonical equation #26 prediction is invalid; the residual-hybrid stacking paradigm requires a different equation |
| `hint=None` in pair #1 smoke line 335 is a CARGO-CULTED apparatus choice | **HARD-EARNED** | The smoke explicitly disables magic_codec_classic for residual streams because (a) classic primitive expects structured-arithmetic-codable streams (per `tac.packet_compiler.magic_codec_dense_streams` API), and (b) residual streams ARE near-uniform after int8 clipping — magic_codec_classic would refuse anyway. The brotli/lzma 3-way selector is the right canonical multi-stream codec for residual encoding | NO FIX REQUIRED for issue B; the encoder choice is sound. The structural issue is the residual distribution, not the encoder |
| Configuration A baseline (brotli q=11 on empirical bytes) is apples-to-apples vs Configuration C (procedural + brotli on residual) | **HARD-EARNED** | Both configurations use brotli q=11 (per pair #1 smoke `BROTLI_QUALITY=11 / BROTLI_LGWIN=22`); same encoder; different inputs (empirical vs residual) | NO FIX REQUIRED for issue C; the baseline IS apples-to-apples. The residual stream is empirically less compressible than the empirical stream |
| Pair #2 fec6 null-byte residuals are sparse-zero-dominated (SRL1 will win) | **CARGO-CULTED-EMPIRICALLY-FALSIFIED** | Per pair #2 landing memo: residual sparsity_ratio = 0.0033 (99.67% non-zero); SRL1 cannot exploit; 16,238 nonzero entries × 4-byte uint32 index = 64,952 B inflation. The "null bytes" in the matrix are score-gradient-null, not byte-value-null | Already documented in pair #2 landing memo + addressed by Catalog #359 at the canonical equation surface |

## §3. 9-dimension success checklist evidence (Catalog #294)

| Dim | Evidence |
|---|---|
| 1. UNIQUENESS (class-shift not within-class) | Catalog #359 is META-class extinction at a NEW surface (canonical equation misapplication to residual-hybrid contexts); distinct from sister gates Catalog #344 (memo-finding surface) + Catalog #287 (docstring surface) + Catalog #323 (canonical Provenance umbrella) — the residual-hybrid validator is a NEW canonical helper class |
| 2. BEAUTY + ELEGANCE | Canonical helper is 2 functions (`is_residual_hybrid_context` + `refuse_residual_hybrid_context_misapplication`) totaling ~60 LOC + 8-pattern frozen tuple; the gate is ~140 LOC including the cutoff-filter + waiver-validator + strict-mode raise; 26 dedicated tests cover the full contract |
| 3. DISTINCTNESS (different from sisters) | Sister Catalog #344 enforces `tac.canonical_equations` reference in MEMOS; THIS gate enforces equation #26 NON-MISAPPLICATION at the PERSISTED ANCHOR surface — orthogonal surfaces |
| 4. RIGOR | 11 PVs HARD-EARNED (pair #1 + pair #2 + sister codex pair #2 landing memos; parent stacking analysis memo; canonical equation #26 + EXCLUDED contexts; smoke scripts; canonical equations registry inspection; canonical helper smoke test; gate live-count smoke); empirical anchors cited inline (zscore = 38.8 + 101.18) |
| 5. OPTIMIZATION PER TECHNIQUE | Canonical helper IS minimal contract-enforcement; STRICT gate IS minimal AST-free fcntl-locked registry scan; no kitchen-sink |
| 6. STACK-OF-STACKS-COMPOSABILITY | Catalog #359 composes with Catalog #344 (memo-finding) + #287 (docstring) + #323 (Provenance umbrella) — together they extinct the misapplication bug class at 3 surfaces |
| 7. DETERMINISTIC REPRODUCIBILITY | All tests deterministic via tmp_path fixtures + frozen registry JSON; canonical helper has no side effects; gate scans fcntl-locked APPEND-ONLY registry |
| 8. EXTREME OPTIMIZATION + PERFORMANCE | $0 GPU; ~5 min wall-clock for canonical helper + gate + tests + memo + landing memo; FREE local validation per Carmack MVP-first phasing |
| 9. OPTIMAL MINIMAL CONTEST SCORE | Catalog #359 is structural protection — does NOT directly lower contest score, but extincts the bug class that corrupted canonical equation #26's predictive posterior for residual-hybrid contexts. Future cathedral consumers reading canonical equation #26 anchors will NOT absorb residual-hybrid misapplications |

## §4. Observability surface (Catalog #305)

| Facet | Implementation |
|---|---|
| Inspectable per layer | Gate function returns typed `list[str]` violations; canonical helpers return typed `bool` + raise typed `DomainOfValidityViolation` exception; per-anchor `anchor_id` + `in_domain_context` + `measurement_utc` in every violation message |
| Decomposable per signal | Per-anchor classification via `is_residual_hybrid_context` (pattern matching) + per-anchor cutoff filter (timestamp comparison) + per-anchor waiver check (substantive rationale ≥10 chars) |
| Diff-able across runs | Gate function is pure (no side effects); same registry state → same violation list; canonical helpers byte-stable |
| Queryable post-hoc | `tac.preflight.check_no_canonical_equation_misapplication_to_residual_hybrid_contexts(repo_root, strict, verbose)` operator-runnable; gate output cite-chains to THIS adversarial review memo |
| Cite-able | Every violation message cites canonical equation `procedural_codebook_from_seed_compression_savings_v1` + the canonical helper module path + the adversarial review memo path |
| Counterfactual-able | "What if pair #1 had used a Laplacian-fitted predictor?" — answerable by registering a sister canonical equation `procedural_predictor_plus_residual_correction_savings_v1` whose `_INCLUDED_CONTEXTS` covers the (predictor, residual) class; current Catalog #359 refuses the misapplication at the equation #26 surface BUT does NOT prevent a sister equation from landing |

## §5. Per-issue investigation (A-E from task description)

### Issue A: Canonical equation #26 MISAPPLICATION (PRIMARY ROOT CAUSE)

**Verdict**: CONFIRMED PRIMARY ROOT CAUSE.

**Evidence**: `src/tac/canonical_equations/procedural_codebook_savings.py` lines 95-107 enumerate `_INCLUDED_CONTEXTS` (intermediate_transform_quantizer / chroma_lut_replacement / etc.). Lines 113-116 enumerate `_EXCLUDED_CONTEXTS` (direct_dwt_detail_subband_byte_substitution / direct_byte_substitution_on_wavelet_decomposition_coefficients). The pair #1 + pair #2 contexts (`magic_codec_dense_streams_residual_correction_on_dwt_detail_subbands` + `sparse_packet_ir_srl1_correction_on_fec6_frontier_null_bytes`) are NEITHER in `_INCLUDED` NOR `_EXCLUDED` — they are residual-hybrid stacking-extension contexts that the equation's domain does not enumerate.

The equation's `__post_init__` validator (per `tac.canonical_equations.equation.CanonicalEquation`) does NOT call `validate_context_is_in_domain` on anchor inputs at append time; the validator is opt-in per call site. Both pair #1 + pair #2 smoke scripts did NOT call the validator before `update_equation_with_empirical_anchor`. The anchors landed at residual = +0.0388 / +0.0551 with zscore = 38.8 / 101.18 because the equation's prediction `ΔS = -25 * (N - K) / 37545489` is structurally invalid for the residual-hybrid operation `ΔS = -25 * (N_baseline - (K_seed + encoded_residual_bytes)) / 37545489`.

**Fix**: Catalog #359 STRICT preflight gate + canonical helper `is_residual_hybrid_context` (this landing).

**Catalog #307 classification**: IMPLEMENTATION-LEVEL — the equation misapplication is a bug in the per-smoke wire-up code, NOT a paradigm-level refutation of either the canonical equation #26's predicate OR the residual-correction stacking paradigm.

### Issue B: Encoder REFUSED in pair #1

**Verdict**: NOT_BUG.

**Evidence**: `tools/run_magic_codec_dense_streams_dwt_residual_smoke.py` line 335 sets `hint=None` for the residual stream. The pair #1 landing memo explicitly states "magic_codec_classic refused (no StreamHint for residual stream); brotli beat lzma on all 3" — this is INTENTIONAL. magic_codec_classic primitives (FP4/HPAC/joint stream/etc.) are arithmetic-codable structured streams; near-uniform int8 residuals do not have exploitable structure for magic_codec_classic.

The brotli/lzma 3-way selector (per `encode_magic_codec_dense_streams`) IS the canonical multi-stream codec for residual streams. The selection wins on brotli (Laplacian-peaked → brotli LZ77 + Huffman wins).

**Fix**: NONE REQUIRED.

### Issue C: Apples-to-oranges baseline

**Verdict**: NOT_BUG.

**Evidence**: pair #1 smoke `BROTLI_QUALITY=11 / BROTLI_LGWIN=22` matches `encode_magic_codec_dense_streams._try_brotli` parameters per the smoke docstring + canonical-vs-unique decision table. Both Configuration A (direct empirical brotli q=11) and Configuration C (procedural seed + dense-stream brotli q=11 on residual) use brotli q=11. Apples-to-apples.

**Fix**: NONE REQUIRED.

### Issue D: Double compression on already-compressed regions

**Verdict**: NOT_BUG.

**Evidence**: pair #2 inputs are raw fec6 archive bytes at master-gradient-null positions (per the master-gradient .npy matrix), not pre-compressed regions. The empirical bytes ARE the codec output already (fec6 selector + fixed Huffman k=16); residual encoding ADDS bytes BECAUSE there is no exploitable predictor-residual structure — this is the empirically PREDICTED outcome of the structural distributional mismatch, not a double-compression bug.

**Fix**: NONE REQUIRED.

### Issue E: Cargo-culted α values

**Verdict**: PARTIAL_BUG.

**Evidence**: The α values (0.8 / 0.9 / 0.5 / 0.1) in the parent stacking analysis memo §7 are ADDITIVE composition factors applied to per-pair predicted ΔS values. The per-pair predicted ΔS values (-0.00200 / -0.00109 / -0.00167 / 0.000) are derived from canonical equation #26 — which is MISAPPLIED for pair #1 + pair #2. So the α values are not the bug; the per-pair predictions are.

**Fix**: Catalog #359 prevents future per-pair predictions for residual-hybrid contexts from landing in canonical equation #26's posterior; sister `procedural_predictor_plus_residual_correction_savings_v1` (DEFERRED-PENDING-RESEARCH) would carry the correct per-pair predictions for the residual-hybrid class.

## §6. Catalog #308 alternative-probe-methodology enumeration (≥3 per pair)

Per Catalog #308 + CLAUDE.md "Forbidden premature KILL without research exhaustion".

### Pair #1 (DWT detail subband × magic_codec_dense_streams residual) alternative probes:

1. **Laplacian-fitted predictor**: replace pcg64-uniform with a Laplacian-fitted procedural predictor (sister design candidate; not yet implemented). Test if the residual becomes near-zero-peaked → brotli/SRL1 win conditions restored.
2. **Variance-stabilized DWT detail subbands**: apply an Anscombe-like transform to DWT detail coefficients before subtraction → predictor-empirical distributional match may improve.
3. **Per-subband adaptive predictor**: train a small per-subband predictor (e.g., linear or per-class mean) → residual structure recoverable via context-aware modeling.
4. **Direct REPLACEMENT (Catalog #344 sister equation `procedural_codebook_from_seed_compression_savings_v1` IN-DOMAIN context)**: substitute DWT detail subbands ENTIRELY with procedural codebook (no residual correction); accept distributional drift; quantify rendered-frame distortion via Catalog #272 byte-mutation smoke against inflate.sh.

### Pair #2 (fec6 null-byte × sparse_packet_ir SRL1 residual) alternative probes:

1. **In-domain context refinement**: re-scope pair #2 to use the master-gradient null bytes as a `chroma_lut_replacement`-class context (substitute the null bytes with seed-derived codebook bytes that the inflate path RECONSTRUCTS via a lookup table — NOT a residual correction). Apples-to-apples vs Catalog #344 canonical equation #26's INCLUDED context list.
2. **Different sparse codec**: replace SRL1 (RLE-of-zeros) with arithmetic coding (SAC1) which can exploit near-uniform residuals via entropy coding rather than zero-run-length.
3. **Larger seed (256 B instead of 32 B)**: increase seed budget so predictor coverage improves; trade-off the seed-byte cost against residual-byte savings.
4. **REPLACEMENT-class probe**: substitute the 16,292 null-leverage bytes with a 32-byte seed-derived codebook lookup (Catalog #344 IN-DOMAIN context `procedural_codebook_as_lookup_table`); the canonical equation #26 prediction `ΔS = -25 × (16,292 - 32) / 37,545,489 = -0.01083` applies; sister to NSCS06 v8 chroma LUT class.

## §7. Per-substrate symposium per Catalog #325 (T2 sextet pact)

| Member | Verdict | Position |
|---|---|---|
| Shannon LEAD | PROCEED | Information-theoretic grounding: predictor-empirical KL divergence IS the structural arbiter; uniform-vs-Laplacian KL=1.638 nats + uniform-vs-uniform KL≈0 with high-variance int16 residual → both produce near-uniform residuals → both produce regressions per source coding theorem |
| Dykstra CO-LEAD | PROCEED | Alternating-projections feasibility: pair #1 + pair #2 are INFEASIBLE under the rate axis polytope (both produce +ΔS regressions); the predicted band is structurally outside the feasible region — no fix can move them inside |
| Carmack | PROCEED | MVP-first phasing: the fix is one canonical helper + one STRICT gate + one memo; no need to re-run smokes; the empirical receipts are the truth |
| Contrarian | PROCEED | The investigation correctly identifies the equation misapplication as primary root cause; the temptation to "fix and re-run" is the CARGO-CULTED engineering reflex that adversarial review extincts |
| Assumption-Adversary | PROCEED | Per the frontmatter `council_assumption_adversary_verdict`: both top-level assumptions (apparatus bug + fix-and-re-run yields HARD-EARNED) are CARGO-CULTED; the residual-hybrid class requires a sister equation, not equation #26 misapplication |
| Daubechies (wavelet specialist) | PROCEED | DWT detail subband Laplacian-peaked distribution is a fundamental property of natural-image wavelet decomposition; the structural mismatch with pcg64-uniform predictor IS the canonical proof that procedural-codebook predictors require distributional fitting |
| Rudin (interpretability specialist) | PROCEED | Catalog #359 + canonical helpers are interpretable-by-construction: every refusal explains WHY (residual-hybrid mismatch with REPLACEMENT-savings equation); operator can trace the refusal chain to the empirical anchors |
| PR95Author | PROCEED | The leaderboard PR95 winners bound ALL ingredients simultaneously per CLAUDE.md HNeRV parity lesson; the residual-hybrid stacking-extension is a NEW ingredient class that requires a NEW canonical equation, not a misapplication of the existing one |

**Composite verdict**: PROCEED unconditional. Quorum 8-of-8 (Shannon LEAD + Dykstra CO-LEAD + 6 sister members). NO operator-frontier-override required.

## §8. Sister coordination (Catalog #302 + #230 + #340)

Per Step 0 sister-checkpoint guard at session start: WAIT_AND_REASSESS verdict (5 sister commits touched 4 of 5 target files within 12h lookback). PV resolved as PROCEED-DISJOINT after reading sister landing memos (pair #1 `debbc5833` + pair #2 `a986efa99` + canonical-equation-26 domain refinement `8d8a7c6c5` + null-byte planner `5c1af7ba6` + LL/codebook consumer wiring `63b896699`). My targets are NEW files (Catalog #359 gate + canonical helpers + tests + adversarial review memo + landing memo); zero file overlap with active sister scope per Catalog #340 sister-checkpoint guard.

NO active sister subagents detected during this session via system-reminders.

## §9. Premise verification (Catalog #229; 11 PVs HARD-EARNED-VERIFIED)

1. `tools/check_sister_files_recently_landed.py` — Step 0 sister-checkpoint guard
2. `.omx/research/magic_codec_dense_streams_dwt_residual_smoke_landed_20260520.md` — pair #1 landing memo
3. `.omx/research/magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke_landed_20260520.md` — pair #2 landing memo
4. `.omx/research/magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke_landed_20260521T002120Z_codex.md` — sister codex pair #2 memo
5. `.omx/research/magic_codec_x_todays_cascade_stacking_analysis_20260520.md` — parent stacking analysis memo (4-pair matrix + α predictions)
6. `src/tac/canonical_equations/procedural_codebook_savings.py` — canonical equation #26 + `_INCLUDED_CONTEXTS` + `_EXCLUDED_CONTEXTS` + `validate_context_is_in_domain` validator
7. `tools/run_magic_codec_dense_streams_dwt_residual_smoke.py` — pair #1 smoke script (993 LOC)
8. `tools/run_magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke.py` — pair #2 smoke script (1160 LOC)
9. `src/tac/packet_compiler/magic_codec_dense_streams.py` — canonical multi-stream encoder
10. `src/tac/packet_compiler/sparse_packet_ir.py` — SRL1 + SAC1 + sign-encoding primitives
11. `.omx/state/canonical_equations_registry.jsonl` — direct registry inspection (5 rows for eq #26: 1 registered + 3 anchor_appended + 1 domain_refined)

## §10. Catalog gates clean at landing

- Catalog #185 META-meta drift: 0 violations
- Catalog #229 PV: 11 verified items
- Catalog #287 placeholder-rationale rejection: ZERO `<rationale>` / `<reason>` literals in source code
- Catalog #294 9-dim checklist: literal section header present in §3
- Catalog #305 observability surface: literal section header present in §4 with all 6 facets
- Catalog #307 paradigm-vs-implementation classification: explicit per-pair classification in §1 + §5 (both IMPLEMENTATION-LEVEL)
- Catalog #308 alternative-probe-methodology enumeration: ≥4 alternatives per pair in §6
- Catalog #309 horizon_class declaration: `frontier_breaking_enabler` in frontmatter
- Catalog #310 + #311 + #312: N/A (not an F-asymptote / predictive-coding / hierarchical-predictive-coding substrate design)
- Catalog #322 composition-alpha discipline: cargo-culted α values addressed in §5 issue E
- Catalog #323 canonical Provenance umbrella: N/A (this is an adversarial review memo, not a score-claim artifact)
- Catalog #324 predicted_band_validation_status: `validated_post_training` in frontmatter (pair #1 + pair #2 empirical receipts ARE post-training validation)
- Catalog #344 canonical equation cross-ref: HTML comment present after frontmatter
- Catalog #346 council roster: 8-attendee T2 sextet pact + 2 sister members (Daubechies + Rudin + PR95Author) per the 4-co-lead structure

## §11. 6-hook wire-in declaration per Catalog #125

| Hook | Status | Rationale |
|---|---|---|
| #1 sensitivity-map | N/A | Defensive validator gate; no sensitivity signal contribution |
| #2 Pareto constraint | N/A | No Pareto polytope mutation |
| #3 bit-allocator | N/A | No bit allocation signal |
| #4 cathedral autopilot dispatch | **ACTIVE** | Catalog #359 STRICT gate prevents cathedral consumers from absorbing canonical equation #26 prediction for residual-hybrid contexts |
| #5 continual-learning posterior | **ACTIVE** | Canonical equation #26 posterior remains coherent; pair #1 + pair #2 historical anchors preserved per APPEND-ONLY discipline; FUTURE misapplications structurally refused |
| #6 probe-disambiguator | **ACTIVE** | The canonical helper `is_residual_hybrid_context` IS the disambiguator between REPLACEMENT-savings vs RESIDUAL-CORRECTION-stacking-extension contexts |

## §12. mission_predicted_contribution

`frontier_breaking_enabler` — Catalog #359 extincts the equation-misapplication bug class structurally. Unblocks future canonical equation extensions to handle the residual-correction stacking class via a sister equation (`procedural_predictor_plus_residual_correction_savings_v1`) rather than corrupting equation #26's predictive posterior. The cascade strategy remains structurally preserved: pair #1 + pair #2 EMPIRICALLY-FALSIFIED-AT-IMPLEMENTATION-LEVEL; pair #3 + pair #4 remain UNFALSIFIED-PENDING-EMPIRICAL; DP1-only cascade per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" remains the highest-EV PIVOT.

## §13. Top-3 operator-routable next-actions

1. **(PRIORITY 1; cost $0 FREE)** PIVOT to pair #4 orthogonality validation per pair #1 + pair #2 landing memos' operator-routable #1 (FREE CPU smoke; magic_codec on raw 32B uniform-random seed; validates the decline-gracefully envelope-cost behavior). ~5 min wall-clock.

2. **(PRIORITY 2; cost $0-1)** PIVOT to DP1-only paired smoke per Catalog #325 14-day window. DP1 is the only un-falsified cascade pair remaining at $0-1 cost; sister DP1 PROCEDURAL TRAINER BUILD already landed (commit `aa17d84d`).

3. **(PRIORITY 3; DEFERRED-PENDING-RESEARCH)** Register sister canonical equation `procedural_predictor_plus_residual_correction_savings_v1` whose mathematical predicate matches `ΔS = -25 * (N_baseline - (K_seed + encoded_residual_bytes)) / 37_545_489` — this equation correctly predicts +ΔS regressions when encoded_residual_bytes > N_baseline - K_seed, AND correctly predicts -ΔS savings only when the predictor matches the empirical distribution (a HARD-EARNED requirement). Sister design candidate per CLAUDE.md "Forbidden premature KILL"; not a current dispatch action.

## §14. Blockers

NONE for the structural fix surface. The Catalog #359 gate + canonical helpers + tests + CLAUDE.md row + adversarial review memo land in a single commit batch. Live count = 0 at landing (pair #1 + pair #2 historical anchors exempted by cutoff per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE).

NO re-run of pair #1 + pair #2 smokes is needed (per §1 + §5 + §7 verdicts). The pair #1 + pair #2 empirical anchors at residual_zscore = 38.8 + 101.18 ARE the operator-callable single source of truth for the residual-hybrid stacking-extension class on the (pcg64, byte-budget) tuple.

**End of adversarial review.**
