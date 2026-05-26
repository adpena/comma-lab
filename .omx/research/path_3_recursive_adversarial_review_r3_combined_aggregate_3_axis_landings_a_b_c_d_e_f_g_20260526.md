<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE — R3-COMBINED AGGREGATE review record for Path 3 candidates A + B' + C' + D + E + F + G (7 substrates landed 2026-05-26). DO NOT mutate after landing. -->
<!-- Catalog #344 canonical equation cross-ref: cross-substrate META review across 7 sister R3 per-substrate memos; FIX-WAVE-R3-COMBINED op-routable queue priority-ranked. FORMALIZATION_PENDING:r3_combined_aggregate_methodology_pending_council_revisions_per_recursive_adversarial_review_protocol_close_paths_item_8_assumption_challenge_axis -->
---
council_tier: T3
council_attendees:
  - Shannon
  - Dykstra
  - Rudin
  - Daubechies
  - Tao
  - Carmack
  - Hotz
  - Quantizr
  - MacKay
  - Selfcomp
  - Ballé
  - Hassabis
  - PR95Author
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED
council_dissent:
  - member: Contrarian
    verbatim: "Per G=NIRVANA R3 verdict (PROCEED_WITH_REVISIONS): 1 IMPLEMENTATION-LEVEL test regression introduced by Wave #1 wire-in. The aggregate R3-COMBINED verdict is PROCEED OVERALL (6 of 7 substrates CLEAN; 3 of 7 reach 3/3 SEAL) BUT G=NIRVANA's counter RESETS per protocol item 3. Per Catalog #307 paradigm-vs-implementation classification, this is IMPLEMENTATION-LEVEL fix only; the paradigm is INTACT. FIX-WAVE-R3 is REQUIRED (1-line test update for G=NIRVANA) before the aggregate R3-COMBINED scope is fully CLEAN."
council_assumption_adversary_verdict:
  - assumption: "T3 tier is appropriate for an AGGREGATE R3-COMBINED review spanning 7 sister landings (post-CONSOLIDATE-OP-1 + post-Wave-#1 state)"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'Council hierarchy: 4-tier protocol' Tier elevation triggers T2→T3 trigger (a): the aggregate finding touches a CLAUDE.md non-negotiable (Recursive adversarial review protocol — close paths — items 1-8) AND touches Catalog #335/#344/#346/#205/#127/#287/#323 across the 7 substrates. R1 + R1' + R2-COMBINED aggregates were T3; consistency. T3 cadence budget ≤3/week per CLAUDE.md is at-budget (R1 was 1st this week; R1' was 2nd; R2-COMBINED was 3rd; R3-COMBINED is the 4th — over budget per Catalog #300 mission_alignment v2 frontmatter discipline) — operator approval per the R3-COMBINED charter overrides per CLAUDE.md 'Mission alignment' operational consequence 1 (operator-frontier-override at ALL tiers; documented escape hatch). The R3-COMBINED review IS the canonical 3rd round of recursive review per protocol; declining to run would violate the protocol non-negotiable."
  - assumption: "Aggregate R3-COMBINED PROCEED verdict (6 of 7 CLEAN; 3 reach 3/3 SEAL; G=NIRVANA RESETS) is structurally sound — per-substrate verdicts honored per protocol item 3"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "Empirical re-verification 2026-05-26T09:11Z: 6 of 7 per-substrate R3 memos return PROCEED (CLEAN PASS); G=NIRVANA returns PROCEED_WITH_REVISIONS (1 test regression introduced by Wave #1). Aggregate test re-verification: A=11p, B'=192p, C'=28p, D=68p, E=27p, F=64p, G=55p+1f (375 PASS + 1 FAIL of 376 total). The 1 FAIL is the G-OP1 finding — Wave #1 wire-in added 4 canonical exports to G's __all__ but did NOT update the strict-equality test in test_basic.py:42-59. Per CLAUDE.md 'Recursive adversarial review protocol — close paths' item 3: 'A round with zero issues is a clean pass. The counter resets to 0 whenever a round finds any issue.' G's counter RESETS 1/3 → 0/3; other 6 substrates advance per item 4."
  - assumption: "The 3-axis methodology empirically validates across post-CONSOLIDATE + post-Wave-#1 state (canonical META-LAYER delegation pattern + canonical posterior emission pattern do NOT regress 3-axis verdicts)"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "Per-substrate Axis 2 empirical re-measurement post-CONSOLIDATE-OP-1: PixelShuffle 2x NHWC max_abs vs PyTorch = 0.0 across A+D+F (improved from R2's FIX-WAVE-R1/R1' anchors of 0.0054/0.0072/post-fix to canonical byte-stable). Per-substrate Axis 1 canonical equation registration: 7 of 7 substrates now declare CANONICAL_EQUATION_IDS at L0 (R2 META op-routable for B'+D+G GAP CLOSED — Wave #1 wire-in resolved). Per-substrate Axis 3 sister-canonical numpy reference (G=NIRVANA): preserved via META-LAYER consumption pattern (cascade_reconstruct retained locally; 7 primitives re-exported from canonical pr95_hnerv_numpy_reference). The 3-axis methodology empirically IMPROVES post-CONSOLIDATE + post-Wave-#1 state for 6 of 7 substrates; the 1 G regression is IMPLEMENTATION-LEVEL test-side fix (NOT architectural)."
  - assumption: "The CONSOLIDATE-OP-1 META-LAYER extraction pattern (canonical helper + delegation) is the canonical structural extinction of the PYTORCH-EXPORT-BOUNDARY-DRIFT bug class across Path 3 future candidates"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "Empirical proof: A=DreamerV3 + F=Z8 pre-CONSOLIDATE both required substrate-local FIX-WAVE patches (commit e1b101888 + 4684dbbab) for the SAME PixelShuffle channel-LAST drift bug. Post-CONSOLIDATE-OP-1, both substrates' max_abs improves to 0.0 (canonical byte-stable). The recurrence bug class (F inherited A's pre-fix code LINE-FOR-LINE per R1' empirical proof) is STRUCTURALLY EXTINCTED — future Path 3 candidates (H/I/J/K queued, plus future L/M/N/O) inherit byte-stable PixelShuffle by importing canonical pr95_hnerv_mlx rather than re-implementing. Per CLAUDE.md 'Bugs must be permanently fixed AND self-protected against' non-negotiable: META-LAYER consolidation IS the canonical structural protection."
council_decisions_recorded:
  - "Aggregate R3-COMBINED verdict: PROCEED (6 of 7 CLEAN) with REVISIONS REQUIRED for G=NIRVANA"
  - "Per-substrate counter advancement: A=1/3→2/3, B'=2/3→3/3 SEAL, C'=2/3→3/3 SEAL, D=2/3→3/3 SEAL, E=1/3→2/3, F=1/3→2/3, G=1/3→0/3 RESET"
  - "**3 SUBSTRATES REACH 3/3 SEAL = PAID CUDA DISPATCH AUTHORIZED**: D=Z6, B'=Z7-Mamba-2-v2, C'=NSCS06-v8 chroma_lut"
  - "1 R3 FINDING G-OP1: G=NIRVANA test_module_exposes_canonical_public_api strict-equality test (test_basic.py:46-58) needs 1-line update to accept 4 Wave #1 canonical exports"
  - "FIX-WAVE-R3-COMBINED REQUIRED — 1 op-routable G-OP1 (1-line test fix) BEFORE R4 fires"
  - "All 7 substrates now declare CANONICAL_EQUATION_IDS at L0 (R2 META #3 op-routable GAP CLOSED for B'+D+G via Wave #1)"
  - "CONSOLIDATE-OP-1 META-LAYER consolidation empirically IMPROVES Axis 2 drift across A+D+F (max_abs = 0.0 canonical byte-stable)"
  - "Per CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA' non-negotiable: paid CUDA dispatch authorization triggers operator-routable next steps for D/B'/C' (paired CUDA + contest-CPU + Catalog #1265 contest-equivalence gate)"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_pursuit
canonical_equation_refs:
  - categorical_posterior_capacity_vs_continuous_gaussian_v1
  - categorical_blahut_arimoto_rate_distortion_v1
  - procedural_predictor_plus_residual_correction_savings_v1
  - procedural_codebook_from_seed_compression_savings_v1
  - mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1
  - scorer_conditional_joint_rate_distortion_floor_v1
  - ego_motion_concentration_prior_v1
  - cross_codec_super_additive_orthogonality_predictor_v1
predicted_band_validation_status: pending_post_training
related_deliberation_ids:
  - path_3_a_recursive_adversarial_review_r3_combined_3_axis_20260526
  - path_3_b_recursive_adversarial_review_r3_combined_3_axis_20260526
  - path_3_c_recursive_adversarial_review_r3_combined_3_axis_20260526
  - path_3_d_recursive_adversarial_review_r3_combined_3_axis_20260526
  - path_3_e_recursive_adversarial_review_r3_combined_3_axis_20260526
  - path_3_f_recursive_adversarial_review_r3_combined_3_axis_20260526
  - path_3_g_recursive_adversarial_review_r3_combined_3_axis_20260526
  - path_3_recursive_adversarial_review_r2_combined_aggregate_3_axis_landings_a_b_c_d_e_f_g_20260526
  - path_3_consolidate_op_1_canonical_mlx_primitives_extraction_landed_20260526
  - path_3_wave_1_posterior_emission_canonical_wire_in_landed_20260526
---

# R3-COMBINED Recursive Adversarial Review — AGGREGATE across Path 3 candidates A + B' + C' + D + E + F + G

**Per binding operator directive 2026-05-26 #3**: *"we also need adversarial review against all landing recursive for math and scientific and engineering rigor and for MLX drift minimization and portability via numpy"*

**Per CLAUDE.md "Recursive adversarial review protocol — close paths"**: R3 of 3 consecutive clean-pass cycles required before code is cleared for L1 dispatch authorization. R3-COMBINED fires AFTER R1 + R1' + R2-COMBINED + CONSOLIDATE-OP-1 + Wave #1 posterior emission landed.

**Aggregate R3-COMBINED verdict**: **PROCEED — 6 of 7 substrates CLEAN; 3 reach 3/3 SEAL; G=NIRVANA needs 1-line FIX-WAVE-R3**

**Cost**: $0 GPU; ~3h wall-clock (per-substrate review ~20-25 min each × 7 + aggregate synthesis ~30 min + empirical re-verification + CONSOLIDATE/Wave-#1 verification ~30 min)

---

## Per-landing R3-COMBINED verdict summary

| Landing | Substrate path | R3 Verdict | Counter advance? | Findings count | FIX-WAVE-R3-COMBINED required? |
|---|---|---|---|---|---|
| **D=Z6 predictive-coding** | `src/tac/substrates/time_traveler_l5_z6/` | **PROCEED — CLEAN PASS** | YES (2/3 → **3/3 SEAL**) | 0 | NO |
| **B'=Z7-Mamba-2-v2 fresh substrate** | `src/tac/substrates/z7_mamba2_v2_fresh_substrate/` | **PROCEED — CLEAN PASS** | YES (2/3 → **3/3 SEAL**) | 0 | NO |
| **C'=NSCS06 v8 chroma_lut** | `src/tac/substrates/nscs06_v8_chroma_lut/` | **PROCEED — CLEAN PASS** | YES (2/3 → **3/3 SEAL**) | 0 | NO |
| **A=DreamerV3 RSSM** | `src/tac/substrates/dreamer_v3_rssm/` | **PROCEED — CLEAN PASS** | YES (1/3 → 2/3) | 0 | NO |
| **E=BoostNeRV against PR110** | `src/tac/substrates/boost_nerv/` | **PROCEED — CLEAN PASS** | YES (1/3 → 2/3) | 0 | NO |
| **F=Z8 hierarchical predictive coding** | `src/tac/substrates/z8_hierarchical_predictive_coding/` | **PROCEED — CLEAN PASS** | YES (1/3 → 2/3) | 0 | NO |
| **G=NIRVANA cascading NeRV** | `src/tac/substrates/nirvana_cascading_nerv/` | **PROCEED_WITH_REVISIONS** | **NO (1/3 → 0/3 RESET)** | **1 (G-OP1)** | **YES (1-line test fix)** |

**Aggregate**: 6/7 substrates CLEAN; 3/7 reach 3/3 SEAL; 1/7 (G=NIRVANA) RESETS due to Wave #1 test regression. **FIX-WAVE-R3-COMBINED REQUIRED for G-OP1 only.**

---

## Per-substrate cumulative counter status (the operator-facing scorecard)

| Substrate | Pre-R3 counter | R3-COMBINED verdict | Post-R3 counter | Status |
|---|---|---|---|---|
| **D=Z6** | 2/3 | CLEAN | **3/3 SEAL** | **PAID CUDA DISPATCH AUTHORIZED** |
| **B'=Z7-Mamba-2-v2** | 2/3 | CLEAN | **3/3 SEAL** | **PAID CUDA DISPATCH AUTHORIZED** (modulo MPS-Win precondition per Phase 3 §6) |
| **C'=NSCS06-v8 chroma_lut** | 2/3 | CLEAN | **3/3 SEAL** | **PAID CUDA DISPATCH AUTHORIZED** (modulo L1 promotion blocker) |
| **A=DreamerV3 RSSM** | 1/3 | CLEAN | **2/3** | R4 next; 1 more round needed |
| **E=BoostNeRV** | 1/3 | CLEAN | **2/3** | R4 next; 1 more round needed |
| **F=Z8** | 1/3 | CLEAN | **2/3** | R4 next; 1 more round needed |
| **G=NIRVANA** | 1/3 | NOT CLEAN | **0/3 RESET** | FIX-WAVE-R3 + R4/R5/R6 needed (3 more rounds after fix) |

---

## **3/3 SEAL substrates — PAID CUDA DISPATCH AUTHORIZED**

Per the R3-COMBINED charter: *"Identify substrates at 3/3 = PAID CUDA DISPATCH AUTHORIZED"*

### D=Z6 predictive-coding — most-advanced; FIRST Path 3 contest-CUDA score candidate

- **Counter**: 3/3 SEAL
- **Canonical equation**: `ego_motion_concentration_prior_v1` (REGISTERED)
- **Sister-canonical status**: META-LAYER consumption pattern preserved (canonical pr95_hnerv_mlx is the source; D=Z6 was the seed convention)
- **L1-PROMOTION** distinction preserved (`ARCHITECTURE_CLASS='z6_predictive_coding_film_conditioned_next_frame_l1_mlx'`)
- **Operator-routable next**: PyTorch port verification + Catalog #1265 contest-equivalence gate empirical run + paired CUDA + contest-CPU per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"

### B'=Z7-Mamba-2-v2 fresh substrate — cargo-cult-first methodology canonical example

- **Counter**: 3/3 SEAL (modulo Phase 3 §6 MPS-Win precondition)
- **Canonical equation**: 1 declared at L0 via Wave #1 (REGISTERED)
- **3-phase cargo-cult-first methodology**: back-to-back CLEAN across R1' + R2 + R3 validates operator directive #2 compliance
- **Operator-routable next**: MPS proxy probe ($0 GPU) to establish MPS-Win + L1 follow-up subagent (Mamba2V2Cell + Z7MCM3 archive)

### C'=NSCS06-v8 chroma_lut — Catalog #359 sister canonical preserved via Wave #1 wire-in extras

- **Counter**: 3/3 SEAL (modulo L1 promotion blocker)
- **Canonical equation**: `procedural_codebook_from_seed_compression_savings_v1` (REGISTERED with Catalog #359 IN_DOMAIN context routing)
- **Cargo-cult #5 EMPIRICALLY CONFIRMED via FAIL_AT_CLASS_1 verdict at L0** — canonical IMPLEMENTATION-LEVEL falsification per Catalog #307
- **Wave #1 wire-in extras** preserve PROCEDURAL_LUT_SENTINEL.hex() as canonical first-class metadata
- **Operator-routable next**: L1 promotion (wire cls_stream consumption at L0 inflate) + per-substrate Catalog #325 symposium within 14-day window 2026-05-26 → 2026-06-09

---

## Cross-substrate META findings

### META finding #1: CONSOLIDATE-OP-1 META-LAYER consolidation EXTINCTS recurrence bug class STRUCTURALLY

**Class**: Pre-CONSOLIDATE, A=DreamerV3 + F=Z8 both required substrate-local FIX-WAVE patches (commits e1b101888 + 4684dbbab) for the SAME PixelShuffle channel-LAST drift bug — F=Z8 inherited A's pre-fix code LINE-FOR-LINE per R1' empirical proof (max_abs=3.77 + 1.51).

**Post-CONSOLIDATE-OP-1**:
- A=DreamerV3 PixelShuffle max_abs vs PyTorch: 0.0 (canonical byte-stable; improved from R2's 0.0072 FIX-WAVE-R1 anchor)
- F=Z8 PixelShuffle max_abs vs PyTorch: 0.0 (canonical byte-stable)
- D=Z6 PixelShuffle max_abs vs PyTorch: 0.0 (canonical byte-stable; D=Z6 was the original sister-canonical seed)

**Future Path 3 candidates** (H/I/J/K queued, plus future L/M/N/O) inherit byte-stable PixelShuffle by importing canonical `pr95_hnerv_mlx` rather than re-implementing. The recurrence bug class is STRUCTURALLY EXTINCTED per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable.

**Status**: positive META; CONSOLIDATE-OP-1 validates META-LAYER consolidation as canonical structural protection.

### META finding #2: Wave #1 posterior emission GAP CLOSURE for B'+D+G canonical equation registration

**Class**: R2 META #3 op-routable identified B' + D + G as "sister-substrate-wide gap" — no declared canonical equation IDs.

**Post-Wave #1**:
- B'=Z7-Mamba-2-v2: CANONICAL_EQUATION_IDS=[1 registered]
- D=Z6: CANONICAL_EQUATION_IDS=['ego_motion_concentration_prior_v1'] (REGISTERED)
- G=NIRVANA: CANONICAL_EQUATION_IDS=[1 registered]

**ALL 7 reviewed substrates** now declare CANONICAL_EQUATION_IDS at L0 (cathedral-consumer-queryable per Catalog #335 + Catalog #344 sister discipline). The R2 META #3 op-routable is GAP-CLOSED across all 7 substrates.

**Status**: positive META; Wave #1 wire-in resolved 3 sister-substrate-wide gaps simultaneously.

### META finding #3: Wave #1 wire-in introduced 1 IMPLEMENTATION-LEVEL test regression at G=NIRVANA

**Class**: G=NIRVANA is the ONLY substrate with a STRICT-equality test (`assert set(mod.__all__) == expected`) for the narrow-public-API contract per Catalog #335. Wave #1 added 4 NEW canonical exports to `__all__` but did NOT update the strict-equality test in `test_basic.py:42-59`.

**Empirical evidence**:
- 55/56 G=NIRVANA test suite PASS
- 1/56 FAIL at `test_module_exposes_canonical_public_api`
- The 4 extra items in `__all__`: `SUBSTRATE_ID`, `ARCHITECTURE_CLASS`, `CANONICAL_EQUATION_IDS`, `emit_landing_posterior_anchor`

**Per Catalog #307 paradigm-vs-implementation classification**: IMPLEMENTATION-LEVEL (1-line test update); paradigm INTACT (narrow-public-API contract still satisfied; Wave #1 additions ARE canonical per Catalog #335).

**Cross-substrate sister scan** (verified empirically 2026-05-26T09:11Z): A/B'/C'/D/E/F do NOT have `test_module_exposes_canonical_public_api` analog (verified via `pytest -k "canonical_public_api or canonical_api or test_module_exposes"` returns 0 selected). G is the ONLY substrate affected by this Wave #1 oversight.

**Status**: 1 R3 FINDING (G-OP1); FIX-WAVE-R3 required.

### META finding #4: G=NIRVANA Axis 3 sister-canonical numpy reference PRESERVED via META-LAYER consumption pattern

**Class**: CONSOLIDATE-OP-2 (LANDED in CONSOLIDATE-OP-1 wave per landing memo §STEP 3) extracted G's 7 canonical numpy primitives to `tac.local_acceleration.pr95_hnerv_numpy_reference`. G's `numpy_reference.py` now RE-EXPORTS the 7 primitives from canonical; `cascade_reconstruct` (substrate-specific composition) retained locally.

**Empirical post-extraction verification** (per CONSOLIDATE-OP-1 landing memo §STEP 4):
- 16/16 canonical numpy reference tests PASS
- 16/16 canonical MLX primitive tests PASS
- 33/33 canonical posterior emission helper tests PASS
- Sister-substrate back-compat preserved (existing test imports `from tac.substrates.nirvana_cascading_nerv.numpy_reference import ...` continue to work)

**G=NIRVANA REMAINS canonical Axis 3 sister-canonical reference** — the seed that produced the canonical helper; now consumes it via META-LAYER consumption pattern. Sister-canonical status STRUCTURALLY PRESERVED (NOT degraded).

**Status**: positive META; CONSOLIDATE-OP-2 validates META-LAYER consolidation pattern for numpy reference surface (sister of MLX surface in META finding #1).

### META finding #5: 3-axis methodology empirically IMPROVES post-CONSOLIDATE + post-Wave-#1 state

**Class**: Per-axis verdict re-measurement post-CONSOLIDATE + Wave-#1:

| Substrate | Axis 1 (rigor) | Axis 2 (MLX drift) | Axis 3 (numpy port) |
|---|---|---|---|
| A=DreamerV3 | CLEAN (2 eqs REGISTERED) | CLEAN (max_abs IMPROVED from 0.0072 → 0.0 canonical byte-stable) | N/A by structural posture |
| B'=Z7-Mamba-2-v2 | CLEAN (1 eq REGISTERED via Wave #1 GAP CLOSURE) | N/A at L0 scope | N/A by construction |
| C'=NSCS06-v8 | CLEAN (Catalog #359 IN_DOMAIN preserved via Wave #1 extras) | N/A by construction (non-MLX) | HARD-EARNED-NATIVE |
| D=Z6 | CLEAN (1 eq REGISTERED via Wave #1 GAP CLOSURE) | CLEAN (max_abs = 0.0 canonical byte-stable) | N/A by structural posture |
| E=BoostNeRV | CLEAN (FIX-WAVE-R1 eq correction preserved) | N/A by construction (PyTorch-only) | N/A by construction |
| F=Z8 | CLEAN (5 eqs REGISTERED — largest citation set) | CLEAN (max_abs = 0.0 canonical byte-stable; IMPROVED from R2 anchor) | N/A by structural posture |
| G=NIRVANA | CLEAN architecture; **1 test-side finding** | N/A at L0 (no MLX renderer) | CLEAN (META-LAYER consumption preserves sister-canonical status) |

**Axis 2 max_abs IMPROVED across A+D+F** from substrate-local FIX-WAVE anchors to canonical byte-stable 0.0. **Axis 1 canonical equation registration GAP CLOSED for 3 substrates** (B'+D+G via Wave #1). **Axis 3 sister-canonical preserved** for G via META-LAYER consumption.

**Status**: positive META; CONSOLIDATE + Wave-#1 net positive across 3 axes (1 test-side regression for G is the only negative).

### META finding #6: PR95-export-boundary-drift bug class STRUCTURALLY EXTINCTED at META layer

**Class**: Per R1/R1'/R2 META observations, the canonical bug class was:
- MLX trainer optimized against MLX_buggy_decoder(weights) → MLX_frames
- PyTorch inflate used CORRECT canonical primitives → MLX-trained-PyTorch-inflated model did NOT match MLX trainer convergence frames
- L0 smoke trainer "loss decreased" did NOT reveal the bug
- Bug surfaced structurally at Catalog #1265 contest-equivalence gate

**Post-CONSOLIDATE-OP-1**:
- All 3 MLX-first substrates (A+D+F) delegate to canonical `pr95_hnerv_mlx` PixelShuffle + bilinear
- Canonical helper produces byte-stable PixelShuffle (0.0 max_abs) + sub-fp32-noise-floor bilinear (1.37e-06 max_abs for general-form)
- Future Path 3 candidates inherit canonical helpers structurally

**Sister CLAUDE.md anchor**: "HNeRV / leaderboard-implementation parity discipline" L9 (Runtime closure: same-runtime source replay required) + "Bugs must be permanently fixed AND self-protected against" non-negotiable.

**Status**: positive META; bug class EXTINCTED at META layer.

### META finding #7: Cargo-cult-first methodology (B' + C') validated across R1' + R2 + R3 — operator directive #2 compliance pays off

**Class**: B' Phase 1 audit (8 NEW CC + 2 NEW HARD-EARNED-PARTIAL) + Phase 3 implementation (16-layer canonical-vs-unique decision table); C' Phase 1 audit (4 CARGO-CULTED-CRITICAL identified) + Phase 3 implementation (all 4 unwinds + cargo-cult #5 EMPIRICALLY CONFIRMED via FAIL_AT_CLASS_1 verdict at L0).

Both substrates reached 3/3 SEAL at R3 via back-to-back R1' CLEAN + R2 CLEAN + R3 CLEAN. The methodology empirically pays off in review quality.

**Status**: positive META; reaffirmed R2 META #7.

---

## FIX-WAVE-R3-COMBINED op-routable queue

**Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable**: every adversarial-review finding MUST be addressed with TWO landings: (a) the fix + (b) a STRICT preflight check.

### G-OP1 (Severity: IMPLEMENTATION-LEVEL; 1-line test fix; NO new STRICT gate)

**File**: `src/tac/substrates/nirvana_cascading_nerv/tests/test_basic.py:46-58`

**Fix**: Expand the `expected` set to include the 4 Wave #1 canonical exports:

```python
expected = {
    "ARCHIVE_GRAMMAR_FIELDS",
    "ARCHIVE_MAGIC",
    "ARCHIVE_VERSION",
    "DEFAULT_BASE_H",
    "DEFAULT_BASE_W",
    "DEFAULT_NUM_LEVELS",
    "DEFAULT_PER_PAIR_LATENT_DIM",
    "NIRVANA1_HEADER_FMT",
    "NIRVANA1_HEADER_LEN",
    "NirvanaCascadingNervConfig",
    "SISTER_SUBSTRATES",
    # WAVE-1 canonical posterior emission wire-in additions (2026-05-26)
    "SUBSTRATE_ID",
    "ARCHITECTURE_CLASS",
    "CANONICAL_EQUATION_IDS",
    "emit_landing_posterior_anchor",
}
```

**Why no new STRICT gate**: G=NIRVANA is the ONLY substrate with this strict-equality test pattern (verified empirically). The existing test IS the structural protection per Catalog #335 narrow-public-API contract. A new STRICT preflight gate would be a quota-brake violation per Catalog #299 (we are at #361; quota brake at #400). The 1-line fix preserves the canonical narrow-public-API contract; no META-meta gate needed because:

1. The narrow-public-API contract IS the canonical Catalog #335 protection (already in place);
2. Sister substrates do NOT have this strict-equality pattern (G is unique);
3. Future Wave #2/#3 wire-ins to G WILL need to update this test — but that is the canonical sister-coherence discipline per Catalog #230 ownership map, not a new bug class needing META protection.

**Acceptance**: After FIX-WAVE-R3 lands the 1-line fix + a new commit advances HEAD, run `.venv/bin/python -m pytest src/tac/substrates/nirvana_cascading_nerv/tests/test_basic.py::test_module_exposes_canonical_public_api -v` → expect PASS.

**Operator-routable**: dedicated FIX-WAVE-R3 successor subagent OR sister cleanup subagent (the 1-line fix is small enough to bundle with any G-related work).

---

## R4 readiness verdict

**R4 UNBLOCKED for A+E+F** (each at 2/3; R4 CLEAN → 3/3 SEAL).
**R4 UNBLOCKED for G** AFTER FIX-WAVE-R3 lands (G resets to 0/3 then needs R4+R5+R6 CLEAN).
**R4 NOT NEEDED for D+B'+C'** — already at 3/3 SEAL.

### Recommended R4 scope

- **Batch 1 (3-substrate R4 fast-track to SEAL)**: A=DreamerV3 + E=BoostNeRV + F=Z8 at 2/3 → 3/3 SEAL IF CLEAN
- **Batch 2 (1-substrate G=NIRVANA post-FIX-WAVE-R3)**: G=NIRVANA at 0/3 → 1/3 (then needs R5 + R6 for SEAL)

**R4 wall-clock estimate**: ~2-3h (smaller than R3 because no CONSOLIDATE/Wave-#1 verification needed; just empirical re-verification of test suites + parity measurements).

### Adversarial perspectives rotation for R4

Per CLAUDE.md "Recursive adversarial review protocol — close paths" item 1 (council member rotation), R4 should foreground different inner council voices than R3:
- **R4 focal axis Carmack + Hotz** rotation: focal on the META-LAYER consolidation benefit — verify A+F's improved max_abs=0.0 anchor preserved across R4 dispatch + verify F=Z8's 5-canonical-equation citation set actually composes per the canonical Pareto-polytope intersection per Dykstra
- **R4 focal axis Daubechies + MacKay** rotation: focal on the canonical-quadruple binding (F=Z8 5-eq citation) per Catalog #312 sister gate
- **R4 assumption-challenge axis per protocol item 8**: focal candidate question: "Is the META-LAYER consolidation pattern (CONSOLIDATE-OP-1 canonical helper + delegation) STRUCTURALLY SUFFICIENT to prevent ALL future Path 3 PixelShuffle drift recurrences, OR does the pattern require an ADDITIONAL META-meta gate enforcing canonical-helper-import per Catalog #176 sister discipline?"

---

## Counter state per CLAUDE.md "Recursive adversarial review protocol — close paths"

- **Before R3-COMBINED**: A=1/3, B'=2/3, C'=2/3, D=2/3, E=1/3, F=1/3, G=1/3
- **Per-landing post-R3-COMBINED**:
  - 6 CLEAN (counter advances +1 each per protocol item 4): A, B', C', D, E, F
  - 1 NOT CLEAN (counter resets to 0 per protocol item 3): G
- **Aggregate post-R3-COMBINED**: A=2/3, B'=**3/3 SEAL**, C'=**3/3 SEAL**, D=**3/3 SEAL**, E=2/3, F=2/3, G=0/3 RESET
- **Post-R4-CLEAN-on-A+E+F + FIX-WAVE-R3-G + R4-G-CLEAN**: A=**3/3 SEAL**, E=**3/3 SEAL**, F=**3/3 SEAL**, G=1/3
- **Post-R5-G-CLEAN + R6-G-CLEAN**: G=**3/3 SEAL** (assuming no further regressions)
- **Operator-declared SEAL (D-1, conservative)**: NOT applicable here (counter-advance path is straightforward; FIX-WAVE-R3 fix is mechanical 1-line)

---

## Discipline applied

- **Catalog #229 PV**: 7 per-substrate R2 memos + R2 aggregate + CONSOLIDATE-OP-1 + Wave #1 posterior emission landing memos + 7 substrate test suites empirically re-verified (375/376 PASS; 1 FAIL = G-OP1 finding) + canonical equation registry empirically queried (42 equations REGISTERED; 8 cited REGISTERED) + canonical helper functions imported + empirical max_abs measurements (PixelShuffle 0.0 / bilinear ≤ 1.37e-06) before any review claim
- **Catalog #110/#113 APPEND-ONLY**: 8 NEW memos (7 per-substrate + 1 aggregate); R2 aggregate + per-substrate R2/R1/R1' memos + FIX-WAVE memos + landing memos NEVER mutated
- **Catalog #117/#157/#174/#235/#289**: commit forthcoming via canonical `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256`
- **Catalog #119**: Co-Authored-By Claude trailer
- **Catalog #287**: every finding closure carries `[empirical:<measurement>]` evidence-tag; no placeholder rationales; G-OP1 finding closure documented as IMPLEMENTATION-LEVEL per Catalog #307
- **Catalog #208**: docs/local-paths — only relative paths cited
- **Catalog #292**: per-axis council member operating-within assumption surfaced explicitly in frontmatter (all 7 per-substrate memos + this aggregate)
- **Catalog #300 v2**: full frontmatter on all 8 memos (tier T2 per-substrate; T3 aggregate; attendees include canonical 4-co-lead structure Shannon+Dykstra+Rudin+Daubechies per 2026-05-19 amendment + sister inner council voices; mission_contribution `frontier_breaking` for the aggregate verdict reflecting 3 substrates reach SEAL; horizon_class frontier_pursuit)
- **Catalog #346**: canonical council roster `validate_council_dispatch_roster` returns complete=True for T3 aggregate (15 attendees; ≥12-of-20 grand council quorum honored)
- **Catalog #340**: sister-checkpoint guard PROCEED verified at start of R3; only L2-INFRA-BUILD + R1'' for H+I+J+K + SISTER-#1265-GATE-Z6PCWM1 in-flight, none touch the 7 R3 review memos per Catalog #230 ownership map
- **Catalog #206**: checkpoint discipline (4 checkpoints emitted; step 1 / 2 / 3 / 4-complete)
- **Catalog #126**: lane `lane_path_3_recursive_adversarial_review_r3_combined_3_axis_landings_a_b_c_d_e_f_g_20260526` pre-registered
- **Catalog #307**: G-OP1 classified as IMPLEMENTATION-LEVEL fix (paradigm INTACT; 1-line test update preserves Catalog #335 narrow-public-API contract)
- **Catalog #299**: NO new STRICT preflight gate proposed (quota brake at #400; current count #361; the 1-line fix preserves canonical Catalog #335 contract without new META gate)
- **Catalog #294 9-dim checklist** (1-9): UNIQUENESS (3rd recursive review aggregate for Path 3; distinct from R1 + R1' + R2-COMBINED by covering post-CONSOLIDATE + post-Wave-#1 state); BEAUTY + ELEGANCE (8 memos ≤700 lines each); DISTINCTNESS (NOT a sister review; AGGREGATE synthesis with cross-substrate META + CONSOLIDATE/Wave-#1 effect tracking); RIGOR (PV + empirical re-measurement + canonical equation verification + sister-canonical cross-comparison + 376-test re-verification + canonical helper byte-stable verification); OPTIMIZATION PER TECHNIQUE (per-axis council members rotate per protocol); STACK-OF-STACKS-COMPOSABILITY (aggregate composes 7 per-substrate verdicts + cross-references R1+R1'+R2 aggregates + CONSOLIDATE-OP-1 + Wave #1); DETERMINISTIC REPRODUCIBILITY (reproducer commands documented per Axis 2 in per-substrate memos); EXTREME OPTIMIZATION + PERFORMANCE (R3-COMBINED review takes ~3h wall-clock vs paid-dispatch cost $0); OPTIMAL MINIMAL CONTEST SCORE (R3 is QUALITY GATE; 3 SEAL substrates unlock paid CUDA dispatch path)
- **CLAUDE.md "Recursive adversarial review protocol — close paths"** items 1-8: this aggregate review IS R3-COMBINED; 6 substrates advance counter +1; 1 substrate (G) resets to 0; item 8 (NEW assumption-challenge axis) satisfied via per-substrate Assumption-Adversary verdicts + aggregate-level 4-row Assumption-Adversary verdict
- **CLAUDE.md "Council conduct" amendment 2026-05-19 4-co-lead structure**: T3 aggregate roster includes all 4 co-leads per Catalog #346 requirement
- **CLAUDE.md "Executing actions with care"**: review-only (NO code modifications); FIX-WAVE-R3 + canonical sister subagents are canonical owners of any source modifications
- **CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"**: D/B'/C' 3/3 SEAL unlocks paid CUDA dispatch authorization at the protocol level — operator-routable next is paired CUDA + contest-CPU on 1:1 contest-compliant hardware per the non-negotiable

---

## Cross-references

- **R3-COMBINED per-substrate review memos**:
  - A=DreamerV3: `.omx/research/path_3_a_recursive_adversarial_review_r3_combined_3_axis_20260526.md`
  - B'=Z7-Mamba-2-v2: `.omx/research/path_3_b_recursive_adversarial_review_r3_combined_3_axis_20260526.md`
  - C'=NSCS06-v8: `.omx/research/path_3_c_recursive_adversarial_review_r3_combined_3_axis_20260526.md`
  - D=Z6: `.omx/research/path_3_d_recursive_adversarial_review_r3_combined_3_axis_20260526.md`
  - E=BoostNeRV: `.omx/research/path_3_e_recursive_adversarial_review_r3_combined_3_axis_20260526.md`
  - F=Z8: `.omx/research/path_3_f_recursive_adversarial_review_r3_combined_3_axis_20260526.md`
  - G=NIRVANA: `.omx/research/path_3_g_recursive_adversarial_review_r3_combined_3_axis_20260526.md`
- **Predecessor aggregate reviews**:
  - R2-COMBINED aggregate: `.omx/research/path_3_recursive_adversarial_review_r2_combined_aggregate_3_axis_landings_a_b_c_d_e_f_g_20260526.md`
  - R1 aggregate (A+D+E): `.omx/research/path_3_recursive_adversarial_review_r1_aggregate_3_axis_landings_a_d_e_20260526.md`
  - R1' aggregate (B'+C'+F+G): `.omx/research/path_3_recursive_adversarial_review_r1_prime_aggregate_3_axis_landings_b_c_f_g_20260526.md`
- **Predecessor FIX-WAVE landings**:
  - FIX-WAVE-R1 (closed A+E R1 findings): `.omx/research/path_3_fix_wave_r1_close_findings_landed_20260526.md` (commit `e1b101888`)
  - FIX-WAVE-R1' (closed F+G R1' findings): `.omx/research/path_3_fix_wave_r1_prime_close_findings_landed_20260526.md` (commit `4684dbbab`)
- **CONSOLIDATE + Wave #1 landings (post-R2; THIS R3 reviewed post-state)**:
  - CONSOLIDATE-OP-1: `.omx/research/path_3_consolidate_op_1_canonical_mlx_primitives_extraction_landed_20260526.md` (commit `caf29acdb`)
  - Wave #1 posterior emission canonical wire-in: `.omx/research/path_3_wave_1_posterior_emission_canonical_wire_in_landed_20260526.md` (commits `f6b432be1` + `3d103dafd`)
  - Cascade doctrine: `.omx/research/path_3_canonical_substrate_development_cascade_doctrine_20260526.md` (commit `fb270e9b6`)
  - MLX-first doctrine: `.omx/research/mlx_first_everywhere_canonical_doctrine_20260526.md` (commit `4107bbf8d`)
- **Canonical references**:
  - Canonical PR95 helper: `tac.local_acceleration.pr95_hnerv_mlx::pixel_shuffle_2x_nhwc` + `bilinear_resize2x_align_corners_false_nhwc` + `bilinear_resize_nhwc`
  - Canonical numpy reference: `tac.local_acceleration.pr95_hnerv_numpy_reference` (7 primitives)
  - Canonical posterior emission helper: `tac.substrates._shared.posterior_emission_helper`
  - Canonical equation registry: `tac.canonical_equations.query_equations()` (42 equations REGISTERED as of 2026-05-26)
  - Canonical Catalog #1265 anchor: `.omx/research/mlx_candidate_contest_equivalence_gate_landed_20260526.md`
- **In-flight sister subagents (NOT in this R3 scope; verified disjoint per Catalog #230 ownership map)**:
  - L2-INFRA-BUILD pid `a72cbf33d2ae2768c` (canonical long-training infrastructure)
  - R1'' for H+I+J+K (sister scope; new H/I/J/K landings)
  - SISTER-#1265-GATE-Z6PCWM1 (D=Z6 specific Catalog #1265 contest-equivalence gate)
- **Lane**: `lane_path_3_recursive_adversarial_review_r3_combined_3_axis_landings_a_b_c_d_e_f_g_20260526` L1 (impl_complete + memory_entry)

---

## Final aggregate verdict

**PROCEED — R3-COMBINED 6 of 7 SUBSTRATES CLEAN; 3 REACH 3/3 SEAL = PAID CUDA DISPATCH AUTHORIZED**

**Substrates at 3/3 SEAL (paid CUDA dispatch authorized)**:

1. **D=Z6 predictive-coding** — most-advanced; first Path 3 contest-CUDA score candidate
2. **B'=Z7-Mamba-2-v2 fresh substrate** — modulo Phase 3 §6 MPS-Win precondition
3. **C'=NSCS06-v8 chroma_lut** — modulo L1 promotion blocker (cls_stream consumption)

**Substrate needing FIX-WAVE-R3 + 3 more clean rounds**: G=NIRVANA (1-line test update at `test_basic.py:46-58`)

**Substrates at 2/3 needing R4**: A=DreamerV3, E=BoostNeRV, F=Z8

### Notable wins from R3-COMBINED

1. **3 substrates reach 3/3 SEAL** — D=Z6, B'=Z7-Mamba-2-v2, C'=NSCS06-v8 chroma_lut all authorized for paid CUDA dispatch
2. **CONSOLIDATE-OP-1 META-LAYER consolidation empirically IMPROVES Axis 2 drift** — A+D+F all post-CONSOLIDATE max_abs PixelShuffle = 0.0 (canonical byte-stable; improved from R2's FIX-WAVE anchors)
3. **Wave #1 posterior emission GAP CLOSURE** — all 7 substrates now declare CANONICAL_EQUATION_IDS at L0 (R2 META op-routable for B'+D+G resolved simultaneously)
4. **PYTORCH-EXPORT-BOUNDARY-DRIFT bug class STRUCTURALLY EXTINCTED** at META layer — future Path 3 candidates inherit byte-stable PixelShuffle by importing canonical
5. **Cargo-cult-first methodology** empirically pays off — B' + C' reach 3/3 SEAL via back-to-back R1' + R2 + R3 CLEAN
6. **G=NIRVANA Axis 3 sister-canonical PRESERVED** via META-LAYER consumption pattern (CONSOLIDATE-OP-2)
7. **FIX-WAVE-R3 op-routable count = 1** (G-OP1 only; 1-line test fix)

### Notable concerns from R3-COMBINED

1. **G=NIRVANA counter RESETS** — Wave #1 wire-in did NOT update G's strict-equality test. This is IMPLEMENTATION-LEVEL (1-line fix) but the protocol-binding-decision per CLAUDE.md item 3 resets counter to 0.
2. **T3 cadence APPROACHING_LIMIT** — R3-COMBINED is the 4th T3 this week (over budget per Catalog #300 mission_alignment v2 frontmatter discipline). Per CLAUDE.md "Mission alignment" operational consequence 1, operator-frontier-override per the R3-COMBINED charter overrides; recommend pause-and-consolidate before R4.
3. **L2-INFRA-BUILD + SISTER-#1265-GATE-Z6PCWM1 in-flight** — D=Z6 3/3 SEAL unlocks operator-routable next steps that overlap with these sister subagents; coordination required per Catalog #230 ownership map.

### Mission alignment per Catalog #300

`frontier_breaking` — the R3-COMBINED review UNLOCKS 3 substrates for paid CUDA dispatch authorization at the protocol level. D=Z6 specifically is the MOST-ADVANCED substrate for first Path 3 contest-CUDA score per CLAUDE.md "Frontier target — NON-NEGOTIABLE, HIGHEST EMPHASIS" + "Submission auth eval — BOTH CPU AND CUDA". The 1 R3 finding (G-OP1) is mechanical 1-line fix; the aggregate effect of R3 is frontier-breaking unlock.

Estimated R4 wall-clock for A+E+F+G batch: ~2-3h (4 substrates; smaller than R3 because no CONSOLIDATE/Wave-#1 verification needed). Estimated G FIX-WAVE-R3: ~10 min (1-line fix + commit + test re-verification).

**The Path 3 substrate-class-shift pursuit is now operator-routable for first contest-CUDA dispatch via D=Z6 (or B'/C' after their respective preconditions).**
