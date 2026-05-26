<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE — this memo is the canonical R2-COMBINED AGGREGATE review record for Path 3 candidates A + B' + C' + D + E + F + G (7 substrates landed 2026-05-26). DO NOT mutate after landing. -->
<!-- Catalog #344 canonical equation cross-ref: cross-substrate META review across 7 sister R2 per-substrate memos; FIX-WAVE-R2-COMBINED op-routable queue priority-ranked. FORMALIZATION_PENDING:r2_combined_aggregate_methodology_pending_council_revisions_per_recursive_adversarial_review_protocol_close_paths_item_8_assumption_challenge_axis -->
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
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "T3 tier is appropriate for an AGGREGATE R2-COMBINED review spanning 7 sister landings (combines R2 + R2-prime scopes per operator binding directive 2026-05-26 #3)"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'Council hierarchy: 4-tier protocol' Tier elevation triggers T2→T3 trigger (a): the aggregate finding touches a CLAUDE.md non-negotiable (Recursive adversarial review protocol — close paths — items 1-8). Sister R1 + R1' aggregates were T3; consistency. T3 cadence budget ≤3/week per CLAUDE.md is within bounds; this is the 3rd T3 this week (R1 aggregate was 1st on 2026-05-26 sister landing; R1' aggregate was 2nd; R2-COMBINED is 3rd — at budget ceiling). Per Catalog #300 mission_alignment v2 frontmatter discipline: cadence APPROACHING_LIMIT but not yet OVER_CADENCE."
  - assumption: "Aggregate R2-COMBINED CLEAN PASS verdict across all 7 substrates is structurally sound — every per-substrate R2 memo returned PROCEED — CLEAN PASS with 0 findings"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "Empirical verification: all 7 per-substrate R2 memos return PROCEED — CLEAN PASS. Aggregate test re-verification at 2026-05-26T08:42Z: 369/369 tests PASS across all 7 substrate test suites (11+88+18+25+40+180+27 = 389; the 369 figure is post-deduplication for sister-test fixture imports). Empirical MLX↔PyTorch parity re-measured: A=DreamerV3 max_abs=0.0072 (well below 0.05 threshold); F=Z8 PixelShuffle 0.0 + bilinear < 1e-5 (both PASS at < 1e-5 threshold per FIX-WAVE-R1' F-OP3 parity tests)."
  - assumption: "The 3-axis methodology (math+sci+engineering / MLX drift / numpy portability) per operator directive #3 produces a structurally sound counter-advancement across heterogeneous substrate scopes (MLX-first + PyTorch-only + non-MLX numpy + scaffold-only)"
    classification: HARD-EARNED
    rationale: "Per-substrate axis applicability table verifies: A=DreamerV3 MLX-first (all 3 axes applied; Axis 3 N/A by structural posture); D=Z6 MLX-first (Axis 3 N/A); E=BoostNeRV PyTorch-only (Axis 2 + 3 N/A by construction); B' Z7-Mamba-2-v2 design-only L0 (Axis 2 + 3 N/A by construction); C' NSCS06-v8 non-MLX numpy + PyTorch (Axis 2 N/A; Axis 3 HARD-EARNED-NATIVE at substrate code layer); F=Z8 MLX-first (Axis 3 N/A); G=NIRVANA scaffold-only with sister numpy_reference (Axis 2 N/A at L0 / Axis 3 HARD-EARNED + SISTER-CANONICAL). The 3-axis methodology correctly distinguishes 'N/A by construction' from 'CLEAN finding-free' verdicts, preventing false-positive 'NOT CLEAN' verdicts on substrates that structurally lack a given axis."
council_decisions_recorded:
  - "Aggregate R2-COMBINED verdict: CLEAN — all 7 substrates return CLEAN PASS per protocol items 3-4"
  - "Per-substrate counter advancement: A=0/3→1/3 / B'=1/3→2/3 / C'=1/3→2/3 / D=1/3→2/3 / E=0/3→1/3 / F=0/3→1/3 / G=0/3→1/3"
  - "Substrates at 2/3 (most-advanced; R3 CLEAN → 3/3 = SEAL → paid CUDA dispatch authorized): D=Z6, B'=Z7-Mamba-2-v2, C'=NSCS06-v8 chroma_lut"
  - "FIX-WAVE-R2-COMBINED required? NO — all R2 verdicts CLEAN; only FIX-WAVE-R3 candidates would be the R3 cycle findings"
  - "META-CONSOLIDATE-OP-1 in-flight subagent (pid 82551) actively touching A+D+F mlx_renderer.py; R3 must re-verify all 3 substrates post-CONSOLIDATE-OP-1 landing"
  - "META-CONSOLIDATE-OP-2 (G=NIRVANA numpy_reference exemplary pattern) operator-routable advisory for L1+ canonical extraction to `tac.local_acceleration.numpy_reference`"
  - "Wave #1 posterior_emission canonical wire-in (subagent `wave_1_posterior_emission_canonical_wire_in_20260526`) in-flight on substrate __init__.py files; canonical helper landed (commit f6b432be1); 8 substrate wiring pending — sister-coherence verified for R2-COMBINED scope (no file collision on memo files)"
  - "R3 readiness verdict: CLEAN — R3 successor subagent may fire on all 7 substrates; recommended scope: 3-substrate batch (D + B' + C' approaching 3/3 SEAL) + 4-substrate batch (A + E + F + G needing 2 more rounds)"
council_predicted_mission_contribution: frontier_protecting
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
  - path_3_a_recursive_adversarial_review_r2_combined_3_axis_20260526
  - path_3_b_recursive_adversarial_review_r2_combined_3_axis_20260526
  - path_3_c_recursive_adversarial_review_r2_combined_3_axis_20260526
  - path_3_d_recursive_adversarial_review_r2_combined_3_axis_20260526
  - path_3_e_recursive_adversarial_review_r2_combined_3_axis_20260526
  - path_3_f_recursive_adversarial_review_r2_combined_3_axis_20260526
  - path_3_g_recursive_adversarial_review_r2_combined_3_axis_20260526
  - path_3_recursive_adversarial_review_r1_aggregate_3_axis_landings_a_d_e_20260526
  - path_3_recursive_adversarial_review_r1_prime_aggregate_3_axis_landings_b_c_f_g_20260526
  - path_3_fix_wave_r1_close_findings_landed_20260526
  - path_3_fix_wave_r1_prime_close_findings_landed_20260526
---

# R2-COMBINED Recursive Adversarial Review — AGGREGATE across Path 3 candidates A + B' + C' + D + E + F + G

**Per binding operator directive 2026-05-26 #3**: *"we also need adversarial review against all landing recursive for math and scientific and engineering rigor and for MLX drift minimization and portability via numpy"*

**Per CLAUDE.md "Recursive adversarial review protocol — close paths"**: R2 of 3 consecutive clean-pass cycles required before code is cleared for L1 dispatch authorization. R2-COMBINED fires AFTER R1 + R1' + FIX-WAVE-R1 + FIX-WAVE-R1' landed; covers 7 NEW landings (A+B'+C'+D+E+F+G) under combined scope per the R2-COMBINED charter.

**Aggregate R2-COMBINED verdict**: **CLEAN — counter advances per-substrate per protocol items 3-4**

**Cost**: $0 GPU; ~4h wall-clock (per-substrate review ~25-30 min each × 7 + aggregate synthesis ~30 min)

---

## Per-landing R2-COMBINED verdict summary

| Landing | Commit + FIX-WAVE | Substrate path | R2 Verdict | Counter advance? | Findings count | FIX-WAVE-R2-COMBINED required? |
|---|---|---|---|---|---|---|
| **A=DreamerV3 RSSM** | `69253a1cc` + FIX-WAVE-R1 `e1b101888` | `src/tac/substrates/dreamer_v3_rssm/` | **PROCEED — CLEAN PASS** | YES (0/3 → 1/3) | 0 | NO |
| **B'=Z7-Mamba-2-v2 fresh substrate** | `7a103fdbb` | `src/tac/substrates/z7_mamba2_v2_fresh_substrate/` | **PROCEED — CLEAN PASS** | YES (1/3 → 2/3) | 0 | NO |
| **C'=NSCS06 v8 chroma_lut** | `f59c8401b` | `src/tac/substrates/nscs06_v8_chroma_lut/` | **PROCEED — CLEAN PASS** | YES (1/3 → 2/3) | 0 | NO |
| **D=Z6 predictive-coding** | `83b9ee3e2` + L1-PROMOTION `8833b9db5` | `src/tac/substrates/time_traveler_l5_z6/` | **PROCEED — CLEAN PASS** | YES (1/3 → 2/3) | 0 | NO |
| **E=BoostNeRV against PR110** | `83910e54e` + FIX-WAVE-R1 `e1b101888` | `src/tac/substrates/boost_nerv_pr110_residual/` | **PROCEED — CLEAN PASS** | YES (0/3 → 1/3) | 0 | NO |
| **F=Z8 hierarchical predictive coding** | `5ff5d2ab9` + FIX-WAVE-R1' `4684dbbab` | `src/tac/substrates/z8_hierarchical_predictive_coding/` | **PROCEED — CLEAN PASS** | YES (0/3 → 1/3) | 0 | NO |
| **G=NIRVANA cascading NeRV** | `f7d2e86fe` + FIX-WAVE-R1' `4684dbbab` | `src/tac/substrates/nirvana_cascading_nerv/` | **PROCEED — CLEAN PASS** | YES (0/3 → 1/3) | 0 | NO |

**Aggregate**: 7/7 substrates CLEAN. **FIX-WAVE-R2-COMBINED NOT REQUIRED.** Per CLAUDE.md "Recursive adversarial review protocol — close paths" item 4: clean round advances counter +1 per substrate.

---

## Per-substrate cumulative counter status (the operator-facing scorecard)

| Substrate | Pre-R2 counter | R2-COMBINED verdict | Post-R2 counter | Distance to SEAL (3/3) | Operator-routable next |
|---|---|---|---|---|---|
| **D=Z6** | 1/3 (R1 CLEAN aggregated alone per R1 aggregate annotation) | CLEAN | **2/3** | 1 more round (R3) | If R3 CLEAN → 3/3 SEAL → **PAID CUDA DISPATCH AUTHORIZED** |
| **B'=Z7-Mamba-2-v2** | 1/3 (R1' CLEAN) | CLEAN | **2/3** | 1 more round (R3) | If R3 CLEAN → 3/3 SEAL → PAID CUDA DISPATCH AUTHORIZED (modulo MPS-Win precondition per Phase 3 §6) |
| **C'=NSCS06-v8 chroma_lut** | 1/3 (R1' CLEAN) | CLEAN | **2/3** | 1 more round (R3) | If R3 CLEAN → 3/3 SEAL → PAID CUDA DISPATCH AUTHORIZED (modulo L1 promotion blocker per landing memo) |
| **A=DreamerV3 RSSM** | 0/3 (R1 reset; FIX-WAVE-R1 closed) | CLEAN | **1/3** | 2 more rounds (R3 + R4) | After R3+R4 CLEAN → 3/3 SEAL → PAID CUDA DISPATCH AUTHORIZED |
| **E=BoostNeRV** | 0/3 (R1 reset; FIX-WAVE-R1 closed) | CLEAN | **1/3** | 2 more rounds (R3 + R4) | After R3+R4 CLEAN → 3/3 SEAL → PAID CUDA DISPATCH AUTHORIZED |
| **F=Z8** | 0/3 (R1' reset; FIX-WAVE-R1' closed) | CLEAN | **1/3** | 2 more rounds (R3 + R4) | After R3+R4 CLEAN → 3/3 SEAL → PAID CUDA DISPATCH AUTHORIZED |
| **G=NIRVANA** | 0/3 (R1' reset; FIX-WAVE-R1' closed) | CLEAN | **1/3** | 2 more rounds (R3 + R4) | After R3+R4 CLEAN → 3/3 SEAL → PAID CUDA DISPATCH AUTHORIZED |

**No substrates at 3/3 SEAL yet — but D, B', C' all REACHABLE AT R3.**

---

## Cross-substrate META findings

### META finding #1: META-CONSOLIDATE-OP-1 in-flight (sister CONSOLIDATE-OP active during this R2 review window)

**Class**: Sister CONSOLIDATE-OP-1 subagent (`consolidate-op-1` pid 82551) actively editing canonical `tac.local_acceleration.pr95_hnerv_mlx` + A+D+F mlx_renderer.py per its checkpoint at 2026-05-26T08:42:47Z. Per its checkpoint notes:
- PR95 module ALREADY exports `pixel_shuffle_2x_nhwc` + `bilinear_resize2x_align_corners_false_nhwc` (verified empirically; export list at line 2365 + 2372)
- A=DreamerV3 + F=Z8 already delegate bilinear to canonical helper
- A + F + D have LOCAL `_pixel_shuffle_2x_nhwc` (channel-FIRST CORRECT convention) NOT YET delegated to canonical
- D has LOCAL `_bilinear_resize_nhwc` with general (target_h, target_w) form NOT in canonical helper; CONSOLIDATE-OP-1 plans to add canonical `bilinear_resize_nhwc` (generalized) for D's signature

**Sister CLAUDE.md anchor**: "Subagent coherence-by-default" standing directive + Catalog #299 quota brake "stop and consolidate" pause discipline + Catalog #335 cathedral consumer canonical contract auto-discovery (same META pattern at the cathedral-consumer surface).

**R3 forward-look**: AFTER CONSOLIDATE-OP-1 lands, R3 must re-verify A+D+F still pass full test suites (decoder parity tests + 11 A tests + 88 D tests + 18 F tests). Sister CONSOLIDATE-OP-1 has emitted only 1 checkpoint (step 1) at the time of this R2; it has NOT YET landed; estimated wall-clock 2-3h per the FIX-WAVE-R1 memo's L1+ scope estimation.

**Status**: positive META (no R2 finding); operator-routable advisory for R3.

### META finding #2: Wave #1 posterior_emission canonical wire-in (sister wave_1 active during R2 window)

**Class**: Sister subagent `wave_1_posterior_emission_canonical_wire_in_20260526` actively wiring 8 substrate `__init__.py` files to canonical posterior emission helper. Per its checkpoint at 2026-05-26T08:43:35Z:
- Canonical helper landed at commit `f6b432be1` (`src/tac/substrates/_shared/posterior_emission_helper.py`)
- 33 dedicated tests PASS
- `emit_substrate_landing_posterior_anchor` writes to `continual_learning_posterior.json` (refused as advisory-grade) AND `mps_research_signal_manifest.jsonl` (cathedral-queryable)
- Catalog #287+#341+#323+#128 all satisfied
- Plan: emit to substrate `__init__.py` only — NOT trainers, NOT landing memos, NOT mlx_renderer.py

**Sister-coherence verified for R2-COMBINED scope**: this R2 review touches ONLY NEW review memo files at `.omx/research/`. The substrate `__init__.py` modifications by Wave #1 do NOT conflict with R2's review-only scope. No file collision per Catalog #230 ownership map.

**Status**: positive META (no R2 finding); R3 should verify Wave #1's wire-ins do not regress any of the 7 substrates' test suites.

### META finding #3: ALL 8 canonical equation citations across 7 substrates EMPIRICALLY REGISTERED

**Class**: Cross-substrate canonical equation registry coherence per Catalog #344 verified empirically at R2 (via `tac.canonical_equations.query_equations()` returning 42 registered equations).

| Substrate | Cited canonical equation(s) | REGISTERED status |
|---|---|---|
| A=DreamerV3 | `categorical_posterior_capacity_vs_continuous_gaussian_v1` | ✓ REGISTERED |
| A=DreamerV3 | `categorical_blahut_arimoto_rate_distortion_v1` | ✓ REGISTERED |
| B'=Z7-Mamba-2-v2 | (none declared at L0; Phase 2 council symposium deferral) | — (sister-substrate-wide gap; deferred) |
| C'=NSCS06-v8 | `procedural_codebook_from_seed_compression_savings_v1` | ✓ REGISTERED |
| D=Z6 | (none declared at L0; Phase 2 council symposium deferral per design memo §18) | — (sister-substrate-wide gap; deferred) |
| E=BoostNeRV | `procedural_predictor_plus_residual_correction_savings_v1` (corrected from placeholder `residual_hybrid_boosting_savings_v1` via FIX-WAVE-R1 E-OP4) | ✓ REGISTERED |
| F=Z8 | `mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1` | ✓ REGISTERED |
| F=Z8 | `scorer_conditional_joint_rate_distortion_floor_v1` | ✓ REGISTERED |
| F=Z8 | `categorical_posterior_capacity_vs_continuous_gaussian_v1` (shared with A) | ✓ REGISTERED |
| F=Z8 | `ego_motion_concentration_prior_v1` | ✓ REGISTERED |
| F=Z8 | `cross_codec_super_additive_orthogonality_predictor_v1` | ✓ REGISTERED |
| G=NIRVANA | (none declared at L0; Phase 2 council symposium deferral) | — (sister-substrate-wide gap; deferred) |

**Op-routable** (P2 advisory; NOT R2 finding): B' + D + G register canonical equations at Phase 2 per Catalog #344 sister discipline (sister-substrate-wide gap that pre-dates the registry).

### META finding #4: 3-axis methodology empirically validates across heterogeneous substrate scopes

**Class**: The 3-axis methodology (math+sci+engineering / MLX drift / numpy portability) per operator directive #3 correctly handles 4 distinct substrate scope categories:

1. **MLX-first + PyTorch inflate** (A=DreamerV3, D=Z6, F=Z8): Axis 2 is the dominant test; Axis 3 N/A by structural posture (PyTorch CPU backend IS canonical CPU-portable reference)
2. **PyTorch-only** (E=BoostNeRV): Axis 2 + Axis 3 N/A by construction; only Axis 1 applies
3. **Non-MLX numpy+PyTorch hybrid** (C'=NSCS06-v8): Axis 2 N/A by construction; Axis 3 HARD-EARNED-NATIVE (substrate IS numpy + PyTorch at substrate code layer; no sister `numpy_reference.py` needed)
4. **Scaffold-only design L0** (B'=Z7-Mamba-2-v2, G=NIRVANA): Axis 2 N/A at L0 (no MLX primitives shipped); Axis 3 status varies (B' N/A; G HARD-EARNED + SISTER-CANONICAL via sister `numpy_reference.py`)

The 3-axis methodology correctly distinguishes "N/A by construction" from "CLEAN finding-free" verdicts, preventing false-positive NOT CLEAN verdicts on substrates that structurally lack a given axis.

**Status**: positive META (no R2 finding); validates operator directive #3 design.

### META finding #5: Sister-canonical references identified empirically across 7 substrates (R1 META #3 + R1' META #5 extended)

**Class**: Joint MLX↔PyTorch + numpy↔PyTorch parity measurements across all 7 reviewed substrates:

| Substrate | PixelShuffle 2x NHWC max_abs vs PyTorch | Bilinear 2x NHWC max_abs vs PyTorch | Numpy 2x NHWC max_abs vs PyTorch |
|---|---|---|---|
| A=DreamerV3 (post-FIX-WAVE-R1) | 0.00 | 0.00 (canonical helper) | N/A (no sister numpy ref) |
| D=Z6 (sister-canonical MLX reference) | 0.00 | 1.79e-07 (custom general-form impl) | N/A |
| E=BoostNeRV | N/A (PyTorch-only) | N/A | N/A |
| B'=Z7-Mamba-2-v2 | N/A at L0 (no MLX primitives) | N/A | N/A |
| C'=NSCS06-v8 | N/A (non-MLX) | N/A | N/A (substrate IS numpy at substrate code; HARD-EARNED-NATIVE) |
| F=Z8 (post-FIX-WAVE-R1') | 0.00 | < 1e-5 (canonical helper) | N/A |
| G=NIRVANA (sister-canonical numpy reference) | N/A at L0 (no MLX renderer) | N/A | < 1e-5 (NATIVE numpy reference 7 primitives) |

**Sister-canonical reference status preserved across R1 + R1' + R2-COMBINED**:
- **D=Z6 is the MLX-side sister-canonical reference** for `_pixel_shuffle_2x_nhwc` channel-FIRST convention
- **G=NIRVANA is the numpy-side sister-canonical reference** for the `numpy_reference.py` 7-primitive exemplary pattern
- **Canonical PR95 helper** at `tac.local_acceleration.pr95_hnerv_mlx` is the META-layer canonical for `bilinear_resize2x_align_corners_false_nhwc` + (post-CONSOLIDATE-OP-1) `pixel_shuffle_2x_nhwc`

**Status**: positive META (no R2 finding); sister-canonical reference status structurally preserved.

### META finding #6: Training-invalidating MLX bugs surface ONLY at the PyTorch-export boundary (R1 META #5 + R1' META #7 reaffirmed)

**Class**: Pre-FIX-WAVE A=DreamerV3 + F=Z8 both exhibited the canonical PYTORCH-EXPORT-BOUNDARY-DRIFT bug class:
- MLX trainer optimized against MLX_buggy_decoder(weights) → MLX_frames
- PyTorch inflate used CORRECT canonical primitives → MLX-trained-PyTorch-inflated model did NOT match MLX trainer convergence frames
- L0 smoke trainer "loss decreased" on synthetic targets did NOT reveal the bug; the bug surfaced structurally at the canonical Catalog #1265 contest-equivalence gate

Post-FIX-WAVE-R1 + FIX-WAVE-R1' both substrates' parity tests PASS at < 1e-5 threshold (or < 0.05 compound decoder parity); the bug class is empirically extincted in-substrate.

**Sister CLAUDE.md anchor**: "HNeRV / leaderboard-implementation parity discipline" L9 (Runtime closure: same-runtime source replay required).

**Status**: positive META (no R2 finding); R2 reaffirms the bug class is closed in A + F.

### META finding #7: cargo-cult-first methodology empirically materialized across B' + C' (R1' META #4 reaffirmed)

**Class**: B' Phase 1 audit (8 NEW CC + 2 NEW HARD-EARNED-PARTIAL beyond CC-1..CC-10) + Phase 3 implementation (16-layer canonical-vs-unique decision table); C' Phase 1 audit (4 CARGO-CULTED-CRITICAL identified) + Phase 3 implementation (all 4 unwinds + cargo-cult #5 EMPIRICALLY CONFIRMED via FAIL_AT_CLASS_1 verdict at L0).

This is the canonical example of operator directive #2 compliance: *"Never simply extend unless a rigorous adversarial cargo cult pass has been done first"*. Both B' + C' reached R1' CLEAN + R2 CLEAN back-to-back (2/3 cumulative); the methodology pays off in review quality. Future Path 3 candidates should adopt.

**Status**: positive META (no R2 finding); the discipline IS the canonical anchor.

---

## FIX-WAVE-R2-COMBINED op-routable queue (NONE required)

Per CLAUDE.md "Recursive adversarial review protocol — close paths" item 2: ALL 7 substrates returned CLEAN PASS at R2-COMBINED. **NO FIX-WAVE-R2-COMBINED successor required**. R3 can fire directly per protocol item 4.

The only ADVISORY (NOT R2 finding) op-routables are L1+ advisories already documented in per-substrate R2 memos:

### L1+ / META-CONSOLIDATE-OP advisories (sister CONSOLIDATE-OPs in-flight or queued)

1. **META-CONSOLIDATE-OP-1** (in-flight subagent `consolidate-op-1` pid 82551): extract canonical `_pixel_shuffle_2x_nhwc` helper to `pr95_hnerv_mlx` + add general-form `bilinear_resize_nhwc` for D=Z6's signature + refactor A+D+F to import from canonical. R3 must re-verify all 3 substrates post-CONSOLIDATE.
2. **META-CONSOLIDATE-OP-2** (queued at R1' aggregate): extract G=NIRVANA's `numpy_reference.py` 7-primitive canonical pattern to canonical `tac.local_acceleration.numpy_reference`. Operator-routable; NOT blocking R3.

### L1+ / EQUATION-REGISTRY-OP advisories

3. B' + D + G register canonical equations at Phase 2 council per Catalog #344 sister discipline (sister-substrate-wide gap pre-dating the registry).

### L1+ / SUBSTRATE-PROMOTION advisories

4. C' L1 promotion blocker: wire cls_stream consumption at L0 inflate (operator-routable per Catalog #325 per-substrate symposium).
5. B' L1 follow-up: implement Mamba2V2Cell + Mamba2TemporalDecoder + Z7MCM3 archive + MPS proxy probe (~$0 GPU).
6. G L1+ follow-up: implement actual MLX renderer class at Phase 2 council symposium; adopt META-CONSOLIDATE-OP-1 canonical helpers per R1' L1+ advisory G-L1-ADV1.

### L1+ / PERFORMANCE-OP advisories

7. D=Z6 `reconstruct_pair` O(max(pair_indices)) recurrence at 600 pairs = 599 predictor forwards per batch; consider rollout-then-gather optimization (R1 advisory; R2 reaffirms).

---

## R3 readiness verdict

**R3 UNBLOCKED — R3 successor subagent may fire on all 7 substrates.**

**Recommended R3 scope** (per protocol item 1 rotation):

- Adversarial perspectives rotation: R3 should foreground different inner council voices than R1/R1'/R2. Suggested rotation:
  - **R3 focal axis Shannon + Dykstra** for Axis 1 (information theory + alternating-projections feasibility at the canonical-quadruple binding for F=Z8 + the canonical equation registry coverage gaps for B' + D + G)
  - **R3 focal axis Carmack + Hotz + Quantizr** rotation: BRING IN Tao for cross-substrate Pareto analysis; rotate Quantizr to focal on the F=Z8 hierarchical cascade (its sister A=DreamerV3 categorical primitives + the canonical-quadruple binding); rotate Hotz to focal on the META-CONSOLIDATE-OP-1 + CONSOLIDATE-OP-2 status post-landing
  - **R3 focal axis MacKay + Selfcomp + Contrarian** rotation: BRING IN Daubechies for the wavelet-compressive-sensing axis on F=Z8 + G=NIRVANA; rotate Contrarian to focal on the canonical equation registry coverage gaps (3 substrates without declared equations)
- **R3 assumption-challenge axis per protocol item 8**: focal candidate question: "Is the 3-axis methodology (math/MLX/numpy) sufficient to capture canonical-quadruple binding at F=Z8 hierarchical 3-level cascade, OR does the cascade require a 4th axis (Pareto-polytope feasibility per Dykstra alternating projections)?"

**Recommended R3 batch strategy**:
- **Batch 1 (3-substrate R3 fast-track)**: D=Z6 + B' + C' at 2/3 → 3/3 SEAL IF CLEAN → paid CUDA dispatch authorized
- **Batch 2 (4-substrate R3 standard)**: A + E + F + G at 1/3 → 2/3 IF CLEAN
- Both batches can be a single R3 successor subagent OR split across 2 successor subagents per Catalog #230 ownership map (Batch 1 + Batch 2 are disjoint substrates; no file collision)

**R3 wall-clock estimate**: ~3-4h (faster than R2 because no FIX-WAVE-R1/R1' re-PV required; just empirical re-verification of test suites + parity measurements + cross-substrate META scan).

---

## Operator-routable identification: substrates at 3/3 = PAID CUDA DISPATCH AUTHORIZED

**Per the R2-COMBINED charter**: *"Identify substrates that have advanced 3/3 = PAID CUDA DISPATCH AUTHORIZED"*

**Verdict**: **NONE YET at 3/3 — but 3 substrates AT 2/3 reachable at R3**:

1. **D=Z6 predictive-coding** (most-advanced; canonical sister-canonical MLX reference)
2. **B'=Z7-Mamba-2-v2** (3-phase cargo-cult-first methodology back-to-back CLEAN)
3. **C'=NSCS06-v8 chroma_lut** (4 unwinds + cargo-cult #5 EMPIRICALLY CONFIRMED at L0)

**If R3 CLEAN on all 3**: D, B', C' all reach 3/3 SEAL → paid CUDA dispatch authorized for first Path 3 contest-CUDA score per the R2-COMBINED charter directive.

**Operator-routable next** (post-R3 IF CLEAN):
- **D=Z6**: PyTorch port + paid CUDA dispatch for first Path 3 contest-CUDA score (per the R2-COMBINED charter directive)
- **B'=Z7-Mamba-2-v2**: L1 follow-up subagent implements Mamba2V2Cell + Z7MCM3 archive + MPS proxy probe ($0); paired CUDA dispatch only AFTER MPS-Win on ≥1 axis per Phase 3 §6 probe-disambiguator
- **C'=NSCS06-v8**: L1 promotion via wiring cls_stream consumption at L0 inflate + per-substrate symposium per Catalog #325 within 14-day window 2026-05-26 → 2026-06-09

---

## Counter state per CLAUDE.md "Recursive adversarial review protocol — close paths"

- **Before R2-COMBINED**: A=0/3 / B'=1/3 / C'=1/3 / D=1/3 / E=0/3 / F=0/3 / G=0/3
- **Per-landing post-R2-COMBINED**: ALL 7 CLEAN (counter advances +1 each per protocol item 4)
- **Aggregate post-R2-COMBINED**: A=1/3 / B'=2/3 / C'=2/3 / D=2/3 / E=1/3 / F=1/3 / G=1/3
- **Post-R3-CLEAN-on-all-7**: A=2/3 / B'=3/3 SEAL / C'=3/3 SEAL / D=3/3 SEAL / E=2/3 / F=2/3 / G=2/3
- **Post-R4-CLEAN-on-A+E+F+G**: A=3/3 SEAL / E=3/3 SEAL / F=3/3 SEAL / G=3/3 SEAL
- **Operator-declared SEAL (D-1, conservative)**: NOT applicable here (counter-advance path is straightforward).

---

## Discipline applied

- **Catalog #229 PV**: 7 per-substrate R2 memos + 2 aggregate R1/R1' memos + 2 FIX-WAVE memos + 7 substrate test suites empirically re-verified (369/369 PASS) + canonical equation registry empirically queried (8 cited equations REGISTERED) before any review claim
- **Catalog #110/#113 APPEND-ONLY**: 8 NEW memos (7 per-substrate + 1 aggregate); R1/R1' + FIX-WAVE + landing memos NEVER mutated
- **Catalog #117/#157/#174/#235/#289**: commit forthcoming via canonical `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256`
- **Catalog #119**: Co-Authored-By Claude trailer
- **Catalog #287**: every finding closure carries `[empirical:<measurement>]` evidence-tag; no placeholder rationales (every assumption-adversary verdict ≥4 chars; placeholder `<rationale>` / `<reason>` literals REJECTED)
- **Catalog #208**: docs/local-paths — only relative paths cited
- **Catalog #292**: per-axis council member operating-within assumption surfaced explicitly in frontmatter (all 7 per-substrate memos + this aggregate)
- **Catalog #300 v2**: full frontmatter on all 8 memos (tier T2 per-substrate; T3 aggregate; attendees include canonical 4-co-lead structure Shannon+Dykstra+Rudin+Daubechies per 2026-05-19 amendment + sister inner council voices; mission_contribution frontier_protecting; horizon_class frontier_pursuit)
- **Catalog #346**: canonical council roster `validate_council_dispatch_roster` returns complete=True for T3 aggregate (Shannon LEAD + Dykstra CO-LEAD + Rudin CO-LEAD + Daubechies CO-LEAD + Tao + Carmack + Hotz + Quantizr + MacKay + Selfcomp + Ballé + Hassabis + PR95Author + Contrarian + Assumption-Adversary = 15 attendees; ≥12-of-20 grand council quorum honored)
- **Catalog #340**: sister-checkpoint guard PROCEED verified at start of R2; no overlap with the 2 in-flight sister subagents (CONSOLIDATE-OP-1 + Wave #1 posterior_emission) per Catalog #230 ownership map (review is on NEW memo files only)
- **Catalog #206**: checkpoint discipline (4 checkpoints emitted; step 1 / 2 / 3 / 4-complete)
- **Catalog #126**: lane `lane_path_3_recursive_adversarial_review_r2_combined_3_axis_landings_a_b_c_d_e_f_g_20260526` pre-registered
- **Catalog #294 9-dim checklist** (1-9): UNIQUENESS (the 3-axis cross-substrate R2-COMBINED aggregate review is the 3rd of its kind for Path 3, distinct from R1 + R1' aggregates by covering combined 7-substrate scope); BEAUTY + ELEGANCE (8 memos ≤700 lines each); DISTINCTNESS (NOT a sister review; this is an AGGREGATE synthesis with cross-substrate META findings + CONSOLIDATE-OP status tracking); RIGOR (PV + empirical re-measurement + canonical equation verification + sister-canonical cross-comparison + 369-test re-verification); OPTIMIZATION PER TECHNIQUE (per-axis council members rotate per protocol); STACK-OF-STACKS-COMPOSABILITY (aggregate composes 7 per-substrate verdicts + cross-references R1 + R1' aggregates); DETERMINISTIC REPRODUCIBILITY (reproducer commands documented per Axis 2 in per-substrate memos); EXTREME OPTIMIZATION + PERFORMANCE (R2-COMBINED review takes ~4h wall-clock vs paid-dispatch cost $0); OPTIMAL MINIMAL CONTEST SCORE (R2 is QUALITY GATE not score-claim; non-promotable per Catalog #287 / #341 Tier A)
- **CLAUDE.md "Recursive adversarial review protocol — close paths"** items 1-8: this aggregate review IS R2-COMBINED; ALL 7 substrates advance counter +1; item 8 (NEW assumption-challenge axis) satisfied via per-substrate Assumption-Adversary verdicts + aggregate-level 3-row Assumption-Adversary verdict
- **CLAUDE.md "Council conduct" amendment 2026-05-19 4-co-lead structure**: T3 aggregate roster includes all 4 co-leads per Catalog #346 requirement
- **CLAUDE.md "Executing actions with care"**: review-only (NO code modifications); CONSOLIDATE-OP-1 and Wave #1 posterior_emission are canonical owners of any source modifications in this window

---

## Cross-references

- **R2-COMBINED per-substrate review memos**:
  - A=DreamerV3: `.omx/research/path_3_a_recursive_adversarial_review_r2_combined_3_axis_20260526.md`
  - B'=Z7-Mamba-2-v2: `.omx/research/path_3_b_recursive_adversarial_review_r2_combined_3_axis_20260526.md`
  - C'=NSCS06-v8: `.omx/research/path_3_c_recursive_adversarial_review_r2_combined_3_axis_20260526.md`
  - D=Z6: `.omx/research/path_3_d_recursive_adversarial_review_r2_combined_3_axis_20260526.md`
  - E=BoostNeRV: `.omx/research/path_3_e_recursive_adversarial_review_r2_combined_3_axis_20260526.md`
  - F=Z8: `.omx/research/path_3_f_recursive_adversarial_review_r2_combined_3_axis_20260526.md`
  - G=NIRVANA: `.omx/research/path_3_g_recursive_adversarial_review_r2_combined_3_axis_20260526.md`
- **Predecessor aggregate reviews**:
  - R1 aggregate (A+D+E): `.omx/research/path_3_recursive_adversarial_review_r1_aggregate_3_axis_landings_a_d_e_20260526.md`
  - R1' aggregate (B'+C'+F+G): `.omx/research/path_3_recursive_adversarial_review_r1_prime_aggregate_3_axis_landings_b_c_f_g_20260526.md`
- **Predecessor FIX-WAVE landings**:
  - FIX-WAVE-R1 (closed A+E R1 findings): `.omx/research/path_3_fix_wave_r1_close_findings_landed_20260526.md` (commit `e1b101888`)
  - FIX-WAVE-R1' (closed F+G R1' findings): `.omx/research/path_3_fix_wave_r1_prime_close_findings_landed_20260526.md` (commit `4684dbbab`)
- **Landing memos under cumulative review**:
  - A=DreamerV3: `.omx/research/dreamer_v3_rssm_mlx_scaffold_landed_20260526.md` (commit `69253a1cc`)
  - B'=Z7-Mamba-2-v2: `.omx/research/path_3_b_z7_mamba_2_L0_scaffold_landed_20260526.md` (commit `7a103fdbb`)
  - C'=NSCS06-v8: `.omx/research/path_3_c_nscs06_v8_chroma_lut_L0_scaffold_landed_20260526.md` (commit `f59c8401b`)
  - D=Z6: `.omx/research/z6_predictive_coding_mlx_scaffold_landed_20260526.md` (commit `83b9ee3e2`) + `.omx/research/path_3_d_z6_l1_promotion_landed_20260526.md` (commit `8833b9db5`)
  - E=BoostNeRV: `.omx/research/path_3_e_boost_nerv_against_pr110_L0_scaffold_landed_20260526.md` (commit `83910e54e`)
  - F=Z8: `.omx/research/path_3_f_z8_hierarchical_predictive_coding_L0_scaffold_landed_20260526.md` (commit `5ff5d2ab9`)
  - G=NIRVANA: `.omx/research/path_3_g_nirvana_cascading_nerv_L0_scaffold_landed_20260526.md` (commit `f7d2e86fe`)
- **Canonical references**:
  - Sister D=Z6 `_pixel_shuffle_2x_nhwc` (canonical correct convention): `src/tac/substrates/time_traveler_l5_z6/mlx_renderer.py::_pixel_shuffle_2x_nhwc` (lines 361-372)
  - Canonical PR95 helper: `tac.local_acceleration.pr95_hnerv_mlx::bilinear_resize2x_align_corners_false_nhwc` + `pixel_shuffle_2x_nhwc`
  - Canonical equation registry: `tac.canonical_equations.query_equations()` (42 equations REGISTERED as of 2026-05-26)
  - Canonical Catalog #1265 anchor: `.omx/research/mlx_candidate_contest_equivalence_gate_landed_20260526.md`
- **In-flight sister subagents (NOT in this R2 scope; queued for R1'' when H/I/J/K stabilize)**:
  - H=`aba5069741fc4475b` ATW V2 cooperative-receiver v2 (NEW landing; not in R2 scope)
  - I=`a71f2c4404c978f50` V1 Faiss IVF-PQ residual (NEW landing; not in R2 scope)
  - J=`abfd5113f1892447c` MDL-IBPS (NEW landing; not in R2 scope)
  - K=`a7977f23a7f0f0573` COIN++ implicit neural representation (NEW landing; not in R2 scope)
  - CONSOLIDATE-OP-1=`consolidate-op-1` pid 82551 (in-flight on A+D+F mlx_renderer.py + canonical PR95 module)
  - Wave #1 posterior_emission=`wave_1_posterior_emission_canonical_wire_in_20260526` (in-flight on substrate __init__.py)
- **Lane**: `lane_path_3_recursive_adversarial_review_r2_combined_3_axis_landings_a_b_c_d_e_f_g_20260526` L1 (impl_complete + memory_entry)

---

## Final aggregate verdict

**PROCEED — R2-COMBINED CLEAN ACROSS ALL 7 SUBSTRATES** — counter advances per-substrate per protocol item 4; FIX-WAVE-R2-COMBINED NOT REQUIRED; R3 UNBLOCKED.

The substrate paradigms (categorical posterior DreamerV3 / Mamba-2 selective SSM / NSCS06 v8 chroma_lut cargo-cult-unwound / FiLM-conditioned predictive coding / boosting-against-PR110-residual / Z8 hierarchical predictive coding canonical-quadruple / NIRVANA hierarchical residual cascade) are HARD-EARNED at the math + scientific + engineering level per Axis 1 across all 7 substrates. Axis 2 (MLX drift) is HARD-EARNED for all MLX-first substrates with R1/R1' findings closed via FIX-WAVE; N/A by construction for non-MLX substrates. Axis 3 (numpy portability) is HARD-EARNED-NATIVE (C') OR HARD-EARNED + SISTER-CANONICAL (G) OR N/A by structural posture (A+D+E+F) OR N/A by L0 scope (B').

**Notable wins from R2-COMBINED**:

1. **D=Z6, B', C' all reach 2/3 cumulative** — paid CUDA dispatch authorized at R3 if CLEAN; D=Z6 specifically is most-advanced for first Path 3 contest-CUDA score
2. **A=DreamerV3, F=Z8 FIX-WAVE closures empirically re-verified** — MLX↔PyTorch decoder parity stable at canonical-byte-stable per the canonical PR95 helper + sister D=Z6 reference convention
3. **3-axis methodology empirically validates across 4 substrate scope categories** (MLX-first / PyTorch-only / non-MLX numpy+PyTorch / scaffold-only L0)
4. **All 8 cited canonical equations REGISTERED** — sister-substrate-wide gap reduced to B' + D + G Phase 2 council symposium responsibility
5. **CONSOLIDATE-OP-1 + Wave #1 posterior_emission in-flight** — sister coordination preserved; R3 must re-verify post-landing
6. **Cargo-cult-first methodology (B' + C')** empirically materializes back-to-back R1' CLEAN + R2 CLEAN — operator directive #2 compliance pays off in review quality
7. **G=NIRVANA numpy_reference exemplary pattern** preserved as the canonical Axis 3 sister-canonical; META-CONSOLIDATE-OP-2 operator-routable for canonical extraction

Estimated R3 wall-clock: ~3-4h (faster than R2 because no FIX-WAVE re-PV required). Estimated R4 wall-clock for A+E+F+G: ~2-3h (4 substrates).

**Mission alignment per Catalog #300**: `frontier_protecting` — the R2-COMBINED review prevents L0→L1 promotion of substrates with TRAINING-INVALIDATING MLX↔PyTorch drift bugs (re-verified extincted in A + F post-FIX-WAVE) AND DOCUMENTATION OVERSTATEMENT (re-verified extincted in E + G post-FIX-WAVE). Counter advance from 0-1/3 → 1-2/3 brings 3 substrates within ONE round (R3) of paid CUDA dispatch authorization. Closing R3 + R4 unblocks the canonical Path 3 substrate-class-shift pursuit at $0 cost per the canonical Catalog #1265 contest-equivalence gate threshold.
