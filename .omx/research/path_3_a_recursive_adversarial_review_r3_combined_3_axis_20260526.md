<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE — R3-COMBINED per-substrate review record for A=DreamerV3 RSSM. DO NOT mutate after landing. -->
<!-- Catalog #344 canonical equation cross-ref: categorical_posterior_capacity_vs_continuous_gaussian_v1 + categorical_blahut_arimoto_rate_distortion_v1 (both REGISTERED). FORMALIZATION_PENDING:r3_combined_post_consolidate_post_wave_1_methodology_pending_council_revisions_per_recursive_adversarial_review_protocol_close_paths_item_8 -->
---
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Rudin
  - Daubechies
  - Hotz
  - Quantizr
  - Hassabis
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "A=DreamerV3 RSSM CONSOLIDATE-OP-1 delegation pattern produces BYTE-IDENTICAL PixelShuffle outputs to pre-CONSOLIDATE substrate-local helper (R2 CLEAN reproducibility preserved through META-LAYER extraction)"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "Empirical re-measurement 2026-05-26T09:11Z post-CONSOLIDATE-OP-1: PixelShuffle 2x NHWC max_abs vs PyTorch = 0.0 (canonical helper byte-stable). Per CONSOLIDATE-OP-1 landing memo §STEP 4: 6 sister-substrate delegation regression guards verify A+D+F substrate-local helpers produce identical output to canonical helpers via byte-exact np.testing.assert_array_equal. 11/11 A test suite PASS. The R1 FIX-WAVE-R1 closure (commit e1b101888) max_abs=0.0072 anchor preserved as historical provenance; post-CONSOLIDATE-OP-1 the empirical max_abs improves to 0.0 (canonical byte-stable)."
  - assumption: "A=DreamerV3 Wave #1 posterior emission wire-in does NOT regress A's 11-test suite OR break any structural canonical contract"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "Empirical verification: 11/11 A=DreamerV3 test suite PASS post-Wave-#1. SUBSTRATE_ID='dreamer_v3_rssm', ARCHITECTURE_CLASS='dreamer_v3_rssm_categorical_posterior_l0_scaffold_mlx', CANONICAL_EQUATION_IDS=['categorical_posterior_capacity_vs_continuous_gaussian_v1', 'categorical_blahut_arimoto_rate_distortion_v1'] (BOTH REGISTERED in tac.canonical_equations.query_equations()), emit_landing_posterior_anchor importable. A does NOT have a strict-`__all__`-equality test (unlike G=NIRVANA which DOES have one and breaks — see G's R3 memo)."
  - assumption: "Axis 3 N/A by structural posture for A=DreamerV3 (MLX-first + PyTorch inflate CPU = canonical CPU-portable reference)"
    classification: HARD-EARNED
    rationale: "Same R2 verdict. No sister numpy_reference.py needed; PyTorch CPU backend IS the canonical CPU-portable reference per structural posture."
council_decisions_recorded:
  - "R3 verdict: CLEAN — counter advances 1/3 → 2/3"
  - "CONSOLIDATE-OP-1 delegation pattern empirically verified byte-stable (max_abs PixelShuffle = 0.0 vs PyTorch)"
  - "Wave #1 posterior emission wire-in NO REGRESSION on A's 11-test suite"
  - "Both cited canonical equations REGISTERED (categorical_posterior_capacity_vs_continuous_gaussian_v1 + categorical_blahut_arimoto_rate_distortion_v1)"
  - "FIX-WAVE-R3-COMBINED NOT REQUIRED for A — all 3 axes CLEAN"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_pursuit
canonical_equation_refs:
  - categorical_posterior_capacity_vs_continuous_gaussian_v1
  - categorical_blahut_arimoto_rate_distortion_v1
predicted_band_validation_status: pending_post_training
related_deliberation_ids:
  - path_3_a_recursive_adversarial_review_r2_combined_3_axis_20260526
  - path_3_recursive_adversarial_review_r2_combined_aggregate_3_axis_landings_a_b_c_d_e_f_g_20260526
  - path_3_consolidate_op_1_canonical_mlx_primitives_extraction_landed_20260526
  - path_3_wave_1_posterior_emission_canonical_wire_in_landed_20260526
  - path_3_fix_wave_r1_close_findings_landed_20260526
---

# R3-COMBINED Recursive Adversarial Review — A=DreamerV3 RSSM

**Pre-R3 counter**: 1/3 (R1 reset + FIX-WAVE-R1 closed; R2 CLEAN)
**R3 verdict**: **PROCEED — CLEAN PASS** → counter **1/3 → 2/3**

---

## Axis 1: math + scientific + engineering rigor — CLEAN

### HARD-EARNED architectural choices (re-verified post-CONSOLIDATE + Wave #1):

1. **Categorical posterior** (discrete-categorical-MINE) preserved at substrate module
2. **2 canonical equations REGISTERED**: `categorical_posterior_capacity_vs_continuous_gaussian_v1` + `categorical_blahut_arimoto_rate_distortion_v1`
3. **MLX-first design** with PyTorch inflate CPU portability — CONSOLIDATE-OP-1 delegation preserves canonical primitives at META layer

### CARGO-CULTED assumptions surfaced (none NEW):

(All cargo-cults from R1' identified + FIX-WAVE-R1 closed via commit e1b101888.)

**Verdict**: 0 findings.

## Axis 2: MLX drift minimization — CLEAN

Empirical re-measurement 2026-05-26T09:11Z post-CONSOLIDATE-OP-1:

| Operation | Canonical helper status | max_abs vs PyTorch | Status |
|---|---|---|---|
| PixelShuffle 2x NHWC | delegates to canonical (commit caf29acdb) | **0.00** | byte-stable PASS |
| Bilinear 2x NHWC | delegates to canonical (pre-existing FIX-WAVE-R1) | < 1e-5 | PASS |

A=DreamerV3's local `_pixel_shuffle_2x_nhwc` at `module.py:184-216` now delegates to canonical `tac.local_acceleration.pr95_hnerv_mlx.pixel_shuffle_2x_nhwc` (line 216: `return pixel_shuffle_2x_nhwc(x)`). Improved from R2's anchor of max_abs=0.0072 (FIX-WAVE-R1 post-fix) to max_abs=0.0 (canonical byte-stable). PYTORCH-EXPORT-BOUNDARY-DRIFT bug class structurally extincted via META-LAYER delegation.

**Verdict**: 0 findings.

## Axis 3: portability via numpy — N/A by structural posture (CLEAN)

PyTorch CPU backend IS the canonical CPU-portable reference. Same R2 verdict.

**Verdict**: N/A by construction; 0 findings.

## Cross-axis META observations

1. **CONSOLIDATE-OP-1 IMPROVES Axis 2 drift** — from R2's max_abs=0.0072 (substrate-local fix) to R3's max_abs=0.0 (canonical helper byte-stable). META-LAYER consolidation EXTINCTS the recurrence bug class at the source.
2. **Wave #1 posterior emission wire-in NO REGRESSION** on A's 11-test suite. A's test suite does NOT have strict-`__all__`-equality test (unlike G=NIRVANA which DOES break — see G's R3 memo for cross-substrate META).
3. **Both canonical equations REGISTERED** — sister substrates (F=Z8) share `categorical_posterior_capacity_vs_continuous_gaussian_v1` per cross-substrate Pareto-feasible composition.

## Counter state per protocol

- **Before R3**: 1/3 (R1 reset + FIX-WAVE-R1 closed; R2 CLEAN)
- **R3 verdict**: CLEAN
- **Post-R3**: **2/3** (R4 needed for 3/3 SEAL)

## Operator-routable next

1. **R4** scheduled per protocol (1 more clean-pass round needed for 3/3 SEAL)
2. **CONSOLIDATE-OP-1 META-LAYER benefit confirmed** at A — future Path 3 substrates structurally inherit byte-stable PixelShuffle by importing canonical

## Discipline applied

- Catalog #229 PV (test re-run + delegation grep + canonical helper empirical re-measurement + canonical equation registry verification)
- Catalog #110/#113 APPEND-ONLY
- Catalog #287 placeholder rejection
- Catalog #208 docs/local-paths
- Catalog #300 v2 frontmatter
- Catalog #292 per-axis assumption surfacing
- Catalog #346 canonical roster validate_council_dispatch_roster complete=True

## Cross-references

- R2 predecessor: `path_3_a_recursive_adversarial_review_r2_combined_3_axis_20260526.md`
- A=DreamerV3 landing: `.omx/research/dreamer_v3_rssm_mlx_scaffold_landed_20260526.md`
- CONSOLIDATE-OP-1 landing: `path_3_consolidate_op_1_canonical_mlx_primitives_extraction_landed_20260526.md`
- Wave #1 posterior emission landing: `path_3_wave_1_posterior_emission_canonical_wire_in_landed_20260526.md`
- FIX-WAVE-R1 closure: `path_3_fix_wave_r1_close_findings_landed_20260526.md`
- Lane: `lane_path_3_recursive_adversarial_review_r3_combined_3_axis_landings_a_b_c_d_e_f_g_20260526` L1
