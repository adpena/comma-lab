<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE — R3-COMBINED per-substrate review record for F=Z8 hierarchical predictive coding. DO NOT mutate after landing. -->
<!-- Catalog #344 canonical equation cross-ref: 5 canonical equations REGISTERED (mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1 + scorer_conditional_joint_rate_distortion_floor_v1 + categorical_posterior_capacity_vs_continuous_gaussian_v1 + ego_motion_concentration_prior_v1 + cross_codec_super_additive_orthogonality_predictor_v1). FORMALIZATION_PENDING:r3_combined_post_consolidate_post_wave_1_methodology_pending_council_revisions_per_recursive_adversarial_review_protocol_close_paths_item_8 -->
---
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Rudin
  - Daubechies
  - Quantizr
  - MacKay
  - Ballé
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "F=Z8 CONSOLIDATE-OP-1 delegation pattern produces BYTE-IDENTICAL PixelShuffle outputs to pre-CONSOLIDATE substrate-local helper AND F=Z8's bilinear is preserved via earlier FIX-WAVE-R1' delegation"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "Empirical re-measurement 2026-05-26T09:11Z post-CONSOLIDATE-OP-1: PixelShuffle 2x NHWC max_abs vs PyTorch = 0.0 (canonical helper byte-stable; F's local _pixel_shuffle_2x_nhwc at mlx_renderer.py:264-297 now delegates to canonical pr95_hnerv_mlx). Per CONSOLIDATE-OP-1 landing memo §STEP 2: F=Z8's bilinear was already-delegating (pre-existing FIX-WAVE-R1' wiring); PixelShuffle migration LANDED in CONSOLIDATE wave. Per Wave #1 landing memo: F's posterior anchor is PRE-FIX-WAVE-R1' with max_abs=3.77 pre-fix provenance citation; post-fix anchor operator-routable per Catalog #110/#113 APPEND-ONLY (NOT a regression; canonical historical provenance discipline)."
  - assumption: "F=Z8 cited 5 canonical equations (including the joint with A=DreamerV3 via categorical_posterior_capacity_vs_continuous_gaussian_v1) ALL REGISTERED in canonical equations registry"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "Empirical query 2026-05-26T09:11Z: tac.canonical_equations.query_equations() returns 42 equations; ALL 5 F=Z8 cited equations REGISTERED: mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1 / scorer_conditional_joint_rate_distortion_floor_v1 / categorical_posterior_capacity_vs_continuous_gaussian_v1 (shared with A) / ego_motion_concentration_prior_v1 / cross_codec_super_additive_orthogonality_predictor_v1. F=Z8 carries the LARGEST canonical equation citation set (5) of any Wave #1 substrate — reflects the hierarchical canonical-quadruple binding per Catalog #312 sister gate."
  - assumption: "F=Z8 64-test suite remains PASS post-CONSOLIDATE-OP-1 + Wave #1 (no regression)"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "Empirical re-verification 2026-05-26T09:11Z: 64/64 z8_hierarchical_predictive_coding test suite PASS. NO regression introduced by either CONSOLIDATE-OP-1 (canonical helper delegation byte-identical) OR Wave #1 (no strict-__all__-equality test in F's suite)."
council_decisions_recorded:
  - "R3 verdict: CLEAN — counter advances 1/3 → 2/3"
  - "CONSOLIDATE-OP-1 PixelShuffle delegation empirically verified byte-stable (max_abs = 0.0 vs PyTorch); IMPROVES on R2's FIX-WAVE-R1' anchor"
  - "Wave #1 posterior emission wire-in NO REGRESSION on F's 64-test suite"
  - "5 canonical equations REGISTERED — F=Z8 carries largest canonical equation citation set of any reviewed substrate"
  - "FIX-WAVE-R3-COMBINED NOT REQUIRED for F — all 3 axes CLEAN"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_pursuit
canonical_equation_refs:
  - mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1
  - scorer_conditional_joint_rate_distortion_floor_v1
  - categorical_posterior_capacity_vs_continuous_gaussian_v1
  - ego_motion_concentration_prior_v1
  - cross_codec_super_additive_orthogonality_predictor_v1
predicted_band_validation_status: pending_post_training
related_deliberation_ids:
  - path_3_f_recursive_adversarial_review_r2_combined_3_axis_20260526
  - path_3_recursive_adversarial_review_r2_combined_aggregate_3_axis_landings_a_b_c_d_e_f_g_20260526
  - path_3_consolidate_op_1_canonical_mlx_primitives_extraction_landed_20260526
  - path_3_wave_1_posterior_emission_canonical_wire_in_landed_20260526
  - path_3_fix_wave_r1_prime_close_findings_landed_20260526
---

# R3-COMBINED Recursive Adversarial Review — F=Z8 hierarchical predictive coding

**Pre-R3 counter**: 1/3 (R1' reset + FIX-WAVE-R1' closed; R2 CLEAN)
**R3 verdict**: **PROCEED — CLEAN PASS** → counter **1/3 → 2/3**

---

## Axis 1: math + scientific + engineering rigor — CLEAN

### HARD-EARNED architectural choices (re-verified post-CONSOLIDATE + Wave #1):

1. **Hierarchical predictive coding 4-level cascade** preserved
2. **5 canonical equations REGISTERED** (largest citation set of any reviewed substrate):
   - `mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1`
   - `scorer_conditional_joint_rate_distortion_floor_v1`
   - `categorical_posterior_capacity_vs_continuous_gaussian_v1` (shared with A=DreamerV3)
   - `ego_motion_concentration_prior_v1` (shared with D=Z6)
   - `cross_codec_super_additive_orthogonality_predictor_v1`
3. **Cross-substrate canonical equation reuse** validates Catalog #344 canonical equations registry design — F=Z8 is THE substrate where hierarchical canonical-quadruple binding empirically materializes

### CARGO-CULTED assumptions surfaced (none NEW):

(All R1' cargo-cults closed via FIX-WAVE-R1' commit 4684dbbab.)

**Verdict**: 0 findings.

## Axis 2: MLX drift minimization — CLEAN

Empirical re-measurement 2026-05-26T09:11Z post-CONSOLIDATE-OP-1:

| Operation | Canonical helper status | max_abs vs PyTorch | Status |
|---|---|---|---|
| PixelShuffle 2x NHWC | delegates to canonical (CONSOLIDATE-OP-1 commit caf29acdb) | **0.00** | byte-stable PASS |
| Bilinear 2x NHWC | delegates to canonical (pre-existing FIX-WAVE-R1') | < 1e-5 | PASS |

F=Z8's local `_pixel_shuffle_2x_nhwc` at `mlx_renderer.py:264-297` now delegates to canonical `tac.local_acceleration.pr95_hnerv_mlx.pixel_shuffle_2x_nhwc` (line 297: `return pixel_shuffle_2x_nhwc(x)`). F=Z8's bilinear was already-delegating per FIX-WAVE-R1' wiring (commit 4684dbbab). Improved from R2's FIX-WAVE-R1' anchor to canonical byte-stable (max_abs = 0.0). PYTORCH-EXPORT-BOUNDARY-DRIFT bug class (the canonical 3.77 + 1.51 anchors from R1') STRUCTURALLY EXTINCTED via META-LAYER delegation.

**Verdict**: 0 findings.

## Axis 3: portability via numpy — N/A by structural posture (CLEAN)

F=Z8's inflate.py uses PyTorch CPU as the canonical CPU-portable reference. Same R2 verdict.

**Verdict**: N/A by construction; 0 findings.

## Cross-axis META observations

1. **CONSOLIDATE-OP-1 EXTINCTS recurrence bug class at F's surface** — F=Z8 inherited A=DreamerV3's pre-fix code LINE-FOR-LINE per R1' empirical proof (max_abs=3.77 + 1.51). Post-CONSOLIDATE-OP-1, F's PixelShuffle is byte-identical to A's via shared canonical helper.
2. **Wave #1 posterior emission anchor is PRE-FIX-WAVE-R1'** (per landing memo §STEP 2). NOT a regression: post-fix anchor operator-routable per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE discipline. The pre-fix anchor preserves the empirical falsification provenance per Catalog #307 paradigm-vs-implementation classification.
3. **F=Z8 5-equation citation set** validates hierarchical canonical-quadruple binding empirical materialization per Catalog #312 sister gate.

## Counter state per protocol

- **Before R3**: 1/3 (R1' reset + FIX-WAVE-R1' closed; R2 CLEAN)
- **R3 verdict**: CLEAN
- **Post-R3**: **2/3** (R4 needed for 3/3 SEAL)

## Operator-routable next

1. **R4** scheduled per protocol (1 more clean-pass round needed for 3/3 SEAL)
2. **Post-fix Wave #1 anchor refresh** (operator-routable; NOT R3 finding) — append-only NEW post-fix anchor citing CONSOLIDATE-OP-1 byte-stable max_abs=0.0

## Discipline applied

- Catalog #229 PV (test re-run + delegation grep + canonical helper empirical re-measurement + canonical equation registry verification)
- Catalog #110/#113 APPEND-ONLY (PRE-FIX-WAVE-R1' anchor preserved per HISTORICAL_PROVENANCE)
- Catalog #287 placeholder rejection
- Catalog #208 docs/local-paths
- Catalog #300 v2 frontmatter
- Catalog #292 per-axis assumption surfacing
- Catalog #346 canonical roster validate_council_dispatch_roster complete=True

## Cross-references

- R2 predecessor: `path_3_f_recursive_adversarial_review_r2_combined_3_axis_20260526.md`
- F=Z8 landing: `path_3_f_z8_hierarchical_predictive_coding_L0_scaffold_landed_20260526.md`
- CONSOLIDATE-OP-1 landing: `path_3_consolidate_op_1_canonical_mlx_primitives_extraction_landed_20260526.md`
- Wave #1 posterior emission landing: `path_3_wave_1_posterior_emission_canonical_wire_in_landed_20260526.md`
- FIX-WAVE-R1' closure: `path_3_fix_wave_r1_prime_close_findings_landed_20260526.md`
- Lane: `lane_path_3_recursive_adversarial_review_r3_combined_3_axis_landings_a_b_c_d_e_f_g_20260526` L1
