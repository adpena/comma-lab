<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE — R3-COMBINED per-substrate review record for D=Z6 predictive-coding. DO NOT mutate after landing. -->
<!-- Catalog #344 canonical equation cross-ref: ego_motion_concentration_prior_v1 + categorical_posterior_capacity_vs_continuous_gaussian_v1 (registered via Wave #1 wire-in). FORMALIZATION_PENDING:r3_combined_post_consolidate_post_wave_1_methodology_pending_council_revisions_per_recursive_adversarial_review_protocol_close_paths_item_8 -->
---
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Rudin
  - Daubechies
  - Quantizr
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "D=Z6 sister-canonical MLX reference status STRUCTURALLY PRESERVED post-CONSOLIDATE-OP-1 (D's local _pixel_shuffle_2x_nhwc now delegates to canonical pr95_hnerv_mlx instead of being the de-facto reference)"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "Empirical re-measurement 2026-05-26T09:11Z: PixelShuffle 2x NHWC max_abs vs PyTorch = 0.0 (canonical helper byte-stable); D's general-form bilinear via canonical bilinear_resize_nhwc max_abs vs PyTorch = 1.37e-06 (well below 1e-5 fp32 noise floor). D's local _pixel_shuffle_2x_nhwc at mlx_renderer.py:361-389 delegates to canonical helper at pr95_hnerv_mlx; D's local _bilinear_resize_nhwc at lines 392-417 delegates to canonical general-form helper. The 'sister-canonical' attribute migrates to META-LAYER STATUS: pr95_hnerv_mlx IS the canonical source; D=Z6 PRESERVES the original channel-FIRST convention by delegation rather than re-implementation. R2 sister-canonical status NOT degraded; it is now META-LAYER-CANONICAL."
  - assumption: "D=Z6 Wave #1 posterior emission wire-in (commit f6b432be1+3d103dafd) preserves the L1-PROMOTION canonical surface declared at commit 8833b9db5"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "Empirical verification: tac.substrates.time_traveler_l5_z6 exports SUBSTRATE_ID='time_traveler_l5_z6', ARCHITECTURE_CLASS='z6_predictive_coding_film_conditioned_next_frame_l1_mlx' (note L1 not L0 — preserves L1-PROMOTION distinction), CANONICAL_EQUATION_IDS=['ego_motion_concentration_prior_v1'] (resolves R2 META op-routable: D no longer 'sister-substrate-wide gap'), emit_landing_posterior_anchor importable. Sister-canonical posterior emission discipline ALIGNED with all 7 reviewed substrates. 68/68 test suite PASS (49 z6 + 19 mlx_renderer)."
  - assumption: "Axis 3 (numpy portability) remains N/A by construction for D=Z6 (MLX-first + PyTorch inflate; no sister numpy_reference.py needed)"
    classification: HARD-EARNED
    rationale: "D=Z6's inflate.py uses PyTorch CPU as the canonical CPU-portable reference; same structural posture as R2 (Axis 3 N/A by structural posture). Wave #1 + CONSOLIDATE-OP-1 do not change this; both wire-ins target MLX renderer surface, not PyTorch inflate."
council_decisions_recorded:
  - "R3 verdict: CLEAN — counter advances 2/3 → 3/3 SEAL"
  - "**3/3 SEAL = PAID CUDA DISPATCH AUTHORIZED** per CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA' (modulo Catalog #1265 contest-equivalence gate empirical verification — operator-routable next)"
  - "Canonical equation registration GAP CLOSED (R2 META op-routable for B'+D+G satisfied for D via Wave #1: ego_motion_concentration_prior_v1)"
  - "Sister-canonical MLX reference status PRESERVED via META-LAYER delegation pattern (canonical pr95_hnerv_mlx becomes the source; D preserves channel-FIRST convention by importing rather than re-implementing)"
  - "FIX-WAVE-R3-COMBINED NOT REQUIRED for D — all 3 axes CLEAN post-CONSOLIDATE-OP-1 + Wave #1"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_pursuit
canonical_equation_refs:
  - ego_motion_concentration_prior_v1
predicted_band_validation_status: pending_post_training
related_deliberation_ids:
  - path_3_d_recursive_adversarial_review_r2_combined_3_axis_20260526
  - path_3_recursive_adversarial_review_r2_combined_aggregate_3_axis_landings_a_b_c_d_e_f_g_20260526
  - path_3_consolidate_op_1_canonical_mlx_primitives_extraction_landed_20260526
  - path_3_wave_1_posterior_emission_canonical_wire_in_landed_20260526
  - path_3_d_z6_l1_promotion_landed_20260526
---

# R3-COMBINED Recursive Adversarial Review — D=Z6 predictive-coding (substrate canonical MLX reference)

**Per binding operator directive 2026-05-26 #3** (3-axis review): math+sci+engineering rigor / MLX drift / numpy portability

**Per CLAUDE.md "Recursive adversarial review protocol — close paths"** R3 of 3 consecutive clean-pass cycles.

**Pre-R3 counter**: 2/3 (R1 CLEAN per R1 aggregate + R2-COMBINED CLEAN)
**R3 verdict**: **PROCEED — CLEAN PASS** → counter **3/3 SEAL** = **PAID CUDA DISPATCH AUTHORIZED**

---

## Axis 1: math + scientific + engineering rigor — CLEAN

### HARD-EARNED architectural choices (re-verified post-CONSOLIDATE-OP-1 + Wave #1):

1. **FiLM-conditioned multi-layer predictor** (canonical Z6 design) — preserved at `multi_layer_film_predictor.py`; test suite 68/68 PASS
2. **L1-PROMOTION canonical surface** (commit `8833b9db5`) — preserved: ARCHITECTURE_CLASS still `_l1_mlx` (not L0)
3. **Canonical equation citation** `ego_motion_concentration_prior_v1` REGISTERED (resolves R2 META op-routable)
4. **Sister-canonical MLX reference convention** PRESERVED via META-LAYER delegation (channel-FIRST PixelShuffle convention is now owned by canonical `pr95_hnerv_mlx`; D imports rather than re-implements)

### CARGO-CULTED assumptions surfaced (none NEW; R2 list re-affirmed):

- 600-pair recurrence O(N) per batch (R1 advisory; R2 reaffirmed; R3 reaffirmed; still L1+ performance op-routable)

**Verdict**: 0 findings.

## Axis 2: MLX drift minimization — CLEAN

Empirical re-measurement 2026-05-26T09:11Z post-CONSOLIDATE-OP-1:

| Operation | Canonical helper | max_abs vs PyTorch | Status |
|---|---|---|---|
| PixelShuffle 2x NHWC | `pixel_shuffle_2x_nhwc` (delegates to canonical) | **0.00** | byte-stable PASS |
| Bilinear general-form NHWC | `bilinear_resize_nhwc` (delegates to canonical) | **1.37e-06** | < 1e-5 PASS |

D=Z6's local `_pixel_shuffle_2x_nhwc` + `_bilinear_resize_nhwc` are now thin delegation wrappers (lines 361-389 + 392-417 of `mlx_renderer.py`) importing from `tac.local_acceleration.pr95_hnerv_mlx`. R2's sister-canonical status STRUCTURALLY PRESERVED: D's correct channel-FIRST convention is the seed that produced the canonical helper; D now consumes it via canonical helper.

**Verdict**: 0 findings.

## Axis 3: portability via numpy — N/A by structural posture (CLEAN)

D=Z6's inflate.py uses PyTorch CPU as the canonical CPU-portable reference. No sister `numpy_reference.py` needed. Same R2 verdict.

**Verdict**: N/A by construction; 0 findings.

## Cross-axis META observations

1. **Wave #1 posterior emission wire-in NO REGRESSION on D's 68-test suite** (49 z6 + 19 mlx_renderer). The 4 new exports (`SUBSTRATE_ID`, `ARCHITECTURE_CLASS`, `CANONICAL_EQUATION_IDS`, `emit_landing_posterior_anchor`) added without breaking any existing test (Wave #1 helpfully did NOT add a strict-`__all__`-equality test for D=Z6).
2. **CONSOLIDATE-OP-1 delegation pattern preserves R2 sister-canonical status** via META-LAYER consumption rather than degrading it. D=Z6 IS THE SOURCE that informed the canonical helper.
3. **L1-PROMOTION distinction preserved** in ARCHITECTURE_CLASS naming (`_l1_mlx` vs sister substrates' `_l0_scaffold_mlx`). Cathedral-consumer-discoverable per Catalog #335.

## Counter state per protocol

- **Before R3**: 2/3 (R1 CLEAN + R2 CLEAN aggregated)
- **R3 verdict**: CLEAN
- **Post-R3**: **3/3 SEAL** → **PAID CUDA DISPATCH AUTHORIZED** per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"

## Operator-routable post-SEAL

Per the protocol-binding-decision-cascade: D=Z6 is THE FIRST Path 3 substrate to reach 3/3 SEAL. Operator-routable next per CLAUDE.md "Frontier target — NON-NEGOTIABLE":

1. **PyTorch port verification** — D=Z6's inflate.py must produce `[contest-CUDA]` parity to MLX trainer convergence
2. **Catalog #1265 contest-equivalence gate** empirical run (paired CUDA + contest-CPU per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA")
3. **600-pair O(N) recurrence performance op-routable** (L1+ advisory carrying from R1) — `rollout_then_gather` optimization
4. **Per-substrate Catalog #325 symposium** within 14-day window 2026-05-26 → 2026-06-09 if not already scheduled

## Discipline applied

- Catalog #229 PV (test re-run + delegation grep + canonical helper empirical re-measurement before any verdict claim)
- Catalog #110/#113 APPEND-ONLY (NEW memo only)
- Catalog #287 placeholder-rationale rejection
- Catalog #208 docs/local-paths (relative paths only)
- Catalog #300 v2 frontmatter
- Catalog #292 per-axis assumption surfacing
- Catalog #340 sister-checkpoint guard PROCEED
- Catalog #287/#323 canonical Provenance for every score literal (axis_tag declared in posterior emission)
- Catalog #346 canonical roster validate_council_dispatch_roster returns complete=True for T2 roster

## Cross-references

- R2 predecessor: `path_3_d_recursive_adversarial_review_r2_combined_3_axis_20260526.md`
- CONSOLIDATE-OP-1 landing: `path_3_consolidate_op_1_canonical_mlx_primitives_extraction_landed_20260526.md`
- Wave #1 posterior emission landing: `path_3_wave_1_posterior_emission_canonical_wire_in_landed_20260526.md`
- D L1-PROMOTION landing: `path_3_d_z6_l1_promotion_landed_20260526.md`
- Canonical Catalog #1265 contest-equivalence anchor: `.omx/research/mlx_candidate_contest_equivalence_gate_landed_20260526.md`
- Lane: `lane_path_3_recursive_adversarial_review_r3_combined_3_axis_landings_a_b_c_d_e_f_g_20260526` L1
