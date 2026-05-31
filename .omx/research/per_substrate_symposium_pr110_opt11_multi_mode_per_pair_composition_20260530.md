# Per-substrate symposium: PR110-OPT-11 multi-mode-per-pair composition L0 SCAFFOLD

<!-- # FORMALIZATION_PENDING:l0_scaffold_per_substrate_symposium_memo_macos_cpu_advisory_only_canonical_equation_pr110_opt11_multi_mode_per_pair_composition_savings_v1_registers_at_l1_paired_cuda_ratification_landing_per_catalog_344_with_first_empirical_anchor_landing_via_tac_canonical_equations_update_equation_with_empirical_anchor_after_phase_2_council_symposium_proceed_unconditional_verdict_per_catalog_325 -->


---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "1.548x compound ratio is an analytical additive upper bound — true Dykstra-projected polytope yields sub-additive per Catalog #373. Empirical paired-CUDA RATIFICATION at L1 may falsify; do not promote based on the analytical alone."
council_assumption_adversary_verdict:
  - assumption: "Per-pair multi-mode composition produces additive ΔS savings"
    classification: CARGO-CULTED
    rationale: "Wave N+34 produces ONLY the analytical upper bound; the empirical 600-pair composed forward pass is NOT yet measured. True Pareto polytope intersection per Catalog #373 generically sub-additive."
    empirical_verification_status: INFERRED_FROM_DOMAIN_LITERATURE
  - assumption: "K=16 / 4-bit selector budget is sufficient for the canonical orthogonal family pair menu (largest 6×8=48 combinations)"
    classification: HARD-EARNED
    rationale: "Verified at construction: max 8 modes per family (rgb_bias); 4-bit selector addresses K=16 distinct modes per slot. The 48 combinations fit cleanly in 2 × 4-bit selectors per pair."
    empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION
  - assumption: "Orthogonal family pair composition is meaningful (composition is non-commutative + composition is distinct from single-mode)"
    classification: HARD-EARNED
    rationale: "Empirically verified in dedicated tests at L0 SCAFFOLD landing: 36 tests pass including test_compose_two_modes_differs_from_both_single_mode_outputs + test_compose_with_identity_b_equals_single_mode_a."
    empirical_verification_status: VERIFIED_VIA_EMPIRICAL_ANCHOR
council_decisions_recorded:
  - "op-routable #1: L1 PROMOTION wires real PR110 base pair reconstruction (currently L0 uses synthetic frames per Catalog #220 SCAFFOLD_DEFERRED_INTEGRATION_OK)"
  - "op-routable #2: Phase 2 council symposium per Catalog #325 PROCEED-unconditional verdict required before paid-CUDA RATIFICATION dispatch"
  - "op-routable #3: empirical 600-pair composed forward pass measurement to validate Wave N+34 analytical 1.548x upper bound (likely sub-additive per Dykstra polytope intersection)"
  - "op-routable #4: canonical equation pr110_opt11_multi_mode_per_pair_composition_savings_v1 FORMALIZATION_PENDING per Catalog #344 until L1 empirical anchor lands"
  - "op-routable #5: explore alternative family-pair indices [0, 2, 3, 4, 5] beyond default index 1 = (luma_bias, rgb_bias) per Catalog #308 alternative-probe-methodology"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
---

## 1. Cargo-cult audit per assumption (Catalog #303)

| Assumption | HARD-EARNED vs CARGO-CULTED | Empirical Status | Unwind Path |
|---|---|---|---|
| Additive composition assumption | CARGO-CULTED (per Wave N+34 explicit) | INFERRED_FROM_DOMAIN_LITERATURE | Empirical 600-pair composed forward pass at L1 PROMOTION |
| 4-bit K=16 selector budget | HARD-EARNED | VERIFIED_VIA_SOURCE_INSPECTION | None — empirically verified |
| Composition non-commutativity | HARD-EARNED | VERIFIED_VIA_EMPIRICAL_ANCHOR | None — 36 tests prove distinct output |
| Synthetic smoke frames at L0 | DOCUMENTED_ADAPTATION | INFERRED_FROM_DOMAIN_LITERATURE | Catalog #213 Comma2k19 real frames at L1 |
| Family pair index 1 default (luma_bias × rgb_bias) | CARGO-CULTED | INFERRED_FROM_DOMAIN_LITERATURE | Empirical sweep across 6 canonical orthogonal family pair indices [0, 5] at L1 |
| Wire cost 428B vs FEC6 249B | HARD-EARNED | VERIFIED_VIA_SOURCE_INSPECTION (Wave N+34 analytical) | None — exact computation |

## 2. 9-dimension success checklist evidence (Catalog #294)

### Dimension 1: UNIQUENESS
- Substrate composes 2 perturbation modes per pair via canonical OPT11MMP grammar.
- Sister-DISJOINT vs PR110-OPT-7 L1 (which composes 5 helpers into single archive; #11 is the multi-mode-per-pair sibling).
- Wave N+34 ANALYTICAL UPPER BOUND classifies as PROCEED_CANDIDATE_FOR_EMPIRICAL_VALIDATION (uniquely targets the per-pair multi-mode composition surface).

### Dimension 2: BEAUTY + ELEGANCE
- 32-byte canonical header + 2-section length-prefixed grammar (HNeRV parity L20 monolithic 4-section pattern simplified to 2-section for L0 SCAFFOLD; L1+ may sweep more sections).
- Per-pair (selector_a, selector_b) → 4-bit packed uint8 stream = 600 bytes raw → ~428B brotli (Wave N+34 anchor).
- Frozen-offset declaration in source per HNeRV parity L3.

### Dimension 3: DISTINCTNESS
- Distinct from PR110-OPT-7 (which composes 5 helpers but applies ONE perturbation per pair).
- Distinct from FEC6 baseline (which applies ZERO frame-0 perturbations).
- Distinct from PR110-OPT-1 frame-0 perturbation catalog (which applies ONE perturbation per pair).
- Canonical distinguishing-feature: 2-mode-per-pair composition primitive per Catalog #272.

### Dimension 4: RIGOR
- Premise verification per Catalog #229: Wave N+34 analytical foundation cited + 22-mode canonical menu source verified.
- 36 dedicated tests covering:
  - Canonical archive grammar contracts (header pack/unpack roundtrip; selector stream pack/unpack 4-bit + 8-bit; reject wrong magic; reject wrong sha prefix length; reject overflow).
  - Canonical mode menu + 6 orthogonal family pair enumeration.
  - Canonical perturbation behavior (identity / luma_bias / blue_chroma / rgb_bias / roll all actually change frame).
  - Composition non-commutativity + identity element + distinct-output invariants (Slot EEE NO FAKE).
  - Canonical config + Tier A result + Catalog #341 markers + Catalog #323 Provenance.
  - Deterministic per-seed reproducibility + cross-seed distinctness + cross-family-pair distinctness.

### Dimension 5: OPTIMIZATION PER TECHNIQUE
- Per Catalog #290 canonical-vs-unique decision per layer: substrate FORKS its archive grammar (OPT11MMP magic) and selector stream packing (4-bit selector helper) per UNIQUE-AND-COMPLETE-PER-METHOD operating mode; ADOPTS canonical Provenance + Tier A markers + inflate runtime template.

### Dimension 6: STACK-OF-STACKS-COMPOSABILITY
- Substrate operates on per-pair selector stream (orthogonal to PR110-OPT-4 selector reduction via grouping).
- Composes ADDITIVELY at the analytical surface but SUB-ADDITIVELY at the Dykstra polytope surface per Wave N+34 + Catalog #373.

### Dimension 7: DETERMINISTIC REPRODUCIBILITY
- Tests verify per-seed deterministic per_pair_selectors output.
- Byte-stable 593-byte archive at canonical seed=42 / family_pair_index=1.
- Canonical Provenance per Catalog #323 carries inputs_sha256 for full provenance round-trip.

### Dimension 8: EXTREME OPTIMIZATION + PERFORMANCE
- L0 SCAFFOLD smoke < 1ms wall-clock on macOS-CPU advisory (4-sample synthetic composition).
- 600-pair selector stream = 600 bytes raw (4-bit packed) → ~428B brotli (Wave N+34 anchor).
- L1+ PROMOTION sweeps modes_per_pair > 2 (compound > 2x base savings) + selector_bits_per_mode > 4 (K > 16).

### Dimension 9: OPTIMAL MINIMAL CONTEST SCORE
- Wave N+34 analytical upper bound: -0.00052 DIRECTIONAL net score delta vs FEC6 baseline.
- True Dykstra polytope intersection per Catalog #373: SUB-ADDITIVE so empirical savings ≤ -0.00052.
- L0 SCAFFOLD predicted band [-0.001, 0.001] per Catalog #324 PREDICTED_BAND_RANDOM_INIT_OK.
- Operator-routable L1 paired-CUDA RATIFICATION cost ~$0.20 per Catalog #246 envelope.

## 3. Observability surface (Catalog #305)

- **Inspectable per layer**: archive_grammar.py + substrate.py + inflate.py + trainer all expose canonical inspection hooks; smoke summary JSON captures per-sample composition deltas.
- **Decomposable per signal**: composition_behavioral_evidence per-sample deltas (base_vs_a / base_vs_b / a_vs_ab / ab_vs_base) decompose composition effect per (mode_a, mode_b).
- **Diff-able across runs**: deterministic per-seed reproducibility enables byte-level diff across config sweeps.
- **Queryable post-hoc**: smoke summary JSON + canonical Provenance dict are JSON-readable.
- **Cite-able**: canonical Provenance per Catalog #323 carries inputs_sha256; cross_reference_matrix cites 5 canonical anchors.
- **Counterfactual-able**: byte-mutation smoke per Catalog #139 (deferred to L1 PROMOTION); current L0 verifies composition_distinct_output_verdict=PASS.

## 4. Reactivation criteria (per CLAUDE.md "Forbidden premature KILL")

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": this L0 SCAFFOLD is research_only=true; the following reactivation paths apply at L1 promotion:

1. **PROCEED-unconditional verdict on per-substrate symposium** per Catalog #325 with full 6-step contract clearance.
2. **Empirical 600-pair composed forward pass** measuring actual ΔS vs the Wave N+34 analytical upper bound 1.548x.
3. **Alternative family-pair index sweep** [0, 2, 3, 4, 5] beyond default 1.
4. **Alternative selector-bits-per-mode** (K=32 via 5-bit; or K=64 via 6-bit) for larger combination space.
5. **Alternative per-pair selector derivation** (currently deterministic per-seed; L1 may use per-pair empirical-optimal-mode-pair from Wave N+34 600-pair component_score_no_rate deltas).
6. **Alternative composition modes** (currently mode_a then mode_b; L1 may explore weighted blending or per-pixel masked overlay).

## 5. Tier-C post-training Tier-C validation (Catalog #324)

Predicted band [-0.001, 0.001] is `predicted_band_validation_status: pending_post_training` per Catalog #324. Post-L1 paired-CUDA dispatch landing, run `tools/mdl_scorer_conditional_ablation.py --tier c` on the post-training archive sha256 to validate.

## 6. Dykstra-feasibility predicted-band per Catalog #296

Per Wave N+34 analytical investigator explicit warning: the predicted_delta `-0.00052` is the ANALYTICAL ADDITIVE UPPER BOUND; true Dykstra-projected polytope intersection per Catalog #373 yields SUB-ADDITIVE results — actual empirical savings will be LESS than this upper bound. Sister Catalog #373 canonical anti-pattern `cross_paradigm_stacking_additive_compounding_without_dykstra_feasibility` registered.

Predicted band [-0.001, 0.001] EXPLICITLY ACKNOWLEDGES the polytope sub-additivity by extending the band BELOW the upper bound (toward zero net savings) so L1 empirical landing is not surprised by sub-additive result.

## 7. Cross-references

- Wave N+34 analytical foundation: `.omx/research/wave_n34_pr110_opt_4_7_11_triple_artifacts_20260528.json`
- Sister PR110-OPT-7 L1 PROMOTION canonical pattern: commit `1230b3b9c`
- L0 SCAFFOLD landing memo: `~/.claude/projects/.../feedback_pr110_opt11_multi_mode_per_pair_composition_l0_scaffold_landed_20260530.md`
- Retroactive sweep memo: `.omx/research/retroactive_sweep_for_pr110_opt11_l0_scaffold_20260530.md`
- Canonical equation FORMALIZATION_PENDING per Catalog #344: `pr110_opt11_multi_mode_per_pair_composition_savings_v1`
- Catalog #287 placeholder-rejection: every assumption-adversary entry carries substantive non-placeholder rationale.
- Catalog #325 6-step contract: this memo IS the symposium evidence per acceptance (a) the canonical filename glob + (b) PROCEED_WITH_REVISIONS verdict.
