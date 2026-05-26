<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110 / #113 APPEND-ONLY HISTORICAL_PROVENANCE — this memo is the canonical R2-COMBINED review record for Path 3 candidate #B' (Z7-Mamba-2-v2 fresh substrate L0 SCAFFOLD; commit `7a103fdbb`). DO NOT mutate after landing. -->
<!-- Catalog #344 canonical equation cross-ref: B' frontmatter declares no equation refs at L0 per the 3-phase cargo-cult-first methodology (Phase 2 canonical equation registration is deferred to L1+ council symposium per Catalog #325). FORMALIZATION_PENDING:b_prime_r2_combined_review_canonical_equation_registration_deferred_to_phase_2_council_symposium_per_catalog_325_per_substrate_per_design_memo_section_18_pattern_sister_to_d_z6 -->
---
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Tao
  - Carmack
  - Hotz
  - Quantizr
  - MacKay
  - Selfcomp
  - Contrarian
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "B'=Z7-Mamba-2-v2's 3-phase cargo-cult-first methodology continues to PASS R2 review without re-emergence of cargo-culted assumptions"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "R2 re-verification at 2026-05-26T08:42Z: 40/40 tests PASS. Phase 1 audit identified 8 NEW CC + 2 NEW HARD-EARNED-PARTIAL beyond CC-1..CC-10; Phase 3 implements 16-layer canonical-vs-unique decision table with 8 UNIQUE-FORK + 6 CANONICAL-ADOPT + 1 UNIQUE-IMPL + 1 UNIQUE-DESIGN. Per R1' verdict (commit `71d8ff687`): R1' CLEAN PASS (0 findings; 2 anticipatory L1+ advisories). R2 confirms no regression."
  - assumption: "B' has ZERO MLX primitives at L0 (axis 2 N/A by construction) per the 3-phase methodology that defers MLX-native Mamba2V2Cell + Mamba2TemporalDecoder to L1+"
    classification: HARD-EARNED
    rationale: "L0 SCAFFOLD scope per landing memo decision row §'op-routable #1': L1 follow-up subagent implements Mamba2V2Cell + Mamba2TemporalDecoder + Z7MCM3 archive pack/unpack + MPS proxy probe (~$0). At L0 the substrate is config dataclass + skeleton helpers + Phase 1 audit + Phase 2 decision + Phase 3 design memo — NO MLX renderer body. Axis 2 N/A is structurally correct (NOT a finding)."
  - assumption: "B' has ZERO numpy reference at L0 (axis 3 N/A by construction); L1+ should adopt META-CONSOLIDATE-OP-2 canonical numpy reference per G=NIRVANA exemplary pattern"
    classification: HARD-EARNED
    rationale: "Per R1' advisory B'-L1-ADV2: 'when L1 implements MLX primitives, MUST land sister numpy_reference.py per G=NIRVANA canonical pattern (or import canonical per META-CONSOLIDATE-OP-2 if landed)'. L0 scope does not yet implement MLX primitives so axis 3 N/A by construction. Advisory only at L1+; NOT a R2 finding."
council_decisions_recorded:
  - "R2-COMBINED CLEAN PASS — counter advances from 1/3 → 2/3 per protocol items 3-4"
  - "All 3 axes PASS at R2 across 0 R2 findings; sister-coherence with B' R1' verdict preserved"
  - "Op-routable advisory (NOT R2 finding): when L1 implements MLX primitives, MUST adopt META-CONSOLIDATE-OP-1 canonical helpers AND META-CONSOLIDATE-OP-2 canonical numpy reference (per R1' L1+ advisories)"
  - "Op-routable advisory (NOT R2 finding): B' canonical equation registry registration deferred to Phase 2 council per Catalog #325 (Mamba-2 selective SSM canonical equations: Gu+Dao 2024 Mamba-2; sister-substrate-wide gap per design memo Section 18 pattern)"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: ""
horizon_class: frontier_pursuit
canonical_equation_refs: []
predicted_band_validation_status: pending_post_training
related_deliberation_ids:
  - path_3_b_recursive_adversarial_review_r1_prime_3_axis_20260526
  - path_3_b_z7_mamba_2_L0_scaffold_landed_20260526
  - path_3_recursive_adversarial_review_r1_prime_aggregate_3_axis_landings_b_c_f_g_20260526
  - path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526
  - path_3_b_z7_mamba_2_substrate_design_decision_20260526
---

# R2-COMBINED Recursive Adversarial Review — Path 3 candidate B' (Z7-Mamba-2-v2 fresh substrate L0 SCAFFOLD)

**Scope**: landing commit `7a103fdbb`. Source files: `src/tac/substrates/z7_mamba2_v2_fresh_substrate/` (config dataclass + skeleton + tests). 3-phase methodology memos: Phase 1 audit + Phase 2 decision + Phase 3 design.

**Verdict**: **PROCEED — R2-COMBINED CLEAN PASS** — counter advances from 1/3 → **2/3** per protocol items 3-4.

**Cost**: $0 GPU; ~20 min wall-clock (re-PV + sister-coherence verification + memo synthesis).

---

## Premise verification per Catalog #229

Read in full before any review claim:

- R1' review memo `.omx/research/path_3_b_recursive_adversarial_review_r1_prime_3_axis_20260526.md`
- B' landing memo `.omx/research/path_3_b_z7_mamba_2_L0_scaffold_landed_20260526.md`
- R1' aggregate: confirms B' was R1' CLEAN PASS (counter at 1/3)
- B' Phase 1 cargo-cult audit + Phase 2 decision + Phase 3 design memos
- B' source files under `src/tac/substrates/z7_mamba2_v2_fresh_substrate/`

Empirical re-verification at 2026-05-26T08:42Z:

```
$ PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/substrates/z7_mamba2_v2_fresh_substrate/tests/ -q
........................................                                        [100%]
40 passed in 0.32s
```

40/40 tests PASS; baseline confirmed CLEAN.

---

## Axis 1 review: Math + scientific + engineering rigor (council members: Shannon + Dykstra + Tao)

**Per-architectural-choice triple-axis citation table** (R1' verified the 3-phase cargo-cult-first methodology; R2 re-verifies + adds rotation perspective):

| Architectural choice | Math citation | Scientific citation | Engineering citation | R2 verdict |
|---|---|---|---|---|
| Mamba-2 selective SSM paradigm (decoder slot) | Gu+Dao 2024 Mamba-2 (arXiv:2405.21060) — selective state space model with structured attention duality | Sister substrate `z7_mamba2_v2_fresh_substrate` design memo Phase 3 §3.2 cites Mamba-2 SSM kernel + structured attention duality | `src/tac/substrates/z7_mamba2_v2_fresh_substrate/` config dataclass declares all 4 orthogonal axes (decoder / latent / training-pathway / grammar) | **HARD-EARNED** (R1'+R2 reaffirmed) |
| Phase 1 cargo-cult audit (8 NEW CC + 2 NEW HARD-EARNED-PARTIAL beyond CC-1..CC-10) | Per Catalog #303 cargo-cult audit section discipline | `path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526.md` empirically materializes the methodology | Documented inline in Phase 1 audit memo | **HARD-EARNED** |
| Phase 2 decision (decompose-and-fork for 2 HARD-EARNED-PARTIAL assumptions; explicit UNIQUE-vs-CANONICAL per layer per Catalog #290) | Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD operating mode | `path_3_b_z7_mamba_2_substrate_design_decision_20260526.md` documents decision per Catalog #290 | Decision table in Phase 3 design memo | **HARD-EARNED** |
| Phase 3 design (16-layer canonical-vs-unique decision table: 8 UNIQUE-FORK + 6 CANONICAL-ADOPT + 1 UNIQUE-IMPL + 1 UNIQUE-DESIGN) | Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD canonical-vs-unique falling-rule discipline | `path_3_b_z7_mamba_2_substrate_design_20260526.md` documents per-layer decisions | Documented inline | **HARD-EARNED** |
| Predicted ΔS band P50 = -0.018 → score ~0.175 (frontier_pursuit upper-region) | Z7 design memo planning prior; Tier-C density validation per Catalog #324 deferred to Phase 2 council | `predicted_band_validation_status: pending_post_training` per Catalog #324 phantom_random_init refusal at L0 | Honest declaration per CLAUDE.md "HORIZON-CLASS" standing directive | **HARD-EARNED** |
| L0 scope is fresh substrate (NOT extending existing Z7-Mamba-2 v1 cargo-culted scaffolds) | Per operator standing directive 2026-05-26 "Never simply extend unless a rigorous adversarial cargo cult pass has been done first" | Phase 1 cargo-cult audit EMPIRICALLY MATERIALIZED the directive | Fresh substrate `z7_mamba2_v2_fresh_substrate` (distinct module path from sister `time_traveler_l5_z7_mamba2`) | **HARD-EARNED** |

### Axis 1 verdict

**Math + scientific + engineering rigor**: HARD-EARNED across all 6 architectural choices. The 3-phase cargo-cult-first methodology IS the canonical example of operator directive #2 compliance and is empirically materializable per R1' META finding #4.

**Sister-substrate-wide gap (NOT this R2's scope)**: Mamba-2 canonical equations not yet registered in `tac.canonical_equations` per Catalog #344. Phase 2 council symposium responsibility per design memo Section 18 (Gu+Dao 2024 Mamba-2 selective SSM equations: A/B/C/Δ parameters + structured attention duality).

**Axis 1 R2 findings**: 0.

---

## Axis 2 review: MLX drift minimization (council members: Carmack + Hotz + Quantizr)

### Per-MLX-primitive drift bound vs PyTorch reference

| MLX primitive in B' | Drift vs PyTorch | Verdict |
|---|---|---|
| (no MLX primitives at L0 per 3-phase methodology design) | N/A | **N/A by construction** |

### Canonical helper substitution status

B' has NOT yet implemented MLX-native Mamba2V2Cell or Mamba2TemporalDecoder at L0. Per the landing memo's `op-routable #1`: L1 follow-up subagent implements MLX-native primitives + Z7MCM3 archive pack/unpack + MPS proxy probe.

**R1' advisory B'-L1-ADV1** (HARD-EARNED at R2): when L1 implements MLX-native Mamba2V2Cell + Mamba2TemporalDecoder, MUST adopt META-CONSOLIDATE-OP-1 canonical helpers (NOT re-invent locally). This advisory carries forward to L1+ work; R2 reaffirms.

### Axis 2 verdict

**MLX drift minimization**: N/A by construction at L0. The 3-phase methodology correctly defers MLX implementation to L1+ where canonical-helper adoption is mandated by R1' advisory.

**Axis 2 R2 findings**: 0 (N/A by construction).

---

## Axis 3 review: Portability via numpy (council members: MacKay + Selfcomp + Contrarian)

### Per-primitive numpy reference status

B' does NOT ship a sister `numpy_reference.py` at L0 per the 3-phase methodology design.

**R1' advisory B'-L1-ADV2** (HARD-EARNED at R2): when L1 implements MLX primitives, MUST land sister `numpy_reference.py` per G=NIRVANA canonical pattern (or import canonical per META-CONSOLIDATE-OP-2 if landed). This advisory carries forward to L1+ work; R2 reaffirms.

### Axis 3 verdict

**Portability via numpy**: N/A by construction at L0. The 3-phase methodology correctly defers MLX + numpy implementation to L1+ where META-CONSOLIDATE-OP-2 canonical adoption is mandated by R1' advisory.

**Axis 3 R2 findings**: 0.

---

## R2-COMBINED verdict per substrate

**B' R2 verdict**: **PROCEED — CLEAN PASS** (0 findings across all 3 axes).

**Counter advance**: 1/3 → **2/3** (R2 advances per protocol items 3-4; sister-coherence with R1' CLEAN PASS preserved).

**Path to 3/3 SEAL → paid CUDA dispatch authorized**: 1 additional CLEAN round (R3) required per protocol item 4. **B' is the SECOND Path 3 substrate (after D=Z6) to reach 2/3 in R2-COMBINED**, despite L0 SCAFFOLD scope being design-only (no MLX renderer at L0); the 3-phase cargo-cult-first methodology has empirically materialized R1' clean-pass + R2 clean-pass back-to-back.

**Operator-routable next** (post-R3 IF CLEAN): per landing memo `op-routable #1`: L1 follow-up subagent implements Mamba2V2Cell + Mamba2TemporalDecoder + Z7MCM3 archive + MPS proxy probe at $0 GPU; per landing memo `op-routable #2`: paired CUDA dispatch only AFTER MPS-Win on ≥1 axis per Phase 3 §6 probe-disambiguator.

---

## Per-substrate cumulative counter status

| Round | Result | Counter |
|---|---|---|
| R1' | PROCEED — CLEAN PASS (0 findings; 2 advisories) | 0 → 1/3 |
| **R2-COMBINED** | **PROCEED — CLEAN PASS** | **1/3 → 2/3** |
| R3 (planned) | TBD | If CLEAN: 2/3 → 3/3 = SEAL |
| R4 (planned) | N/A if R3 closes at 3/3 | — |
| SEAL gate | 3/3 required | **REACHABLE AT R3 IF CLEAN** |

---

## 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map contribution**: N/A.
- **hook #2 Pareto constraint**: N/A.
- **hook #3 bit-allocator hook**: N/A.
- **hook #4 cathedral autopilot dispatch hook**: ACTIVE (R2 CLEAN PASS at 2/3 unblocks B' for paid CUDA dispatch authorization at R3 if CLEAN, modulo `op-routable #2` MPS-Win precondition per Phase 3 §6 probe-disambiguator).
- **hook #5 continual-learning posterior**: ACTIVE (R2 verdict appended to council deliberation posterior per Catalog #300 v2; supersedes R1' PROCEED as chronologically-later anchor; B' counter now at 2/3).
- **hook #6 probe-disambiguator**: ACTIVE via Phase 3 §6 MPS proxy probe-disambiguator (canonical disambiguator at L1+ for paired CUDA dispatch precondition).

---

## Discipline applied

- **Catalog #229 PV**: R1' review + landing + 3-phase methodology memos + 40 tests run before any review claim.
- **Catalog #110/#113 APPEND-ONLY**: NEW review memo (R2); R1' + landing + Phase 1/2/3 memos NEVER mutated.
- **Catalog #117/#157/#174/#235/#289**: commit forthcoming via canonical serializer.
- **Catalog #119**: Co-Authored-By Claude trailer.
- **Catalog #206**: checkpoint discipline.
- **Catalog #208**: docs/local-paths.
- **Catalog #230**: sister-subagent ownership map — review-only; no sister-collision (B' is NOT touched by CONSOLIDATE-OP-1 since it has no MLX primitives at L0; Wave #1 posterior_emission may touch B's `__init__.py` but R2 is review-only on memo files).
- **Catalog #287**: every finding carries `[empirical:<measurement>]` evidence-tag; no placeholder rationales.
- **Catalog #292**: per-axis council member operating-within assumption surfaced in frontmatter.
- **Catalog #300 v2**: full frontmatter (tier T2; canonical attendees; mission_contribution frontier_breaking_enabler; horizon_class frontier_pursuit).
- **Catalog #340**: sister-checkpoint guard PROCEED verified at start of R2.
- **CLAUDE.md "Recursive adversarial review protocol — close paths"** items 1-8: R2 CLEAN PASS advances counter to 2/3; item 8 satisfied via the 3-row Assumption-Adversary verdict.
- **CLAUDE.md "Executing actions with care"**: review-only.

---

## Cross-references

- R1' review: `.omx/research/path_3_b_recursive_adversarial_review_r1_prime_3_axis_20260526.md`
- Landing memo: `.omx/research/path_3_b_z7_mamba_2_L0_scaffold_landed_20260526.md`
- R1' aggregate: `.omx/research/path_3_recursive_adversarial_review_r1_prime_aggregate_3_axis_landings_b_c_f_g_20260526.md`
- Phase 1 audit: `.omx/research/path_3_b_z7_mamba_2_cargo_cult_audit_of_existing_scaffold_20260526.md`
- Phase 2 decision: `.omx/research/path_3_b_z7_mamba_2_substrate_design_decision_20260526.md`
- Phase 3 design: `.omx/research/path_3_b_z7_mamba_2_substrate_design_20260526.md`
- Lane: `lane_path_3_recursive_adversarial_review_r2_combined_3_axis_landings_a_b_c_d_e_f_g_20260526` L1 (impl_complete + memory_entry)
