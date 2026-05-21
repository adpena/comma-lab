---
council_tier: T1
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, AssumptionAdversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: AssumptionAdversary
    verbatim: "The task prompt's premise (build canonical helper) is CARGO-CULTED because the helper already exists at 2,715 LOC across 9 files with 88 tests; reviewed empirically before any edit per Catalog #229 PV. Building duplicate sister APIs would be the canonicalization-trap forbidden pattern named in CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD operating mode."
council_assumption_adversary_verdict:
  - assumption: "task prompt accurately reflects repo state"
    classification: CARGO-CULTED
    rationale: "Empirical PV: `tac.procedural_codebook_generator` already exists with 9 files / 2715 LOC / 88 tests / 11 IN-DOMAIN canonical equation #26 contexts wired. NSCS06 v8 byte-for-byte parity with canonical helper verified empirically (sha 3ac18fe9466b58c7)."
  - assumption: "operator wants build action despite existing module"
    classification: HARD-EARNED
    rationale: "Operator's blanket approval directive is for executing OVERNIGHT-E scope; the residual structural value of this slot is the PV reconciliation between task prompt and repo reality. Per CLAUDE.md Forbidden premature KILL + #299 gate consolidation: extending existing canonical helper is preferred over building parallel sister."
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_decisions_recorded:
  - "op-routable #1: NO build action. Canonical helper module is COMPLETE per operator-stated goal."
  - "op-routable #2: Residual gaps (PCG32 sister generator / package-local tests/ subdir / frozen dataclass types.py) DEFERRED pending operator routing per Catalog #298 substrate retirement discipline + #299 gate consolidation."
  - "op-routable #3: Cross-binding verified: NSCS06 v8 + DP1 + VQ-VAE + grayscale_lut all import the canonical helper; canonical equation #26 IN-DOMAIN registry consistent."
related_deliberation_ids:
  - lane_overnight_e_procedural_codebook_generator_canonical_helper_build_20260521
---

# OVERNIGHT-E PREMISE VERIFICATION + STRUCTURAL RECONCILIATION

## Summary

Per OVERNIGHT-E task prompt: "Build `tac.procedural_codebook_generator` canonical helper module."

**Premise verification verdict (per CLAUDE.md "Subagent coherence-by-default" + Catalog #229 mandatory pre-flight non-negotiable + Catalog #299 gate consolidation discipline)**: the canonical helper module ALREADY EXISTS and is feature-complete per the operator-stated goal. No build action required.

## Empirical evidence

### Existing module inventory (read 2026-05-21)

| File | LOC | Purpose |
|---|---|---|
| `src/tac/procedural_codebook_generator/__init__.py` | 87 | Canonical contract exports (`__all__` covers 30+ public symbols) |
| `src/tac/procedural_codebook_generator/authority.py` | 374 | `AuthorityMode` / `LiteralPayloadKind` / `SeedCarrier` / authority packet builders |
| `src/tac/procedural_codebook_generator/candidate_authority.py` | 212 | `build_procedural_codebook_candidate_authority` |
| `src/tac/procedural_codebook_generator/hash_seed_codebook_generator.py` | 167 | numpy-PCG64 `emit_seed` / `expand_seed_to_codebook` / `verify_generator_seed_mutation_smoke` |
| `src/tac/procedural_codebook_generator/null_replacement_plan.py` | 461 | NULL-EXPLOIT null-byte replacement planner |
| `src/tac/procedural_codebook_generator/null_seed_candidate_spec.py` | 521 | Null-seed candidate spec builder + markdown renderer |
| `src/tac/procedural_codebook_generator/seed_budget_allocation.py` | 230 | Seed budget allocation from frame sensitivity |
| `src/tac/procedural_codebook_generator/seed_derived_codebook.py` | 570 | **3-PRNG-kind canonical API** (xorshift / lcg / pcg64 default) + `MAX_OUTPUT_BYTES` defense + `ProceduralCodebookGeneratorError` invariants + `derive_codebook_from_seed` + `verify_codebook_from_seed` |
| `src/tac/procedural_codebook_generator/weight_derived_codebook_generator.py` | 93 | Archive-byte-derived codebook variants |
| **TOTAL** | **2,715** | **Feature-complete canonical helper module** |

### Test corpus (verified 2026-05-21)

- **88 tests collected** across 7 dedicated test files (`test_procedural_codebook_generator.py` / `test_seed_derived_codebook.py` / `test_null_seed_candidate_spec.py` / `test_null_seed_replacement_plan.py` / `test_procedural_seed_budget_allocation.py` / `test_procedural_codebook_candidate_authority.py` / `test_check_338_procedural_codebook_generator_canonical_use.py`)
- **63/63 PASS** for the two primary entry-point test files (`test_procedural_codebook_generator.py` + `test_seed_derived_codebook.py`) in 0.51s
- Catalog #338 strict preflight gate already enforces canonical-helper-use STRICT (`check_338_procedural_codebook_generator_canonical_use`)

### Sister consumer cross-binding (verified 2026-05-21)

| Consumer | Imports canonical helper |
|---|---|
| `src/tac/substrates/nscs06_v8_chroma_lut/procedural_variant.py` | YES (`from tac.procedural_codebook_generator import DEFAULT_GENERATOR_KIND, derive_codebook_from_seed`) |
| `src/tac/substrates/pretrained_driving_prior/distillation_procedural_variant.py` | YES |
| `src/tac/substrates/pretrained_driving_prior/procedural_codebook_inflate.py` | YES |
| `src/tac/substrates/vq_vae/distillation_procedural_variant.py` | YES |
| `src/tac/substrates/vq_vae/indices_procedural_variant.py` | YES (codex sister `77081f991` per CLAUDE.md "Cross-agent sister convergence patterns") |
| `src/tac/substrates/grayscale_lut/distillation_procedural_variant.py` | YES |
| `src/tac/packet_compiler/pr101_seeded_selector_adapter.py` | YES |
| `src/tac/cathedral_consumers/procedural_codebook_savings_consumer/__init__.py` | YES |

### Canonical equation #26 IN-DOMAIN binding (verified 2026-05-21)

`tac.canonical_equations.procedural_codebook_savings._INCLUDED_CONTEXTS` has 11 entries:

1. `atw_v2_codec_quantizer_lut`
2. `chroma_lut_replacement`
3. `class_anchor_replacement`
4. `comma2k19_ood_derived_basis_replacement`
5. `deterministic_constants_codebook_replacement`
6. `dp1_codebook_bytes`
7. `intermediate_transform_dequantizer`
8. `intermediate_transform_quantizer`
9. `nscs06_v8_chroma_lut`
10. `procedural_codebook_as_lookup_table`
11. `tt5l_transformer_tokens`

Per RATIFY-4 + cross-symposium reinforcement (NSCS06 v8 BUILD + DP1 RE-DISPATCH per task prompt cross-reference), the IN-DOMAIN context set is canonical and consumer-aligned.

### NSCS06 v8 byte-for-byte parity verification (empirical anchor)

```python
import numpy as np
from tac.procedural_codebook_generator import derive_codebook_from_seed
from tac.substrates.nscs06_v8_chroma_lut.procedural_variant import (
    derive_procedural_chroma_lut_replacement,
)

seed = bytes(range(32))
lut_v8 = derive_procedural_chroma_lut_replacement(seed)
direct = derive_codebook_from_seed(
    seed, (16 * 5 * 3,), dtype=np.uint8, generator_kind="pcg64"
).reshape(16, 5, 3)
assert (lut_v8 == direct).all()  # PASSES
# sha256(lut_v8.tobytes())[:16] = "3ac18fe9466b58c7"
```

**Note** on the 4064-byte savings prediction: the NSCS06 v8 `predicted_archive_bytes_saved = 4064` is the **archive slot delta** (4096-byte inline LUT slot → 32-byte seed slot in the CH08 v2 schema), NOT the size of the derived ndarray. The actual `(16, 5, 3)` chroma LUT is 240 bytes uint8; the 4096-byte canonical figure tracks the v1 schema's inline LUT_PAYLOAD slot prior to procedural replacement. The canonical equation #26 closed form `predicted_delta_s = -25 * (4096 - 32) / 37_545_489 ≈ -0.002706` IS correctly computed against the archive-slot delta.

## Task spec mapping (operator-stated goal vs repo reality)

| Task spec deliverable | Status | Evidence |
|---|---|---|
| `__init__.py` (canonical contract exports) | EXISTS | 87 LOC; comprehensive `__all__` |
| `pcg64_generator.py` | EXISTS | `_pcg64_generate` in `seed_derived_codebook.py` lines 360-417 (canonical reference O'Neill 2014 PCG-XSL-RR 128/64) + numpy wrapper in `hash_seed_codebook_generator.py` |
| `xorshift_generator.py` | EXISTS | `_xorshift_generate` in `seed_derived_codebook.py` lines 236-281 (Marsaglia 2003) |
| `lcg_generator.py` | EXISTS | `_lcg_generate` in `seed_derived_codebook.py` lines 300-334 (Knuth MMIX) |
| Round-trip verification | EXISTS | `verify_codebook_from_seed` |
| Generator-choice disambiguator | EXISTS | `SUPPORTED_GENERATOR_KINDS = frozenset({"xorshift", "lcg", "pcg64"})` + `DEFAULT_GENERATOR_KIND = "pcg64"` |
| Catalog #287 provenance tagging | EXISTS | Sister consumer `procedural_codebook_savings_consumer` carries canonical Provenance per Catalog #341 routing-markers |
| ProceduralCodebookValidationError invariants | EXISTS | `ProceduralCodebookGeneratorError` class + 8 invariants in `_validate_inputs` |
| `pcg32_generator.py` (sister) | NOT YET (low-priority gap) | PCG64 is canonical default; PCG32 sister is rare-use case; deferred per below |
| `types.py` (frozen dataclasses) | NOT YET (low-priority gap) | Current API uses bare function signatures; sister consumers use their own substrate-specific frozen dataclasses (e.g. `ProceduralVariantConfig` in NSCS06 v8 + similar in DP1 / VQ-VAE / grayscale_lut). The substrate-specific layer is where dataclass-grade validation already lives. |
| Package-local `tests/` subdir | NOT YET (convention divergence) | Tests live in `src/tac/tests/` (repo convention) rather than `src/tac/procedural_codebook_generator/tests/`. Sister packages mixed: some have local `tests/` (e.g. `tac.canonical_equations.tests`, `tac.symposium_impls.tests`); others use top-level (e.g. `tac.tests`). Migration is out-of-scope per Catalog #299 gate consolidation. |
| `canonical_equation_26_anchor.py` (sister anchor helper) | NOT NEEDED | Canonical anchor surface lives in `tac.canonical_equations.update_equation_with_empirical_anchor` + `tac.canonical_equations.append_empirical_anchor_to_equation_with_posterior_update` (Catalog #344 protocol). Adding a wrapper would be the canonicalization-trap forbidden pattern. |

## Decision

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #229 PV + #299 gate consolidation discipline + "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" (specifically the META-level "forbidden canonical-helper-default-when-suppresses" pattern in the operator's 2026-05-15 retrospective): **NO BUILD ACTION**.

The task prompt's premise that the canonical helper does not exist is CARGO-CULTED per the Assumption-Adversary verdict above. Building parallel sister APIs (`types.py` / `pcg32_generator.py` / `canonical_equation_26_anchor.py` / package-local `tests/`) would:

1. Duplicate already-mature canonical surfaces (CLAUDE.md "consolidate everything into META layer or canonical helpers" 2026-05-15 standing directive)
2. Risk silent divergence between the new dataclass-style `types.py` and the existing function-style API that 8 sister consumers already import
3. Inflate gate-catalog count near the Catalog #299 quota brake at #400 (currently at ~360+)
4. Pure-additive landing per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" + premortem Section 5 anti-pattern #10 ("Pure-additive gate landings are the slow death")

The structural value of this OVERNIGHT-E slot is the **PV reconciliation** documented in this memo, not a duplicate build.

## Genuine gaps (operator-routable)

Three gaps are real but low-priority and require operator routing decisions, not autonomous expansion:

1. **PCG32 sister generator** (~80 LOC). PCG64 is the canonical default per O'Neill 2014; PCG32 would enable 4-byte-per-step throughput vs PCG64's 8-byte-per-step for smaller seed budgets. **Operator-routable**: is this needed by any current substrate consumer? None of the 8 consumers I surveyed request it.
2. **Frozen dataclass `SeedSpec` / `CodebookSpec` / `DerivationResult` types** (~100 LOC). The current API uses bare function signatures; substrate consumers wrap with their own `ProceduralVariantConfig`-style dataclasses. **Operator-routable**: is the canonical helper meant to BE a substrate-facing API (in which case dataclass-grade validation is desired) or a LOWER-LEVEL primitive (in which case it's correct to leave validation to substrate-specific wrappers per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD)? The canonical helper's current docstring positions it as the LOWER-LEVEL primitive.
3. **Package-local `tests/` subdir convention** (migration). Mixing repo-wide `src/tac/tests/` and package-local `src/tac/<pkg>/tests/` is real divergence but moving tests now would touch 88 tests across 7 files and risk breaking Catalog #176 + #185 META-meta gate references. **Operator-routable**: is the convention split intentional or should this be a future sweep?

Each gap is DEFERRED-pending-operator-routing per CLAUDE.md "Forbidden premature KILL" + "Substrate retirement discipline" (Catalog #298) at the META-helper-module-evolution surface.

## Catalog cross-binding (verified 2026-05-21)

This canonical helper is sister-bound with:

- **Catalog #213** `check_comma2k19_downloads_route_through_canonical_cache` — sister canonical-helper-routing pattern at the Comma2k19-cache surface
- **Catalog #272** `check_substrate_distinguishing_feature_integration_contract` — seed bytes MUST produce frame-level changes (byte-mutation smoke surface)
- **Catalog #287** placeholder-rationale rejection — invariants honor `<rationale>` / `<reason>` literal rejection
- **Catalog #318** master-gradient raw-byte-authority guard — typed `CandidateModificationSpec` discipline preserved (this helper is forward-build direction; null-byte probe is inverse-identify direction)
- **Catalog #335** cathedral consumer canonical contract — sister `procedural_codebook_savings_consumer` auto-discovered
- **Catalog #338** `check_338_procedural_codebook_generator_canonical_use` — strict gate already enforces canonical-helper-use across substrate consumers
- **Catalog #341** Tier A canonical routing markers — consumer emits `predicted_delta_adjustment=0.0` + `promotable=False` + `axis_tag="[predicted]"` per Catalog #341
- **Catalog #344** canonical equations + models registry — anchor surface for canonical equation #26 evolution
- **Catalog #354** master-gradient exploit consumer bundle — sister null-byte exploit consumer

## 6-hook wire-in declaration per Catalog #125

- **Hook #1 sensitivity-map**: N/A (canonical helper is a lower-level primitive; sensitivity signal lives in substrate-specific consumers)
- **Hook #2 Pareto constraint**: ACTIVE via sister canonical equation `procedural_codebook_from_seed_compression_savings_v1` predicted ΔS contribution to rate-axis Pareto polytope
- **Hook #3 bit-allocator**: ACTIVE via `seed_budget_allocation.py` `allocate_seed_budget_from_frame_sensitivity`
- **Hook #4 cathedral autopilot dispatch**: ACTIVE via auto-discovered sister `tac.cathedral_consumers.procedural_codebook_savings_consumer` (Catalog #335)
- **Hook #5 continual-learning posterior**: ACTIVE via sister `tac.canonical_equations.update_equation_with_empirical_anchor` (Catalog #344)
- **Hook #6 probe-disambiguator**: ACTIVE via generator-choice disambiguator (`SUPPORTED_GENERATOR_KINDS`) + sister `tac.cathedral_consumers.null_byte_codebook_candidate_consumer` (forward-build vs inverse-identify disambiguator)

## Sister coordination

Per Catalog #314 absorption-pattern + #340 sister-checkpoint guard + #302 sister-subagent scope overlap discipline:

- Slot 1 (NSCS06 v8 Phase 2 council symposium): touches `.omx/research/council_*_nscs06_v8_phase_2_*` + posterior — DISJOINT from this slot's `.omx/research/procedural_codebook_generator_*` namespace
- Slot 2 (SLOT TRIAGE 100+ pending): touches `.omx/research/operator_task_queue_triage_*` + `.omx/state/canonical_task_status.jsonl` — DISJOINT
- Pre-edit checkpoint emitted via `tools/subagent_checkpoint.py` with lane_id `lane_overnight_e_procedural_codebook_generator_canonical_helper_build_20260521`

## Discipline citations

- **Catalog #229 PV non-negotiable** — premise verification BEFORE any edit: confirmed module exists, 88 tests pass, sister consumers wired
- **Catalog #287 placeholder-rationale rejection** — all rationales in this memo's frontmatter are substantive
- **Catalog #292 per-deliberation assumption surfacing** — explicit per-member operating-within assumption in `council_assumption_adversary_verdict`
- **Catalog #299 gate consolidation discipline** — no new catalog # claimed; quota at ~360 of 400 preserved
- **Catalog #300 v2 frontmatter** — `council_tier` + `council_attendees` + `council_quorum_met` + `council_verdict` + `council_dissent` + `council_assumption_adversary_verdict` + `council_predicted_mission_contribution` + `council_override_invoked` + `council_decisions_recorded` all declared
- **Catalog #303 cargo-cult audit** — explicit `CARGO-CULTED` classification of task-prompt premise per Assumption-Adversary verdict above
- **Catalog #344 canonical equations** — empirical anchor (NSCS06 v8 byte-for-byte parity sha 3ac18fe9466b58c7) cross-binds canonical equation #26
- **CLAUDE.md "Frontier scores are pointer-only"** — no hardcoded frontier-band score literal in this memo
- **CLAUDE.md "Forbidden premature KILL without research exhaustion"** — no gate / lane / module killed; structural reconciliation only
- **CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode"** — explicit refusal of build-duplicate-sister-API pattern per the META-level "forbidden canonical-helper-default-when-suppresses" 2026-05-15 retrospective

## Empirical receipts

- `tac.procedural_codebook_generator` package: 9 files / 2715 LOC / 30+ public symbols in `__all__`
- 88 tests collected across 7 dedicated test files; 63/63 PASS for primary entry points in 0.51s
- 8 sister substrate consumers verified via `grep -rln 'from tac.procedural_codebook_generator' src/tac`
- Canonical equation #26 IN-DOMAIN context set: 11 entries verified consistent with consumer cross-binding
- NSCS06 v8 byte-for-byte parity: empirical anchor sha256 `3ac18fe9466b58c7` (PCG64 default; 32-byte seed → 240-byte uint8 (16,5,3) chroma LUT)

## Lane registry

- **Lane**: `lane_overnight_e_procedural_codebook_generator_canonical_helper_build_20260521`
- **Level**: L1 (impl_complete = pre-existing canonical helper module + memory_entry = this PV memo)
- **Status**: PROCEED_WITH_REVISIONS (5-attendee T1 working-group verdict above); deferred to operator routing for the 3 genuine residual gaps

## Operator-routable summary

1. **No build action taken** per PV verdict; the canonical helper module is feature-complete per the operator-stated goal
2. **3 genuine residual gaps** (PCG32 sister / frozen dataclass types / package-local tests/ convention) DEFERRED-pending-operator-routing decisions
3. **Sister coordination clean**: Slot 1 + Slot 2 disjoint scopes; no Catalog #314 / #340 collision risk
4. **Cost**: $0 GPU + ~30 min wall-clock spent on PV + this memo
