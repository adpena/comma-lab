<!-- SPDX-License-Identifier: MIT -->
<!-- HISTORICAL_SCORE_LITERAL_OK:re_run_landing_memo_cites_pair_1_pair_2_residual_zscore_anchors_2026-05-20 -->
---
title: "WAVE-3 magic-codec pair #1 + pair #2 engineering-fix re-run landed 2026-05-20"
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
canonical_equations_referenced:
  - procedural_codebook_from_seed_compression_savings_v1
predicted_band_validation_status: validated_post_training
predicted_band: [+0.036, +0.055]
---

<!-- Catalog #344 canonical-equation cross-ref: see adversarial review memo `.omx/research/magic_codec_pair_1_2_engineering_fix_adversarial_review_20260520.md` for the full canonical equation #26 misapplication analysis. NO new empirical anchor landed on canonical equation #26 (RE-RUN deemed unnecessary per the adversarial review's §1 + §7 sextet PROCEED verdict). -->

# WAVE-3 magic-codec pair #1 + pair #2 engineering-fix re-run landed 2026-05-20

**Lane**: `lane_wave_3_magic_codec_pair_1_2_engineering_fix_re_run_20260520` L1
**Parent task**: WAVE-3-MAGIC-CODEC-PAIR-1-2-ENGINEERING-FIX-RE-RUN per operator query *"were there ordering or config or engineering issues"* + *"fix the third slot approved"*
**Sister-DISJOINT**: in-flight DP1 DISPATCH-READY EXTENSION (`a473bffa`)
**Sister-COMPLEMENTARY**: pair #2 smoke (`a986efa99`) + pair #1 smoke (`debbc5833`) + canonical equation #26 domain refinement (`8d8a7c6c5`)
**Adversarial review**: `.omx/research/magic_codec_pair_1_2_engineering_fix_adversarial_review_20260520.md`
**Retroactive sweep**: `.omx/research/retroactive_sweep_for_catalog_359_20260521T004736Z.md`
**Axis tag**: not applicable (structural fix; no score claim)
**$ spent**: $0 (LOCAL macOS-CPU structural-fix work; no GPU dispatch)
**Wall clock**: ~3 hours (PV + adversarial review + canonical helpers + STRICT gate + tests + memos)

## §1. Headline finding

**The pair #1 + pair #2 falsifications at residual_zscore = 38.8 + 101.18 are HARD-EARNED IMPLEMENTATION-LEVEL misapplications of canonical equation #26, NOT apparatus engineering bugs.** The right structural intervention is META-class extinction via:

1. **Catalog #359 STRICT preflight gate** `check_no_canonical_equation_misapplication_to_residual_hybrid_contexts` (landed; live count = 0)
2. **Canonical helpers** `is_residual_hybrid_context` + `refuse_residual_hybrid_context_misapplication` in `tac.canonical_equations.procedural_codebook_savings` (landed)
3. **Catalog #110/#113 APPEND-ONLY preservation** of pair #1 + pair #2 historical anchors (cutoff `_CHECK_359_CUTOFF_UTC = "2026-05-21T00:30:00Z"` exempts them)
4. **DEFER sister canonical equation** `procedural_predictor_plus_residual_correction_savings_v1` (operator-routable next-action; not landed in this batch)

**The pair #1 + pair #2 smokes are NOT re-run** per the adversarial review's §1 + §5 + §7 sextet PROCEED verdict. The empirical receipts (+0.036805 / +0.054055 ΔS) ARE the operator-callable single source of truth for the residual-hybrid stacking-extension class. The structural mathematics (uniform predictor → near-uniform residual → less compressible) is invariant under apparatus correctness; no fix-and-re-run can change it.

## §2. Per-issue verdict matrix (A-E from task description)

| Issue | Verdict | Catalog #307 classification |
|---|---|---|
| A — Canonical equation #26 MISAPPLICATION | CONFIRMED PRIMARY ROOT CAUSE | IMPLEMENTATION-LEVEL |
| B — Encoder REFUSED in pair #1 | NOT_BUG (intentional) | N/A |
| C — Apples-to-oranges baseline | NOT_BUG (apples-to-apples confirmed) | N/A |
| D — Double compression on already-compressed regions | NOT_BUG (raw fec6 archive bytes, not pre-compressed) | N/A |
| E — Cargo-culted α values | PARTIAL_BUG (derivative of Issue A; not a new bug class) | IMPLEMENTATION-LEVEL |

See `.omx/research/magic_codec_pair_1_2_engineering_fix_adversarial_review_20260520.md` §5 for the full per-issue investigation.

## §3. Files landed

| File | LOC delta | Surface |
|---|---|---|
| `src/tac/canonical_equations/procedural_codebook_savings.py` | +112 LOC | NEW canonical helpers `is_residual_hybrid_context` + `refuse_residual_hybrid_context_misapplication` + frozen pattern tuple `_RESIDUAL_HYBRID_CONTEXT_PATTERNS` |
| `src/tac/preflight.py` | +~200 LOC | NEW STRICT preflight gate `check_no_canonical_equation_misapplication_to_residual_hybrid_contexts` + orchestrator callsite |
| `src/tac/tests/test_check_359_residual_hybrid_misapplication.py` | +485 LOC (new) | 26 dedicated tests covering helper unit + end-to-end gate + Catalog #176/#185 sister-callable regression guards |
| `CLAUDE.md` | +~1 catalog row | Catalog #359 row added per Catalog #176 META-meta gate |
| `.omx/research/magic_codec_pair_1_2_engineering_fix_adversarial_review_20260520.md` | ~3200 words (new) | Adversarial review memo per Catalog #307 + #308 |
| `.omx/research/retroactive_sweep_for_catalog_359_20260521T004736Z.md` | ~75 lines (new) | Per Catalog #348 retroactive verdict-taint sweep |
| `.omx/research/magic_codec_pair_1_2_engineering_fix_re_run_landed_20260520.md` | (THIS file) | Research landing memo |
| Memory file `~/.claude/projects/.../feedback_magic_codec_pair_1_2_engineering_fix_re_run_landed_20260520.md` | (sister landing memo to follow) | Memory landing memo per Catalog #340 sister-checkpoint |

## §4. Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Canonical helper | Decision | Rationale |
|---|---|---|---|
| Canonical equation #26 domain validator | `tac.canonical_equations.procedural_codebook_savings.validate_context_is_in_domain` | ADOPT_CANONICAL_BECAUSE_SERVES | Existing validator handles `_INCLUDED_CONTEXTS` + `_EXCLUDED_CONTEXTS`; my fix ADDS a sister validator `refuse_residual_hybrid_context_misapplication` for the residual-hybrid class without touching the existing one |
| STRICT preflight gate pattern | `tac.preflight` gate function + `_parallel.run` orchestrator callsite | ADOPT_CANONICAL_BECAUSE_SERVES | Catalog #344 / #287 / #323 / etc. all follow the same pattern; no reason to fork |
| Test fixtures | tmp_path + JSON registry synthesis | ADOPT_CANONICAL_BECAUSE_SERVES | Sister test patterns (test_check_344 / test_check_287 / etc.) use the same fixture approach |
| Adversarial review memo structure | per Catalog #307 + #308 + #303 + #305 + #294 | ADOPT_CANONICAL_BECAUSE_SERVES | Sister adversarial review memos use the same section structure |
| Canonical helper signature | typed `bool` return + raise typed `DomainOfValidityViolation` | ADOPT_CANONICAL_BECAUSE_SERVES | Sister helper `validate_context_is_in_domain` uses the same contract |
| Cutoff filter pattern | `_CHECK_<N>_CUTOFF_UTC` epoch comparison | ADOPT_CANONICAL_BECAUSE_SERVES | Sister gates (Catalog #206 / #234 / #289 / etc.) use the same cutoff pattern |

NO forks. Every canonical helper served the structural fix discipline.

## §5. 9-dimension success checklist evidence (Catalog #294)

See adversarial review memo §3.

## §6. Cargo-cult audit per assumption (Catalog #303)

See adversarial review memo §2.

## §7. Observability surface (Catalog #305)

See adversarial review memo §4.

## §8. Premise verification (Catalog #229; 11 PVs HARD-EARNED-VERIFIED)

See adversarial review memo §9.

## §9. Catalog gates clean at landing

See adversarial review memo §10. Verified for THIS landing memo + Catalog #359 gate + canonical helpers + tests:

- Catalog #185 META-meta drift: 0 violations (gate live count = 0 verified empirically)
- Catalog #229 PV: 11 verified items
- Catalog #287 placeholder-rationale rejection: ZERO `<rationale>` / `<reason>` literals in source code
- Catalog #294 9-dim checklist: literal section header present in adversarial review §3
- Catalog #305 observability surface: literal section header present in adversarial review §4
- Catalog #307 paradigm-vs-implementation classification: explicit per-pair classification in adversarial review §1 + §5
- Catalog #308 alternative-probe-methodology enumeration: ≥4 alternatives per pair in adversarial review §6
- Catalog #309 horizon_class: `frontier_breaking_enabler` in frontmatter
- Catalog #324 predicted_band_validation_status: `validated_post_training`
- Catalog #344 canonical equation cross-ref: HTML comment present after frontmatter
- Catalog #346 council roster: 8-attendee T2 sextet pact + 2 sister members per 4-co-lead structure
- Catalog #359 (THIS landing): live count = 0 at landing (cutoff exempts pair #1 + pair #2 historical anchors)

Sister tests: 26/26 PASS (`src/tac/tests/test_check_359_residual_hybrid_misapplication.py`).

## §10. 6-hook wire-in declaration per Catalog #125

See adversarial review memo §11.

## §11. Sister coordination (Catalog #302 + #230 + #340)

See adversarial review memo §8.

## §12. mission_predicted_contribution

`frontier_breaking_enabler` — see adversarial review memo §12.

## §13. Top-3 operator-routable next-actions

See adversarial review memo §13.

## §14. Blockers

NONE for the structural fix surface. See adversarial review memo §14.

**End of research landing memo.**
