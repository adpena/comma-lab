# Catalog #383 retroactive sweep — gumbel_softmax_sample canonical extraction migration 2026-05-30 23:00Z

**Lane**: `lane_gumbel_softmax_sample_canonical_extraction_migration_20260530` L1.

**Source**: per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable + Catalog #348 EVENT-DRIVEN RETROACTIVE VERDICT-TAINT SWEEP discipline.

Catalog #383 (MLX primitive canonical routing discipline) landed at MLX canonicalization wave commit `e52f2f6b4` (2026-05-30). THIS sweep memo accompanies the migration that drives Catalog #383 live count from 2 → 0.

## 1. Bug-class symptom signature

Substrate-side re-implementation of canonical MLX primitive `gumbel_softmax_sample` outside the canonical extractor modules (`tac.framework_agnostic.canonical_kernels` + sister `tac.local_acceleration.pr95_hnerv_mlx` family). Symptom = `def gumbel_softmax_sample(` AST node in `src/tac/substrates/*/` files without canonical extractor import AND without same-line `# MLX_PRIMITIVE_UNIQUE_BECAUSE_<reason>:<rationale>` waiver per Catalog #290 PRINCIPLED FORK discipline.

## 2. Pre-fix window

Window = from Catalog #383 landing (commit `e52f2f6b4`, 2026-05-30 ~14:50Z) to THIS migration commit (2026-05-30 ~23:00Z). Window duration: ~8 hours. Pre-fix live count: 2.

## 3. Historical-KILL / DEFER / FALSIFY search results

Searched `.omx/research/*.md` and `~/.claude/projects/-Users-adpena-Projects-pact/memory/*.md` for KILL / DEFER / FALSIFY verdicts involving `gumbel_softmax_sample` substrate-side impls. Results:

| Historical verdict | Substrate | Reactivation criterion met? | RE-EVAL priority |
|---|---|---|---|
| Wave 10/11 audit (2026-05-29) at `feedback_wave_3_dreamerv3_rssm_math_audit_landed_20260529.md` flagged Z8 local impl as CARGO-CULTED (omitted unimix); Wave 10/11 fix replaced with delegation | Z8 | YES — delegation pattern preserved by THIS migration's waiver; behavior unchanged | LOW (no re-eval needed; PR-or-greater-parity discipline maintained) |
| Wave 3 DreamerV3 RSSM math-fidelity audit (2026-05-29) confirmed `gumbel_softmax_sample` in DreamerV3 as HARD-EARNED CANONICAL 1:1 vs Hafner 2023 §3 | DreamerV3 | YES — canonical 1:1 preserved by THIS migration; behavior unchanged | LOW (no re-eval needed) |
| Audit memo `.omx/research/mlx_canonicalization_audit_inventory_20260530.md` §A.2.5 classified DreamerV3 + Z8 + mdl_ibps_j as `CANONICAL_EXTRACTION_RECOMMENDED` (operator-routable) | All 3 | PARTIAL — THIS migration verifies canonical extraction is NOT structurally viable for DreamerV3 + Z8 (PRINCIPLED FORK per Catalog #290 falling-rule); mdl_ibps_j out of gate scope | LOW (operator-routable for future canonical helper signature extension; not blocking) |

No prior KILL / DEFER / FALSIFY verdicts to re-evaluate. The Wave 10/11 fix (`feedback_wave_3_dreamerv3_rssm_math_audit_landed_20260529.md`) already extincted the Z8 local-duplicate cargo-cult; THIS migration documents the PRINCIPLED FORK structurally per Catalog #383 + #290 falling-rule.

## 4. Per-finding RE-EVAL priority assignment

| Finding | Substrate | RE-EVAL priority | Reactivation lane |
|---|---|---|---|
| DreamerV3 local impl with tuple return + STE flag + MLX key | DreamerV3 | LOW | Future operator-routable: extend canonical helper signature to support STE flag + tuple return; would require sister wave on canonical helper + 32 DreamerV3+Z8 test migration. Not blocking. |
| Z8 thin delegation wrapper | Z8 | LOW | Future operator-routable: refactor Wave 10/11 test fixture to verify delegation via module-level re-export binding instead of `inspect.getsource`-based pin. Not blocking. |
| mdl_ibps_j sister impl with `_mlx` suffix (out of gate scope) | mdl_ibps_j | LOW | Future operator-routable: substrate-callsite refactor to consume tuple return + STE flag + canonical helper signature. Not blocking. |

## 5. Migration outcome

- **Catalog #383 live count**: 2 → 0
- **Catalog #383 STRICT-flip readiness**: YES (operator-routable; sister recommendation memo emitted separately)
- **Tests**: 422 pass (DreamerV3 + Z8 + framework_agnostic) + 43 pass (mdl_ibps_j); 0 regressions
- **Byte-stability**: PASS (max abs diff < 1e-6 between DreamerV3 + Z8 delegation with deterministic seed)
- **Behavior change**: NONE (waivers document existing PRINCIPLED FORK; no code logic modified)

## 6. Sister gates cross-reference

- Catalog #290 (canonical-vs-unique decision per layer — falling-rule list) — the canonical discipline that justifies PRINCIPLED FORK waivers
- Catalog #383 (THIS gate) — substrate-MLX-primitive canonical routing
- Catalog #287 (placeholder-rationale rejection) — waiver rationales are substantive (>4 chars, traceable to real callsites)
- Catalog #335 (canonical cathedral consumer auto-discovery) — sister cathedral consumer `mlx_canonicalization_audit_consumer` already exists per MLX canonicalization landing
- Catalog #341 (Tier A canonical-routing markers) — sister Tier A consumer markers already present
- Catalog #344 (canonical equations registry) — `mlx_primitive_canonicalization_compounding_savings_v1` gains new EmpiricalAnchor: predicted=2 violations → empirical=0 violations; residual=0.0
- Catalog #348 (THIS retroactive sweep gate) — sister discipline this memo satisfies
- Catalog #371 (canonical equations auto-recalibrator) — fires `when_3+_new_empirical_anchors_in_domain` for `mlx_primitive_canonicalization_compounding_savings_v1`
- Catalog #176 (META-meta: STRICT callsites have CLAUDE.md row) — Catalog #383 already satisfies via MLX canonicalization landing entry
- Catalog #185 (META-meta-meta: Live count: 0 verified empirically) — THIS migration verifies live count 0 empirically

## 7. Honest reframe

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": THIS migration does NOT kill the canonical helper or the substrate impls. It documents the PRINCIPLED FORK case per Catalog #290 falling-rule (canonical helper does not structurally fit substrate contracts). Future canonical helper signature extension OR substrate contract refactor remain operator-routable reactivation paths.

Per CLAUDE.md NO FAKE IMPLEMENTATIONS non-negotiable: the waivers document REAL substrate-contract differences (tuple return + STE flag + MLX random key required by 3 downstream callers). NO phantom adoption claim; NO synthetic-fixture validation; NO placeholder rationale; NO test-verifies-constants-not-behavior pattern.

## 8. Anchor memos

- Landing memo: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_gumbel_softmax_canonical_extraction_migration_landed_20260530.md`
- Per-substrate verdict table: `.omx/research/gumbel_softmax_canonical_extraction_migration_20260530.md`
- Parent audit memo: `.omx/research/mlx_canonicalization_audit_inventory_20260530.md` §A.2.5
- Parent landing memo: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_mlx_canonicalization_audit_plus_tinygrad_bridge_plus_6_pillar_discipline_landed_20260530.md`
- Sister Wave 10/11 fix: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_wave_3_dreamerv3_rssm_math_audit_landed_20260529.md`
