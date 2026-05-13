# CANON-1 Wave B + C — Consolidated CLOSURE memo for already-addressed decisions 2026-05-12

**Source ledger**: `.omx/research/canonicalization_dedup_oss_rigor_ledger_20260512.md`
(15 total CANON-1 decisions; this memo closes 7 of them).

**Lane**: `lane_canon_b_c_remaining_decisions_20260512` (L1, Phase 2).

**Scope**: Document the closure status of every CANON-1 decision the
WAVE-CANON-B-C subagent does NOT directly implement — either because a sister
subagent already landed it, the recommended action is NO-ACTION, or operator-
discretion deferral is the standing decision.

## Per-decision closure table

| ID | Decision | Status | Closure rationale |
|---|---|---|---|
| CANON-1.A | Substrate tradition fragmentation | LANDED (WAVE-A-2 d7574355) | Option C explicit-taxonomy resolution + 9 TRADITION 2 inventory rows + xray classifier wire-in |
| CANON-1.B | Canonical scorer-loss helper + 14 sister migrations | LANDED (WAVE-A-1 4f6896ad) | `tac.substrates._shared.score_aware_loss_real_scorer_test_kit` + 14 sister substrate regression tests; Catalog #164 STRICT @ 0 |
| CANON-1.C | SIREN dual-implementation | CLOSED — see this memo §1 | Already documented in residual_basis/__init__.py + substrates/__init__.py docstrings (residual = sidecar; full substrate = renderer replacement) |
| CANON-1.D | Hand-curated list audit | LANDED (this session 1f188d75) | META audit memo at `.omx/research/canon_1_d_hand_curated_list_audit_20260512.md` — 4 HIGH candidates DEFERRED-pending-operator |
| CANON-1.E | Transactional catalog claim | LANDED (this session 6d15a46c) | `tools/claim_catalog_number.py --commit-via-serializer` + 4 new tests |
| CANON-1.F | Catalog #125 lexical scanner | CLOSED-already-addressed (see `canon_1_f_lexical_scanner_closure_20260512.md`) | Current implementation already accepts all 3 spelling forms + informal aliases |
| CANON-1.G | Modal upload-race policy (D2) | LANDED (FIX-I 71f8ffa9) | Catalog #165 STRICT @ 0 via mtime-stability check |
| CANON-1.H | tac.sensitivity_map axis-level reweighting | LANDED (COUNCIL-A1 48ee9201) | `src/tac/sensitivity_map/axis_weights.py` |
| CANON-1.I | lane_g_v3 L3 promotion via GHA contest_cpu | LANDED-WORKFLOW-pending-trigger (WAVE-8-GHA 31854ecc) | GHA workflow `.github/workflows/contest_cpu_eval.yml` ready; awaits operator-side workflow_dispatch trigger to produce the empirical anchor that flips lane_g_v3.contest_cpu gate |
| CANON-1.J | Substrate + CompressAI canonical_inventory wire-in | LANDED (WAVE-A-2 d7574355) | 9 TRADITION 2 rows + 6 substrate-class allowlist entries + cost-table wire-in |
| CANON-1.K | Catalog text drift remaining ~15 entries | CLOSED-at-0 — see this memo §2 | Current `check_claude_md_catalog_text_matches_preflight_strict_value(strict=True)` returns 0 violations |
| CANON-1.L | 230 phantom Vast.ai tracker entries cleanup | LANDED (WAVE-8-VASTAI-CLEAN) | 230 → 0 records per `feedback_wave8_vastai_phantom_cleanup_LANDED_20260512.md` |
| CANON-1.M | recovered_*/ body-cleavage (~106 MB) | DEFERRED-pending-operator (WAVE-8-RECOVERED-CLEANUP eddf4fd0) | Audit + GC plan ledger landed; 0 directories eligible for body-cleavage without operator approval (all contain git-tracked HISTORICAL_PROVENANCE artifacts) |
| CANON-1.N | 23 remaining untagged constants | CLOSED-NO-ACTION — see this memo §3 | Per Wave 2/H recommendation, alarm-fatigue risk dominates incremental tagging value; documented closure |
| CANON-1.O | PR mining further expansion (PR110-115+) | DEFERRED-operator-discretion — see this memo §4 | Subagent 7 completed PR81-104; further PR mining is operator-discretion, not blocking |

## §1 — CANON-1.C SIREN/cool_chic/wavelet dual-implementation explicit closure

**Verification 2026-05-12** at SHA `1f188d75`:

`src/tac/residual_basis/__init__.py:1-52` declares the package as
SCAFFOLD-level (Level 0 SKETCH) residual-basis lanes operating on PR106-family
decoded outputs. The docstring explicitly states each module emits
`score_claim=False`, `promotion_eligible=False`,
`ready_for_exact_eval_dispatch=False` in its result manifest.

`src/tac/substrates/__init__.py:34-141` declares the explicit "Substrate
implementation traditions (CANON-1.A taxonomy)" section enumerating both
TRADITION 1 (`src/tac/substrates/<name>/`) and TRADITION 2
(`src/tac/<name>_as_renderer.py`), with literature citations per family.

Per CANON-1.C recommendation: residual_basis modules are PRIMITIVES
(sidecar/compositional layering) while substrates are RENDERERS
(canvas/replacement). This taxonomy is documented in BOTH `__init__.py`
docstrings + the canonical taxonomy memo at
`.omx/research/substrate_tradition_taxonomy_20260512.md` (landed in WAVE-A-2).

**No further docstring or code work is required for CANON-1.C.** The dual-
implementation is preserved with explicit semantics; future subagents reading
either `__init__.py` will not confuse residual-basis with full-substrate work.

## §2 — CANON-1.K catalog text drift CLOSED-at-0

**Empirical verification 2026-05-12**:

```python
from tac.preflight import check_claude_md_catalog_text_matches_preflight_strict_value
result = check_claude_md_catalog_text_matches_preflight_strict_value(strict=False, verbose=False)
# result == [] (0 violations)
```

The 8 entries WAVE-A-2 fixed in the additive 2026-04-27 A-L block + the
remaining ~15 entries flagged by UUU audit are all CLOSED. Catalog #159
STRICT enforces the invariant going forward; any future text drift is
caught at preflight time.

**CANON-1.K is CLOSED. No bulk text-fix pass needed.**

## §3 — CANON-1.N 23 untagged constants — NO-ACTION

Per Wave 2/H recommendation surfaced in the original CANON-1 ledger row:

> NO-ACTION on remaining (alarm-fatigue risk) OR scanner-improvement pass.

The cost-benefit analysis: tagging additional constants past the top 20
gives diminishing return on review-signal-to-noise. The 23 remaining are
edge-case constants whose non-tagging does not block any STRICT preflight
check.

**Operator override available**: if alarm fatigue is NOT a concern and
operator wants exhaustive tagging, the path is a single-subagent pass that
(a) enumerates remaining untagged constants via AST grep, (b) tags each
with appropriate evidence-axis label, (c) runs Catalog #D regression. ~30
LOC tool + ~50 manual taggings.

**Default recommendation**: KEEP NO-ACTION. The tagged top-20 covers the
high-leverage constants; lower-tier constants are auto-defaulted to
`[advisory only]` per the existing tagging convention.

## §4 — CANON-1.O PR mining DEFERRED-operator-discretion

Per Subagent 7 prior landing (`feedback_public_pr_mining_expansion_pr50_80_pr105_115_landed_20260512.md`):
20 typed mechanism rows from 10 un-mined PRs (PR53/56/60/63/64/65/67/79/104/105)
landed; top-5 EV/byte at PR106 r2 includes 3 NEW pose codecs.

Further expansion (PR110-115+) is **operator-discretion**, not blocking on any
in-flight wave. Per CLAUDE.md "limit subagent use" + "Match the scope of your
actions to what was actually requested": the subagent does NOT proactively
mine additional PRs without explicit operator request.

**Reactivation criteria**: operator explicitly requests "mine PR110-115" or
"complete PR mining for the un-mined corpus."

## Operator decisions surfaced

1. **CANON-1.I lane_g_v3 L3 promotion**: trigger the GHA workflow at
   `.github/workflows/contest_cpu_eval.yml` to produce the [contest-CPU GHA
   Linux x86_64] anchor. Once it lands, mark `lane_g_v3.contest_cpu` gate
   satisfied — first L3 lane in the registry.

2. **CANON-1.M recovered_*/ cleavage**: per WAVE-8-RECOVERED-CLEANUP
   (eddf4fd0) deferral, operator decides whether to (a) apply the GC plan
   with manual per-instance review, (b) keep the 106 MB indefinitely, or
   (c) explicitly waive the HISTORICAL_PROVENANCE artifacts for cleavage.

3. **CANON-1.D follow-ups**: operator approves a follow-up subagent (post-
   cap-reset) to land HCL-1 (`tools/generate_catalog_md_table.py`) as the
   highest-leverage canonicalization. Other 3 HIGH candidates can follow
   sequentially or in parallel.

4. **CANON-1.O PR mining**: explicitly request expansion to PR110-115 if
   the operator wants the un-mined corpus closed. NOT done by default.

5. **CANON-1.N untagged constants**: keep NO-ACTION (default) or override
   with explicit "complete the tagging" directive.

## 6-hook wire-in declaration (per Catalog #125)

This is a META closure memo, not an empirical landing. All 6 hooks N/A:

1. **Sensitivity-map**: N/A — META consolidation memo.
2. **Pareto constraint**: N/A — no feasibility region change.
3. **Bit-allocator**: N/A — no per-tensor importance change.
4. **Cathedral autopilot dispatch**: N/A — META consolidation.
5. **Continual-learning posterior**: N/A — no empirical anchor.
6. **Probe-disambiguator**: N/A — closures of single-interpretation decisions.

## Forbidden patterns honored

- ZERO `/tmp` paths.
- ZERO score claims (CANON-1.I closure references the WORKFLOW landing, not
  a contest-CPU anchor; the anchor itself awaits operator GHA trigger).
- ZERO MPS-derived strategic decisions.
- ZERO KILL verdicts (every DEFERRED row has explicit reactivation criteria;
  every CLOSED row has explicit "what made this closed" rationale).

## Cross-references

- `.omx/research/canonicalization_dedup_oss_rigor_ledger_20260512.md` (the 15-decision ledger)
- `.omx/research/canon_1_d_hand_curated_list_audit_20260512.md` (CANON-1.D LANDED)
- `.omx/research/canon_1_f_lexical_scanner_closure_20260512.md` (CANON-1.F closure detail)
- `.omx/research/substrate_tradition_taxonomy_20260512.md` (CANON-1.A canonical taxonomy)
- `feedback_wave_a_1_canonical_scorer_helper_14_substrate_migration_landed_20260512.md` (CANON-1.B)
- `feedback_wave_a_2_taxonomy_inventory_drift_landed_20260512.md` (CANON-1.A + 1.J + 1.K)
- `feedback_wave8_vastai_phantom_cleanup_LANDED_20260512.md` (CANON-1.L)
- `feedback_wave8_recovered_cleanup_DEFERRED_pending_operator_decision_20260512.md` (CANON-1.M)
- `feedback_wave8_gha_contest_cpu_workflow_LANDED_20260512.md` (CANON-1.I workflow)
- `feedback_session_2026_05_12_lessons_learned_canonicalization_discipline.md` (PERMANENT KNOWLEDGE)
