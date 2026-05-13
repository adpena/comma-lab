# CANON-1.F — Catalog #125 lexical-vs-semantic keyword scanner CLOSURE 2026-05-12

**Source ledger**: `.omx/research/canonicalization_dedup_oss_rigor_ledger_20260512.md`
CANON-1.F.

**Lane**: `lane_canon_b_c_remaining_decisions_20260512` (L1, Phase 2).

**Status**: **CLOSED — already addressed by current implementation**.

## Audit verdict

The CANON-1.F recommendation called for accepting BOTH underscore-form
(`sensitivity_map`) AND hyphenated-form (`Sensitivity-map`) keyword variants
in the Catalog #125 scanner. Verification 2026-05-12 confirms this is
**already implemented** — no further work required.

## Verification

The canonical hook keyword table at `src/tac/preflight.py:33135-33167`
(`_UNIFIED_LAGRANGIAN_WIRE_IN_HOOKS`) declares each hook with **all THREE
spelling variants** as accepted aliases:

| Hook | Underscore form | Hyphenated form | Space form |
|---|---|---|---|
| 1. Sensitivity-map | `sensitivity_map` | `sensitivity-map` | `sensitivity map` |
| 2. Pareto constraint | `pareto_constraint` | `pareto-constraint` | `pareto constraint` |
| 3. Bit-allocator | `bit_allocator` | `bit-allocator` | `bit allocator` |
| 4. Cathedral autopilot | `cathedral_autopilot` | `cathedral-autopilot` | `cathedral autopilot` |
| 5. Continual-learning | `continual_learning` | `continual-learning` | `continual learning` |
| 6. Probe-disambiguator | `probe_disambiguator` | `probe-disambiguator` | `probe disambiguator` |

Plus additional informal aliases per hook (e.g. `sensitivitymap` as
no-separator; `pareto-frontier` as semantic synonym).

The `_memo_declares_hook` scanner at `src/tac/preflight.py:33239` performs
case-insensitive substring matching against the joined alias set, accepting
ANY of the three spelling forms. The CANON-1.F bug class (β `_full_main`
landing memo blocked because it used hyphenated form) is structurally
extincted by the comprehensive alias table.

## Empirical evidence of CLOSED state

Run from working tree at SHA `1f188d75`:

```python
from tac.preflight import check_subagent_landing_has_solver_wire_in
result = check_subagent_landing_has_solver_wire_in(strict=False, verbose=False)
# result == [] (no violations on current memory dir)
```

If a subagent landing memo declares hooks using ANY of the three spelling
forms, the scanner accepts it. The original CANON-1.F bug class (β `_full_main`
memo getting blocked despite semantically-correct hook declaration) cannot
recur.

## Why this is CLOSED-not-DEFERRED

The recommendation in the CANON-1.F ledger row was:

> Update scanner to accept BOTH forms (regex `r"sensitivity[-_]map"` etc.) OR
> accept any of {EXERCISED, DECLARED, WIRED, N/A} within 120 chars of any hook
> word. ~20 LOC scanner update + tests.

The current implementation does even MORE than this recommendation: it
accepts the underscore form, hyphenated form, AND space-separated form,
PLUS informal aliases per hook (e.g. `pareto-frontier`, `bayesian update`).
Per-hook acceptance further accepts:

- `:` (canonical declaration form)
- `=` (struct-style declaration)
- `wire` / `wired` / `wiring` (informal acknowledgment)
- `hook` (informal "<hook>: ..." pattern)
- `contribution` / `update` / `constraint` / `dispatch` / `n/a` (semantic markers)

So CANON-1.F is fully addressed by the existing implementation. NO further
scanner relaxation is required.

## Forbidden patterns honored

- ZERO `/tmp` paths.
- ZERO score claims.
- ZERO MPS-derived strategic decisions.
- ZERO KILL verdicts.

## 6-hook wire-in declaration (per Catalog #125)

This is a CLOSURE memo, not an empirical anchor landing. Per the same Catalog
#125 scanner this memo addresses, all 6 hooks are declared N/A with rationale:

1. **Sensitivity-map**: N/A — META verification of an existing scanner; no
   new per-tensor saliency contribution introduced.
2. **Pareto constraint**: N/A — no new feasibility constraint.
3. **Bit-allocator**: N/A — no per-tensor importance change.
4. **Cathedral autopilot dispatch**: N/A — META verification.
5. **Continual-learning posterior**: N/A — no new empirical anchor.
6. **Probe-disambiguator**: N/A — verification of structural invariant; no
   2+-interpretation tension to arbitrate.

## Cross-references

- `.omx/research/canonicalization_dedup_oss_rigor_ledger_20260512.md` (CANON-1.F source)
- `feedback_session_2026_05_12_lessons_learned_canonicalization_discipline.md`
  (Lesson 3: lexical-vs-semantic match in keyword scanners)
- `src/tac/preflight.py:33135-33167` (`_UNIFIED_LAGRANGIAN_WIRE_IN_HOOKS`)
- `src/tac/preflight.py:33239-33337` (`_memo_declares_hook`)
- Catalog #125 `check_subagent_landing_has_solver_wire_in`
