# Retroactive Sweep for Catalog #373 — 2026-05-28T14:10:00Z

**Lane**: `lane_canonical_anti_patterns_layer_3_plus_5_20260528`
**Gate**: `check_compound_stack_proposal_acknowledges_known_anti_patterns`
**Memo**: `.omx/research/canonical_anti_patterns_layer_3_plus_5_landed_20260528.md`

## Bug-class symptom signature

Compound stacking work re-discovers a registered canonical anti-pattern
via paid GPU dispatch because the proposing landing memo does NOT cite
the matched anti-pattern + canonical_unwind_path (registered at
`tac.canonical_anti_patterns` Layer 1+2 landed 2026-05-28) BEFORE the
operator approves dispatch. Pre-fix the bug class manifests as:

* Memo proposes compound stack (trigger tokens `compound stack` /
  `compounding order` / `stacking` / `stack-of-stacks` / `stack of stacks`
  / `combined codec`).
* Stack proposal structurally overlaps with a registered anti-pattern's
  `recurrence_conditions` / `forbidden_pattern_predicate` (detected via
  `match_stack_against_anti_patterns` min_confidence=0.5).
* Memo lacks acknowledgment cite + lacks waiver + lacks empirical
  falsification ratification.
* Operator approves dispatch based on memo body alone.
* Paid GPU dispatch runs; anti-pattern manifestation produces stale /
  saturated / corrupted result; the empirical anchor mirrors the
  registered anti-pattern's `falsification_band`.

## Pre-fix window

Layer 1+2 landed 2026-05-28 with 12 initial anti-patterns. From Layer
1+2 landing forward, the bug class is structurally TRACKABLE (the
registry is queryable) but NOT structurally REFUSED at the memo surface.
The cutoff for Catalog #373 STRICT enforcement is 2026-05-29 per the
"Strict-flip atomicity rule" — Wave N+1 + earlier memos are exempt.

Pre-fix window: 2026-05-28 (Layer 1+2 landing) → 2026-05-28 (Layer 3+5
landing; same day). Empirical exposure window: 0 days. No paid GPU
dispatch wave fired in this window that could have re-discovered an
anti-pattern via the pre-fix gap.

## Historical KILL / DEFER / FALSIFY search results

Searched `.omx/research/` for landing memos dated 2026-05-28 with
compound-stack triggers AND anti-pattern matches WITHOUT acknowledgment.
Found:

* `canonical_anti_patterns_registry_layer_1_plus_2_landed_20260528.md` — discusses anti-patterns + compound stacking as the SUBJECT MATTER, not as a NEW proposal; the memo IS the registry landing. NOT in violation under any interpretation.
* `dykstra_pareto_polytope_solver_wire_in_dim1_phase4_landed_20260528.md` — discusses Pareto polytope intersection (sister mathematics); no compound stack PROPOSED.
* `canonical_anti_patterns_layer_3_plus_5_landed_20260528.md` — THIS memo. Discusses anti-patterns + compound stacking as the SUBJECT MATTER (Layer 3 + Layer 5 wire-in); cites anti_pattern_id + canonical_unwind_path tokens verbatim per Layer 3 acceptance path (a).

**Result**: 0 historical FALSIFY / KILL / DEFER verdicts predicate on the
pre-Catalog-#373 bug class. The Layer 1+2 landing was simultaneous with
the gap window; Layer 3 + Layer 5 landing closes the gap before any
re-discovery incident.

## Per-finding RE-EVAL-priority assignment

| Finding | Memo | Anti-pattern matched | Re-eval priority | Action |
|---|---|---|---|---|
| (none) | (none) | (none) | (n/a) | (n/a) |

No historical findings to re-evaluate per the 0-day pre-fix window.

## Acknowledgment-cite tokens used in THIS landing memo

Per CLAUDE.md "Forbidden premature KILL" + the canonical anti-patterns
registry's `canonical_unwind_path` field discipline:

* Anti-pattern citation `brotli_plus_lzma_chained_anti_pattern_v1` —
  referenced in the Layer 5 test fixture for end-to-end cathedral
  autopilot integration; the test's synthetic candidate's stack_spec
  explicitly triggers the brotli+lzma compounding-order anti-pattern.
* Canonical unwind path: the test asserts the cathedral autopilot
  invocation payload surfaces `canonical_unwind_paths_recommended` for
  the matched anti-pattern; the matched anti-pattern's registered
  unwind path is "choose ONE high-quality entropy coder (brotli q=11)
  standalone rather than chaining LZMA after brotli (which saturates
  at ~1.001 ratio)".

THIS landing memo IS a research artifact + landing memo (filename
matches `_landed_<YYYYMMDD>.md`); the memo body discusses anti-patterns
+ compounding as the IMPLEMENTATION SUBJECT MATTER, not as a NEW
compound stack PROPOSAL. The Layer 3 gate's acceptance path (a) is
satisfied because the memo cites anti_pattern_id tokens verbatim +
references `canonical_unwind_path` as a top-level concept; the gate
verifies acknowledgment via the canonical token-prefix detection
(`anti_pattern_id=` / `anti-pattern: ` / `canonical_unwind_path` /
`matched anti-pattern` / `ANTI_PATTERN_MATCH_RATIFIED`).

## Cross-references

* Catalog #373 gate function: `src/tac/preflight.py::check_compound_stack_proposal_acknowledges_known_anti_patterns`
* Layer 1+2 registry: `tac.canonical_anti_patterns.match_stack_against_anti_patterns`
* Layer 5 sister: `tac.dykstra_pareto_solver.AntiPatternConstraint` +
  `tools/cathedral_autopilot_autonomous_loop.py::invoke_dykstra_pareto_solver_on_candidates`
* Sister Catalog #372 (Slot 1 Dykstra Pareto solver invoker)
* Sister Catalog #344 (canonical equations memo-reference; #344 is the
  POSITIVE registry surface, #373 is the NEGATIVE registry surface)

## Verdict

LIVE_COUNT_VERIFIED_ZERO at landing. No historical re-eval required.
Forward-going enforcement per the canonical 5-cascade acceptance
contract (cite / waiver / ratification / pre-cutoff exempt / no-trigger-
no-violation).
