# DDM IP1 decision-integrity ports

Tags: `[no-triality]` `[p0-ledger-ok]`  
Date: 2026-08-11  
Axis: `[scorer-free apparatus]`  
Score claim: false  
Pointer moved: false

## Conclusion

All four LV2 PORT-NOW decision-integrity gaps are implemented as typed, fail-closed extensions of
their existing canonical surfaces. No terminal object was consumed or fabricated.

1. `tac.verdicts.verdict_payload()` and its write wrapper now accept an optional
   `InstrumentClaimScope`. When used, the canonical verdict payload carries
   `instrument_capacity`, `object_capacity`, `claim_units`, and `scope_ok`; an object capacity above
   instrument capacity refuses before emission. Omitting the option preserves the legacy payload
   shape.
2. `hr1_prestage` now emits a typed same-parent freshness receipt for `fit`, `map`, `selector`, and
   `correction`. Missing, malformed, or unequal producer/consumer parent SHA-256 values refuse, and
   there is no waiver field.
3. The existing activation ledger now admits explicit `folded` and `queued` evidence and can join
   any JSON-serialized compiled DSL config to `FIRED`, `FOLDED`, or `queued`. Its typed receipt names
   the config hash, ledger hash, exact denominator, joined rows, and missing rows. The reporting CLI
   returns nonzero and still prints a refusal receipt when any non-default lever lacks evidence.
4. The existing EG1 policy module now exposes a vehicle-neutral `StopVerdict` grammar. It derives
   `continue`, `handoff`, or `stop` only from strict same-parent dominance over caller-supplied
   complete additive score components. No contest formula, TR1 constant, target, or component name
   is baked into this port.

The real v7.5.2 compiler, not a hand-copied fixture, supplied the activation positive control. Its
9 non-default levers joined 9/9 to the live 251-row ledger as `FIRED`. A missing-lever CLI control
returned nonzero with a typed `REFUSE` receipt.

## Verification

- Focused suite: **151 passed** across verdict emission, HR1 pre-stage binding, activation ledger,
  and EG1 policy tests.
- Static checks: `ruff check` clean on all ten Python files; `git diff --check` clean.
- Review gate: two clean `review_tracker.py` passes recorded for every entity-bearing Python file;
  `src/tac/verdicts/__init__.py` contains no reviewable class/function entity.
- m94 alone has 21 parameterized/explicit cases, including safe greater-than/equality controls,
  four over-capacity refusals, malformed numeric refusals, typed payload embedding, and unused-shape
  preservation.
- No `REVIEW_GATE_OVERRIDE=1` was used.

## RECALL EVIDENCE

Sources searched and inspected:

- Governing and live state: `PROGRAM.md`, byte-identical `CLAUDE.md`/`AGENTS.md`,
  `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, the IP1 charter, and the
  common contract.
- Canonical implementations and lineage: commits `09b361bcc8`, `3f2dbb2c14`, `436edf452c`, and
  `81ca1affef`; `tac.verdicts.emit`; `hr1_prestage`; `activation_ledger`; the real v7.5.2 compiler;
  `ddm_endgame_policy`; and their focused tests.
- Research content queries: `m94`, `instrument capacity`, `claim units`, `m37`, `same parent`,
  `freshness`, `activation terminal config`, `stop`, `handoff`, and `complete-score dominance` across
  `.omx/research/`, `CANONICAL_RESEARCH_INDEX*`, the sub-0.15 DAG, current task status, and the
  harness bridge.
- Primary receipts: `.omx/research/ddm_na2_negative_audit_20260803.md`,
  `.omx/research/ddm_q43a_20260806/RECEIPT.md`,
  `.omx/research/default_off_comprehensive_sweep_20260710.md`, the 251-row
  `.omx/state/lever_activation_ledger.jsonl`, and EG1 E2 plus its arithmetic receipt.
- Canonical equations: `.venv/bin/python tools/list_canonical_equations.py --json`, queried for
  freshness/staleness, activation, stopping/handoff, dominance, and instrument/claim-unit terms.

Findings beyond the charter seeds changed the implementation:

- The activation reader already documents that its historical writer became vacuous on the live
  vehicle and that factory names do not automatically join constructed lever names. The terminal
  port therefore consumes the compiled config's explicit active-lever names and reports its full
  denominator; it does not use `never_fired()` as terminal truth.
- The live ledger contained 243 `fired`, 7 `measured`, and one invalid historical `built` row, but no
  typed `folded` or `queued` event. Those two dispositions were added explicitly rather than inferred
  from free-text reasons.
- EG1's full E2 implementation was already in-tree and tested, but it embeds TR1-era score constants
  and target logic. The plan narrowed to a parameterized dominance layer inside that module, leaving
  the historical policy intact.
- The m94 source law phrases equality conservatively, while the newer IP1 charter explicitly pins
  refusal to `object_capacity > instrument_capacity`. The implementation follows the charter's
  exact comparison and tests equality as the safe boundary.
- The queried canonical task/harness stores did not contain a newer IP1 implementation row. LV2's
  four PORT-NOW rows remained the current scoped authority.

## Boundaries and non-measurements

- Scorer-free: no SegNet, PoseNet, evaluator, Modal, GPU, MPS, or paid job ran.
- No terminal config or terminal parent exists yet, so no terminal activation, freshness, scope, or
  stop-policy receipt was claimed.
- No candidate, archive, learned payload, score component, or score delta was measured or created.
- `upstream/` and all three protected files were untouched.
- The shared worktree's unrelated edits and staged index were preserved; no stash, reset, broad add,
  or destructive command was used.
- The exact and own-vehicle frontier pointers are unchanged.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: JS1 reseal owner; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hr1_preflight/instrument_scope/`; fire trigger: before the first terminal-base scorer or realization verdict is admitted; action: construct `InstrumentClaimScope` from the real claim denominator and persist the emitted verdict receipt.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: JS1 content-binding owner; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hr1_preflight/content_bindings/`; fire trigger: when PS135 terminal roles bind and again before each selected candidate is scored; action: bind each real fit/map/selector/correction producer parent to its consumer parent and persist the freshness receipts.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: JS1 reseal and costate owner; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hr1_preflight/activation_audit/`; fire trigger: after the exact terminal DSL compiles and before any realization command is enabled; action: run `tools/report_terminal_activation_join.py` on that compiled JSON and resolve every refusal with an explicit fired, folded, or queued ledger event.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: JS1 event-controller owner; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/event_policy/`; fire trigger: before Stage-0 stop rules or terminal-finisher admission are sealed; action: build same-parent complete-component dominance receipts from the current vehicle and consume the derived `StopVerdict`.

## LIVE-HYPOTHESES

- Claim-unit scoping may prevent an instrument-limited negative from becoming a false object/family
  closure, because m94 documents several measured reversals caused by that exact capacity mismatch.
- Same-parent freshness may reject a superficially valid terminal fit after the selected parent
  changes, because q43a already exposed a stale-premise receiver mismatch on this campaign.
- The compiled-config join may surface a non-default terminal lever that historical activation
  summaries miss, because the ledger's own audit proves both writer-vacuity and name-space drift.
- A parameterized complete-score dominance policy may keep Stage-0 training active when an axis-local
  finisher looks attractive but loses after all components are summed, which is the decision error
  EG1 E2 was designed to prevent.

## DEAD-ENDS

- Do not create a second verdict API for m94; the measured VW1 adoption failure was caused by
  bypassing the embeddable canonical producer.
- Do not treat `never_fired()` or a bare ledger row count as terminal configuration truth; the reader
  itself proves that surface can measure a retired writer rather than the live vehicle.
- Do not infer `FOLDED` or `queued` from prose in a `reason` string; they now have typed ledger events.
- Do not reuse EG1's TR1 constants, target bars, packet grammar, or stale effect sizes in the current
  vehicle; only the same-parent complete-score dominance grammar was ported.
- Do not permit parent-hash waivers or force-fit adapters for fits, maps, selectors, or corrections;
  absent or unequal custody is a refusal.
