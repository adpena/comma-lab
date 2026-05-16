# L5 v2 Cathedral Source-Scope Loader Guard

Date: 2026-05-16
Author: Codex
Scope: L5 v2 staircase / Cathedral autopilot planning ingestion

## Verdict

`score_claim=false`; `promotion_eligible=false`; `ready_for_exact_eval_dispatch=false`.

Manual Cathedral autopilot loaders now enforce the same literature-anchor
source-scope blockers as JSONL candidate ingestion. Literature anchors from
substrate-composition rankings and probe-disambiguator outputs cannot receive
the Z1 class-shift rank reward unless they carry the required source-scope
fields:

- `source_supports`
- `paper_claim_scope`
- `pact_must_prove`
- `decode_complexity_evidence`

## Failure Class

`cathedral_manual_loader_literature_source_scope_bypass`

The generic JSONL path already called
`literature_source_scope_blockers(...)`, but L5-relevant manual loaders built
`CandidateRow` objects directly and could bypass the blocker:

- `load_candidates_from_substrate_composition_ranking`
- `load_candidates_from_probe_disambiguator_output`

That made literature support look stronger than the actual Pact evidence and
could bias L5 v2 ranking without paired byte-closed proof.

## Landed Fix

Centralized blocker insertion in
`tools/cathedral_autopilot_autonomous_loop.py` via
`_append_literature_source_scope_blockers(...)`, then wired that helper into:

- JSONL candidate ingestion
- substrate-composition ranking ingestion
- probe-disambiguator planning-row ingestion

Regression tests assert the blocker lands and suppresses the literature-anchor
rank reward for unscoped planning rows.

## Verification

```bash
.venv/bin/python -m ruff check \
  tools/cathedral_autopilot_autonomous_loop.py \
  src/tac/tests/test_cathedral_autopilot_autonomous_loop.py \
  src/tac/tests/test_cathedral_autopilot_substrate_composition_wire.py

.venv/bin/python -m pytest \
  src/tac/tests/test_cathedral_autopilot_autonomous_loop.py \
  src/tac/tests/test_cathedral_autopilot_substrate_composition_wire.py -q
```

Observed:

- `ruff`: all checks passed
- `pytest`: `176 passed in 0.62s`

## Next Audit Hook

Continue reviewing L5 v2 planning ingress paths for loaders that construct
`CandidateRow` directly. Any future loader with `literature_anchor` support
must call `_append_literature_source_scope_blockers(...)` or delegate through
the canonical row parser.
