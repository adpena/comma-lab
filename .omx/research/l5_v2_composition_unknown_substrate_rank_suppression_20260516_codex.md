# L5 v2 Composition Unknown-Substrate Rank Suppression

Date: 2026-05-16
Author: Codex
Scope: Cathedral autopilot substrate-composition ranking ingestion

## Verdict

`score_claim=false`; `promotion_eligible=false`; `ready_for_exact_eval_dispatch=false`.

Unknown substrate IDs in composition-ranking rows now receive an explicit
rank-suppressing blocker instead of passing through as clean planning rows.

## Failure Class

`composition_ranking_unknown_substrate_attention_leak`

The composition filter intentionally did not crash on unknown substrate names,
but the loader also left those rows clean. A typo or arbitrary substrate ID
could therefore consume Cathedral ranking attention even though it was not in
the canonical composition inventory.

## Landed Fix

`load_candidates_from_substrate_composition_ranking(...)` now compares
`substrate_ids` against `canonical_substrate_inventory()` and adds:

`composition_matrix_unknown_substrate:<substrate_id>`

for each unknown ID. Rows with unknown substrate IDs also have
`expected_information_gain=0.0` so the default EIG/$ ranking does not elevate
them.

## Verification

```bash
.venv/bin/python -m ruff check \
  tools/cathedral_autopilot_autonomous_loop.py \
  src/tac/tests/test_cathedral_autopilot_substrate_composition_wire.py

.venv/bin/python -m pytest \
  src/tac/tests/test_cathedral_autopilot_substrate_composition_wire.py \
  src/tac/tests/test_cathedral_autopilot_autonomous_loop.py -q
```

Observed:

- `ruff`: all checks passed
- `pytest`: `176 passed in 0.61s`
