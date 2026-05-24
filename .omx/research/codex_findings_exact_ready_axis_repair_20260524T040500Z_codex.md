# Codex Findings - Exact-Ready Axis Repair

UTC: 2026-05-24T04:05:00Z
Author: Codex
Lane: `codex_exact_ready_repair_plan_20260524`

## Finding

After exact-ready identity hardening, the operator briefing exposed `21` raw
stale exact-ready rows. The split was not uniform:

- `12` PR101 nonlocal exact-ready queues had only `score_axis_missing`.
- `9` queues had terminal/result-review/runtime-custody blockers and were not
  repairable by metadata backfill.

Suppressing all rows as generic stale work would lose signal. Hand-editing
historical queues in place would lose custody. The durable fix is a
non-destructive repair copy plus refreshed retraction manifest.

## Landing

`tac.optimizer.exact_ready_axis_repair` and
`tools/repair_exact_ready_score_axis.py` now:

- audit source queues first;
- repair only rows whose sole blocker is `score_axis_missing`;
- copy repaired queues into a new `.omx/research/` directory;
- stamp `score_axis=contest_cuda` and `target_score_axis=contest_cuda`;
- backfill top-level runtime tree/content custody from the nested runtime
  manifest when legacy rows carried it there;
- mark runtime-consumption proof status as present only when a proof path is
  already declared;
- rerun exact-ready audit on each copy;
- keep score, promotion, rank/kill, and dispatch authority false.

The suppression classifier now gives axis-only legacy rows a distinct
classification:
`retracted_legacy_exact_ready_score_axis_missing`.

`tools/operator_briefing.py` now reports the repaired handoff state without
double-counting superseded consumer reports for the same candidate/archive
identity.

## Live Artifacts

- Repair report:
  `.omx/research/exact_ready_score_axis_repair_20260524T040713Z.json`
- Repaired queue directory:
  `.omx/research/exact_ready_score_axis_repair_20260524T040713Z/`
- Refreshed suppression audit:
  `.omx/research/exact_ready_queue_retraction_refresh_20260524T035655Z.json`
- Refreshed canonical suppression manifest:
  `.omx/research/exact_ready_queue_retraction_manifest_20260510_codex.json`
- Paused materializer consumer handoff:
  `.omx/research/materializer_exact_eval_consumer_axis_repair_20260524T040713Z.json`
- Paused experiment queue:
  `.omx/research/materializer_exact_eval_consumer_axis_repair_20260524T040713Z.experiment_queue.json`

## Result

Operator briefing exact-ready hygiene is now green:

- `raw_stale_ready_row_count=21`
- `suppressed_ready_row_count=21`
- `stale_ready_row_count=0`

The repaired PR101 queues produced `12` authorized paused dry-run materializer
consumer rows, `0` blocked rows, and no dispatch attempt. These are still not
score, promotion, rank/kill, or paid-dispatch authority; they are ready for the
normal claim-plus-dispatch gate.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_exact_ready_axis_repair.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_optimizer_exact_ready_audit.py src/tac/tests/test_exact_ready_axis_repair.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_operator_briefing.py -k 'materializer_exact_ready_handoff_summary'`
- `.venv/bin/python -m pytest -q src/tac/tests/test_exact_ready_axis_repair.py src/tac/tests/test_optimizer_exact_ready_audit.py src/tac/tests/test_materializer_exact_eval_consumer.py src/tac/tests/test_operator_briefing.py`
- `.venv/bin/python -m ruff check src/tac/optimizer/exact_ready_axis_repair.py src/tac/tests/test_exact_ready_axis_repair.py tools/repair_exact_ready_score_axis.py src/tac/optimizer/exact_ready_audit.py src/tac/tests/test_optimizer_exact_ready_audit.py tools/operator_briefing.py src/tac/tests/test_operator_briefing.py`

Latest broad suite result: `81 passed`; ruff and py_compile passed.

## Remaining Work

The paused experiment queue is exact-ready handoff work, not an exact auth eval.
Next step is to route these `12` rows through the dispatch-claim lifecycle and
decide whether any are worth exact CUDA spend given the current frontier and
known PR101 result-review anchors.
