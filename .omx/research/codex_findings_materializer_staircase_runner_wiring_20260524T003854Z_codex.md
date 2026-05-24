---
schema: codex_findings_v1
author: codex
created_at_utc: 2026-05-24T00:38:54Z
lane_id: lane_codex_queue_executor_materializer_tranche_20260524
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
---

# Materializer Staircase Runner Wiring

## Finding

The byte-shaving materializer runner could already build an
`experiment_queue.v1`, initialize state, run a bounded local worker, and emit
queue observation artifacts. The missing automation bridge was immediate
staircase/DAG emission from that same queue plus an operator-callable SSH
executor dry-run. Without that bridge, agents had to hand-carry the generated
queue through separate tools, which slowed iteration and risked losing the exact
queue/state pair that produced the candidate set.

This tranche wires `tools/run_byte_shaving_materializer_campaign.py` to emit
`staircase_dag.json` and `staircase_dispatch_plan.json` directly from the
generated execution queue and initialized SQLite state. It can also invoke
`tools/run_staircase_ssh_executor.py` in dry-run mode against the emitted plan.
The dry-run remains false-authority and never executes remote work, but it now
validates that the queue-owned SSH executor can consume the generated plan
without a separate manual bridge.

The runner fails closed if the SSH dry-run subprocess exits nonzero or does not
emit a JSON object. That prevents broken executor wiring from being buried in
the command telemetry while the local worker continues.

## Landed Surfaces

- `tools/run_byte_shaving_materializer_campaign.py`: adds staircase plan
  emission, SSH executor dry-run flags, remote repo-root mapping, and strict
  JSON-output validation for the dry-run.
- `src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`: covers
  staircase artifact emission, SSH dry-run command construction, remote-root
  parsing, and fail-closed JSON validation.

## Verification

- `PYTHONPATH=. .venv/bin/python -m ruff check tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`
- `PYTHONPATH=. .venv/bin/python -m pytest -q src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_ssh_experiment_queue_executor.py src/tac/tests/test_staircase_dag.py`
  passed with `34 passed`.
- Expanded queue regression:
  `PYTHONPATH=. .venv/bin/python -m pytest -q src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_ssh_experiment_queue_executor.py src/tac/tests/test_staircase_dag.py src/tac/tests/test_experiment_queue.py`
  passed with `79 passed`.
- `PYTHONPATH=. .venv/bin/python -m py_compile tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`
  passed.

## Frontier Status

No exact auth eval was dispatched in this tranche. `reports/latest.md` still
lists `[contest-CPU Linux x86_64]` best at `0.1920282830` and
`[contest-CUDA T4]` best at `0.2053300290`.

## Next Tranche

Next is the artifact mobility layer: make SSH workers return materialized
candidate trees through an explicit shared-storage or sync contract, then allow
the runner to execute bounded SSH work rather than only dry-run it. After that,
the campaign runner becomes the single command that builds candidates, emits a
staircase plan, saturates local/peer resources, observes queue telemetry, and
hands exact-ready artifacts to the strict auth-eval dispatch gate.
