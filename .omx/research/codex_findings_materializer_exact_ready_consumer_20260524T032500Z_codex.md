# Codex Findings - Materializer Exact-Ready Consumer

UTC: 2026-05-24T03:25:00Z
Author: Codex
Lane: `codex_materializer_exact_ready_consumer_20260524`

## Finding

Materializer harvest already produced exact-ready queues and bridge reports, but
there was no queue-owned consumer that could assemble those artifacts into the
next dry-run dispatch preparation step. That left exact-ready artifacts as
manual handoff state rather than scheduler-owned work.

## Landing

Added `comma_lab.scheduler.materializer_exact_eval_consumer`, plus
`tools/build_materializer_exact_eval_consumer.py`, to consume bridge reports or
exact-ready queue artifacts. The consumer dedupes candidates by archive SHA,
runtime content SHA, and runtime tree SHA, re-runs exact-ready audit plus
`exact_dispatch_authority`, and emits a paused `experiment_queue.v1`.

The generated queue remains dry-run only: claim steps use
`tools/claim_lane_dispatch.py --dry-run`, dispatch steps use
`tools/parallel_dispatch_top_k.py --dry-run`, and the report is forced through
the planning/proxy false-authority boundary. Contest CPU/CUDA auth results
remain the only score authority.

## Verification

- `PYTHONPATH=. .venv/bin/python -m pytest -q src/tac/tests/test_materializer_exact_eval_consumer.py src/tac/tests/test_materializer_chain_harvest_scheduler.py`
- `PYTHONPATH=. .venv/bin/python -m ruff check src/comma_lab/scheduler/materializer_exact_eval_consumer.py src/comma_lab/scheduler/__init__.py src/tac/tests/test_materializer_exact_eval_consumer.py tools/build_materializer_exact_eval_consumer.py`
- `PYTHONPATH=. .venv/bin/python -m py_compile src/comma_lab/scheduler/materializer_exact_eval_consumer.py src/comma_lab/scheduler/__init__.py tools/build_materializer_exact_eval_consumer.py`

Result: focused exact-ready consumer and harvest scheduler suite passed with
`25 passed`; ruff and py_compile passed.

## Remaining Work

This is a dry-run queue builder, not an execution actuator. The next step is a
bounded end-to-end run on a real harvested exact-ready queue, followed by a
separate operator-explicit paid dispatch path that preserves claim lifecycle,
terminal claim closure, and contest-axis custody.
