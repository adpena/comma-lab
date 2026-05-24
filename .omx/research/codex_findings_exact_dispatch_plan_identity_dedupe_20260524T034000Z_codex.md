# Codex Findings - Exact-Ready Handoff Identity Guards

UTC: 2026-05-24T03:40:00Z
Author: Codex
Lane: `codex_exact_dispatch_plan_identity_dedupe_20260524`

## Finding

The materializer exact-ready consumer and exact-eval dispatch planner were too
dependent on `candidate_id` and loose exact-ready queue shape. Candidate ids are
operator-facing labels; the dispatch identity that matters for duplicate
paid-eval suppression is canonical candidate archive SHA, consumed runtime
content SHA, and score axis. Runtime-tree SHA remains custody metadata and an
exact-readiness requirement, but wrapper-tree churn alone must not create a
second dispatch identity when consumed runtime content is identical.

## Landing

`build_materializer_exact_eval_dispatch_plan` and
`build_materializer_exact_eval_consumer_queue` now compute stable handoff
identity for every exact-ready row, block missing or inconsistent identity
metadata before authority checks, validate dispatch-ready/top-k queue shape, and
reject duplicate stable identities when candidate ids differ. SHA alias fields
now fail closed when `candidate_archive_sha256`, `archive_sha256`, and
`expected_archive_sha256` disagree, with canonical precedence matching the
exact-readiness/audit path.

Plan rows, consumer rows, and queue metadata carry explicit false-authority
fields so dry-run and execute-plan artifacts cannot be mistaken for score or
promotion authority. The consumer CLI now defaults to active floor guards, can
require at least one authorized row, and ignores rebuildable generated consumer
JSON artifacts through `.gitignore`.

## Verification

- `PYTHONPATH=. .venv/bin/python -m pytest -q src/tac/tests/test_materializer_exact_eval_consumer.py src/tac/tests/test_materializer_chain_harvest_scheduler.py tests/test_comma_lab_research_state.py`
- `PYTHONPATH=. .venv/bin/python -m ruff check src/comma_lab/scheduler/materializer_exact_eval_consumer.py src/comma_lab/scheduler/materializer_exact_eval_dispatch_plan.py src/tac/tests/test_materializer_exact_eval_consumer.py src/tac/tests/test_materializer_chain_harvest_scheduler.py tools/build_materializer_exact_eval_consumer.py tests/test_comma_lab_research_state.py`
- `PYTHONPATH=. .venv/bin/python -m py_compile src/comma_lab/scheduler/materializer_exact_eval_consumer.py src/comma_lab/scheduler/materializer_exact_eval_dispatch_plan.py tools/build_materializer_exact_eval_consumer.py`

Result: focused exact-ready handoff suite passed with `42 passed`; ruff and
py_compile passed.

## Remaining Work

This still produces plans and queue rows, not score authority. Real paid
dispatch remains gated on explicit operator spend, lane claims, and contest
CPU/CUDA auth result custody.
