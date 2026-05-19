# Codex Session Summary: Canonical Task-Status Bootstrap - 2026-05-19T00:02:51Z

## Scope

Closed the remaining single-source-of-truth bootstrap items from
`.omx/research/codex_routing_directive_canonical_task_status_single_source_of_truth_20260518.md`:

- ITEM_5: persistent `/goal` v2 consumes `.omx/state/canonical_task_status.jsonl`.
- ITEM_6: directive-task backfill is registered in canonical task status.
- ITEM_7: Claude `TaskCreate` / `TaskUpdate` work intended for Codex, autopilot,
  or dashboards must mirror into canonical task status.

## Authority Model

A Pact task or claim is authoritative only when the following chain is present:

1. A dated design or findings memo states the contract.
2. `.omx/state/canonical_task_status.jsonl` records ownership and latest status.
3. A durable commit SHA anchors any protocol or implementation change.
4. Focused validation proves the ledger and affected code paths still parse.
5. Read models such as DuckDB or HF are treated as mirrors, not the source of
   truth.

Claude-private `TaskCreate` state, chat prose, and read-model mirrors are not
independent authority. They become actionable when mirrored into the shared
ledger or explicitly relayed through a directive.

## Evidence

- `.omx/research/codex_persistent_goal_v2_20260518.md` is committed in
  `b4b49a6ca` and makes canonical task status the primary work queue.
- `tools/extract_canonical_tasks_from_directive.py --register-all --actor codex
  --session-id 019de465 --json` registered four previously untracked Z6-v2
  directive tasks.
- `AGENTS.md` now documents the Canonical Task-Status Mirror rule under the
  event-driven collaboration protocol.

## Verification

Run before completion rows:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_canonical_task_status.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check tools/extract_canonical_tasks_from_directive.py tools/canonical_task_status.py src/tac/canonical_task_status
.venv/bin/python tools/canonical_task_status.py --validate
git diff --check
```

## Residual Risk

Repo-local verification can prove every extractable directive task has a
canonical row. It cannot prove coverage of Claude-private `TaskCreate` state
that was never written to a repo surface. The fail-closed rule is therefore
intentional: unmirrored private tasks are advisory until registered.
