# OMX parent Markdown state-sync stomp guard - 2026-05-17

## Bug class

The parent-scope Markdown scan showed that `.omx/state/current_focus.md` and
`.omx/state/next_experiments.md` are the current non-research control-plane
authority for L5-v2, TT5L, Rule #6, FEC6, and dispatch discipline.

`src/comma_lab/state_sync.py` was still legacy April Track-B machinery that
rendered both files from `.omx/state/promoted_result.json`. Running
`state sync` could therefore overwrite the current May 17 L5/Rule #6 queue
with a stale promoted-floor template:

- `# Current Focus - <promoted_at>`
- `## Floor`
- `## next steps` with old Modal/Kaggle text

That is a no-signal-loss/control-plane stomp risk, not a score result.

## Fix

Removed lossy state-sync ownership of:

- `.omx/state/current_focus.md`
- `.omx/state/next_experiments.md`

The sync path still repairs the canonical promoted-result summary,
`reports/latest.md`, `.omx/research/findings.md`, `.ralph/run_log.md`, ledgers,
and stale managed-session manifests. It no longer treats the live `.omx/state`
frontier queue as a projection of an old promoted-result record.

## Regression coverage

`src/tac/tests/test_state_sync.py` now asserts that `sync_repo()` preserves
both state docs byte-for-byte and does not report them in `changed_paths`.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_state_sync.py src/tac/tests/test_state_cli.py src/tac/tests/test_frontier_scan.py -q
.venv/bin/python -m ruff check src/comma_lab/state_sync.py src/tac/tests/test_state_sync.py
```

Observed:

- `15 passed`
- `All checks passed`

## Authority

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`

No provider dispatch was launched and no lane claim was opened by this guard.
