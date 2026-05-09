# Preflight developer clean-cache ratchet (2026-05-09)

## Change

Added a dedicated `preflight_developer_clean` cache for the default
`python -m tac.preflight` developer scope.

This is intentionally separate from `preflight_all_clean`:

- developer cache covers the bounded strict edit/dispatch safety gate;
- all/release cache covers exhaustive custody and historical scans;
- artifact/training inputs still run live and do not use the codebase-only
  clean cache.

The full-preflight fingerprint now also excludes ignored
`.omx/research/artifacts/**`, so writing profiler JSON no longer invalidates
the next clean-cache hit.

## Evidence

Commands:

```bash
.venv/bin/python -m pytest src/tac/tests/test_preflight_all_clean_cache.py -q
/usr/bin/time -p .venv/bin/python -m tac.preflight
/usr/bin/time -p .venv/bin/python -m tac.preflight
```

Results:

- Cache tests: 4 passed.
- First default developer preflight after source edits: `PREFLIGHT PASSED`,
  `real 9.99`.
- Second unchanged default developer preflight: cache hit,
  `real 0.66`.

## Classification

Performance/DX guardrail. No score promotion. No remote/GPU/eval dispatch.

Reactivation criteria:

- If the default developer preflight exceeds 30 seconds, the CLI timeout should
  fail the run and trigger a fresh latency profile.
- If release/custody preflight remains above the operator budget, continue
  migrating repeated scans onto `tac.source_index` or a native index backend.
