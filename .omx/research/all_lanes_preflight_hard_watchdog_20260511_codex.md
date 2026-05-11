# All-lanes preflight hard watchdog (2026-05-11)

## Scope

Operator concern: routine preflight must never become a multi-minute velocity
tax. The all-lanes preflight already has a 30s soft DX budget, timing JSON,
parallel execution, and SourceIndex-backed scan reuse. This pass hardened the
remaining stall class: an in-process gate that ignores cooperative cancellation
and keeps the CLI alive after the budget expires.

## Change

- `tools/all_lanes_preflight.py` now starts a hard process watchdog for normal
  bounded runs.
- Default budget remains `30.0s`.
- Default hard-watchdog grace is `2.0s`.
- `--allow-slow-preflight` still disables the budget for explicit profiling or
  release/custody sweeps.
- `PACT_ALL_LANES_PREFLIGHT_HARD_WATCHDOG=0` can disable the hard watchdog for
  diagnostics only.

The watchdog is a last-resort process exit. Cooperative subprocess gates still
receive shrinking timeouts and return structured timing failures first.

## Verification

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_all_lanes_preflight_timing_profile.py \
  src/tac/tests/test_preflight_cli_timeout.py \
  src/tac/tests/test_preflight_proactive_checks.py
```

Result: `46 passed in 8.05s`.

```bash
.venv/bin/python tools/all_lanes_preflight.py \
  --timings \
  --timings-json reports/all_lanes_preflight_timing_20260511_codex.json
```

Result: `29/29 passed`, wall `2.717217s`, serial sum `13.171107s`,
estimated speedup `4.847278x`.

Current hot steps:

1. Gate #8 tooling consolidation inventory: `2.602238s`
2. Gate #0 dispatch CLI/shell hazards: `1.912344s`
3. Gate #3 semantic-label contract: `1.372337s`
4. Gate #10 untracked source inventory: `1.351885s`
5. Gate #19 PR91 HPM1 fail-closed custody: `0.740852s`

## Score-lowering relevance

This is not a score claim. It preserves score-lowering velocity: every local
edit, custody hardening, and dispatch-readiness check stays bounded, timed,
and fail-closed instead of quietly slowing the training/eval loop.
