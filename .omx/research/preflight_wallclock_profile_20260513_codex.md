# Preflight Wall-Clock Profile (Codex, 2026-05-13)

## Summary

All-lanes preflight was rerun after the PR106 sidecar-profile correction and
passed under the normal dispatch gate:

- command: `.venv/bin/python tools/all_lanes_preflight.py --timings --timings-json experiments/results/preflight_all_timing_20260513_codex_after_pr106_parser_fix.json`
- result: `ALL 30 PREFLIGHT CHECKS PASSED`
- wall time: `2.02s`
- serial step sum: `11.47s`
- workers: `8`
- estimated speedup: `5.68x`
- default hard budget: `30.0s` via `tools/all_lanes_preflight.py`
- hard watchdog: enabled by default with `2.0s` grace

This discharges the earlier long-preflight concern for the current tree. A
future all-lanes run exceeding 30s should fail closed unless invoked with the
explicit `--allow-slow-preflight` profiling/debug override.

## Hot Steps

The largest current steps are:

1. `GATE #8 tooling consolidation inventory`: `1.986640s`
2. `GATE #3 semantic-label contract`: `1.373865s`
3. `GATE #10 untracked source inventory`: `0.976718s`
4. `GATE #19 PR91 HPM1 fail-closed custody`: `0.728704s`
5. `GATE #0 dispatch CLI/shell hazards`: `0.714972s`

None are close to the 30s dispatch budget. The next optimization pass should
target the first three gates only if they regress above the current profile or
if hosted CI shows materially different timing.

## Artifact

- `experiments/results/preflight_all_timing_20260513_codex_after_pr106_parser_fix.json`

## Score-Lowering Relevance

Fast preflight is not a score claim, but it directly improves score-lowering
velocity: exact-eval packets can cycle through custody gates without burning
operator time, while the normal gate still blocks proxy-score authority, stale
custody, untracked source, and non-dispatch-ready forensic lanes.
