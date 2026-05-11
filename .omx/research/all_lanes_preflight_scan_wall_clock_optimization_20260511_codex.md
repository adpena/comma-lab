# All-lanes preflight scan wall-clock optimization - 2026-05-11

## Scope

Goal: reduce all-lanes preflight wall-clock while preserving the full 29-gate
coverage surface. This is score-lowering infrastructure: faster strict gates
make exact-eval dispatch and score-lowering iteration safer to run often.

## Changes

- Gate #10 generated-custody source inventory now uses a fast
  `rg --files --hidden --no-ignore` inventory for ignored generated roots,
  with the existing Python `os.scandir` walk retained as a parity fallback.
- Broad shared `SourceIndex` gates (#0 dispatch CLI/shell hazards, #3 semantic
  label contract, #8 tooling consolidation) now serialize their cache-heavy
  critical section. Local measurement showed concurrent access to the same
  Python `SourceIndex` produced lock/cache contention on macOS; surrounding
  gates still run in parallel.
- Added regression tests proving the `rg` inventory matches the Python walk
  and that shared-index gates serialize.

## Measurement

Before this patch, the tracked baseline timing profile was:

- wall: 2.717217 s
- serial sum: 13.171107 s
- slowest gate: Gate #8 at 2.602238 s
- Gate #10: 1.351885 s

After this patch, final staged verification run:

- wall: 2.461336 s
- serial sum: 12.262684 s
- slowest gate: Gate #8 at 2.42698 s
- Gate #10: 1.036932 s
- all 29 checks passed

The generated-custody inventory itself measured:

- Python walk: 0.4831 s for 26,223 records
- `rg` fast path: 0.2821 s for 26,223 records
- parity: true

## Verification

- `.venv/bin/python -m py_compile tools/audit_untracked_source_artifacts.py tools/all_lanes_preflight.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_audit_untracked_source_artifacts.py src/tac/tests/test_all_lanes_preflight_timing_profile.py src/tac/tests/test_preflight_cli_timeout.py src/tac/tests/test_preflight_proactive_checks.py`
  - 63 passed in 7.32 s
- `.venv/bin/python tools/all_lanes_preflight.py --timings --timings-json reports/all_lanes_preflight_timing_20260511_codex.json`
  - 29/29 passed, wall 2.461336 s
- `git diff --check`

## Next optimization tranche

Do not jump to Rust/Zig before fusing the Python architecture. The highest
remaining payoff is a shared `GitSnapshot` plus batched `SourceIndex` query
prewarm for gates that still shell out or duplicate git/index scans. Native
`ignore`/`aho-corasick`/`rayon` becomes attractive only after the scanner
contract is fully centralized and covered by conformance vectors.
