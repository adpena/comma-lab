# Preflight scanner fast relative-path hardening (2026-05-11)

## Scope

Operator goal: keep all-lanes preflight comprehensive while driving wall-clock
latency down. The measured slow gates were broad source scanners:

- Gate #0 `check_dispatch_cli_shell_hazards`
- Gate #8 `audit_tooling_consolidation`

Both already use `SourceIndex` for candidate selection. Profiling showed the
next bottleneck was repeated canonical path resolution while converting known
repo-contained paths to repo-relative strings.

## Change

The two scanners now use a fast `Path.relative_to(repo_root).as_posix()` path
for files already known to be under the repository, with fallback to
`tac.repo_io.repo_relative` for out-of-tree safety. No guard family was removed.

An attempted nested worker-pool scan was measured and rejected because it
increased all-lanes wall time on this Mac through thread/lock overhead. The
landed patch is the measured simpler path.

## Verification

Focused tests:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_dispatch_cli_shell_hazards.py \
  src/tac/tests/test_audit_tooling_consolidation.py \
  src/tac/tests/test_source_index.py \
  src/tac/tests/test_all_lanes_preflight_timing_profile.py
```

Result: `50 passed in 0.93s`.

Full preflight:

```bash
.venv/bin/python tools/all_lanes_preflight.py \
  --timings \
  --timings-json /tmp/all_lanes_preflight_fast_rel.json \
  --timeout-s 30
```

Result: 29/29 checks passed; wall `2.49s`; serial sum `12.12s`; estimated
parallel speedup `4.88x`.

Measured hot-gate movement in this tranche:

- Gate #0 dispatch CLI/shell hazards: ~`2.30s` warm baseline -> `1.77s`
- Gate #8 tooling consolidation inventory: ~`2.63s` warm baseline -> `2.39s`
- all-lanes wall: ~`2.73s` warm baseline -> `2.49s`

## Classification

This is a DX/performance hardening change, not a score claim. It preserves the
existing guard semantics and leaves the 30s all-lanes crash budget intact.
