# Preflight DX cold-path slice — source-index callsite contracts

<!-- generated_at: 2026-05-09T20:34:00Z -->
<!-- evidence_grade: dx_hardening; no score claim; no dispatch -->

## Motivation

The operator set the DX bar: normal preflight should fail closed if it exceeds
30 seconds, and cold scans should keep moving toward a single-pass,
parallelized source-index substrate rather than reopening every file in every
check.

Read-only performance scouting found `check_callsite_contracts_satisfied` was
one of the largest Python scans, previously around `6.3s` in a no-cache profile.

## Change

- Added `reconstruct_poses` to the shared `SourceIndex` hot substring facts.
- Bumped persistent source-facts schema to `pact.source_text_facts.v3` so older
  caches cannot hide the new needle.
- Migrated `check_callsite_contracts_satisfied` to
  `SourceIndex.files_containing_substrings(...)` before AST parsing.
- Added a regression test proving 40 irrelevant files are fact-scanned but only
  the candidate callsite file is AST-parsed.

## Verification

Focused timing after schema bump:

```bash
/usr/bin/time -p .venv/bin/python - <<'PY'
from pathlib import Path
from tac.source_index import source_index_context
from tac.preflight import check_callsite_contracts_satisfied
root = Path(".")
with source_index_context(root) as index:
    violations = check_callsite_contracts_satisfied(repo_root=root)
    print("violations", len(violations))
    print(index.stats())
PY
```

Result:

- violations: `0`
- AST parses: `5`
- wall: `1.93s`

Full preflight behavior:

- normal budgeted run:
  `.venv/bin/python -m tac.preflight --timeout-s 30` fails closed after 30s
  when the full-clean cache is invalidated.
- explicit slow release override:
  `.venv/bin/python -m tac.preflight --allow-slow-preflight` passed in `42.00s`.
- subsequent normal hot-cache run:
  `.venv/bin/python -m tac.preflight --timeout-s 30` passed in `0.71s` with
  `[preflight-all-cache] OK`.

## Remaining DX work

This is one cold-path slice, not the full performance endpoint. The next
highest-EV migrations are:

1. batch custody/concurrency checks #127/#128/#130/#131/#132/#138 into one
   shared facts object;
2. create a `ShellLaneIndex` for `scripts/remote_lane_*.sh` arity,
   provenance, runtime, NVDEC, and shell-hazard checks;
3. source-index candidate-filter `preflight_loader_format_safety`,
   `check_no_off_manifold_pose_zeros`, tag-custody checks, and continual
   learning lock checks;
4. keep Rust/Rayon/PyO3 as a lexical index backend only until parity tests prove
   identical candidate sets against the Python oracle.
