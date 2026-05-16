# Parallel Dispatch Runtime-Custody Hardening

Date: 2026-05-16
Owner: codex
Scope: `tools/parallel_dispatch_top_k.py`

## Finding

The exact-ready queue schema path already runs the richer exact-ready audit,
but generic `dispatch_ready` / `top_k` ranked inputs fell back to
`_candidate_blockers()`. That fallback validated archive path, archive bytes,
archive SHA-256, target mode, and evidence semantics, but did not require a
runtime tree SHA-256 before paid exact-eval dispatch.

This created a false-authority path where a generic row with
`ready_for_exact_eval_dispatch=true`, valid archive custody, and no runtime
custody could pass dispatch input loading.

## Fix

- Added runtime-tree SHA extraction for top-level and nested runtime custody
  fields.
- Added `runtime_custody:runtime_tree_sha256_missing_or_invalid` to generic
  exact-eval dispatch blockers.
- Updated generic ready-row fixtures to include runtime custody by default.
- Added a regression that removes `runtime_tree_sha256` and verifies dispatch
  input loading refuses the row.

## Evidence

- `.venv/bin/python -m pytest src/tac/tests/test_dispatch_command_builder_shapes.py tests/test_parallel_dispatch_top_k_exact_ready_audit.py -q`
- `.venv/bin/python -m ruff check tools/parallel_dispatch_top_k.py src/tac/tests/test_dispatch_command_builder_shapes.py tests/test_parallel_dispatch_top_k_exact_ready_audit.py`
- `.venv/bin/python -m py_compile tools/parallel_dispatch_top_k.py src/tac/tests/test_dispatch_command_builder_shapes.py tests/test_parallel_dispatch_top_k_exact_ready_audit.py`
- `git diff --check -- tools/parallel_dispatch_top_k.py src/tac/tests/test_dispatch_command_builder_shapes.py tests/test_parallel_dispatch_top_k_exact_ready_audit.py`

## Remaining Canonicalization

Archive/runtime SHA extraction is still repeated across dispatch, exact-ready
audit, operator briefing, prediction-band anchors, and L5-v2 probe validation.
The next structural hardening pass should move these predicates into one
shared `tac` custody authority helper and have all surfaces consume it.
