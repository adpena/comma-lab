# Preflight Check 168 SourceIndex Optimization - 2026-05-13

## Scope

Optimize Catalog #168, `check_ast_walker_handles_both_assign_and_annassign`,
without reducing coverage. This is a DX wall-clock change only; it does not
change score authority or dispatch eligibility.

## Change

The check previously attempted an `rg` subprocess prefilter even when the
shared `SourceIndex` was active. The fast path now uses
`SourceIndex.files_containing_substrings(..., require_all=("isinstance",
"ast.Assign"))` directly, then runs the existing AST-level detector on those
candidate files.

This is a conservative prefilter: the real violation requires both an
`isinstance(...)` call and an `ast.Assign` type reference before the AST logic
can produce a finding. Files that cannot produce a Catalog #168 violation are
not parsed.

## Evidence

- Focused regression tests:
  `.venv/bin/pytest -q src/tac/tests/test_check_168_ast_walker_handles_assign_and_annassign.py src/tac/tests/test_all_lanes_pr106_sidecar_runtime_gate.py`
- Result: `33 passed`
- Explicit slow all-scope timing sweep:
  `PACT_PREFLIGHT_SOURCE_INDEX_PREWARM=1 .venv/bin/python -m tac.preflight --scope all --allow-slow-preflight --timings-json reports/preflight_all_timing_check168_indexed_20260513.json`
- Result: `PREFLIGHT PASSED`
- Measured Catalog #168 elapsed time in that sweep: `1.539379s`
- Prior review baseline from subagent Godel: `8.47s`

## Coverage Guard

Added a regression test that monkeypatches
`tac.preflight._rg_python_files_matching_regex` to raise during a
`source_index_context(...)` run. The check must still catch an Assign-only AST
walker. This prevents accidental reintroduction of the redundant `rg` path.

## Classification

`performance_optimization`, `coverage_preserving`, `dispatch_safe`.

