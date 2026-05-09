# Preflight bounded default and source-index acceleration (2026-05-09)

## Verdict

Routine developer preflight must be a bounded edit-loop gate. The exhaustive
release/custody surface remains available, but it is now explicit:

- default: `.venv/bin/python -m tac.preflight`
- full: `.venv/bin/python -m tac.preflight --scope all --allow-slow-preflight`

This supersedes the prior Catalog #145 interpretation that made `--scope all`
the CLI default. That policy preserved observability but caused the normal CLI
to trip the 30s wall-clock budget as the strict surface grew.

## Evidence

- Default preflight: `real 10.54`, passed.
- Explicit full preflight: `real 49.68`, passed with existing warn-only backlog
  surfaced.
- `tools/all_lanes_preflight.py`: 29/29 checks passed.
- Focused tests:
  - `test_source_index.py`
  - `test_loader_format_safety.py`
  - `test_silent_defaults_audit.py`
  - `test_callsite_contracts_source_index.py`
  - `test_callsite_contracts_and_no_mps_decision.py`
  - `test_preflight_hook.py`
  - `test_codex_round6_medium2_preflight_cli_default_scope.py`
  - selected preflight meta strict-call regression

## Implementation

- `SourceIndex.files_containing_substrings` now handles non-default needles by
  scanning cached source text once per file and batching all unknown needles.
- Hot preflight scanners reuse source-index text/AST and candidate prefilters
  instead of opening every file repeatedly.
- `tac.preflight` default scope is `dev`; `all`/`release` stay explicit.
- Catalog #145 self-protection now enforces bounded default plus explicit full
  choices, not a slow default.
- `tools/preflight_hook.py` uses a 30s default timeout and switches
  `PREFLIGHT_FULL=1` to `--scope all --allow-slow-preflight`.

## No signal loss

No score, release, or dispatch promotion rule changed. Release/custody sweeps
still use the full gate; routine local development no longer pays that cost on
every edit. Any exact-eval dispatch still requires the normal lane claim,
custody, archive/runtime, component, and evidence-grade protocols.
