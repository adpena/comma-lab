# Codex Findings - Ruff Preflight Force-Exclude

Date: 2026-05-18 18:30:48 UTC
Author: Codex

## Finding

The repository-level Ruff configuration and CI path already correctly exclude
generated custody trees such as `experiments/results` while preserving the
expanded blocking F821 scan over `src/`, `experiments/`,
`submissions/robust_current/`, `scripts/`, and `tools/`.

The remaining mismatch was local-hook specific: `tools/preflight_hook.py`
passed explicit staged file paths to Ruff without `--force-exclude`. Ruff checks
explicit paths even when they match configured excludes unless
`--force-exclude` is present, so the local hook could still lint generated or
ignored files that CI intentionally excludes.

## Patch

- Added `--force-exclude` to the local staged-file Ruff F821 invocation in
  `tools/preflight_hook.py`.
- Added a regression assertion in `src/tac/tests/test_ci_ruff_scope.py` so CI,
  repository config, and local hook semantics stay aligned.

## Verification

- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_ci_ruff_scope.py]`
  - Result: `3 passed in 0.15s`
- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check --force-exclude --select F821 src/ experiments/ submissions/robust_current/ scripts/ tools/]`
  - Result: `All checks passed!`

## Residual

This does not change the already-recorded ITEM_3 master-gradient projector
state. It is a separate preflight consistency hardening patch prompted by the
operator's Ruff reconfiguration concern.
