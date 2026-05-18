# Codex Findings - Ruff Isolated F821 Policy

Date: 2026-05-18 20:09:52 UTC
Author: Codex

## Finding

The Ruff hard gate needed one more policy separation. The generated-artifact
scope was already hardened with `force-exclude = true`, but the blocking F821
commands still loaded project config. Because `src/tac/preflight.py` has a
deliberate broad-lint carve-out (`["ALL"]`) for legacy churn, a project-configured
F821 scan could miss undefined names in the preflight mega-file.

Correct policy:

- F821 is a hard, config-independent NameError gate over `src/`, `experiments/`,
  `submissions/robust_current/`, `scripts/`, and `tools/`.
- Generated custody trees remain excluded: `experiments/archive` and
  `experiments/results`.
- Broad style/refactor Ruff remains project-configured and nonblocking until
  the historical baseline is intentionally burned down.

## Patch

- `.github/workflows/ci.yml`: blocking F821 command now uses `--isolated`,
  `--ignore-noqa`, and explicit generated-tree excludes.
- `tools/preflight_hook.py`: staged-file F821 hook now uses the same isolated
  policy, so per-file style ignores or local noqa markers cannot suppress
  undefined-name checks.
- `src/tac/tests/test_ci_ruff_scope.py`: added subprocess probes proving
  synthetic F821 fails under `src/tac/preflight.py` while a synthetic generated
  path under `experiments/results` stays excluded.
- `pyproject.toml`: global broad-lint config now ignores mathematical Unicode
  notation rules RUF001/RUF002/RUF003 and recognizes legacy WPS noqa markers as
  external suppressions. This reduces nonblocking noise without weakening F821.

## Verification

- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_ci_ruff_scope.py]`
  - Result: `8 passed in 0.22s`
- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check pyproject.toml tools/preflight_hook.py src/tac/tests/test_ci_ruff_scope.py]`
  - Result: `All checks passed!`
- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check --isolated --force-exclude --select F821 --ignore-noqa --exclude experiments/archive --exclude experiments/results src/ experiments/ submissions/robust_current/ scripts/ tools/]`
  - Result: `All checks passed!`
- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check --force-exclude --select RUF102 src/ experiments/ submissions/robust_current/ scripts/ tools/]`
  - Result: `All checks passed!` with one pre-existing warning from
    `submissions/robust_current/inflate_renderer.py:5348`, where a custom
    lowercase pseudo-noqa marker is intentionally left untouched to avoid a
    review-gate-heavy contest-runtime edit in this Ruff policy slice.
- `[empirical:PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/canonical_task_status.py --validate]`
  - Result: `{"rows": 52, "status": "valid"}`
- `[empirical:git diff --check]`
  - Result: clean

## Residual

Explicit command-line selection of RUF001/RUF002/RUF003 still reports historical
math-notation diagnostics because an explicit `--select` overrides normal broad
lint policy. That is expected and is not the hard gate. The production hard
gate is the isolated F821 command above.

The contest runtime still contains one lowercase custom pseudo-noqa marker.
The isolated hard F821 gate uses `--ignore-noqa`, so that marker cannot hide
undefined names and no longer adds noise to the hard gate. Cleaning the marker
itself should happen in a runtime-reviewed patch, not bundled into this
configuration-only hardening.
