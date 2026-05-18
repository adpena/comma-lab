# Codex Findings: Ruff Runtime Broad-Lint Reconfiguration

timestamp_utc: 2026-05-18T20:21:26Z
agent: codex
evidence_axis: tooling-hardening
score_claim: false
research_only: false

## Finding

The isolated Ruff F821 safety gate is correctly configured and currently green:

```bash
.venv/bin/ruff check --isolated --force-exclude --select F821 --ignore-noqa \
  --exclude experiments/archive --exclude experiments/results \
  src/ experiments/ submissions/robust_current/ scripts/ tools/
```

The noisy surface was the project-wide Ruff configuration. Broad style linting
was still parsing `submissions/robust_current/inflate_renderer.py`, a giant
contest runtime that carries historical custom `# noqa: scorer-at-inflate`
markers. Ruff 0.15.10 warns on that marker before reporting style findings.

## Change

`pyproject.toml` now excludes only
`submissions/robust_current/inflate_renderer.py` from broad project Ruff. The
hard F821 gate remains isolated from project config and still scans the contest
runtime through CI and `tools/preflight_hook.py`.

Regression coverage in `src/tac/tests/test_ci_ruff_scope.py` now proves both
conditions:

- project Ruff no longer emits the invalid custom-noqa warning for the runtime;
- isolated F821 still catches undefined names when the stdin filename is
  `submissions/robust_current/inflate_renderer.py`.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_ci_ruff_scope.py -q
.venv/bin/ruff check --force-exclude --select RUF100 \
  src/tac/tests/test_ci_ruff_scope.py submissions/robust_current/inflate_renderer.py
.venv/bin/ruff check --isolated --force-exclude --select F821 --ignore-noqa \
  --exclude experiments/archive --exclude experiments/results \
  src/ experiments/ submissions/robust_current/ scripts/ tools/
.venv/bin/ruff check --force-exclude src/tac/tests/test_ci_ruff_scope.py
```

All focused checks passed. A broad all-tree Ruff statistics probe still reports
the existing legacy style baseline, but the runtime custom-noqa warning is no
longer injected into that baseline.

## Residual Risk

Broad Ruff remains intentionally non-blocking. Promoting more of it to a hard
gate should happen by rule family and ownership surface, not by one-shot
mechanical cleanup across generated artifacts, historical experiments, and
contest-runtime code.
