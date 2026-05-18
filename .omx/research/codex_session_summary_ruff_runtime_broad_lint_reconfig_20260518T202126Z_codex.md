# Codex Session Summary: Ruff Runtime Broad-Lint Reconfiguration

timestamp_utc: 2026-05-18T20:21:26Z
agent: codex
score_claim: false
research_only: false

## Landed

- Reconfigured project Ruff broad lint to exclude the giant contest runtime
  `submissions/robust_current/inflate_renderer.py`.
- Kept the hard undefined-name safety gate unchanged: CI and preflight still
  run isolated `F821` with `--ignore-noqa` over `submissions/robust_current/`.
- Added regression tests proving the broad runtime exclude does not weaken the
  isolated F821 path.
- Recorded adversarial finding in
  `.omx/research/codex_findings_ruff_runtime_broad_lint_reconfig_20260518T202126Z_codex.md`.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_ci_ruff_scope.py -q`
- `.venv/bin/ruff check --force-exclude --select RUF100 src/tac/tests/test_ci_ruff_scope.py submissions/robust_current/inflate_renderer.py`
- `.venv/bin/ruff check --isolated --force-exclude --select F821 --ignore-noqa --exclude experiments/archive --exclude experiments/results src/ experiments/ submissions/robust_current/ scripts/ tools/`
- `.venv/bin/ruff check --force-exclude src/tac/tests/test_ci_ruff_scope.py`
- `git diff --check`

## Next

Resume the canonical queue at OP-SYN-1 or v2 ITEM_7. Do not promote
project-wide Ruff to a hard gate until rule families are burned down by owner
surface; the current broad baseline remains intentionally non-blocking.
