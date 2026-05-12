# Preflight GHA worker fanout greenup - 2026-05-12

## Problem

CI run `25766705005` on commit `d312363c` failed the 30s developer-preflight
watchdog in the `Preflight validators (5 layers)` step. The failure was not a
semantic preflight violation; it was scheduling:

- `check_state_writers_strict_load_for_mutating_path`: 14.125s, passed
- `check_dispatch_cli_shell_hazards`: 12.554s, passed
- `check_no_bare_writes_to_shared_state`: 8.768s, passed
- `check_no_mps_fallback_default`: 7.083s, still running at timeout
- `check_authoritative_tag_requires_custody_metadata`: 5.422s, passed

With the GitHub Actions default capped at 2 workers, broad full-tree scans
started in long waves and the watchdog fired before the second wave completed.

## Fix

`src/tac/preflight.py::_preflight_parallel_worker_count()` now defaults to 8
workers on both local machines and GitHub-hosted runners. SourceIndex prewarm
remains opt-in, so the fix is fanout of independent checks, not a hidden
reduction in coverage.

## Verification

Local GitHub-mode cold preflight proof:

```bash
env GITHUB_ACTIONS=true HOME=/tmp/pact-ci-home-scheduled-8w \
  PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE=1 \
  PYTHONPATH=src:upstream:$PWD \
  .venv/bin/python -m tac.preflight \
  --timings-json /tmp/pact_preflight_ci_scheduled_8w.json
# PREFLIGHT PASSED; wall_elapsed_s = 8.192850
```

The slow checks all still ran and passed. The top recorded checks were:

- `check_state_writers_strict_load_for_mutating_path`: 8.094965s
- `check_authoritative_tag_requires_custody_metadata`: 7.399235s
- `check_no_bare_writes_to_shared_state`: 7.344986s
- `check_continual_learning_writes_use_lock`: 5.839871s
- `check_no_mps_fallback_default`: 5.805328s

## Evidence boundary

This is a DX wall-clock scheduling fix, not a preflight-scope reduction. If CI
still exceeds 30s after this, the next correct fix is one-pass SourceIndex
fusion for the broad source scanners, not disabling checks.

