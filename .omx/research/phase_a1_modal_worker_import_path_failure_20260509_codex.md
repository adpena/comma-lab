# Phase A1 Modal Worker Import-Path Failure - 2026-05-09

## Classification

- Lane: `track1_phase_a1_score_gradient`
- Instance/job id: `track1_phase_a1_score_gradient_latentalign_lr2e6_20260509T011929Z_modal`
- Modal call id: `fc-01KR553TPH27G73HMHH56MDZH0`
- Status: `failed_modal_recover_exception`
- Failure class: remote worker import-path/runtime bug
- Score claim: `false`
- Promotion eligible: `false`

## Evidence

The recovery command:

```bash
.venv/bin/python experiments/modal_phase_a1_score_gradient_pr101.py recover \
  --label track1_phase_a1_score_gradient_latentalign_lr2e6_20260509T011929Z_modal
```

returned:

```text
FATAL: recover failed: ModuleNotFoundError: No module named 'tac'
```

The failure was raised by the Modal function result, not by a local missing
dependency. The remote worker imports `tac.submission_archive.safe_extract_zip`
inside `run_phase_a1_t4()` before spawning subprocesses that receive the
explicit `PYTHONPATH`. The worker process itself did not have
`/workspace/pact/src` on `sys.path`.

## Fix

`experiments/modal_phase_a1_score_gradient_pr101.py` now installs both local
and Modal-worker repo import roots in `sys.path` at module import time:

- `<repo>/src`
- `<repo>/upstream`
- `<repo>`
- `/workspace/pact/src`
- `/workspace/pact/upstream`
- `/workspace/pact`

The recover path also now closes active Modal claims if `FunctionCall.get()`
raises a non-timeout exception, preventing failed Modal runs from remaining as
phantom active dispatches.

Regression test:

```bash
.venv/bin/python -m pytest tests/test_modal_phase_a1_recover_paths.py -q
```

## Claim Closure

The recover command appended a terminal claim row:

```text
failed_modal_recover_exception
```

with notes that include the Modal call id and exception type. This closes the
`2026-05-09T01:19:59Z` active dispatch row for this instance/job id.

## Reactivation

Relaunch A1 only with a fresh label after the import-path fix is committed and
pushed. The next dispatch should preserve the same conservative latent-aligned
training configuration unless a separate score-domain reason changes it.
