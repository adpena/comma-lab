# Modal auth-eval detached claimed recovery (2026-05-11)

## Summary

Codex centralized Modal auth-eval dispatch/recovery custody into
`src/tac/deploy/modal/auth_eval.py` and wired both CUDA and CPU auth-eval
wrappers through the same contract:

- dispatch claim required before Modal provider work;
- optional `--detach` launch via Modal `.spawn()`;
- canonical `modal_auth_eval_spawn.json` + `modal_call_id.txt` sentinels;
- canonical harvest through `tools/recover_modal_auth_eval.py`;
- recovered `contest_auth_eval.json` and `inflated_outputs_manifest.json`
  preserved in the local output directory;
- terminal dispatch-claim rows written at recovery.

This is score-lowering infrastructure, not a score claim. It removes a custody
blocker for paired PR103/PR106 CUDA-vs-CPU raw-output mechanism work and for
future exact CUDA dispatches that should not tie up local wall clock.

## Rigor

The CPU wrapper now supports uploaded submission runtimes, matching the CUDA
wrapper. This matters because CPU/CUDA drift analysis is not apples-to-apples
unless both axes consume the same archive bytes and the same runtime tree.

Runtime transport zipping is centralized and fail-closed:

- deterministic ZIP metadata;
- Python bytecode and host metadata filtered;
- symlinks, hidden files, and secret-looking files rejected.

## Verification

```bash
.venv/bin/python -m py_compile \
  src/tac/deploy/modal/auth_eval.py \
  experiments/modal_auth_eval.py \
  experiments/modal_auth_eval_cpu.py \
  tools/recover_modal_auth_eval.py

.venv/bin/python -m pytest -q \
  src/tac/tests/test_modal_auth_eval.py \
  src/tac/tests/test_modal_auth_eval_recovery.py \
  src/tac/tests/test_plan_dual_device_auth_eval.py
```

Result: `31 passed`.

## Next score-lowering use

Launch a claimed detached CUDA auth eval for the PR103-on-PR106 packet with
uploaded runtime `submissions/pr103_pr106_final_runtime`, harvest the raw
`inflated_outputs_manifest.json`, then pair it with a claimed detached Modal
CPU run using the same archive/runtime. That closes the current mechanism gap:
whether the observed CPU/CUDA score difference is renderer-output drift or
scorer-device drift.
