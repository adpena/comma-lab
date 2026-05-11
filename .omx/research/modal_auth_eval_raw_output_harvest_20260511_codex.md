# Modal auth-eval raw-output harvest hardening (2026-05-11)

## Scope

Follow-up to the PR103-on-PR106 CPU/CUDA exact-pair drift analysis. The paired
score artifacts share the same archive and runtime content, but the mechanism
classification is still `same_archive_runtime_raw_outputs_unmeasured` because
the harvested Modal artifacts did not include `inflated_outputs_manifest.json`.

## Fix

`experiments/contest_auth_eval.py` already writes
`inflated_outputs_manifest.json` and embeds its aggregate hash into provenance.
The missing link was the Modal wrapper harvest surface:

- `experiments/modal_auth_eval.py` now harvests
  `work_dir/inflated_outputs_manifest.json`;
- `experiments/modal_auth_eval_cpu.py` now harvests the same file;
- `src/tac/tests/test_modal_auth_eval.py` locks the CUDA harvest behavior and
  verifies both wrappers reference the raw-output manifest path.

This is a custody fix only. It does not create a score claim and does not
retroactively add raw-output hashes to older Modal runs.

## Verification

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_modal_auth_eval.py \
  src/tac/tests/test_contest_auth_eval.py \
  src/tac/tests/test_analyze_cpu_cuda_eval_drift.py \
  src/tac/tests/test_plan_dual_device_auth_eval.py \
  src/tac/tests/test_harvest_cuda_cpu_axis_profile_registry.py
```

Result: `71 passed`.

## Score-lowering relevance

Future paired CUDA/CPU auth evals can now answer the question that matters for
exploiting the axis gap:

- if raw-output aggregate hashes match, score drift localizes to loader/scorer
  device behavior;
- if raw-output aggregate hashes differ, score drift localizes to
  runtime/inflate device behavior.

That distinction determines whether to spend effort on runtime arithmetic,
DALI/PyAV loader parity, or scorer-kernel-aware training losses.
