# HDM8 Modal Postfilter Import Closure Fix - 2026-05-15

## Scope

Hardening fix for the CUDA-in-loop HDM8 postfilter/selector sweep actuator.
This is infrastructure repair, not a score claim.

## Failure

First relaunch attempt:

- lane_id: `lane_hdm8_cuda_prefix_selector_aggressive_v1_20260515`
- instance_job_id: `hdm8_cuda_prefix_aggressive_v1_20260515T022426Z`
- Modal call_id: `fc-01KRMQ6M9W4C2D6N7H2748X926`
- output_dir:
  `experiments/results/modal_hdm8_postfilter_sweep/hdm8_cuda_prefix_aggressive_v1_20260515T022426Z`

Remote runner log showed:

```text
ImportError: cannot import name 'modal_auth_eval' from 'experiments' (unknown location)
```

Classification:

- provider/import-closure bug
- no score claim
- no model/lane negative
- terminal dispatch ledger row:
  `stale_superseded_by_modal_import_mount_fix`

## Fix

`experiments/modal_hdm8_postfilter_sweep.py` now:

- resolves `REPO_ROOT` to `/workspace/pact` inside the Modal image when mounted;
- mounts `experiments/__init__.py`;
- mounts `experiments/modal_auth_eval.py`;
- keeps the HDM8 sweep as a narrow proxy-prefix CUDA scorer with
  `score_claim=false` and `promotion_eligible=false`.

Regression:

- `src/tac/tests/test_modal_hdm8_postfilter_sweep.py::test_modal_runtime_dependency_mounts_include_import_closure`

Verification:

```bash
.venv/bin/ruff check experiments/modal_hdm8_postfilter_sweep.py src/tac/tests/test_modal_hdm8_postfilter_sweep.py
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_modal_hdm8_postfilter_sweep.py -q
```

Result:

- ruff: passed
- pytest: `12 passed`

## Reactivation

Relaunch the same HDM8 CUDA-prefix selector sweep after this patch from a clean
main commit, using a new `instance_job_id`.
