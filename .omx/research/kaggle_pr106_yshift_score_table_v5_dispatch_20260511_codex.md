# Kaggle PR106 y-shift score-table v5 dispatch — codex

Date: 2026-05-11

Generated at: 2026-05-11T09:02:30Z

Lane: `lane_pr106_yshift_score_table`

Provider: Kaggle private GPU script kernel

Fresh kernel: `adpena/comma-lab-pr106-yshift-score-table-v5`

Kernel URL: <https://www.kaggle.com/code/adpena/comma-lab-pr106-yshift-score-table-v5>

Job ID: `kaggle_pr106_yshift_score_table_v5`

Why v5 exists:

- v4 fixed Kaggle source/archive materialization and reached y-shift CUDA stage.
- v4 failed because Kaggle assigned a Tesla P100 (`sm_60`) while its default
  PyTorch build supported only `sm_70+`.
- v5 adds a deterministic P100 fallback in the generated launcher: when the
  visible CUDA device is `sm_60` and the active torch wheel lacks `sm_60`, it
  installs pinned `torch==2.4.1+cu121`, `torchvision==0.19.1+cu121`, and
  `torchaudio==2.4.1+cu121` from the official PyTorch CUDA 12.1 wheel index,
  then re-execs once before scorer work.

Dispatch claim:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id lane_pr106_yshift_score_table \
  --platform kaggle \
  --instance-job-id kaggle_pr106_yshift_score_table_v5 \
  --status active_dispatching
```

Source dataset:

- ref: `adpena/comma-lab-pr106-yshift-source`
- URL: <https://www.kaggle.com/datasets/adpena/comma-lab-pr106-yshift-source>
- updated with v5 source and claim ledger via `kaggle datasets version`
- dataset status after version: `ready`

Kernel bundle:

- local staging: `/tmp/pact_kaggle_pr106_yshift_dispatch_v5/kernels/comma-lab-pr106-yshift-score-table-v5`
- schema: `kaggle_pr106_yshift_score_table_bundle_v2`
- dataset sources:
  - `adpena/comma-lab-pr106-yshift-source`
  - `adpena/comma-lab-private-assets`
- `score_claim=false`
- `promotion_requires=contest_auth_eval_json_adjudication`

Push/status:

- `kaggle kernels push -p /tmp/pact_kaggle_pr106_yshift_dispatch_v5/kernels/comma-lab-pr106-yshift-score-table-v5`
  returned `Kernel version 1 successfully pushed`.
- `kaggle kernels status adpena/comma-lab-pr106-yshift-score-table-v5`
  returned `KernelWorkerStatus.RUNNING`.
- Initial logs had no emitted rows.

Claim discipline:

- This is not a score claim.
- MPS is not used for auth eval.
- CPU, CUDA, and macOS advisory axes remain distinct.
- Scorers are used by the CUDA producer/eval path, not by `inflate.py`.

Harvest command after terminal status:

```bash
uv run --with kaggle kaggle kernels output \
  adpena/comma-lab-pr106-yshift-score-table-v5 \
  -p reports/raw/kaggle_pr106_yshift_score_table_v5_20260511
```

Expected classification after harvest:

- `completed_positive`: exact CUDA score improves after component/formula
  recomputation and archive/runtime custody review.
- `completed_negative`: exact CUDA score regresses but score table and trust
  region are preserved.
- `failed_kaggle_runtime`: dependency, DALI/NVDEC, mount, upstream clone/LFS, or
  Kaggle environment failure.
- `failed_kaggle_gpu_arch_unsupported`: pinned fallback still lacks `sm_60` or
  cannot run on P100.
- `indeterminate`: terminal output lacks contest-auth JSON or custody closure.
