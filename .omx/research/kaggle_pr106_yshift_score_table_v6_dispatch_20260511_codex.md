# Kaggle PR106 y-shift score-table v6 dispatch — codex

Date: 2026-05-11

Generated at: 2026-05-11T09:14:08Z

Lane: `lane_pr106_yshift_score_table`

Provider: Kaggle private GPU script kernel

Fresh kernel: `adpena/comma-lab-pr106-yshift-score-table-v6`

Kernel URL: <https://www.kaggle.com/code/adpena/comma-lab-pr106-yshift-score-table-v6>

Job ID: `kaggle_pr106_yshift_score_table_v6`

Why v6 exists:

- v5 proved the P100 PyTorch fallback reaches real scorer execution.
- v5 failed at the default score-table batch shape with CUDA OOM.
- v6 preserves the same candidate grid and method, but reduces runtime memory:
  - `batch_pairs=2`
  - `candidate_batch_size=16`
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Dispatch claim:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id lane_pr106_yshift_score_table \
  --platform kaggle \
  --instance-job-id kaggle_pr106_yshift_score_table_v6 \
  --status active_dispatching
```

Source dataset:

- ref: `adpena/comma-lab-pr106-yshift-source`
- URL: <https://www.kaggle.com/datasets/adpena/comma-lab-pr106-yshift-source>
- updated with v6 source and claim ledger via `kaggle datasets version`
- dataset status after version: `ready`

Kernel bundle:

- local staging: `/tmp/pact_kaggle_pr106_yshift_dispatch_v6/kernels/comma-lab-pr106-yshift-score-table-v6`
- schema: `kaggle_pr106_yshift_score_table_bundle_v2`
- dataset sources:
  - `adpena/comma-lab-pr106-yshift-source`
  - `adpena/comma-lab-private-assets`
- `score_claim=false`
- `promotion_requires=contest_auth_eval_json_adjudication`

Push/status:

- `kaggle kernels push -p /tmp/pact_kaggle_pr106_yshift_dispatch_v6/kernels/comma-lab-pr106-yshift-score-table-v6`
  returned `Kernel version 1 successfully pushed`.
- `kaggle kernels status adpena/comma-lab-pr106-yshift-score-table-v6`
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
  adpena/comma-lab-pr106-yshift-score-table-v6 \
  -p reports/raw/kaggle_pr106_yshift_score_table_v6_20260511
```

Expected classification after harvest:

- `completed_positive`: exact CUDA score improves after component/formula
  recomputation and archive/runtime custody review.
- `completed_negative`: exact CUDA score regresses but score table and trust
  region are preserved.
- `failed_kaggle_cuda_oom`: smaller P100 batch still does not fit.
- `failed_kaggle_runtime`: dependency, DALI/NVDEC, mount, upstream clone/LFS, or
  Kaggle environment failure.
- `indeterminate`: terminal output lacks contest-auth JSON or custody closure.
