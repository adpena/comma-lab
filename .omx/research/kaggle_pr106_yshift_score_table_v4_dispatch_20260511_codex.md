# Kaggle PR106 y-shift score-table v4 dispatch — codex

Date: 2026-05-11

Generated at: 2026-05-11T08:51:39Z

Lane: `lane_pr106_yshift_score_table`

Provider: Kaggle private GPU script kernel

Fresh kernel: `adpena/comma-lab-pr106-yshift-score-table-v4`

Kernel URL: <https://www.kaggle.com/code/adpena/comma-lab-pr106-yshift-score-table-v4>

Job ID: `kaggle_pr106_yshift_score_table_v4`

Why v4 exists:

- v1/v2/v3 all failed before scorer work because of Kaggle source transport,
  not because of the PR106 y-shift method.
- `kaggle datasets files adpena/comma-lab-pr106-yshift-source` proved Kaggle
  expands the uploaded tarball into `pact_pr106_yshift_source_bundle/...` and
  expands the embedded source archive into `inputs/pr106_archive/0.bin`.
- v4 launcher supports both raw tarball and expanded-tree mount shapes, and
  reconstructs `inputs/pr106_archive.zip` deterministically when Kaggle expands
  the nested ZIP.

Dispatch claim:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id lane_pr106_yshift_score_table \
  --platform kaggle \
  --instance-job-id kaggle_pr106_yshift_score_table_v4 \
  --status active_dispatching
```

Source dataset:

- ref: `adpena/comma-lab-pr106-yshift-source`
- URL: <https://www.kaggle.com/datasets/adpena/comma-lab-pr106-yshift-source>
- updated with v4 source and claim ledger via `kaggle datasets version`
- dataset status after version: `ready`
- source bundle: `pact_pr106_yshift_source_bundle.tar.gz`

Kernel bundle:

- local staging: `/tmp/pact_kaggle_pr106_yshift_dispatch_v4/kernels/comma-lab-pr106-yshift-score-table-v4`
- schema: `kaggle_pr106_yshift_score_table_bundle_v2`
- dataset sources:
  - `adpena/comma-lab-pr106-yshift-source`
  - `adpena/comma-lab-private-assets`
- `score_claim=false`
- `promotion_requires=contest_auth_eval_json_adjudication`

Push/status:

- `kaggle kernels push -p /tmp/pact_kaggle_pr106_yshift_dispatch_v4/kernels/comma-lab-pr106-yshift-score-table-v4`
  returned `Kernel version 1 successfully pushed`.
- `kaggle kernels status adpena/comma-lab-pr106-yshift-score-table-v4`
  returned `KernelWorkerStatus.RUNNING`.
- `kaggle kernels logs adpena/comma-lab-pr106-yshift-score-table-v4`
  had no emitted log rows at initial check.

Claim discipline:

- This is not a score claim.
- MPS is not used for auth eval.
- CPU, CUDA, and macOS advisory axes remain distinct.
- Scorers are used by the CUDA producer/eval path, not by `inflate.py`.

Harvest command after terminal status:

```bash
uv run --with kaggle kaggle kernels output \
  adpena/comma-lab-pr106-yshift-score-table-v4 \
  -p reports/raw/kaggle_pr106_yshift_score_table_v4_20260511
```

Expected classification after harvest:

- `completed_positive`: exact CUDA score improves after component/formula
  recomputation and archive/runtime custody review.
- `completed_negative`: exact CUDA score regresses but score table and trust
  region are preserved.
- `failed_kaggle_runtime`: dependency, DALI/NVDEC, mount, upstream clone/LFS, or
  Kaggle environment failure.
- `indeterminate`: terminal output lacks contest-auth JSON or custody closure.
