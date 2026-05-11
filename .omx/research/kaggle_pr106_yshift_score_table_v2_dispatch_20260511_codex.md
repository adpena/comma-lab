# Kaggle PR106 y-shift score-table v2 dispatch — codex

Date: 2026-05-11

Generated at: 2026-05-11T08:41:55Z

Lane: `lane_pr106_yshift_score_table`

Provider: Kaggle private GPU script kernel

Kernel: `adpena/comma-lab-pr106-yshift-score-table`

Kernel URL: <https://www.kaggle.com/code/adpena/comma-lab-pr106-yshift-score-table>

Job ID: `kaggle_pr106_yshift_score_table_v2`

Dispatch claim:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id lane_pr106_yshift_score_table \
  --platform kaggle \
  --instance-job-id kaggle_pr106_yshift_score_table_v2 \
  --status active_dispatching
```

Source dataset:

- ref: `adpena/comma-lab-pr106-yshift-source`
- URL: <https://www.kaggle.com/datasets/adpena/comma-lab-pr106-yshift-source>
- dataset status after create: `ready`
- source bundle: `pact_pr106_yshift_source_bundle.tar.gz`
- source bundle size: 5.6 MiB local staging

Kernel bundle:

- local staging: `/tmp/pact_kaggle_pr106_yshift_dispatch_v2/kernels/comma-lab-pr106-yshift-score-table`
- schema: `kaggle_pr106_yshift_score_table_bundle_v2`
- dataset sources:
  - `adpena/comma-lab-pr106-yshift-source`
  - `adpena/comma-lab-private-assets`
- `score_claim=false`
- `promotion_requires=contest_auth_eval_json_adjudication`

Push/status:

- `kaggle datasets create -p /tmp/pact_kaggle_pr106_yshift_dispatch_v2/source_dataset`
  created the private source dataset.
- `kaggle kernels push -p /tmp/pact_kaggle_pr106_yshift_dispatch_v2/kernels/comma-lab-pr106-yshift-score-table`
  returned `Kernel version 2 successfully pushed`.
- `kaggle kernels status adpena/comma-lab-pr106-yshift-score-table`
  returned `KernelWorkerStatus.RUNNING`.

Claim discipline:

- This is not a score claim.
- MPS is not used for auth eval.
- CPU, CUDA, and macOS advisory axes remain distinct.
- Scorers are used by the CUDA producer/eval path, not by `inflate.py`.

Harvest command after terminal status:

```bash
uv run --with kaggle kaggle kernels output \
  adpena/comma-lab-pr106-yshift-score-table \
  -p reports/raw/kaggle_pr106_yshift_score_table_v2_20260511
```

Expected classification after harvest:

- `completed_positive`: exact CUDA score improves after component/formula
  recomputation and archive/runtime custody review.
- `completed_negative`: exact CUDA score regresses but score table and trust
  region are preserved.
- `failed_kaggle_runtime`: dependency, DALI/NVDEC, mount, upstream clone/LFS, or
  Kaggle environment failure.
- `indeterminate`: terminal output lacks contest-auth JSON or custody closure.

## Terminal update — 2026-05-11T08:43:00Z

`kaggle kernels status adpena/comma-lab-pr106-yshift-score-table` returned
`KernelWorkerStatus.ERROR`.

Logs show failure before scorer work:

```text
FileNotFoundError: required source bundle 'pact_pr106_yshift_source_bundle.tar.gz'
not found under ['/kaggle/src', '/kaggle/input']
```

Classification: `failed_kaggle_runtime`.

Interpretation: the source dataset itself was created and reached `ready`, but
the existing kernel slug did not mount the new dataset source for version 2.
This is a provider attachment/stale-kernel-metadata failure, not a method result
and not a PR106 y-shift score claim.

Terminal claim recorded with the same lane/job:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim --force \
  --lane-id lane_pr106_yshift_score_table \
  --platform kaggle \
  --instance-job-id kaggle_pr106_yshift_score_table_v2 \
  --status failed_kaggle_runtime
```

Next action: push the same source-bundle contract under a fresh kernel slug so
Kaggle cannot reuse stale dataset-source attachment state.
