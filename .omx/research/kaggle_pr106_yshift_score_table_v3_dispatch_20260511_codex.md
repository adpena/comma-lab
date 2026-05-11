# Kaggle PR106 y-shift score-table v3 dispatch — codex

Date: 2026-05-11

Generated at: 2026-05-11T08:45:08Z

Lane: `lane_pr106_yshift_score_table`

Provider: Kaggle private GPU script kernel

Fresh kernel: `adpena/comma-lab-pr106-yshift-score-table-v3`

Kernel URL: <https://www.kaggle.com/code/adpena/comma-lab-pr106-yshift-score-table-v3>

Job ID: `kaggle_pr106_yshift_score_table_v3`

Why v3 exists:

- v1 failed because Kaggle did not expose copied source files next to
  `run_kernel.py`, and the stale private-assets wheel lacked
  `tac.deploy.pr106_yshift`.
- v2 fixed source transport with a dataset tarball, but reused the old kernel
  slug. The run failed before scorer work because the new source dataset was
  not mounted under `/kaggle/input`.
- v3 uses a fresh kernel slug with the same source-bundle contract so Kaggle
  must attach dataset sources from first creation.

Dispatch claim:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id lane_pr106_yshift_score_table \
  --platform kaggle \
  --instance-job-id kaggle_pr106_yshift_score_table_v3 \
  --status active_dispatching
```

Source dataset:

- ref: `adpena/comma-lab-pr106-yshift-source`
- URL: <https://www.kaggle.com/datasets/adpena/comma-lab-pr106-yshift-source>
- updated with v3 claim ledger via `kaggle datasets version`
- dataset status after version: `ready`
- source bundle: `pact_pr106_yshift_source_bundle.tar.gz`

Kernel bundle:

- local staging: `/tmp/pact_kaggle_pr106_yshift_dispatch_v3/kernels/comma-lab-pr106-yshift-score-table-v3`
- schema: `kaggle_pr106_yshift_score_table_bundle_v2`
- dataset sources:
  - `adpena/comma-lab-pr106-yshift-source`
  - `adpena/comma-lab-private-assets`
- `score_claim=false`
- `promotion_requires=contest_auth_eval_json_adjudication`

Push/status:

- `kaggle kernels push -p /tmp/pact_kaggle_pr106_yshift_dispatch_v3/kernels/comma-lab-pr106-yshift-score-table-v3`
  returned `Kernel version 1 successfully pushed`.
- `kaggle kernels status adpena/comma-lab-pr106-yshift-score-table-v3`
  returned `KernelWorkerStatus.RUNNING`.
- `kaggle kernels logs adpena/comma-lab-pr106-yshift-score-table-v3`
  had no emitted log rows at initial check.

Claim discipline:

- This is not a score claim.
- MPS is not used for auth eval.
- CPU, CUDA, and macOS advisory axes remain distinct.
- Scorers are used by the CUDA producer/eval path, not by `inflate.py`.

Harvest command after terminal status:

```bash
uv run --with kaggle kaggle kernels output \
  adpena/comma-lab-pr106-yshift-score-table-v3 \
  -p reports/raw/kaggle_pr106_yshift_score_table_v3_20260511
```

Expected classification after harvest:

- `completed_positive`: exact CUDA score improves after component/formula
  recomputation and archive/runtime custody review.
- `completed_negative`: exact CUDA score regresses but score table and trust
  region are preserved.
- `failed_kaggle_runtime`: dependency, DALI/NVDEC, mount, upstream clone/LFS, or
  Kaggle environment failure.
- `indeterminate`: terminal output lacks contest-auth JSON or custody closure.

## Terminal update — 2026-05-11T08:48:00Z

`kaggle kernels status adpena/comma-lab-pr106-yshift-score-table-v3` returned
`KernelWorkerStatus.ERROR`.

Logs again showed failure before scorer work:

```text
FileNotFoundError: required source bundle 'pact_pr106_yshift_source_bundle.tar.gz'
not found under ['/kaggle/src', '/kaggle/input']
```

Additional provider evidence:

```bash
kaggle datasets files adpena/comma-lab-pr106-yshift-source
```

showed Kaggle expanded the uploaded tarball into
`pact_pr106_yshift_source_bundle/...` and also exposed the embedded PR106
archive as `inputs/pr106_archive/0.bin` rather than
`inputs/pr106_archive.zip`.

Classification: `failed_kaggle_runtime`.

Interpretation: the source dataset is present, but Kaggle materializes archive
uploads as expanded directories. The method did not run; this is not a PR106
y-shift result and not a score claim.

Terminal claim recorded with the same lane/job:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim --force \
  --lane-id lane_pr106_yshift_score_table \
  --platform kaggle \
  --instance-job-id kaggle_pr106_yshift_score_table_v3 \
  --status failed_kaggle_runtime
```

Next action: launcher v4 supports both source tarballs and pre-expanded Kaggle
dataset trees, and reconstructs `inputs/pr106_archive.zip` deterministically
from `inputs/pr106_archive/0.bin` when Kaggle expands the nested ZIP.
