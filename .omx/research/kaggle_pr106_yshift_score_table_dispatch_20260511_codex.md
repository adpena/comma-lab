# Kaggle PR106 y-shift score-table dispatch — codex

Date: 2026-05-11

Lane: `lane_pr106_yshift_score_table`

Provider: Kaggle private GPU script kernel

Kernel: `adpena/comma-lab-pr106-yshift-score-table`

Kernel URL: <https://www.kaggle.com/code/adpena/comma-lab-pr106-yshift-score-table>

Job ID: `kaggle_pr106_yshift_score_table`

Dispatch claim:
- `tools/claim_lane_dispatch.py claim --lane-id lane_pr106_yshift_score_table --platform kaggle --instance-job-id kaggle_pr106_yshift_score_table --status active_dispatching`
- Claim status after push: active.

Bundle:
- Built with `tools/kaggle_build_pr106_yshift_score_table.py`.
- Local bundle path used for push: `/tmp/pact_kaggle_pr106_yshift_dispatch/comma-lab-pr106-yshift-score-table`.
- Bundle schema: `kaggle_pr106_yshift_score_table_bundle_v1`.
- Bundle embeds `.omx/state/active_lane_dispatch_claims.md`; bundle writer refused to emit before the matching active claim existed.

Purpose:
- Generate a CUDA scorer-backed PR106 y-shift candidate table on Kaggle.
- Build a charged `pr106_yshift_sidechannel_archive.zip` from the score table.
- Run exact CUDA `experiments/contest_auth_eval.py` via `scripts/remote_lane_pr106_yshift_sidechannel.sh`.

Claim discipline:
- `score_claim=false` until harvested `contest_auth_eval.json` is adjudicated.
- MPS is not used for auth eval.
- Scorers are used only by the compression/eval producer path, not at inflate time.
- CPU/CUDA/macOS axes remain distinct.

Initial status:
- `kaggle kernels push` returned: `Kernel version 1 successfully pushed`.
- `kaggle kernels status adpena/comma-lab-pr106-yshift-score-table` returned:
  `KernelWorkerStatus.RUNNING`.

Next harvest command:

```bash
uv run --with kaggle kaggle kernels output \
  adpena/comma-lab-pr106-yshift-score-table \
  -p reports/raw/kaggle_pr106_yshift_score_table_20260511
```

Expected classification after harvest:
- `completed_positive`: exact CUDA score beats PR106/A1-relevant comparator after formula recomputation and archive/runtime custody review.
- `completed_negative`: exact CUDA candidate regresses but leaves score-table evidence for trust-region updates.
- `failed_kaggle_runtime`: dependency, DALI/NVDEC, mount, upstream clone/LFS, or Kaggle environment failure.
- `failed_kaggle_push`: not applicable; push succeeded.

## v1 terminal update — 2026-05-11T08:37:52Z

`kaggle kernels logs adpena/comma-lab-pr106-yshift-score-table` returned
`KernelWorkerStatus.ERROR`.

Failure class: `failed_kaggle_runtime`.

Root cause: v1 assumed Kaggle script-kernel upload would expose copied repo
paths next to `run_kernel.py`. The runtime instead reached the stale mounted
wheel from `adpena/comma-lab-private-assets`, and that
`tac-1.0.5-py3-none-any.whl` did not include `tac.deploy.pr106_yshift`.

Terminal claim recorded:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim --force \
  --lane-id lane_pr106_yshift_score_table \
  --platform kaggle \
  --instance-job-id kaggle_pr106_yshift_score_table \
  --agent codex:gpt-5.5 \
  --status failed_kaggle_runtime \
  --notes "Kaggle v1 failed: script kernel uploaded only run_kernel.py; source paths unavailable; dataset wheel tac-1.0.5 missing tac.deploy.pr106_yshift"
```

This is not a method result and not a PR106 y-shift score claim.

Supersession: v2 source-bundle fix is recorded in
`kaggle_pr106_yshift_score_table_v2_source_bundle_20260511_codex.md`.
