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

## Harvest classification

Generated at: 2026-05-11T11:34:00Z

Terminal status: `completed_negative_diagnostic_cuda_p100`

Raw artifact directory used for custody harvest:
`reports/raw/kaggle_pr106_yshift_score_table_v6_20260511/`

Kaggle output paging note:

- The Kaggle CLI `kernels output` downloaded only the first output page and
  printed a `Next page token`.
- The paginated Kaggle SDK API reported 4 pages / 1970 files.
- The custody harvest downloaded the score-table/build/eval artifacts directly
  through the paginated API and intentionally skipped only
  `pr106_yshift_score_table/yshift_run/eval/inflated/0.raw` (3.6 GB,
  rebuildable, manifest-hashed).
- After extracting hashes and terminal classification, local ignored raw
  harvest directories were removed to keep preflight's runtime-source baseline
  clean. Reproduction uses the command and hashes in this ledger, not a tracked
  raw artifact.

Score-table result:

- `score_table.npy` SHA-256:
  `d3f5deabe293c4e903883b901b0d6e71efb10fcf41f396a4bb58ab74dfed3f43`
- `candidate_grid.npy` SHA-256:
  `2b5856a525f1bed8a759f9a3c23cbbd6c4070f56cbf4576b394ab6780fb99ecf`
- table shape: `1200 x 343`
- strict-improvement frame count: `419`
- best improvement max/mean/min, no-rate objective:
  `0.01517552137374878 / 0.0009982656920328736 / 0.0`
- manifest says `ready_for_builder=true` and `score_claim=false`.

Built archive / exact diagnostic eval:

- archive: `pr106_yshift_sidechannel_archive.zip`
- archive bytes: `186663`
- archive SHA-256:
  `c0e187bf521383541edcb308697d1b247696773a3d6757041bb8072d0ff7e87a`
- source archive SHA-256:
  `cb9976bd33468475aac54a98c3baff996101c144b00e8d7e2c5107c86cda6182`
- runtime tree SHA-256:
  `33e15cab9cb09e8a3b4375a7c75a383d760aaa07dfee7e5ef94296fe387f7392`
- inflated output aggregate SHA-256:
  `ae4a38ee08677bc25aa11762b976ef8cf1ff07c6cc7ba0502f5b21a07d8f2361`
- device: Kaggle `Tesla P100-PCIE-16GB`, torch `2.4.1+cu121`,
  CUDA `12.1`, `gpu_t4_match=false`
- canonical score:
  `0.20965010844392395`
- components:
  - seg: `0.00066536` -> score contribution `0.066536`
  - pose: `0.00003543` -> score contribution `0.018822858443923972`
  - rate: `0.12429125`
- `contest_auth_eval.json` marks evidence grade `B`, lane tag
  `[diagnostic-auth-eval]`, `score_claim=false`,
  `promotion_eligible=false`, `rank_or_kill_eligible=false`.

Adversarial classification:

- This is a legitimate consumed-byte negative for the measured y-shift
  sidechannel configuration: the charged archive changed, inflate consumed the
  sidechannel, and the scorer returned a worse score.
- This is not a `[contest-CUDA]` promotable score claim because Kaggle P100 is
  a diagnostic CUDA axis, not the contest T4-equivalent axis.
- This does not kill PR106 sidechannels broadly. It falsifies the independent
  per-frame y-shift table reduction at step `1.0` on this archive/runtime.
- Preserved signal: the table identifies 419 locally improving frame
  corrections under the one-frame-at-a-time no-rate objective, but the composed
  charged archive loses after rate/coupling/exact eval. Future reactivation
  needs coupling-aware selection, joint latent+y-shift scoring, or a stricter
  Lagrangian that includes charged bits and pair interactions before build.

## Post-harvest hardening

Generated at: 2026-05-11T11:55:00Z

Repeated bug class found:

- `kaggle kernels output` exposes a next-page token but the wrapper did not
  follow it, so a naive harvest can miss terminal artifacts while flooding the
  transcript with staged source-tree paths.
- The PR106 y-shift/latent remote scripts printed
  `contest_cuda_score=... [contest-CUDA]` whenever `--device cuda` completed,
  even when `contest_auth_eval.json` correctly marked the run
  `score_claim=false`, `evidence_grade=B`, and `lane_tag=[diagnostic-auth-eval]`.
- `experiments/contest_auth_eval.py` marked non-T4 CUDA diagnostics with
  `score_axis="cuda"` instead of `diagnostic_cuda`, creating mixed semantics.

Durable fixes:

- `src/tac/deploy/kaggle/kaggle_output_ingest.py` now downloads Kaggle outputs
  through the paginated SDK API, supports include/exclude regexes, skips
  rebuildable raw inflated videos and cache files by default, and writes a
  compact `kaggle_output_download_manifest.json` with downloaded SHA-256s,
  counts by skip reason, and only a bounded skipped sample.
- `src/tac/auth_eval_schema.py::auth_eval_completion_summary` and
  `python -m tac.auth_eval_schema completion-summary` are the canonical compact
  completion-log surfaces for remote wrappers. They report the evaluator's own
  `score_claim`, `score_claim_valid`, `lane_tag`, `score_axis`,
  `evidence_grade`, hardware, and recomputed score fields.
- `scripts/remote_lane_pr106_yshift_sidechannel.sh` and
  `scripts/remote_lane_pr106_latent_sidecar.sh` now log
  `auth_eval_summary=...` from the canonical CLI instead of inventing a
  promotable label from the requested CUDA device or carrying duplicate
  embedded Python snippets.
- `experiments/contest_auth_eval.py` now emits `diagnostic_cuda` for non-T4
  CUDA diagnostic axes.

Authenticated quiet harvest proof:

- command:
  `PYTHONPATH=src uv run --with kaggle --with requests python -m tac.deploy.kaggle.kaggle_output_ingest --manifest /tmp/pact_kaggle_yshift_v6_manifest.json --download-dir reports/raw/kaggle_pr106_yshift_score_table_v6_quiet_20260511 --output-root reports/raw/kaggle_ingested_20260511 --download --include-output-pattern '^pr106_yshift_score_table/'`
- compact manifest:
  `reports/raw/kaggle_pr106_yshift_score_table_v6_quiet_20260511/kaggle_output_download_manifest.json`
- pages/files: 4 pages, 1970 files seen, 17 matched run artifacts, 18 files
  downloaded including log, 1953 skipped (`not_matched_by_include_patterns=1952`,
  `matched_exclude_patterns=1`)
- quiet artifact footprint: 3.8 MB instead of the noisy source-tree harvest;
  `inflated/0.raw` remains intentionally skipped and manifest-hashed by
  `inflated_outputs_manifest.json`.
- The quiet harvest was an ignored local custody check and was deleted after
  verification for the same preflight-baseline reason above.

Verification:

- `bash -n scripts/remote_lane_pr106_yshift_sidechannel.sh`
- `bash -n scripts/remote_lane_pr106_latent_sidecar.sh`
- `.venv/bin/python -m pytest src/tac/tests/test_kaggle_output_ingest.py src/tac/tests/test_auth_eval_schema.py src/tac/tests/test_contest_auth_eval.py src/tac/tests/test_kaggle_pr106_yshift_score_table.py src/tac/tests/test_kaggle_pr106_latent_score_table.py -q`
- `.venv/bin/ruff check --select F821,F401 experiments/contest_auth_eval.py src/tac/deploy/kaggle/kaggle_output_ingest.py src/tac/auth_eval_schema.py src/tac/tests/test_kaggle_output_ingest.py src/tac/tests/test_auth_eval_schema.py src/tac/tests/test_contest_auth_eval.py`
- `/usr/bin/time -p .venv/bin/python tools/all_lanes_preflight.py` -> all 29
  checks passed; wall clock `real 2.42s`.
