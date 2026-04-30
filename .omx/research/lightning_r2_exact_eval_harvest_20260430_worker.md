# Lightning r2 Exact-Eval Harvest - 2026-04-30

Worker: Codex
Scope: `owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r2`

No score claim is made in this report. r2 failed before archive copy, CUDA auth
eval, `contest_auth_eval.json`, and adjudication artifacts existed.

## Status

- Local strict supply-chain scan passed on 2026-04-30T20:07:16Z:
  `lightning-sdk=2026.4.10`, `lightning=null`, `violation_count=0`.
- SDK status refreshes moved r2 from `Running` to `Failed`.
- Latest inspected SDK state:
  - SDK job:
    `owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x-r2`
  - local queue job:
    `owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r2`
  - status: `Failed`
  - `completed_at`: `null`
  - `failed_at`: `null`
  - `failure_reason`: `null`
  - machine: `T4`
  - SDK artifact path:
    `/teamspace/jobs/owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x-r2/artifacts`
  - command SHA:
    `cf1ab6a1c81b0fa69273007ec4c3efcd54f25bdce1f8f3856bb2bb3bddc21e08`

## Artifact Check

- Expected archive identity in queue:
  - SHA-256:
    `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`
  - bytes: `686557`
- Local candidate source archive still matches that identity:
  `experiments/results/lane_g_v3_owv3_byte_plan_sweep_worker_b/best_byte_feasible/archive_lane_g_v3_owv3.zip`.
- SDK artifact tree for the r2 artifact path returned zero files.
- SDK logs are available only after terminal status. After r2 failed, logs ended
  with:

```text
OSError: [Errno 30] Read-only file system:
'/teamspace/jobs/owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x-r2/artifacts/lightning_queue_metadata.json'
```

## Diagnosis

r2 is an infrastructure failure before eval. The queue command wrote runtime
artifacts directly under the SDK-reported `/teamspace/jobs/<job>/artifacts`
path. Live SDK evidence shows that path is a read-only artifact view inside a
Studio-backed Batch Job. The run failed while writing
`lightning_queue_metadata.json`, before archive copy, SHA/byte preflight,
`contest_auth_eval.py --device cuda`, and adjudication.

No exact artifacts were harvested. No `contest_auth_eval.json` or
`adjudication_provenance.json` exists for r2, so the expected SHA/bytes and
adjudication gate could not be applied to an eval artifact.

## Commands Run

```bash
.venv/bin/python scripts/scan_lightning_supply_chain.py --strict --quiet
.venv/bin/python scripts/launch_lightning_batch_job.py refresh-status \
  --job-name owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r2 \
  --teamspace comma-lab \
  --user adpena
shasum -a 256 experiments/results/lane_g_v3_owv3_byte_plan_sweep_worker_b/best_byte_feasible/archive_lane_g_v3_owv3.zip
wc -c < experiments/results/lane_g_v3_owv3_byte_plan_sweep_worker_b/best_byte_feasible/archive_lane_g_v3_owv3.zip
```

SDK log/artifact inspection used `lightning_sdk.Job` and `Teamspace` directly;
logs were used only for infrastructure diagnosis, not score parsing.

## Robustness Patch

Changed files:

- `.omx/state/lightning_batch_jobs.json` (SDK status refresh only)
- `src/tac/deploy/lightning/batch_jobs.py`
- `scripts/launch_lightning_batch_job.py`
- `src/tac/tests/test_lightning_batch_jobs.py`
- `.omx/research/lightning_r2_exact_eval_harvest_20260430_worker.md`

Patch summary:

- Added `default_exact_eval_output_dir()` so exact-eval jobs default to
  `<repo-dir>/experiments/results/lightning_batch/<job-name>`, a writable
  Studio workspace path.
- Added `lightning_sdk_artifact_path()` and recorded the SDK artifact view
  separately in queue records.
- Added fail-closed validation rejecting `/teamspace/jobs/...` as an exact-eval
  command output target.
- Updated CLI help to describe the writable default.
- Added focused tests for writable default output, SDK artifact path recording,
  and rejection of the read-only SDK artifact view.

## Verification

Passed:

```bash
.venv/bin/python -m py_compile src/tac/deploy/lightning/batch_jobs.py scripts/launch_lightning_batch_job.py
.venv/bin/python -m pytest src/tac/tests/test_lightning_batch_jobs.py -q
for f in src/tac/deploy/lightning/batch_jobs.py scripts/launch_lightning_batch_job.py src/tac/tests/test_lightning_batch_jobs.py .omx/research/lightning_r2_exact_eval_harvest_20260430_worker.md; do
  out=$(git diff --check --no-index -- /dev/null "$f" || true)
  if [ -n "$out" ]; then printf '%s\n' "$out"; exit 1; fi
done
```

Pytest result: `20 passed in 0.20s`.

No touched shell scripts required `bash -n`.
