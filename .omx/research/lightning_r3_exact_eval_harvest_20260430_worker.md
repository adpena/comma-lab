# Lightning r3 Exact-Eval Harvest - 2026-04-30

Worker: Codex
Scope: `owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r3`

No score claim is made in this report. r3 failed before
`contest_auth_eval.json`, `contest_auth_eval.adjudicated.json`, and
`adjudication_provenance.json` existed.

## Status

- SDK job:
  `owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x-r3`
- Latest SDK refresh: `2026-04-30T20:26:38Z`
- Status: `Failed`
- Machine requested in queue metadata: `g4dn.2xlarge`
- SDK job attribute machine: `T4`
- SDK artifact path:
  `/teamspace/jobs/owv3-byte-feasible-exact-cuda-20260430-codex-lightning-t4-g4dn2x-r3/artifacts`
- Queue remote output dir:
  `/teamspace/studios/this_studio/pact/experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r3`

## Artifact Check

- Queue expected archive SHA-256:
  `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`
- Queue expected archive bytes: `686557`
- Mirrored diagnostic archive:
  `experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r3/archive.zip`
- Local mirrored archive identity matches expected SHA and bytes.
- `validate-artifacts --require-adjudication` correctly failed because
  `contest_auth_eval.json` is absent.
- The Studio SSH view of `remote_output_dir` did not exist, but the SDK
  artifact view contained the nested job filesystem path:
  `.../artifacts/pact/experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r3`.
- Mirrored locally for diagnosis:
  - `archive.zip`
  - `auth_eval.log`
  - `lightning_queue_metadata.json`
  - `lightning_runner_preflight.json`
  - `lightning_supply_chain_scan.json`
  - `eval_work/provenance.json`
- Not mirrored: `eval_work/inflated/0.raw` (`3.5G`) and duplicate extracted
  payload files. The archive already preserves the compressed payload custody.

## Root Cause

R3 is an infrastructure/runtime failure, not compression evidence.

The job passed strict Lightning supply-chain scan, CUDA/T4 runner preflight,
archive SHA/byte preflight, archive extraction, inflate, and raw-output strict
validation. It failed when `experiments/contest_auth_eval.py` invoked upstream
`evaluate.py --device cuda`; upstream `frame_utils.py` imported
`nvidia.dali.fn`, but the exact-eval Python environment did not have
`nvidia.dali` installed:

```text
ModuleNotFoundError: No module named 'nvidia.dali'
```

Because upstream evaluation exited with return code `1`, no
`contest_auth_eval.json`, copied result JSON, report, or adjudication
provenance was produced. The archive is byte-custodied only; it has no exact
CUDA score evidence from r3.

## Commands Run

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py refresh-status \
  --job-name owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r3 \
  --teamspace comma-lab \
  --user adpena

ssh -i ~/.ssh/lightning_rsa \
  s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai \
  "find <remote_output_dir> ...; find <sdk_artifact_path> ..."

ssh -i ~/.ssh/lightning_rsa \
  s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai \
  "tail -n 120 <sdk_artifact_path>/pact/experiments/results/lightning_batch/.../auth_eval.log"

scp -i ~/.ssh/lightning_rsa <selected diagnostic artifacts> \
  experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r3/

shasum -a 256 experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r3/archive.zip
wc -c < experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r3/archive.zip

.venv/bin/python scripts/launch_lightning_batch_job.py validate-artifacts \
  --artifact-dir experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r3 \
  --expected-archive-sha256 e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec \
  --expected-archive-size-bytes 686557 \
  --require-adjudication
```

## Robustness Patch

Changed files:

- `src/tac/deploy/lightning/batch_jobs.py`
- `src/tac/tests/test_lightning_batch_jobs.py`
- `.omx/state/lightning_batch_jobs.json` (SDK status refresh)
- `.omx/state/lightning_supply_chain_scan_20260430_codex_r3_harvest.json`
  (local strict scan)
- `experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex_lightning_t4_g4dn2x_r3/` (diagnostic mirror)
- `.omx/research/lightning_r3_exact_eval_harvest_20260430_worker.md`

Patch summary:

- Added `nvidia.dali.fn` import to the Lightning exact-eval runner preflight.
  Future exact-eval jobs fail before archive copy, inflate, and CUDA eval spend
  if the upstream evaluator dependency is missing.
- Recorded the DALI preflight module name in `lightning_runner_preflight.json`.
- Added focused test coverage that exact-eval commands include the DALI
  fail-closed preflight check.

## Verification

Passed:

```bash
.venv/bin/python -m py_compile src/tac/deploy/lightning/batch_jobs.py scripts/launch_lightning_batch_job.py
.venv/bin/python -m pytest src/tac/tests/test_lightning_batch_jobs.py -q
.venv/bin/python scripts/scan_lightning_supply_chain.py --strict --quiet \
  --json-out .omx/state/lightning_supply_chain_scan_20260430_codex_r3_harvest.json
```

Pytest result: `20 passed in 0.18s`.
Local strict supply-chain scan: `status=OK`, `violation_count=0`.

`validate-artifacts --require-adjudication` failed closed, as expected, because
the failed r3 job did not produce `contest_auth_eval.json` or adjudication
artifacts.
