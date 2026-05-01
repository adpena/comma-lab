# Lightning Worker A Exact-Eval Harvest Status - 2026-04-30

Worker: Worker A / Codex
Scope:

- `pfp16_paired_calibration_20260430_codex_lightning_t4_r2`
- `owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4`

No score claim is made in this note. No artifacts were harvested or validated
because both jobs remained `Running` at the explicit SDK refreshes below.

## Preflight

- Live MCP helpers were found locally and killed by exact command pattern.
- `check_no_live_mcp_processes(strict=True)` then passed.
- Local strict Lightning supply-chain scan passed:
  `.omx/state/lightning_supply_chain_scan_20260430_worker_a_harvest.json`
  records `status=OK`, `violation_count=0`, `lightning=null`, and
  `lightning-sdk=2026.4.10`.

## SDK Status Refresh

PFP16 paired calibration:

- SDK job:
  `pfp16-paired-calibration-20260430-codex-lightning-t4-r2`
- Local queue job:
  `pfp16_paired_calibration_20260430_codex_lightning_t4_r2`
- Final explicit refresh time: `2026-04-30T22:38:18Z`
- Status: `Running`
- `completed_at`: `null`
- `failed_at`: `null`
- `failure_reason`: `null`
- Expected archive SHA-256:
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`
- Expected archive bytes: `686635`
- Local artifact dir target:
  `experiments/results/lightning_batch/pfp16_paired_calibration_20260430_codex_lightning_t4_r2`
- Required adjudication: yes, `required_device=cuda`, `required_samples=600`,
  result copy `contest_auth_eval.adjudicated.json`.

OWV3 R5 rank-1 exact CUDA:

- SDK job:
  `owv3-r5-rank1-exact-cuda-20260430-codex-lightning-t4`
- Local queue job:
  `owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4`
- Final explicit refresh time: `2026-04-30T22:38:26Z`
- Status: `Running`
- `completed_at`: `null`
- `failed_at`: `null`
- `failure_reason`: `null`
- Expected archive SHA-256:
  `16ab95220c8add11b0bc40fb632bc8421f8bb8ad1cfba145f0b6058075237518`
- Expected archive bytes: `686468`
- Local artifact dir target:
  `experiments/results/lightning_batch/owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4`
- Required adjudication: yes, `required_device=cuda`, `required_samples=600`,
  result copy `contest_auth_eval.adjudicated.json`.
- Queue metadata still requires paired readjudication against
  `pfp16_paired_calibration_20260430_codex_lightning_t4_r2`.

## Harvest Decision

No `harvest-local` or `validate-artifacts --require-adjudication` command was
run for these two jobs in this turn because neither job reached a terminal
completed status. There are no local harvested artifact directories for these
two job names yet.

## Commands Run

```bash
ps -axo pid=,command= | rg -i "mcp|model context protocol|chrome_devtools|roblox_studio_mcp" || true
kill <exact MCP helper PIDs>
kill -9 <remaining exact MCP helper PIDs>
.venv/bin/python - <<'PY'
from tac.preflight import check_no_live_mcp_processes
check_no_live_mcp_processes(strict=True, verbose=True)
print('check_no_live_mcp_processes(strict=True): PASS')
PY

.venv/bin/python scripts/scan_lightning_supply_chain.py \
  --json-out .omx/state/lightning_supply_chain_scan_20260430_worker_a_harvest.json \
  --strict \
  --quiet

.venv/bin/python scripts/launch_lightning_batch_job.py refresh-status \
  --job-name pfp16_paired_calibration_20260430_codex_lightning_t4_r2 \
  --teamspace comma-lab \
  --user adpena

.venv/bin/python scripts/launch_lightning_batch_job.py refresh-status \
  --job-name owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4 \
  --teamspace comma-lab \
  --user adpena

# Final explicit refresh pass used for the status above:
.venv/bin/python scripts/launch_lightning_batch_job.py refresh-status \
  --job-name pfp16_paired_calibration_20260430_codex_lightning_t4_r2 \
  --teamspace comma-lab \
  --user adpena

.venv/bin/python scripts/launch_lightning_batch_job.py refresh-status \
  --job-name owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4 \
  --teamspace comma-lab \
  --user adpena

find experiments/results/lightning_batch -maxdepth 1 -type d \
  \( -name 'pfp16_paired_calibration_20260430_codex_lightning_t4_r2' \
     -o -name 'owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4' \) \
  -print -exec find {} -maxdepth 1 -type f -print \;
```

State touched by the refresh/preflight:

- `.omx/state/lightning_batch_jobs.json`
- `.omx/state/lightning_supply_chain_scan_20260430_worker_a_harvest.json`
- `.omx/research/lightning_worker_a_exact_eval_harvest_status_20260430.md`

## R6 Rank-1 Monitor Addendum

Scope:

- `owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1`
- SDK job:
  `owv3-r6-rank1-exact-cuda-20260430-codex-lightning-t4-r1`

No score claim is made in this addendum. The job remained non-terminal at the
read-only SDK poll below, so no harvest or artifact validation was run.

Read-only SDK attribute poll:

- Poll time: `2026-04-30T23:44:09Z`
- Status: `Running`
- `completed_at`: `null`
- `failed_at`: `null`
- `failure_reason`: `null`
- SDK artifact path:
  `/teamspace/jobs/owv3-r6-rank1-exact-cuda-20260430-codex-lightning-t4-r1/artifacts`
- Local state's last recorded refresh before the read-only poll:
  `2026-04-30T23:42:42Z`, status `Running`
- Local artifact directory:
  `experiments/results/lightning_batch/owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1`
- Local artifact directory status: absent at monitor time.
- Expected archive SHA-256:
  `9f7528bade11bf9cdf3df68f8073d11f196a6d5f48475a8680c21fb58c878c91`
- Expected archive bytes: `686531`

Queued adjudication is paired-run forensic tolerant:

- Required device/samples: `cuda`, `600`
- Paired baseline score: `1.037045485927815`
- Paired baseline PoseNet: `0.00316404`
- Paired baseline SegNet: `0.00401966`
- Max PoseNet relative: `1.002`
- Max SegNet relative: `1.002`
- `allow_component_gate_forensic_success=true`

If this job completes, harvest through the state-aware SSH wrapper rather than
hand-copying the SDK artifact tree:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py harvest-ssh \
  --state-path .omx/state/lightning_batch_jobs.json \
  --job-name owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1 \
  --ssh-target "$LIGHTNING_SSH_TARGET" \
  --expected-archive-sha256 9f7528bade11bf9cdf3df68f8073d11f196a6d5f48475a8680c21fb58c878c91 \
  --expected-archive-size-bytes 686531 \
  --require-adjudication
```

Then run the strict final-deploy component gate separately before any promotion
discussion:

```bash
.venv/bin/python scripts/adjudicate_contest_auth_eval.py \
  --contest-json experiments/results/lightning_batch/owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1/contest_auth_eval.json \
  --provenance experiments/results/lightning_batch/owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1/adjudication_final_deploy_provenance.json \
  --archive experiments/results/lightning_batch/owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1/archive.zip \
  --result-copy experiments/results/lightning_batch/owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1/contest_auth_eval.final_deploy_gated.json \
  --baseline-score 1.043987524793892 \
  --predicted-band 0.0 1.043987524793892 \
  --regression-threshold 1.043987524793892 \
  --baseline-archive-bytes 686635 \
  --baseline-posenet-dist 0.00346442 \
  --baseline-segnet-dist 0.00400656 \
  --max-posenet-relative 1.002 \
  --max-segnet-relative 1.002 \
  --component-reference-label pfp16_final_deploy_bundle_20260430 \
  --required-device cuda \
  --required-samples 600
```

Commands run for this addendum:

```bash
jq '.[] | select((.spec.name? == "owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1") or (.queue.job_name? == "owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1")) | {status, job_status: .job.status, job_refreshed_at: .job.refreshed_at_utc, sdk_job_name: .job.name, sdk_artifact_path: .queue.sdk_artifact_path, remote_output_dir: .queue.remote_output_dir, local_artifact_dir: .queue.local_artifact_dir, expected_archive_sha256: .queue.expected_archive_sha256, expected_archive_size_bytes: .queue.expected_archive_size_bytes}' .omx/state/lightning_batch_jobs.json

.venv/bin/python - <<'PY'
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('LIGHTNING_DISABLE_VERSION_CHECK', '1')
sys.path.insert(0, str(Path('src').resolve()))
from tac.deploy.lightning.batch_jobs import job_status_snapshot
from lightning_sdk import Job

job = Job(
    name='owv3-r6-rank1-exact-cuda-20260430-codex-lightning-t4-r1',
    teamspace='comma-lab',
    user='adpena',
)
print(json.dumps(job_status_snapshot(job), indent=2, sort_keys=True))
PY

find experiments/results/lightning_batch/owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1 -maxdepth 2 -type f -print
```

## R6 Harvest Addendum

R6 completed after the monitor pass and was harvested locally.

Status:

- Completed refresh: `2026-04-30T23:47:45Z`
- Local validation: `2026-04-30T23:48:37Z`
- SDK artifact root:
  `/teamspace/jobs/owv3-r6-rank1-exact-cuda-20260430-codex-lightning-t4-r1/artifacts`
- Local artifact dir:
  `experiments/results/lightning_batch/owv3_r6_rank1_exact_cuda_20260430_codex_lightning_t4_r1`

Exact result:

- `score_recomputed_from_components=1.0393166493980681`
- `archive_size_bytes=686531`
- `archive_sha256=9f7528bade11bf9cdf3df68f8073d11f196a6d5f48475a8680c21fb58c878c91`
- `avg_posenet_dist=0.00323147`
- `avg_segnet_dist=0.00402421`
- `device=cuda`
- `gpu_model=Tesla T4`
- `n_samples=600`

Strict final-deploy gate:

- Exit code: `2`
- `LANE_STATUS=REGRESSION_AND_COMPONENT_GATE_REVIEW_REQUIRED`
- `REGRESSION_TRIGGERED=1`
- `COMPONENT_GATE_TRIGGERED=1`
- PoseNet violation: observed `0.00323147`, reference `0.00316404`,
  relative `1.0213113614240024`, max `1.002`.
- SegNet passed: observed `0.00402421`, reference `0.00401966`,
  relative `1.0011319365319455`, max `1.002`.

Classification:

R6 is valid A++ exact CUDA/T4 forensic evidence and a scoped negative for this
candidate/config. It is not promotable and not an OWV3 family KILL.
