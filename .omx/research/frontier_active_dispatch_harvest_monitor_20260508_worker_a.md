# Frontier Active Dispatch Harvest Monitor - Worker A - 2026-05-08

generated_at_utc: 2026-05-08T01:43:15Z
scope: active Lightning/job harvest and no-signal-loss monitoring only
remote_actions_taken: none
local_mutations_taken: ledger only

## Repo / State Baseline

- Branch: `main`
- HEAD: `c3cda1530cd3235ccbd5ebb359a9c65bd7cb5f90`
- Required context commits verified as ancestors of HEAD:
  - `c3cda153` - Test lossy coarsening dispatch guards
  - `4d1061b4` - add lossy_coarsening_analytical Lightning T4 CUDA dispatcher + harvester
  - `dadb33b9` - Harden monolithic frontier packet gates
- Worktree was already dirty before this audit. Existing modified harvest files observed:
  - `experiments/arch_shrink_x0.4_lightning_harvest.py`
  - `experiments/lossy_coarsening_lightning_harvest.py`
- This audit did not submit, stop, cancel, force-harvest, or remotely query any Lightning job.

## Active Claims

Source: `.omx/state/active_lane_dispatch_claims.md` lines 21-24 as of this audit.

| lane_id | agent | job_id | status | predicted_eta_utc | note |
|---|---|---|---|---|---|
| `lossy_coarsening_analytical_cuda` | `claude_lab` | `lossy-coarsening-cuda-20260508T013829Z` | `active_dispatching` | `2026-05-08T03:08:34Z` | FULL Lightning T4 contest-CUDA auth eval via `experiments/lossy_coarsening_lightning_cuda_test.py` |
| `arch_shrink_x0.4_lightning` | `claude_lab` | `arch-shrink-x0-4-lightning-20260508T010514Z` | `active_dispatching` | `2026-05-08T19:05:19Z` | FULL Lightning T4 train + archive + auth eval via `experiments/arch_shrink_x0.4_lightning_full.py` |
| `arch_shrink_x0.4_lightning` | `claude_lab` | `arch-shrink-x0-4-lightning-20260508T010445Z` | `stale_superseded_dry_run_replaced_by_real` | empty | Closed dry-run row; do not harvest as live work. |

## Lightning State Files

- `.omx/state/lightning_active_jobs.json`
  - JSON shape: array.
  - Rows: 3.
  - `arch-shrink-x0-4-lightning-20260508T010445Z`: `terminal_status=dry_run_completed`; `terminated_at_utc=2026-05-08T01:05:05Z`.
  - `arch-shrink-x0-4-lightning-20260508T010514Z`: nonterminal row; `submit_result.status_at_submit=Pending`; expected local auth eval path `experiments/results/lightning_batch/arch-shrink-x0-4-lightning-20260508T010514Z/contest_auth_eval.json`.
  - `lossy-coarsening-cuda-20260508T013829Z`: nonterminal row; `submit_result.status_at_submit=Pending`; expected local auth eval path `experiments/results/lightning_batch/lossy-coarsening-cuda-20260508T013829Z/contest_auth_eval.json`.
- `.omx/state/lightning_batch_jobs.json`
  - JSON shape: array.
  - Rows: 55.
  - No rows matched `arch-shrink-x0-4-lightning` or `lossy-coarsening-cuda`; current source of truth for these two jobs is `.omx/state/lightning_active_jobs.json`.

## Local Artifacts Present / Missing

### `arch-shrink-x0-4-lightning-20260508T010445Z`

- Local batch mirror exists: `experiments/results/lightning_batch/arch-shrink-x0-4-lightning-20260508T010445Z`
- Present:
  - `source_manifest.json`
- Missing locally:
  - `archive.zip`
  - `contest_auth_eval.json`
  - `auth_eval.log`
  - `run.log`
  - `heartbeat.log`
- Disposition: dry-run row is terminal/superseded; do not harvest.
- Source manifest:
  - `generated_at_utc=2026-05-08T01:04:45Z`
  - `git_head=ee2919fd5c514aef29ffc6be73337e1140382bb6`
  - `file_count=2326`
  - `total_bytes=223568444`
  - `artifact_paths=[]`

### `arch-shrink-x0-4-lightning-20260508T010514Z`

- Local batch mirror exists: `experiments/results/lightning_batch/arch-shrink-x0-4-lightning-20260508T010514Z`
- Present:
  - `source_manifest.json`
- Missing locally:
  - `archive.zip`
  - `contest_auth_eval.json`
  - `auth_eval.log`
  - `run.log`
  - `heartbeat.log`
- Source manifest:
  - `generated_at_utc=2026-05-08T01:05:14Z`
  - `git_head=002fc7d23235cb49836552a9a52844571b4c8474`
  - `file_count=2327`
  - `total_bytes=223589472`
  - `artifact_paths=[]`
- Interpretation: mirror is staging-only so far. Absence of eval artifacts locally is not a failure while the job is nonterminal.

### `lossy-coarsening-cuda-20260508T013829Z`

- Local batch mirror exists: `experiments/results/lightning_batch/lossy-coarsening-cuda-20260508T013829Z`
- Present in batch mirror:
  - `source_manifest.json`
- Missing locally from batch mirror:
  - `contest_auth_eval.json`
  - `auth_eval.log`
  - `run.log`
  - `heartbeat.log`
- Local prebuilt archive artifacts exist outside the batch mirror:
  - `experiments/results/lossy_coarsening_20260508T013829Z/build_manifest.json`
  - `experiments/results/lossy_coarsening_20260508T013829Z/archive.zip`
  - `experiments/results/lossy_coarsening_20260508T013829Z/submission_dir/`
- Local archive custody:
  - bytes: `156404`
  - sha256: `ab8a8a13c70b3d3bbf2ce3d8a81a77691b776a6e0fb1cbe9ce504dc3f59c1b28`
  - ZIP members: single stored member `x`, 156304 bytes uncompressed/compressed.
- Build manifest:
  - `rel_err_budget=0.05`
  - `rel_err_actual_int8=0.03855950900557584`
  - `rel_err_actual_fp32_smoke=0.03481125033136704`
  - `evidence_grade=[CPU-build]`
  - `score_claim=false`
  - `promotion_eligible=false`
- Source manifest:
  - `generated_at_utc=2026-05-08T01:38:30Z`
  - `git_head=4d1061b4d55c8f5259fbfc0e50c0d545bfb7a8d1`
  - `file_count=2348`
  - `total_bytes=224028659`
  - `artifact_paths` includes the local archive and submission directory.
- Interpretation: local build custody is present; contest-CUDA score artifact is still missing locally.

## Shell Harvest Readiness

- No `LIGHTNING_*` variables were present.
- No `REMOTE` variable was present.
- Imported defaults resolve to:
  - `default_ssh_target=''`
  - `default_teamspace=''`
  - `default_user=''`
  - `default_remote_pact='/teamspace/studios/this_studio/pact'`
- Local harvest prerequisites present:
  - `rsync`: `/usr/bin/rsync`
  - `ssh`: `/usr/bin/ssh`
  - `jq`: `/usr/bin/jq`
  - `zipinfo`: `/usr/bin/zipinfo`
  - `lightning_sdk`: import OK, version `2026.05.06post2`
- Conclusion:
  - Bare harvester invocations are not ready in this shell because the default SSH target, teamspace, and user are empty.
  - State-derived explicit invocations are possible because `submit_result.user`, `submit_result.teamspace`, `source_manifest.remote`, and `source_manifest.remote_pact` are present in local state/manifests.
  - SSH authentication was not probed in this audit. Run the read-only SSH probe below before relying on rsync harvest.

## Tooling Notes

- `experiments/lossy_coarsening_lightning_harvest.py`
  - Reads `.omx/state/lightning_active_jobs.json`.
  - Selects the most recent nonterminal `lossy_coarsening_analytical_cuda` row by default, or exact `--job-name`.
  - `--once` performs one SDK status check; if the job is terminal, it rsyncs artifacts, parses `contest_auth_eval.json`, appends evidence, appends a terminal claim, and marks the active-jobs row terminal.
  - `--force-harvest` skips SDK status. Do not use it while the job may still be running.
- `experiments/arch_shrink_x0.4_lightning_harvest.py`
  - Same control flow for `arch_shrink_x0.4_lightning`.
  - Default poll interval is 300 seconds.
- `experiments/lossy_coarsening_lightning_cuda_test.py`
  - Dispatch/build script; do not run for monitoring.
- `experiments/arch_shrink_x0.4_lightning_full.py`
  - Full train + archive + auth-eval dispatch script; do not run for monitoring.

## Exact Next Safe Commands

Local state check only:

```bash
jq '[.[] | {lane_id, job_name, submitted_at_utc, terminal_status, machine, status_at_submit: .submit_result.status_at_submit, expected_artifact_dir, expected_auth_eval_json}]' .omx/state/lightning_active_jobs.json
```

Read-only local artifact presence check:

```bash
for d in \
  experiments/results/lightning_batch/arch-shrink-x0-4-lightning-20260508T010514Z \
  experiments/results/lightning_batch/lossy-coarsening-cuda-20260508T013829Z; do
  test -f "$d/contest_auth_eval.json" && echo "present $d/contest_auth_eval.json" || echo "missing $d/contest_auth_eval.json"
done
```

Prepare state-derived environment for lossy coarsening harvest:

```bash
STATE=.omx/state/lightning_active_jobs.json
LOSSY_JOB=lossy-coarsening-cuda-20260508T013829Z
LOSSY_MANIFEST=experiments/results/lightning_batch/${LOSSY_JOB}/source_manifest.json
LOSSY_USER="$(jq -r --arg job "$LOSSY_JOB" '.[] | select(.job_name == $job) | .submit_result.user // empty' "$STATE")"
LOSSY_TEAMSPACE="$(jq -r --arg job "$LOSSY_JOB" '.[] | select(.job_name == $job) | .submit_result.teamspace // empty' "$STATE")"
LOSSY_SSH="$(jq -r '.remote // empty' "$LOSSY_MANIFEST")"
LOSSY_REMOTE_PACT="$(jq -r '.remote_pact // "/teamspace/studios/this_studio/pact"' "$LOSSY_MANIFEST")"
```

Read-only SSH probe for lossy coarsening:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 "$LOSSY_SSH" "test -d '$LOSSY_REMOTE_PACT/experiments/results/lightning_batch/$LOSSY_JOB' && echo remote_artifact_dir_present"
```

Single-shot status/harvest for lossy coarsening. This does not submit or stop remote work; if the job is terminal, it will harvest and update local ledgers:

```bash
.venv/bin/python experiments/lossy_coarsening_lightning_harvest.py \
  --job-name "$LOSSY_JOB" \
  --ssh-target "$LOSSY_SSH" \
  --remote-pact "$LOSSY_REMOTE_PACT" \
  --teamspace "$LOSSY_TEAMSPACE" \
  --user "$LOSSY_USER" \
  --once
```

Prepare state-derived environment for arch shrink harvest:

```bash
STATE=.omx/state/lightning_active_jobs.json
ARCH_JOB=arch-shrink-x0-4-lightning-20260508T010514Z
ARCH_MANIFEST=experiments/results/lightning_batch/${ARCH_JOB}/source_manifest.json
ARCH_USER="$(jq -r --arg job "$ARCH_JOB" '.[] | select(.job_name == $job) | .submit_result.user // empty' "$STATE")"
ARCH_TEAMSPACE="$(jq -r --arg job "$ARCH_JOB" '.[] | select(.job_name == $job) | .submit_result.teamspace // empty' "$STATE")"
ARCH_SSH="$(jq -r '.remote // empty' "$ARCH_MANIFEST")"
ARCH_REMOTE_PACT="$(jq -r '.remote_pact // "/teamspace/studios/this_studio/pact"' "$ARCH_MANIFEST")"
```

Read-only SSH probe for arch shrink:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 "$ARCH_SSH" "test -d '$ARCH_REMOTE_PACT/experiments/results/lightning_batch/$ARCH_JOB' && echo remote_artifact_dir_present"
```

Single-shot status/harvest for arch shrink. This does not submit or stop remote work; if the job is terminal, it will harvest and update local ledgers:

```bash
.venv/bin/python experiments/arch_shrink_x0.4_lightning_harvest.py \
  --job-name "$ARCH_JOB" \
  --ssh-target "$ARCH_SSH" \
  --remote-pact "$ARCH_REMOTE_PACT" \
  --teamspace "$ARCH_TEAMSPACE" \
  --user "$ARCH_USER" \
  --once
```

Continuous polling only after the explicit args above have been prepared and SSH probe succeeds:

```bash
.venv/bin/python experiments/lossy_coarsening_lightning_harvest.py --job-name "$LOSSY_JOB" --ssh-target "$LOSSY_SSH" --remote-pact "$LOSSY_REMOTE_PACT" --teamspace "$LOSSY_TEAMSPACE" --user "$LOSSY_USER" --poll-interval-sec 180
.venv/bin/python experiments/arch_shrink_x0.4_lightning_harvest.py --job-name "$ARCH_JOB" --ssh-target "$ARCH_SSH" --remote-pact "$ARCH_REMOTE_PACT" --teamspace "$ARCH_TEAMSPACE" --user "$ARCH_USER" --poll-interval-sec 300
```

Do not run:

```bash
.venv/bin/python experiments/lossy_coarsening_lightning_cuda_test.py ...
.venv/bin/python experiments/arch_shrink_x0.4_lightning_full.py ...
.venv/bin/python experiments/*_harvest.py --force-harvest
```

unless the operator explicitly authorizes a new dispatch or SDK status is known terminal and force-harvest is the only recovery path.
