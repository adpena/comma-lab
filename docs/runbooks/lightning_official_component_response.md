# Lightning Official Component-Response Runbook

Updated: 2026-05-01

## Purpose

Run official component-response curves on Lightning after the perturbation
archive producer has emitted an `official_component_response_plan_v1` plan and
all referenced archives exist. This path does not build perturbation archives.
It only stages declared inputs, queues a CUDA Batch Job, harvests compact JSON
evidence, and validates the artifacts.

Score and component-response authority remains:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

The Lightning Batch job runs
`experiments/profile_component_sensitivity_official.py --device cuda`, which
calls `experiments/contest_auth_eval.py` for every response point.

Do not install or invoke the PyPI `lightning` package or `lightning` CLI.
Use `lightning-sdk` through `scripts/launch_lightning_batch_job.py` and SSH
wrappers only.

## SSH Credential Preflight

Keep concrete Studio users, SSH keys, tokens, and account identifiers out of
source. Configure a local SSH alias and export only the alias name:

```sshconfig
Host scratch-studio-devbox
  HostName ssh.lightning.ai
  User <studio-ssh-user>
  IdentityFile ~/.ssh/<private-key-file>
  IdentitiesOnly yes
  BatchMode yes
  StrictHostKeyChecking accept-new
```

Do not use `StrictHostKeyChecking no` for reproducible custody. Avoid duplicate
`Host` blocks for the same alias; OpenSSH merges them and can silently offer
more identities than intended.

Before staging, run the repo preflight. It resolves `ssh -G`, records sanitized
identity/public-key metadata, then performs a `BatchMode` auth probe. It does
not submit a job.

```bash
.venv/bin/python scripts/lightning_repro_workspace.py \
  --remote "$LIGHTNING_SSH_TARGET" \
  --ssh-check-only \
  --ssh-diagnostics-out ".omx/state/${RUN_ID}_ssh_preflight.json"
```

If this fails with `Permission denied (publickey)`, add the resolved `*.pub`
key named in the diagnostic to Lightning Studio/account SSH keys, or correct
the alias `User`/`IdentityFile`. Use `ssh-keygen -lf <key>.pub` to compare the
fingerprint without exposing private key material.

## Required Inputs

The following files must already exist before any non-dry-run dispatch:

- Baseline archive ZIP.
- Optional baseline `contest_auth_eval.json`, if the zero point should reuse
  existing exact CUDA custody.
- Official perturbation plan JSON.
- Every archive referenced by each nonzero plan point.
- Any per-point `contest_auth_eval_json` referenced by the plan.

For reproducible staging, keep plan paths repo-relative or inside the repo.
For remote Batch Jobs, point archives and per-point eval JSON inside the plan
must be relative to the plan file. Host-local absolute paths
(e.g. `<operator-home>/...` or `<wsl-home>/...`) are non-portable and are blocked at submit. If a legacy plan has
a stale top-level `baseline_contest_auth_eval_json`, pass
`--baseline-contest-auth-eval-json "$LIGHTNING_REMOTE_PACT/$BASELINE_JSON"`;
the explicit CLI path is the authority.
The non-dry-run `component-response` command requires both
`--source-manifest` and `--local-perturbation-plan`; it blocks submit when the
staged manifest does not include the plan-listed archives.

## Dry Run

Set runtime-only values:

```bash
export RUN_ID=official_component_response_$(date -u +%Y%m%dT%H%M%SZ)
export BASELINE_ARCHIVE=<repo-relative-baseline-archive.zip>
export BASELINE_JSON=<repo-relative-baseline-contest_auth_eval.json>
export RESPONSE_PLAN=<repo-relative-official-component-response-plan.json>
export LIGHTNING_SSH_TARGET=<ssh-config-alias>
export LIGHTNING_REMOTE_PACT=/teamspace/studios/this_studio/pact
export LIGHTNING_UPSTREAM_DIR=/teamspace/studios/this_studio/upstream
export LIGHTNING_TEAMSPACE=comma-lab
export LIGHTNING_STUDIO=lossy-compression-challenge
export LIGHTNING_SDK_USER=<lightning-account-user>
export LIGHTNING_MACHINE=T4
```

Run local supply-chain scan first:

```bash
.venv/bin/python scripts/scan_lightning_supply_chain.py \
  --strict \
  --quiet \
  --json-out ".omx/state/${RUN_ID}_local_lightning_supply_chain_scan.json"
```

Queue only a local dry-run record:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py component-response \
  --state-path .omx/state/lightning_batch_jobs.json \
  --job-name "$RUN_ID" \
  --baseline-archive "$LIGHTNING_REMOTE_PACT/$BASELINE_ARCHIVE" \
  --baseline-contest-auth-eval-json "$LIGHTNING_REMOTE_PACT/$BASELINE_JSON" \
  --perturbation-plan "$LIGHTNING_REMOTE_PACT/$RESPONSE_PLAN" \
  --repo-dir "$LIGHTNING_REMOTE_PACT" \
  --upstream-dir "$LIGHTNING_UPSTREAM_DIR" \
  --machine "$LIGHTNING_MACHINE" \
  --studio "$LIGHTNING_STUDIO" \
  --teamspace "$LIGHTNING_TEAMSPACE" \
  --user "$LIGHTNING_SDK_USER" \
  --python-bin .venv/bin/python \
  --local-baseline-archive "$BASELINE_ARCHIVE" \
  --infer-expected-baseline-archive \
  --local-artifact-dir "experiments/results/lightning_batch/$RUN_ID" \
  --queue-metadata lane=official_component_response \
  --queue-metadata response_plan="$RESPONSE_PLAN" \
  --dry-run
```

Inspect the printed command. It must include the strict supply-chain scan,
hash-pinned DALI bootstrap, CUDA runner preflight, official profiler with
`--device cuda`, input preflight, heavy eval cleanup, and
`validate-component-response-artifacts`.

## Stage And Submit

Stage the source tree and all response artifacts. Add one `--artifact` flag for
every perturbation archive and per-point JSON named by `$RESPONSE_PLAN`.

```bash
.venv/bin/python scripts/lightning_repro_workspace.py \
  --remote "$LIGHTNING_SSH_TARGET" \
  --remote-pact "$LIGHTNING_REMOTE_PACT" \
  --run-id "$RUN_ID" \
  --manifest-out ".omx/state/${RUN_ID}_manifest.json" \
  --requirements-mode uv-sync \
  --ssh-diagnostics-out ".omx/state/${RUN_ID}_ssh_preflight.json" \
  --artifact "$BASELINE_ARCHIVE" \
  --artifact "$BASELINE_JSON" \
  --artifact "$RESPONSE_PLAN" \
  --artifact <each-plan-point-archive-or-json>
```

Run the composable pre-submit doctor before SDK submission:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py doctor \
  --ssh-target "$LIGHTNING_SSH_TARGET" \
  --teamspace "$LIGHTNING_TEAMSPACE" \
  --user "$LIGHTNING_SDK_USER" \
  --machine "$LIGHTNING_MACHINE" \
  --require-ssh \
  --require-remote-supply-chain \
  --require-machine-inventory \
  --json-out ".omx/state/${RUN_ID}_lightning_doctor.json"
```

Then submit the non-interruptible Batch Job:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py component-response \
  --state-path .omx/state/lightning_batch_jobs.json \
  --job-name "$RUN_ID" \
  --baseline-archive "$LIGHTNING_REMOTE_PACT/$BASELINE_ARCHIVE" \
  --baseline-contest-auth-eval-json "$LIGHTNING_REMOTE_PACT/$BASELINE_JSON" \
  --perturbation-plan "$LIGHTNING_REMOTE_PACT/$RESPONSE_PLAN" \
  --repo-dir "$LIGHTNING_REMOTE_PACT" \
  --upstream-dir "$LIGHTNING_UPSTREAM_DIR" \
  --machine "$LIGHTNING_MACHINE" \
  --studio "$LIGHTNING_STUDIO" \
  --teamspace "$LIGHTNING_TEAMSPACE" \
  --user "$LIGHTNING_SDK_USER" \
  --python-bin .venv/bin/python \
  --remote-preflight-ssh-target "$LIGHTNING_SSH_TARGET" \
  --local-baseline-archive "$BASELINE_ARCHIVE" \
  --infer-expected-baseline-archive \
  --source-manifest ".omx/state/${RUN_ID}_manifest.json" \
  --local-perturbation-plan "$RESPONSE_PLAN" \
  --local-artifact-dir "experiments/results/lightning_batch/$RUN_ID" \
  --queue-metadata lane=official_component_response \
  --queue-metadata response_plan="$RESPONSE_PLAN" \
  --require-passed
```

No `--dry-run` means real dispatch. The command fails before SDK submission if
the source manifest does not include the baseline archive, response plan, and
all plan-listed archives, or if the remote Studio tree fails the strict
Lightning supply-chain scan.

## Monitor, Harvest, Validate

Refresh job state through the SDK wrapper:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py refresh-status \
  --state-path .omx/state/lightning_batch_jobs.json \
  --job-name "$RUN_ID"
```

After the job completes, harvest compact response evidence through SSH. This
copies root JSON artifacts plus per-point `contest_auth_eval.json`,
`provenance.json`, `report.txt`, and profiler stdout/stderr logs. It does not
copy inflated raw frame directories.

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py harvest-component-response-ssh \
  --state-path .omx/state/lightning_batch_jobs.json \
  --job-name "$RUN_ID" \
  --ssh-target "$LIGHTNING_SSH_TARGET" \
  --expected-baseline-archive-sha256 <baseline-archive-sha256> \
  --expected-baseline-archive-size-bytes <baseline-archive-bytes> \
  --require-passed
```

The harvest command derives the persisted SDK artifact mirror from the recorded
`remote_output_dir` and `sdk_artifact_path`; do not hand-compose
`/teamspace/jobs/...` or Studio output paths for promotion-grade harvests.

If the artifacts are already local, validate them directly:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py validate-component-response-artifacts \
  --artifact-dir "experiments/results/lightning_batch/$RUN_ID" \
  --expected-baseline-archive-sha256 <baseline-archive-sha256> \
  --expected-baseline-archive-size-bytes <baseline-archive-bytes> \
  --require-passed
```

The validated packet must include `lightning_queue_metadata.json`,
`official_component_response_inputs.json`,
`official_component_response_summary.json`, the three component curve JSONs,
`lightning_supply_chain_scan_pre.json`, `lightning_supply_chain_scan.json`,
`lightning_dali_bootstrap.json`, `lightning_dali_requirements.txt`, and
`lightning_runner_preflight.json`.
