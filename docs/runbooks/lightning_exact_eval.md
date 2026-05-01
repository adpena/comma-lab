# Lightning Exact Eval Runbook

Updated: 2026-04-30

## Purpose

Use Lightning only as an auditable CUDA exact-eval runner. Score authority
remains `archive.zip -> inflate.sh -> upstream/evaluate.py` through
`experiments/contest_auth_eval.py --device cuda`; human logs are not score
evidence.

Do not install the PyPI package named `lightning` and do not invoke the
`lightning` CLI for discovery. Exact-eval queue tooling uses `lightning-sdk`
through repo wrappers with `LIGHTNING_DISABLE_VERSION_CHECK=1`.

## Credential Handling

Keep SSH targets, key paths, and Lightning account details out of source.
Preferred setup is an SSH config alias:

```sshconfig
Host scratch-studio-devbox
  HostName ssh.lightning.ai
  User <studio-ssh-user>
  IdentityFile ~/.ssh/<private-key-file>
  IdentitiesOnly yes
  BatchMode yes
```

Then export runtime-only values:

```bash
export LIGHTNING_SSH_TARGET=scratch-studio-devbox
export LIGHTNING_REMOTE_PACT=/teamspace/studios/this_studio/pact
export LIGHTNING_UPSTREAM_DIR=/teamspace/studios/this_studio/upstream
export LIGHTNING_TEAMSPACE=comma-lab
export LIGHTNING_STUDIO=lossy-compression-challenge
export LIGHTNING_SDK_USER=<lightning-account-user>
export LIGHTNING_MACHINE=T4
```

Do not commit shell history, `.env` files, SSH keys, API tokens, or concrete
personal SSH users.

## Dry-Run Queue

The safe default is a local supply-chain scan plus a Lightning Batch dry-run.
No SSH staging or spendful job submission happens unless explicitly requested.

```bash
.venv/bin/python scripts/lightning_exact_eval_repro.py \
  --job-name owv3_exact_eval_$(date -u +%Y%m%dT%H%M%SZ) \
  --archive experiments/results/lane_g_v3_owv3_byte_plan_sweep_worker_b/best_byte_feasible/archive_lane_g_v3_owv3.zip \
  --baseline-json experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/eval/contest_auth_eval.json \
  --predicted-band 1.00 1.10 \
  --regression-threshold 1.20 \
  --max-posenet-relative 1.05 \
  --max-segnet-relative 1.002 \
  --queue-metadata lane=owv3_byte_feasible \
  --queue-metadata evidence=exact_cuda_candidate
```

The wrapper writes:

- `.omx/state/<job>_lightning_exact_eval_repro_plan.json`
- `.omx/state/<job>_local_lightning_supply_chain_scan.json`
- `.omx/state/<job>_lightning_batch_record.json`

The generated Batch command copies the exact archive bytes into the job output
directory, verifies SHA-256 and byte count, runs CUDA auth eval, copies
`contest_auth_eval.json`, and runs JSON-based adjudication.

## Stage And Submit

Stage source plus explicit artifacts with a deterministic manifest, then submit
the non-interruptible T4 Batch Job:

```bash
.venv/bin/python scripts/lightning_exact_eval_repro.py \
  --job-name owv3_exact_eval_$(date -u +%Y%m%dT%H%M%SZ) \
  --stage-workspace \
  --submit \
  --archive experiments/results/lane_g_v3_owv3_byte_plan_sweep_worker_b/best_byte_feasible/archive_lane_g_v3_owv3.zip \
  --baseline-json experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/eval/contest_auth_eval.json \
  --predicted-band 1.00 1.10 \
  --regression-threshold 1.20 \
  --max-posenet-relative 1.05 \
  --max-segnet-relative 1.002 \
  --queue-metadata lane=owv3_byte_feasible \
  --queue-metadata evidence=exact_cuda_candidate
```

Use `--requirements-mode uv-sync` by default. Use
`--requirements-mode verify-only --python-bin <remote-python>` only when a
prebuilt remote environment has already been audited and should not be changed.

If remote source/artifact custody was verified separately, `--submit` may be
combined with `--allow-unstaged-submit`, but this should cite the custody
manifest in `--queue-metadata`.

Do not set `--output-dir` to `/teamspace/jobs/...`; that path is the SDK
artifact view and can be read-only inside a running Studio job. The default
output directory is:

```text
<LIGHTNING_REMOTE_PACT>/experiments/results/lightning_batch/<job-name>
```

## Harvest And Validate

After a Studio-backed job reaches a terminal completed state, harvest through
the state-aware SSH wrapper. It derives the persisted SDK artifact mirror from
the recorded `sdk_artifact_path` and `remote_output_dir`, copies only canonical
evidence files, and validates archive identity plus adjudication:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py harvest-ssh \
  --state-path .omx/state/lightning_batch_jobs.json \
  --job-name <job-name> \
  --ssh-target "$LIGHTNING_SSH_TARGET" \
  --expected-archive-sha256 <archive-sha256> \
  --expected-archive-size-bytes <archive-bytes> \
  --require-adjudication
```

If the artifacts are already available locally, validate the harvest using
expected archive identity and adjudication:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py validate-artifacts \
  --artifact-dir experiments/results/lightning_batch/<job-name> \
  --expected-archive-sha256 <archive-sha256> \
  --expected-archive-size-bytes <archive-bytes> \
  --require-adjudication
```

For promotion-grade claims, the harvested packet must include
`lightning_queue_metadata.json`, `archive.zip`, `contest_auth_eval.json`,
`contest_auth_eval.adjudicated.json`, `adjudication_provenance.json`,
`auth_eval.log`, `adjudication.log`, `eval_provenance.json`, `report.txt`,
`lightning_supply_chain_scan_pre.json`, `lightning_supply_chain_scan.json`,
`lightning_dali_bootstrap.json`, `lightning_dali_requirements.txt`, and
`lightning_runner_preflight.json`.

## Manual Building Blocks

When debugging the wrapper, the two underlying commands are:

```bash
.venv/bin/python scripts/lightning_repro_workspace.py \
  --remote "$LIGHTNING_SSH_TARGET" \
  --remote-pact "$LIGHTNING_REMOTE_PACT" \
  --run-id <run-id> \
  --requirements-mode uv-sync \
  --artifact <repo-relative-archive.zip>
```

Before non-dry-run SDK submission, run:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py doctor \
  --ssh-target "$LIGHTNING_SSH_TARGET" \
  --teamspace "$LIGHTNING_TEAMSPACE" \
  --user "$LIGHTNING_SDK_USER" \
  --machine "$LIGHTNING_MACHINE" \
  --require-ssh \
  --require-remote-supply-chain \
  --require-machine-inventory \
  --json-out ".omx/state/<run-id>_lightning_doctor.json"
```

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py exact-eval \
  --job-name <job-name> \
  --archive "$LIGHTNING_REMOTE_PACT/<repo-relative-archive.zip>" \
  --repo-dir "$LIGHTNING_REMOTE_PACT" \
  --upstream-dir "$LIGHTNING_UPSTREAM_DIR" \
  --machine "${LIGHTNING_MACHINE:-T4}" \
  --studio "$LIGHTNING_STUDIO" \
  --teamspace "$LIGHTNING_TEAMSPACE" \
  --user "$LIGHTNING_SDK_USER" \
  --python-bin .venv/bin/python \
  --remote-preflight-ssh-target "$LIGHTNING_SSH_TARGET" \
  --expected-archive-sha256 <archive-sha256> \
  --expected-archive-size-bytes <archive-bytes> \
  --adjudicate \
  --baseline-score <baseline-recomputed-score> \
  --predicted-band <low> <high> \
  --regression-threshold <threshold> \
  --dry-run
```
