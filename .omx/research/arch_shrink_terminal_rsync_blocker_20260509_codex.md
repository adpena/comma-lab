# Arch-Shrink Lightning Terminal Harvest Blocker — 2026-05-09

## Status

`arch_shrink_x0.4_lightning` is no longer an active score-producing job in
this workspace. A single-shot harvest found the Lightning job terminal, but
artifact download failed before any score artifact could be parsed.

- Lane: `arch_shrink_x0.4_lightning`
- Job: `arch-shrink-x0-4-lightning-20260508T024304Z`
- Observed SDK status: `stopped`
- Harvester command:
  `.venv/bin/python experiments/arch_shrink_x0.4_lightning_harvest.py --job-name arch-shrink-x0-4-lightning-20260508T024304Z --teamspace comma-lab --user adpena --ssh-target s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai --once`
- Terminal claim status recorded locally:
  `failed_artifact_rsync_rc_255`
- Failure text: `Permission denied (publickey)`

## Evidence Semantics

No `archive.zip`, `contest_auth_eval.json`, or `auth_eval.log` was harvested.
There is no score claim and no candidate promotion. This is an infrastructure
harvest blocker, not a measured model regression.

The harvester attempted both canonical remote paths:

- `/teamspace/studios/this_studio/pact/experiments/results/lightning_batch/arch-shrink-x0-4-lightning-20260508T024304Z/`
- `/teamspace/jobs/arch-shrink-x0-4-lightning-20260508t024304z/artifacts/pact/experiments/results/lightning_batch/arch-shrink-x0-4-lightning-20260508T024304Z/`

Both failed with rsync rc `255` due SSH public-key denial.

## Classification

- Result class: `infrastructure_harvest_blocker`
- Score claim: `false`
- Promotion eligible: `false`
- Family falsified: `false`
- Measured configuration retired: `false`

## Follow-up

- Recover artifacts via a valid Lightning SSH identity or Lightning SDK
  artifact download path.
- If artifacts are recovered, run the normal result-review packet: archive
  bytes, archive SHA-256, runtime-tree SHA, component recomputation, logs, and
  terminal claim supersession.
- Do not relaunch arch-shrink until either artifacts are recovered or the
  missing-artifact state is explicitly classified as unrecoverable.
