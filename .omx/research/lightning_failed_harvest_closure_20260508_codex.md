# Lightning failed harvest closure - 2026-05-08

generated_at_utc: 2026-05-08T01:54:40Z
scope: terminal closure for active Lightning score-lowering jobs
score_claim: false
dispatch_attempted: false
remote_submit_stop_cancel: false

## Summary

Two active Lightning jobs were status-polled through the existing harvest
companions with explicit state-derived `teamspace`, `user`, `ssh_target`, and
`remote_pact` arguments. Both jobs returned terminal status `failed`. Neither
expected remote artifact directory existed at the canonical
`experiments/results/lightning_batch/<job_name>/` path, so no
`contest_auth_eval.json`, score, archive, or log artifact was harvested.

The harvesters now close this failure mode explicitly instead of leaving a
phantom active claim:

- `lossy_coarsening_analytical_cuda`
  - job: `lossy-coarsening-cuda-20260508T013829Z`
  - SDK status: `failed`
  - artifact rsync: `rc=23` because remote artifact directory was missing
  - terminal claim status recorded: `failed_artifact_rsync_rc_23`
  - active job terminal status recorded locally: `failed_artifact_rsync_rc_23`
- `arch_shrink_x0.4_lightning`
  - job: `arch-shrink-x0-4-lightning-20260508T010514Z`
  - SDK status: `failed`
  - artifact rsync: `rc=23` because remote artifact directory was missing
  - terminal claim status recorded: `failed_artifact_rsync_rc_23`
  - active job terminal status recorded locally: `failed_artifact_rsync_rc_23`

## Engineering Fixes Landed In This Tranche

- `tac.deploy.lightning.harvest_env` now validates missing provider and
  artifact-rsync configuration before SDK/rsync calls.
- Harvesters select `--progress` on macOS rsync and `--info=progress2` on GNU
  rsync, preventing local toolchain incompatibility from hiding the true remote
  failure.
- Terminal failed jobs whose artifact rsync fails now append terminal claim
  rows and mark `.omx/state/lightning_active_jobs.json` terminal locally.

## Evidence Status

- Evidence grade: `invalid` for score purposes.
- Reason: terminal provider status plus missing remote artifact directory; no
  `contest_auth_eval.json` was available.
- Reactivation criterion: relaunch after correcting the remote job artifact
  creation path and preserving an active dispatch claim before submit.
