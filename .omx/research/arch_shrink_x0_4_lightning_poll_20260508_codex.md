# Arch Shrink x0.4 Lightning Poll - Codex - 2026-05-08

generated_at_utc: 2026-05-08T06:29:39Z
lane_id: `arch_shrink_x0.4_lightning`
job_name: `arch-shrink-x0-4-lightning-20260508T024304Z`
scope: read-only live custody poll; no score claim; no evidence JSONL append

## Status

- Harvester command:
  `.venv/bin/python experiments/arch_shrink_x0.4_lightning_harvest.py --job-name arch-shrink-x0-4-lightning-20260508T024304Z --teamspace comma-lab --user adpena --ssh-target s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai --once`
- Harvester result: `status=running` at `2026-05-08T06:29:26Z`.
- Remote artifact mirror:
  `/teamspace/jobs/arch-shrink-x0-4-lightning-20260508t024304z/artifacts/pact/experiments/results/lightning_batch/arch-shrink-x0-4-lightning-20260508T024304Z`
- Latest heartbeat observed: `2026-05-08T06:28:35Z`, GPU `94 %`, memory `3655 MiB`.
- Latest train log observed: epoch `130/3000`, Phase 1, about `94.5s/ep`, ETA `75.3h`.
- Half-frame training signal is active: `hf_fires=150/150 (1.00)` and `hf_target_prob=1.000`.

## Terminal Artifacts

Absent as of this poll:

- `archive.zip`
- `contest_auth_eval.json`
- `auth_eval_work/contest_auth_eval.json`
- `auth_eval.log`

## Classification

Evidence grade: `in_progress / invalid for scoring`.

This is neither a win nor a negative. The only admissible conclusion is that
the active Lightning job is still training and has not reached final archive
packaging or exact CUDA auth eval.

## Next Action

Poll later with the same job name. Do not relaunch while the active dispatch
claim remains open and the remote heartbeat is current. If the SDK status
becomes terminal, run the harvester and then complete exact result review
before appending any `[contest-CUDA]` evidence row.

## Poll Update - 2026-05-08T06:48:50Z

- Harvester result: `status=running` at `2026-05-08T06:46:22Z`.
- Remote heartbeat remains current through `2026-05-08T06:45:36Z`, GPU `100 %`,
  memory `3655 MiB`.
- Latest train log observed: epoch `140/3000`, Phase 1, about `94.6s/ep`,
  ETA `75.1h`.
- Half-frame instrumentation remains active:
  `hf_fires=150/150 (1.00)`, `hf_warp_diff=0.0280`,
  `hf_target_prob=1.000`.
- Terminal artifacts remain absent: no `archive.zip`, no
  `contest_auth_eval.json`, no `auth_eval_work/contest_auth_eval.json`, no
  `auth_eval.log`.

### Architecture String Audit

The train log's `NO motion, NO warp` string is consistent with the intended
Q-FAITHFUL architecture. `q_faithful_dilated_88k` routes
`variant=quantizr_faithful` to `JointFrameGenerator`, which consumes the odd
mask and deployed pose stream and intentionally discards `mask_t`; it is not
the legacy `AsymmetricPairGenerator` warp path. The current profile still sets
`use_zoom_flow=True` as historical half-frame compatibility plumbing, so the
training loop constructs `ego_flow` and the shim swallows it. That is awkward
metadata/plumbing, not evidence that the live job is training the wrong
architecture. A follow-up review should decide whether to split this into an
explicit `single_mask_halfframe=True` contract to remove the overloaded
`use_zoom_flow` signal.

## Poll Update - 2026-05-08T07:00:33Z

- Harvester result: `status=running` at `2026-05-08T07:00:33Z`.
- Remote heartbeat remains current through `2026-05-08T06:59:37Z`, GPU
  `100 %`, memory `3655 MiB`.
- Latest train log observed: epoch `150/3000`, Phase 1, about `94.5s/ep`,
  ETA `74.8h`.
- Half-frame instrumentation remains active:
  `hf_fires=150/150 (1.00)`, `hf_warp_diff=0.0267`,
  `hf_target_prob=1.000`.
- Terminal artifacts remain absent: no `archive.zip`, no
  `contest_auth_eval.json`, no `auth_eval.log`.

Classification remains `in_progress / invalid for scoring`. Do not relaunch
or close the active claim while the heartbeat is current.

## Poll Update - 2026-05-08T07:11:54Z

- Harvester result: `status=running` at `2026-05-08T07:11:54Z`.
- `--once` was set, so the harvester exited non-terminal.
- No terminal score artifact was harvested in this poll.
- L3 live-strategy review recommends continuing to observe and not stopping
  the job. It also flags that the observed ~94.5s/epoch pace makes the active
  3000-epoch T4 run unlikely to finish archive build plus exact auth eval
  inside the current 18h cap, so this dispatch may primarily produce checkpoint
  and loss-curve signal unless resumed or harvested terminally later.

Classification remains `in_progress / invalid for scoring`. No score claim.
