# Active Dispatch Harvest Status - 2026-04-30

Checked at: 2026-04-30T14:29:23Z; refreshed after new live instance appeared at 2026-04-30T14:30:32Z

Scope: Vast active instances only. No code edits. No instance destruction. `contest_auth_eval.json` treated as authoritative completion signal.

## Vast live inventory

`/Users/adpena/Projects/pact/.venv/bin/vastai show instances --raw` returned exactly 3 live instances on the final refresh:

| instance_id | label | actual_status | cur_state | gpu_util | dph_total | ssh | status |
|---|---|---|---|---:|---:|---|---|
| 35885106 | lane_hm_s_2026-04-30_b_a2 | running | running | 27.999973 | 0.2592407407407407 | ssh8.vast.ai:15106 | training in progress |
| 35899850 | lane_19_logit_margin_2026-04-30_b_a4 | running | running | 37.999966 | 0.2632407407407408 | ssh2.vast.ai:19850 | training in progress |
| 35904766 | lane_sa_segmap_clone_2026-04-30_codex_a1 | running | running | 99.999908 | 0.28972222222222227 | ssh4.vast.ai:24766 | early setup / no lane logs yet |

The local `.omx/state/vastai_active_instances.json` tracker contains many historical records and does not reflect live Vast state by itself. The active-dispatch table also has stale rows for retry attempts; live Vast state is authoritative for this pass.

## Instance 35885106

Lane: `lane_hm_s_2026-04-30_b_a2`

Remote workspace: `/workspace/pact/lane_hm_s_segmap_homography_results`

Observed process:

`/opt/conda/bin/python -u experiments/train_segmap.py --variant kl_distill --arch segmap_homography ... --epochs 600 ...`

Status:

- SSH reachable.
- Vast reports `actual_status=running`, `cur_state=running`.
- Heartbeat latest line: `2026-04-30T14:24:03Z`, GPU 64%, 6694 MiB.
- Training log latest epoch: `epoch=396` of 600.
- Lane-local `contest_auth_eval.json` count: 0.
- Lane-local files present: `provenance.json`, `run.log`, `heartbeat.log`, `train.log`.
- No lane-local `archive*.zip`, renderer, masks, checkpoint, or authoritative contest eval output found.

Harvest decision: no harvest performed. The lane is not complete and has no authoritative lane-local `contest_auth_eval.json`.

Risk note: this active lane is running `--variant kl_distill`, which conflicts with the current repo instruction that KL distill is dead. I did not stop or destroy it because it is not conclusively complete and artifact harvest is not possible yet.

## Instance 35899850

Lane: `lane_19_logit_margin_2026-04-30_b_a4`

Remote workspace: `/workspace/pact/lane_19_logit_margin_results`

Observed process:

`/opt/conda/bin/python -u src/tac/experiments/train_renderer.py --profile lane_19_logit_margin --tag lane_19_logit_margin --device cuda ... --use-qat --no-auth-eval-on-best`

Status:

- SSH reachable.
- Vast reports `actual_status=running`, `cur_state=running`.
- Heartbeat latest line: `2026-04-30T14:28:00Z`, GPU 49%, 2623 MiB.
- Training log latest epoch: `ep 430/1980`, Phase 2.
- Lane-local `contest_auth_eval.json` count: 0.
- Lane-local files present: `provenance.json`, `run.log`, `heartbeat.log`, `train.log`, `train/training_state_lane_19_logit_margin.pt`.
- No lane-local `archive*.zip` or authoritative contest eval output found.

Harvest decision: no harvest performed. The lane is not complete and has no authoritative lane-local `contest_auth_eval.json`.

## Instance 35904766

Lane: `lane_sa_segmap_clone_2026-04-30_codex_a1`

Tracker registration: `2026-04-30T14:29:23Z`, script `scripts/remote_lane_sa_segmap_clone.sh`

Remote workspace: `/workspace/pact`

Status:

- SSH reachable.
- Vast reports `actual_status=running`, `cur_state=running`.
- Vast duration at refresh: 74.9171793460846 seconds.
- Process list only showed container launch wrappers (`/.launch`); no lane script, trainer, or `contest_auth_eval.py` process yet.
- `/workspace/pact` exists, but no lane result directory, lane-local logs, heartbeat, or `contest_auth_eval.json` were present.
- Generic repo archives under `submissions/` exist, but they are not lane-local completion artifacts.

Harvest decision: no harvest performed. The lane is too early and has no authoritative lane-local `contest_auth_eval.json`.

## Non-live active-dispatch rows

`.omx/state/active_dispatches.md` still lists current rows for `lane_12_nerv_2026-04-30_b`, `lane_8_multipass_2026-04-30_b`, and `lane_17_imp_10cycle_2026-04-30T124951Z`, plus an older `lane_19` attempt ID. These rows were not live in Vast at query time. I found no local `launch_lane_with_retry.py` process still running for those dispatches, and `/tmp/dispatch_lane_*` logs matching those names were not present.

Exact live conclusion:

- `lane_12_nerv_2026-04-30_b`: no live Vast instance at query time.
- `lane_8_multipass_2026-04-30_b`: no live Vast instance at query time.
- `lane_17_imp_10cycle_2026-04-30T124951Z`: no live Vast instance at query time.
- `lane_19_logit_margin_2026-04-30_b`: live attempt is `35899850` (`_a4`), not the stale `_a1` ID in active_dispatches.
- `lane_sa_segmap_clone_2026-04-30_codex_a1`: newly live during this pass as `35904766`; not listed in active_dispatches at the time I read it.

## Actions taken

- Queried live Vast inventory with `.venv/bin/vastai show instances --raw`.
- SSH-probed all live instances.
- Inspected lane-local logs and artifact/eval paths.
- Did not copy artifacts, because no active lane had a lane-local authoritative `contest_auth_eval.json`.
- Did not destroy any instances.
