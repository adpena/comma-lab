# Codex Findings: PR95 MLX Full Source-Video Runtime Control Profile

Generated: 2026-05-25T15:08Z
Agent: Codex
Topic: PR95/HNeRV MLX reproduction lane

## Result

Landed and executed the first canonical PR95 MLX full source-video runtime
control profile. The profile is intentionally non-promotional and false-authority:
it is a local MLX timing and custody signal, not a score claim.

Artifact root:

- `experiments/results/pr95_mlx_full_source_video_runtime_profile_20260525T150639Z/`

Queue:

- `experiment_queue.json`
- `matrix_manifest.json`
- SQLite state: `.omx/state/experiment_queue_pr95_mlx_full_source_video_runtime_profile_20260525T150639Z.sqlite`

## Control Profile Contract

Profile id: `full_pr95_source_video_runtime`

Pinned settings:

- stages: `1,5,8`
- batch size: `1`
- synthetic pairs: `1`
- base channels: `36`
- latent dim: `28`
- source video: `upstream/videos/0.mkv`
- source-video output HW: `384,512`
- source-video pair index: `0`
- source-video loss: `rgb_yuv6_mse`
- write PR95 public archive export: true
- prove runtime consumption: true
- write source-video preprocess smoke: true
- local MLX concurrency: `1`

This removes ad hoc PR95 smoke knobs from the operator path and makes the PR95
control arm queue-owned.

## Execution Evidence

Queue validation:

- `tools/experiment_queue.py validate`: valid, 3 experiments, 3 steps
- `tools/experiment_queue.py observe`: healthy, 3 succeeded, 0 blockers, 0
  orphaned steps, 0 definition-drift steps

Executed stages:

| Stage | Module | Optimizer | Process seconds | Manifest train seconds | Runtime proof |
|---:|---|---|---:|---:|---|
| 1 | `stage1_v328_ce` | AdamW baseline | 1.7903 | 0.0430 | proven |
| 5 | `stage5_c1a_l7` | AdamW baseline | 1.7799 | 0.0406 | proven |
| 8 | `stage8_muon_finetune` | Muon + AdamW | 1.5366 | 0.0359 | proven |

Aggregate queue telemetry:

- local MLX runs: 3
- failures: 0
- elapsed seconds sum: 5.1068
- elapsed seconds mean: 1.7023
- artifact records: 21
- artifact bytes tracked: 1,015,547

## Exact-Readiness State

Each artifact remains correctly non-promotional:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

Active blockers are expected and useful:

- local PR95 timing smoke is not a score
- PR95 stage hparams and cosine schedules are not fully source-matched
- PR95 QAT/C1a/resume semantics are not ported to MLX
- PyTorch export forward parity is not established
- receiver proof is missing
- byte-closed contest archive export is not sufficient without exact eval
- scorer network loss is not wired to MLX
- full-frame inflate parity against the source runtime has not been run
- RGB+YUV6 preprocess loss is not full scorer loss
- runtime-consumption smoke is not score authority

## Why This Matters

This establishes the PR95 control lane as an executable queue/DAG surface rather
than a one-off local smoke. The same profile can now be used to measure scaling
when we add source-faithful curriculum, QAT/C1a, PyTorch export parity, receiver
proofs, and byte-closed full-frame parity.

It also gives a practical speed anchor: full-size source-video single-step MLX
control smokes are cheap enough to run as routine regression checks while the
automated final rate attack lane continues independently.

## Next Patch Set

1. Add a multi-step/full-run PR95 MLX curriculum profile that keeps the same
   queue authority and false-authority contract.
2. Wire PR95 scorer loss targets into MLX training as a distinct profile that
   remains advisory until calibrated against contest CPU/CUDA.
3. Add PyTorch export parity for the MLX checkpoint produced by this control
   lane.
4. Add byte-closed full-frame inflate parity against the source PR95 runtime.
5. Promote receiver proof from a blocker string to an executable postcondition.
6. Reuse this profile schema for HNeRV variants, BoostNeRV bolt-ons, NeRV-family,
   and non-NeRV substrates instead of adding new ad hoc smoke scripts.
