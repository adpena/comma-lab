# FA1 Stage-Transition Soft Velocity Blend Route

status: QUEUED-DESIGN
source: ddm_fa1 rank-1 `ADOPT-CLASS`
consumer_trainer: `experiments/train_levelset_witness_realized_through_R_mlx.py`
budget: USD 0 unless a future charter explicitly owns a launch

## Route

The admissible FA1 signal is not "adopt FlowAdam." It is a default-off Pact-local stage-boundary treatment:

`StageTransitionSoftVelocityBlend(previous_state, reset_state, alpha, clip_rms)`

The treatment must be tested only as a deterministic replay/backtest before any trainer-control use.

## Required Backtest

Backtest name: `fa1_stage_transition_soft_velocity_blend_replay`

Inputs:

- one stage-boundary checkpoint with optimizer state before the reset,
- the deterministic mini-batch / gradient sequence used for the first post-boundary steps,
- the current reset baseline,
- the `v <- v_prev` control named by the GC15 reset-operator corpus,
- one bias-corrected reset control at matched update RMS.

Metrics:

- effective-LR spike at the boundary,
- update RMS and clip RMS,
- descent alignment against recent gradients,
- component replay deltas for d_seg-facing loss terms,
- no scorer, no archive, no long training continuation.

Adoption gate:

- default-off only,
- wins against both controls at matched update RMS,
- writes a replay receipt with the checkpoint sha, batch identity, config, and exact touched trainer commit.

## Non-Adoptions

- Do not import FlowAdam as an optimizer.
- Do not use the FlowAdam EMA detector as a schedule controller.
- Do not project FlowAdam paper benchmark percentages into Pact score units.

