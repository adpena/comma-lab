# PR95 MLX Queue-Owned Long Training Plan

Timestamp: 2026-06-01T19:09:28Z
Author: Codex
Status: LANDED_QUEUE_OWNERSHIP_FIX

## Finding

The prior PR95 29,650-epoch MLX long-training launch artifact was not a live
training run: `train.pid` pointed at no running process and `train.log` was
empty. Treating that as "launched" would be false progress.

## Fix

`tools/run_pr95_mlx_long_training.py` now supports dry-run execution planning:

- `--dry-run-execute` emits a plan whose `recommended_execution` is the same
  command with `--execute`.
- `--dry-run-execute-smoke` emits a plan whose `recommended_execution` is the
  same command with `--execute-smoke`.

This moves long training under `experiment_queue.v1` ownership: SQLite state,
logs, postconditions, pause/resume/rewind, and observer recovery now own the
run instead of a loose shell PID.

## Artifact

Queue root:
`/Volumes/VertigoDataTier/pact/pr95_mlx_long_training_queue_20260601T190928Z`

Ready queue state:
`/Users/adpena/Projects/pact/.omx/state/experiment_queue_pr95_mlx_long_training_29650ep_seed0_queue_owned_clean.sqlite`

The queued step is `local_mlx`, total epochs are 29,650, and the queued command
contains `--execute`. It remains a `[macOS-MLX research-signal]` control arm
only until PyTorch export parity, full-frame inflate parity, and exact CPU/CUDA
auth gates pass.

## Score-Lowering Implication

We are close on long-training mechanics: the previous timing smoke measured
~0.219 seconds per full-video step, so the raw 29,650-step pass is plausibly
hours, not days, before checkpoint/export overhead. We are not yet close enough
on score movement from this lane because the current loss is RGB/YUV6
reconstruction, not the full SegNet/PoseNet contest objective.

Next required score-lowering step: bind scorer-aware P18/P19 loss or calibrated
MLX scorer response into the queued local training lane, then export/archive
only byte-closed local winners.
