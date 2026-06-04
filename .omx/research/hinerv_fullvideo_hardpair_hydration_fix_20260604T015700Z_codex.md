# HiNeRV full-video hard-pair hydration fix

Axis: `[macOS-MLX research-signal]`; no score claim.

## Finding

Sparse-only hard-pair hydration was a real negative on the live 600-pair HiNeRV
candidate. The epoch 11,749 sparse hard-pair export was receiver-proof-ready
and compact at 121,518 bytes, but full-video MLX replay scored 87.901691 versus
87.162211 for the epoch 11,499 full-hydration baseline. The sparse run preserved
rate but trained against a changed local target distribution instead of the
full-video contest objective.

## Fix

Hard-pair prioritization is now a sampler pressure by default while keeping all
full-video targets hydrated. Sparse arbitrary source-pair target hydration is
still available only through `--sparse-prioritized-target-hydration` for explicit
memory experiments. Runner metadata records the consumed mode so downstream
feedback cannot confuse sampler priority with changed target coverage.

The same landing also exposes `--hi-nerv-pr95-curriculum-total-epochs`, allowing
bounded/resumed HiNeRV chunks to preserve a PR95-scale curriculum clock (for
example 29,650) instead of silently shrinking stage timing to the chunk length.

## Evidence

- `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_sparse_hardpair_epoch11749_full600_mlx_replay_20260604T015006Z/mlx_response_gpu_full600.json`
- `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_epoch11499_full600_mlx_replay_20260604T013738Z/mlx_response_gpu_full600.json`
- `uv run pytest src/tac/tests/test_compact_renderer_mlx_spine_runner.py -q -k 'prioritized_pair or priority_hydration or pr95_curriculum_total_epochs'`
- `uv run ruff check tools/run_compact_renderer_mlx_spine_runner.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py`
- `uv run python -m py_compile tools/run_compact_renderer_mlx_spine_runner.py`

## Next

Stop the stale sparse-only successor and relaunch from epoch 11,499 with
full-video hydration, hard-pair sampler pressure, and a PR95-scale curriculum
clock. Harvest the first checkpoint through archive export, receiver proof,
full-video MLX replay, and feedback before exact spend.
