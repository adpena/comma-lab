# Codex Findings: Pact-NeRV MLX Replay + Parity Queue Closure

UTC: 2026-05-28T03:45Z

## Verdict

The Pact-NeRV DiffusionBlocks queue is no longer only an MLX advisory smoke
plan. It now owns the deterministic replay and MLX-to-PyTorch bridge artifacts
needed before any byte-closed archive promotion can be considered.

## Landed Surface

- Added `mlx_local_replay_bundle.v1` for MLX-local deterministic replay:
  repo head/status hash, platform, package versions, redacted env hash, exact
  argv, replay commands, input/output artifact hashes, and fail-closed exact
  blockers.
- Added `pact_nerv_ia3_mlx_pytorch_forward_parity.v1` for IA3 export parity:
  MLX renderer state export, PyTorch sister strict load, forward tensor compare,
  `.pt` manifest, raw-state byte count, and archive-native byte-tax marker.
- Wired both artifacts into the queue after the existing DiffusionBlocks
  schedule, diffusion distilled smoke, IA3 MLX smoke, and PR95 MLX blockwise
  control steps.

## Evidence

Executed:

```bash
.venv/bin/python tools/experiment_queue.py \
  --queue .omx/research/pact_nerv_diffusion_blocks_replay_parity_smoke_20260528Tlocal/queue.json \
  --state .omx/research/pact_nerv_diffusion_blocks_replay_parity_smoke_20260528Tlocal/queue_state.sqlite \
  run-worker --execute --max-steps 6 --max-parallel 2 \
  --noncanonical-state-rationale "local replay parity smoke state intentionally kept beside committed queue artifact" \
  --output .omx/research/pact_nerv_diffusion_blocks_replay_parity_smoke_20260528Tlocal/worker_result.json
```

Result:

- 6 steps started.
- 6 steps succeeded.
- 0 failures.
- 0 orphaned steps.
- no ready steps remain.

Key artifact facts:

- `pact_nerv_ia3_mlx_pytorch_forward_parity.json`:
  `parity_passed=true`, `max_abs_diff_255=0.0`, `mean_abs_diff_255=0.0`,
  26 tensors, 31,232 raw state bytes, 40,949 `.pt` bytes.
- `mlx_local_replay_bundle.json`:
  `local_replay_ready=true`, `missing_artifacts=[]`,
  `contest_exact_eval_ready=false`.

## Authority

All rows remain advisory:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

The exact blockers are intentional: byte-closed archive packing, receiver proof,
and contest CPU/CUDA auth eval must still sign the candidate.
