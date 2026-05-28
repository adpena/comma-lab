# Pact-NeRV DiffusionBlocks MLX Queue Wiring

Codex landed the DiffusionBlocks implication as an executable local-MLX queue rather than a prose-only research note.

## Contract

- Paper basis: `arxiv:2506.14202v3`.
- Queue schema: `pact_nerv_diffusion_blocks_mlx_queue.v1`.
- Schedule schema: `pact_nerv_diffusion_blocks_schedule.v1`.
- Mathematical objective:
  `argmin_theta sum_b E_u in I_b[D_b(x_u,u,c)-x_0]^2 + lambda_byte*DeltaBytes + lambda_axis*AxisShiftPenalty + sum_ij psi_ij(stack_interaction)`.
- Interaction axes: pixel, bit, byte, frame, pair, region, boundary, batch, full-video.
- Entropy positions: before entropy coder, at entropy coder, after entropy coder.
- Receiver contract: deterministic student only; no MLX runtime in inflate; teacher is compress-time only.

## Executed Queue

- Queue artifact: `.omx/research/pact_nerv_diffusion_blocks_mlx_smoke2_20260528Tlocal/queue.json`.
- Worker proof: `.omx/research/pact_nerv_diffusion_blocks_mlx_smoke2_20260528Tlocal/worker_result.json`.
- External output root: `/Volumes/VertigoDataTier/experiments/results/pact_nerv_diffusion_blocks_mlx_smoke2_20260528Tlocal`.
- Steps executed:
  - `emit_diffusion_blocks_schedule`
  - `run_pact_nerv_ia3_mlx_renderer_smoke`
  - `run_pr95_mlx_blockwise_control_smoke`
- Worker result: 3 started, 3 succeeded, 0 failed.

## Authority

All rows are `[macOS-MLX research-signal]` and fail closed:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

Exact score authority still requires PyTorch or numpy forward parity, full-frame inflate/runtime proof, and paired contest CPU/CUDA auth eval.

## Verification

- `ruff` passed on the new scheduler, tools, tests, IA3 MLX trainer, and MLX renderer.
- `pytest src/tac/tests/test_pact_nerv_diffusion_blocks_queue.py -q` passed: 4 tests.
- Queue validation passed for `pact_nerv_diffusion_blocks_mlx_smoke2_20260528Tlocal`.
- Queue worker executed the full local queue with all postconditions satisfied.
