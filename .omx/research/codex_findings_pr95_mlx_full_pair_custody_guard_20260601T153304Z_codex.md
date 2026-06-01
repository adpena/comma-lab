# PR95 MLX Full-Pair Custody Guard

## Finding

The PR95 MLX timing/export CLI defaulted source-video training to pair `[0]`
when `--source-video-pair-index` was omitted. That made it too easy for a
source-video training/export row to look like a full-video PR95 candidate while
only emitting one latent row.

## Landing

- `tools/run_pr95_mlx_timing_smoke.py` now resolves source-video training pair
  selection as explicit indices, else `--source-video-pair-count`, else
  `range(--synthetic-pairs)`.
- `tools/build_pr95_mlx_optimizer_matrix_queue.py` uses the same selection rule
  before emitting queue commands.
- The regression test proves a 3-pair source-video training plan emits
  `[0, 1, 2]`, not `[0]`.

## Empirical Artifact

Full-coverage advisory smoke:

- path:
  `/Volumes/VertigoDataTier/pact/pr95_mlx_stage8_source_yuv6_600pair_1step_codex_v1`
- command family: `tools/run_pr95_mlx_timing_smoke.py --stage 8 --steps 1
  --synthetic-pairs 600 --train-on-source-video-pairs
  --source-video-pair-count 600 --write-pr95-public-archive-export`
- archive: `pr95_public_archive.zip`
- archive bytes: `247340`
- archive SHA-256:
  `9f4c8cd8bfe1eba26ed516975fad93d2b08bb3d3936ae712aeaa75cfb845387c`
- parsed latent shape: `[600, 28]`
- target shape: `[600, 2, 3, 384, 512]`

This artifact is `[macOS-MLX advisory]` and has no score, promotion, or exact
authority. Its value is custody: future PR95/HNeRV compact-base sweeps can now
prove that they trained and exported the intended full video before spending on
receiver proof or exact CPU/CUDA gating.

## Next Execution

The next score-lowering step is not another partial export. Run the same
full-pair path with the scorer-faithful compact-base loss once wired, then
receiver-proof only byte-closed archives whose local full-video evidence can
plausibly beat the current frontier byte budget.
