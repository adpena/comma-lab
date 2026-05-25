# PR95 MLX Stage 7 sigma_sweep Curriculum Build Landed

Generated: 2026-05-25
Agent: Codex
Axis: [macOS-MLX research-signal]
Lane: `lane_pr95_mlx_stage_7_sigma_sweep_curriculum_build_20260525`

## Goal

Close the recovered PR95 8-stage MLX timing spine by adding Stage 7
`stage7_sigma_sweep` before Stage 8 Muon finetune. Stage 7 is a Stage 6
continuation: AdamW remains at `3e-5`, loss family remains
`l7_softplus_seg_loss`, C1a lambda remains `0.02`, QAT remains enabled, and
the distinguishing sweep parameter is C1a sigma `0.2 -> 0.1`.

This is not a contest score claim. It is local MLX replacement-training
infrastructure and queue-owned timing/profile signal. Exact CPU/CUDA auth eval
and byte-closed runtime proof remain mandatory before score, promotion, rank, or
kill authority.

## Landed Surface

- `PR95_STAGE_MODULES[7] = "stage7_sigma_sweep"`
- `PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS[7] =
  "pr95_stage7_adamw_sigma_sweep_mlx"`
- Descriptor `pr95_stage7_adamw_sigma_sweep_mlx` records:
  - optimizer: `mlx.optimizers.AdamW`
  - AdamW LR: `3e-5`
  - stage epochs: `3000`
  - loss family: `l7_softplus_seg_loss`
  - C1a lambda: `0.02`
  - C1a sigma: `0.1`
  - QAT: `true`
  - Muon: `false`
  - false-authority fields: all false
- `full_pr95_source_video_runtime` now emits all stages:
  `[1, 2, 3, 4, 5, 6, 7, 8]`.

## Empirical Receipts

Stage 7 100-step local MLX smoke:

| Metric | Value |
|---|---:|
| State bytes | 915,944 |
| Seconds per step | 23.394 ms |
| Examples per second | 42.746 |
| Last loss | 0.0832345 |
| Score claim | false |
| Promotion eligible | false |
| Ready for exact eval dispatch | false |

The Catalog #313 row is
`pr95_mlx_stage_7_sigma_sweep_curriculum_build_synthetic_timing_smoke_3step`.
The 3-step JSON receipt is stored at
`.omx/research/codex_pr95_stage7_sigma_sweep_probe_20260525T1710Z/stage7_sigma_sweep_3step_smoke.json`.

## Queue Receipt

The full queue-owned profile was regenerated into:

- `.omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/pr95_mlx_full_profile_manifest.json`
- `.omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/pr95_mlx_full_profile_queue.json`

Summary:

- plan count: `8`
- queue id: `codex_pr95_stage6_stage7_full_profile_20260525T1714Z`
- manifest SHA-256: `b7abd34ca9b50c20b0dc058518959349be221fb48d109cf4360ef62007270912`
- queue SHA-256: `e52dc7548d04bfb4c93d4f602f9c340f562c605206282b80b686b27b702cc2ae`
- score/rank/promotion authority: false

## Remaining Gaps

- Source-video training is still `synthetic_timing_only`, not source faithful.
- PR95 source-video loader and scorer-loss training still need full MLX wiring.
- QAT/C1a resume semantics need source checkpoint parity.
- PyTorch export forward parity must be established on source checkpoints.
- Byte-closed public archive export and runtime-consumption proof must pass
  before exact auth eval dispatch.

## Next Action

Use the 8-stage queue as the PR95 reproduction control arm while implementing
source-faithful MLX training and export parity. The next build should move from
timing-proxy correctness to source-video pair training, receiver/runtime proof,
and byte-closed archive smoke.
