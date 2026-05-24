# Codex Findings: PR95/HNeRV MLX Reproduction Lane

UTC: 2026-05-24T08:11:18Z

## Landing

- Added native MLX PR95/HNeRV reproduction primitives in `src/tac/local_acceleration/pr95_hnerv_mlx.py`.
- Added executable local timing-smoke CLI in `tools/run_pr95_mlx_timing_smoke.py`.
- Added tests covering MLX pixel shuffle, align-corners-false 2x bilinear resize, decoder output shape, PyTorch state-dict name/layout export, PR95 Stage 8 Muon/AdamW partitioning, runtime-profile false-authority, and byte-closed smoke archive determinism.

## Timing Smoke Anchor

Artifact root: `experiments/results/pr95_hnerv_mlx_timing_smokes_20260524T081106Z`

These are local MLX synthetic timing smokes at PR95 real topology (`base_channels=36`, `batch_size=1`, `synthetic_pairs=1`, `steps=1`). They are cost and execution signals only, not quality, score, promotion, rank/kill, or dispatch authority.

| Stage | Module | Seconds/step | Smoke archive bytes | Smoke archive sha256 |
|---:|---|---:|---:|---|
| 1 | `stage1_v328_ce` | 0.34745379199739546 | 2910 | `57cb02e72f8b853cbbdf765a9436feacd0bb04ad751acd1dc59b7d84c63566dc` |
| 5 | `stage5_c1a_l7` | 0.04001095803687349 | 2908 | `0480ce9a31b863556e2078e2bae65312bc8e652ed1d67415b09305f5282f99cc` |
| 8 | `stage8_muon_finetune` | 0.04175970901269466 | 2923 | `1c282064a81b8ca5a513ecdcc184badf0fd824fc2bbe52eea32f519ace8b1c60` |

Stage 1 includes the first compile/warmup cost in this run; Stage 5/8 reused warmed MLX kernels. Future queue-owned profiling should run warmup plus multi-step steady-state windows.

## Queue Consumption Proof

`tools/build_optimizer_candidate_queue.py` consumed all three representation-training sidecars and wrote:

- `experiments/results/pr95_hnerv_mlx_timing_smokes_20260524T081106Z/optimizer_candidate_queue.json`
- `schema=optimizer_candidate_queue_v1`
- `n_candidates=3`
- `dispatch_ready=0`

Rows carry `best_local_backend=mlx` and remain planning-only.

## PyTorch Export Parity Proof

The first implementation pass exposed a real parity bug: the MLX pixel-shuffle
reshape used spatial-major channel ordering, while PyTorch `PixelShuffle(2)`
uses channel-major ordering (`c * r^2 + i * r + j`). The fix is covered by
`test_pixel_shuffle_2x_nhwc_matches_pytorch_channel_major_order`.

`test_public_pr95_pytorch_state_load_matches_mlx_forward` now loads the public
PR95 `HNeRVDecoder` source model with `base_channels=4`, transfers its PyTorch
`state_dict` into `HNeRVDecoderMLX`, and verifies a paired forward pass:

- max absolute output drift <= `1e-4`
- mean absolute output drift <= `1e-5`
- output layout `(B, 2, 3, 384, 512)` matches PR95 PyTorch

This is source-model forward parity for the decoder topology and state layout.
It is still not a public checkpoint/runtime/score proof.

## Authority Boundary

The smoke archive is byte-closed for queue plumbing but intentionally refuses exact readiness:

- synthetic targets do not establish contest quality;
- PR95 runtime-consumption proof is missing;
- receiver proof is missing;
- PyTorch forward parity on a source checkpoint is still required;
- byte-closed contest archive export is still required;
- exact CPU/CUDA auth eval is still required before any score claim.

All emitted manifests preserve `score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false`, and related false-authority flags.

## Next Implementation Hooks

- Add source-checkpoint PyTorch-to-MLX load and paired forward-parity smoke.
- Add PR95 codec export bridge that converts MLX parameters/latents into the existing PR95 NumPy/PyTorch archive grammar.
- Replace synthetic targets with local MLX-compatible decoder/loss timing windows; full source-faithful MLX training remains blocked until the scorer-loss gradient path is real or explicitly bridged.
- Compile Stage 1/5/8 smoke manifests into `experiment_queue.v1` through local training queue surfaces.
