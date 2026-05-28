# Codex Findings: PR95 MLX Stage 8 Source-Video Queue Control Arm

UTC: 2026-05-28T13:05:49Z

## Scope

Queue-owned PR95/HNeRV MLX Stage 8 control-arm smoke using real source-video pairs, not synthetic-only targets.

Artifacts:

- `.omx/research/pr95_mlx_stage8_source_video_queue_no_cliff_20260528T130308Z/queue.json`
- `.omx/research/pr95_mlx_stage8_source_video_queue_no_cliff_20260528T130308Z/worker_result.json`
- `.omx/research/pr95_mlx_stage8_source_video_queue_no_cliff_20260528T130308Z/manifest.json`
- `.omx/research/pr95_mlx_stage8_source_video_queue_no_cliff_20260528T130308Z/representation_training_manifest.json`
- `.omx/research/pr95_mlx_stage8_source_video_queue_no_cliff_20260528T130308Z/pr95_public_archive.zip`

## Findings

- The queue executed one local MLX training step group successfully: `success_count=1`, `failure_count=0`, with no failed postconditions.
- The executed command used `--train-on-source-video-pairs`, pair indices `0` and `371`, and `--source-video-loss-surface rgb_yuv6_mse`.
- Stage 8 used the source-faithful optimizer descriptor `pr95_stage8_muon_adamw_mlx`.
- The harvested MLX drift default flowed into the execution path as `--mlx-gpu-drift-conv2d-override-preset blocks02_kahan_fp32`.
- Runtime telemetry reported `seconds_per_step=0.03019218749977881` and `examples_per_second=33.12115096023851` for this tiny bounded smoke.
- The source-video loader and native MLX YUV6 preprocess smoke both reported ready.
- The local MLX forward drift attestation passed with `max_abs=0.000030517578125`, `mean_abs=0.0000037957861422910355`, `atol_max=0.002`, and `atol_mean=0.0001`.
- A PR95-compatible archive export was emitted at `pr95_public_archive.zip`, `230434` bytes, SHA-256 `89a55cd81c8913a49e0e168efe4d5cfeeab52ef8f0d7c27eb3fb76ad1eeeb869`.

## Authority

This is `[macOS-MLX research-signal]` and local training-control evidence only. It is not a contest score, not promotion authority, and not exact-eval dispatch authority.

Remaining blockers are explicit in the manifests:

- full PR95 stage hparams, schedules, QAT C1A, and resume semantics are not yet 1:1 ported;
- SegNet/PoseNet network loss is not yet wired to the MLX training objective;
- full-frame inflate parity and exact CPU/CUDA auth eval are required before any score claim;
- the emitted archive export is byte-closed but not yet runtime-consumed by the PR95 public inflate path in this run.

## Integration

This closes the loop from drift-scope recommendation to queue-owned source-video-backed MLX execution. The next automation step is to make this a single operator-facing control-arm queue builder so plan generation, queue generation, execution, telemetry harvest, and refusal gates are not manually chained.
