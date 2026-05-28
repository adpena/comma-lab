# Codex Findings: PR95 MLX Stage 8 256-Step Source-Video Control Run

UTC: 2026-05-28T13:13:48Z

## Scope

Queue-owned bounded PR95/HNeRV MLX Stage 8 source-video training run.

Artifacts:

- `.omx/research/pr95_mlx_stage8_source_video_256x16_queue_20260528T131156Z/queue.json`
- `.omx/research/pr95_mlx_stage8_source_video_256x16_queue_20260528T131156Z/worker_result.json`
- `.omx/research/pr95_mlx_stage8_source_video_256x16_queue_20260528T131156Z/manifest.json`
- `.omx/research/pr95_mlx_stage8_source_video_256x16_queue_20260528T131156Z/queue_performance.json`

## Findings

- The queue executed successfully: `success_count=1`, `failure_count=0`, with no failed postconditions.
- The run used 16 real source-video pair targets, pair indices `0..15`, with `rgb_yuv6_mse`.
- The run completed 256 Stage 8 optimizer steps in `16.771938624995528` worker seconds.
- Runtime profile reported `seconds_per_step=0.05564628206252564` and `examples_per_second=17.970652538409905`.
- Final local proxy loss was `0.13475480675697327`; the earlier 2-step/2-pair control-arm smoke recorded `0.15645083785057068`, so the local training path is doing real optimization work instead of only exercising plumbing.
- The MLX GPU drift attestation passed with `max_abs=0.000030517578125`, `mean_abs=0.0000021553069018409587`, `atol_max=0.002`, and `atol_mean=0.0001`.
- The emitted PR95-compatible archive export is byte-closed and declared through packet-compiler bridge metadata, but runtime-consumption proof remains missing for this run.

## Authority

This is `[macOS-MLX research-signal]` and local training-control evidence only. It is not a contest score, promotion authority, rank/kill authority, or exact-eval dispatch authority.

Remaining blockers:

- full SegNet/PoseNet scorer network loss is not wired into MLX training;
- PR95 full schedule/QAT/resume fidelity is not yet complete;
- full-frame inflate parity and PR95 public runtime consumption proof are required;
- exact CPU/CUDA auth eval is required before any score claim.

## Integration

This run validates the new single-entry queue-builder path at a more useful scale. The next PR95 MLX tranche should turn this from bounded control smoke into resumable multi-stage local MLX training with scorer-loss wiring, checkpoint custody, and byte-closed receiver proof.
