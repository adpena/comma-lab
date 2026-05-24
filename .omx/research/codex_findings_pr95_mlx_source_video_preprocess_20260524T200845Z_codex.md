# Codex Findings: PR95 MLX Source-Video Preprocess Wiring

Date: 2026-05-24T20:08:45Z
Agent: Codex
Lane: `lane_pr95_hnerv_mlx_reproduction`
Evidence axis: `[macOS-MLX research-signal]`

## Finding

The PR95 MLX reproduction lane no longer stops at synthetic scorer-preprocess
fixtures. A bounded source-video smoke now decodes real PR95 pair frames from
`upstream/videos/0.mkv` through the upstream CPU `frame_utils.py` semantics,
downsamples them to scorer resolution through native MLX, emits YUV6 targets,
and proves gradient reachability through the MLX preprocessing path.

## Implemented Wiring

- Added exact-path upstream source-video frame loading in
  `tac.local_acceleration.pr95_hnerv_mlx_training`, avoiding ambient
  `sys.path` imports so the smoke cannot silently consume the wrong
  `frame_utils.py`.
- Added queue-visible source-video preprocess manifests with schema
  `pr95_hnerv_mlx_source_video_preprocess_smoke_v1`.
- Wired `tools/run_pr95_mlx_timing_smoke.py` flags, recommended execution,
  exact-readiness blocker updates, representation manifest sidecars, and
  false-authority postconditions.
- Wired `tools/build_pr95_mlx_optimizer_matrix_queue.py` so optimizer matrix
  plans can require source-video preprocess proof before queue steps reconcile.
- Wired `tac.optimization.representation_training_probe_integration` so
  harvested candidates carry `source_video_preprocess` signal and blockers in
  the canonical representation-training consumer payload.

## Empirical Anchor

Command:

```bash
.venv/bin/python tools/run_pr95_mlx_timing_smoke.py \
  --stage 1 --steps 1 --batch-size 1 --synthetic-pairs 1 \
  --seed 31 --base-channels 4 --latent-dim 8 \
  --output-dir experiments/results/codex_pr95_mlx_source_video_preprocess_20260524T200817Z \
  --write-source-video-preprocess-smoke \
  --source-video-path upstream/videos/0.mkv \
  --source-video-upstream-dir upstream \
  --source-video-pair-index 0 \
  --source-video-output-hw 8,10 \
  --source-video-gradient-shape 1,2,8,10,3
```

Observed:

- `source_video_loader_ready=true`
- `source_video_preprocess_ready=true`
- `frame_reader_kind=upstream_pyav_cpu`
- `video_sha256=2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`
- `source_frame_pair_shape=[1,2,874,1164,3]`
- `scorer_rgb_shape=[1,2,8,10,3]`
- `yuv6_output_shape=[1,2,4,5,6]`
- `gradient_probe.gradient_reachable=true`

Queue-owned execution was also proven through `experiment_queue.v1`:

```bash
.venv/bin/python tools/build_pr95_mlx_optimizer_matrix_queue.py \
  --stage 1 --seed 37 --steps 1 --batch-size 1 --synthetic-pairs 1 \
  --base-channels 4 --latent-dim 8 \
  --output-root experiments/results/codex_pr95_mlx_source_video_queue_20260524T201029Z \
  --queue-output experiments/results/codex_pr95_mlx_source_video_queue_20260524T201029Z/queue.json \
  --manifest-output experiments/results/codex_pr95_mlx_source_video_queue_20260524T201029Z/matrix_manifest.json \
  --queue-id codex_pr95_source_video_queue_smoke \
  --local-mlx-concurrency 1 \
  --write-source-video-preprocess-smoke \
  --source-video-path upstream/videos/0.mkv \
  --source-video-upstream-dir upstream \
  --source-video-pair-index 0 \
  --source-video-output-hw 8,10 \
  --source-video-gradient-shape 1,2,8,10,3
```

Worker result: `success_count=1`, `failure_count=0`,
`failed_postconditions=[]`, `elapsed_seconds=0.8421468330197968`.

Independent read-only subagent review also found no concrete bug in the lane.
It verified false authority preservation, explicit `upstream/frame_utils.py`
loading, PR95 pair/frame indexing, queue postconditions, harvest wiring, ignored
experiment artifacts, and a current-code `/tmp` source-video smoke with
`score_claim=false`, `promotion_eligible=false`, and
`ready_for_exact_eval_dispatch=false`.

## Exact-Readiness Interpretation

This removes only the stale class-level claim that the source-video loader is
not ported. It does not create score authority, quality authority, dispatch
authority, or promotion authority. Remaining blockers are correct:

- PR95 MLX stage training is still synthetic timing only.
- The real source-video preprocess is not yet wired into the training loop.
- Scorer loss remains unwired to the MLX source-video preprocess.
- Stage hparams, cosine schedules, QAT/resume semantics, PyTorch export parity,
  byte-closed contest archive export, receiver proof, and exact CPU/CUDA auth
  eval remain required before score claims.

## Next Engineering Target

Wire the real source-video pair provider into the PR95 MLX Stage 1/5/8 training
loop, replace synthetic targets with source-backed batches, and keep the same
queue-owned exact-readiness refusal contract until PyTorch parity and
byte-closed archive export prove the runtime path.
