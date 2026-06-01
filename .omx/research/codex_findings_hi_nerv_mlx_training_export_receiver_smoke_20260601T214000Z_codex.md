# Codex Findings: HiNeRV MLX Training Export Receiver Smoke

UTC: 2026-06-01T21:40:00Z

## Verdict

HiNeRV now has a real MLX train/export/archive/receiver path in the compact
renderer spine runner. This supersedes the earlier initialized-adapter-only
smoke: `--execute-family hi_nerv` trains on real video targets, exports the
trained EMA state through HIV1, emits an HPRC spine projection, runs the
generated receiver proof, and reports exact remaining blockers.

This is still false-authority MLX-local evidence, not score movement. The smoke
is one pair and one epoch with no SegNet/PoseNet teachers, no full-video MLX
prefilter, no local CPU replay, and no exact CPU/CUDA auth.

## Durable Smoke Artifact

Command:

```bash
/Users/adpena/Projects/pact/.venv/bin/python tools/run_compact_renderer_mlx_spine_runner.py \
  --execute-family hi_nerv \
  --num-pairs 1 \
  --epochs 1 \
  --batch-pairs 1 \
  --compact-latent-dim 4 \
  --compact-embed-dim 4 \
  --compact-decoder-channel 4 \
  --source-video-path /Users/adpena/Projects/pact/upstream/videos/0.mkv \
  --output-dir /Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_training_smoke_20260601T213927Z \
  --repo-root /Volumes/VertigoDataTier/pact/codex_hinerv_execute_gate_20260601T2120
```

Result:

- report: `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_training_smoke_20260601T213927Z/compact_renderer_mlx_spine_runner_report.json`
- archive: `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_training_smoke_20260601T213927Z/hi_nerv_mlx_training/archive.zip`
- archive bytes: `26173`
- archive sha256: `6e84c5e4542e96ae49ae4a0f2f2b3d497881fd6bdbd56ebee55d9a8015c1ff47`
- receiver proof: `runtime_consumption_proof_ready=true`
- receiver raw bytes during proof: `6104016`
- receiver raw retained: `false`

## Exact Blockers Remaining

- `hi_nerv_pr95_faithful_curriculum_requires_min_8_epochs`
- `hi_nerv_real_segnet_posenet_teachers_not_both_attached`
- `full_video_mlx_scorer_replay_not_attached`
- `local_cpu_replay_not_executed`
- `contest_cpu_cuda_exact_eval_not_executed`
- no full-coverage compact-base candidate yet

## Next Engineering Step

Run the same path with `--epochs >= 8` to activate the PR95-faithful staged
curriculum, then attach real SegNet and PoseNet teachers with the existing
`--segnet-distillation-weight` and `--pose-distillation-weight` flags. The next
score-lowering gate is full-video MLX prefilter plus local CPU replay; exact
CPU/CUDA stays blocked until a local replay winner exists.
