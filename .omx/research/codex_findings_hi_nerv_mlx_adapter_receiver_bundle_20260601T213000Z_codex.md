# Codex Findings: HiNeRV MLX Adapter Receiver Bundle

UTC: 2026-06-01T21:30:00Z

## Verdict

HiNeRV is no longer a pure planner-gated missing-adapter lane. The repo now has
a real MLX renderer/export/receiver bundle path:

- `tac.substrates.hi_nerv.mlx_renderer.HinervSubstrateMLX`
- `tac.substrates.hi_nerv.archive_candidate.export_hi_nerv_mlx_archive`
- HIV1 decoder-state codec support via the shared int8/int4/int2/fp16 envelope
- raw contest receiver output through `tac.substrates._shared.inflate_runtime`
- `tools/run_compact_renderer_mlx_spine_runner.py --execute-family hi_nerv`

This is still not score movement. The runner explicitly marks the run as an
initialized adapter smoke, not trained evidence, and blocks on score-aware
training, full-video MLX scorer replay, local CPU replay, and exact CPU/CUDA
auth.

## Durable Smoke Artifact

Command:

```bash
/Users/adpena/Projects/pact/.venv/bin/python tools/run_compact_renderer_mlx_spine_runner.py \
  --execute-family hi_nerv \
  --num-pairs 1 \
  --compact-latent-dim 4 \
  --compact-embed-dim 4 \
  --compact-decoder-channel 4 \
  --output-dir /Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_adapter_smoke_20260601T212946Z \
  --repo-root /Volumes/VertigoDataTier/pact/codex_hinerv_execute_gate_20260601T2120
```

Result:

- report: `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_adapter_smoke_20260601T212946Z/compact_renderer_mlx_spine_runner_report.json`
- archive: `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_adapter_smoke_20260601T212946Z/hi_nerv_mlx_adapter_smoke/archive.zip`
- archive bytes: `25312`
- archive sha256: `747f9b1063c04e420f491fe1be51ed67556b05f965dccfb89e9e7106d7a29930`
- HPRC projection bytes: `9441`
- receiver proof: `runtime_consumption_proof_ready=true`
- receiver raw bytes during proof: `6104016`
- receiver raw retained: `false`

## Blockers Preserved

The runner reports these blockers by design:

- `hi_nerv_score_aware_training_not_executed`
- `hi_nerv_initialized_adapter_smoke_not_score_fit`
- `full_video_mlx_scorer_replay_not_attached`
- `local_cpu_replay_not_executed`
- `contest_cpu_cuda_exact_eval_not_executed`
- no full-coverage compact-base candidate yet

## Next Engineering Step

Replace the initialized adapter smoke with the actual HiNeRV MLX training lane:
real SegNet/PoseNet teacher caches, PR95-faithful staged optimizer/QAT/C1a
pressure, export of the trained EMA state through the same archive candidate,
full-video MLX prefilter, local CPU replay, and exact CPU/CUDA only after local
win.
