# Compact Renderer MLX Spine Runner Landed - 2026-06-01

## Verdict

Implemented the MLX/Metal-first compact renderer spine runner so PR95/HNeRV
MLX checkpoints can enter the same archive-bound representation spine as
RNeRV, SR-NeRV, BoostNeRV, PVQ-NeRV, RT-VQ-NeRV, and PACT-NeRV/VQ candidates.
The PyTorch PR95 Stage-8 lane remains a control/calibration path; this runner
is the portable MLX-first path for new compact-base work.

## What Landed

- `tools/run_compact_renderer_mlx_spine_runner.py`
  - plan-only backlog rows for compact renderer families;
  - optional PR95 MLX smoke execution;
  - executable PACT-NeRV-VQ MLX smoke/export path against real video targets;
  - per-family backend matrix so missing RNeRV/SR-NeRV/BoostNeRV/RT-VQ
    trainers are migration rows, not fake promotion candidates;
  - adaptation from `tools/run_pr95_mlx_long_training.py` reports into:
    - compact spine adapter report,
    - HPRC representation spine projection,
    - acquisition report,
    - bounded runner plan.
- `tools/emit_compact_renderer_spine_adapter.py`
  - PR95/HNeRV aliases;
  - declared pair coverage metadata.
- `src/tac/substrates/hprc/spine_acquisition.py`
  - stack roles for BoostNeRV, PVQ-NeRV, RT-VQ-NeRV;
  - SR-NeRV promoted to charged primary-carrier policy.
- `src/tac/substrates/hprc/spine_bounded_runner.py`
  - short-coverage rows no longer disappear; they remain selected as
    scale/migration work with fail-closed blockers.
- `src/tac/optimization/archive_bound_candidate_runtime_bridge.py`
  - generated receiver proofs now accept contest-style `.raw/` output
    directories, not only single-file outputs.
- `tools/build_hprc_spine_bounded_runner.py`
  - accepts receiver proof reports as first-class bounded-runner inputs.

## Smoke Artifact

PR95 checkpoint-adapter smoke:

Command:

```bash
.venv/bin/python tools/run_compact_renderer_mlx_spine_runner.py \
  --execute-pr95-mlx-smoke \
  --max-frames 4 \
  --smoke-epochs-per-stage 1 \
  --training-loss-surface rgb_yuv6_mse \
  --output-dir /Volumes/VertigoDataTier/pact/compact_renderer_mlx_spine_runner_smoke_codex \
  --overwrite
```

Result:

- Report:
  `/Volumes/VertigoDataTier/pact/compact_renderer_mlx_spine_runner_smoke_codex/compact_renderer_mlx_spine_runner_report.json`
- Mode: `executed_pr95_mlx_smoke_and_adapted`
- Selected checkpoint: epoch 8
- Exported weights: 927027 bytes
- Exported latents: 352 bytes
- Declared coverage: 2 pairs, correctly blocked below 600-pair full video
- Authority: `[macOS-MLX research-signal]`, no score claim, no promotion

PACT-NeRV-VQ byte-closed MLX smoke:

```bash
.venv/bin/python tools/run_compact_renderer_mlx_spine_runner.py \
  --execute-family pact_nerv_vq \
  --num-pairs 1 \
  --epochs 1 \
  --batch-pairs 1 \
  --compact-latent-dim 4 \
  --compact-embed-dim 4 \
  --compact-codebook-size 8 \
  --compact-decoder-channel 4 \
  --output-dir /Volumes/VertigoDataTier/pact/compact_renderer_mlx_spine_runner_pact_vq_smoke_codex_v3 \
  --overwrite
```

Result:

- Report:
  `/Volumes/VertigoDataTier/pact/compact_renderer_mlx_spine_runner_pact_vq_smoke_codex_v3/compact_renderer_mlx_spine_runner_report.json`
- Archive:
  `/Volumes/VertigoDataTier/pact/compact_renderer_mlx_spine_runner_pact_vq_smoke_codex_v3/pact_nerv_vq_mlx_training/archive.zip`
- Archive bytes: 12394
- Archive SHA-256:
  `05724288f9ad648006fe3046282a2d01eec625911261f35cc51e765273949f22`
- Receiver proof: `runtime_consumption_proof_passed=true`, output kind
  `directory`, output bytes `5674`
- Selected runner row keeps exact/full-video blockers, but no longer says
  receiver proof is missing.

## Blockers Preserved

- `archive_zip_runtime_receiver_proof_not_yet_emitted`
- `full_video_scorer_value_per_byte_not_yet_measured`
- `contest_cpu_cuda_exact_eval_missing`
- `declared_pair_coverage_below_full_video` for the 1-pair PACT-VQ smoke
- `no_full_coverage_compact_base_candidate`
- `no_full_coverage_candidate_under_any_hard_byte_ceiling`
- `mlx_local_report_is_advisory_not_score_authority`

## Next Score-Lowering Action

Run full-coverage MLX compact-base sweeps under hard byte ceilings, then admit
only receiver-proven residual sections whose measured full-video
`delta_nonrate + rate_cost < 0`. The immediate target is replacing the
~927 KB smoke-export decoder footprint with PR95-scale or smaller packed
weights/latents while preserving the same spine, receiver proof, and exact gate.
