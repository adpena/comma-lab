# Compact PACT/VQ coder-aware QAT foreground launch - Codex findings

UTC: 2026-06-01T22:45:00Z

## Scope

Recovered the next compact PACT-NeRV-VQ score-lowering action after the
completed 600-pair baseline and int2 section-value replay. All rows here are
`[macOS-MLX research-signal]` and false-authority until receiver proof,
full-video MLX replay, section-value profile, and exact CPU/CUDA gating close.

## Prior Completed Artifact

- Baseline archive:
  `/Volumes/VertigoDataTier/pact/compact_pact_vq_ch48_score_bound_full600_2000ep_codex_20260601T194633Z/pact_nerv_vq_mlx_training/archive.zip`
- Baseline archive bytes: 192,810
- Best post-hoc codec: `int2_scale_bundled`
- Best post-hoc codec bytes: 54,930
- Best post-hoc codec SHA-256:
  `bebcadb846182fbc666b3151202fe46e8e15a8a991c6bbbb7085cc65dacc66d7`
- Full-video MLX advisory for the int2 artifact: `canonical_score` 90.20324809883085.

Section-value replay protected `decoder_qw` and `selectors_rc`, demoted
`codebooks_q` for that artifact, and left residuals absent/demoted unless a
future candidate satisfies `delta_nonrate + rate_cost < 0`.

## Launch Failure Classification

Three detached `nohup` launches exited silently with empty logs and no durable
output. A foreground one-epoch 600-pair startup probe then succeeded with the
same runner/runtime surface:

- Probe output:
  `/Volumes/VertigoDataTier/pact/compact_pact_vq_qat4_startup_probe_codex_20260601T223555Z`
- Probe archive bytes: 38,149
- Probe archive SHA-256:
  `1ec59304cb565f4634fbd79a2645ae4b43bc36bd702e6c7b3b02bb784578a1b2`
- Probe receiver proof: passed.
- Probe blockers:
  `contest_cpu_cuda_exact_eval_not_executed`,
  `some_sections_missing_value_per_byte_measurement`.

Verdict: the detached failure is a launch-method/process-lifetime issue, not a
current runner, scorer-upstream, receiver-proof, or QAT runtime blocker.

## Active Long Run

Foreground-managed long run:

- Claim lane:
  `lane_compact_pact_vq_qat4_int2_full600_foreground_mlx_20260601`
- Job:
  `compact_pact_vq_qat4_int2_full600_2000ep_foreground_20260601T224126Z`
- Output:
  `/Volumes/VertigoDataTier/pact/compact_pact_vq_ch48_qat4_int2_full600_2000ep_foreground_codex_20260601T224126Z`
- Storage plan:
  `/Volumes/VertigoDataTier/pact/compact_pact_vq_ch48_qat4_int2_full600_2000ep_foreground_codex_20260601T224126Z_storage_plan.json`
- Foreground log:
  `/Volumes/VertigoDataTier/pact/compact_pact_vq_ch48_qat4_int2_full600_2000ep_foreground_codex_20260601T224126Z.foreground.log`

Configuration highlights:

- `--num-pairs 600`
- `--epochs 2000`
- `--batch-pairs 128`
- `--compact-decoder-channel 48`
- `--compact-decoder-codec int2_scale_bundled`
- `--segnet-distillation-objective boundary_argmax_hinge`
- `--segnet-distillation-weight 0.05`
- `--pose-distillation-weight 0.0005`
- `--coder-aware-qat`
- `--coder-qat-quant-bits 4`
- `--coder-qat-quant-residual-weight 0.001`
- `--coder-qat-magnitude-weight 0.0001`
- `--coder-qat-delta-weight 0.0002`
- `--upstream-dir /Users/adpena/Projects/pact/upstream`

Early telemetry confirmed the run entered real training under foreground
custody. Promotion remains blocked until the finished archive is harvested,
receiver-proven, codec-swept if needed, replayed through full-video MLX
section-value profiling, and exact-gated only if local evidence becomes
frontier-plausible.
