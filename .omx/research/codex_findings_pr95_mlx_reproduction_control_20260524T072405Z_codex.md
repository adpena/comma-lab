# Codex Findings: PR95 MLX Reproduction Control And Local Training Queue

created_at_utc: 2026-05-24T07:24:05Z
lane_id: lane_pr95_mlx_reproduction_control_20260524
research_only: false
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false

## Verdict

PR95 should be treated as the source-faithful control arm for the local
substrate-training program, not as a one-off HNeRV clone. The durable goal is:

1. reproduce PR95's training recipe and byte grammar exactly enough to anchor a
   control;
2. port differentiable compute to MLX/Metal while keeping NumPy as the portable
   archive/export boundary;
3. compare local MLX scorer/training signals against local CPU and exact
   contest CPU/CUDA anchors without collapsing evidence axes;
4. reuse the same queue contract for HNeRV variants, BoostNeRV bolt-ons,
   broader NeRV-family models, and non-NeRV representations.

MPS is not a decision substrate for this lane. Existing MPS timing is historical
control noise only; the actionable timing question is direct MLX/Metal.

## Public Source Map

- PR: https://github.com/commaai/comma_video_compression_challenge/pull/95
- Title: `hnerv_muon submission (0.20)`
- Author: `AaronLeslie138`
- Head SHA: `9bdce26f2a4f996828c4e3fa2b87c454a0e8fcc9`
- Merge commit: `fa67764d323f13a9be41906e96426cc556d85459`
- Public archive URL: https://github.com/user-attachments/files/27332334/archive.zip
- Blog/writeup URL: https://aaronleslie.dev/blog/comma-compression

Public PR body reports CPU/GitHub Action evaluation:

- PoseNet: `0.00003494`
- SegNet: `0.00061212`
- bytes: `178,417`
- score: displayed `0.20`, described as `0.1987`

Public CUDA/T4 action comment reports:

- PoseNet: `0.00017185`
- SegNet: `0.00070728`
- bytes: `178,417`
- score: displayed `0.23`

This is a useful PR95-specific example of CPU/CUDA drift. It does not imply a
universal ordering across submissions.

## Local Custody

- Source root:
  `experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon`
- Archive:
  `experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/archive.zip`
- Archive bytes: `178,417`
- Archive SHA-256:
  `e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a`
- Single stored ZIP member: `0.bin`

Existing local integration surfaces:

- `tools/run_pr95_local_training_probe.py`
- `src/tac/optimization/pr95_muon_local_training_integration.py`
- `src/tac/optimization/representation_training_probe_integration.py`
- `src/tac/optimization/local_training_runtime_profile.py`

This pass adds:

- `src/comma_lab/scheduler/local_training_queue.py`
- `tools/build_local_training_execution_queue.py`

The PR95 plan-only path now emits `recommended_execution`, and the generic
representation-training sidecar carries that execution block into the queueable
surface.

## Recipe

Architecture:

- HNeRV-style decoder
- 28-dimensional latent per frame pair, roughly 600 pairs
- 229K parameters
- evaluation/training resolution: `384x512`
- inflate upsamples to camera resolution `874x1164`
- archive grammar: stored ZIP `0.bin`; length-prefixed Brotli streams for
  metadata, int8 decoder state, and delta-coded uint8 latents

Training curriculum:

| Stage | Module | Epochs | Key optimizer/loss notes |
| --- | --- | ---: | --- |
| 1 | `stage1_v328_ce` | 3000 | AdamW, CE segmentation warmup |
| 2 | `stage2_v331_softplus` | 5650 | AdamW, tau-Softplus margin |
| 3 | `stage3_v332_smooth` | 1500 | AdamW, smooth disagreement |
| 4 | `stage4_v332_qat` | 500 | AdamW, quantization-aware training |
| 5 | `stage5_c1a_l7` | 9000 | AdamW, L7 hard-pixel weighting, C1a entropy |
| 6 | `stage6_lambda_sweep` | 2000 | C1a lambda sweep |
| 7 | `stage7_sigma_sweep` | 3000 | C1a sigma sweep |
| 8 | `stage8_muon_finetune` | 5000 | Muon hidden weights + AdamW rest |

Total epochs: `29,650`.

The public README states about 50 hours on one unspecified GPU from random init.
That implies an average of about `6.1 s/epoch` across the full source recipe.
Direct MLX timing is not yet measured; a correct estimate must smoke at least
Stage 1, Stage 5, and Stage 8 because QAT, C1a sampling, resize kernels, and
Muon Newton-Schulz alter the kernel mix.

Planning formula:

```text
T_total =
  T_target_precompute
  + sum_stage(
      stage_epochs * seconds_per_train_epoch
      + ceil(stage_epochs / eval_every) * (seconds_per_eval + seconds_per_archive_probe)
    )
  + T_export_zip
  + T_auth_eval_anchor
```

MLX answer today: unknown until direct timing smoke. The control expectation is
`~50h` if MLX matches the author's unspecified single GPU average; better or
worse is determined by measured Stage 1/5/8 MLX probes, not by MPS.

## Port Design

Keep in Python/NumPy:

- archive parse/build grammar;
- int8 quant/dequant metadata;
- latent min/scale and temporal delta coding;
- Brotli and ZIP emission;
- SHA/custody manifests;
- export/import interchange.

Move to MLX/Metal:

- HNeRV forward/backward;
- QAT straight-through estimator;
- C1a entropy regularizer;
- Muon parameter partition and update;
- resize kernels only after explicit PyTorch parity probes.

Scorer boundary:

- First MLX port target is training/rendering, not score authority.
- Use local MLX scorer-response only as research signal.
- Use local CPU spot checks for calibration.
- Use exact contest CPU/CUDA only for score/rank/promotion.

## Queue Wiring Landed

`local_training_execution_queue_plan.v1` now compiles local representation
training plans into `experiment_queue.v1` with:

- backend-aware resource mapping (`local_mlx`, `local_cpu`, `local_cuda`,
  `local_mps`, or `local`);
- output manifest and representation sidecar postconditions;
- JSON false-authority postconditions;
- experiment metadata carrying representation/substrate family, backend, and
  device;
- queue controls for independent CPU/MLX/CUDA/MPS concurrency.

This is the missing automation layer for loading many local training probes and
letting the queue run them under custody. It is deliberately generic; PR95 is
one source-faithful control, not the only target.

## Next Engineering Gates

1. Implement `pr95_mlx_backend` adapter:
   - MLX `HNeRVDecoder`;
   - MLX latents and training state;
   - NumPy state export compatible with existing PR95 codec;
   - optimizer partition matching Stage 8 Muon semantics.
2. Add parity probes:
   - PyTorch vs MLX forward output at eval resolution;
   - resize parity for bicubic upsample and bilinear scorer downsample;
   - NumPy export -> PR95 codec -> inflate output smoke.
3. Queue timing smokes:
   - Stage 1, 25-100 epochs;
   - Stage 5, 25-100 epochs;
   - Stage 8, 25-100 epochs with Muon.
4. Wire timing results into `trainer_runtime_profile_observation.v1`.
5. Build candidate queue rows from the MLX manifests, then feed byte-shaving
   operation-set planning and exact-readiness only after byte-closed export.

## Verification

Focused verification for this pass is recorded in the session output. The new
local-training queue tests cover queue compilation, MLX/CPU resource mapping,
truthy-authority rejection, CLI writing, and PR95 plan-only queueability.
