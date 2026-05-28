# Codex Findings: Diffusion Blocks For PACT-NeRV And MLX Long Training

- generated_at_utc: 2026-05-28T03:08:00Z
- source: xhigh sidecar literature pass on https://arxiv.org/abs/2506.14202
- scope: PACT-NeRV, PR95/HNeRV MLX reproduction, local MLX long training, numpy-portable substrate runtimes
- authority: research and implementation planning only; no score claim; no promotion authority

## Core Signal

The paper reframes deep residual networks as discretized reverse-diffusion
dynamics and trains residual stacks block-wise. The useful PACT translation is
not generic image diffusion; it is deterministic blockwise denoising/refinement
for video substrates, where each block owns a fixed sigma/difficulty band and
exports as a deterministic MLX/numpy-compatible runtime component.

The high-value idea is memory-local training. Only one block's activations,
gradients, parameters, and optimizer state need to be live during a block update.
That can unlock larger PACT-NeRV/HNeRV/NeRV-family settings on the local M-series
GPU without cloud spend, while preserving a byte-closed deterministic inflate
path.

## Project Mapping

Direct wiring targets:

- `experiments/train_substrate_pact_nerv_diffusion_trajectory.py`
- `experiments/train_substrate_pact_nerv_diffusion_distilled.py`
- `tools/run_pr95_mlx_long_training.py`
- `src/tac/local_acceleration/pr95_hnerv_mlx_long_training.py`
- `src/tac/substrates/_shared/mlx_score_aware_full_main.py`
- `src/tac/substrates/_shared/numpy_portable_inflate.py`

Reusable primitives to canonicalize:

- `DiffusionBlockSchedule`
- `BlockDenoiser`
- `NoiseConditioner`
- `EulerIntegrator`
- `BlockManifest`
- `PortableBlockRuntime`

The primitive op set should remain tinygrad-like and portable: linear, conv2d,
depthwise conv, simple norm, FiLM/AdaLN conditioning, SiLU/GELU, fixed resize,
pixel shuffle, and only byte-justified small attention.

## Design Direction

`PACT-NeRV-DB`: split the existing PACT-NeRV/NeRV decoder trajectory into
`B=2/3/4` deterministic denoising blocks. Each block receives frame/pair/region
conditioning, a fixed sigma or difficulty coordinate, and a noisy/residual state,
then predicts a clean frame, residual, or feature-space refinement. Inference is
a fixed Euler schedule. Random sampling is forbidden for contest inflate.

`ScorerDifficultySchedule`: replace generic log-normal sigma bands with equal
cumulative difficulty mass computed from MLX scorer response, master-gradient
anchors, P18 SegNet waterfill rows, P19 PoseNet-null subsets, and P11 selector
cells. This is the bridge from the paper's block partitioning to contest-specific
inverse steganalysis.

`PR95BlockwiseMLX`: use blockwise training as a memory-reduction control arm for
PR95/HNeRV reproduction. The first question is whether local MLX can train larger
hidden/channel settings or longer frame subsets than the current end-to-end path.

## Queueable Experiments

1. `pact_nerv_diffusion_blocks_mlx_smoke`: `B=2/3`, 8-32 pair subset, local MLX,
   compare memory peak, seconds/step, RGB/YUV6 loss, scorer proxy deltas, and
   archive byte tax against the current PACT-NeRV baseline.
2. `pr95_mlx_blockwise_smoke`: source-video frames, one block trained at a time,
   measure whether memory savings unlock larger base channels or latent width.
3. `diffusion_block_schedule_ablation`: uniform sigma vs paper-style
   equal-probability sigma bands vs scorer-difficulty bands.
4. `blockwise_numpy_portability_smoke`: export a tiny MLX block checkpoint,
   replay deterministic Euler inference in numpy, and record tolerance/hash
   contract.
5. `block_archive_byte_tax_audit`: charge block headers, schedule constants,
   noise-conditioning metadata, duplicated modules, and runtime LOC before any
   exact-eval promotion.

## Risks

- HNeRV upsampling stacks violate the same input/output dimension assumption of
  simple residual diffusion blocks. Use feature-space projectors or explicit
  upsample-stage blocks.
- Blockwise training reduces memory, not archive bytes by itself. It is useful
  only if it enables a better trained, smaller, or more portable substrate.
- PoseNet-local signals remain advisory. Start with RGB/YUV6 and SegNet-sensitive
  region objectives before trusting PoseNet-null claims.
- Extra conditioning and block manifests can erase byte gains; byte-tax audit is
  mandatory.

## Next Build Step

Create a fail-closed queue builder for `pact_nerv_diffusion_blocks_mlx_smoke`
that emits only local research artifacts: block schedule manifest, MLX timing
smoke, memory peak, numpy export parity, byte-tax estimate, and explicit blockers
for score/promotion/exact dispatch.
