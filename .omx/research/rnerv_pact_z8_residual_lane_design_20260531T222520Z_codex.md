# RNeRV/PACT-NeRV + Z8 Residual Lane Design

UTC: 2026-05-31T22:25:20Z
Author: Codex
Authority: design memo only; no score claim; no promotion authority
Primary target: reduce contest auth eval score by collapsing archive.zip bytes
while preserving SegNet/PoseNet through receiver-proven runtime bytes.

## Executive Thesis

Z8 hierarchical predictive coding is now faithful enough to be interesting, but
the byte profile proves it is not rate-competitive in its current form. The
live Z8 packets are dominated by wavelet detail bytes: `wavelet_blob` accounts
for roughly 99.7-99.98% of inner `0.bin`. The best custody-valid profiled Z8
ZIP is `q0156` at 23,376,927 bytes, rate term 15.5657. The smaller
`quantized_detail_probe` is 10,195,155 inner bytes but still lacks ZIP/receiver
custody in the current profile.

That byte scale is incompatible with sub-0.19 frontier scores unless distortion
goes essentially to zero, which is not the present regime. Therefore the next
high-upside lane should not be another global compressor pass over the same
residual. It should replace most stored residual coefficients with a compact
per-video generator, then spend a much smaller Z8 residual budget only where
the contest scorer is sensitive.

The proposed lane is:

`RNeRV/PACT-NeRV base renderer + Z8 scorer-aware residual sidecar`

## Contest Objective

The lane optimizes the actual contest action, not PSNR alone:

`S = 100*d_seg(full_video) + sqrt(10*d_pose(full_video)) + 25*archive_zip_bytes/N`

where `N` is the contest byte normalizer. For local work, MLX and CPU replay may
rank and falsify candidates, but only exact contest CPU/CUDA replay can promote
a score claim.

## Architecture

1. Base renderer
   - Train a compact per-video RNeRV/PACT-NeRV-style INR to reconstruct the full
     contest video.
   - The model is allowed long compress-time training in contest mode.
   - All weights, latents, configs, and decoder code must be archive-bound bytes.
   - No hypernetwork teacher, checkpoint, cache, or hidden latent sidecar may be
     required at decode time.

2. Z8 residual sidecar
   - Decode the base renderer output, compute residuals against target frames,
     and represent the residual in the existing Z8 wavelet hierarchy.
   - Store only scorer-critical residual atoms: coefficients, blocks, bitplanes,
     selector modes, or repair operations that pass full-video P18/P19 gates.
   - Use Z8 as a residual codec, not as a near-lossless full-frame carrier.

3. Joint scorer allocation
   - Use full-video exact-reduced P18/P19 surfaces as allocation authority.
   - SegNet contributes dense boundary/class argmax sensitivity.
   - PoseNet contributes per-axis Mahalanobis sensitivity and null-subset
     detection so saved bytes are not spent into pose-sensitive pairs.
   - Rate is accepted only as measured archive.zip byte delta after hard
     materialization; no differentiable byte-gradient claim unless a real
     quantizer/entropy-code VJP is implemented and validated.

4. Residual entropy stack
   - Immediate baseline: existing per-subband portfolio
     (RLE/byteplane/range/ANS-compatible coding).
   - Next classical step: EBCOT-lite block bitplane truncation with
     P18/P19-weighted rate-distortion slopes.
   - Next structural step: SPIHT/SPECK-style significance trees over Z8
     cross-scale wavelet coefficients, implemented from first principles.
   - Next learned step: tiny deterministic coefficient predictor or Mamba/Dreamer
     prior whose model bytes are fully included in the archive.

5. Repair/postfilter layer
   - Deterministic SegNet/PoseNet repair is allowed only when it is cheaper than
     residual bytes and receiver-consumed.
   - It must be encoder-side materialization plus deterministic receiver decode,
     not eval-time adaptation.

6. Runtime and archive contract
   - `archive.zip` is the paid contest packet.
   - `inflate.sh` must consume the packed bytes and reconstruct outputs without
     scorer state, network access, mutable caches, or sidecars.
   - Every candidate must carry hashes, argv/config/env, source archive/runtime
     hashes, false-authority flags, cleanup disposition, and exact-axis blocker
     or dispatch plan.

## Mathematical Grounding

The optimal target is a constrained, full-video, nonsmooth variational solve:

`min_theta S_full(theta, archive_bytes, runtime)`

subject to legal hard archive projection and deterministic decode. The practical
optimization stack is:

- Full-video chunked VJP accumulation: chunking is an exact reduction strategy,
  not stochastic promotion authority.
- Bundle/trust-region handling for nonsmooth SegNet argmax boundaries.
- Implicit KKT/Dykstra differentiation for allocation layers where implemented.
- Hard quantization and archive projection decide acceptance; STE/soft quant are
  proposal mechanisms only.
- Low-rank or blockwise interaction surfaces cover coefficient, block, subband,
  frame, pair, region, boundary, and full-video atoms.

The first-order waterfill approximation is valid only after the current
full-video surface is measured. The lane should relinearize after accepted
archive mutations.

## Immediate Experiments

1. Repair `quantized_detail_probe` custody.
   Acceptance: valid ZIP contains the profiled 10.2 MB `0.bin`, receiver proof
   passes, full local replay recorded, exact CPU/CUDA blockers preserved. If it
   stays <=10.5 MB and does not regress local SegNet/PoseNet versus `q0156`, it
   becomes the Z8 rate-axis baseline.

2. Minimal RNeRV/PACT base renderer archive adapter.
   Acceptance: one adapter emits byte-closed archive rows with all model/latent
   bytes included, deterministic `inflate.sh` decode, byte profile, replay
   bundle, and no hidden state.

3. Base plus residual smoke.
   Acceptance: train a compact base on the contest video, compute Z8 residuals,
   pack residual sidecar, and report archive.zip bytes plus local scorer replay.
   The key metric is residual `wavelet_blob` collapse, not PSNR alone.

4. EBCOT-lite residual sidecar.
   Acceptance: block/bitplane streams inside Z8HPC1, P18/P19 slope-driven pass
   truncation, 600-pair byte-closed ZIP <=15 MB with no q0156 local replay
   regression; stretch <=8 MB.

5. Learned coefficient predictor.
   Acceptance: predictor-model-plus-residual bytes save at least 15% over the
   same quantized packet, deterministic decode, bundled entropy tables, and no
   receiver sidecar.

6. Production/fleet mode toggle.
   Acceptance: contest mode permits full-video overfit and long training;
   production mode switches to corpus/CVaR validation, robustness constraints,
   and no single-video-only promotion.

## Anti-Patterns To Block

- PSNR/MS-SSIM-only wins with worse SegNet/PoseNet or larger archive.zip.
- Inner `0.bin` byte wins without valid contest ZIP custody.
- MLX-positive/full-replay-negative promotion.
- Hidden NeRV latents, hypernetwork state, training checkpoint dependencies, or
  generated files outside the archive/runtime tree.
- Global entropy-coder forcing after the per-subband portfolio already beats it.
- Treating runtime code as free once residual bytes collapse. The present Z8
  Python runtime overhead is already large relative to a 0.05 rate-term budget.
- Reusing falsified Compound C or any provenance-unclean stack member.

## External Anchors

- RNeRV / VINRB: https://arxiv.org/abs/2506.24127
- Embedded / geometric-transform Mamba for learned video compression:
  https://arxiv.org/html/2603.07912v1
- HNeRV: https://arxiv.org/abs/2304.02633
- EBCOT / JPEG2000 / HTJ2K:
  https://www.ee.nthu.edu.tw/~cwlin/courses/multimedia/notes/EBCOT.pdf,
  https://jpeg.org/jpeg2000/index.html,
  https://ds.jpeg.org/whitepapers/jpeg-htj2k-whitepaper.pdf
- SPIHT: https://www.spiht.com/
- Constriction entropy coding: https://github.com/bamler-lab/constriction

## Handoff Pointer

Use this memo as the comprehensive design reference for the compact goal prompt:

`.omx/research/rnerv_pact_z8_residual_lane_design_20260531T222520Z_codex.md`
