# Codex Findings: Z8 Wavelet Quantization Research

UTC: 2026-05-31T22:18:39Z
Scope: read-only online research plus local artifact/code inspection
Authority: research memo only; no score claim; no promotion authority

## Local Anchor

Z8HPC1 is a deterministic monolithic `0.bin` grammar with `wavelet_blob` as the
frame-affecting payload and `wyner_ziv_blob` as the second proven
frame-affecting surface. `decoder_blob`, `indices_blob`, and
`dreamer_state_blob` remain archive-bound custody/provenance rather than
pixel-consuming authority.

The current profile is overwhelmingly a wavelet detail-byte problem:

- contest rate targets: 0.10 rate ~= 150,181 bytes; 0.20 ~= 300,363 bytes;
  1.00 ~= 1,501,819 bytes.
- best custody-valid Z8 ZIP in this profile: `q0156`, 23,376,927 bytes,
  rate term 15.5657, `wavelet_blob` 99.87% of inner `0.bin`.
- smallest observed inner packet: `quantized_detail_probe`, 10,195,155 bytes,
  rate term 6.7885, `qi16_zero_rle`, but no ZIP/receiver custody in this
  profile.
- existing code already has per-subband quantization/entropy portfolio,
  `constriction` range coding, byteplane/RLE modes, RD-waterfill, full-video
  P18/P19 authority gates, and coefficient materialization.

Implication: generic entropy-mode polishing is now second-order. Competitive Z8
needs to remove, predict, or synthesize most detail coefficients, then spend
bits through full-video SegNet/PoseNet surfaces.

## Source Findings

- EBCOT/JPEG 2000 is the closest classical template. Taubman's EBCOT partitions
  each subband into independent code-blocks, produces embedded bitstreams, and
  chooses optimized truncation points by rate-distortion slope. JPEG says JPEG
  2000 supports ROI access and non-iterative optimal rate control; HTJ2K keeps
  code-blocks <=4096 samples, quantizes/codes blocks independently, and uses
  PCRD-opt to discard coding passes to meet rate/distortion targets.
  Sources: https://www.ee.nthu.edu.tw/~cwlin/courses/multimedia/notes/EBCOT.pdf,
  https://jpeg.org/jpeg2000/index.html,
  https://ds.jpeg.org/whitepapers/jpeg-htj2k-whitepaper.pdf

- SPIHT/EZW/SPECK are useful because they exploit cross-scale significance
  trees over wavelet coefficients. This maps to Z8's detail hierarchy, but
  importing old SPIHT code is a licensing/patent footgun; use the idea, not the
  code. Source: https://www.spiht.com/

- Learned wavelet codecs now combine DWT/lifting transforms with learned
  entropy models over inter/intra-subband dependencies, explicitly analogous to
  EZW/SPIHT/EBCOT, and report gains beyond JPEG2000 when learned entropy models
  are paired with traditional CDF filters. Source:
  https://link.springer.com/article/10.1007/s00530-023-01192-w

- Learned image compression's durable lesson is not "use a giant autoencoder";
  it is hyperprior/context modeling plus exact entropy tables. Ballé/Minnen
  style models use quantized latents plus priors for arithmetic-coded
  bitstreams; TensorFlow Compression warns that compression/decompression need
  identical range-coding tables. Sources:
  https://research.google/pubs/variational-image-compression-with-a-scale-hyperprior/,
  https://arxiv.org/abs/1809.02736,
  https://www.tensorflow.org/tutorials/generative/data_compression,
  https://interdigitalinc.github.io/CompressAI/entropy_models.html

- Task-oriented/video-coding-for-machines work supports the contest-specific
  posture: optimize bytes for downstream machine reliability, not human PSNR.
  Region/semantic bit allocation is valid only when reduced into the same
  full-video coefficient surface that protects SegNet/PoseNet. Source:
  https://link.springer.com/article/10.1186/s13640-025-00682-3

- Neural video/INR/sequence models are relevant as context predictors and base
  representations, not as advisory hidden state. DCVC-style work gets gains
  from temporal/spatial contexts and entropy models; HNeRV suggests an INR base
  can carry semantic/low-frequency video content; Mamba is attractive for
  linear-time coefficient-context prediction. Sources:
  https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Neural_Video_Compression_With_Diverse_Contexts_CVPR_2023_paper.pdf,
  https://arxiv.org/abs/2304.02633,
  https://arxiv.org/abs/2312.00752

- Native tooling worth using: keep `constriction` for Python/Rust range/ANS
  parity; use PyO3/maturin only after the block coder's semantics are stable;
  use `bitstream-io` for exact bit packing if custom bitplane packets land.
  Use OpenJPEG/Grok/HTJ2K as external baselines/oracles, not as the contest
  runtime unless dependency/license/runtime custody is solved. Sources:
  https://github.com/bamler-lab/constriction,
  https://github.com/PyO3/maturin,
  https://docs.rs/bitstream-io/latest/x86_64-apple-darwin/bitstream_io/,
  https://www.openjpeg.org/?menu=doc,
  https://github.com/GrokImageCompression/grok

## Ranked Implementation Plan

1. Repair custody for `quantized_detail_probe`.
   Package the 10.2 MB inner probe into a valid ZIP, prove the ZIP contains the
   profiled `0.bin`, run receiver proof and full local replay, and preserve
   exact CPU/CUDA blockers. Acceptance: valid archive/runtime custody, receiver
   proof executed, local replay not worse than `q0156`; if ZIP <=10.5 MB it
   becomes the new Z8 rate-axis baseline.

2. Prototype `EBCOT-lite` inside Z8HPC1.
   Replace whole-subband streams with small coefficient code-blocks, dead-zone
   scalar quantization, bitplane/significance/refinement packets, and actual
   byte-measured truncation. Use full-video P18/P19 gradients for block/pass
   slopes. Acceptance: 600-pair byte-closed ZIP <=15 MB with no local replay
   regression versus `q0156`; stretch target <=8 MB before exact dispatch.

3. Prototype Z8 zero-tree/significance-tree coding.
   Build a SPIHT/SPECK-inspired significance-map codec over parent/child
   wavelet detail coefficients, with no imported legacy code. Acceptance:
   >10% ZIP reduction over the best block/portfolio coder at matched local
   SegNet/PoseNet replay; reject if tree metadata or scan order hurts boundary
   regions.

4. Add a conditional coefficient entropy model.
   Start with hand-built contexts: level, orientation, block neighborhood,
   parent LL energy, pair/frame, P18/P19 protection class. Then test a tiny
   Mamba/learned tree predictor only if model bytes are included. Acceptance:
   net model-plus-residual byte saving >=15% over the same quantized coefficient
   packet, deterministic decode, exact entropy tables bundled.

5. Build an INR/HNeRV-base plus Z8-residual hybrid.
   Use the base representation for low-frequency/semantic content and reserve
   Z8 wavelet bits for scorer-critical residuals. Acceptance: residual
   `wavelet_blob` <=100k-500k bytes after model bytes, with local SegNet/PoseNet
   within frontier-relevant range before any exact auth claim.

6. Native Rust acceleration after semantics settle.
   Move block bitplane packing, entropy coding, and decode hot loops behind a
   Rust core with Python bindings. Acceptance: byte-identical outputs to Python,
   >=5x encode speedup, decoder within contest inflate budget, no new
   dependency-custody blockers.

## Saturated / Avoid

- Do not force one entropy coder globally; local profile already shows the
  per-subband portfolio beats global range coding on mixed sparse/dense bands.
- Do not spend effort on decoder/meta/index bytes until `wavelet_blob` collapses
  by orders of magnitude.
- Do not treat MLX loss, hidden Mamba/Dreamer state, decoded tensor parity, or
  inner `0.bin` bytes as archive/runtime authority.
- Do not import full JPEG2000/SPIHT runtimes into `inflate.sh` as a shortcut
  without license, dependency, runtime-tree SHA, and exact replay custody.
- Do not let class-region or boundary prose spend bits until it is projected
  into the archive-fresh full-video coefficient surface.
