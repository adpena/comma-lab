# Codex Findings: Optimal Synergy Stack For Score Lowering

UTC: 2026-06-01T16:34:00Z

## Claim

The most important part of the fully optimized stack is the score-priced
allocation layer around a tiny byte-closed learned receiver. The architecture
family matters, but only after every charged bit is priced against the real
contest action:

`S = 100*d_seg + sqrt(10*d_pose) + 25*archive_zip_bytes/source_video_bytes`

The upstream evaluator fixes the byte water level:

`lambda_byte = 25 / 37,545,489 = 6.658589531221714e-7 score/byte`

So the optimal rule is not subjective quality, MSE, PSNR, or generic
compression. Spend a byte only if its measured full-video P18/P19 non-rate
improvement is greater than this price. Everything else is dead-zoned,
quantized away, implicit, or never emitted.

## Why This Is The Center

PR95-style HNeRV worked because the carrier grammar is tiny: a compact learned
program plus tiny latents. Z8/HPRC became rate-bound because explicit signal
fields, especially wavelet/detail/residual fields, can easily dominate the
archive even when distortion is excellent. The winning stack must therefore
start from the smallest high-signal program and only add residual/token bytes
when the contest scorer says they pay.

The core object is:

`argmin_theta,latents,tokens,codec S(theta, latents, tokens, archive_bytes)`

subject to:

- archive bytes are fully charged inside `archive.zip`;
- `inflate.sh` is deterministic decode-only;
- no hidden scorer state, no sidecars, no eval-time adaptation;
- all MLX/proxy gradients are proposal-only until receiver proof and exact
  CPU/CUDA eval gate.

## Optimal Stack

### Layer 0: Authority Contract

Use the pinned upstream evaluator contract as the root:

- source custody: evaluator, frame utils, scorer modules, scorer weights,
  workflow axes;
- real rate denominator: source video bytes, not inflated raw bytes;
- SegNet: last frame of each pair only, 5-class argmax flip at `384x512`;
- PoseNet: both frames, YUV6 pair tensor, first-six pose MSE;
- exact byte price as the allocator water level.

### Layer 1: Primary Carrier

Primary carrier should be PR95/HNeRV-scale learned receiver or a successor with
the same tiny-packet discipline:

- HNeRV/RNeRV/PACT-NeRV base renderer;
- hard ceilings around `178k`, `216k`, `285k`;
- weights, latents, selectors, config, and runtime all inside `archive.zip`;
- no raw wavelet/detail float fields as primary representation.

This matches the research direction where HNeRV adds content-adaptive embeddings
and redesigned blocks for faster, better video regression, while HiNeRV improves
INR video coding with hierarchical encodings and a refined
training/pruning/quantization pipeline.

### Layer 2: Hierarchical/Token Grammar

Use hierarchy and tokenization only where bytes pay:

- Tree/Hi/SR modes for nonuniform temporal and spatial allocation;
- RT/VQ-NeRV-style residual tokenization for fine detail when continuous
  residuals are too expensive;
- section-wise selector streams only if their value-per-byte beats lambda;
- C3/Cool-Chic-style multiresolution latent grids as a competing section
  family when codebook/table/model overhead amortizes.

The local rule stays stricter than the papers: every token is contest-priced.
No residual sidecar because it improves PSNR; only residual sidecars with
negative `delta_nonrate + rate_cost`.

### Layer 3: Full-Video P18/P19 Allocator

The allocator must keep SegNet and PoseNet separate until incidence projection:

- `s_seg`: last-frame-only argmax-boundary/DeepFool/Crammer-Singer flip-risk
  on five-class logits;
- `s_pose`: both-frame PoseNet pixel/pair Fisher or finite-difference null
  response, weighted by `5/sqrt(10*d_pose)`;
- pair latent incidence projects both fields onto weights, latents, tokens,
  selectors, wavelet bands, regions, boundaries, frames, pairs, batches, and
  full-video atoms;
- KKT/Dykstra/water-fill chooses bytes only above the fixed contest byte price.

Full-video exact reduction is the authority lane. Chunking is allowed only as
an exact deterministic reduction, not stochastic promotion.

### Layer 4: Residual Knowledge, Not Residual Bloat

Z8/HPC is valuable as a scorer-aware analyzer and tiny-token generator, not as
the main packet:

- use Z8 wavelet/detail coefficients to learn where the compact base fails;
- map P18/P19 surfaces to candidate residual tokens;
- code residuals with significance trees, bitplanes, RLE/range/ANS, VQ, or
  learned priors;
- admit residual sections only when exact replay says they pay.

Raw-float detail storage is disallowed as a winning grammar.

### Layer 5: Learned Priors

Mamba/Dreamer are not magic sidecars. They are useful if they reduce charged
bytes:

- Mamba: temporal/coefficient-token entropy prior or direct-transform latent
  predictor, with all model/state bytes charged;
- Dreamer/RSSM: categorical mode allocator over zero/quant/VQ/protected
  states, charged as archive bytes;
- procedural driving prior: only as charged deterministic decoder code and
  constants, replacing latents where road/ego geometry is predictable.

If model bytes exceed entropy savings, demote automatically.

### Layer 6: Section Entropy Compiler

Every section uses the cheapest bit-exact coder for its distribution:

- decoder weights: tensor grouping, byte maps, split Brotli/range only if
  measured gap remains;
- latents/tokens/selectors: RLE/range/ANS/byteplane/VQ chosen per section;
- sparse residuals: significance order plus occupancy/value separation;
- container: single stored `0.bin` ZIP member unless runtime needs otherwise.

On PR95/fec6-like packets this layer is nearly saturated; on developing
substrates it is a diagnostic that prevents raw entropy leaks.

### Layer 7: Runner And Learning Loop

One bounded runner should own:

1. select contract-backed work;
2. train/export compact base under hard byte ceiling;
3. compute full-video MLX/PyTorch advisory P18/P19 surfaces;
4. materialize byte-closed packet;
5. prove receiver consumption through `inflate.sh`;
6. run local full-video replay if available;
7. exact-gate only plausible candidates;
8. posterior-update every positive, negative, failure, and blocker;
9. demote sections/families with durable negative evidence.

## The Design To Pursue First

The immediate highest-EV stack is:

`PR95/HNeRV-scale base -> Hi/RT tokenized latent grammar -> full-video P18/P19 allocator -> tiny VQ/HPRC/Z8 residual tokens only if priced -> section entropy compiler -> receiver proof -> exact gate`

The first killer experiment is not a broad sweep. It is a hard-ceiling compact
base continuation under the real scorer objective, with section neutralization
and residual admission governed by:

`delta_nonrate + 25*delta_bytes/37,545,489 < 0`

If the base cannot converge enough under the byte ceiling, pivot within the same
packet spine to HiNeRV/RT-NeRV/C3/Cool-Chic-style latents. Do not pivot to a
byte-heavy explicit residual codec unless the byte-value profile proves it.

## Research Anchors

- HNeRV: content-adaptive embeddings and redesigned HNeRV blocks improve video
  regression quality and convergence speed.
  https://arxiv.org/abs/2304.02633
- HiNeRV: hierarchical encodings plus training/pruning/quantization improve INR
  video compression rate-quality.
  https://papers.neurips.cc/paper_files/paper/2023/hash/e5dc475c370ff42f2f96dddf8191a40c-Abstract-Conference.html
- RT-NeRV: residual tokenization addresses the cost of continuous residual
  support in hybrid NeRV.
  https://arxiv.org/abs/2403.12401
- Cool-chic video: ultra-low-parameter overfitted neural video codec with
  temporal inter coding.
  https://arxiv.org/abs/2402.03179
- C3/Cool-Chic lineage: multiresolution latent grids, synthesis transform, and
  learned entropy model optimized per image/video.
  https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_C3_High-Performance_and_Low-Complexity_Neural_Compression_from_a_Single_Image_CVPR_2024_paper.pdf
- Geometric Transformation-Embedded Mamba: direct transform, quantization,
  entropy coding, and temporal priors via cascaded Mamba.
  https://arxiv.org/abs/2603.07912

## Non-Negotiable Failure Modes

- No MLX/proxy score authority.
- No scorer in receiver.
- No sidecars.
- No stochastic minibatch promotion.
- No residual/token bytes without measured value-per-byte.
- No conflation of source video bytes and inflated raw bytes.
- No one-off tools that do not feed the packet spine, allocator, posterior, or
  bounded runner.
