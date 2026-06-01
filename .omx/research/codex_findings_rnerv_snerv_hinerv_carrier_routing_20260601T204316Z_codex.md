# Codex Findings: RNeRV / SNeRV / HiNeRV Carrier Routing

UTC: 2026-06-01T20:43:16Z
Agent: Codex
Axis: compact learned receiver / archive-rate reduction

## Verdict

RNeRV is useful here, but only as an enhancer/search prior over the winning
compact learned carrier. It should not displace SNeRV or HiNeRV as the current
top-priority carrier layer unless a byte-closed export proves lower paid
decoder+latent entropy under the shared packet spine.

SNeRV and HiNeRV are the primary carrier hypotheses:

- HiNeRV supplies hierarchical multi-scale latent structure and specialized
  pruning/quantization pressure.
- SNeRV supplies a spectra-preserving wavelet/frequency split, aligning with
  the Z8 finding that wavelet structure is valuable only when it avoids bulk
  coefficient storage.

RNeRV / Rabbit NeRV contributes component search, recurrence, and possible
charged latent generation. FFNeRV-flow is a pose-channel enhancer. BoostNeRV is
a temporal-affine/conditional-decoder bolt-on. These are second-order
synergy components, not separate score-authority carriers.

SR-NeRV is also a second-order component here, but a higher-priority one than
FFNeRV-flow or BoostNeRV. The actual upstream scorer requires contest output at
`1164x874` while both PoseNet and SegNet resize to `512x384` before scoring, and
SegNet uses only the last frame of each pair. That creates a resolution-axis
dead-zone: a compact carrier can encode internally at or below scorer
resolution, then use a charged SR/upscale step to emit legal `1164x874` frames.
This must be proven by a low-res -> SR -> contest-output ->
scorer-downsample mirror check before promotion.

## Literature Anchors

- RNeRV / VINRB: arXiv 2506.24127, "How to Design and Train Your Implicit
  Neural Representation for Video Compression". The paper frames RNeRV as a
  state-of-the-art configuration from a NeRV-family component library and a
  training-time-aware benchmark; useful here as an architecture-search prior.
  Project/code: https://mgwillia.github.io/vinrb/ and
  https://github.com/mgwillia/vinrb.
- HiNeRV: arXiv 2306.09818, "Video Compression with Hierarchical
  Encoding-based Neural Representation". The useful piece for this repo is the
  frames+patches hierarchical representation plus pruning/quantization-aware
  codec framing.
- SNeRV: arXiv 2501.01681 / ECCV 2024, "Spectra-preserving Neural
  Representation for Video". The useful piece is using DWT/frequency structure
  to reduce spectral-bias failure while keeping HF synthesis implicit rather
  than storing explicit residual fields.
- SR-NeRV: arXiv 2505.00046, "Improving Embedding Efficiency of Neural Video
  Representation via Super-Resolution". The useful piece is the low-internal-
  resolution representation plus super-resolution output principle, adapted here
  to the scorer's mandatory downsample.

## Code Landing

This memo corresponds to a code slice that:

1. Adds `snerv` as a first-class `HprcRepresentationFamily` without shifting
   existing compact-family numeric ids.
2. Adds `snerv` / `hi_nerv` to the MLX compact-runner target-family rows as
   primary carriers.
3. Keeps `rnerv`, `sr_nerv`, and `boostnerv` marked as migration-required
   enhancer/search-prior rows until MLX trainer/exporter/runtime custody exists,
   with SR-NeRV ranked as the top enhancer because it attacks a structural
   resolution-axis rate dead-zone.
4. Adds architecture-prior and allowed-enhancer metadata to target rows and
   campaign rows so acquisition can route work without prose interpretation.
5. Adds a raw-byte HiNeRV HIV1 archive projection bridge:
   `build_hi_nerv_spine_from_archive`, plus CLI support through
   `tools/build_hprc_representation_spine_projection.py --family hi_nerv`.
   This consumes existing charged HiNeRV archive bytes into decoder, latent,
   and receiver-state HPRC sections without importing a PyTorch runtime or
   granting score authority.
6. Adds regression coverage proving the SNeRV spine projection remains
   false-authority and existing SR/VQ/PVQ/RT-VQ ids are stable.

## Promotion Rule

No RNeRV/SNeRV/HiNeRV row is score authority until it emits:

- trained decoder/program bytes;
- trained latent/token/selector/codebook bytes;
- charged runtime/config bytes;
- archive.zip custody;
- receiver proof through inflate.sh;
- full-video MLX replay for section value per byte;
- exact CPU/CUDA blocker or dispatch packet.

MLX-local RNeRV/SNeRV/HiNeRV rows remain advisory. Exact contest authority
stays with byte-closed archive/runtime packets only.
