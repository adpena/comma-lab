# Codex Findings: NeRV Source Refresh And Current Control Surface

written_at_utc: 2026-06-03T04:21:19Z
axis_status: false_authority_research_and_control_only
families: [pr95_hnerv, hi_nerv, snerv]

## Contest Scorer Contract

The upstream scorer remains the only authority. Local upstream commit checked by
the research pass is `11ad728`, and the scoring contract is:

- score = `100 * SegNet + sqrt(10 * PoseNet) + 25 * (archive_bytes / original_bytes)`
- SegNet consumes only the second frame of each two-frame pair after resize to
  512x384.
- PoseNet consumes both frames after RGB-to-YUV6 and resize to 512x384, then
  scores the first six pose dimensions.
- The upstream RGB-to-YUV6 helper is not a gradient-safe PoseNet training path;
  score-aware training must use the differentiable replacement/roundtrip
  surfaces already in this repo.

This preserves the concrete asymmetry: frame 1 buys SegNet and PoseNet leverage,
while frame 0 buys PoseNet leverage only. All rate decisions must still pay the
fixed contest byte price.

## PR95 Control Refresh

Popper's upstream pass verified PR95 as the control arm, not a generic HNeRV
baseline. Public PR95 is commaai PR #95, head
`9bdce26f2a4f996828c4e3fa2b87c454a0e8fcc9`, merged as `fa67764d...`, with
public comments reporting CUDA 0.23 and CPU 0.20. Local PR95 files under
`data/working/upstream/submissions/hnerv_muon` byte-match the PR95 head raw
files.

The actionable implementation lesson is still: source-faithful staged scorer
training plus tight codec/export discipline. PR95 uses a 28-d latent-per-pair
HNeRV decoder, eval-size 384x512, int8 quantized decoder + brotli, latent
min/scale, and delta/zigzag latent stream. The staged schedule is CE, tau
softplus, smooth disagreement, QAT, L7+C1a, lambda sweep, sigma sweep, and
final Muon. Muon is final-stage only; stages before that are expected to report
no Muon.

Current live HiNeRV telemetry at epoch 19094 is therefore not missing Muon:
`pr95_stage_index=5`, `pr95_stage_uses_muon=0`, and final stage 8 has not been
reached. The run should continue unless later telemetry reaches stage 8 without
Muon.

## HiNeRV Refresh

The HiNeRV/HNeRV/SR-NeRV/RNeRV pass found the official HiNeRV controls that
still matter for our lane:

- hierarchical temporal/local feature grids
- trilinear interpolation path
- ConvNeXt/depthwise-MLP blocks
- overlapped patch/frame equivalence
- adaptive pruning
- QuantNoise/QAT and entropy-coded bitstream controls, including official 8/7/6
  quant levels

This landing closed one concrete control gap: intermediate 6/7-bit HiNeRV
QuantNoise/waterfill actions are now executable through the shared planner,
export-side bitstream preparation, MLX train-time per-tensor fake quant, and
compact-runner validation. This is not score authority; it is a real control
surface for future scorer-valued training/export runs.

Remaining HiNeRV blockers are architecture/source parity for official feature
grid geometry and patch equivalence, score-valued decoder-weight waterfill,
trained byte-section measurement, receiver-closed full-600 archive/runtime
evidence, and exact CPU/CUDA replay.

## SNeRV Refresh

Copernicus verified that our SNeRV lane is receiver-safe and contest-useful but
not official SNeRV OSS/paper faithful for MFU/HFR/TUB. The current local lane is
a contest fork until `SNERV_OFFICIAL_MFU_HFR_TUB_PARITY_PROOF` lands.

Faithfully wired:

- official source detection and fail-closed parity audit
- official `--modelsize -> fc_dim` false-authority control surface
- real SNAR1 receiver archive/runtime grammar with metadata, LF payload,
  decoder payload, and step maps
- Haar/db1 no-pywavelets receiver path

Still forked or under-optimized:

- local defaults use multi-level `db2`/deterministic NumPy MFU/HFR rather than
  official J=1 Haar learned MFU/HFR/TUB
- temporal path is algebraic/delta/Haar-lowpass, not source-faithful SNeRV_T
  prev/current/next TUB behavior
- full-600 explicit LF storage remains rate-fatal unless representation changes

The current code path also now records invalid official SNeRV modelsize controls
instead of aborting the whole budget build. That preserves negative config
signal without losing viable candidates.

## Immediate Engineering Consequence

Do not claim SNeRV or HiNeRV is fully optimized yet. The next high-EV code work
is not another small advisory memo; it is:

1. HiNeRV: bind official architecture parity controls and use 6/7-bit QuantNoise
   in scorer-valued training/export sweeps.
2. HiNeRV: attach decoder-weight saliency/waterfill to the real long trainer and
   measure trained byte sections.
3. SNeRV: either prove official MFU/HFR/TUB behavior or mark all source-parity
   surfaces as contest-fork only.
4. SNeRV: attack LF representation with learned/scorer-preserving LF/HF
   generation or low-res/SR rather than further lossless LF prediction.
5. Cross-stack: keep every local/MLX/advisory row false-authority until
   receiver-closed full-600 archive/runtime plus exact contest-axis replay.

## Source Anchors

- Upstream contest repo: https://github.com/commaai/comma_video_compression_challenge
- PR95: https://github.com/commaai/comma_video_compression_challenge/pull/95
- HiNeRV paper: https://openreview.net/pdf?id=CpoS56pYnU
- HiNeRV repo: https://github.com/hmkx/HiNeRV
- HNeRV paper: https://arxiv.org/abs/2304.02633
- HNeRV repo: https://github.com/haochen-rye/HNeRV
- SNeRV paper: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07231.pdf
- SNeRV repo: https://github.com/qwertja/SNeRV
- SR-NeRV paper: https://arxiv.org/abs/2505.00046
- RNeRV/VINRB paper: https://arxiv.org/abs/2506.24127
- RNeRV/VINRB repo: https://github.com/mgwillia/vinrb
