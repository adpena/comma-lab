# Codex Findings: PixelShuffle Name Disambiguation

UTC: 2026-05-24T09:02:06Z
Evidence grade: `[repo-forensic/advisory]`

## Question

Was there a submission called `pixelshuffle` or a close variant, and is it
related to the current HNeRV/NeRV-family optimization work?

## Finding

There is no current GitHub PR hit named `pixelshuffle`, `pixel shuffle`, or
`PixelShuffle` in the checked public PR search paths. In the repo, the name
refers to two separate things:

1. Historical internal post-filter lanes:
   - `pixelshuffle_h64_long1000`
   - `psd_h64_long1000`
   - `PixelShuffle+Dilated` / `PSD` / `PixelShuffle-Downscale`

2. A generic neural decoder primitive used in many NeRV/HNeRV-style submission
   runtimes:
   - PR95/PR101/PR106-derived decoders use `Conv -> PixelShuffle(2)` style
     upsampling cells, often with sin activation and bilinear skip.
   - This is architecture lineage, not a public submission literally called
     `pixelshuffle`.

## Evidence

- `docs/speculative_lanes.md` records `PixelShuffle+Dilated hybrid` as a
  high-priority historical experiment and says training launched as
  `psd_h64_long1000`.
- The same file records `PixelShuffle half-resolution processing` and says
  training launched as `pixelshuffle_h64_long1000`.
- `reports/raw/2026-04-09-pixelshuffle-h64-best/pixelshuffle_h64_long1000_proxy_summary.json`
  records a CPU proxy artifact for the `pixelshuffle_h64_long1000` lane.
- `reports/raw/2026-04-09-psd-h64-best/psd_h64_long1000_proxy_summary.json`
  records a CPU proxy artifact for the `psd_h64_long1000` lane.
- `src/comma_lab/task_codec/architectures.py` still exposes a `pixelshuffle`
  architecture spec that points at the legacy
  `experiments/train_postfilter_pixelshuffle_dilated.py` entrypoint.
- `src/tac/architectures.py` has the migrated `PixelShuffleDilatedPostFilter`
  and legacy aliases including `pixelshuffle_dilated`.
- `src/tac/quantization.py` has a loader heuristic for legacy checkpoints whose
  metadata did not correctly identify the PixelShuffle/PSD architecture.
- `submissions/*/model.py` and `submissions/*/inflate.py` contain many
  `nn.PixelShuffle(2)` decoder blocks for HNeRV/NeRV-family submissions.

## Interpretation

The useful current connection is architectural, not naming-based. The historical
`pixelshuffle_h64_long1000`/`psd_h64_long1000` lines were post-filter experiments
from the older robust-current stack. They are related to scorer-response and
SegNet/PoseNet Pareto exploration, but they are not the same object as PR95,
PR106, BoostNeRV, HNeRV, or current byte-closed renderer submissions.

The current PR95/PR101/PR106-family relevance is that PixelShuffle is the
standard upsample operator inside the decoder. This matters for MLX/Metal work:
PixelShuffle lowering/parity, polyphase layout, channel packing, and fused
conv-shuffle kernels are plausible local acceleration targets. It does not
reopen the old `pixelshuffle_h64_long1000` post-filter as a promotion lane
without fresh evidence.

## False-Authority Boundary

This memo is a name-disambiguation artifact only. It is not a score claim,
promotion claim, rank/kill claim, or exact-eval dispatch recommendation.
Historical CPU proxy artifacts are not interchangeable with current exact
contest CPU/CUDA authority.

## Actionable Follow-Up

- Treat `PixelShuffle` as a decoder primitive to optimize/lower in the
  PR95/HNeRV/NeRV-family MLX/Rust/Metal path.
- Do not spend current tranche time resurrecting the old
  `pixelshuffle_h64_long1000` post-filter lane unless a newer exact or
  component-response artifact changes its EV.
- If the optimizer matrix needs a PixelShuffle dimension, encode it as a
  primitive/fusion/layout axis, not as a standalone historical submission.
