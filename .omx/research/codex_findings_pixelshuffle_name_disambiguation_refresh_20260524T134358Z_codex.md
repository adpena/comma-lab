# Codex Findings: PixelShuffle Name Disambiguation Refresh

UTC: 2026-05-24T13:43:58Z
Evidence grade: `[repo-forensic/live-github-advisory]`

## Question

Was there a submission called `pixelshuffle`, `pixel shuffle`, `PixelShuffle`,
or a close variant, and is it related to current HNeRV/NeRV/BoostNeRV work?

## Finding

No current public PR title/head in `commaai/comma_video_compression_challenge`
matched `pixelshuffle`, `pixel shuffle`, or `PixelShuffle` in the live GitHub
checks from this workspace.

The repo contains two distinct meanings that can be easy to conflate:

1. Historical internal post-filter lanes:
   - `pixelshuffle_h64_long1000`
   - `psd_h64_long1000`
   - `PixelShuffle+Dilated` / `PSD`

2. A generic neural decoder primitive:
   - `nn.PixelShuffle(2)` is used by many NeRV/HNeRV-style public and internal
     decoders as an upsampling block.
   - This includes HNeRV-family submissions such as PR95 and later variants,
     plus other public submissions whose runtime source contains `PixelShuffle`.

## Evidence

- Prior memo:
  `.omx/research/codex_findings_pixelshuffle_name_disambiguation_20260524T090206Z_codex.md`.
- `docs/speculative_lanes.md` records `PixelShuffle+Dilated hybrid` launched as
  `psd_h64_long1000`.
- `docs/speculative_lanes.md` records `PixelShuffle half-resolution processing`
  launched as `pixelshuffle_h64_long1000`.
- `reports/raw/2026-04-09-pixelshuffle-h64-best/pixelshuffle_h64_long1000_proxy_summary.json`
  records a historical CPU proxy result for the internal pixelshuffle lane:
  `current_workflow_score=1.99`, `pose_distortion=0.0728246`,
  `seg_distortion=0.0056208`, `current_workflow_rate=0.02301653`,
  `current_workflow_archive_bytes=864167`.
- `reports/raw/2026-04-09-psd-h64-best/psd_h64_long1000_proxy_summary.json`
  records a historical CPU proxy result for PSD:
  `current_workflow_score=1.85`, `pose_distortion=0.05271273`,
  `seg_distortion=0.00551752`, `current_workflow_rate=0.02301653`,
  `current_workflow_archive_bytes=864167`.
- `src/comma_lab/task_codec/architectures.py` still exposes
  `variant="pixelshuffle"` with the legacy
  `experiments/train_postfilter_pixelshuffle_dilated.py` entrypoint.
- `submissions/robust_current/inflate_postfilter.py` includes
  `PixelShuffleDilatedPostFilter`, maps `pixelshuffle_dilated` and `psd` to
  that class, and maps `pixelshuffle` to `PixelShufflePostFilter`.
- Live GitHub code search found `PixelShuffle` in public submission source
  paths such as `submissions/hnerv_muon/src/model.py`,
  `submissions/neural_inflate/inflate.py`,
  `submissions/jas0xf_adversarial_neural_representation/inflate.py`,
  `submissions/kitchen_sink/src/model.py`,
  `submissions/belt_and_suspenders/src/model.py`, and
  `submissions/hnerv_lc_v2_scale095_rplus1/hnerv_model.py`.

## Interpretation

`pixelshuffle_h64_long1000` and `psd_h64_long1000` were older internal
post-filter experiments in the `robust_current` stack. They are related to
SegNet/PoseNet response exploration, but they are not PR95, PR106, BoostNeRV,
or the current byte-closed HNeRV/NeRV renderer family.

The current useful relationship is lower-level: PixelShuffle is part of the
decoder substrate used by many NeRV/HNeRV-family candidates. For the MLX,
Metal, Accelerate, Rust, and queue/DAG optimization tranche, it should be
modeled as a primitive/fusion/layout axis: conv-to-PixelShuffle lowering,
polyphase channel layout, parity checks, and fused kernels. It should not be
treated as a standalone current promotion lane without new exact or
component-response evidence.

## False-Authority Boundary

This is a name-disambiguation and routing memo only. It is not a score claim,
promotion claim, rank/kill claim, or exact-eval dispatch recommendation.
Historical CPU proxy rows are not interchangeable with current contest-CPU or
contest-CUDA auth evidence.

## Queue / DAG Follow-Up

- Add or reuse a `pixelshuffle_primitive` dimension only for substrate/kernel
  acceleration and architecture-family tagging.
- Do not resurrect `pixelshuffle_h64_long1000` as an active lane unless a new
  exact-axis artifact or calibrated component-response result changes EV.
- For HNeRV/BoostNeRV/NeRV-family local MLX work, include PixelShuffle lowering
  in the native-hotspot profile list beside decoder convolution, activation,
  quant/dequant, and archive parse/inflate overhead.
