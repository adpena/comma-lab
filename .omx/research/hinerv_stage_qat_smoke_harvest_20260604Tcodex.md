# HiNeRV Stage-QAT Smoke Harvest - 2026-06-04

Axis: `[macOS-CPU/MLX local:false-authority]`. No score, rank, promotion, kill, or exact-eval dispatch authority.

## Verdict

`BYTE_ATTRACTIVE_BUT_RENDERER_DEGENERATE_NOT_A_FRONTIER_CANDIDATE`

The stage-QAT smoke produced a small archive, but it is not a candidate. The receiver cache quality gate failed with `FUNDAMENTAL_RENDERER_OUTPUT_DEGENERATE`.

## Evidence

- Output root: `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_stage_qat_runner_smoke_20260604T2111Z_codex`
- Archive: `106351` bytes, SHA-256 `85c1c44936560f87d0cf300392bdb8cba2cfe16abad47f7a9ba239960ca80890`
- Local MLX prefilter: sampled 2 pairs only, `canonical_score=94.93991807867756`
- Local terms: Seg `50.74869692325592`, Pose `44.12040638989815`, Rate `0.07081476552349604`
- Receiver cache quality: failed
- SegNet last RGB: `std=0.517579197883606`, `dynamic_range=4.9344635009765625`, `mae=101.44928741455078`
- PoseNet YUV6 pair: `std=0.43814340233802795`, `dynamic_range=3.3023681640625`, `mae=69.05482482910156`

## Decision

- Do not Modal-dispatch this archive.
- Keep the 106 KB archive size as capacity signal only.
- The next HiNeRV local run should repair renderer dynamic range and treat receiver cache quality as a stop condition before full-video replay.

## Source Artifacts

- Runner report SHA-256: `da206a672a8b315abe6309beb51a6682632768e724588a6b7a506dcbad16b091`
- Training artifact SHA-256: `643621a25eacb579637cc229fbeb6a2134261692023ab6ca50d12ae0c17f7e9a`
- Local MLX prefilter SHA-256: `1c6e89fc280fed96a3e90d4022de79f184b1a243238737dffd9c0ab46132d8cc`
- Receiver cache quality report SHA-256: `8ae41a9345a28807e7fb4708bd6e2983949d0609927db231835a0606f5fa1bb5`
