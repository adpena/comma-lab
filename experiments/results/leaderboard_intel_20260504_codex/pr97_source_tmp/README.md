# vibe-coder-final-boss

Interactive writeup of `vibe_coder_final_boss`, a submission to the **comma 2026 video compression challenge**. Final score: **0.22878**.

The site walks through the four pieces of the pipeline:

1. **Lossless arithmetic mask codec** (135 KB) forked from PR #81 with five composable encoder-side wins on top.
2. **92K-parameter FP4-quantized U-Net generator** found by 195 short-budget proxy experiments under a Karpathy-style autoresearch loop.
3. **2.4 KB per-pair sidecar** of mask flips, pose deltas, and frame-1 warps that invert the SegNet/PoseNet eval discriminators.
4. **Per-byte engineering** (flat-FP4 model packing, per-dim pose quantization, single-file zip).

## Stack

- Vite 5 + React 19 + TypeScript
- Tailwind CSS 3
- Hand-rolled SVG visualizations, no chart libs

## Develop

```bash
npm install
npm run dev      # localhost:5179
npm run build    # → dist/
npm run preview  # serve dist/
```

## Deploy (Cloudflare Pages)

Connect this repo to Cloudflare Pages with these build settings:

- Build command: `npm run build`
- Build output directory: `dist`
- Node version: 20.19+

## Layout

```
src/
  components/      Hero, Section, Prose, Abstract, Postscript, References
  interactive/     ScoreCalculator, CascadeAnimator, TileLayoutInteractive,
                   CodecTimeline, SVDRankSlider, SidecarPipeline, DropPruning,
                   ScoreJourney, ArchiveComposition, Architecture
  App.tsx          One file wires every section together.
public/
  writeup_assets/  Static images + the rendered mask used by tile layout viz.
```

## Submission

The actual submission code lives in the parent challenge repo at
`submissions/vibe_coder_final_boss/`. This repo is just the writeup.
