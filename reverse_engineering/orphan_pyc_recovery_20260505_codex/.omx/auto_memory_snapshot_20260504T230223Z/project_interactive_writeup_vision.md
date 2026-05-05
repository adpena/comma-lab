---
name: Interactive writeup vision — marimo notebook + byhand.ai aesthetic
description: Clean, understated interactive writeup with real data visualizations. marimo notebook for pipeline diagram, training curves, PoseNet rank-1 viz, weight distributions, competitive analysis. Portfolio-grade.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## Aesthetic Reference
byhand.ai — clean, minimal, the work speaks for itself. No flashy design.
Real data, real visualizations, interactive exploration.

## Key Visualizations (with actual data)

1. Training trajectory: proxy 2.01 → 0.407 over 3000 epochs
   - Phase transitions marked (pixel → scorer → hard-pair)
   - SegNet and PoseNet components split

2. PoseNet rank-1 discovery: singular value bar chart
   - dim 0 = 99.8% variance, dims 1-5 nearly zero
   - Jacobian effective rank 1.008

3. Weight distribution: kurtosis 33.6 histogram
   - Signal energy by band (27% carry 90%)
   - Mixed-precision opportunity visualization

4. Architecture: animated pipeline diagram
   - mask → CLADE renderer → zoom warp → gate*residual → frame pair
   - Show data flow with actual frame thumbnails

5. Competitive landscape: score decomposition
   - Us vs Quantizr vs szabolcs: stacked bar (seg + pose + rate)
   - Interactive: hover for details

6. Forensic analysis: heatmaps on real frames
   - Boundary artifact score map
   - PoseNet sensitivity map
   - eval_roundtrip distortion map

7. Compression Pareto frontier: archive size vs quality
   - CRF sweep, FP4 vs int4+LZMA2, with/without Brotli

## Tech Stack
- marimo notebook (Python-native, interactive)
- matplotlib/plotly for charts
- Real data from experiments/results/
- Deployable to Cloudflare Pages or GitHub Pages

## When to Build
After auth scores are in and WILDE/SHIRAZ results are available.
The visualizations need final numbers to be meaningful.
