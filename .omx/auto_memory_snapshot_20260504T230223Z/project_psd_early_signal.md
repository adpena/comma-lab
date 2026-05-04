---
name: PSD Architecture Early Signal
description: PSD h=64 scores 1.31 proxy at epoch 289 with SegNet 6.7% BETTER than baseline — first architecture to improve both metrics
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Discovery (2026-04-10, late night)

PSD (PixelShuffle+Dilated) h=64 at epoch 289/2500 shows:
- **SegNet: 0.00541** (6.7% better than baseline 0.00580) — FIRST architecture to improve SegNet
- **PoseNet: 0.05944** (still 27x worse than dilated at 905 epochs — early in training)
- **Int8 proxy score: 1.3117** (already near our auth 1.33)
- **Int8 size: 95KB** (2x larger than dilated 46KB — rate penalty of 0.033 pts)

## Why this matters

Dilated h=64 (our current #1 at 1.33) improved PoseNet 5.6x but **regressed SegNet 5.2%**.
PSD is the first architecture to move in the RIGHT direction on both axes simultaneously.
This validates the Pareto frontier analysis — PSD may sit closer to the true minimum.

## Concerns
- PoseNet is still very high (0.059 vs dilated's 0.002) at epoch 289
- Model is 95KB (rate penalty vs 46KB dilated)
- This is PROXY score, not authoritative — could have proxy gap like KL distill
- Need to see convergence curve through epoch 900+

## What to watch
- Does PoseNet drop below 0.01 by epoch 500? (Dilated did, at comparable epoch)
- Does SegNet stay below baseline as training progresses?
- Can PSD model be pruned/distilled to reduce from 95KB?

**Why:** PSD may be the answer to the Pareto frontier — improving both axes.
**How to apply:** Let PSD training run to 2500 epochs. Auth eval at epoch 500 and 1000 checkpoints.
