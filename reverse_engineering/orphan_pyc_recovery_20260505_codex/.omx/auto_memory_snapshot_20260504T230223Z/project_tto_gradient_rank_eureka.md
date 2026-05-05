---
name: TTO Gradient Rank Eureka — PoseNet Output MSE is Rank-6
description: FUNDAMENTAL discovery — PoseNet output MSE gives rank-6 gradient (6 pose dims). Embedding loss gives rank-256. This is THE bottleneck.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## The Discovery (Karpathy + Tao + Shannon, 2026-04-14)

PoseNet maps 589K pixels → 6 numbers. The gradient of MSE(pred[:6], target[:6]) has
Jacobian rank AT MOST 6. This means TTO can only push frames along 6 directions in
pixel space per step. SegNet has near-full-rank gradient (per-pixel logits).

SegNet gradient: ~10^-2 per pixel (strong, full-rank)
PoseNet gradient: ~10^-7 per pixel (weak, rank-6)
Ratio: SegNet dominates by 100,000x regardless of loss weights.

## Why TTO Plateaus at ~0.012

TTO exhausts all 6 PoseNet gradient directions in ~50 steps. Then early stopping kills
at step 151. No amount of lr/weight tuning can exceed rank-6 exploration.

## The Fix: Embedding Loss

Use MSE on PoseNet intermediate features (before final linear layer) instead of
output MSE. If embedding dim = 256, gradient rank = 256 — 42x more pixel-space
directions. This is `posenet_embedding_loss` in `tac.losses` (already implemented,
task #125, never used in TTO).

## Implementation

In `coupled_trajectory_optimize`, replace `compute_posenet_constraint_loss` with
a version that extracts PoseNet embeddings and computes MSE there. The GT embedding
targets need to be precomputed from GT frames (same as pose_targets but at the
embedding layer instead of the output layer).

## Expected Impact

Current TTO: 0.017 → 0.012 (30% improvement, rank-6 limited)
With embedding loss: 0.017 → potentially 0.002 (10x improvement, rank-256)
Score impact: sqrt(10 * 0.002) = 0.14 vs sqrt(10 * 0.012) = 0.35 — saves 0.21 points
Projected score: 0.22 (seg) + 0.14 (pose) + 0.10 (rate) = 0.46 — BEATS QUANTIZR

## Priority

THIS IS THE SINGLE MOST IMPORTANT CODE CHANGE FOR THE COMPETITION.
Implement embedding loss in TTO v3 IMMEDIATELY after v1 results come in.
