---
name: Master Council Verdict (2026-04-10)
description: Einstein/Tao/Contrarian/full panel — boundary weight 30-40x too low, PoseNet cap, safety threshold, score projection
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## The Three Structural Gaps

1. **Boundary weight 30-40x too low**: At bw=10, 88% of gradient lands on non-boundary pixels
   that CANNOT improve SegNet. At bw=150, 89% lands on boundaries. Set to 150 for next runs.

2. **PoseNet gradient not capped**: pose=0.00218 is already 33x better than competitor.
   Improving pose from 0.00218→0.001 saves only 0.048 score. Same effort on seg saves 0.21.
   FIXED: clamp pose_dist min=0.001 in kl_distill_scorer_loss.

3. **Spatial scale mismatch**: corrections at 1164x874 arrive at SegNet diluted 2.27x.
   Competitor's PixelShuffle at ~582x437 is 13% closer to SegNet's 512x384.

## Game Theory (Contrarian)

Safety threshold: seg < 0.004. If competitor fixes PoseNet to match ours, their best
possible score is 1.19. We need seg < 0.004 to stay below that. Currently at 0.00610.
Need 34% SegNet improvement.

**Recommendation**: Submit current 1.33 as insurance NOW. Continue pushing for sub-1.1.

## Score Projection

| Changes | Expected Score |
|---------|---------------|
| boundary_weight=150 | 1.20-1.25 |
| + PoseNet cap | 1.15-1.20 |
| + 3000 epochs | 1.10-1.15 |
| + error replay | 1.05-1.12 |

Conservative target: **1.05-1.15 by April 20**.

## Collier's Unifying Principle

Techniques work when they operate in the geometric/semantic space where the scorer
makes decisions. Successes (dilated, KL distill, hard-frame) all respect the scorer's
coordinates. Failures (preprocessing, Jacobian) operated in raw pixel space.

## Rubin's Essence

The filter corrects in the wrong coordinate space for SegNet. Corrections at 1164x874
get diluted 2.27x before SegNet sees them. A SegNet-resolution correction branch
(512x384) would eliminate this attenuation entirely.
