# Geometric Mean Scoring Analysis

## Current Scoring Formula

The comma.ai video compression challenge uses an additive composite metric:

```
score = 100 * seg_dist + sqrt(10 * pose_dist) + 25 * rate
```

This is a **weighted linear combination** with a sqrt transform on PoseNet.
Lower is better.

## Did Anyone Propose Geometric Mean Scoring?

**Short answer: No.** There is no public discussion of a geometric mean alternative
on the challenge GitHub repo (commaai/comma_video_compression_challenge).

The closest related discussion is **Issue #33**: "Metric feedback: PoseNet seems
to overweight low-value temporal variation outside the decision-critical scene"
(opened and closed April 2026). The author (dwallener) argued that PoseNet's
feature-similarity metric penalizes designs that compress non-critical regions
(hood, dashboard) more aggressively. The maintainer closed it, noting that
PoseNet's texture sensitivity is inherent to convolutional architectures and
that navigating this sensitivity IS the challenge.

No alternative formula was proposed in that issue or in any PR discussion.

## Geometric Mean: What Would It Look Like?

A geometric mean score would be:

```
geo_score = (seg_dist * pose_dist * rate) ^ (1/3)
```

This has fundamentally different properties:

| Property | Additive (current) | Geometric mean |
|----------|-------------------|----------------|
| Scale sensitivity | Linear in each (sqrt for pose) | Equal relative weight |
| Zero handling | One zero term reduces score | One zero term makes score zero |
| Improvement incentive | Diminishing returns on pose (sqrt) | Equal % improvement matters equally |
| Dimension coupling | Independent | Multiplicative coupling |

## Leaderboard Under Both Formulas

Known submissions with full component breakdowns:

| Submission | seg | pose | rate | Additive Score | Geo Score | Add Rank | Geo Rank |
|------------|-----|------|------|---------------|-----------|----------|----------|
| mask2mask (PR#53) | 0.00264 | 0.000654 | 0.01029 | 0.60 | 0.01185 | 1 | 1 |
| neural_inflate (PR#49) | 0.00434 | 0.0715 | 0.02443 | 1.89 | 0.02058 | 2 | 3 |
| v5 best (ours) | 0.00217 | 0.031 | 0.00401 | 0.87 | 0.01379 | - | - |
| dilated h64 (ours) | 0.006 | 0.060 | 0.046 | 1.33 | 0.02530 | - | - |
| asym v3 (ours) | 0.00210 | 0.048 | 0.100 | 1.00 | 0.02168 | - | - |

**Note**: Most public submissions (roi_v2, svtav1 variants) only report total
scores, not component breakdowns. Full geometric ranking requires all three
components.

### Computation Details

For mask2mask:
```
geo = (0.00264 * 0.000654 * 0.01029)^(1/3) = (1.776e-8)^(1/3) = 0.01185
```

For our v5 best:
```
geo = (0.00217 * 0.031 * 0.00401)^(1/3) = (2.696e-7)^(1/3) = 0.01379
```

## Analysis: How Would Rankings Change?

The geometric mean **preserves the same winner** (mask2mask) because it
dominates in all three dimensions. However, it would change the relative
distances between submissions:

1. **Geometric mean compresses the range.** The additive spread is 0.60-1.89
   (3.15x ratio). The geometric spread would be 0.01185-0.02058 (1.74x ratio).
   This makes the competition tighter.

2. **Our v5 result looks better under geometric mean.** Under additive scoring,
   we trail mask2mask by 0.27 (0.87 vs 0.60). Under geometric mean, the gap
   shrinks to 0.00194 (0.01379 vs 0.01185), a ratio of 1.16x vs 1.45x.
   This is because our seg and rate are competitive; only pose drags us down.

3. **PoseNet dominance is dampened.** The current formula applies sqrt to pose
   but linear 100x to seg, creating a regime where PoseNet matters most at
   high distortion (see optimal_allocation.py analysis). Geometric mean treats
   a 10x PoseNet improvement identically to a 10x SegNet improvement, removing
   the nonlinear PoseNet emphasis.

4. **Neural_inflate would rank worse under geometric mean** relative to our
   v5 result, because its high rate (0.024 vs our 0.004) gets penalized more
   uniformly in the geometric formulation.

## Strategic Implications

The current additive formula with sqrt(pose) creates a **PoseNet-first strategy**
at high pose distortion (where we operate). This is empirically confirmed by our
optimal allocation analysis showing d(score)/d(pose) >> d(score)/d(seg) at
pose=0.031.

A geometric mean would instead favor **balanced improvement across all three
dimensions**. Under geometric scoring:

- Our v5 result (seg=0.00217, rate=0.004) would be quite competitive since
  those dimensions are already strong
- The entire game becomes "close the PoseNet gap" regardless of formula,
  since that is our weakest dimension by far

**Bottom line**: The scoring formula does not change our strategy. PoseNet
improvement is the highest-leverage move under both formulas. The geometric
mean would merely make our current position look relatively better compared
to neural_inflate, but mask2mask would still lead.
