# Geometric Mean Scoring Analysis

## Current Scoring Formula

The comma.ai video compression challenge uses an additive composite metric:

```
score = 100 * seg_dist + sqrt(10 * pose_dist) + 25 * rate
```

This is a **weighted linear combination** with a sqrt transform on PoseNet.
Lower is better.

## Did Anyone Propose Geometric Mean Scoring?

No. No public discussion of a geometric mean alternative exists on the challenge
repo (commaai/comma_video_compression_challenge).

The closest related discussion is **Issue #33** (opened and closed April 2026):
dwallener argued PoseNet overweights non-critical regions. The maintainer closed
it, noting that navigating PoseNet's texture sensitivity IS the challenge.

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
| mask2mask (PR#53) | 0.00264 | 0.000654 | 0.01029 | 0.60 | 0.00261 | 1 | 1 |
| neural_inflate (PR#49) | 0.00434 | 0.0715 | 0.02443 | 1.89 | 0.01964 | 2 | 2 |
| v5 best (ours) | 0.00217 | 0.031 | 0.00401 | 0.87 | 0.00646 | - | - |

**Note**: dilated h64 (1.33) and asym v3 (1.00) are excluded because their
archived component breakdowns do not reproduce their authoritative scores
under the formula. Only entries with verified component-to-score consistency
are shown. Most public submissions (roi_v2, svtav1 variants) only report
total scores, not component breakdowns.

### Computation Details

For mask2mask:
```
geo = (0.00264 * 0.000654 * 0.01029)^(1/3) = (1.776e-8)^(1/3) = 0.00261
```

For our v5 best:
```
geo = (0.00217 * 0.031 * 0.00401)^(1/3) = (2.697e-7)^(1/3) = 0.00646
```

## Analysis: How Would Rankings Change?

The geometric mean **preserves the same winner** (mask2mask) because it
dominates in all three dimensions. However, it would change the relative
distances between submissions:

1. **Geometric mean compresses the range.** The additive spread is 0.60-1.89
   (3.15x ratio). The geometric spread is 0.00261-0.01964 (7.52x ratio).
   Geometric mean actually amplifies the spread because it penalizes balanced
   weakness across all dimensions more than additive scoring does.

2. **Our v5 result looks worse under geometric mean.** Under additive scoring,
   we trail mask2mask by ratio 1.45x (0.87 vs 0.60). Under geometric mean,
   the ratio widens to 2.48x (0.00646 vs 0.00261). Our 47x PoseNet deficit
   (0.031 vs 0.000654) gets fully exposed without the sqrt dampening.

3. **PoseNet dominance is dampened.** The current formula applies sqrt to pose
   but linear 100x to seg, creating a regime where PoseNet matters most at
   high distortion (see optimal_allocation.py analysis). Geometric mean treats
   a 10x PoseNet improvement identically to a 10x SegNet improvement, removing
   the nonlinear PoseNet emphasis.

4. **Neural_inflate still ranks worse under geometric mean** relative to our
   v5 result (0.01964 vs 0.00646), because its higher distortions across all
   three dimensions compound multiplicatively.

## Strategic Implications

The current additive formula with sqrt(pose) creates a **PoseNet-first strategy**
at high pose distortion (where we operate). PoseNet contributes 63.7% of our
score despite having the lowest marginal sensitivity (8.98 vs 100 for SegNet),
because our pose distortion (0.031) is 47x worse than mask2mask's.

A geometric mean would favor **balanced improvement across all three
dimensions**. Under geometric scoring:

- Our v5 result would be penalized harder (2.48x ratio vs 1.45x under additive)
  because the geometric mean exposes our 47x PoseNet deficit without the sqrt
  dampening the current formula provides
- The game is "close the PoseNet gap" regardless of formula, since that is
  our weakest dimension by far

**Bottom line**: The scoring formula does not change our strategy. PoseNet
improvement is the highest-leverage move under both formulas. Under geometric
mean our position actually looks worse relative to mask2mask (2.48x vs 1.45x
ratio), reinforcing that PoseNet is the binding constraint.
