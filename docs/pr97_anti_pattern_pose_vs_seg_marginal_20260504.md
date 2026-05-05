# PR97 anti-pattern: pose vs seg marginal value at PR106's operating point (2026-05-04)

## Discovery

PR97 ("vibe_coder_final_boss") is the worst-scoring entry in the PR family
on the dashboard at 0.251633 — +0.042 worse than PR106 despite using
+10,921 MORE bytes. Audited the components to understand the failure mode.

## Component delta (PR97 vs PR106)

```
PR97 (vibe_coder_final_boss):
  archive_bytes:  197,160  (+10,921 vs PR106)
  pose_avg:       0.000637  (18× HIGHER than PR106's 0.000034)
  seg_avg:        0.000405  (40% LOWER than PR106's 0.000671)
  rate_unscaled:  0.005251  (5.9% higher than PR106)
  TOTAL:          0.251633  (+0.042 worse than PR106)

Score-contribution delta (positive = PR97 worse):
  pose contribution:  +0.061534  ← PR97 explodes here
  seg contribution:   -0.026630  ← PR97 wins here (substantial)
  rate contribution:  +0.007272  ← PR97 marginal loss from extra bytes
  TOTAL Δ:            +0.042176  ← net much worse
```

PR97 deliberately traded pose for seg — and got demolished. They reduced
SegNet distortion by 40% (a substantial lossy-image-side win) but their
PoseNet distortion went 18× higher. The score formula `sqrt(10 * pose_avg)`
punished them.

## The marginal-value calculation

The score formula:
```
score = 100 * seg_avg + sqrt(10 * pose_avg) + 25 * rate
```

Marginal sensitivity at PR106's operating point:
```
d(seg_contribution) / d(seg_avg)   = 100
d(pose_contribution) / d(pose_avg) = 10 / (2 * sqrt(10 * pose_avg))
                                    = 5 / sqrt(10 * 0.000034)
                                    = 5 / 0.01844
                                    = 271
```

**Pose marginal sensitivity is 2.71× the seg marginal sensitivity at PR106's
operating point.** A unit reduction in pose_avg buys 2.71× the score
reduction of the same unit reduction in seg_avg.

This INVERTS the CLAUDE.md "SegNet is 77x more important than PoseNet"
heuristic — that was at the old 1.x score operating point where pose_avg
was high enough that the sqrt was much shallower. At PR106's near-zero
pose_avg, the sqrt is steep and the marginal value flips.

## Total contribution vs marginal value — both matter

At PR106:
- SegNet TOTAL contribution: 0.0671 (3.67× larger than pose's 0.0183)
- POSE marginal sensitivity: 2.71× larger than seg

So:
- If you must pick ONE component to attack at fixed budget, optimize POSE
- If you've already pushed pose to its floor, then attack SegNet for the
  larger total payback

PR97's mistake was trading pose AWAY for seg gains. They went the wrong
direction on both axes (lost where the marginal was steep, won where it
was shallow).

## Implications for our PR106-stacking lanes

This validates the **score-aware sidechannel paradigm direction**:

| Lane | Targets | Mechanism |
|---|---|---|
| `lane_pr106_latent_sidecar` | pose-related latent dim per pair | per-pair correction trained against scorer |
| `lane_pr106_yshift_sidechannel` | sub-pixel pose alignment | per-frame (dy, dx) correction |

Both target POSE-relevant errors at PR106's operating point — exactly the
high-marginal-value direction. The 6-variant paradigm catalogue includes
SegNet-relevant variants (LRL1 luma residual, qpose14 seg_tile_actions)
but those are TERTIARY at PR106's pose-marginal-dominated operating point.

The apogee_intN bit-width reduction is a different axis (rate, with pose+seg
distortion as side-effects). Need empirical data to know which component
dominates the distortion side-effect — that's exactly what the int5 dispatch
will surface.

## Decision: NO new lane

PR97's anti-pattern is informative for prioritization, not a new lane to
chase. The existing pre-registered lanes are pointed correctly. This memo
is preventive — it documents WHY a future agent shouldn't chase a SegNet-
focused lane at PR106's operating point until pose is exhausted.

## Cross-refs

- PR97 source: `experiments/results/lightning_batch/exact_eval_public_pr97_vibe_coder_final_boss_t4_dup_20260504T0930Z/auth_eval.log`
- PR106 reference: `experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_adapter_t4_20260504T1330Z/auth_eval.log`
- Sister training-recipe finding: `docs/pr106_vs_pr101_training_recipe_finding_20260504.md`
- Sidechannel paradigm INDEX: `docs/INDEX_score_aware_sidechannel_thread_20260504.md`
- Pre-registered lanes (both target pose):
  - `lane_pr106_latent_sidecar` (latent-space pre-decode)
  - `lane_pr106_yshift_sidechannel` (pixel-space post-decode)
- CLAUDE.md "SegNet paradigm shift" section: cumulative reference (now needs
  update — the 77× ratio is at the OLD operating point, not PR106's)
