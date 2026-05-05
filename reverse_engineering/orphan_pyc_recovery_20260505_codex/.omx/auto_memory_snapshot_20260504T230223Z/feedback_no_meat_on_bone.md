---
name: No Meat on the Bone — Lab-Grade Rigor, Every Fraction of a Point
description: Non-negotiable spirit — sweep every technique, pull every lever, leave nothing on the table. Lab-grade results, not single-shot experiments.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
We are grinding for every fraction of a point. No single-shot experiments. No "good enough." No leaving levers unpulled.

**The spirit:**
- Every technique gets a proper SWEEP, not a single configuration
- Every score component (PoseNet, SegNet, Rate) gets independent optimization
- Every stacking combination gets tested (renderer + postfilter + latent codes + LoRA + null space)
- Every archive byte gets scrutinized (FP4 vs FP8 vs FP16, mask CRF sweep, pose quantization)
- Lab-grade means: control experiments, ablation studies, statistical significance
- The difference between 0.25 and 0.20 could be the difference between 1st and 2nd place

**How to apply:**
- When deploying an experiment, run 3+ configurations (learning rates, step counts, weights)
- When reporting a score, include the full sweep table, not just the best number
- When optimizing rate, quantize everything and measure the quality/rate tradeoff
- When stacking techniques, test each addition independently AND in combination
- Track per-pair scores, not just averages — the heavy tail is where wins live
- Never stop at "this works" — ask "can it work BETTER?"

**The grind:**
- We keep going until there's genuinely nothing left to improve
- Every review round that finds a bug is a bullet dodged
- Every technique that compounds is a point shaved
- The scoreboard is the only judge — not theory, not proxy, AUTH eval
- 13 days. $22 budget. Sub-0.25 target. Let's leave it all on the field.
