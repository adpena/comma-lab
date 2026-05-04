---
name: Deep Understanding of the Scoring Function
description: 18 experiments reveal the scoring function's sensitivities — PoseNet dominates, preprocessing kills it, two paths to 2.08
type: project
---

After 18+ scored experiments, we understand the scoring function deeply:

**Score = 100*seg + sqrt(10*pose) + 25*rate**

At operating point 2.08:
- SegNet: ~0.575 pts (100 * 0.00575)
- PoseNet: ~0.969 pts (sqrt(10 * 0.094))  
- Rate: ~0.576 pts (25 * 0.023)

**PoseNet dominates** — it's ~47% of the total score. But it has diminishing returns (sqrt function). SegNet and rate are ~28% each.

**Two equivalent paths to 2.08:**
1. ffmpeg lanczos + unsharp=0.35 + sharpness=1 → better SegNet (0.00575)
2. Python bicubic + binomial USM 0.40 + sharpness=1 → better PoseNet (0.08521)

**Why:** Different decode interpolation methods trade SegNet for PoseNet. Neither wins overall.

**How to apply:** To break below 2.08, need a technique that improves BOTH, or reduces rate without hurting either. ALL preprocessing approaches are dead ends (kills PoseNet by 21-105%). Only encoder params and decode filter tuning are safe.
