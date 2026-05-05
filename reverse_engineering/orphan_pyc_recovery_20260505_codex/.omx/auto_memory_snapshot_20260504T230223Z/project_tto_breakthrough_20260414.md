---
name: TTO Breakthrough and Battle Plan (2026-04-14, VALIDATED)
description: TTO warm-start preserves PoseNet=0.0000 from GT. Validated by v5a/v5b (auth 0.41-0.43). Now 13 days left.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## STALE NOTICE
Original: April 14. Updated: April 20. TTO hypothesis VALIDATED (v5a auth 0.43, v5b auth 0.41).
These are [unlimited-compute] scores. Contest-compliant remains 0.87.

## TTO Warm-Start Breakthrough (2026-04-14) -- VALIDATED
- coupled_trajectory_optimize with init_frames=GT: PoseNet stays 0.0000, SegNet improves to 0.011
- With init_frames from RENDERER: v5a auth 0.43 (MSE), v5b auth 0.41 (embedding) [unlimited-compute]
- THE GREAT GRADIENT BUG was found and fixed (make_scorers_differentiable())
- Embedding loss (512D) strictly superior to output MSE (6D)

## Score Math (Yousfi's formula)
- Our 0.87 = SegNet 0.22 + PoseNet 0.56 + Rate 0.10 [contest-compliant]
- Our 0.41 = SegNet 0.15 + PoseNet 0.16 + Rate 0.10 [unlimited-compute, TTO v5b]
- Quantizr 0.33 = SegNet 0.06 + PoseNet 0.07 + Rate 0.20 [contest-compliant, PR#55]

## Current Focus (April 20, 13 days left)
1. **Lane 1 (Contest):** FiLM + eval roundtrip in renderer training -> sub-0.40
2. **Lane 2 (Unlimited):** TTO v7 hinge + roundtrip -> sub-0.25
3. **Lane 3 (Research):** Stack everything -> theoretical floor

## Validated Techniques
- TTO warm-start from renderer output: WORKS
- Embedding loss 512D: strictly superior
- Hinge loss: implemented, untested at scale
- Eval roundtrip: implemented, untested at scale
- Two-phase TTO: implemented, untested at scale
