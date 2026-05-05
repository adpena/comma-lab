---
name: KL Distill PoseNet Issue — SECOND CONFIRMATION
description: KL distill with sw=30 (council_v1_fixed) scored 2.05 auth vs 1.43 proxy. PoseNet at 0.081 vs our 1.33's 0.002.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Second KL distill authoritative eval: 2.05 (proxy was 1.43)

| Checkpoint | Proxy | Auth | PoseNet (auth) | SegNet (auth) |
|-----------|-------|------|---------------|--------------|
| 1st KL (with cap, sw=100) | 1.25 | 1.85 | 0.05725 | 0.00493 |
| 2nd KL (no cap, sw=30) | 1.43 | **2.05** | **0.08095** | 0.00546 |
| Promoted (standard, dilated) | ~3.4 | **1.33** | 0.00218 | 0.00610 |

## Root Cause Analysis

KL distill fundamentally over-weights SegNet at the expense of PoseNet:
- SegNet improves beautifully (0.00610 → 0.00546, 10% better)  
- PoseNet collapses (0.00218 → 0.08095, 37x worse)
- The proxy under-reports PoseNet damage (α_p ≈ 1.2 per eval)
- But PoseNet was already bad in training — the proxy faithfully reported
  pose=0.066 (30x worse than the 1.33 checkpoint's 0.002)

The problem is NOT the proxy being unfaithful. The problem is that KL distill
with ANY sw > ~5 pushes SegNet so hard that PoseNet has no gradient budget.

## CRITICAL LESSON

Standard loss (proven_baseline) remains the ONLY technique that has produced
a good authoritative score. KL distill improves SegNet but at catastrophic
PoseNet cost. The scoring formula's sqrt(10*pose) makes PoseNet regressions
very expensive — going from 0.002 to 0.08 costs 0.75 score points.

## What to Do

1. Stop all KL distill experiments with sw > 5
2. Test KL distill with sw=1-3 (very light SegNet pressure)
3. OR: use standard loss + boundary_weight for SegNet improvement
4. OR: two-phase training — standard loss until PoseNet converges,
   then switch to KL distill for SegNet fine-tuning
5. The PSD architecture advantage may ONLY work with standard loss
