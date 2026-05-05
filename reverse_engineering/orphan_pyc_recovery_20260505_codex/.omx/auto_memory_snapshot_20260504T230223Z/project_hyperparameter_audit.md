---
name: Hyperparameter Audit — Einstein/Tao Derivations (2026-04-10)
description: Every arbitrary value audited. Three highest-EV changes identified. alpha_seg=5000 is wrong (should be 200). boundary_weight optimal is ~20.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Key Findings

1. **alpha_seg=5000 is ~20x too aggressive.** Formula-derived value is 11.5 * alpha_pose ≈ 230.
   Code default is already 200. The "5000" was a chat-session council guess, not validated.
   The running dual-saliency experiments use 5000 — may explain why they're slower than KL distill.

2. **boundary_weight optimal is ~20** (from 1/boundary_fraction where boundaries are ~5% of pixels).
   We're using 5-10. This means boundary pixels get 1/4 to 1/2 of optimal gradient pressure.

3. **524x394 encoding introduces double interpolation** into the scorer's resize path.
   Scorer resizes to 512x384 internally. Encoding at exactly 512x384 would eliminate scale jitter.
   Zero cost to test (codec rebuild, no retraining).

## Three Highest-EV Changes
1. hard_frame_ratio=0.3 + error_replay_every=200 → estimated **0.10-0.25** score improvement
2. boundary_weight sweep 10→20 → estimated **0.05-0.15**
3. Test 512x384 encoding + sharpness sweep → estimated **0.03-0.10**

## Validated (keep as-is)
- hidden=64, CRF=34, film-grain=22, segnet_loss_weight=100, alpha_seg=200 (code default),
  ema_decay=0.997, grad_clip=1.0, symmetric int8, boundary_mask threshold=0.5

## True guesses needing sweeps
- lr=5e-4 (try 1e-3), focal_gamma=2 (try 3.5), sharpness=1 (try 0 and 2),
  kernel=3 (try 5 with dilation), 524x394 (try 512x384)

**Critical insight**: The dual-saliency experiments running with alpha_seg=5000 are using
a value 20x higher than the formula-derived optimum. This likely explains their slower
convergence vs KL distill.
