---
name: PoseNet-SegNet Pareto Frontier Discovery
description: GIF visualization revealed 117% of gains from PoseNet, SegNet regressed 5.2%. True score minimum lies on unexplored Pareto frontier.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Discovery (2026-04-10)

The color-coded SegNet diff GIF made it visually obvious what the numbers showed:
our dilated h=64 post-filter is a **PoseNet optimization machine** that trades SegNet.

### Score decomposition (1.33 auth)
- PoseNet contribution: **-0.2028** (117% of total 0.1728 improvement)
- SegNet contribution: **+0.0300** (5.2% regression vs baseline)
- Raw: PoseNet 5.6x better, SegNet 5.2% worse

### The Pareto frontier
Three regimes exist:
1. **PoseNet-optimal**: dilated h=64 (current, score 1.33). PoseNet 0.00218, SegNet 0.00610
2. **SegNet-optimal**: KL distill (DEAD, score 2.05). SegNet 0.00546, PoseNet 0.08095
3. **Unexplored middle**: the true minimum lies here

### Marginal sensitivities at current operating point
- d(score)/d(seg) = 100.0 (constant)
- d(score)/d(pose) = 3.4 (diminishing via sqrt)
- SegNet is 29.6x more valuable per unit, but PoseNet had 40x more room

### Council-recommended techniques to find the minimum
1. **Two-phase training**: freeze conv1/conv2, fine-tune conv3 with focal_ste_loss (EV: 0.015-0.025)
2. **Boundary-aware residual masking**: attenuate corrections at class boundaries (EV: 0.010-0.020)
3. **Gradient projection (PCGrad)**: project out SegNet gradients opposing PoseNet (EV: 0.010-0.020)
4. **CRF/rate sweep**: rate is 43% of score, never systematically optimized (EV: 0.030-0.080)
5. **Standard (non-dilated) h=64 + two-phase**: may find better tradeoff point

### Headroom
- Fix SegNet to baseline: 1.303 (-0.030)
- Halve PoseNet: 1.290 (-0.043)
- CRF 36: 1.293 (-0.040)
- All combined realistic: ~1.22

**Why:** The visualization surfaced a strategic blind spot — we optimized one axis to exhaustion.
**How to apply:** Always evaluate both SegNet and PoseNet independently. Design experiments that explore the Pareto frontier, not just one extreme.
