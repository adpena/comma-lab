---
name: Experiment Landscape and Theoretical Minimum
description: Full map of all techniques, their targets, status, and the theoretical score floor (~0.975) as of 2026-04-10
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Current Floor: 1.33 (dilated h=64, authoritative confirmed)

## Technique Map — Nothing Abandoned

| Technique | Attacks | Expected Gain | Status |
|-----------|---------|--------------|--------|
| Dilated architecture | PoseNet (5.6x proven) | DEPLOYED | Promoted at 1.33 |
| Per-channel int8 quantization | Both (better fidelity) | -0.02 to -0.05 | Implemented in tac v0.9.0 |
| Dual saliency (alpha_seg=5000) | SegNet (-0.15 to -0.25) | Tier 1 | Running on local MPS |
| KL distill (T=5→2→1 stepwise) | SegNet soft targets | -0.10 to -0.20 | Ready, deploy to Lightning |
| STE boundary weighting (5x) | SegNet boundaries | Stacked with dual sal | Running on local MPS |
| Hard-frame upsampling (worst 20%) | SegNet | -0.10 to -0.20 | Needs ~30 lines in Trainer |
| DualHead architecture (1x1 seg + 3x3 pose) | Both | -0.05 to -0.10 | Needs implementation |
| Pair-aware 6ch input | PoseNet temporal | -0.03 to -0.08 | Implemented, undeployed |
| Test-time optimization (5 Adam steps) | Both | High variance | Concept, inflate-time |
| h=96 dilated scaling | Both | Scaling law | Modal A10G ep 759 |
| Multi-pass inflate (CNN twice) | Both | +20-30% of first pass | Free, inflate-time |
| LSQ learned step size | Both | -0.03 to -0.06 | Implemented, undeployed |

## Theoretical Minimum (all techniques compounding)

- SegNet: 0.00610 → 0.003 (dual sal + KL distill + hard-frame + boundary STE)
- PoseNet: 0.00218 → 0.001 (dilated + pair-aware + TTO)
- Rate: 0.0230 (locked by codec params)
- **Score: 100*0.003 + sqrt(10*0.001) + 25*0.023 = 0.30 + 0.10 + 0.575 = 0.975**

## Key Insight: Saliency Inversion

Current PoseNet-only saliency (alpha=20) actively constrains corrections at SegNet
boundary pixels. SegNet has 590x marginal leverage over PoseNet at this operating point.
Flipping to dual saliency with alpha_seg=5000 is the highest-EV single change.

**Why:** The reconstruction loss penalizes modifications at low-PoseNet-saliency regions,
which is exactly where SegNet class boundaries are. We're telling the CNN "don't touch
the pixels that matter most for 62% of our score."

**How to apply:** All new training runs should use `--use-dual-saliency --alpha-seg 5000`.
