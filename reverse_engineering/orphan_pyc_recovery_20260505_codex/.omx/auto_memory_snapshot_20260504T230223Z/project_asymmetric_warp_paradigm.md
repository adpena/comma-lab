---
name: Asymmetric Warp Paradigm — The Path to Beat Quantizr
description: Council binding decision 2026-04-12: adopt Quantizr's warp-based pair generation. frame2=direct render, frame1=warp(frame2)+gated residual. Option C killed.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## The Paradigm (deobfuscated from Quantizr PR #53)

```
frame2 = renderer(mask2)                    # Direct render (anchor)
flow, gate, residual = motion(mask1, mask2) # Motion from BOTH masks  
frame1 = warp(frame2, flow) + gate * residual  # Geometric + correction
```

PoseNet sees a geometric warp between frames → near-zero distortion (0.00066).

## Council Decisions (all binding, 2026-04-12)

1. **Option F replaces Option C** — warp paradigm, not shared backbone
2. **Keep learned embeddings** (5→6 dim) — already have this
3. **Add coordinate grid conditioning** — 2 extra input channels, free spatial info
4. **Implement AsymmetricPairGenerator** as NEW class alongside existing
5. **Keep GroupNorm(1)** for now — sweep later, don't confound
6. **Keep CLADE U-Net** — our building blocks are more expressive than Quantizr's
7. **Flow bounds:** max 12px, residual ±20, gate sigmoid — domain priors

## Implementation Order
1. Coord grid → MaskRenderer (10 lines)
2. MotionPredictor output_channels=6 
3. AsymmetricPairGenerator class
4. Profile config `pair_mode: "asymmetric"`
5. Training run

## What We Keep
- MaskRenderer (CLADE U-Net), ResBlock, channel widths 36/60
- Standard loss, Fridrich curriculum
- Export/inflate pipeline (update for pair-wise)

## Kill/Success
- Kill: PoseNet regresses 2x after 1000 epochs
- Success: Proxy < 0.85 (current 0.92), PoseNet improvement specifically
