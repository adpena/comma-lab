---
name: 7 Eureka Moments — Council Shower Thoughts (2026-04-12)
description: Deep insights from the council after marathon session. Scoring formula asymmetry, SegNet odd-frame freedom, YUV null space, scorer resolution exploit. Game-changing.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Eureka 1: Scoring Formula Asymmetry
dS/d(pose) = sqrt(10)/(2*sqrt(pose)) → diminishing returns. At pose=0.0005, further PoseNet improvement is worth less than SegNet or rate work. **Find the knee of the sqrt curve and stop there.**

## Eureka 2: SegNet Only Sees Even-Indexed Frames (deeper)
ODD frames (frame_t in each pair) are INVISIBLE to SegNet. They can be optimized PURELY for PoseNet + compressibility. 600 of 1200 frames have ONE FEWER constraint. Make odd frames smooth/simple → better rate.

## Eureka 3: Scorer Resolution = 384x512 (MASSIVE)
BOTH PoseNet and SegNet downscale to 384x512. Neither can distinguish a 384x512 frame upsampled to 874x1164 from the original. **Store at scorer resolution → 5.2x rate reduction.** Must verify round-trip fidelity.

## Eureka 4: YUV420 Chroma Null Space
Chroma subsampling averages 2x2 blocks. Any perturbation summing to zero within a 2x2 block is invisible. 294,912 free dimensions per frame. Use to reduce TV (improve H.265 compressibility) at zero scorer cost.

## Eureka 5: SegNet Argmax Stability
Only BOUNDARY pixels (~2-5%) need high-frequency detail. Region interiors can be flat color/smooth gradient. Hard-quantize interiors to class-mean colors, spend bits only on boundaries.

## Eureka 6: Unlimited Compress Time = Store Minimal Recipe
Pre-compute scorer outputs at compress time. Store: masks (239B) + PoseNet targets (7KB) + noise seed (64B) + correction tensor (10-50KB). Total: 10-60KB.

## Eureka 7: DALI Divergence Is Your Friend
Generated frames bypass video decode entirely (TensorVideoDataset loads raw). DALI divergence only affects GT side. Pre-compute PoseNet targets using DALI decode at compress time → eliminates 29x calibration problem.
