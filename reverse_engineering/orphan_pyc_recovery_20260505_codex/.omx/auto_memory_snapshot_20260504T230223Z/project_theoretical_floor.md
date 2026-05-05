---
name: Theoretical Score Floor Under Mask-Rendering Paradigm
description: Council analysis — sub-0.30 achievable, sub-0.10 stretch, floor is effectively zero. Rate term is free.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Rate anomaly (Tao)
mask2mask's 386KB archive with our 37.5MB uncompressed reference gives rate=0.0103,
rate_term=0.257. But the rate formula is archive/uncompressed — at 386KB/37.5MB
the rate should be much lower. The stated rate_term of 0.257 suggests the scorer
uses a different uncompressed size. VERIFY with our eval pipeline.

## Key insight: rate is effectively free
Under mask-rendering, the archive is ~200-400KB total. Rate term = 0.003-0.01.
The score is almost entirely seg_term + pose_term.

## Theoretical floor by component
- SegNet: 0 (masks ARE the ground truth argmax — renderer just needs to stay in decision regions)
- PoseNet: 0 in theory, ~0.0001-0.0005 in practice (noise floor of the metric)  
- Rate: ~0.003 at 400KB archive

## Score projections
| Quality | seg | pose | archive | score |
|---------|-----|------|---------|-------|
| mask2mask actual | 0.00264 | 0.00066 | 386KB | 0.60 |
| Better renderer | 0.00100 | 0.00040 | 250KB | 0.17 |
| Optimal | 0.00050 | 0.00030 | 200KB | 0.11 |
| Stretch | 0.00020 | 0.00020 | 150KB | 0.07 |

## Why seg→0 is achievable (Einstein)
The masks ARE SegNet's argmax on GT frames. The renderer only needs to produce
frames where SegNet's argmax matches the input masks — NOT reproduce pixel values.
This is strictly easier than image reconstruction.

## The tradeoff vanishes (Pareto)
Under mask-rendering, better renderer → better seg AND better pose simultaneously.
No Pareto tradeoff. The only coupling is model size vs quality, but rate is so
cheap that models up to 5MB are affordable.

## Scorer structure insight
Score = 100*S + sqrt(10*P) + 25*rate
- Only first 6 PoseNet outputs matter (not all 12)
- Only last-frame SegNet argmax (not both frames)
- Byte accounting is analytically separable from distortion
- Parameter optimization is nonconvex and partly discrete

**Why:** This defines our optimization target for the GPU lane.
**How to apply:** Build the best possible mask→frame renderer. Rate doesn't matter.
SegNet is near-free (masks ARE ground truth). PoseNet quality is the differentiator.
