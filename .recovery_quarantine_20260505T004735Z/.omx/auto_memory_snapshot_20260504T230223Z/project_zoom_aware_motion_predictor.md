---
name: Zoom-aware MotionPredictor — 4ch output, smaller hidden, 11KB savings
description: With RadialZoomWarp handling flow (rank-1), MotionPredictor only needs gate(1)+residual(3)=4ch. Can shrink hidden from 24→16. Saves 11KB at int4+LZMA2. Architecturally equivalent to Quantizr (no flow prediction).
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## The Idea

RadialZoomWarp provides the dominant flow signal (rank-1, 99.8% of PoseNet variance).
The MotionPredictor currently predicts flow(2)+gate(1)+residual(3) = 6 output channels.
With zoom handling flow, it only needs gate(1)+residual(3) = 4 channels.

Gate is spatially varying (per-pixel blending mask), residual handles occlusions
and parallax corrections. These are simpler signals than optical flow →
fewer hidden channels needed.

## Param Impact

| Config | Params | FP4 KB | Int4+LZMA2 KB |
|--------|--------|--------|---------------|
| 6ch h=24 (current) | 27,702 | 14.9 | 7.4 |
| 4ch h=24 (zoom) | 27,268 | 14.6 | 7.3 |
| 4ch h=16 (zoom+slim) | 13,316 | 7.2 | 3.5 |
| 4ch h=12 (zoom+tiny) | 8,164 | 4.4 | 2.2 |

Savings at 4ch h=16: 11KB (int4+LZMA2) or 7.7KB (FP4).
This is MORE than the zoom scalars archive cost (1.2KB).

## Architecture Equivalence

Our approach: frame_t = zoom_warp(frame_t1) + gate * residual
Quantizr:    frame_t = FiLM(shared_features, pose)

Both treat frame_t1 as static (SegNet-only) and frame_t as conditioned.
The difference: we use geometric zoom + learned corrections, he uses FiLM.
Our radial zoom is geometrically exact for the rank-1 PoseNet signal.

## Implementation

Train with output_channels=4 from the start (not 6).
Pass ego_flow from RadialZoomWarp during training.
The MotionPredictor never learns flow — zoom provides it.

Requires: new AsymmetricPairGenerator constructor option or
modification to forward() to handle output_channels=4.
