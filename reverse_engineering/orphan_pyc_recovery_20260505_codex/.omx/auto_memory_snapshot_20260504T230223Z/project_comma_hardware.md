---
name: Comma Hardware — EON + AR0231AT Camera Specs
description: Challenge video from Comma EON (2018), AR0231AT sensor, exact intrinsics verified
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Device: Comma EON (2018)

Video segment: `b0c9d2329ad1606b|2018-07-27--06-03-57/10/video.hevc` from comma2k19 dataset.

## Camera Intrinsics

| Property | Native (1164x874) | Scorer (512x384) |
|----------|-------------------|-------------------|
| Focal length | 910 px | 910 * (512/1164) = 400.3 px |
| Principal point | (582, 437) | (256, 192) |
| Resolution | 1164 x 874 | 512 x 384 |

Note: the 910px focal length is at native resolution. At scorer resolution (512x384), it scales to ~400px. But since the scorers resize internally via preprocess_input, the 910px value is what matters for geometric computations on the original frames.

## Sensor: OnSemi AR0231AT
- 1/2.7" CMOS BSI, 3.0μm pixels
- Native 1920x1080, processed to 1164x874
- 120 dB dynamic range, LED flicker mitigation
- YUV 4:2:0 output

## Video Specs
- 20 fps, 1200 frames, 60 seconds
- HEVC (H.265) Main profile, ~5 Mbps
- Container: MKV, 35.8 MB

**Why:** Exact intrinsics needed for ego-motion flow, road plane homography, vanishing point constraints. Wrong principal point → wrong flow fields → wrong PoseNet targets.

**How to apply:** Use `tac.camera` module for ALL camera-related constants. Native resolution for geometric computations, scorer resolution for loss evaluation.
