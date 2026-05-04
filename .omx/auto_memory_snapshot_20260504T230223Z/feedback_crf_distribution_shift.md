---
name: Postfilter is CRF-Specific — Distribution Shift Kills Transfer
description: Dilated filter trained on CRF 34 destroyed PoseNet 7.3x on CRF 35 video. Must retrain per CRF.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
Auth eval CRF 35 + dilated postfilter (trained on CRF 34): score 2.08 (WORSE than baseline 1.51).
PoseNet: 0.0898 (7.3x worse than baseline 0.0123). SegNet: 0.00581 (fine).

The filter's corrections are specific to CRF 34's quantization artifacts.
CRF 35 produces different artifact patterns → corrections are actively harmful.

**Why:** CRF changes the quantization step size, which changes the spatial pattern of
compression artifacts. The filter learned to undo CRF 34's specific patterns. On CRF 35
video, those corrections shift pixels in the wrong direction for PoseNet.

**How to apply:** ANY change to encoder settings (CRF, preset, resolution, film-grain)
requires retraining the postfilter from scratch on the new compressed video.
Never test a postfilter on video compressed with different settings than it was trained on.
This is a fundamental distribution matching requirement — same as training on the actual
submission archive (lesson learned earlier in the campaign).
