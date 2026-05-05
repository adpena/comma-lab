---
name: YUV Channel GIF Visualization
description: User wants a YUV channel comparison (original vs baseline vs ours) for the writeup — shows what PoseNet actually sees
type: project
---

PoseNet's `debug_run` in upstream `modules.py` converts RGB→YUV 4:2:0 (BT.601) and renders the Y00 (luma half-res) channel as a 2-frame animated GIF showing consecutive frames.

**Goal:** Create a 3-panel YUV channel comparison:
- Original Y00 channel | Baseline compressed Y00 | Ours (post-filtered) Y00

**Why this matters for the writeup:**
- PoseNet sees YUV, not RGB. The Y00 channel at half-res is where compression artifacts concentrate.
- Our post-filter corrects the YUV representation the scorer uses, not the RGB that looks fine to humans.
- Visually demonstrates WHY the filter works — viewers can see the artifacts our CNN removes.
- The upstream `rgb_to_yuv6` function (frame_utils.py) does: BT.601 Y, average 2×2 for U/V subsampling, produces 6 channels: y00, y10, y01, y11, U_sub, V_sub.

**Implementation:** Use `frame_utils.rgb_to_yuv6` from upstream to convert frames, then display the y00 channel for all three versions side by side as an animated GIF.

**Priority:** Polish item for writeup. Build after the main comma-format GIF is finalized.
