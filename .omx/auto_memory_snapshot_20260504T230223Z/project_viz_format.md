---
name: Comma Contest GIF Format
description: Exact layout of the comma.ai competition README visualization — must replicate for writeup
type: reference
---

The contest README GIF is a 4-panel animated layout at 4× speed:

**Top row (video panels):**
- Top-left: "speed 4x — original — 37.5MB" — raw uncompressed dashcam video
- Top-right: "speed 4x — compressed — 1.4MB" — compressed/reconstructed video

**Bottom row (error panels):**
- Bottom-left: "segnet errors" — binary white-on-black mask showing pixels where argmax(compressed) != argmax(original)
- Bottom-right: "posenet errors" — running line chart of per-frame pose distortion over all 1200 frames

**Our version should be 6-panel (or 3-column):**
- Top: Original / Baseline (no filter) / Ours (post-filter)
- Bottom: SegNet error masks for each / PoseNet line chart comparing both

This shows our post-filter reducing errors visually. The binary disagreement mask is the key — viewers can SEE the white pixels disappearing where our filter corrects.
