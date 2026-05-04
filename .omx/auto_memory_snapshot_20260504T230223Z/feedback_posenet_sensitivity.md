---
name: PoseNet Extreme Sensitivity to Preprocessing
description: ANY blur or degradation outside the driving corridor catastrophically hurts PoseNet — learned from 4 scored experiments
type: feedback
---

Never apply Gaussian blur or spatial degradation to any part of the frame before encoding. PoseNet uses the ENTIRE frame for camera pose estimation, including distant features (buildings, horizon line, road perspective vanishing points). Even sigma=0.8, blend=0.25 (barely visible degradation) caused +90% PoseNet distortion increase.

**Why:** Scored experiments proved this: J (2.52, +104% PoseNet), K (2.47, +90% PoseNet). The scoring function sqrt(10*posenet) amplifies PoseNet damage. A 90% increase in PoseNet from 0.094 to 0.178 costs 0.37 score points — more than all possible rate savings combined.

**How to apply:** 
- Do NOT use spatial blur for preprocessing (Gaussian, bilateral, guided filter)
- Safe alternatives: chroma-only degradation, hqdn3d temporal denoise (preserves edges), sky-only replacement
- If using ML masks for preprocessing, the mask must protect 90%+ of the frame (essentially everything except sky)
- Any future preprocessing experiment must explicitly justify why PoseNet won't be hurt
