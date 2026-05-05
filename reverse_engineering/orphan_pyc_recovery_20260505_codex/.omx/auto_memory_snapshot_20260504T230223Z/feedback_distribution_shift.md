---
name: Distribution Shift Kills Learned Filters
description: Post-filter v1 failed because it was trained on FG=0 archive but applied to FG=22. Always train on target distribution.
type: feedback
---

When training learned post-filters / correction models for the comma video compression
challenge, ALWAYS train on the exact archive distribution the filter will be applied to.

**Why:** Post-filter v1 (2026-04-07) was trained on the FG=0 archive (PoseNet 0.367, 
catastrophic baseline). It learned to add grain-like high-frequency texture as recovery.
Proxy results: -71% PoseNet, total score 2.47 → 1.64.

When applied to the canonical FG=22 archive (PoseNet 0.087), the filter ADDED its 
trained correction on top of the existing film grain, OVERCORRECTING and producing 
score 2.35 (worse than the 2.08 floor by 0.27).

**How to apply:**
- The training baseline frames must come from the SAME encoded archive that will be 
  used at submission time
- If switching from FG=0 training data to FG=22, ALL model weights from the previous 
  training are invalidated
- Validate with proxy AND real scorer before relying on a filter
- The proxy → real scorer transfer is fragile; large proxy improvements may be 
  artifacts of training distribution, not real generalization
