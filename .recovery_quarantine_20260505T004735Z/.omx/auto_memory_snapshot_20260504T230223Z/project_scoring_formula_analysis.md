---
name: Scoring Formula Deep Analysis
description: Why SegNet is 100x and PoseNet has sqrt — comma's real product reasons, not arbitrary
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Key finding (2026-04-10)

The 100x SegNet weight is NOT arbitrary. It reflects comma's real product priorities:

### SegNet = "Did I preserve what the scene IS?"
- 5-class segmentation: road, lane markings, undrivable/sky, movable objects
- Failure = lane markings wash out, road edges become ambiguous, vehicles blend into background
- SegNet has NO backup sensor — if compressed video breaks classification, driving stack has no fallback
- Used for: visualization, supervision, evaluation, fallback logic, model-development tooling
- Linear weight (100x) because any regression is directly harmful

### PoseNet = "Did I preserve how the scene is MOVING?"
- 6-DOF ego-motion from consecutive 2-frame pairs
- Failure = temporal/geometry cues damaged, motion consistency lost
- PoseNet HAS backup: IMU + wheel odometry provide redundant ego-motion signal
- sqrt weight because: (a) already noisy/high-variance, (b) correctable by other sensors
- Connected to world-model and simulator work at comma

### Our position (seg=0.00610, pose=0.00218)
We're saying: "I preserved motion/geometry cues excellently, but I may be blurring semantic structure"
This is the WRONG priority per the formula. SegNet matters more and we made it worse.

### Rate in context
- `rate = compressed_size / uncompressed_size` — linear at 25x
- Smaller files = more routes retained locally, cheaper upload, faster training
- Contest does NOT score runtime/latency/energy — only distortion + size

### For comma four deployment
- Our 46KB CNN runs ~30ms CPU — potentially deployable
- But contest doesn't measure encode/decode speed, memory, thermal
- The technique is directly relevant to their data pipeline even if not real-time deployable

**Why:** This analysis corrects our assumption that PoseNet matters more for safety.
**How to apply:** SegNet improvements should be prioritized over PoseNet. The PCGrad/MRS-adaptive
approach should steer toward recovering SegNet regression while holding PoseNet gains.
