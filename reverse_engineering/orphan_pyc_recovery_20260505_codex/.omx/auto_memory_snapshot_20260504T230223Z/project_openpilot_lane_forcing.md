---
name: openpilot policy model for lane-based speed/zoom estimation at compress time
description: Use openpilot's lane detection and path planning models at compress time (unlimited compute, contest-compliant) to derive physically-informed zoom scalars. Warm-start for gradient optimization.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

openpilot's driving models understand the exact road geometry from this camera (EON AR0231AT).
At compress time: use lane detection → polynomial fit → inter-frame displacement → zoom scalars.
At inflate time: just apply the stored/computed zoom (no openpilot model needed).

Potential sources:
- openpilot supercombo model (lane lines, path, lead car)
- comma2k19 calibration data (per-segment camera pose)
- openpilot's calibration daemon (refined extrinsics)

Contest compliant: unlimited compute at compress time per rules.
Best approach: openpilot warm-start + PoseNet gradient polish.
