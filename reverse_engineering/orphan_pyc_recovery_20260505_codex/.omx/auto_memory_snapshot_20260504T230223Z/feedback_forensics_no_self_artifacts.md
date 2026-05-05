---
name: Forensic tools must not introduce the artifacts they detect
description: boundary_artifact_score used zero-padded gradients while detecting zero-padding artifacts. Fixed to replicate padding. Also: use max() not geometric mean (independent detection features), 8-connected not 4-connected (SRNet 3x3 kernels).
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

Three forensics bugs found by Fridrich:

1. Zero-padded gradient computation in a zero-padding detection tool.
   The tool's own padding biased the border measurement downward.
   Fix: replicate padding on finite difference gradients.

2. Geometric mean artifact score collapses to zero when ANY metric is zero.
   But each metric captures an independent detection feature (like SRNet).
   Fix: use max() of normalized metrics.

3. 4-connected boundary detection misses diagonal class transitions
   (10-20% undercount for angled lane markings). SRNet uses full 3x3.
   Fix: 8-connected boundary detection.

**How to apply:** Every analysis tool must be audited for self-contamination.
If a tool measures X, it must not introduce X in its own computation.
