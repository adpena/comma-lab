---
name: Transcendent Council Session — Phenomenology of Fake Driving
description: Radial flow fields, 7x7 patch locality, warp-gen baseline, score formula audit needed, sparse patch TTO
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Key Insights

### Fridrich: PoseNet reads RADIAL FLOW FIELDS, not pixels
PoseNet doesn't measure ego-motion — it measures optical flow patterns that CORRELATE
with ego-motion. Lane markings have known depth (ground plane), so their flow magnitude
is a pure velocity signal. Generated frames need correct radial flow consistency between
consecutive frames, not correct RGB values.

From music: vibrato detected by RATE OF CHANGE, not absolute pitch. PoseNet detects
motion by rate of pixel displacement between frames.

### Yousfi: PoseNet has 7x7 receptive field on conv1
Cannot see global structure. Builds ego-motion from LOCAL texture displacements.
Frames don't need globally coherent geometry — just correct 7x7 patches at the
right locations with the right displacement.

Sparse patch idea: optimize ONLY the 50 highest-gradient PoseNet locations.
Leave everything else as flat semantic color. 99% less optimization surface.

### Hotz: Deterministic warp-gen (no neural network)
1. Render frame N from mask (flat colors + class texture)
2. WARP by RAFT flow to get frame N+1
3. Blend with mask N+1 for occlusion
No training. Deterministic. If this scores < 1.5 it obsoletes the neural renderer.

### Quantizr: SCORE FORMULA AUDIT NEEDED
auth=1.00 with seg=0.210, pose=0.692, rate=0.100 doesn't add up under the formula
100*seg + sqrt(10*pose) + 25*rate = 0.21 + 2.63 + 2.50 ≠ 1.00
Wait — the formula is 100*seg_dist + sqrt(10*pose_dist) + 25*rate where seg/pose
are raw DISTORTION values, not the weighted contributions.
MUST VERIFY: what exactly are the "seg" and "pose" numbers in our auth eval?

### Contrarian: Frames are intermediate, not output
Optimal PoseNet frames would be deeply alien — high-contrast edge patterns at specific
spatial frequencies. A hoverfly doesn't look like a wasp to a human, it looks like a
wasp to a BIRD. Our frames should look like driving video to the SCORERS, not to us.

## Binding Experiments (all actionable)
1. AUDIT score components (30 min) — verify formula before allocating effort
2. WARP-GEN baseline (2h, no GPU) — deterministic mask+RAFT warp pipeline
3. PoseNet gradient visualization (1h) — what pixels does PoseNet actually use?
4. Marginal return equalizer (1h) — Pareto-optimal allocation
5. Sparse patch optimization (3h) — TTO on only 50 PoseNet-sensitive 7x7 patches
6. Paper hero figure — animate PoseNet gradient heatmap across 60 frames
