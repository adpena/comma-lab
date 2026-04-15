# Human Insights — For Council Agents

These are direct insights from the human (adpena) that must be incorporated into all council deliberations, paper sections, and experiment designs.

## Insight 1: Pair-wise Analysis, Not Frame-wise

"We are talking about PAIRS here typically and not just frames."

PoseNet evaluates non-overlapping consecutive pairs (frame_2k, frame_2k+1). There are 600 pairs in the test video. ALL difficulty analysis, TTO optimization, and renderer evaluation must be pair-centric, not frame-centric. A "hard frame" is meaningless in isolation — what matters is a "hard pair" where the ego-motion estimation between two consecutive frames is difficult to preserve through compression.

## Insight 2: Mathematical Precision on Hard Pairs

"We can mathematically determine and model the hard vs easy pairs prior using sophisticated and mathematical analysis."
"We can be extremely precise and optimal here."

This is not hand-waving. The camera model is KNOWN (AR0231AT, fx=910px, pp=(582,437)@1164x874, 20fps). The ego-motion for each pair is KNOWN (PoseNet outputs 6-DOF). The mask topology for each pair is KNOWN (SegNet 5-class output). From these, we can:

1. Derive expected optical flow fields analytically from the 6-DOF pose
2. Compute pair difficulty metrics from mask topology (boundary density, rare class prevalence, temporal transitions)
3. Predict PoseNet sensitivity per-pair without running PoseNet (geometric analysis of flow magnitude vs texture structure)
4. Pre-compute ALL of this at compress time (unlimited compute) and store as compact metadata in the archive

The mathematical framework should be rigorous enough for a CVPR systems paper.

## Insight 3: Optimal Time Budget Allocation

"We must allocate our usage of time within the 30 minute budget optimally."

The 30-minute contest budget is a resource to be optimized, not just a constraint to satisfy. Currently the renderer uses only 186s of the 1522s available. That's 12% utilization of a scarce resource. Even under the strict scorer rule (no scorers at inflate time), we should explore what useful compute we can do in the remaining 1336s WITHOUT loading scorers.

Possibilities:
- Multi-pass rendering (render, measure self-consistency, re-render hard regions)
- Ensemble of renderer variants (run 3 architectures, blend outputs)
- Iterative refinement using only the renderer's own gradients (no scorer needed)
- Resolution-adaptive rendering (high-res for hard pairs, standard for easy)

## Insight 4: Adaptive Behavior Engineering

"Perhaps we can engineer adaptive behavior in."

The inflate script should be SMART about how it spends its time budget. Not a fixed pipeline but an adaptive one that:
- Measures its own elapsed time
- Adjusts strategy based on remaining budget
- Allocates more compute to pairs it expects to be hard (from pre-computed metadata)
- Gracefully degrades if running out of time

## Insight 5: Comma Production Relevance

"We want to recall what the optimal production config and deployment would be for comma ai's training pipelines and on device and everything."

The paper and site should connect our techniques to comma's actual production pipeline:
- Fleet data compression for upload bandwidth
- Training data pipeline efficiency
- On-device decoder for visualization
- Asymmetric compute model (heavy server-side compress, light edge inflate)

## Insight 6: Strict Scorer Rule (Canonical)

"I agree with the strict reading personally, record that as the canonical answer."

NO scorers at inflate time. Period. This is a binding decision. All contest-compliant work must operate without PoseNet or SegNet at inflate time. TTO is compress-time only. Any inflate-time feature that loads scorers must be labeled "non-compliant" and off by default.
