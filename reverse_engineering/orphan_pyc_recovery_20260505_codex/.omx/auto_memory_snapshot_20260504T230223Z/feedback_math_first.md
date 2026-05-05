---
name: Mathematical Analysis First — No Arbitrary Designs
description: Every design decision must be derived from data statistics and information theory, not guessed.
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
The user strongly emphasizes: start with mathematical analysis, then implement.
Multiple means of finding optimal exist, but mathematical analysis is the foundation.

## Applied to entropy coder design:
- Measure actual class distribution from our masks (not estimated)
- Compute actual spatial autocorrelation (pixel-to-neighbor conditional probability)
- Compute actual temporal redundancy (frame-to-frame change rate)
- Derive the Shannon limit from measured statistics
- Design the coder to approach that limit
- Every parameter (block size, context window, etc.) should be derived from the data

## Applied to all architecture decisions:
- Channel widths: derived from capacity analysis, not round numbers
- Learning rates: derived from loss landscape curvature
- Loss weights: derived from Pareto MRS condition (we already do this)
- Quantization levels: derived from weight distribution statistics

## Applied to scorer optimization:
- The scorer uses only 6 PoseNet outputs and last-frame SegNet argmax
- Every design choice should exploit this specific structure
- Byte accounting is analytically separable from distortion
- Parameter optimization is nonconvex and partly discrete

## What "no arbitrariness" means:
- No magic numbers without justification
- No "let's try 64 channels because it's a nice power of 2"
- Every hyperparameter should have a derivation or at minimum a sensitivity analysis
- If a choice is arbitrary, document WHY it's arbitrary and what the alternatives are

**Why:** Arbitrary choices compound into suboptimal systems. Mathematical grounding
ensures every component is pulling in the right direction.
**How to apply:** Before implementing, derive. Before training, analyze. Before deploying, verify.
