---
name: Writeup Story Arc and Demo Ideas
description: The Pact narrative, discovery arc, visualization ideas, and publishable findings for paper/site
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## The Story: "The Pact"

A human and an AI system made a pact to push the boundaries of neural video compression.
The project name IS the story.

## Discovery Arc (each step was a paradigm shift)
1. Codec tweaks (2.16) — AV1/H265 parameter optimization
2. Dilated postfilter (1.33) — first neural component, 5.6x PoseNet gain
3. Asymmetric warp renderer (1.00) — frame2=render, frame1=warp+gate*residual
4. Lagrangian annealing (0.87) — brief λ reduction found better PoseNet basin
5. TTO warm-start v1 (0.59 proxy) — renderer→gradient descent against scorers
6. Embedding loss v3 (RUNNING) — rank-512 gradient breaks rank-6 bottleneck

## Publishable Findings

### 1. Gradient Rank Bottleneck (Novel)
PoseNet output MSE provides rank-6 gradient (6 pose dimensions) in a 589K-pixel space.
Embedding MSE provides rank-512. This is a general result for ANY test-time optimization
against a neural scorer — the gradient rank of the loss determines the effective
degrees of freedom for pixel-level optimization.

### 2. Inverse Steganalysis Framing (Novel connection)
Contest = inverse steganalysis. Embed (masks, ego-motion) into cover (frames) such that
detector (scorer) cannot distinguish from original. Connects 30 years of information
hiding research to neural video compression. Fridrich+Yousfi (DDELab) designed both
the steganalysis detectors AND this contest.

### 3. Lagrangian Annealing Transient (Novel)
Reducing constraint strength on a converged model creates a brief transient where the
model finds a better trade-off on the Pareto frontier. The optimal strategy: perturb→
snapshot→discard late epochs. This is simulated annealing for Lagrangian multipliers.

### 4. Skunkworks Council Methodology (Novel)
13 virtual domain experts with structured adversarial deliberation. Recursive review
protocol (3 consecutive clean passes). Cross-disciplinary integration at scale.

## Demo/Visualization Ideas

### The Drift Curve (beautiful data)
Animate the 11-checkpoint auth eval sweep (ep12500→16999) showing the model drifting
under weakened Lagrangian. U-shaped curve with two minima. Interactive D3 chart.

### TTO Before/After
Side-by-side video: renderer frames vs TTO-refined frames. PoseNet heat map overlay
showing where TTO concentrated changes. Show the 6 gradient directions from output MSE
vs the 512 from embedding loss as a dimensionality reduction visualization.

### Gradient Rank Visualization (paper figure)
PCA of the Jacobian: 6 principal components from output MSE vs 512 from embedding loss.
Show how the pixel-space gradient is constrained to a low-dimensional subspace.
This would be THE figure for the paper.

### Score Timeline
Git history IS the research timeline. Gource video of the repo evolution.
D3 score timeline: every auth eval plotted chronologically with architecture labels.

## Shannon's Shower Thought (implement)
The scoring formula 100*seg + sqrt(10*pose) + 25*rate has KNOWN multipliers.
The optimal allocation of model capacity between SegNet, PoseNet, and rate is a
convex optimization in 3 variables. Solve analytically and visualize on the site.

## Fridrich's Shower Thought (implement)
DALI decode divergence (29x PoseNet) reveals DECODER FINGERPRINT sensitivity.
Adding noise during TTO could make frames robust to decoder variation — more
natural-looking to PoseNet. Test with Gaussian noise augmentation in TTO v4.
