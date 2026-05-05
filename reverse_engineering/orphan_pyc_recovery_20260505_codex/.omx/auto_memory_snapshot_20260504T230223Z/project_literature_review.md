---
name: Literature Review — Multi-Task Gradient Methods and Task-Aware Compression
description: Key papers for citations, novel techniques to implement, and our novelty claim for arXiv paper
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Multi-Task Gradient Conflict Methods (ranked by strength)

| Method | Paper | Key Idea | Implement? |
|--------|-------|----------|------------|
| GradNorm | Chen et al., ICML 2018 | Dynamic loss weight tuning by gradient norm balancing | Maybe |
| MGDA | Sener & Koltun, NeurIPS 2018 | Min-norm point in convex hull of task gradients | No (heavy) |
| PCGrad | Yu et al., NeurIPS 2020 | Project conflicting gradients perpendicular | Done (approx) |
| CAGrad | Liu et al., NeurIPS 2021 | Maximize worst-case local improvement | YES — strongest |
| Nash-MTL | Navon et al., ICML 2022 | Nash bargaining for proportional fairness | Maybe |
| Aligned-MTL | Senushkin et al., CVPR 2023 | Converge to specified Pareto point via weights | YES — fits us |
| FAMO | Liu et al., NeurIPS 2023 | O(1) overhead adaptive balancing | Backup option |

## Task-Aware Compression Prior Art

- **VCM MPEG Standard** (ISO/IEC 23888-2) — task-driven compression now standardized
- **Neural Wrapping** (Khan et al., CVPR 2025) — pre/post-processing around standard codec
  - Closest to our approach but targets human perceptual metrics, not frozen task networks
- **Sandwiched Compression** (Du et al., Google) — neural pre+post processor concept
- **Rate-Distortion-Perception Tradeoff** (Blau & Michaeli, ICML 2019) — fundamental impossibility

## Pareto Frontier Navigation

- **Pareto HyperNetworks** (Navon et al., ICLR 2021) — learn ENTIRE frontier with one hypernetwork
- **Controllable Pareto MTL** (Lin et al., 2020) — weighted Chebyshev scalarization

## Our Novelty Claim (for arXiv)

Training a CNN post-filter by directly backpropagating through frozen scorer networks
(PoseNet + SegNet) to minimize a specific scoring formula is a concrete instance of
"compression for machines" not yet documented in academic literature.

Distinguishing features:
1. Frozen perception networks as differentiable loss (not jointly trained)
2. Explicit multi-objective score formula with known coefficients
3. Pareto frontier analysis revealing antagonistic coupling
4. Non-opposing gradient method for PoseNet-SegNet decoupling
5. MRS-adaptive weights from first-order optimality on the Pareto frontier
6. Formal Lean 4 verification (even of vacuous results — honest science)
7. 25+ documented negative results

**Why:** These citations position our arXiv paper in the literature.
**How to apply:** Cite all in the paper. Implement CAGrad and Aligned-MTL as stronger alternatives.
