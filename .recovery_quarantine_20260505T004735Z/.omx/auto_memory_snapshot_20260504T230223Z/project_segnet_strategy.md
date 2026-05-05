---
name: SegNet Attack Strategy
description: Council-recommended approach to close 0.147 SegNet gap — KL distillation, temperature, boundary weighting
type: project
---

SegNet contributes 38.9% of our score (0.581 out of 1.49). neural_inflate has 0.434 — 0.147 better.
Closing this gap = 0.15 score improvement, the single biggest opportunity.

**Implemented approaches (tac v0.8.0):**
1. `kl_distill` loss: Hinton-style KL divergence on temperature-softened SegNet distributions.
   T anneals 5.0→1.0. Boundary-weighted (10× at class boundaries). T² scaling.
   Needs --temperature-start 5.0 --temperature-end 1.0 (pydantic enforces this).
2. `temperature` loss: soft-cosine with temperature annealing (old approach, less principled)
3. `focal_ste` loss: focal cross-entropy with STE for hard argmax training
4. Boundary mask: compute_boundary_mask with fixed preprocess_input + T dimension

**Council diagnosis (Hinton):** Match full class probability distribution, not just argmax.
At high T, SegNet reveals inter-class relationships. Model learns WHERE to push pixels.
At low T, focuses on flipping specific boundary pixels. Need 500+ epochs minimum.

**Status:** kl_distill is implemented, reviewed, bug-fixed, validated with pydantic.
Smoke-tested locally (15 epochs). Ready for nuclear Modal run.

**Why:** SegNet is the largest score component and has 98.4% headroom remaining.
**How to apply:** Deploy kl_distill on H100 as the nuclear run. Don't run locally — needs 500+ epochs to show results.
