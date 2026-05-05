---
name: Always Push Both CPU and GPU Lanes in Parallel
description: Non-negotiable — every session must advance both CPU postfilter and GPU renderer lanes simultaneously. Dual submission plan (fudgemallow).
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## The Rule

Both CPU and GPU lanes must be advancing at ALL times. If either lane goes quiet for more than a few turns, proactively bring it back up and propose the next action.

**Why:** The CPU lane (1.33, #2) is our floor — it ships regardless. The GPU lane (targeting sub-0.50) is how we catch Quantizr. Both feed into the dual submission plan ("fudgemallow" = best-scoring approach).

**How to apply:**
- Every session: check status of both lanes
- If one lane is idle, propose an experiment to launch
- Use the optimal pipeline: MLX Phase 1 locally (free), Modal A10G Phase 2 (paid), MPS for quick tests
- CPU lane: postfilter training, CRF sweep, TTO, ensemble, LSQ
- GPU lane: renderer training (MaskRenderer, DP-SIMS, wavelet), FP4 QAT
- Monitor both training logs for convergence signals

**Compute assignment:**
- M5 Max MLX → GPU lane Phase 1 (free, fast)
- Modal A10G → GPU lane Phase 2 + CPU lane serious runs
- Kaggle P100 → long overnight runs (either lane)
- Local MPS → quick smoke tests only
