---
name: Honest State of Play — Apr 11 Evening
description: Grand council comprehensive accounting. What's real vs what's built but unproven.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## What's REAL (proven, auth-verified, working)
- CPU lane auth score: 1.33 (dilated h=64, CRF 34)
- CRF 35 retrain converging: proxy 1.493 at epoch 189
- auth_eval.sh zero-pollution tool
- 144 passing tests (CPU lane)
- Paper 85% complete for CPU story

## What's PROMISING (training, converging, but unproven)
- CRF 35 retrain → projected auth ~1.27-1.35 when converged
- Renderer Phase 1 learning (loss 0.415→0.037 in 10 epochs)
- Renderer Phase 2 (scorer loss) NOT YET STARTED

## What's BUILT but NEVER TRAINED (4,691 lines, zero tests)
- WaveletRenderer (137K params, 376 lines)
- DP-SIMS (1-3.7M params, 778 lines)
- DiffusionRenderer + distillation (2.5-9.5M params, 1192 lines)
- VQ-VAE codec (955K params, 880 lines)
- MLX renderer port (312K params, 898 lines)

## What's BROKEN or STALE
- GPU lane inflate path: NOT validated end-to-end
- GPU lane test coverage: ZERO
- FP4 QAT: fixed but never verified with a training run
- Renderer was killed (running stale code with broken QAT)
- CRF 36 retrain: very slow convergence, may never beat 1.33

## Council's 24-hour plan
1. Let CRF retrains run uninterrupted (CPU lane, proven)
2. Auth eval CRF 35 at epoch 300 (~5 hours from now)
3. DO NOT start new architectures until MaskRenderer Phase 2 produces a score
4. Write GPU lane paper section as "in-progress/future work"
5. Fix site status.json (DONE)
6. Write 10-15 focused tests for renderer + mask_codec + fp4

## Council verdicts on untrained architectures
- Diffusion: KILL (2-week minimum project)
- VQ-VAE: KILL (codec redesign, not drop-in)
- Wavelet: QUEUE (ablation after MaskRenderer proves viability)
- DP-SIMS: QUEUE (replacement if MaskRenderer fails)

**Why:** Honest accounting prevents wishful thinking.
**How to apply:** Focus on what's converging (CRF 35). Don't touch what's working.
