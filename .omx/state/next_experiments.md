# next experiments — COMPREHENSIVE (no time constraints)

## promoted floor: 1.33 (dilated_h64)

## mandate: all techniques on merit, time is not a constraint

## immediate launches (parallel across all platforms)

### M5 Max (MPS/MLX, local)
1. TTO on current 1.33 checkpoint — zero risk, 30 seconds
2. GatedDilated h=64 smoke — `--profile gated_dilated_smoke`
3. Wavelet renderer Phase 1 via MLX — `--profile wavelet_renderer_smoke`

### Modal A10G (CUDA 24GB)
4. DP-SIMS smoke — `--profile dp_sims_smoke` (needs deploy script)
5. VQ-VAE smoke — `--profile vqvae_smoke` (needs deploy script)

### Kaggle P100 (CUDA 16GB, 30h/week)
6. Dilated h=32 — `--profile dilated_h32_smoke`
7. PairAware h=64 — needs profile creation

### Lightning T4 (CUDA 16GB)
8. PCGrad proven_baseline — `--profile pareto_pcgrad`
9. Extreme PoseNet — `--profile extreme_posenet`

## phase 2: full training (promote smoke winners)
- Best renderer -> 2500 epoch full on Modal
- Best postfilter variant -> 2500 epoch full on Kaggle
- Diffusion teacher smoke -> if any renderer shows promise

## phase 3: advanced
- Diffusion distillation
- Two-stage pipeline (renderer + postfilter)
- Per-class specialists
- Multi-resolution fusion
- Ensemble/SWA across architectures

## previously killed — now revived
- DP-SIMS (778 lines, complete)
- VQ-VAE (880 lines, complete)
- Diffusion teacher + distillation (1192 lines, complete)
- Wavelet renderer (376 lines, complete)
- TTO (348 lines, complete, never deployed)

## success criteria
- Any smoke test scoring < 1.30 proxy -> promote to full training
- Any full training scoring < 1.20 authoritative -> promote to submission
- Renderer variant scoring < 1.00 -> paradigm shift, all-in
