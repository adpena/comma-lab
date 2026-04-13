# Adversarial Council Sweep — 2026-04-13
## Tripartite Pact + Quantizr: Full Pipeline Review

### FIXED (10 bugs, all committed)

| # | Bug | File | Severity | Impact |
|---|-----|------|----------|--------|
| 1 | kill_loss_threshold=100 too low for Lagrangian Phase 2 | train script | CRITICAL | **Every run dies at epoch ~2700** |
| 2 | _full_eval evaluates stride-1 pairs; scorer uses stride-2 | train script | CRITICAL | **Best checkpoint selection wrong** |
| 3 | _full_eval doesn't pass explicit residual_scale=1.0 | train script | MEDIUM | Warmup eval measures wrong model |
| 4 | rho growth guard > 0 instead of > 1e-6 | train script | MEDIUM | rho never stops growing |
| 5 | Modal template tv_weight 0.05 vs dataclass 0.1 | deploy | CRITICAL | Silent config drift |
| 6 | Modal template target_bytes 200000 vs dataclass 256000 | deploy | CRITICAL | Silent config drift |
| 7 | Flow normalization max(H,W) instead of per-axis | renderer.py | IMPORTANT | 25% y-flow underscale |
| 8 | MPS _manual_grid_sample extrapolates outside border | renderer.py | IMPORTANT | MPS/CUDA gradient divergence |
| 9 | residual_scale not initialized in dp_sims path | train script | LOW | Scoping fragility |
| 10 | Modal template log-every 25 vs default 50 | deploy | LOW | Doubled logging overhead |

### REMAINING (not yet fixed, lower priority)

| # | Bug | File | Severity | Notes |
|---|-----|------|----------|-------|
| 11 | inflate_renderer.py loads asymmetric .pt as DPSIMSRenderer | inflate | CRITICAL for deploy | Needs pair_mode dispatch |
| 12 | Phase 3 resume loads scheduler state into wrong T_max | train script | MEDIUM | Only affects Phase 3 resume |
| 13 | auth_eval can't load .bin exports (only .pt) | deploy | MEDIUM | Blocks .bin eval before submission |
| 14 | _full_eval skips 1164x874 upscale that auth_eval does | train script | LOW | Minor metric divergence |
| 15 | warp_quality telemetry warps wrong direction | train script | LOW | Corrupted diagnostic |
| 16 | HintedPairGenerator doesn't guard against 6-ch motion predictor | renderer.py | LOW | Runtime error if misused |
| 17 | make_coord_grid cache key uses torch.device object | renderer.py | LOW | Potential cache misses |

### Previously caught (3 bugs from round 1)

| # | Bug | Status |
|---|-----|--------|
| A | residual_scale added to wrong class (HintedPairGenerator not AsymmetricPairGenerator) | FIXED |
| B | Resume loads explosive Lagrangian state without clamping | FIXED |
| C | Modal deploy template overrides all council fixes with old values | FIXED |

### Total: 13 bugs fixed across 2 review rounds. 7 remaining (lower priority).
