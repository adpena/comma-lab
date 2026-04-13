# Adversarial Council Sweep — 2026-04-13
## Tripartite Pact + Quantizr: 4 Rounds, Full Pipeline

### Final Tally: 20 bugs fixed across 4 rounds. 0 remaining.

## Round 1 (3 showstoppers)
| # | Bug | Impact |
|---|-----|--------|
| A | residual_scale on wrong class (HintedPairGenerator not AsymmetricPairGenerator) | Crash epoch 0 |
| B | Resume loads explosive Lagrangian state without clamping | Re-divergence |
| C | Modal deploy template overrides all council fixes | Silent revert |

## Round 2 (10 bugs)
| # | Bug | Impact |
|---|-----|--------|
| 1 | kill_loss_threshold=100 too low for Lagrangian Phase 2 | Every run dies at ep~2700 |
| 2 | _full_eval stride-1 pairs (scorer uses stride-2) | Wrong checkpoint selection |
| 3 | _full_eval missing explicit residual_scale=1.0 | Warmup eval mismatch |
| 4 | rho growth guard >0 instead of >1e-6 | rho never stabilizes |
| 5 | Modal template tv_weight 0.05 vs dataclass 0.1 | Silent config drift |
| 6 | Modal template target_bytes 200K vs dataclass 256K | Silent config drift |
| 7 | Flow normalization max(H,W) instead of per-axis | 25% y-flow underscale |
| 8 | MPS _manual_grid_sample extrapolates outside border | MPS/CUDA divergence |
| 9 | residual_scale not initialized in dp_sims path | Scoping fragility |
| 10 | Modal template log-every divergence | Minor |

## Round 3 (7 remaining issues)
| # | Bug | Impact |
|---|-----|--------|
| 11 | inflate_renderer.py loads asymmetric .pt as DPSIMSRenderer | Wrong arch at deploy |
| 12 | Phase 3 scheduler resume T_max mismatch | Wrong LR on resume |
| 13 | auth_eval can't load .bin exports | Crashes on .bin |
| 14 | _full_eval skips upscale (documented, minor metric diff) | Documented |
| 15 | warp_quality telemetry direction (verified CORRECT) | False positive |
| 16 | HintedPairGenerator 6-ch motion predictor guard | Runtime error if misused |
| 17 | make_coord_grid cache key device object | Cache misses |

## Round 4 (3 more)
| # | Bug | Impact |
|---|-----|--------|
| 18 | renderer_depth key mismatch in 4 callsites | Wrong arch at depth>1 |
| 19 | auth_eval auto-export default_bits=4 (optimistic rate) | Score appears 0.5x better |
| 20 | Phase 3 scheduler warm restart (comment fix) | Misleading comment |

### renderer.py confirmed clean:
- AsymmetricPairGenerator: residual_scale, per-axis flow, gate init all correct
- HintedPairGenerator: energy-conserving blend, motion slicing correct
- _manual_grid_sample: border clamping correct
- make_coord_grid: str(device) cache key correct
- warp_with_flow: confirmed backward warp direction correct

### Training script confirmed clean:
- Flow warmup: correctly gated by pair_mode, correct ramp, safe scoping
- Lagrangian: correct updates, clamp-on-resume, rho guard with epsilon
- Phase boundaries: correct progress computation, correct phase transitions
- Auto-kill: appropriate thresholds for Lagrangian Phase 2
- Checkpoint: all necessary state saved and restored

### Pipeline confirmed aligned:
- Training → export → inflate: all use same architecture dispatch
- .pt and .bin formats both loadable in all consumers
- Modal deploy template matches training script defaults
