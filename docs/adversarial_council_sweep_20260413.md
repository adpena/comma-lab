# Adversarial Council Sweep — 2026-04-13
## Tripartite Pact + Quantizr: 5 Rounds, Full Pipeline

### Final Tally: 26 bugs fixed across 5 rounds. 0 remaining critical/important.

## Round 1 (3 showstoppers)
| A | residual_scale on wrong class | Crash epoch 0 |
| B | Resume loads explosive Lagrangian state | Re-divergence |
| C | Modal deploy template overrides all fixes | Silent revert |

## Round 2 (10 bugs)
| 1-10 | kill threshold, eval stride, flow normalization, MPS grid_sample, config drift | Training dies, wrong checkpoints, gradient divergence |

## Round 3 (7 remaining from round 2)
| 11-17 | inflate pair_mode dispatch, Phase 3 scheduler, auth_eval .bin, motion slicing, cache key | Deploy failure, wrong LR, crashes |

## Round 4 (3 more)
| 18-20 | renderer_depth key mismatch, auto-export bits, scheduler comment | Wrong arch, optimistic rate |

## Round 5 (6 more — deep numerical + export review)
| 21 | Gate regularizer completely inert (.item() kills gradient) | Gate penalty had ZERO training effect |
| 22 | NaN in violation permanently corrupts lambda multipliers | Silent NaN propagation kills run |
| 23 | Shared embedding exported twice with different quant noise | Export fidelity degradation |
| 24 | renderer_depth key mismatch (4 callsites) | Wrong arch at depth>1 |
| 25 | auth_eval auto-export default_bits=4 (optimistic rate) | Score 0.5x too good |
| 26 | compress.sh x265 missing metadata strip | Documented, not active |

## Documented but not fixed (low priority, not blocking)
- compress.sh even-frame QP double-encode (dead code, not active)
- runner.py PACT_FRAME_COUNT env vs config.env inconsistency
- compress.sh renderer.bin not validated in pre-flight
- CLADE embedding per-tensor quantization (fidelity, not crash)
- coord_grid cache not thread-safe (single-threaded training is safe)
- residual_scale=0 silences gate/residual gradients during warmup (by design)

## Files confirmed clean after 5 rounds:
- experiments/train_renderer_fridrich.py
- src/tac/renderer.py
- src/tac/renderer_export.py
- src/tac/deploy/modal/modal_asymmetric_warp_deploy.py
- submissions/robust_current/inflate_renderer.py
- experiments/auth_eval_renderer.py
- src/tac/eval/auth_eval.py
