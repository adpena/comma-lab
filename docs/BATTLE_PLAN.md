# Battle Plan: April 23 - May 3 (10 days remaining)

## CRITICAL: ALL scores before April 21 were INVALID
Measurement bugs (48x64 masks, wrong archive, overlapping pairs, eval_roundtrip=False) invalidated every prior score. First honest measurement: 2.01 on April 21.

## Current State

### Leaderboard
| Rank | Score | Submission |
|------|-------|-----------|
| 1 | **0.33** | quantizr (PR#55) |
| 2 | 0.60 | mask2mask (PR#53) |
| 3 | 1.89 | neural_inflate (PR#49) |
| — | **2.03** | Our best (MPS, CRF50 matched poses) |

### Running Experiments (2 × RTX 4090, ~$0.48/hr combined)
| Instance | Config | Phase | Best |
|----------|--------|-------|------|
| #2 (ssh6:10476) | Float + EMA, no Fridrich | Phase 2 | ~1.15 proxy |
| #3 (ssh5:17296) | Float + EMA + Fridrich | Phase 2 | ~1.00 proxy |

Both: 103K params, DSConv, FiLM, CLADE, CRF50 masks, eval_roundtrip, hinge, EMA 0.997.

### Key Findings (April 21-23)
1. **QAT from epoch 0 is 8.7x slower than float** — proxy 0.665 vs 0.644 at equivalent epochs. Train float, quantize post-hoc. (Confirmed by head-to-head)
2. **Fridrich losses help Phase 1 by 5-7%** but may compete in Phase 2. Phase-specific application recommended.
3. **Difficulty map: top 20% pairs are 227x worse** — fixing them via postfilter = 0.932 net point improvement.
4. **Poses MUST match archive masks** — CRF50 poses on CRF56 masks = 27x PoseNet regression.
5. **EMA was built but never wired** — now wired into all phases + export.
6. **Hybrid CG is DEAD** — contest rules require scorer weights in archive (90MB = catastrophic rate).

## After Training Completes (~8h)

1. Take the winner (lower proxy)
2. Export ASYM (130KB)
3. Pose TTO with `--masks` CRF50 (~1h)
4. Postfilter training against hard pairs (~2h, curriculum: all→top 30%→top 10%)
5. Build archive: renderer(130KB) + postfilter(46KB) + masks(421KB) + poses(16KB) = 613KB
6. Full e2e eval: inflate_renderer.py → upstream evaluate.py on Modal T4
7. Ship as submission

## Score Projection
| Config | Distortion | Rate | Total |
|--------|-----------|------|-------|
| Winner + poses (no postfilter) | ~0.8-1.2 | 0.37 | ~1.2-1.6 |
| Winner + poses + postfilter | ~0.3-0.6 | 0.41 | ~0.7-1.0 |
| Next iteration + half-frame masks | ~0.3-0.5 | 0.28 | ~0.6-0.8 |

## Next Training Iteration (if time, 9 days left)
- Fridrich Phase 1+3 only (off in Phase 2)
- KL distillation (T=2.0) as auxiliary SegNet loss in Phase 2
- Half-frame masks (600 odd → ~210KB)
- Smaller model (75K params → ~100KB ASYM)
- Post-hoc FP4 comparison

## Budget
- Vast.ai: ~$8 spent, ~$16 remaining of $24
- Modal: fresh monthly credits
- Deadline: May 3, 11:59 PM AOE
