# Battle Plan: April 25 - May 3 (8 days remaining)

## Current State

### Leaderboard
| Rank | Score | Submission | Architecture |
|------|-------|-----------|-------------|
| 1 | **0.33** | Quantizr (PR#55) | JointFrameGenerator, 87K, GroupNorm, 5-stage |
| 2 | 0.36* | szabolcs-cs (PR#56) | Shared latent + affine, block FP 1.017 bpw |
| 3 | 1.89 | neural_inflate (PR#49) | — |
| — | **0.407** | Our proxy (Phase 2 best) | AsymWarp 103K, CLADE, DSConv, FiLM |

*Not yet CI-evaluated

### Session 2026-04-24 Results (45 commits)
- Proxy: 2.01 → **0.407** (4.9x improvement)
- Training: 3000 epochs complete on RTX 4090 (Phase 1 + Phase 2 + Phase 3)
- Post-training pipeline running (pose TTO in progress)
- Radial zoom validated: 10.6x PoseNet improvement from 600 scalars
- eval_roundtrip fixed across 15 files (3 distinct bug patterns)
- HWC/CHW mismatch found and fixed (50x scorer inflation)

### Key Discoveries
1. **PoseNet rank-1**: Jacobian effective rank 1.008. Dim 0 = 99.8% variance. Radial zoom from FoE is optimal.
2. **SegNet is 77x more important than PoseNet** at our operating point. SegNet gap vs Quantizr is 5.7x.
3. **Masks = 66-89% of archive**. CRF50 is optimal. Rate lever is model compression, not mask compression.
4. **Int4+LZMA2**: 2.18 bits/weight (27.5KB vs 64.5KB FP4). Council-approved format.
5. **Lane marking displacement** encodes vehicle speed. Zero-archive-cost motion estimation.
6. **Fridrich losses** (UNIWARD, L-inf, Markov) are our competitive edge — no competitor uses steganalytic shaping.

## Experiments Queued

### Iteration 1: WILDE vs SHIRAZ (A/B test)
| | WILDE | SHIRAZ |
|---|---|---|
| Architecture | 32/48, 181K params | Same |
| Strategy | Freeze/unfreeze + error_boost 9→49x | Focal STE γ=2, continuous |
| Unique | Quantizr-adapted brute force | Principled gradient surgery |
| GPU | A100 (~20h, ~$13) | A100 (~20h, ~$13) |
| Both have | CLADE, DSConv, dilation, replicate pad, Fridrich, per-class weights, SWA, EMA |

### Iteration 2: GREEN
- Winner's recipe + zoom-aware 4ch MotionPredictor (output_channels=4)
- Zoom handles flow (rank-1), MotionPredictor handles gate+residual only
- Int4+LZMA2 export for smallest archive

### Post-Training Pipeline (per model)
1. Export → 2. Pose TTO → 3. QAT → 4. Fridrich refinement → 5. Weight compression → 6. Archive build → 7. Auth eval

## Timeline

| Day | Action |
|-----|--------|
| Apr 25 (today) | Pose TTO + auth eval on current model. Deploy WILDE + SHIRAZ. |
| Apr 26 | WILDE/SHIRAZ training runs (~20h each on A100) |
| Apr 27 | Results. Winner → post-training pipeline → auth eval |
| Apr 28 | GREEN (iteration 2) on A100. ~20h. |
| Apr 29 | GREEN results. Best model → final polish |
| Apr 30 | Multi-pass Fridrich. Int4+LZMA2. Final archive optimization. |
| May 1 | Auth eval on final model. Paper/writeup finalization. |
| May 2 | Submission PR preparation. 5-pass adversarial council review. |
| May 3 | **DEADLINE.** Submit. |

## Budget
- Vast.ai: ~$4 spent, ~$20 remaining of $24 (replenishing today)
- A100: $0.67/hr, 4090: $0.27/hr
- Two A100 runs (WILDE + SHIRAZ): ~$26 → needs top-up
- H100 ($2/hr): reserved for nuclear final run if needed

## Paper & Portfolio
- docs/paper/*.md — updated with rank-1 discovery, Fridrich framing, competitive analysis
- README.md — rewritten for OSS release
- Quarto → arXiv PDF + HTML site
- marimo notebook — interactive visualizations with real data
- Cloudflare Pages — landing page (byhand.ai aesthetic)

## openpilot Integration (Paper Section + Future Work)
- openpilot supercombo model at compress time → lane detection, path planning, calibration
- Contest-compliant: unlimited compute at compress time
- Demo: openpilot-derived zoom scalars match gradient-optimized zoom scalars
- Production story: compress → deploy → collect new data → retrain → compress
- Continual learning feedback loop for fleet-wide compression
- Paper section: "From competition to production" (docs/paper/05_production.md)

## Score Projections

| Scenario | SegNet | PoseNet | Rate | Total |
|----------|--------|---------|------|-------|
| Current model + FP4 | 0.330 | 0.080 | 0.175 | **0.585** |
| WILDE (conservative) | 0.200 | 0.100 | 0.180 | **0.480** |
| WILDE (aggressive) | 0.050 | 0.070 | 0.180 | **0.300** |
| GREEN + int4+LZMA2 | 0.050 | 0.060 | 0.135 | **0.245** |
| Quantizr | 0.061 | 0.072 | 0.200 | **0.333** |

## Non-Negotiable Rules (from CLAUDE.md)
- Auth score > 1.0 is UNACCEPTABLE
- eval_roundtrip EVERYWHERE, ZERO exceptions
- NO scorers at inflate time
- Full e2e pipeline test before ANY score claim
- 5 consecutive adversarial council passes before submission PR
