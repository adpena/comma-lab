---
name: 6-MONTH STRATEGIC PLAN — Phase 1-4, target floor 0.12-0.20 (moonshot <0.12)
description: 2026-04-29 PM. Master codex grand council 22-voice strategic recheck under 6-month team-parallel + post-contest budget. 4-phase plan with 23 lanes total (8 contest deadline + 15 post-contest). Floor target: Phase 4 central 0.12-0.20, moonshot <0.12. Section 3 highest-EV next 24h: Hybrid AV1+residual STC smoke (already executed; CPU result invalid, needs CUDA SegNet validation). Compress-time-unlimited insight: BUY scorer-margin reductions and representation simplification first, byte polish later.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Phase plan

### Phase 1 (Days 0-7, contest deadline May 3)
Target: best contest-eligible archive. 8 lanes ranked by EV:
1. **proven baseline / compliance lock** (serial)
2. **archive diet pose-delta** (independent)
3. **STC hybrid smoke** (independent, codex top-1 redesign)
4. **water-filling Ω-W qint export** (independent after renderer anchor)
5. **LCT / learnable class targets** (dependent on mask/render path) — READY (8/8 tests)
6. **GP / radial pose / pose-space table** (dependent on renderer) — Lane GP fix-A FINALLY landed in commit 8746793e
7. **PSD standard only** (independent, high PoseNet risk)
8. **multi-pass inflate** (only if scorer-free or precomputed, within 30-min bound)

### Phase 2 (Weeks 2-4)
Target: full top-3 STC redesigns + Joint ADMM + wavelet
- Full STC boundary codec rebuild (one-majority-plus-exceptions structural bug fixed)
- Joint ADMM stream allocator (Boyd) — multi-objective rate/seg/pose/archive convex projection
- Wavelet residual codec (Mallat) — multi-scale sparsity
- NeRV mask codec (Cool-Chic / C3) — coordinate-MLP overfit on 1200-frame sequence
- DARTS-S full sweep (was restricted; expanded under budget)

### Phase 3 (Weeks 5-12)
Target: theoretical floor reduction research. Add post-contest extensions:
- multi-pass compress optimization (UNLIMITED encoder time)
- bit-level archive optimizer (gradient search over latents, quantization, class targets)
- Bayesian MDL / evidence analysis (MacKay + Schmidhuber)
- IMP / sparse qint model (full 10-20 cycle)
- RAFT / radial pose preimage lane
- SegNet logit-margin boundary fitting (use scorer at compress time per user direction)
- Ballé hyperprior residual codec
- Decoder systems rewrite + profiling

### Phase 4 (Weeks 13-24)
Target: **0.12-0.20 central, <0.12 moonshot**. Integration phase, not novelty accumulation:
- best semantic representation
- best pose representation
- best residual
- best entropy coder
- best quantization
- best distillation
Validation: three independent full reproductions, frozen evaluator container, ablation table, public archive scripts. Kill criteria: anything not contributing to integrated frontier or paper section.

Paper deliverables per phase:
- Phase 1: contest submission writeup
- Phase 2: empirical R(D) frontier and negative results
- Phase 3: theoretical floor analysis + Bayesian framework
- Phase 4: final paper, appendices, artifact checklist, negative-results catalog

## Score arithmetic (codex council math)

- Saving 1 KB → 0.000666 score gain
- Saving 15 KB → 0.01 score gain
- Saving 60 KB → 0.04 score gain
- SegNet -1e-4 → 0.01 score gain
- PoseNet -1e-4 (at 0.0005 baseline) → 0.007 score gain

**Therefore**: unlimited compute should buy SCORER-MARGIN reductions and representation simplification FIRST. The best use of a day is NOT saving 3KB — it's finding a representation that saves 50KB or reduces SegNet by 1e-4.

## "Compress time is unlimited" applied systematically

- 12-hour ADMM at compress time to equalize byte utility across streams
- Full RAFT or stronger optical flow over all 1200 frames → fit a compact motion model → store only parameters
- Run SegNet/PoseNet logit analysis to find margins; encode only fragile regions
- Train hyperpriors with rate-distortion loss
- Gradient search over latents, quantization assignments, class targets, even archive bytes
- Multi-pass: pre-encode → analyze residual → re-encode with informed prior → repeat

## Adversarial review scaling (6-month budget)

3-clean-pass review per landing is now MANDATORY. Three orthogonal axes:
- **Math**: objective, bounds, marginal rates, non-vacuous derivations
- **Engineering**: archive bytes, runtime, determinism, decoder dependencies, upstream purity, preflight coverage
- **Scientific rigor**: controls, proxy/auth gaps, stale artifacts, seed variance, hypothesis answered

Every landing must include: prediction, result, delta decomposition, manifest, kill decision, paper note.

## Highest-EV next 24h (codex council Section 3 verdict)

**Hybrid AV1+residual STC smoke** — predicted EV:
- 55% chance of usable positive result
- 25% chance of >0.03 contest-relevant score delta through 45KB+ net savings
- 15% chance reveals STC not worth near-term contest risk
- ~100% chance improves paper by producing real entropy/overhead table
- 6-month value: gates whether mask stream can plausibly fall under 80-120KB

ALREADY ATTEMPTED via CPU smoke (commit 5e8c7697); result INVALID due to AV1-decoded fallback as "clean" reference. Required: CUDA SegNet on GT video → re-run smoke with proper clean argmax. ~$0.20 Modal T4, ~10 min.

## Cross-refs

- /tmp/codex_runs/master_6month_strategic_recheck.log (full 7.6K-line transcript)
- /tmp/codex_runs/master_30day_design.log (24K-line per-lane math/impl/review)
- project_grand_council_final_designs_20260429.md (4-day council, still valid for Phase 1)
- project_stc_redesign_verdict_20260429.md (top-3 STC redesigns)
- project_lane_stc_av1_residual_smoke_prelim_20260429.md (smoke result, INVALID)
- feedback_budget_30_day_team_parallel_20260429.md (the budget pivot memory)
- reports/silent_defaults.md (246 CRITICAL silent-default overrides — Sherlock surfaced)
