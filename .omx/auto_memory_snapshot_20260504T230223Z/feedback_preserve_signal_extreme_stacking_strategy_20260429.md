---
name: Strategy correction — preserve signal, extreme stacking, multi-pass pipeline
description: 2026-04-29 PM. User corrects codex aggressive-kill-list framing. Quantizr + Selfcomp left the door open: unlimited-compute compress time + multi-pass pre/during/post pipeline techniques + extreme stacking are our edge. Don't kill EUREKA lanes — they preserve signal for stacking.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Rule**: do NOT aggressively kill lanes. Preserve signal for extreme stacking and optimization opportunities.

**Why**:
- Quantizr (0.33) and Selfcomp (0.38) are in the same architectural family (FiLM-like / analytical-pose / small-renderer / block-FP-style weight compression).
- Both leveraged the family's basics — but neither pursued:
  1. **Extreme stacking** (combining multiple orthogonal techniques per archive)
  2. **Unlimited-compute compress time** (the contest only constrains inflate to 30 min on a single GPU; compress is unbounded)
  3. **Multi-pass pre/during/post pipeline techniques** (pre = data preparation/synthesis, during = training-time tricks, post = encoder-side optimization on the trained checkpoint)
- These are the OPEN DOORS where our deep mathematical + research + labwork advantage compounds.

**How to apply**:
- Keep EUREKA lanes (PA/FC/SH/TR/PD/AL) ALIVE. They are signal preservation, not primary EV.
- Codex's recent "demote AL/FC/PA to bolt-ons" framing is correct ONLY for budget-allocation purposes (don't make them the sole primary lanes), NOT for kill purposes (still dispatch as cheap probes).
- Continue iterating on extreme-stacking compositions (e.g. SC++ + SH + PD + FR-Ω + AL + archive-diet stack).
- Pursue multi-pass pre/during/post pipeline:
  - **Pre-compress**: Lane PA pose-as-affine init from frozen PoseNet. Lane MAE-V mask augmentation. Lane SAUG self-augmentation. Pre-compute optimal grayscale via SGD before encoding (Lane AL).
  - **During-compress**: SC++ training with KL distill T=2.0 + eval_roundtrip + EMA + curator-outlier-pair-weights (Lane WC-S).
  - **Post-compress**: SH arithmetic on qints, PD pose deltas, FR-Ω Fridrich-cost block-FP, archive diet.
- Recommit to compute-budget elasticity: don't artificially constrain to $150 if $300-500 nets more lanes.

**Strategic framing**:
- Quantizr stopped optimizing at 0.33 with his own admission "sub 0.30 possible just by sweeping conv dims, but I stopped". That door is OPEN.
- Selfcomp at 0.38 admitted "underfit segnet due to no architecture search" + "more can be gained on weight self-compression". Doors OPEN.
- We have: 80 STRICT preflight checks (engineering rigor neither has), Modal-trusted pipeline, 78+ memory files of accumulated research, 14 dispatched lanes, tactical infrastructure (review_tracker, lane_class_proofs, contest-CUDA auth eval, council protocol).

**Counter-evidence to codex's "kill list"**:
- Codex's "any archive >350KB" rule is correct per Pareto, but doesn't mean kill techniques that produce <350KB archives. Multi-pass post-compress can shrink 350KB → 250KB without retraining.
- Codex's "EUREKA only as bolt-ons" is correct for primary-budget — but bolt-ons compose. SH+PD+FR-Ω+AL stacked could exceed any single new training lane.

Cross-refs:
- project_codex_theoretical_floor_brutal_20260429 (codex floor verdict, somewhat-aggressive kill list)
- project_grand_council_brutal_forecast_20260429 (council kill list, also aggressive)
- project_selfcomp_portfolio_tonight_20260429 (full 14-lane portfolio)
- project_council_stacking_lanes_to_03_20260429 (extreme stacking proposal)
