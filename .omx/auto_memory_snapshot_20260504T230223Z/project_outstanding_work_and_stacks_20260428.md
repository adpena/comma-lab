---
name: 2026-04-28 outstanding work catalog + stack composition strategy
description: Comprehensive recall of every lane experiment + outstanding deployment/review work + 5 candidate stacks (conservative → moonshot) for sub-Quantizr 0.33 score. Filed during overnight wave per user request to prioritize undeployed lanes.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Session start state (2026-04-27 morning)**: Lane A 1.15 frontier, 14 strict preflight checks, 1 lane script.

**Session current state (2026-04-28 ~10:30 UTC)**: 36 strict preflight checks, 22+ lane scripts written, 8+ subagents spawned for hardening, 7 codex review rounds, 14+ HIGH/MEDIUM bugs found+fixed.

# Lane Inventory (every experiment + status)

## TIER 1 — Validated landed (truth in hand)
- **Lane A 1.15** — frontier; pose TTO warm-start from baseline poses
- **Lane B-alt 1.146** — brotli compression of archive (-0.0037)
- **Lane F V1 2.73, V2 1.79, V3 1.85** — FP4 architecturally hostile to dilated-h64 ASYM (20× PoseNet penalty); chapter closed
- **Lane M+N V1 2.35** — radial-zoom 1-DOF wrong subspace; padding-with-zeros bug
- **Lane D V1** — killed @62%, plateau (LR starvation hypothesis)

## TIER 2 — In-flight on Vast.ai (overnight wave)
- **Lane G v3** — KL distill weight=0.002 (post-fix); pose TTO retry; predicted [1.10, 1.18]
- **Lane I** — Cool-Chic CCh1 renderer-replacement; predicted [0.95, 1.30]
- **Lane V** — Quantizr replica 88K + half-frame from epoch 0 + KL distill; predicted [0.50, 1.10] BIGGEST SWING; ~12h
- **Lane M-V2 v2** — radial-zoom proper baseline-padded 6-DOF; predicted [1.10, 1.30]
- **Lane W** (just-relaunched Iceland) — per-pair-weighted Self-Compression; predicted [0.85, 1.05]
- **Lane K** (just-relaunched Denmark) — DSConv 88K from-scratch; predicted [0.85, 1.10]; ~12h
- **Lane OS-V2** (just-relaunched NC) — openpilot supercombo seeding V2; predicted [0.95, 1.10]

## TIER 3 — Code shipped, never deployed (highest-priority dispatch queue)
- **Lane S** — per-channel SC + Lagrangian; CRASHED on motion.head 6-vs-4 mismatch; needs profile fix + redeploy
- **Lane Ω-V2** — Lagrangian-learnable per-element bits; survived 7 codex rounds; READY TO DISPATCH
- **Lane Ω-V1** — water-fill heuristic version; superseded by V2
- **Lane SI-V2** — Lagrangian-learnable saliency threshold for masks
- **Lane PS-V2** — Lagrangian-learnable per-class SegNet weights
- **Lane LR-V2** — learnable-rank LoRA pose
- **Lane LM-V2** — endpoint-tracking lane-mark zero-cost poses (V1 corr=0.017, V2 should hit 0.30+)
- **Lane V-V2** — annealed half-frame (mask_half_sim_prob 0→1)
- **Lane RM** — Riemannian SE(3) pose optimizer; predicted [1.05, 1.15]
- **Lane GH** — Ghost-module renderer; predicted [1.05, 1.30]
- **Lane GH-DARTS / Lane K-DARTS / Lane I-DARTS** — architecture search variants
- **Lane SZ Phase 2** — szabolcs no-masks paradigm; predicted [0.30, 0.50] MOONSHOT
- **Lane G v3-V2** — KL SNR-target Lagrangian
- **Lane Ω-V3** — rate-frontier sweep across 5 budgets
- **Lane S-V2** — auto-warmup via convergence detection
- **Lane W-V2** — continuous Lagrangian per-pair weights
- **Lane F-V4** — mixed-precision FP4 + per-layer sensitivity (in-flight subagent)
- **Lane M-V3** — PoseNet-embedding-space pose distillation (in-flight subagent)
- **Lane D-V3** — half-frame validation + KL distill + annealed (in-flight subagent)
- **Lane MOS** — TODO: combine M-V3 PoseNet-embedding + OS supercombo seeding (user-suggested today)

## TIER 4 — Bayesian sweep framework + canonical examples
- **Lane A-Sweep** — Optuna over pose TTO hyperparams
- **Lane QAT-Sweep** — Optuna over FP4 QAT schedule

# Stack Composition Strategy

Lane A's score 1.15 = pose 0.22 + seg 0.46 + rate 0.46. Distortion at architectural floor, rate is 60% of remaining wedge. Stacks that COMPOSE orthogonally:

## Stack A — Conservative (high confidence, ~$3 deploy)
- Lane A renderer (anchor)
- Lane G v3 (KL distill aux for SegNet polish)
- Lane LR (LoRA pose -0.005 rate)
- Lane LM-V2 (zero-cost pose dim 0, -0.010 rate, V2 fixes calibration)
- Lane SI-V2 (Lagrangian saliency mask, -0.05 rate)
- **Predicted total: ~1.08 [contest-CUDA]**

## Stack B — Aggressive Rate (medium confidence, ~$5 deploy)
- Lane S/W/Ω-V2 (per-channel/element SC on Lane A renderer, -0.10 rate)
- Lane I (Cool-Chic CCh1 mask compression, -0.15 rate)
- Lane LR + LM-V2 (pose savings)
- **Predicted total: ~0.88 [contest-CUDA]**

## Stack C — Architectural (medium-low confidence, ~$15 deploy)
- Lane V (Quantizr replica 88K + half-frame from epoch 0)
- Lane W applied to Lane V's checkpoint (per-pair-weighted SC)
- Lane LM-V2 zero-cost poses
- **Predicted total: ~0.45 [contest-CUDA]** if Lane V lands in lower band

## Stack D — Moonshot (low confidence, paper-worthy, ~$5 deploy)
- Lane SZ Phase 2 (szabolcs no-masks paradigm)
- Lane LM-V2 zero-cost poses (the szabolcs archive lacks poses anyway)
- **Predicted total: 0.30-0.50 [contest-CUDA]** if SZ replica works

## Stack E — Lane MOS (user-suggested 2026-04-28, ~$0.80 deploy)
- Lane OS supercombo pose seeding (replaces baseline-pose init)
- Lane M-V3 PoseNet-embedding distillation (compress pose info)
- Lane LR LoRA on the distilled poses
- **Predicted total: ~1.10 [contest-CUDA]** (incremental, low risk)

# Composition Rules (orthogonality matrix)

Independent (compose multiplicatively):
- Renderer compression (S/W/Ω-V2) × Mask compression (I/SI) × Pose compression (LR/LM)
- All three are orthogonal axes

Mutually exclusive (pick one):
- Architecture: dilated-h64 (Lane A baseline) OR DSConv-88K (K) OR Cool-Chic (I) OR Ghost (GH) OR Quantizr-replica (V) OR szabolcs (SZ)
- Quantization scheme: per-tensor (FP4 dead) OR per-channel SC (S) OR hard-pair-weighted SC (W) OR per-weight Hessian (Ω)

# Deployment Priority Order (after current overnight wave)

1. **Lane Ω-V2** — math-validated through 7 codex rounds, never deployed
2. **Lane S V2** — fix motion.head profile, redeploy
3. **Lane MOS** — combine M-V3 + OS-V2 (user-suggested)
4. **Lane SI-V2** — Lagrangian saliency, paired with whatever renderer wins overnight
5. **Lane LR-V2** — pair with whichever renderer wins overnight
6. **Lane SZ Phase 2** — moonshot if other stacks underperform

**Council consensus**: Stack B (aggressive rate composition) is the highest-EV path that doesn't require Lane V succeeding. Stack C is the swing bet.

# Related memories (cross-references)
- `project_lane_taxonomy_stacking_strategy_20260427` — earlier session taxonomy
- `project_arbitrariness_audit_full_catalog_20260427` — heuristic→learnable plan
- `project_lane_omega_bit_budget_hessian_aware_quantization` — Lane Ω design
- `project_lane_w_hard_pair_self_compress_premise_20260427` — Lane W premise
- `project_szabolcs_full_re_20260426` — Lane SZ source
- `feedback_lane_s_motion_head_shape_mismatch_20260428` — Lane S bug to fix
- `feedback_vastai_nvdec_host_variation` — NVDEC failure pattern (7/12 tonight)
- `feedback_metabug_checks_30_31_32_added_20260427` — preflight progression
