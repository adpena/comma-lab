---
name: Full lane sweep — Selfcomp stacking compatibility for 70 scripts
description: 2026-04-29 sweep of 70 remote_lane_*.sh + 50+ tac modules categorized by stack-EV with Selfcomp paradigm. 9 STRONG stackers, 7 neutral, 8 superseded. Lane FR-Ω (Fridrich-cost block-FP) + DARTS-S (arch search Selfcomp skipped) are highest-EV new additions.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
Codebase inventory: 70 `scripts/remote_lane_*.sh` scripts + 50+ `src/tac/*.py` modules. After Selfcomp 0.38 RE, swept all for stacking compatibility.

## 🟢 9 STRONG STACKERS (re-deploy on top of Selfcomp paradigm)

- **g_kldistill_pose_tto family** (`losses.kl_distill_scorer_loss`): KL distill T=2.0 → Lane SC++ (Quantizr's 0.05 edge over Selfcomp)
- **omega_hessian_qat** (in flight): replaces block-FP per-tensor exp with per-weight Hessian-prior → Lane SO
- **w_hard_pair_self_compress** (in flight): hard-pair weighting drives block-FP exponent precision per-pair
- **gh_darts_ratio / k_darts_channels** (`darts.py`): DARTS arch search on (hidden, block_hidden, num_blocks) — exactly what Selfcomp admitted he skipped → Lane DARTS-S
- **j_nwc_neural_weight_compression** (`entropy_bottleneck.py`): replaces xz with neural arithmetic coder on qints (Ballé eureka) → Lane EBR-S
- **m_v3_pose_from_embedding**: "pose from embedding" IS Selfcomp's frame_affine_embedding — confirms paradigm
- **hm_homography_motion**: replace Selfcomp's 6-DOF affine with 8-DOF homography → Lane HM-S
- **wc_curator_outlier**: Cosmos Curator outlier weighting deprioritizes trivial pairs in block-FP → Lane WC-S
- **fridrich.py + fridrich_losses.py**: cost-weighted block-FP exponents (Lane FR-Ω) + variable-σ LUT (Lane FR-MM) — highest novel-design EV

## 🟡 7 NEUTRAL (orthogonal but not multiplicative)

a_pose_tto, ge_geodesic_pose, rm_riemannian_pose_tto, ea_entropy_archive, ec_engineered_corrections, mae_v, saug_v2

## 🔴 8 SUPERSEDED (Selfcomp dominates)

- f_v3/v4/v5 FP4/FP8: block-FP @ 1.017bpw beats FP4 @ 4bpw
- i_coolchic_masks: grayscale-LUT mask cheaper
- d_halfframe / h_halfframe / v_halfframe: single-mask + affine duality is half-frame done right (no PoseNet break)
- gp_gaussian_process_pose: Runge-dead
- uniward_texture: killed standalone (council 5/5)
- b_fp4_qat / qat_sweep: superseded
- lr_lora_pose_tto / lr_v2: TTO replaced by analytical
- lm_v2_endpoint_tracking: TTO-adjacent

## ⚪ 2 ORTHOGONAL BIG BETS (independent paths, continue running)

q_faithful_jointgen (true Quantizr replica), sz_phase2_full (moonshot)

## Updated lane portfolio after sweep (10 lanes, ~$36 / 18h parallel)

MM, SA, SC++, SO (original 4 from Selfcomp RE) + FR-Ω, HM-S, DARTS-S, WC-S, FR-MM (5 from sweep) + EBR-S deferred.

Top stack predictions:
- SC++ + FR-Ω + DARTS-S = ~0.25 (frontier, sub-Quantizr)
- SO + FR-Ω = ~0.28
- SC++ alone = ~0.33

## How to apply

- Top priority: implement segmap_renderer + block_fp_codec + mask_grayscale_lut (subagent in flight)
- Tonight: dispatch MM (2h), SA/SC++/SO/FR-Ω/HM-S in parallel (12-14h Modal each, ~$5-7 each)
- Tomorrow: dispatch DARTS-S based on SA result (18h, $6)
- Recursive review: spawn codex:adversarial-review + feature-dev:code-reviewer per landing; 3 clean-pass gate per CLAUDE.md
