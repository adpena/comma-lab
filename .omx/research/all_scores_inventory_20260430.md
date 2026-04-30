# All-Scores Forensic Inventory (2026-04-30)

**Compiled by:** ALL-SCORES-FORENSIC-AGENT (#298)
**Inputs:** `experiments/results/**/contest_auth_eval.json`, `experiments/results/lane_*_modal/harvested_artifacts/_harvest_summary.json` + `_stdout_tail.txt`, `experiments/results/recovered_*/`, memory file index, `.ralph/run_log.md`, `.omx/research/council_*`.
**Goal:** complete catalog of every measured/attempted score across the historical record, plus per-lane forensic classification (APPROACH_KILLED / ENGINEERING_BUG / METHODOLOGY_BUG / CONFIG_BUG / LEGITIMATE_REGRESSION / INDETERMINATE).

> **Tag legend:** `[contest-CUDA]` valid for strategy. `[contest-CPU advisory]` CPU not CUDA — INVALID for kill/promote per CLAUDE.md MPS-non-negotiable. `[Modal-T4-CUDA]` Modal A100/T4 trusted. `[MPS-PROXY]` advisory only. `[empirical:<path>]` static/offline measurement. `[prediction]` not measured.

---

## A. CUDA / Modal-CUDA AUTHORITATIVE SCORES (the only kill/promote-valid set)

| Rank | Lane | Score | PoseNet d | SegNet d | Rate (KB) | Date | Source path | Tag |
|------|------|-------|-----------|----------|-----------|------|-------------|-----|
| 1 | Lane G v3 (KL distill weight=0.002 retry on Lane A anchor) | **1.05** | 0.00345 | 0.00401 | 694 | 2026-04-28 | `experiments/results/lane_g_v3_landed/contest_auth_eval.json` | `[contest-CUDA]` |
| 2 | Lane G v3 + Ω-W-V2 stack (mixed-precision FP4 + per-pair) | **1.07** | 0.00564 | 0.00404 | 643 | 2026-04-30 | `experiments/results/lane_g_v3_omega_w_v2_stack_landed/contest_auth_eval.json` | `[contest-CUDA]` |
| 3 | Lane A (pose TTO baseline on dilated h64) | **1.15** | 0.00497 | 0.00461 | 694 | 2026-04-27 | `experiments/results/lane_a_landed/contest_auth_eval.json` | `[contest-CUDA]` |
| 4 | Lane M-V2 (rank-1 radial-zoom 1-DOF) | **1.84** | 0.07603 | 0.00505 | 694 | 2026-04-28 | `experiments/results/lane_m_v2_landed/contest_auth_eval.json` | `[contest-CUDA]` |
| 5 | Lane F-V2 (FP4 QAT on Lane A) | **1.79** | 0.101 (20× Lane A) | unchanged | 694 | 2026-04-27 | memory `project_lane_f_v2_fp4_architectural_bottleneck_20260427` | `[contest-CUDA]` |
| 6 | Lane M+N (radial-zoom + L∞) | **2.35** | — | — | — | 2026-04-27 | memory `project_lane_mn_radial_zoom_negative_20260427` | `[contest-CUDA]` |
| 7 | Lane B (pose TTO on dilated-h64 baseline) | **2.40** | 0.246 (23× baseline) | 0.0037 | 685 | 2026-04-26 | run_log + `project_lane_b_pose_tto_proxy_auth_gap` | `[contest-CUDA]` |
| 8 | SHIRAZ v4 (181K renderer + poses) | **2.70** | 0.257 | 0.00750 | 519 | 2026-04-26 | run_log | `[contest-CUDA]` |
| 9 | Lane F V1 (FP4 QAT on dilated-h64 + bug) | **2.73** | 0.391 | unchanged | 559 | 2026-04-27 | memory `project_lane_f_fp4_qat_regression_20260427` | `[contest-CUDA]` (but config bug) |
| 10 | Lane H crf56 (denoise high-CRF) | **3.20** | 0.518 | 0.00555 | 559 | 2026-04-27 | `experiments/results/lane_h_crf56/auth_eval/contest_auth_eval.json` | `[contest-CUDA]` |
| 11 | Verified baseline (dilated-h64 + CRF50) | **0.90** | 0.0107 | 0.00240 | 293 | 2026-04-25 | memory `project_cuda_gate_result_20260425` | `[contest-CUDA]` (no scorers yet) |
| 12 | Verified baseline (h64 + CRF50 + pose ASYM) | **2.29** | — | — | — | 2026-04-27 (recomputed) | memory `project_verified_baseline_2_29` | `[contest-CUDA]` |

**Quantizr leader (external):** 0.33. **Selfcomp #2 (external):** 0.38. **Mask2Mask #3 (external):** 0.60.

---

## B. CPU / Modal-T4-CPU advisory scores (NOT valid for strategy)

These were eval'd with `device=cpu` per provenance. CPU PoseNet drift ~9.5% smaller-than-CUDA; SegNet drift ~0%. Per CLAUDE.md "MPS auth eval is NOISE" non-negotiable, NOT valid for kill/promote.

| Lane | Score | PoseNet d | SegNet d | Rate (KB) | Source | Tag | Notes |
|------|-------|-----------|----------|-----------|--------|-----|-------|
| Lane MM v2 (encoder-only grayscale-LUT) | **2.63** | 0.174 | 0.00560 | 1134 | `lane_lane_mm_v2_modal/.../contest_auth_eval.json` | `[contest-CPU advisory]` | Council Round 7 retag — needs CUDA to formalize FALSIFICATION |
| Lane UNIWARD v8 | **1.14** | 0.00450 | 0.00461 | 694 | `lane_uniward_v8_modal/.../contest_auth_eval.json` | `[contest-CPU advisory]` + **NO-OP** | masks.mkv SHA-identical to Lane A (Council B audit) |
| Lane UNIWARD v7 | **53.61** | 62.69 | 0.283 | 346 | `lane_uniward_v7_modal/.../contest_auth_eval.json` | `[contest-CPU advisory]` + **48x64 mask bug** | Pre-Check-76 catastrophic mask resolution |
| Lane GP v2 | **89.66** | 149.95 | 0.505 | 681 | `lane_lane_gp_v2_modal/.../contest_auth_eval.json` | `[contest-CPU advisory]` | Runge phenomenon polynomial fit |
| Lane GP v3 | **89.67** | 149.95 | 0.505 | 693 | `lane_lane_gp_v3_modal/.../contest_auth_eval.json` | `[contest-CPU advisory]` | Same root-cause as v2; Fix A landed but didn't budge |

---

## C. Lanes that CRASHED or NEVER PRODUCED A SCORE (forensic targets)

These dispatched to GPU, spent money, but produced NO usable score. Full per-lane forensic in section D.

| Lane | RC | Crash kind | Bug class | Memory ref / log path |
|------|----|------------|-----------|----------------------|
| Lane FL (RAFT-derived poses) | 137 | OOM-kill | RAFT model OOM on T4 | `lane_lane_fl_modal/_stdout_tail.txt` |
| Lane FL v2 | 137 | OOM-kill | Same | `lane_lane_fl_v2_modal/...` |
| Lane GP (v1) | 1 | device mismatch | Tensor on cpu+cuda in `save_pose_gp` | `lane_lane_gp_modal/...` |
| Lane MAE-V | 1 | ModuleNotFoundError | `pydantic` not installed in Modal image | `lane_lane_mae_v_modal/...` |
| Lane MM (v1) | 3 | Dead-flag | `--hard` not in `build_lane_mm_archive` argparse | `lane_lane_mm_modal/...` |
| Lane Omega-Hessian | 1 | CUDA assert | Probably indexing OOB in renderer.py:471 | `lane_lane_omega_hessian_modal/...` |
| Lane S (self-compress full retrain) | 1 | (unknown — log truncated) | likely OOM after Phase 1 init | `lane_lane_s_modal/...` |
| Lane SA (SegMap clone — plain) | 1 | OOM (T4 7GB > 14.56GB free) | SegMapTrainer never chunked | `lane_lane_sa_modal/...` |
| Lane SA v2 | 1 | OOM | Same | `lane_lane_sa_v2_modal/...` |
| Lane SA v3 (A10G retry) | 1 | OOM (21GB > 22GB) | Same — A10G shared 22GB; chunking still needed | `lane_lane_sa_v3_modal/...` |
| Lane SC++ (v1) | 1 | OOM (T4 7GB) | SegMapTrainer | `lane_lane_sc_plus_plus_modal/...` |
| Lane SC++ v2 | 1 | OOM (T4 7GB) | SegMapTrainer | `lane_lane_sc_plus_plus_v2_modal/...` |
| Lane SC++ v3 (A10G retry) | 1 | OOM (21GB) | SegMapTrainer | `lane_lane_sc_plus_plus_v3_modal/...` |
| Lane SO (Hessian block-FP) | 1 | OOM (T4 7GB) | SegMapTrainer | `lane_lane_so_modal/...` |
| Lane SO v2 | 1 | OOM (T4 7GB) | SegMapTrainer | `lane_lane_so_v2_modal/...` |
| Lane UNIWARD v1 | 1 | FATAL: missing baseline_dilated_h64_0_90/renderer.bin | Anchor path | `lane_lane_uniward_modal/...` |
| Lane UNIWARD v2 | 1 | bash-as-python SyntaxError | Heredoc bug in remote_lane_uniward_texture.sh | `lane_uniward_v2_modal/...` |
| Lane UNIWARD v3 | 1 | NameError: sys not defined | Missing import in inline python | `lane_uniward_v3_modal/...` |
| Lane UNIWARD v4 | 1 | TypeError: unexpected `mode` kwarg | Calling `apply_saliency_weighted_compression` with wrong sig | `lane_uniward_v4_modal/...` |
| Lane UNIWARD v5 | 1 | ValueError: saliency_inv must be 2-D bool | Threshold cast missing | `lane_uniward_v5_modal/...` |
| Lane UNIWARD v6 | 1 | NameError: json not defined | Missing import | `lane_uniward_v6_modal/...` |
| Lane W (hard-pair self-compress) | 1 | ValueError: --resume-from is quantised binary | Resume from .bin (ASYM magic) instead of .pt | `lane_lane_w_modal/...` |
| Lane W v2 | 124 | TIMEOUT (8h hit) | Stage 2 train hung at "training" | `lane_lane_w_v2_modal/...` |
| Lane Q-FAITHFUL (v1) | 2 | argparse: --tag required | Missing CLI flag | `lane_q_faithful_modal/...` |
| Lane Q-FAITHFUL v2 | 1 | CONFIG ERROR: variant=quantizr_faithful not FP4A-exportable | --auth-eval-on-best gate vs variant mismatch | `lane_q_faithful_v2_modal/...` |
| Lane V (Quantizr replica 88K) | crash | RuntimeError: 1-ch input vs 3-ch weight | Channel mismatch in HintedRenderer alpha_map | memory `project_lane_v_crashed_channel_mismatch_20260428` |
| Lane V-V2 (annealed half-frame) | crash | Same | Inherits Lane V channel bug | memory `project_killed_lanes_forensic_audit_20260428` |
| Lane F-V4 (mixed-precision FP4 last gasp) | n/a | run finished but score worse | Hardware: NVFP4 needs Blackwell; 4090 is Ada (CC 8.9) | memory `project_lane_f_fp4_qat_regression_20260427` |
| Lane D-V3 (annealed half-frame + KL fix) | crash | Half-frame broken (mask_t channel dispatch) | end_value=0.5 ≠ inflate-time 1.0 (train/test distribution mismatch) | memory `project_killed_lanes_forensic_audit_20260428` |
| Lane J-JBL (Jaccard + boundary label smoothing) | exit | combined_jbl_distill_loss not wired in train_renderer | Loss-mode validator allowed but dispatch missing | memory `project_killed_lanes_forensic_audit_20260428` |
| Lane I (Cool-Chic CCh1 replacement) | crash | parametrize-strip mismatch on .pt load | Stage 3 export crashed after train succeeded | memory `project_lane_i_crashed_parametrize_strip_20260428` |
| Lane GP v4 (smooth-basis pose-fit) | n/a | KILLED at design phase | Council #271 empirical proof (white-noise pose dims 1-5) | memory `project_lane_gp_v4_killed_basis_fit_infeasible_20260430` |
| Lane STC (clean-source FALSIFICATION WITHDRAWN) | n/a | Local MPS measurement invalid | MPS-PROXY violation; CUDA confirm pending | memory `project_lane_stc_clean_source_FALSIFIED_20260429` |

---

## D. Hidden-gem inventory (lanes killed for the WRONG reason)

These are lanes whose "kill" was an engineering / methodology / config bug — the underlying approach is approximately untested. Re-engineering cost typically <$2/lane.

(See `recoverable_lanes_re_engineering_plans_20260430.md` companion file.)

---

## E. Lane G v3 anchor confirmation cross-platform

| Platform | Lane | Score | Note |
|----------|------|-------|------|
| Vast.ai 4090 (canonical) | Lane G v3 | 1.05 | `lane_g_v3_landed/contest_auth_eval.json` |
| Modal T4 (CUDA) | Lane G v3 | 1.04 | `experiments/results/modal_auth_eval_9b20bdfca246.json` (run_log 2026-04-29) |

Modal pipeline trusted within 0.01 noise floor. CUDA-T4 ≈ CUDA-4090 for these distortions.
