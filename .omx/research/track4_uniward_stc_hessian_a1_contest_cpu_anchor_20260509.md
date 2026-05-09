# Track 4 UNIWARD/STC/Hessian-A1 — [contest-CPU] anchor + falsification 2026-05-09

## Verdict

Track 4 of the Fields-Medal Grand Council Phase-1 multi-track (verdict commit
`9b44c2f6`, memory
`feedback_grand_council_fields_medal_theoretical_floor_20260509`) is empirically
falsified at the dispatch-screen level. ALL Track 4 candidates produced by
`tools/build_uniward_stc_hessian_a1_v1.py` score WORSE than the A1 baseline on
the leaderboard ranking axis (`[contest-CPU]` Linux x86_64).

**Contest-CPU anchor (best Track 4 candidate):**

- archive: `experiments/results/track4_uniward_stc_hessian_a1_blocks4_7bit_20260509_codex/archive.zip`
- archive sha256: `864f7eecb0825111ed8fd7625ef434f0e216deaab28b11fcdb3295c17d35c754`
- archive bytes: `177,903`
- score: `0.19869389522684905` `[contest-CPU]` GHA Linux x86_64 ubuntu-24.04
  / 20260413.86, n_samples 600
- workflow_run: `25597258234` (https://github.com/adpena/comma_video_compression_challenge/actions/runs/25597258234)
- adjudicated_json: `experiments/results/track4_blocks4_7bit_cpu_eval_gha_20260509/contest_auth_eval.adjudicated.json`
- evidence_grade: `contest-CPU-1to1`
- delta vs A1 baseline (`0.192847577437`): **+0.005846 (WORSE)**

## Candidate ladder + scoring

| Candidate | Bytes | Δ bytes | Score | Axis | Δ vs A1 baseline |
|---|---:|---:|---:|---|---:|
| **A1 baseline** | 178,262 | — | **0.192848** | `[contest-CPU]` GHA | — |
| `blocks4_7bit` | 177,903 | −359 | **0.198694** | **`[contest-CPU]` GHA** | **+0.005847** |
| `blocks4_7bit` | 177,903 | −359 | 0.198704 | `[macOS-CPU calibrated]` | +0.005856 |
| `blocks3_7bit` | 178,027 | −235 | 0.211735 | `[macOS-CPU calibrated]` | +0.018887 |
| `target176000` | 177,668 | −594 | 0.213005 | `[macOS-CPU calibrated]` | +0.020157 |
| `target174000` | 177,624 | −638 | 0.227295 | `[macOS-CPU calibrated]` | +0.034447 |
| `blocks0_7bit` | 177,559 | −703 | 0.233267 | `[macOS-CPU calibrated]` | +0.040419 |
| `target164000` | 171,109 | −7,153 | 0.260516 | `[macOS-CPU calibrated]` | +0.067668 |

## macOS↔Linux x86_64 calibration confirmed

The best Track 4 candidate's macOS prediction was 0.198704; GHA Linux x86_64
actual 0.198694; delta **−9e-6** (within the 6e-6 epsilon documented in
`feedback_macos_x86_64_epsilon_calibrated_tag_20260508`). macOS-CPU advisory
remains a credible predictor of `[contest-CPU]` for HNeRV-class candidates.

## Root cause

The v1 builder uses `mean(θ²)` per tensor as the Fisher-proxy saliency
(empirical-Bayes approximation when no gradient access is available; van Trees
§2.4). On A1's substrate, this proxy is empirically **anti-correlated** with
true score-saliency:

A1 was trained with `experiments/train_score_gradient_pr101_finetune.py` —
the loss is `score_gradient(score_components(decoder(latent)))`, which
propagates `d(seg)/d(theta) + d(pose)/d(theta)` through every parameter. By
the converse of the Cramer-Rao argument: parameters with HIGH `mean(θ²)` are
exactly those that the score-gradient supervision pushed AWAY from zero
because they were score-relevant; small `mean(θ²)` tensors are NOT "low
Fisher" — they were trained TOWARDS zero by the score gradient because they
are score-IRrelevant directions.

Coarsening "low-Fisher" tensors hits exactly the score-naive directions in
the parameter manifold; even at sub-1e-3 distortion the score penalty exceeds
the byte savings.

This is a sharper version of the cliff documented in
`feedback_three_lossy_anchors_show_rel_err_squared_objective_falsified_20260508`
(rel_err ≥ 0.04 on score-NAIVE substrates). On A1's score-AWARE substrate, the
cliff is at rms ≈ **3.5e-4** (4e-4 distortion lost 0.006 score).

## Per CLAUDE.md kill-as-last-resort

This is **DEFERRED-pending-research**, NOT KILLED. Reactivation criteria:

1. **Replace mean(θ²) with score-gradient-derived saliency.** Run
   `upstream/evaluate.py` with autograd hooks to capture per-tensor
   `d(score)/d(theta)`, aggregate as Fisher diagonal `E[(d/dtheta)²]`. Cost:
   one evaluate.py run with grad enabled (~5-30 min).
2. **STC syndrome trellis on the latent_blob, not decoder weights.** UNIWARD's
   spatial weighting natively applies to a (600 × 28) latent grid. The STC
   Filler 2011 paper specifically targets ±1 / ±2 embedding into a spatially-
   correlated cover, not weight-tensor coarsening.
3. **Stay below 1e-3 distortion AND extract >1 KB savings BEFORE coarsening.**
   All Track 4 candidates have <750 B savings at distortions in 4e-4 to 8e-3.
4. **Sanity-check on a non-score-gradient substrate.** If Track 4 also fails
   on raw PR101 (untrained for score gradient), the v1 algorithm itself is
   wrong; if it succeeds on raw PR101, the failure is specifically because
   A1's score-gradient training has saturated all per-tensor headroom.

## Implication for council Phase 1 dispatch order

A1's CUDA score is `0.226352` `[contest-CUDA]` (HNeRV-cluster CUDA-CPU gap
0.033, consistent with the empirical profile). Track 4 on the SAME substrate
loses on the CPU axis; CUDA dispatch on Track 4's blocks4_7bit candidate is
dominated by other Phase-1 paths:

- **Track 1 (Ballé hyperprior end-to-end)** — predicted -0.030 to -0.040;
  highest EIG/$ ceiling-breaker per council §2.
- **Track 5a (CUDA verify A1 itself)** — locks the CUDA medal-band claim
  before Phase-2 substrate moves.

Do NOT spend further dispatch budget on Track 4 candidates without first
implementing one of the 4 reactivation criteria.

## Custody artifacts

- Builder: `tools/build_uniward_stc_hessian_a1_v1.py` (commits `524dd5de` + `9e0c1988`)
- Tests: `src/tac/tests/test_build_uniward_stc_hessian_a1_v1.py` (4 pass)
- Track 4 candidates: `experiments/results/track4_uniward_stc_hessian_a1_*/`
- Best candidate: `experiments/results/track4_uniward_stc_hessian_a1_blocks4_7bit_20260509_codex/`
- Promotion card: `experiments/results/track4_uniward_stc_hessian_a1_blocks4_7bit_20260509_codex/promotion_card.json`
- Adjudicated GHA result: `experiments/results/track4_blocks4_7bit_cpu_eval_gha_20260509/contest_auth_eval.adjudicated.json`
- Lane registry: `track1_paradigm_delta_track4_uniward_stc_hessian_a1` at L2
- Memory: `feedback_track_4_uniward_stc_hessian_a1_landed_20260509.md`

## Cross-references

- Council verdict (Track 4 design): `.omx/research/grand_council_fields_medal_theoretical_floor_20260509.md`
- A1 anchor: `.omx/research/phase_a1_latent_aligned_contest_cpu_anchor_20260509_codex.md`
- Codex Track 4 ladder: `.omx/research/track4_uniward_hessian_a1_ladder_20260509_codex.md`
- macOS↔Linux x86_64 epsilon: memory `feedback_macos_x86_64_epsilon_calibrated_tag_20260508`
- Cliff prior (rel_err ≥ 0.04 on score-naive substrates): memory `feedback_three_lossy_anchors_show_rel_err_squared_objective_falsified_20260508`
- Substrate-vs-codec composition: memory `feedback_substrate_vs_codec_composition_meta_pattern_20260508`
- Dual CPU+CUDA mandate: memory `feedback_dual_cpu_cuda_auth_eval_mandatory_20260508`
- GHA dispatch helper landed: memory `project_pr107_cpu_eval_score_anchor_gha_20260508`
