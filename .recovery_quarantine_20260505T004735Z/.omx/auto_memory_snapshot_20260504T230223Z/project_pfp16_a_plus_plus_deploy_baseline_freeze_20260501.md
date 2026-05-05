---
name: PFP16 A++ deploy baseline FROZEN — 1.043987524793892 [contest-CUDA T4] Wave 0 evidence-hygiene COMPLETE
description: 2026-05-01 Wave 0 freeze of the contest-grade deploy baseline. Three independent archive copies verified matching SHA-256 0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f at 686,635 bytes. All score components reconciled against contest formula. Deploy-ready until any new contest-CUDA score lands below 1.044.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Frozen baseline (deploy-ready) — WITH EVAL-NOISE BAND

| Field | Value |
|---|---|
| **Score (this measurement)** | **1.043987524793892** |
| **Eval-noise band** | **[1.037, 1.044]** (see correction below) |
| **Tag** | **[contest-CUDA Grade A++ T4 single-measurement]** |
| **Bytes** | **686,635** |
| **SHA-256** | **`0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`** |
| Eval samples | 600 |
| Schema version | 1 |
| Frozen at | 2026-04-30 ~16:25Z (T4 contest-CUDA on Lightning) |
| Verified at | 2026-05-01 ~08:30Z (this memory) |
| **Noise discovery** | 2026-05-01 ~09:25Z (`project_owv3_r7_state_correction_20260501.md`) |

**EVAL-NOISE CORRECTION** (added 2026-05-01 ~09:25Z): The OWv3 R5+R6 references in `experiments/sweep_owv3_byte_plan.py:52-83` carry `paired_pfp16_score: 1.037045485927815` for the SAME SHA `0af839ab...ed7f` / 686635 bytes. This is the SAME archive evaluated in two different sessions producing 1.044 and 1.037 — a 0.007 eval-noise band. The OWv3 deferral memory predicted 0.005-0.01 cycle-to-cycle noise; this is the empirical measurement of that band. Any future "frontier" claims should cite the band, not the single 1.044 number. Score derivative arithmetic in the Shannon checkpoint remains correct against THIS measurement's components but the absolute frontier has 0.007 uncertainty.

## Score breakdown (verified against contest formula `100·seg + sqrt(10·pose) + 25·bytes/37545489`)

```
avg_posenet_dist:   0.00346442
avg_segnet_dist:    0.00400656
rate_unscaled:      0.01828808

score_pose_contribution    = sqrt(10 × 0.00346442) = 0.18612952479389186
score_seg_contribution     = 100 × 0.00400656     = 0.400656
score_rate_contribution    = 25 × 686635 / 37545489 = 0.45720200000000005

score_recomputed_from_components = 1.043987524793892 ✓
```

## Three deploy-archive copies — all SHA-verified intact

| Location | Bytes | SHA matches |
|---|---|---|
| `experiments/results/lane_g_v3_pfp16/archive_lane_g_v3_pfp16.zip` | 686635 | ✓ |
| `experiments/results/lane_g_v3_pfp16/exact_cuda_20260430T1353Z/archive_lane_g_v3_pfp16.zip` | 686635 | ✓ |
| `experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/archive/archive.zip` | 686635 | ✓ |

All three are byte-identical and SHA-identical. Any of the three can be submitted to the contest without re-evaluation.

## Provenance bundle (the contest-CUDA evidence chain)

`experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex/`:
- `archive_sha256.txt` — pinned SHA + bytes (single source of truth)
- `contest_auth_eval.json` — score_recomputed_from_components 1.043987524793892
- `report.txt` — upstream evaluate.py output (600 samples, contest-CUDA T4)
- `auth_eval.log` — full eval log (10.7K)
- `gpu.txt` — T4 confirmation (18B)
- `eval_provenance.json` — full provenance (3.4K)

## What this freezes

1. **Lower-bound deploy guarantee:** if no further dispatch lands a sub-1.044 [contest-CUDA] score, this is the contest submission. Wave 1 dispatches (β/Lane 17/Lane 19) attempt to improve, but failure means we ship 1.044.

2. **Anchor for all sensitivity / OWv3 / Lane 19 bit-allocation work:** every "predicted ΔScore" claim in PARADIGM-β material (`src/tac/owv3_sensitivity_weighted.py`, `src/tac/sensitivity_map.py`) is computed against THIS baseline's component values.

3. **Distance to leader:** Quantizr 0.33. Score gap 0.71. Worth $XX of GPU time per the Shannon-floor execution plan.

4. **Distance to Selfcomp 0.38:** 0.66 score points. Selfcomp/SC++ is the second leader; reactivating SA/SC++ training (Wave 2) requires recovering A10G OOM workaround + Council C bf16+chunk fix (already landed).

## Reactivation criteria (when this freeze should be re-baselined)

- ANY new contest-CUDA score < 1.044 → THIS file becomes obsolete; create a successor freeze memory
- Lane 17 IMP cycle 0 if validated (currently retracted KILL pending Lightning Studio GPU mode + train_distill swap) → may produce sub-1.044 if 88K param sparse model converges
- Lane 19 logit-margin if SegNet boundary improvement holds → sub-1.044 candidate
- OWv3 r6+ sweep if a candidate predicted < 1.040 [contest-CUDA pred] lands and CUDA-confirms

## Cross-refs

- `project_shannon_floor_execution_state_checkpoint_20260501.md` (master plan)
- `project_lane_g_v3_landed_1_05_20260428.md` (the parent Lane G v3 anchor at 1.0489)
- `project_owv3_byte_feasible_candidate_dispatch_deferred_20260501.md` (OWv3 r5 within-noise deferral)
- `feedback_grand_council_imp_permanent_fix_review_20260430.md` (5 metabug extincts that hardened this baseline)
- `project_lane_17_imp_killed_cycle_0_198_regression_20260430.md` (Lane 17 KILL retraction context)
