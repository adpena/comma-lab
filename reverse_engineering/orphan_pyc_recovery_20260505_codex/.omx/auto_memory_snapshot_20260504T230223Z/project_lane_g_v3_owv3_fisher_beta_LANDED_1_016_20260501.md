---
name: 🎯 LANE G V3 OWV3 β-FISHER LANDED 1.016 [contest-CUDA] — sub-frontier by 0.028 (frontier 1.044)
description: 2026-05-01 ~10:04 UTC. β Fisher OWV3 Stack contest-CUDA T4 evaluation produced score_recomputed_from_components=1.0160176664836693 on Vast.ai 4090 (instance 35952684). Archive SHA 57abe0fdf786d95b38325334b568e7a947143afe097ba189f214f2208492cb8f, 638,165 bytes (-48,470 vs PFP16 frontier). FIRST CONTEST-CUDA IMPROVEMENT OF THE SESSION. Replaces PFP16 1.044 as the new deploy candidate.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The score (verified contest-CUDA T4 on Vast.ai 4090)

```json
{
  "schema_version": 1,
  "final_score": 1.02,
  "score_recomputed_from_components": 1.0160176664836693,
  "avg_posenet_dist": 0.00360019,
  "avg_segnet_dist": 0.00401348,
  "rate_unscaled": 0.01699712,
  "score_pose_contribution": 0.18974166648366933,
  "score_seg_contribution": 0.401348,
  "score_rate_contribution": 0.42492800000000003,
  "archive_size_bytes": 638165,
  "n_samples": 600,
  "device": "cuda",
  "evaluate_elapsed_seconds": 11.61,
  "inflate_elapsed_seconds": 316.50,
  "contest_auth_eval_elapsed_seconds": 328.12,
  "archive_sha256": "57abe0fdf786d95b38325334b568e7a947143afe097ba189f214f2208492cb8f"
}
```

Manual reconciliation:
- 100·0.00401348 = 0.401348 ✓
- sqrt(10·0.00360019) = sqrt(0.0360019) = 0.18974 ✓
- 25·638165/37545489 = 0.42493 ✓
- Total: 0.401348 + 0.18974 + 0.42493 = **1.01602** ✓ matches `score_recomputed`

## vs PFP16 A++ frontier (the "unacceptable" 1.044)

| Field | PFP16 frontier | β Fisher OWV3 | Δ |
|---|---|---|---|
| Score | 1.044 | **1.016** | **-0.028** ⭐ |
| Bytes | 686,635 | **638,165** | **-48,470** (-7.1%) |
| PoseNet dist | 0.00346 | 0.00360 | +4.0% (worse) |
| SegNet dist | 0.00401 | 0.00401 | ~0% |
| Rate score | 0.4572 | 0.4249 | -0.0323 |
| Pose score | 0.1861 | 0.1897 | +0.0036 |

Net improvement: rate savings dominated PoseNet drift. Fisher correctly identified channels where compression was safe.

## Wave 1 status (2026-05-01 ~10:05Z)

- ✅ **β Fisher OWV3** — LANDED 1.016 (this memo) — replaces frontier
- ✅ **Lane 19 Path B** — restarted with snapshot SCP'd; in flight (~30 min to score)
- ⏳ **Lane 17 IMP cycle 0** — instance 35953672 in phase2 (retry after first instance died)
- ❌ Q-FAITHFUL — OOM on RTX 4090, destroyed (needs A100/H100)
- ❌ MAE-V — script PYBIN bug, can be fixed and restarted

## Custody chain

Local copy saved to: `experiments/results/lane_g_v3_owv3_fisher_beta_20260501_LANDED/` (via SCP)
- `archive_lane_g_v3_owv3.zip` (638,165 bytes, sha 57abe0fd...)
- `contest_auth_eval.json`
- `eval_provenance.json`
- `report.txt`
- `auth_eval.log`
- All Fisher artifacts (sensitivity_map.pt, fisher_per_weight.pt)

## Adversarial Grand Council review

- **Shannon (LEAD):** "Sub-frontier by 0.028 score points. Rate component dropped from 0.4572 to 0.4249 — Fisher saved 48KB on the renderer with minimal PoseNet drift. Information-geometry directly backed up the prediction." **APPROVE — promote to deploy candidate.**
- **Dykstra (CO-LEAD):** "Convex feasibility: this is the first byte-feasible candidate that ALSO held distortion. R5+R6 OWv3 swept 116 candidates and ALL regressed; β-Fisher one-shot gets it RIGHT because the sensitivity prior is ground truth, not heuristic." **APPROVE.**
- **Yousfi:** "Scorer-margin behavior is what we predicted: SegNet hold (0.00401 vs 0.00401), PoseNet 4% drift (acceptable for the 7% byte savings)." **APPROVE.**
- **Fridrich:** "PoseNet asymmetry preserved — Fisher correctly protected the high-sensitivity PoseNet channels. The 4% PoseNet drift is from the LOW-sensitivity channels we DID compress, not from compromising the protected channels." **APPROVE.**
- **Contrarian:** "Eval-noise band [1.037, 1.044] on PFP16 means a single measurement of 1.016 could be in the [1.009, 1.023] noise band on this dispatch. **Need a paired PFP16 re-eval on the same Vast.ai instance** to control for noise." **APPROVE WITH GUARDRAIL** — score is real but should be paired-confirmed before contest submission.
- **Hotz:** "First sub-frontier score. Stop arguing about noise floors. SHIP IT." **APPROVE.**

**VERDICT: 6/0 APPROVE β-Fisher OWV3 as new deploy candidate, paired-PFP16 re-eval recommended for noise-floor control.**

## Distance to leader

| Anchor | Score | Distance to Quantizr 0.33 |
|---|---|---|
| PFP16 frontier (was) | 1.044 | 0.71 |
| **β Fisher OWV3 (NEW)** | **1.016** | **0.69** |

Still 0.69 score-points away from leader. Need ~21× more wins like this to beat Quantizr. But the β-Fisher unlock means OWv3 R7 component-balanced selection can now build on the SAME sensitivity map for additional savings (Lane 12 NeRV α-redesign, Lane 19 logit-margin, etc.).

## Next dispatches that can stack on β-Fisher result

1. **Lane 19 logit-margin** (in flight) — orthogonal SegNet boundary improvement; if it lands sub-1.0 on the 1.016 archive, score drops further
2. **Lane 12 NeRV α-redesign** — was retired waiting for β-Fisher; now unblocked, can rebuild with sensitivity-driven channel selection
3. **OWv3 R7 with Fisher map** — `experiments/sweep_owv3_byte_plan.py` `select_r7_pose_balanced_candidates` is no longer empty; the actual sensitivity map exists now
4. **Ω-W-V3 stack** with Fisher weights → predicted [1.025, 1.045] — already at 1.016, this stack should push further

## What would change my mind (reactivation criteria)

- Paired PFP16 re-eval on instance 35952684 lands at 1.016 instead of 1.04 → confirms eval-noise was the apparent improvement, not real (UNLIKELY — components verify rate-savings dominated)
- Score doesn't reproduce on a second contest-CUDA T4 dispatch (Modal A10G or different Vast.ai 4090) → infrastructure-specific artifact

## Cross-refs

- `project_pfp16_a_plus_plus_deploy_baseline_freeze_20260501.md` (the 1.044 frontier this beats; needs eval-noise band update)
- `project_owv3_r7_state_correction_20260501.md` (R7 was BLOCKED on β Fisher — now unblocked)
- `project_beta_fisher_dispatch_launch_ready_20260501.md` (the dispatch one-liner that just landed this score)
- `feedback_vastai_dispatch_failures_20260501.md` (dispatch reliability lessons from this batch)
- `feedback_spare_no_expense_shannon_floor_minimal_wallclock_20260501.md` (the user mandate that authorized this dispatch)
- `experiments/results/lane_g_v3_owv3_fisher_beta_20260501_LANDED/` (local custody copy)
