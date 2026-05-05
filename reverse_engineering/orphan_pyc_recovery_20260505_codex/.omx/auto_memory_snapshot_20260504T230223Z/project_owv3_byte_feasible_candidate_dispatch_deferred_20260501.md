---
name: OWv3 owv3_0018 byte-feasible candidate — contest-CUDA dispatch DEFERRED (within noise floor)
description: 2026-05-01 ~02:50 UTC. Codex agent #305 (PARADIGM-β) produced 135 OWv3 byte-plan sweep candidates; best candidate `owv3_0018_bbr0p69_protect0p0014_aggr1em05` (686,557 bytes) tested on Modal T4-CPU (advisory) at 1.0339. CPU↔CUDA drift correction (PoseNet ×1.105) yields predicted [contest-CUDA] = 1.0429, vs PFP16_A++ frontier 1.0440. Δ = -0.0011 is within noise floor (~0.005-0.01). Vast.ai dispatch DEFERRED until refined sweep produces a candidate with predicted < 1.040.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## Candidate details

- Source: `experiments/results/lane_g_v3_owv3_byte_plan_sweep_20260430_codex_r5/best_byte_feasible/archive_lane_g_v3_owv3.zip`
- SHA-256: `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`
- Bytes: 686,557 (78 bytes UNDER PFP16_A++ frontier; 7,517 bytes UNDER Lane G v3 anchor)
- Knobs: `bit_budget_ratio=0.69`, `protect_threshold=0.0014`, `aggressive_threshold=1e-05`
- Sweep: 135 candidates evaluated, 116 byte-feasible vs frontier, owv3_0018 chosen by `closest_archive_bytes_not_exceeding_frontier_no_score_claim`

## Modal T4-CPU advisory result (2026-05-01 ~01:00 UTC)

```json
{
  "score": 1.0339,
  "posenet_dist": 0.00309012,
  "segnet_dist": 0.00400950,
  "rate": 0.01828601,
  "archive_bytes": 686557
}
```

Source: `experiments/results/modal_auth_eval_e1deda126d86.json` (Modal app `ap-D2kAf4VfCOgvAzqvSFNB6F`).

## CPU↔CUDA drift projection

Per `all_scores_inventory_20260430.md` Section B and `feedback_modal_pipeline_trusted_lane_g_v3_1_04_20260429`:
- T4-CPU PoseNet drift: ~9.5% smaller than CUDA (CPU under-reads)
- T4-CPU SegNet drift: ~0% (no significant drift)

Applied to OWv3:
- PoseNet CUDA-pred: 0.00309 × 1.105 = 0.00342 → √(10·pose) ≈ 0.1849 (was 0.1759)
- SegNet CUDA-pred: 0.00401 (unchanged) → 100·seg = 0.401
- Rate: 0.01829 (unchanged) → 25·rate = 0.4571
- **CUDA-pred score: 0.401 + 0.1849 + 0.4571 = 1.0429 [contest-CUDA prediction]**

## Comparison

| Lane | Score | Tag | Bytes | Δ vs frontier |
|---|---|---|---|---|
| Lane G v3 anchor | 1.0500 | [contest-CUDA] | 694,074 | +0.0060 |
| PFP16_A++ frontier | 1.0440 | [contest-CUDA] | 686,635 | — |
| **OWv3 owv3_0018 (predicted)** | **1.0429** | **[contest-CUDA prediction]** | **686,557** | **-0.0011** |

## DEFER decision rationale

The Δ vs frontier (-0.0011) is BELOW the contest-CUDA measurement noise floor:
- Modal CUDA reproduced Lane G v3 within 0.01 of Vast.ai's 1.05 baseline (`feedback_modal_pipeline_trusted_lane_g_v3_1_04_20260429`)
- Cycle-to-cycle measurement noise on the same archive runs ~0.005-0.01 score points

A single Vast.ai 4090 contest-CUDA dispatch (~$0.50, ~30 min) cannot reliably distinguish 1.0429 from 1.0440. Possible outcomes:
- ~50% probability OWv3 lands at 1.040-1.044 → confirms it's a marginal new frontier (worth +0.004)
- ~50% probability OWv3 lands at 1.044-1.050 → confirms it's tied/slightly regressing (no progress)

Cost-benefit: $0.50 GPU time for binary noise-vs-signal answer with 50/50 prior is not justified when the candidate is this marginal.

## What would change my mind (reactivation criteria)

- A refined sweep (r6+) producing a candidate with predicted [contest-CUDA] < **1.040** (-0.004 below frontier) would have enough margin above noise to justify dispatch.
- OR: 2-3 different byte-feasible candidates from the sweep all predicting <1.043, suggesting a consistent sub-frontier band → run ONE eval to verify the band.
- OR: User explicitly approves dispatch as part of broader frontier-validation campaign.
- OR: A complementary lane lands a contest-CUDA result that brings PFP16_A++ frontier UP (e.g., PFP16 retrain regresses to 1.05+), making OWv3 the new frontier candidate worth confirming.

## Grand Council adversarial review

KILL subject: defer dispatch (NOT kill the candidate; it stays in the byte-feasibility table).

Council vote (5+ inner-council members):
- **Shannon (LEAD)**: information value of the dispatch = log2(P(win)/(1-P(win))) is ~0 bits when P=0.5; not worth $0.50 GPU time per CLAUDE.md cost-benefit discipline.
- **Dykstra (CO-LEAD)**: convex feasibility region for OWv3 candidate INCLUDES PFP16_A++ frontier; the sweep didn't produce a candidate strictly inside the lower-score half-space.
- **Yousfi**: scorer-margin is below empirical eval noise; need >2σ separation before promotion.
- **Fridrich**: agree with Yousfi — Council Q4 9/10 revert-on-regression threshold (1.10×) doesn't apply here, but the symmetric promote-on-improvement threshold should be ≥0.005 score points.
- **Contrarian**: pushed for "just dispatch and see" but withdrew when reminded of the 50/50 information value; if the sweep gets a stronger candidate, dispatch is justified then.
- **Hotz**: not enough signal to bother; do the cheap thing first (refine sweep), pay the $0.50 only when needed.

VERDICT: 6/0 DEFER until refined sweep candidate predicted < 1.040 OR user explicit approval.

## Internal consistency checks performed

- **Verified Modal eval result is reproducible**: 2 separate Modal dispatches (`ap-8t26hdYqVjez4RHL6TipdV` v1 and `ap-D2kAf4VfCOgvAzqvSFNB6F` v2) both produced 1.03/1.0339 score on the same archive sha256.
- **CPU↔CUDA drift coefficient verified against historical data**: Lane G v3 Modal-T4-CPU 1.04 vs Vast.ai-CUDA 1.05 = 0.01 drift on a smooth-PoseNet anchor; OWv3 PoseNet is similar magnitude so ×1.105 multiplier is conservatively-correct.
- **Archive contents verified contest-compliant**: `renderer.bin` (292,019 bytes) + `masks.mkv` (421,483) + `optimized_poses.pt` (15,620). No extraneous files.
- **Decode verified by sweep**: `decode_verified: true` in `byte_plan_summary.json`.

## Cross-refs

- `experiments/results/lane_g_v3_owv3_byte_plan_sweep_20260430_codex_r5/byte_plan_summary.json` (135-candidate sweep)
- `experiments/results/modal_auth_eval_e1deda126d86.json` (Modal CPU advisory result)
- `all_scores_inventory_20260430.md` Section B (CPU↔CUDA drift coefficients)
- `feedback_modal_pipeline_trusted_lane_g_v3_1_04_20260429.md` (Modal-CUDA noise floor)
- `project_lane_g_v3_landed_1_05_20260428.md` (anchor)
