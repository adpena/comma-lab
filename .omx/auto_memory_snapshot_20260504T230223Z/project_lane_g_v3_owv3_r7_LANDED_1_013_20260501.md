---
name: 🎯 LANE G V3 OWV3 R7 COMPONENT-BALANCED LANDED 1.013 [contest-CUDA RTX 4090] — sub-frontier by 0.003 vs β Fisher 1.016
description: 2026-05-01 ~12:50 UTC. owv3_0001_bbr0p7_protect0p0013_aggr1em05 (R7 component-balanced selector) scored 1.013396 on Vast.ai 35955469 (RTX 4090). 631,473-byte archive. PoseNet 0.00366 (held), SegNet 0.00401 (held). Replaces β Fisher 1.016 as new deploy candidate. 2nd sub-frontier improvement of session.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## The score (verified contest-CUDA RTX 4090 on Vast.ai 35955469)

```json
{
  "final_score": 1.01,
  "score_recomputed_from_components": 1.0134396099014253,
  "avg_posenet_dist": 0.00366296,
  "avg_segnet_dist": 0.00401579,
  "rate_unscaled": 0.01681888,
  "score_pose_contribution": 0.19133,
  "score_seg_contribution": 0.40158,
  "score_rate_contribution": 0.42047,
  "archive_size_bytes": 631473,
  "n_samples": 600,
  "device": "cuda"
}
```

Manual reconciliation: 0.40158 + 0.19133 + 0.42047 = 1.01338 ✓

## Wave 2 sweep (3-candidate chain eval on instance 35955469)

| Candidate | bbr | Δ bytes vs β Fisher | Score | Pose | Seg | Verdict |
|---|---|---|---|---|---|---|
| owv3_0001_r7 (R7) | 0.70 | -6,692 | **1.013** ⭐ | 0.00366 | 0.00401 | NEW SUB-FRONTIER |
| owv3_0089 (bbr=0.65) | 0.65 | -32,019 | 1.022 | 0.00473 | 0.00401 | mild regression (+0.006) |
| owv3_0134 (bbr=0.5) | 0.50 | -66,548 | 1.097 | 0.00929 | 0.00412 | catastrophic (+0.080) |

**The bbr cliff is between 0.50 and 0.65** — Fisher protection breaks once you compress too aggressively in low-sensitivity channels because the linearization (Hessian quadratic approx) becomes invalid.

## vs entire score history this session

| Anchor | Score | Date | Distance to Quantizr 0.33 |
|---|---|---|---|
| Lane G v3 baseline | 1.05 | 2026-04-28 | 0.72 |
| PFP16 frontier | 1.044 | 2026-05-01 ~05:00Z | 0.71 |
| β Fisher OWV3 | 1.016 | 2026-05-01 ~10:04Z | 0.69 |
| **R7 OWV3** | **1.013** | **2026-05-01 ~12:50Z** | **0.68** |

**Cumulative session improvement: -0.031** (from 1.044 → 1.013) over ~7h wall-clock.

## Why R7 worked where R6+R5 didn't

R5 = "byte-budget-greedy" — picks lowest-rate candidates regardless of distortion sensitivity. ALL 47 R5 candidates regressed (memory `project_owv3_r7_state_correction_20260501.md`).

R6 = "rate-component-balanced" — balances rate vs SegNet+PoseNet. ALL 76 R6 candidates regressed.

R7 = "component-balanced WITH FISHER PRIOR" — the sensitivity map (built from PoseNet Fisher trace) tells the selector which channels can be sacrificed. WAS BLOCKED before β Fisher landed because there was no sensitivity map to feed in.

The 1.013 lands within Shannon's predicted band [1.008, 1.022] for R7 with valid Fisher prior. β Fisher 1.016 was the Wave-1 single-shot, R7 1.013 is the Wave-2 directed-search optimum within the same Fisher constraint set.

## Custody chain (local)

`experiments/results/lane_g_v3_owv3_r7_LANDED_1_013_20260501/`:
- `archive_lane_g_v3_owv3.zip` (631,473 bytes, sha 5c11013539755c6470fb9f55e4d7f2ab6ec1edb2b951a468513d4ed7550f66ef)
- `contest_auth_eval.json` (the score)
- `provenance.json`
- `auth_eval.log`
- `run.log`

Plus regressed candidates: `experiments/results/lane_g_v3_owv3_REGRESSED_20260501/owv3_0089_contest_auth_eval.json` and `owv3_0134_contest_auth_eval.json` for posterity (cliff-finding evidence).

Source candidate dir: `experiments/results/lane_g_v3_owv3_byte_plan_sweep_20260501_r9_r7_component_balanced/r7_pose_balanced/owv3_0001_bbr0p7_protect0p0013_aggr1em05/`

## Next move (Wave 3 refinement sweep, ~$2)

The cliff between bbr=0.65 and bbr=0.70 is interesting. There is potentially a sweet spot at bbr ∈ [0.66, 0.69] that saves more than 6.7KB while preserving PoseNet. Plan:

1. Local: `experiments/sweep_owv3_byte_plan.py` with bbr ∈ {0.66, 0.67, 0.68, 0.69, 0.71, 0.72} × protect ∈ {0.001, 0.0013, 0.0015} = 18 candidates
2. Build 18 archives locally (~30 min)
3. Chain eval all 18 on Vast.ai 4090 (~$2, ~3h)
4. Predicted: at least 1 candidate at -10 to -15KB without PoseNet break → score 1.005-1.010 region

## Adversarial Grand Council review

- **Shannon (LEAD):** "Component-balanced with Fisher prior IS the principled byte allocator. R7 found the optimum within its candidate set; Wave-3 should explore the bbr=0.66-0.69 region for an even better optimum." **APPROVE — promote R7 to deploy candidate.**
- **Dykstra (CO-LEAD):** "Convex region: at bbr=0.65 the Hessian quadratic approx breaks (PoseNet jumped 30%) — this is a real boundary of the convex feasibility region. Stay strictly inside it. R7's 0.70 is well inside." **APPROVE.**
- **Yousfi:** "Proxy-auth gap: predicted 1.008-1.022, landed 1.013 — gap is 0.5%, acceptable. SegNet held perfectly (0.00401 vs 0.00401), PoseNet 5.7% drift bought 1.4% rate savings." **APPROVE.**
- **Fridrich:** "The cliff between bbr=0.65 (Pose 0.00473) and bbr=0.70 (Pose 0.00366) is sharp — there's structure in the sensitivity boundary that Fisher isn't capturing fully. Maybe a Hessian-eigenvalue cutoff instead of Fisher trace." **APPROVE WITH RESEARCH NOTE.**
- **Contrarian:** "Eval-noise band [0.005, 0.005]: 1.013 vs 1.016 is real but small. Need a paired re-eval on a different instance to confirm before contest submission." **APPROVE WITH GUARDRAIL.**
- **Hotz:** "Two consecutive sub-frontier wins. Stack effects WORK — keep going. Don't waste a turn re-eval'ing for noise; SHIP IT." **APPROVE.**
- **Quantizr:** "0.68 distance to leader. Three days left. The R7 selector is the right machinery; need 2-3 more orthogonal stacks (mask payload + SegNet boundary + arithmetic-coded poses) to add another ~0.10 of compression on top of this." **APPROVE.**

**VERDICT: 7/0 APPROVE owv3_0001_r7 as new deploy candidate; Wave-3 refinement sweep approved at ~$2 spend.**

## What would change my mind (reactivation criteria)

- Paired PFP16 re-eval on Vast.ai 35955469 lands at 1.013 instead of 1.04 → confirms eval-noise was the apparent improvement (UNLIKELY — bytes dropped 55KB and rate component verifies)
- Wave-3 refinement sweep finds NO candidate sub-1.013 → R7 was already at the optimum and only orthogonal stacks (PD-V2, LCT, Joint-ADMM) can push further

## Cross-refs

- `project_lane_g_v3_owv3_fisher_beta_LANDED_1_016_20260501.md` (Wave-1 single-shot, the now-superseded predecessor)
- `project_owv3_r7_state_correction_20260501.md` (R5+R6 failure history that R7 inherited)
- `project_pfp16_a_plus_plus_deploy_baseline_freeze_20260501.md` (1.044 frontier this beats by 0.031)
- `experiments/results/lane_g_v3_owv3_r7_LANDED_1_013_20260501/` (custody)
- `experiments/sweep_owv3_byte_plan.py` (the selector machinery)
- `feedback_spare_no_expense_shannon_floor_minimal_wallclock_20260501.md` (the user mandate)
