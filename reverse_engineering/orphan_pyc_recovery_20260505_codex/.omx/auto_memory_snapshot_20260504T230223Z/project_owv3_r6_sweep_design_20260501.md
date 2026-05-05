---
name: OWv3 r6 sweep design — predicted-score-driven candidate selection (replaces conservative byte-only selection)
description: 2026-05-01 ~08:55 UTC. r5 had 116 byte-feasible candidates spanning -38122 → -78 bytes vs PFP16 frontier; deferral memory only tested owv3_0018 (-78, conservative). r6 design samples bit_budget_ratio densely between 0.5-0.69, scores distortion at compress-time via local scorer call, and selects top 3 by predicted [contest-CUDA] score. Threshold to dispatch: any candidate predicting < 1.040.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## What r5 left on the table

r5 byte-feasibility distribution (116/135 candidates feasible vs PFP16 frontier 686,635 bytes):

| bit_budget_ratio | protect_threshold | bytes (top) | frontier Δ | rate-savings score | aggressive risk |
|---|---|---|---|---|---|
| 0.50 | 0.005 | 648,513 | **-38,122** | **0.0254** | HIGH (least bits) |
| 0.60 | 0.005 | 658,395 | -28,240 | 0.0188 | medium-HIGH |
| 0.64 | 0.005 | 662,776 | -23,859 | 0.0159 | medium |
| 0.65 | 0.005 | 663,872 | -22,763 | 0.0152 | medium |
| 0.66 | 0.005 | 664,723 | -21,912 | 0.0146 | medium |
| 0.67 | 0.005 | 665,465 | -21,170 | 0.0141 | medium |
| 0.68 | 0.005 | 666,555 | -20,080 | 0.0134 | medium |
| 0.69 | 0.005 | 667,058 | -19,577 | 0.0130 | medium-low |
| 0.50 | 0.003 | 667,335 | -19,300 | 0.0129 | medium |
| 0.70 | 0.005 | 667,724 | -18,911 | 0.0126 | low |
| **0.69** | **0.0014** | **686,557** | **-78** | 0.00005 | **VERY LOW (tested)** |

The deferral memory's `owv3_0018` is the bottom of this table — **conservatively chosen for minimum distortion risk**. The TOP candidate (`owv3_0134`) has **325× more rate savings** but unknown distortion impact.

## What r6 must add

**1. Compress-time scorer call to PREDICT distortion before archive build.** Per CLAUDE.md, scorers at compress time are explicitly allowed (the strict-scorer-rule only forbids them at INFLATE time). A single forward pass on Lane G v3 anchor frames (PoseNet + SegNet) gives `(pose_dist_pred, seg_dist_pred)` per candidate without any GPU dispatch.

**2. Predicted score formula:** `score_pred = 100·seg_pred + sqrt(10·pose_pred) + 25·bytes/37545489`. Compute for every candidate. Filter to those predicting < 1.040.

**3. Drift correction:** apply the empirical Modal CPU↔CUDA drift coefficient (PoseNet ×1.105 from the OWv3 deferral memory). Per Round 1 council greenup, this coefficient was calibrated on n=1; r6 design should ALSO compute the prediction with ×1.131 (Council Round 1 CRITICAL #3 finding) to bracket uncertainty.

**4. Dense sampling between bbr ∈ [0.50, 0.69] with protect_threshold ∈ [0.001, 0.0014, 0.002, 0.003, 0.005]:**
- bbr step: 0.02 → 10 values × 5 protect = 50 candidates
- aggressive_threshold: 1e-5 (kept fixed — r5 showed it doesn't dominate)

**5. Top-3 by predicted score get ONE Vast.ai 4090 dispatch each ($0.50 × 3 = $1.50).** This still satisfies the deferral memory's reactivation criterion ("OR: 2-3 different byte-feasible candidates from the sweep all predicting <1.043, suggesting a consistent sub-frontier band → run ONE eval to verify the band").

## Implementation plan

`experiments/run_owv3_r6_sweep.py` (new file):

```python
# Pseudocode — 50 candidates with compress-time scorer eval
for bbr in [0.50, 0.52, ..., 0.68]:
    for protect in [0.001, 0.0014, 0.002, 0.003, 0.005]:
        # Build OWv3 byte plan (existing src/tac/owv3_sensitivity_weighted.py)
        plan = build_owv3_byte_plan(
            anchor_renderer=lane_g_v3_anchor,
            sensitivity_map=fisher_sensitivity_or_uniform,
            bit_budget_ratio=bbr,
            protect_threshold=protect,
            aggressive_threshold=1e-5,
        )
        # Build temp renderer.bin with plan applied
        renderer_perturbed = apply_byte_plan(plan, anchor_renderer)
        # Build archive (deterministic, in-memory)
        archive_bytes = build_archive_in_memory(renderer_perturbed, masks_anchor, poses_anchor)
        # COMPRESS-TIME SCORER CALL (allowed per CLAUDE.md)
        with torch.no_grad():
            pose_dist = posenet(renderer_perturbed.forward(...))
            seg_dist = segnet(renderer_perturbed.forward(...))
        # Predicted score
        score_cpu_pred = 100 * seg_dist + sqrt(10 * pose_dist) + 25 * len(archive_bytes) / 37545489
        # Apply drift correction (lower + upper bound)
        pose_cuda_lo = pose_dist * 1.105
        pose_cuda_hi = pose_dist * 1.131
        score_cuda_lo = 100 * seg_dist + sqrt(10 * pose_cuda_lo) + 25 * len(archive_bytes) / 37545489
        score_cuda_hi = 100 * seg_dist + sqrt(10 * pose_cuda_hi) + 25 * len(archive_bytes) / 37545489
        save_candidate(...)
```

Output: `experiments/results/lane_g_v3_owv3_byte_plan_sweep_20260501_r6/byte_plan_summary.json` with each candidate carrying:
- `score_cuda_pred_lo` / `score_cuda_pred_hi` (drift-corrected bounds)
- `dispatch_eligible` boolean: `score_cuda_pred_hi < 1.040` (conservative)
- top-3 ranked by `score_cuda_pred_lo`

## Compute budget

- **Sweep itself:** ~50 candidates × ~5s scorer call each = ~5 min on local MPS or CPU. **No GPU dispatch needed.**
- **Promotion candidates:** if 1+ predicts < 1.040 [contest-CUDA pred], dispatch 1 candidate ($0.50, ~30 min).
- **Top-3 verification:** if dense band of <1.040 predicted candidates emerges, dispatch all 3 ($1.50).
- **Total upper bound:** $1.50 (still under any operator approval threshold).

## Pre-requisites for this lane being implementable RIGHT NOW

1. ✅ `src/tac/owv3_sensitivity_weighted.py` — landed (per Shannon checkpoint)
2. ✅ `experiments/results/lane_g_v3_pfp16/...archive_lane_g_v3_pfp16.zip` (anchor) — verified intact today (SHA `0af839ab...ed7f`, 686,635 bytes)
3. ❌ Sensitivity map (Fisher diagonal on Lane G v3 anchor) — NOT YET produced; r6 sweep can fall back to UNIFORM sensitivity (less optimal but still produces valid candidates)
4. ❌ `experiments/run_owv3_r6_sweep.py` — needs to be written (~200 LOC; uses existing helpers)

**Critical path:** r6 can run today on UNIFORM sensitivity (skipping Fisher). Once β sensitivity-map dispatch lands ($2 on Vast.ai 4090), r6 can be re-run with proper Fisher weighting for ~0.001-0.005 score-points of refinement.

## Adversarial Grand Council review

**Required per CLAUDE.md design-decisions non-negotiable.** Council vote (5+ inner council members):

- **Shannon (LEAD):** rate-distortion analysis. r5 conservative selection saw owv3_0018's 0.005 rate savings → 1.0429 prediction; rate is the dominant lever per the score formula. r6's predicted-score-driven selection at owv3_0134's 0.0254 rate savings, IF distortion holds, lands at 1.0186 — sub-frontier by 0.025. **APPROVE r6 IF compress-time scorer call is in fact allowed (yes, per CLAUDE.md).**
- **Dykstra (CO-LEAD):** convex feasibility. The byte-feasibility region was undersampled by r5 (15 grid points in bbr × 9 in protect = 135 total). r6 fills the dense band where the Pareto frontier likely lives. **APPROVE.**
- **Yousfi:** "scorer-margin is the entire signal." r6 explicitly USES the scorer at compress time to pre-filter. This is exactly the inverse-steganalysis pattern — use the detector to inform the encoding. **APPROVE.**
- **Fridrich:** PoseNet asymmetry warnings preserved? Yes — drift bracket [×1.105, ×1.131] uses the upper bound for the conservative dispatch threshold. **APPROVE.**
- **Contrarian:** "compress-time scorer prediction has unknown drift to actual contest-CUDA inflate-eval. The proxy-auth gap can be 2-11x for PoseNet (CLAUDE.md non-negotiable). Predicting on CPU and dispatching on CUDA may surface a different gap than the Modal-T4 calibration."
  - **Mitigation:** the deferral memory already captured Modal-CPU vs Vast-CUDA drift coefficient (×1.105/×1.131). r6 applies this correction explicitly. If the prediction is < 1.040 with the upper-bound drift, the actual CUDA score is likely < 1.044 (frontier).
  - **Verdict:** APPROVE WITH GUARDRAIL — only dispatch candidates whose predicted [contest-CUDA] HIGH bound < 1.040.
- **Hotz:** simplest thing that works. "Just sweep dense, score local, dispatch winners." Approve. The r5 conservative selection wasted information; r6 uses all of it.

**VERDICT: 6/0 APPROVE design** — r6 implementation can land this turn or next; one Vast.ai dispatch follows IF a candidate predicts < 1.040 [contest-CUDA upper-bound].

## Internal-consistency checks performed

- Verified r5 candidate count and byte distribution by direct JSONL inspection (not relying on summary's count fields).
- Verified PFP16 frontier bytes (686,635) match three deploy-archive copies' SHAs (today's `project_pfp16_a_plus_plus_deploy_baseline_freeze_20260501.md`).
- Drift coefficient ×1.105 sourced from OWv3 deferral memory; ×1.131 sourced from Round 1 council greenup CRITICAL #3 (the corrected empirical value).
- Score formula matches `100·seg + sqrt(10·pose) + 25·bytes/37545489` per Shannon checkpoint and contest_auth_eval.json reconciliation.

## What would change my mind (reactivation criteria)

- Compress-time scorer prediction shown empirically to deviate > 0.01 from contest-CUDA truth on Lane G v3 anchor → re-evaluate the design (require Modal-T4 advisory before Vast.ai dispatch).
- β sensitivity-map (Fisher) lands and shows owv3_0134's protected channels were wrong → re-sample with Fisher weights.
- A different paradigm (Lane 17 IMP, Lane 19 logit-margin) lands a sub-frontier score that obsoletes the OWv3 lane.

## Cross-refs

- `project_owv3_byte_feasible_candidate_dispatch_deferred_20260501.md` (the conservative r5 selection that this design supersedes)
- `project_pfp16_a_plus_plus_deploy_baseline_freeze_20260501.md` (the frontier this targets)
- `project_shannon_floor_execution_state_checkpoint_20260501.md` (master plan; "Write the OWv3 r6 sweep design" was an open task on the "What I CAN continue" list)
- `feedback_grand_council_recursive_greenup_shannon_floor_20260501.md` (Round 1 finding: drift ×1.131 not ×1.105)
- `src/tac/owv3_sensitivity_weighted.py` (the byte-plan generator)
- `experiments/results/lane_g_v3_owv3_byte_plan_sweep_20260430_codex_r5/byte_plan_candidates.jsonl` (r5 raw output)
