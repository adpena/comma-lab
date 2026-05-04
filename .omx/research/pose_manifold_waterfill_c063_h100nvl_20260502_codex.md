# Pose Manifold Water-Fill Candidates - 2026-05-02 Codex

## Evidence Boundary

- Evidence grade: `diagnostic_planning_non_promotable`.
- Score claim: `False`.
- Required score truth: `archive.zip -> inflate.sh -> upstream/evaluate.py via experiments/contest_auth_eval.py --device cuda`.
- H100/L40S/A100 diagnostics remain diagnostic until identical bytes pass T4/equivalent CUDA auth eval.

## Frontier Anchor

- Label: `C-063-H100NVL`.
- Archive bytes: `276223`.
- Archive SHA-256: `83615afd130afa08e972e4a02476612397bffea53327caf3591891f8317aa52d`.
- Score recomputed from components: `0.31500175622241344`.
- Avg PoseNet: `0.0004909`.
- Avg SegNet: `0.00061012`.

## Output Artifacts

- Plan JSON: `experiments/results/pose_manifold_waterfill_c063_h100nvl_20260502_codex/pose_manifold_waterfill_plan.json`.
- Dispatch recommendations: `experiments/results/pose_manifold_waterfill_c063_h100nvl_20260502_codex/exact_eval_recommendations.json`.

## Top Recommendation

- Candidate: `ls_c059_weighted_pairs_top32_h100`.
- Status: `ready_for_t4_confirmation`.
- Archive SHA-256: `877fc5ac13e9fbd5c4158a9c7fa9dec3354057522b086004a4a28c6822456fe8`.
- Archive bytes: `276423`.
- Requires T4 confirmation: `True`.
- Exact eval command:

```bash
.venv/bin/python -u experiments/contest_auth_eval.py --archive /Users/adpena/Projects/pact/experiments/results/vast_harvest/archive_eval_ls_c059_weighted_pairs_top32_h100_20260502/archive.zip --inflate-sh submissions/robust_current/inflate.sh --upstream-dir upstream --device cuda --keep-work-dir --work-dir experiments/results/pose_manifold_waterfill_c063_h100nvl_20260502_codex/ls_c059_weighted_pairs_top32_h100_t4_eval_work
```

- Dispatch guard: claim the lane first and do not duplicate an active Lightning T4 promotion claim.

## Build-Only Macro Specs

- Macro specs emitted: `4`.
- These specs require a closed archive builder before exact eval; their expected utility is formula-only planning signal.

