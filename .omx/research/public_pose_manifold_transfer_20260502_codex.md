# Public Pose Manifold Transfer Planning - 2026-05-02

## Scope

Worker E implemented a deterministic comparison/planning surface for PR65,
PR67, and C-059 pose-manifold transfer. The tool is:

- `experiments/compare_public_pose_manifolds.py`

The generated planning artifact is:

- `experiments/results/public_pose_manifold_transfer_20260502/pose_manifold_transfer_policy.json`

## Evidence Boundary

- `evidence_grade`: `diagnostic_planning_non_promotable`
- `score_claim`: `false`
- `promotion_eligible`: `false`
- `no_remote_jobs_launched`: `true`

The artifact uses formula terms only as planning utility. It does not build an
archive, launch remote work, run an eval, claim a score, rank a submission, or
promote/retire a method. Any selected policy must become a closed archive and
pass exact CUDA auth eval on identical bytes before it can support a score or
paper claim.

## Inputs

- Public PR summary:
  `experiments/results/top_submission_reverse_engineering_refresh_20260502/public_pr_summary.json`
- Public PR anatomy:
  `experiments/results/top_submission_reverse_engineering_refresh_20260502/archive_anatomy.json`
- C-059 auth eval:
  `experiments/results/lightning_batch/exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z/contest_auth_eval.json`
- C-059 component trace:
  `experiments/results/lightning_batch/exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z/component_trace.json`
- C-059 pose plan:
  `experiments/results/pose_atom_plan_c059_20260502/pose_atom_policies.json`
- PR65/PR67 atom ledger:
  `experiments/results/top_submission_reverse_engineering_20260502T0206Z/pr65_pr67_atom_allocation_ledger.json`

## Output Summary

The policy JSON compares PR65 and PR67 against C-059 with explicit evidence
grades, byte notes, and risk notes. It emits top-16, top-32, and top-64
non-promotable pair-index policies. The planner records PR65 as a global pose
manifold signal with no pair-local trace and PR67 as the better pair-transfer
source where local PR67/C-059 deltas exist.

The generated policy retained 64 ranked opportunities. The top policies remain
formula-only planning proposals and carry `requires_exact_cuda_stack_eval=true`.

## Verification

Run locally:

```bash
.venv/bin/python -m py_compile experiments/compare_public_pose_manifolds.py src/tac/tests/test_compare_public_pose_manifolds.py
.venv/bin/python -m pytest src/tac/tests/test_compare_public_pose_manifolds.py
.venv/bin/python experiments/compare_public_pose_manifolds.py
jq -e '.score_claim == false and .promotion_eligible == false and .no_remote_jobs_launched == true and ([.. | objects | select(has("score_claim")) | .score_claim] | all(. == false))' experiments/results/public_pose_manifold_transfer_20260502/pose_manifold_transfer_policy.json
git diff --check -- experiments/compare_public_pose_manifolds.py src/tac/tests/test_compare_public_pose_manifolds.py .omx/research/public_pose_manifold_transfer_20260502_codex.md experiments/results/public_pose_manifold_transfer_20260502/pose_manifold_transfer_policy.json
```
