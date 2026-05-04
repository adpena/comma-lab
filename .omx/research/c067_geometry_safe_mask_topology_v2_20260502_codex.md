# C067 Geometry-Safe Mask/Topology V2 Gate

Date: 2026-05-02
Tool: `experiments/plan_c067_geometry_safe_mask_topology_v2.py`
Output:
`experiments/results/c067_geometry_safe_mask_topology_v2_20260502/c067_geometry_safe_mask_topology_v2_plan.json`

## Purpose

After the same-family PMG, CMG3A multimask, micro-mask reencode, and AMR1
postdecode repair exact negatives, do not dispatch another byte-only mask or
topology archive unless it passes a geometry/pose-safety gate first.

The gate is planning-only. It loads exact CUDA component traces and refuses:

- identical archive SHA repeats of measured exact negatives;
- global PMG/CMG3/CMG3A topology replacement bases already shown to collapse
  PoseNet/SegNet;
- selected atoms that overlap catastrophic exact-negative pair indices;
- candidates without byte-closed archive provenance;
- candidates above the unchanged-distortion sub-0.300 byte gate;
- active dispatch conflicts from `.omx/state/active_lane_dispatch_claims.md`.

## Live Result

The live run reviewed 26 candidates and surfaced 0 dispatchable archives.

This is the correct aggressive decision: the current non-retraining
mask/topology family is not under-tested; it is over-risked. The exact negative
traces show broad PoseNet collapse from global geometry changes:

- CMG3 top1/top2 and CMG3A body candidates collapse across nearly all pairs.
- PMG/topology candidates are byte-attractive but exact CUDA negative.
- Multimask reconciler variants collapse without pair/atom-level pose-safety
  evidence.
- Micro-mask and AMR1 repair candidates remain exact-negative and should be
  treated as trace generators, not as primary sub-0.300 dispatches.

## Grand-Council Priority Decision

Do not launch another same-family PMG/multimask/micro-mask/AMR1 topology job
from the existing candidate pool.

The next non-training candidate design must be a decoded-baseline delta or
overlay that preserves global C067 geometry, records pixel-disagreement inside
a strict trust region, avoids catastrophic exact-negative pair indices, and
ships as a byte-closed archive before exact CUDA diagnostic. If a candidate
cannot satisfy those conditions locally, it should not reach remote exact eval.

## Verification

- `.venv/bin/python -m py_compile experiments/plan_c067_geometry_safe_mask_topology_v2.py src/tac/tests/test_plan_c067_geometry_safe_mask_topology_v2.py`
- `.venv/bin/python -m pytest src/tac/tests/test_plan_c067_geometry_safe_mask_topology_v2.py -q`
  - `6 passed in 0.08s`
- `.venv/bin/python experiments/plan_c067_geometry_safe_mask_topology_v2.py --force`
  - `candidate_count=26`
  - `dispatchable_candidate_count=0`
  - `decision=no_same_family_remote_dispatch`

## Reactivation Criteria

This lane can be reactivated only by one of:

1. A byte-closed decoded-baseline delta/overlay archive below the sub-0.300 byte
   gate with selected atoms outside catastrophic exact-negative pairs.
2. A pose-regenerated geometry proof that converts the global topology change
   from a stale-pose failure into a component-safe exact CUDA diagnostic.
3. A learned topology export with the required Lane12/L2 clearance and exact
   archive custody.

No broad method kill is claimed. The measured implementation families are
blocked from duplicate remote dispatch until the safety gate is satisfied.
