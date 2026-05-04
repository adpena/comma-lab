# C067 Decoded Delta Overlay Mask Topology - 2026-05-02

## Scope

Implemented a local-only decoded-baseline delta/overlay planner:

- `experiments/plan_c067_decoded_delta_overlay_mask_topology.py`
- schema: `c067_decoded_delta_overlay_mask_topology_plan_v1`
- output:
  `experiments/results/c067_decoded_delta_overlay_mask_topology_20260502/c067_decoded_delta_overlay_mask_topology_plan.json`
- score status: `score_claim=false`
- remote status: `remote_jobs_dispatched=false`

The planner preserves the C067 decoded mask tensor as global geometry and uses
donor mask tensors only as local overlay proposals. It emits a deterministic
charged payload spec only when all changed pixels are inside an explicit
pair/class trust region and no selected pair overlaps recent catastrophic
exact-negative pairs. It does not emit a contest archive because no reviewed
inflate runtime consumes the new `CDO1` overlay payload yet.

## Inputs

- Base decoded mask:
  `experiments/results/c063_trace_weighted_mask_grammar_plan_20260502_codex/decoded_mask_array.npy`
- Geometry-safety gate:
  `experiments/results/c067_geometry_safe_mask_topology_v2_20260502/c067_geometry_safe_mask_topology_v2_plan.json`
- Trust-region seed:
  `experiments/results/c067_postdecode_mask_repair_candidate_20260502/c067_postdecode_mask_repair_waterfill_pair_class_plan.json`
- Donor decoded masks:
  `cmg3a_body200`, `cmg3_nonzero_top2`, `cmg3_nonzero_top1`,
  and `cmg3_rowspan_stride1` from
  `experiments/results/c067_multimask_reconciliation_20260502/`

## Result

The planner reviewed 4 donor overlay candidates and emitted 0 safe payload
specs.

Trust-region pairs derived from the postdecode pair/class waterfill plan:

```text
79, 153, 212, 216, 230
```

All 5 trust-region pairs overlap the catastrophic pair set in the recent
exact-negative geometry plan. Therefore every donor overlay failed closed.

Local byte/pixel screens before the veto:

| Candidate | Selected pixels | Selected fraction | Best payload bytes | Blocker |
|---|---:|---:|---:|---|
| `cmg3_rowspan_stride1` | 2610 | 0.000022125244 | 2176 | catastrophic pairs 79,153,212,216,230 |
| `cmg3_nonzero_top2` | 4853 | 0.000041139391 | 3296 | catastrophic pairs 79,153,212,216,230 |
| `cmg3a_body200` | 4994 | 0.000042334663 | 3532 | catastrophic pairs 79,153,212,216,230 |
| `cmg3_nonzero_top1` | 8283 | 0.000070215861 | 4344 | catastrophic pairs 79,153,212,216,230 |

## Interpretation

This is useful negative signal. The overlay payload contract is now concrete:
`CDO1` records the base tensor SHA-256, donor tensor SHA-256, selected
pair/class trust region, run struct, selected pixel count, and deterministic
payload hashes. But the current high-prior trust region is not safe under the
new exact-negative pair veto.

No candidate became dispatchable. The next valid local move is not to relax the
veto; it is to find a trust region whose selected pairs survive the
exact-negative set, or to produce a stricter exact-negative classification that
distinguishes global topology-collapse pairs from local decoded-overlay atoms.

## Verification

Commands run:

```bash
.venv/bin/python -m py_compile experiments/plan_c067_decoded_delta_overlay_mask_topology.py src/tac/tests/test_plan_c067_decoded_delta_overlay_mask_topology.py
.venv/bin/python -m pytest src/tac/tests/test_plan_c067_decoded_delta_overlay_mask_topology.py -q
.venv/bin/python experiments/plan_c067_decoded_delta_overlay_mask_topology.py --force
```

Observed focused pytest result: `4 passed`.

No remote dispatch was performed and no SJ-KL files were touched.
