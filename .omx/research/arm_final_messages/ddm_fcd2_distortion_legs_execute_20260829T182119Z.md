The fcd1 union is `INSTANCE-REFUSED-POSE-GATE`; the family remains open.

- Fresh n600 compensation plus full diminishing-returns refinement reached `d_pose=0.00027348054805362656`, versus base `0.0000063656845167356244`—42.96× worse.
- Two closes were byte-identical: `176,463 B`, SHA `d4f6b932…`.
- Final exact rate saving was 3,729 B, but publication failed. Therefore full scorers, seals, and Modal dispatch were correctly not fired; net ΔS was not measured.
- The RR5/DX2 splice incompatibility and missing refusal receipt were permanently repaired. Six tests pass with two review passes per Python file.
- Handoff: [ddm_fcd2_distortion_legs_execute_20260829.md](/Users/adpena/Projects/pact/.omx/research/ddm_fcd2_distortion_legs_execute_20260829.md)
- Commit: `1326458d5bb8d44fa20559b904a0f8513939d306`
- Own frontier unchanged: `S=0.14811799921260607 @ 180,215 B [contest-CUDA T4, n600]`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_fcd1_field_for_coder_diagonal/`; fire trigger: a new candidate-bound carrier produces two identical union closes satisfying `d_pose_after <= base + 1e-8`. Then publish and run the two full scorers.
- **FOLDED** — owner: MAIN; consumer store: same; fire trigger: the published union’s realized refusal is ≤5× the canonical band. Then evaluate batch0, batch2, and batch1 independently.
- **FOLDED** — owner: MAIN; consumer store: same; fire trigger: three independent batches each refuse by >5×. Only then record the #1295 family closure.

## LIVE-HYPOTHESES

- A richer pose carrier may clear the hard tail: the current 12-dimensional carrier reached int12 limits and 597/600 refinement rows stopped with no improving step.
- Individual batches may be pose-safe because each has roughly one-third the union’s edit density and carrier costs are non-additive.
- Pose-aware token selection may retain some byte savings while avoiding edits transverse to the candidate’s pose Jacobian.

## DEAD-ENDS

- The exact union with the current fresh GN/refinement carrier is closed at INSTANCE scope: repeat-identical pose remains 42.96× base.
- More iterations of the same refinement are closed: no solve budget was exhausted, and refinement moved mean pose only `2.7875e-6`.
- Parsing RR5/DX2 carrier bytes as plain Rice is closed as an apparatus defect and now regression-tested.
- Carried compensation, estimated/additive rate credits, and treating B/H labels as realized SegNet flips remain invalid.
- Scoring, sealing, or dispatching a publish-refused archive is closed by the charter’s ordering contract.