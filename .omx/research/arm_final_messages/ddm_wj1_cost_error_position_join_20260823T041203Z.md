Implemented, measured, retained, independently replayed, and committed WJ1 as `72975fcaa1`.

Key result: cost and render-manufactured error are strongly associated on DX2.

- Gross manufactured support: 28,602 positions carrying 6,846.84 modeled bytes.
- Top 1% cost set captures 26,016 positions and 6,841.83 modeled bytes.
- Versus independence: 90.96× count enrichment and 257.48× bit enrichment.
- Association remains positive within every class, including Lane.
- The 22,321 headline is a net stage delta, not a position set; the valid membership mask has 28,602 gross transitions.
- This is incumbent modeled mass, not realized savings. No score or frontier claim was made.

Artifacts:

- [Full measurement report](/Users/adpena/Projects/pact/.omx/research/ddm_wj1_cost_error_position_join_20260823.md:1)
- [Resumable measurement runner](/Users/adpena/Projects/pact/experiments/ddm_wj1_cost_error_position_join.py:1)
- [Completion receipt](/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_wj1_cost_error_position_join/measurement_v1/COMPLETED_VERIFICATION.json)
- [JF1 target position list](/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_wj1_cost_error_position_join/measurement_v1/retained/targets/top_10pct_render_manufactured_positions.npy)

The retained store contains 27 artifacts totaling 1,283,879,716 bytes. Independent replay verified every threshold/class cell and target-list column. Ruff, formatting, compilation, self-test, two recorded review passes, deterministic resume, and the bounded payload-retention audit passed. Repository-wide developer preflight still reports eight unrelated shared-tree/global failures; neither WJ1 file is implicated.

OWN-VEHICLE FRONTIER: UNMOVED — DX2 remains `S=0.14821987563243377`, 180,368 bytes, contest-CUDA T4 n600.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: `ddm_jf1_joint_field_model_refit`; consumer store: `.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/wj1_target_consumer/`; fire trigger: WJ1 status is `COMPLETE` and the consumer independently matches position-list SHA-256 `bb1c42698e38deb94d9bee8edbdf44261a40a95554defef38d6088730be5da7d`. Run a retained field-coarsen plus real model-refit rung against JF1’s null-refit control.

## LIVE-HYPOTHESES

- Cost×futility ordering may outperform cost-only ordering after HPAC refitting because its top-1% bit enrichment is 257.48× independence.
- The later-repaired subset may be the best first rung because 2,249.70 of its 2,252.10 modeled bytes lie in the top 1%.
- Class-aware caps may outperform Lane-only targeting because 69.39% of gross-manufactured modeled mass lies outside Lane.

## DEAD-ENDS

- Position-level independence is closed on this DX2 instance.
- A class-only explanation is closed because every class retains within-class enrichment.
- Treating `+22,321` as a membership set is closed; it is only a net difference.
- Fixed-model harvesting is closed by LD1: all six tested coarsenings increased archive size.
- Coder swaps, reordering, and storage-layout attacks remain closed and were not reopened by WJ1.

