Implemented and landed the real QBFLOW scorer-in-loop trainer in commit `755f31e9ef2b2f09ce5ac3a0a1efc42b8907476d`.

Key artifacts:

- [Trainer](/Users/adpena/Projects/pact/experiments/ddm_qbt1_qbflow_trainer.py)
- [Focused tests](/Users/adpena/Projects/pact/experiments/tests/test_ddm_qbt1_qbflow_trainer.py)
- [Charter receipt](/Users/adpena/Projects/pact/.omx/research/ddm_qbt1_qbflow_trainer_build_20260827.md)
- [Bounded-smoke result](/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/smoke_n1_20260827_final/RESULT.json)
- [Compiled launch request](/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/COMPILED_LAUNCH_REQUEST.json)

Measured:

- `8 passed`; Ruff, py_compile, and payload-retention gate passed.
- Two visible review passes covered 53 trainer and 9 test entities per pass, without override.
- n=1 CPU smoke completed in 17.46 s, retaining 189 MiB across 1,269 files.
- Checkpoint resume reproduced the exact 107,539-B archive, SHA-256 `0bde7801…57ebe`.
- n32 materialization projection: 92,084,098,826 B, giving 32,469,952,758 B projected headroom below the 116-GiB ceiling.
- Draft config validates; live launch correctly refuses because lanes and authorization are absent.
- No Metal, Modal, n600 scorer, or contest evaluation was launched.

The request disposition is `QUEUED_STAGE03_04_FIRE_STAGE05_BLOCKED`. The existing QBW1 rate receipt is not accepted as a scorer control because it lacks the required real renderer/scorer outputs and 32 retained payloads. Existing unrelated AU1, lane-registry, and WD3 worktree changes remain untouched.

## NEXT_IF_RESUMED

- **Disposition: QUEUED.** Owner: MAIN QBFLOW joint-training owner. Consumer store: `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow`. Fire trigger: verify commit/review hashes, confirm no duplicate lane or full-n600 scorer job, claim Metal and scorer lanes, and rerun live storage plus ≤116-GiB admission before stages 03/04.
- **Disposition: QUEUED-WITH-A-FIRE-ORDER.** Owner: MAIN QBW1 control owner. Consumer store: QBT1 stage-05 control receipt input. Fire trigger: the governed QBFLOW n32 budget is known; materialize a real same-budget QBW1 archive and all 32 scorer payloads with custody and recomputed arithmetic.
- **Disposition: BLOCKED.** Owner: MAIN stage-05 admission owner. Consumer store: `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow`. Fire trigger: both the governed n32 QBFLOW result and validated same-budget QBW1 control exist.

## LIVE-HYPOTHESES

- Joint pose6 descent from update zero may escape the historical post-hoc pose wall because official PoseNet gradients now shape the rendered interiors from birth.
- Boundary-flow and step-transition prequantization may reduce trained rate; both reduced real smoke archive bytes, but require realized scorer A/Bs.
- The equal-mass 16-pair job may fit Metal memory because the conservative projection is below the sealed ceiling; this remains unmeasured on Metal.

## DEAD-ENDS

- The 30+2 optimizer split is closed because it overweights the two-pair chunk; use the equal-mass 16+16 schedule.
- The current QBW1 stage-02 rate receipt is not a same-budget scorer control.
- Precision choices based only on first-order sensitivity are closed; no adoption without realized scorer survival.
- Fixed-high-beta hosc and serialized boundary payloads remain closed by prior evidence and the frozen QBF1 ABI.

Own-vehicle frontier unchanged: gb1 — S 0.14811799921260607 @ 180,215 B `[contest-CUDA T4 n600]`.