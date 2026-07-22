---
schema: dag_feed.v1
utc: 2026-07-22T01:34:04Z
task: 603
lane_id: lane_ddm_target_receipt_pose_rung0_603_20260722
research_only: true
execution_allowed: false
evidence_axis: "[macOS-CPU real-target subset n64 apparatus]"
verdict_scope: exact existing target custody and bounded n64 apparatus only
---

# DAG FEED — DDM target receipt and Pose rung zero

## Producers

- Existing source: complete C1 solved-pair Y0/Y1 scorer-resolution planes plus GT Pose6 cache.
- Typed custody/materializer: `tac.optimization.direct_description_real_target_rung0`.
- Local-only consumer: `tools/run_direct_description_real_target_rung0.py`.

## New typed signals

- `direct_description_full_precision_target_planes.v1`: hashes every source receipt, official
  provenance record, archive/cache, 50 chunk manifests and 100 plane files; binds aggregate/source
  geometry, original producer path/git/source SHA, current upstream snapshot/evaluator SHA, and the
  deterministic n64 projection/Pose6-code recipes.
- `DirectDescriptionRealTargetRung0ConfigV1`: seed-1234, n64, exact target-receipt SHA, three explicit
  `candidate_search` stages, `execution_allowed=false`, `research_only=true`, `score_claim=false`.
- `DirectDescriptionRealTargetCheckpointV1`: canonical JCS envelope binding typed config/DSL/argv,
  target receipt/projection/Pose6 codes, current archive/output, objective, optimizer cursor, and full
  history at every stage.
- `direct_description_real_target_pose_rung0.v1`: exact integer-debt trajectory, producer code
  custody, three primary plus three resume checkpoint hashes, and terminal byte-identity receipt.

## Six-hook disposition

1. Sensitivity map: no scorer sensitivity row is emitted; the measured changes are apparatus debt.
2. Pareto constraint: exact archive bytes are counted, but d_seg/d_pose are absent, so no contest
   Pareto or score admission occurs.
3. Bit allocator: the Pose stage demonstrates a real target-code term is callable; it does not infer
   score-unit value per byte.
4. Cathedral/autopilot: no dispatch edge; execution, candidate, and score flags remain false.
5. Continual learning: blocker-register custody is updated, but no empirical posterior is updated
   from non-scorer diagnostics.
6. Probe disambiguator: target hash/source geometry, checkpoint reload, stopped-then-resumed execution,
   and primary/resume byte comparison are the fail-closed disambiguators.

## Edge-state delta

- `full_precision_sha_bound_target_receipt`: `RED -> GREEN_CUSTODY_SCOPE`.
- `fresh_v3_family_pose_in_objective_rung_zero`: `RED -> GREEN_CUSTODY_SCOPE`.
- Fixed register total: `4/19 -> 6/19` scoped green.
- `four_rung_ladder`, `n600_same_artifact_closure`, `contest_cpu`, `contest_cuda`, completion,
  attestation, operator GO, and PRIMARY execution remain red.

Pointer `0.1910828242 [contest-CPU]` unchanged. No scorer, evaluator, remote execution, provider,
paid dispatch, candidate archive, or score authority was used.
