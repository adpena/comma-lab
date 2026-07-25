# Codex session summary — DDM EV2 per-pair allocation producer

Date: 2026-07-25 UTC  
Lane: `ddm_ev2_per_pair_allocation_producer`  
Authority: research-only; MAIN review required

## Landed

- SHA-bound EV2 materializer and focused regression suite.
- 162-cell allocation table, 600 per-pair rows, strict MS5-loader projection,
  RD1 null-costate backfill, and canonical headline replay.
- Registered MS4D post-admission preflight receipts.
- Findings memo with construction-lineage proof and scoped falsifier.

## Result

Exact construction lineage permits no exclusive final-byte
`{pair, scorer-derived cell}` ownership in the current jointly coded C1
object. The full LP1 mass is conserved as typed `UNALLOCATED`
(`134,211 / 134,211 B`), the 30% falsifier fires at `FORMULATION` scope,
`0 / 162` lambda values become computable, and all four headline blockers
remain.

The MS3 complete bundle was admitted. The registered waterfill gate then
refused `PF3_RECEIVER_OBJECT_AND_TYPED_RATE_HOME_ABSENT`; no full solve,
launch, score, promotion, or pointer movement occurred.

## MAIN review focus

Review the same-object separation from EV1 accounting bytes, the exact ZIP
home proof, strict MS5 projection, falsifier scope, and the coarser seven-home
successor before merging.
