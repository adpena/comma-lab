The v-family harvest is complete: 121/121 roots read and routed, 0 remaining. I found no unconsumed current-floor candidate; every result was folded into an existing consumer.

Artifacts:

- [Harvest receipt](/Users/adpena/Projects/pact/.omx/research/ddm_vh2_vehicle_harvest_drain_20260813.md) — SHA-256 `d4057acb…`
- [Routed-artifact ledger](/Users/adpena/Projects/pact/.omx/research/ddm_vh2_vehicle_harvest_drain_20260813_routed_artifacts.jsonl) — 121 artifact rows, 22 group rows, SHA-256 `de939c5c…`
- [Verified commit bundle](/Volumes/APDataStore/pact/ddm_vh2_20260813/6b1353e89e.bundle) — final serializer head `6b1353e89ef705c7e5cfba259b0a9e1bc62747c9`, 20,417 B, SHA-256 `9a6cc578…`

The shared checkout’s Git database rejected writes with `Operation not permitted`, so the two artifacts remain untracked there. The shared index is empty; the unrelated dirty canonical ledger was preserved. Serializer custody was completed in a clean fallback clone and retained in the verified SSD bundle. Temporary clones totaling about 2 GB were deleted afterward.

Measured now: exhaustive filesystem coverage, bytes, hashes, read surfaces, consumer joins, and ledger custody. Not measured: rendering, R, SegNet, PoseNet, complete S, or contest evaluation. No scorer, trainer, Modal job, payload materialization, deletion, or pointer movement occurred.

The strongest apparent leads were already consumed:

- V19/V19c primitives → G1, current correction DSL, and task 1029.
- V13/V14 worldsheet evidence → canonical describe-line equation and G1.
- V4d dim-0 pose offset → current CP1/PJ2 pose stack.

Own-vehicle frontier remains LC2 `S=0.16959899569230852 @ 187,226 B` `[contest-CUDA T4, n600]`. Effective/local floor remains CP135 `S=0.16195513827824176 @ 186,252 B`.

## NEXT_IF_RESUMED

- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: VH2 ms-family successor. Consumer store: a hash-pinned ms routed-artifact JSONL beside the receipt. Fire trigger:** declare and exhaustively read the current 49-root ms frame after loading CN5 exclusions and existing ms consumers.
- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: VH2 j-family successor. Consumer store: a hash-pinned j routed-artifact JSONL beside the receipt. Fire trigger:** the ms frame is terminal or explicitly skipped, then declare and read all 45 j roots.
- **Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: MAIN probe-outcomes ledger custodian. Consumer store: `.omx/state/probe_outcomes.jsonl`. Fire trigger:** separately custody its pre-existing 66 additions, one deletion, and earlier 48 VH2 rows before semantically ingesting the new 121 routes.

## LIVE-HYPOTHESES

- A representation-level TF1/v19 join could escape V19c saturation by changing categorical support and coordinates rather than adding atoms from the closed vocabulary. G1 compilation and RE1’s cheap categorical carriage make this plausible, but no retained current-receiver proposal bank exists.
- The unharvested ms or j families may contain a current-floor signal absent from v; they remain plausible because neither family has yet been reconciled against current consumers.

## DEAD-ENDS

- Fresh V19/V19b/V19c replay: closed because G1 and the current DSL already consume the exact receipts and primitives.
- Treating V13/V14 as unread candidates: closed because their canonical-equation and G1 joins are explicit, while their historical Pose axis does not transfer.
- Reopening V4d’s dim-0 offset: closed because CP1/PJ2 already implement it and its archived row is far above the live floor.
- Direct canonical-ledger append during this arm: closed because it would absorb unrelated dirty work; the dedicated ingestion ledger preserves the complete drain safely.