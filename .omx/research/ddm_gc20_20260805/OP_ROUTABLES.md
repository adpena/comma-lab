# DDM GC20 operator routables

id: ddm_gc20_20260805_op_routables
date: 2026-08-05
axis: [macOS-CPU advisory, scorer-free research]
promotion_claim: false

## Immediate disposition

| Rank | Status | Action | Fire condition | Why |
|---:|---|---|---|---|
| 1 | FIRED | Preserve the JD3 full-v3 ENTRY window. | Already fired outside GC20; do not touch the live run. | ENTRY had the best joint slope despite transient seg toll. |
| 2 | QUEUED-WITH-FIRE-ORDER | Consume v3 endpoint with JD3/GC19 endpoint predicates. | Full-v3 endpoint appears and receipts are complete. | Avoid mid-window mutation and avoid judging by single-axis movement. |
| 3 | QUEUED-WITH-FIRE-ORDER | Byte-close/parse-back alive endpoint before terminal/rate composition. | Endpoint passes realized hold and basic endpoint predicate. | GC19 says terminal/rate actions matter only after object custody. |
| 4 | QUEUED-WITH-FIRE-ORDER | OD3/SQ2 terminal pose/rate compose. | Byte-closed endpoint exists and same-row pose baseline is measured. | OD2/OD3 show seg-first can recover pose by repair capacity. |
| 5 | QUEUED-WITH-FIRE-ORDER | Realized-hold rollback/pose-pressure retreat. | Any v3 gate violates realized d_seg hold. | Removes ENTRY state-dependent antagonism without killing the family. |
| 6 | QUEUED-WITH-FIRE-ORDER | Lane-guard ratchet v3b/v4 smoke. | Endpoint shows Lane give-back with guard slack. | PC2 says Road/Lane is central; CX1 says current ratchet is not engaged. |
| 7 | QUEUED-WITH-FIRE-ORDER | EN1 margin-weight A/B. | Next clean boundary after v3, same seed/schedule, one variable. | Consumer is built; effect must be measured with pose collateral. |
| 8 | QUEUED-WITH-FIRE-ORDER | Receiver-close OD5 generator/worldsheet packet. | Native representation and parse-back are available. | Per-flip and offset streams are folded; generator packets remain open. |
| 9 | QUEUED-WITH-FIRE-ORDER | Conflict controller A/B. | Endpoint logs show actual seg/pose gradient conflict. | Controller complexity is justified only by measured conflict. |

## Folded routes

| Status | Route | Fold reason | Reopen condition |
|---|---|---|---|
| FOLDED | PE3-as-is shipped hybrid | Receiver-byte-positive but scorer-negative: `S = 1.852721897902562 @ 432,428 B`. | Conditioning-only or receiver-consumed rider with matched scorer survival. |
| FOLDED | SL2 explicit edit carriage | Carriage pricing dead at roughly three orders of magnitude. | Not reopened as explicit values; use as teacher/verifier only. |
| FOLDED | Distill-from-solve/KD shipping | Prior settlement says route is dead on current recipe. | Reopen only if weights-as-carrier training route changes the mechanism. |
| FOLDED | OD4 sparse per-flip stream | Sparse per-flip weak-packet formulation killed, not the generator family. | Worldsheet/task-description packet with receiver closure. |
| FOLDED | LR2 offset field | Every offset-shipping rung loses; pose-null is AC-only. | New native pose-null DOF, not DC shifts or constants. |
| FOLDED | Pure qo1 byte crumbs | 0.0001651 S is only 0.0284% of the current gap. | Only as cleanup after score-moving object is selected. |

## Operator sequence

1. Do nothing to the live v3 window.
2. At endpoint, compute the JD3/GC19 endpoint predicate: joint S, realized hold survival, live/EMA basis, d_seg/d_pose/rate, and stop reason.
3. If endpoint is alive, take custody of bytes first. Then route OD3/SQ2 terminal pose/rate compose.
4. If endpoint breaches hold, retreat pose pressure or rollback before any new lever is stacked.
5. Only after endpoint disposition, run clean one-variable A/Bs: lane-guard ratchet if slack exists, EN1 margin-weight if boundary is clean, conflict controller if gradients conflict.
6. Keep PE3-as-is, explicit SL2 carriage, OD4 sparse, LR2 offsets, and pure byte crumbs folded unless their listed reopen conditions are met.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
