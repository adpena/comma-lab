# OD1 typed blockers - 2026-08-05

Status: `BLOCKERS_TYPED / SCORER-FREE`.

Axis: `[macOS-CPU advisory / blocker ledger]`.
`score_claim=false`, `promotion_eligible=false`, `n600_scorer_job=false`.

## Blockers

| id | type | verdict scope | evidence | owner | unblock condition |
|---|---|---|---|---|---|
| `OD1_BLOCKER_PE2_SURVIVAL_ROWS_PENDING` | queued scorer evidence | PE1 full, PE1 surgical, BF1 scorer survival only | PE2 proved receiver consumption but did not run scorer; MAIN hot state says batch was fired/staged | MAIN | terminal pe2 n600 receipts with recomputed S, axis, bytes, d_seg, d_pose |
| `OD1_BLOCKER_SEG_BASE_CAP_BOUND` | optimization terminality | SQ2 solved-field base | SQ2 n32 eta rose to `0.9112579957356077`, but 0/32 converged and 21/32 were cap-best at 100 | OD1 next build / MAIN fire | event-continuation run with terminal census and no cap-default ambiguity |
| `OD1_BLOCKER_TRANSIENT_POSE_EROSION` | composition, not route kill | raw seg-only solved paint | SQ2 raw solved paint pose term `0.8813937215422536`, +`0.7968937215422536` versus the R8 pose bank | OD1 pose stage | final composition recovers pose before R8/final gate; raw transient is not judged as final |
| `OD1_BLOCKER_FRAME0_CARRIAGE_POPULATION` | denominator | JS1 frame0 C-PRIME/k=4 | JS1 evidence is sampled/subset and not bankable as n600 | OD1 next build | n>=32 stratified/random survival with retained seg and measured pose recovery |
| `OD1_BLOCKER_Q3_FIRST_SCOPED_NEGATIVE` | formulation negative | Q3 as first base only | Q31 n32 survival `0.2303538325`, only 33.1% of bar; 32/32 cap-best | none for base; OD1 for repair use | do not use Q3-first as base; use Q3 only as repair/correction subspace with null proof |
| `OD1_BLOCKER_SCHEDULE_RESEAL` | DSL/compiler readiness | fixed-stage schedule | Sched1 verdict `THREE_CLEAN_PASSES_RESEAL_REQUIRED`, ready_to_fire false | MAIN/DSL owner | event-continuation compiler resealed or explicit known-good ticket builder |
| `OD1_BLOCKER_HINGE_TREATMENT` | objective weight | DirectDescription correction solver path | BO1 found hinge weight `0.05` too quiet relative to boundary CE pressure | OD1 if that path is used | run/adopt #888 hinge A/B or explicit measured hinge setting |
| `OD1_BLOCKER_POSE_MECHANISM_LABEL_CONFLICT` | scope conflict | older #366 terminal-pose language versus Addendum 4 | P0 ledger older language said terminal pose never trained; Addendum 4 now binds seg-first then joint-descent pose recovery after | OD1 spec / MAIN | keep mechanism scoped: old language superseded for OD1; final tickets use 2026-08-05 ordering law |
| `OD1_BLOCKER_AUTHORITY_PROMOTION` | authority | any advisory or subset row | current package is scorer-free and macOS advisory only | MAIN | exact full n600 on contest-CPU or contest-CUDA with archive custody |

## Non-Blockers

- PE2 receiver consumption itself is no longer a blocker for PE1 full, PE1 surgical, and BF1. It is landed in `PE2_RECEIPT_20260805.md`; survival remains pending.
- SQ2 pose erosion is not a route kill. It is a transient-stage blocker that must be cleared by Stage 2 before R8.
- Q31 does not kill Q3 as a repair subspace. It kills Q3-first as the production base under the measured n32 formulation.

Own-vehicle frontier line: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved.
