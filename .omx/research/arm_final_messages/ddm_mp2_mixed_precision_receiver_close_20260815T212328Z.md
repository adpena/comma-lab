The exact pointer did not move. Three original candidates were measured on the same `[macOS-CPU env-mismatch advisory]` n600 axis; all failed the `ΔS < -3.5e-6` gate:

| Candidate | Bytes | Δd_seg | Δd_pose | Exact ΔS | Verdict |
|---|---:|---:|---:|---:|---|
| mixed q3/q4 | 181,936 | +1.14e-6 | +5.8376e-4 | +0.0466761627 | REJECT |
| FiLM keep87 | 182,629 | +6.40e-7 | +5.3643e-4 | +0.0442738621 | REJECT |
| FiLM keep75 | 182,288 | +1.07e-6 | +4.9212e-4 | +0.0413659285 | REJECT |

Key outcomes:

- keep62/50/37/25 were not launched after MAIN’s binding pose-field stop.
- The keep75-minus-keep87 differential was built and receiver-closed: 182,734 B, SHA `37194782ed5c…`, deterministic repeat identical, 38/38 tensors exact.
- That differential saves only 25 B, correcting the non-transferable 341 B conditional estimate. Its n600 row is queued behind WD3.
- Stage 2’s retained Brotli q0–q11 race found no improvement: q11 ties the incumbent at 22,161 B.
- No candidate was admitted, so no T4 fire order or Modal dispatch was created.
- Verification passed: 11 tests, Ruff, compilation, arithmetic/artifact assertions, and two review passes per Python file.

The complete handoff is [ddm_mp2_mixed_precision_receiver_close_20260815.md](/Users/adpena/Projects/pact/.omx/research/ddm_mp2_mixed_precision_receiver_close_20260815.md). Code landed in `34f0cc9a0a`; the memo landed in `66172672b8`.

Vehicle frontier unchanged: **HV1 ep0634 S = 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600]**, archive SHA `80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`.

## NEXT_IF_RESUMED

- Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: MAIN. Action: run the differential n600 advisory. Consumer store: `/Volumes/APDataStore/pact/ddm_mp2_mixed_precision_receiver_close_20260815/advisory_n600_cpu/score_gated_film_row_prune_keep75_minus_keep87/attempt_0000/contest_auth_eval.json`. Fire trigger: WD3 releases the scorer lane, no n600 scorer is active, hashes match, and all `._*` files are cleaned.
- Disposition: QUEUED-WITH-A-FIRE-ORDER. Owner: MAIN/MZ2 carrier successor. Action: build the pose-gated rank/refit and adaptive-quant race. Consumer store: `/Volumes/APDataStore/pact/ddm_mp2_mixed_precision_receiver_close_20260815/carrier_rank/`. Fire trigger: the differential is terminal and a strided per-coefficient pose-sensitivity field exists.
- Disposition: PAUSED-TO-FOLD. Owner: MAIN/operator. Action: terminate parked queue supervisor PID 77093 without restoring its receipt alias, then reconcile its state. Consumer store: `/Volumes/APDataStore/pact/ddm_mp2_mixed_precision_receiver_close_20260815/ADVISORY_QUEUE_STATE.json`. Fire trigger: process-control authority is available and keep75’s terminal JSON/receipt are reverified.

## LIVE-HYPOTHESES

- The differential may improve pose despite saving only 25 B; the same marginal removals reduced d_pose by `4.43e-5` conditionally.
- Pose-null or pose-positive FiLM rows likely exist because greater pruning improved pose from keep87 to keep75.
- Rank/refit or adaptive per-cell depth may reduce carrier bytes even though lossless Brotli recoding tied, provided selection includes real pose sensitivity.

## DEAD-ENDS

- Mixed q3/q4, keep87, and keep75 are rejected instances; their pose damage overwhelms rate savings.
- Blindly scoring keep62/50/37/25 is closed until the pose-sensitivity prerequisite exists.
- Exact Brotli-quality search on the current carrier is closed at instance scope: 12/12 rows produced no saving.
- Treating the conditional 341 B keep75-minus-keep87 difference as the standalone differential prize is closed; the real receiver-closed saving is 25 B.
- Re-running `pz4a` on pose coefficients is closed; it previously added 2,232 B and targets the wrong sensitivity surface.