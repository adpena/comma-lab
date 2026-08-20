Implemented the default-off advisory decode path and retained all evidence. Full findings are in the [handoff memo](/Users/adpena/Projects/pact/.omx/research/ddm_wc1_advisory_decode_wallclock_20260815.md).

| Population | Wall time | Identity result |
|---|---:|---|
| n4 prefix | 8.4980 s | Token and 24,416,064 raw bytes matched |
| n32 prefix | 32.6311 s | Token and 195,328,512 raw bytes matched |
| n600 | Not measured | Governor paused before decode; terminal timeout `rc=124` |

The n600 tree remained paused from four seconds after launch until the canonical 1,200.706-second timeout. It materialized no token, render, or raw payload. The [blocker receipt](/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815/receipts/FULL_N600_BLOCKED.json) binds the first and last pause events, terminal [done receipt](/Users/adpena/Projects/pact/.omx/tmp/codex_runs/wc1_base_optimized_n600.done), pressure facts, and fire order.

No admission gate was issued. MP2 remains on its existing slow path; LH2 was confirmed not to consume this decoder. Shipping, upstream, the evaluation mirror, and live MP2 directories were untouched.

Validation passed: 11 tests, Ruff, bytecode compilation, zero payload-retention findings across six Python surfaces, and two review-tracker passes.

The seven scoped artifacts are uncommitted. The required serializer failed while staging with `unable to create temporary file: Operation not permitted`. HEAD remains `bef8e76dbc1eefe6c998f717499fc5a61443bb51`, and the index is empty. I did not bypass the serializer.

Own-vehicle frontier remains `S=0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600]`.

## NEXT_IF_RESUMED

- Disposition: **QUEUED-WITH-A-FIRE-ORDER**. Owner: **MAIN**. Consumer store: `/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815/receipts/`. Fire trigger: three consecutive memory-blackbox rows show `pressure=normal` and `available_gib>=64`, with the prior done receipt terminal. Action: launch a fresh watched optimized n600 row, then the scalar twin and fresh/cached micro-edit rows serially; finalize only after all identity gates pass.
- Disposition: **READY-TO-LAND**. Owner: **MAIN**. Consumer store: Pact Git history. Fire trigger: the managed environment permits Git index/object writes. Action: recompute the seven post-edit SHA-256 values and rerun the serializer with the same explicit file list.

## LIVE-HYPOTHESES

- Full n600 may clear the 700-second bar: linear scaling from the real n32 result projects roughly 612 seconds, although full-field copying and scheduling could scale differently.
- The selected MP2 micro-edit should reuse the base token cache because it inherits the token stream, HPAC model, and correction table while changing only the semantic section.
- Four workers should remain admissible after pressure normalizes because measured maximum worker RSS was about 1.50 GB and the governor adds a 25% margin.

## DEAD-ENDS

- Prefix timing cannot serve as a full-field verdict; n600 raw identity and decoder digests remain mandatory.
- Another n600 launch under current WARN pressure is closed; the 64 GiB safety floor remains unmet.
- Manually overriding the memory governor is closed; no override authority was provided.
- Wiring WC1 into LH2 is closed because LH2 never calls this advisory decoder.
- Concurrent producers sharing one WC1 run ID are closed; the runner now enforces a nonblocking single-flight lock.