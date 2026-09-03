The six FPC1 blockers are cleared for the required real n=2 pipeline. The run produced an independent driver/receiver byte match, but the live Git landing is blocked by sandbox write denial.

Key evidence:

- Fresh archive: 180,496 B, SHA `fa048304…2b7848`.
- Driver and receiver: identical 12,208,032 B raw payloads, SHA `c7452f74…6692e3c`.
- Advisory n=2: `d_seg=0.00030772`, `d_pose=0.00010887`, recomputed S `0.1839524542323636` `[macOS-CPU advisory]`, `score_claim=false`.
- Exact AFR1 replay remains 180,002 B, SHA `cbb8d928…d405bf25`.
- Tests: 8 passed; Ruff and diff checks clean; two review passes recorded.
- Full receipt: [RESULT.json](/Volumes/VertigoDataTier/pact/ddm_fpc2_full_pipeline_ports/full/RESULT.json)
- Handoff memo: [ddm_fpc2_full_pipeline_ports_20260903.md](/Users/adpena/Projects/pact/.omx/research/ddm_fpc2_full_pipeline_ports_20260903.md)

The serializer created a verified 15-file fallback commit because `.git` rejected writes:

- Isolated commit: `4608f607b8ce13eaa309d9ce34e17413962381be`
- Base: `2c32e2767b50cef31d27e62c61da1f862158a033`
- [Commit bundle](/Volumes/VertigoDataTier/pact/ddm_fpc2_full_pipeline_ports/receipts/commit_serializer_fallbacks/20260903T180101.226565Z-75291/intended-commit.bundle), SHA `f2b6560b…0f007b`
- Bundle verification and all 15 recorded content hashes passed. The shared index remained untouched.

One acceptance boundary remains: the historical QS5 receipt embeds the old script’s source hash, so its complete hash cannot survive the mandated CLI edits. The extracted kernel reproduced the exact selected values instead; returning the old receipt as new provenance would be false.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `/Volumes/VertigoDataTier/pact/ddm_fpc2_full_pipeline_ports/receipts/commit_serializer_fallbacks/20260903T180101.226565Z-75291`; fire trigger: on a Git-write-enabled host, verify base `2c32e276…`, import commit `4608f607…`, verify its 15-file tree, and rerun the eight tests.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `/Volumes/VertigoDataTier/pact/ddm_fpc2_full_pipeline_ports`; fire trigger: land the crash-resumable chunked n=600 trainer, rerun system-aware memory preflight, claim the scorer lane, then execute the retained CUDA launch ticket.

## LIVE-HYPOTHESES

- Chunking can preserve the bounded trainer’s semantics because EMA, target lineage, update order, roundtrip, archive construction, and receiver contracts are explicit.
- The n=2 loss descent shows that scorer-aware training is active, but only a stratified/population run can establish its n=600 sign or magnitude.

## DEAD-ENDS

- Treating the n=2 advisory result as a contest score is closed: it is a contiguous-prefix plumbing smoke.
- Reusing the retained token field with a changed archive token stream is closed; the receiver now refuses it.
- Calling two receiver invocations “driver/receiver closure” is closed; the final proof uses an independent quantized-state driver.
- Reproducing QS5’s whole historical receipt after changing its self-censused source is closed as false provenance.
- Firing the n=600 ticket before its chunked consumer and lane claim is closed.

**OWN-VEHICLE FRONTIER: S 0.14797617125559104 @ 180,002 B `[contest-CUDA T4, n600]` — AFR1 unchanged; this arm moved no exact pointer.**