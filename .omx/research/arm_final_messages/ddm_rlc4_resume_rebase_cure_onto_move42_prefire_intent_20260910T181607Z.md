Executed through the charter’s required STOP: **the real candidate passes full n600 byte identity, but the frozen prefire producer refuses its receiver digest.**

- **Archive:** 180,178 B, **60 B smaller** than move42. SHA-256: `eaf17a7038a4671bf39c3071192b15eec7d7513d471d520b6d13bc9acf94a60b`
- **Passed:** independent twin encodes, complete cold public raw comparison, 49-row manifest verification, 51-file census, and four smoke checks.
- **Blocker:** refreshing `MANIFEST.sha256` changes the normalized receiver digest. The unchanged producer returned rc3: `PREFIRE_RISK_EVIDENCE_REFUSED: receiver risk endpoints differ`.
- **Not produced:** valid intent, seal, authorization, dispatch, or candidate exact score. Normal-seal controls could not run against a valid intent.
- **Custody:** all payloads retained. Git writes were sandbox-blocked; the fallback bundle independently verifies all 66 selected files. MAIN landing remains owed. Two serializer invocations occurred: one preflight refusal, then one actual Git write attempt.

The **1,032.725 s timing projection is diagnostic only**. Conditional rate credit is −0.000039951537 S, not a measured contest result.

[Full findings](/Users/adpena/Projects/pact/.omx/research/ddm_rlc4_resume_rebase_cure_onto_move42_prefire_intent_20260910.md) · [Verified bundle and handoff](/Users/adpena/Projects/pact/.omx/research/ddm_rlc4_20260910/FINAL_HANDOFF.json)

Frontier unchanged: **S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600], move42**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN. Consumer: `ddm_rlc4_20260910/FINAL_HANDOFF.json`. Trigger: Git-writable MAIN harvests the verified bundle. Land its exact contents.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN/pr12-family. Consumer: `ddm_rlc4_20260910/RECEIVER_MANIFEST_CONFLICT.json`. Trigger: harvest of the actual refusal. Resolve the contract conflict prospectively and decide evidence reuse without retimestamping.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner: newly chartered producer. Consumer: `/Volumes/VertigoDataTier/pact/ddm_rlc2_cure_on_move42`. Trigger: resolved contract and explicit resumed charter while move42 remains current. Produce the valid intent and required refusal controls.

## LIVE-HYPOTHESES

- The candidate may improve the exact T4 score: retained public raw identity supports the measured 60-byte saving.
- T4 runtime may fit the ceiling: historical diagnostics support the projection, but candidate timing remains untested.

## DEAD-ENDS

- Restoring the old manifest would recreate stale hashes; it cannot honestly resolve this instance.
- Treating historical timing as candidate authority fails receiver correspondence.
- Repeating encoding from scratch is unnecessary: complete streams, archives, checkpoints, and raw proof are retained.