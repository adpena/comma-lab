Implemented and verified, **not landed on MAIN**. Serializer **rc 17**: Git object writes were denied; verified fallback commit **`a62939b58a7b099a86960a10057d35000a373b6a`** contains code, tests, and evidence.

| Requirement | Code | Verification |
|---|---|---|
| Fresh content/provenance join and retained-root projection | `validate_first_measurement_runtime_custody` | 15 drift cases; real run3 replay; run4 dry validation |
| Identical seal/leg custody; unchanged legacy timing validation | Completion and v3 revalidation | Synthetic completion, unequal-object and source-drift refusals |
| Typed consumer batches with ancestry, manifest and freshness checks | `_pf_freeze_history`, `_pf_contract` | Real Git controls; 18 negatives; stale-row refusal |

**231 tests passed; Ruff passed; two review passes completed.** The 17-file manifest and seven-fix draft batch are prepared. Full requirement mapping: [memo](/Users/adpena/Projects/pact/.omx/research/ddm_ffi6_completion_custody_and_consumer_fix_rows_20260910.md).

Run3 passes content validation, reproduces the original tree-hash refusal, and remains non-completable. Run4 passes dry custody and completion-facts validation across 48 files. **Appending the new freeze row makes run4’s existing intent stale**; that refusal remains enforced.

No real completion, seal, authorization, fire, timing sample, scorer run, volume write, or pointer change.

Frontier unchanged: **S 0.1372848557085275 @ 180,466 B [contest-CUDA T4 n600]**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — MAIN:** consume [LANDING_FIRE_ORDER.json](/Users/adpena/Projects/pact/.omx/research/ddm_ffi6_20260910/LANDING_FIRE_ORDER.json); trigger: bundle verification at harvest. Land the packet, bind the exact commit, and ratify/append the draft batch.
- **FOLDED — MAIN/pr17:** consumer store `ddm_ffi6_20260910/NEXT_FIRE_ORDERS.json`; trigger: batch ratification. Resolve run4’s full-completion disposition under latest-row freshness.

## LIVE-HYPOTHESES

- Run4’s retained evidence may support lifecycle closure after MAIN’s contract disposition: its custody and completion-facts checks pass.

## DEAD-ENDS

- Equating normal and retained-root tree hashes: disproved by run3’s identical content at different roots.
- Replaying or completing run3: refused by its terminal failure and nonce state.
- Treating run4’s dry pass as completion: freshness remains unresolved; no seal was written.