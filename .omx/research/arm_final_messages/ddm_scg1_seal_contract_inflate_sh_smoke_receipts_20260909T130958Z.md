Implementation is complete, but the required Git landing is blocked by the sandbox’s read-only object store. The serializer produced a validated fallback commit instead.

Key results:

- Candidate seals now require structured candidate/frontier receipts for both direct token-decode reach and public `inflate.sh` reach to the CUDA gate.
- Fire refuses missing or invalid receipts. The waiver is restricted to already-scored custody replay.
- Both historical sj1 seals return `SEAL_PUBLIC_SMOKE_MISSING`; the prior-law prediction had 2 falsifiers because `public_entrypoint_smoke` is absent.
- `review_tracker mark-file` now rescans the current AST.
- Anchor assertion sweep: 39/39 converted; zero literal count assertions remain.
- Tests: 822/822 runnable changed tests passed; 137/137 focused tests, including 50/50 seal tests. One MLX test could not collect without Metal and was AST-verified.
- No scorer, Modal dispatch, payload render, exact evaluation, or pointer movement occurred.

Artifacts:

- [Evidence memo](/Users/adpena/Projects/pact/.omx/research/ddm_scg1_seal_contract_inflate_sh_smoke_receipts_20260908.md)
- [Fallback receipt](/Volumes/APDataStore/pact/ddm_scg1/receipts/commit_serializer_fallbacks/20260909T130455.443501Z-99636/receipts.jsonl)
- [Fallback bundle](/Volumes/APDataStore/pact/ddm_scg1/receipts/commit_serializer_fallbacks/20260909T130455.443501Z-99636/intended-commit.bundle)
- Fallback commit: `3bddfcd09b9598c6ab256922f9191d2fca7e7279`
- Bundle verification: clean isolated checkout, exactly 39 declared files, 39/39 content hashes matched, zero staged shared-worktree paths.
- Registered follow-on: `ddm_scg1_seal_contract_inflate_sh_smoke_receipts_20260908::ITEM_7`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: the fallback receipt above; fire trigger: when Git object-write authority is available, fetch the bundle and cherry-pick `3bddfcd09b9598c6ab256922f9191d2fca7e7279`. The two intervening shared commits have zero path overlap.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN / next sj1 custody-seal producer; consumer store: `.omx/state/canonical_task_status.jsonl`, task `…::ITEM_7`; fire trigger: before any future sj1 joint/pass3 replay or fire, capture both receiver legs against candidate and current-frontier trees and issue a successor structured seal.

## LIVE-HYPOTHESES

- Fresh structured receipts should validate the unchanged sj1 archive bytes because the historical prose says both candidate legs succeeded; this remains untested because frontier-control receipts and complete identities were never recorded.
- The custody waiver should remain exceptional because reproducible trees can normally produce fresh paired receipts before fire.

## DEAD-ENDS

- Library-path decode identity is not public-entrypoint proof; rc1 demonstrated the failure.
- The sj1 falsifier prose cannot substitute for structured candidate/frontier receipts.
- A public “first two pairs” smoke is not expressible because `file_list` selects videos.
- Stale review-tracker censuses can no longer be marked reviewed.
- Literal anchor-count assertions are eliminated from the bounded test scope.
- Retrying direct Git writes in this managed sandbox is closed by the object-store permission denial; the retained serializer bundle is the recovery path.

sj1 S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]