Reclaim completed and landed in commit `ff1095d49`.

- Freed **65,923,223,552 measured bytes (61.395786 GiB)** on Vertigo.
- Deleted 18 fully certified rebuildable raws; retained 141/159 census rows.
- All retained blobs remain identity-current; all 20 reproducer chains revalidated.
- No scorer, Modal, archive, runtime, submission, `upstream/`, or sj1 mutation.
- Final Vertigo availability at executor completion: **68.868034 GiB**.
- Verification: Ruff passed, 6 tests passed, two review passes completed, Catalog #344 strict passed.

Artifacts: [memo](/Users/adpena/Projects/pact/.omx/research/ddm_vr3_both_ssds_full_certify_or_block_reclaim_20260908.md), [159-row ledger](/Users/adpena/Projects/pact/.omx/research/ddm_vr3_reclaim_20260908.jsonl), [certifier](/Users/adpena/Projects/pact/experiments/ddm_vr3_certified_raw_reclaim.py), [tests](/Users/adpena/Projects/pact/experiments/tests/test_ddm_vr3_certified_raw_reclaim.py).

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — disposition: certify the 26 unreferenced but certificate-incomplete rows in a successor ledger; owner: next Vertigo storage custodian; consumer store: the VR3 ledger and `/Volumes/VertigoDataTier/pact/ddm_vr3_reclaim_20260908/`; fire trigger: Vertigo falls below 50 GiB available, followed by fresh identity, SHA, reference, reproducer, and `lsof` checks.

## LIVE-HYPOTHESES

- Some of the 26 unreferenced rows may form another certified batch: they have the same 3.66 GB decoded-raw shape and no repository path hit, but their archive/runtime/decode chains remain unproven.
- The two surplus certified JF2 raws may provide another 6.821769 GiB quickly: their chains remain valid, but every apply-time check must be repeated.

## DEAD-ENDS

- Pre-remount hashes cannot be reused across a device-identity change; all 21 affected raws required rehashing.
- AP could not absorb 60 GiB while preserving its verified 10 GiB floor; local cleanup would not free Vertigo.
- SR3 compression remains blocked for these custody namespaces without a per-tree protection lift.
- Deletion outside the exact allowlist remains closed for this run: 139 rows lacked complete certificates, including referenced, submission, and live-sj1 paths.
- Hand deletion/moves, path-size pseudo-hashes, citation rewrites, and archive/runtime/sj1 mutation remain forbidden.

sj1 S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]

