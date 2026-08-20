Completed the custody move and reached the storage target.

- Moved two inactive families totaling 132.880 GiB allocated.
- Both manifests are `complete_verified`; all 76 files passed full path/size/SHA-256 equality.
- Vertigo free space rose from 127.725 GiB to 234.004 GiB: **106.279 GiB recovered**.
- Original paths now resolve through verified APDataStore symlinks.
- Memo: [ddm_sc3_storage_custody_move_20260813.md](/Users/adpena/Projects/pact/.omx/research/ddm_sc3_storage_custody_move_20260813.md)
- Receipts: [experiments manifest](/Volumes/APDataStore/pact/cold_store_experiments_20260813/SC3_MOVE_MANIFEST.json), [DQS1 manifest](/Volumes/APDataStore/pact/cold_store_external_dqs1_20260813/SC3_MOVE_MANIFEST.json)

The memo remains uncommitted because Git object writes failed with `Operation not permitted`. Both serializer attempts left the staged index empty. Final memo SHA-256: `48f115e8a2ff2a3e87d53c5aeacf0784cc420b790b853aba5867757238be8e6a`.

This was scorer-free: no Modal call, training, archive mutation, or exact evaluation. Pointer unmoved; own-vehicle frontier remains LC2 at `S=0.16959899569230852 @ 187,226 B` `[contest-CUDA T4, n600]`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: Git history; fire trigger: Git index/object writes become permitted; action: recompute the memo SHA and serializer-commit only the memo with the recorded `[no-triality] [p0-ledger-ok]` message.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN storage custody; consumer store: `/Volumes/APDataStore/pact/<next-family>/SC3_MOVE_MANIFEST.json`; fire trigger: Vertigo free space falls below 150 GiB or payload retention preflight fails; action: certify and move one narrow inactive family with serialized heavy-read phases.

## LIVE-HYPOTHESES

- Old, narrowly named cold/HPRC families may provide additional safe recovery because several have May/June newest-file dates, but each still needs current-consumer and full-manifest proof.
- Broad `evidence` and `experiments` roots likely contain movable subfamilies because they dominate allocation, but only partition-level certification can separate them safely from live consumers.

## DEAD-ENDS

- Whole-tree moves of broad high-allocation roots are closed: age and consumer quiescence were not proven at that scope.
- Moving `regen2`, live solve/terminal directories, the RE1 probability store, PZ4R `direct_v6`, or active fire-order paths is closed by the charter.
- Root mtime, successful `rsync`, sampled hashes, and raw ExFAT counts are insufficient custody proof.
- Concurrent heavy Vertigo readers are closed: measured I/O contention made serialization the required execution form.
- Retrofitting checkpoints into an already-running verifier is closed: the first verifier completed uninterrupted without per-file destination checkpoints; the second used the corrected checkpoint path.