Reclaimed 218.667 GiB from Vertigo and exceeded the safety threshold.

- Vertigo free space: 23.996 → 242.663 GiB.
- Moved 234,889,540,091 logical bytes from `nerv_long_training_campaigns`.
- Verified 4,404/4,404 data files; complete manifests and largest-20 subset match.
- Manifest SHA-256: `81d865cd3d2ddf0ba81a5a9b76423ceac6e65fab8c5d67404f1557dd28e4443a`.
- Original path now resolves through a symlink to APDataStore.
- Classified all 38/38 directories meeting the ≥5 GiB denominator.
- Protected recent/live trees and the AP custody mirror were untouched.

The duplicate Vertigo copy was removed after verification; those bytes remain preserved at the AP destination but are no longer locally recoverable on Vertigo.

Artifacts:

- [Durable memo](/Users/adpena/Projects/pact/.omx/research/ddm_sr2_vertigo_space_reclaim_20260811.md)
- [MOVED.json](/Volumes/APDataStore/pact/vertigo_coldstore_20260811/_manifests/nerv_long_training_campaigns/MOVED.json)
- Memo SHA-256: `3b7b9e4977691efc5ab0ab24418b5a4a843ddfdc9b143239b7fb6b14a6bc6305`

The memo is uncommitted. The required serializer failed before staging because managed Git object/index writes returned `Operation not permitted`; the shared index remains empty.

No scorer, Modal job, training, archive mutation, or exact evaluation ran. Pointer unmoved. Own-vehicle frontier remains `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: Git history; fire trigger: Git object/index writes become permitted. Serializer-commit only the memo using SHA-256 `3b7b9e4977691efc5ab0ab24418b5a4a843ddfdc9b143239b7fb6b14a6bc6305` and message `storage: record verified Vertigo cold move [no-triality] [p0-ledger-ok]`.
- **QUEUED-WITH-FIRE-ORDER** — owner: MAIN storage custody; consumer store: `/Volumes/APDataStore/pact/vertigo_coldstore_20260811/_manifests/`; fire trigger: Vertigo free space falls below 150 GiB. Partition `pact/evidence` and `pact/experiments` by newest descendant and current consumer before certifying another move.

## LIVE-HYPOTHESES

- `pact/evidence` and `pact/experiments` may contain substantial movable cold subtrees: together they allocate about 603 GiB and have old root mtimes, but their current consumers remain unresolved.
- Narrowly named historical HPRC and compact-renderer trees may certify quickly if more headroom is needed because they predate the 14-day cutoff and their cited paths can be preserved with symlinks.

## DEAD-ENDS

- A mirrored subtree is not proof that its complete parent directory is already mirrored; the scopes differ.
- Root mtime alone cannot establish whole-tree coldness or consumer quiescence.
- Raw ExFAT file counts are invalid equality evidence because 5,161 AppleDouble sidecars were generated.
- Successful `rsync` completion alone is not deletion authority; complete hashes and largest-file checks are required.
- Parallel APDataStore hashing is closed for this corpus: two streams were slower than one, while eight streams caused severe random-read thrashing.

