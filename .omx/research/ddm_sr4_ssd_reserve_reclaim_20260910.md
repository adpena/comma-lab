# ddm_sr4 — SSD reserves restored with retained payload custody

Status: STORAGE COMPLETE, final custody audit VERIFIED. Git landing is separately recorded in
`ddm_sr4_20260910/SERIALIZER_RESULT.json` after this packet is frozen. No main-commit claim is
made by this pre-serializer memo. Axis: [macOS filesystem custody, scorer-free].

## MEASURED

| Volume | Before free GiB | After dedup GiB | Final free GiB | Target GiB |
|---|---:|---:|---:|---:|
| /Volumes/VertigoDataTier | 35.822582 | 116.549309 | 85.187397 | 80 |
| /Volumes/APDataStore | 32.024902 | 32.024536 | 62.720215 | 60 |

Free-space snapshots: `DF_BEFORE.txt`, `DF_AFTER_DEDUP.txt`, `DF_AFTER.txt` in the evidence directory.

| Method | Exact file accounting | Measured phase free-space change |
|---|---:|---:|
| 25 certified hardlink replacements on Vertigo | 91,560,240,000 logical duplicate B | +86,679,662,592 B on Vertigo |
| 9 certified AP-to-Vertigo moves | 32,961,686,400 retained payload B | +32,959,234,048 B on AP; -33,674,596,352 B on Vertigo |
| Tool-owned snapshot pruning | 0 expired / 32 inspected | 0 B |
| Payload deletion | 0 files | 0 B |

Phase `df` deltas include metadata and concurrent unrelated activity; they are measured net changes,
not a claim of exclusive physical-byte attribution. Per-action intervals overlap and are not summed.
APFS already shared some duplicate extents: BX1 control released effectively no additional blocks.
One of two separate 3,662,409,600-byte raws can release at most 3.411 GiB, not the charter's two-copy total.

## CUSTODY AND VERIFICATION

`FINAL_CUSTODY_AUDIT.json` passes all 25 hardlink sources against their full-hash-bound destination
inode, size and mtime; all 9 MOVED sources are absent with intact manifests and destinations whose
identities still match their completed full-hash receipts. The canonical validator passes 25 file
certificates: 16 pre-existing records plus 9 new records. Initial independent recursive and filename
censuses found the same 16 file records; 2 legacy directory records were excluded by canonical rules.
Existing-record audit is metadata/header validation, not a fresh full-payload hash of every old redirect.
Source and destination were fully hashed at every actual mutation. Final verification reused unchanged
hash-bound identities. The audit initially needed a parser correction (`path` versus `source` in move
receipts); no storage mutation failed. The corrected audit completed with rc 0.

All original hardlinked raw names remain usable. Hardlinked files share future writes; regeneration
must use fresh output paths. The twelve DWC1/RLC1 and two SJ1-pass6/RLC5 raw paths already shared inodes
before this arm and receive no new reclaim credit. The move-42 RP1/RLC pair, G85 public pair, three
additional public raws and twenty cold duplicates were certified and hardlinked here.

Nine older advisory AP raws were safely externalized using `tac.artifact_moved.move_with_manifest`.
Their destinations live under `/Volumes/VertigoDataTier/pact/ddm_sr4_20260910/retained/ap_offload/`.
Full source and destination hashes preceded source retirement; every original path now has a
`.MOVED.json` resolver certificate. Consumers must use the canonical MOVED resolver or the recorded
copy-back command. No archive or runtime was moved. GB1 identity raws were excluded because their
memo binds a sealed candidate. BO2/RD2 retain exact original data forks and copy-back commands;
missing historical archive paths mean no renderer regeneration claim is made for them.

`dedup.sh` and `move_ap.sh` completed with rc 0. Their stage certificates and append-only ledger
support disk resume. Interrupted-copy policy retains partial bytes and retries at a fresh destination;
no interrupted copy occurred. Each move checked that its destination would retain at least 80 GiB free.

## CENSUS AND CERTIFIED DELETION LIST

`DU_TOPLEVEL.txt`: 1,460 visible top-level Pact entries across both SSDs, no du errors.
`RAW_CENSUS.json`: 237 regular exact-size 0.raw paths, no walk errors, with explicit protected-path
exclusions. `PUBLIC_RAW_CENSUS.json` and `PUBLIC_RAW_HASHES.jsonl`: 21/21 named-public and charter-lineage
paths fully hashed in this run, four SHA groups. This bounded public classification is not a claim
that every path in the 237-entry census is public or freshly hashed. Historical full-hash/stat bindings
cover 130/237; sampled duplicate leads are explicitly not equality proofs. Current full hashes were
required before all 25 hardlink replacements and all 9 moves.

The six named large bulk roots remain intact at whole-tree scope: no complete tree rebuild certificate
was found in the inspected evidence. Twenty nearby raw decode chains yielded three fully verified
archive/runtime certificates and seventeen refusals. HV1 and JG4 raws were retained by offload; WD2
remains in place. These optional deletion routes are FOLDED because the requested reserves are met.
`CERTIFIED_DELETION_LIST.json` is empty. No operator deletion is requested.

All 32 local source snapshots were newer than three days. The actual tool-owned `prune_snapshots`
call removed none; the protected RLC5-run3 snapshot remains present. Local-disk reclaim was not pursued.

## RECALL EVIDENCE


Queries: `hardlink|certify.and.move|SSD reserve`, `reclaim`, `prune_snapshots`, and exact names
`ddm_pk4_20260813|ddm_pfs1_20260729|ddm_qs1_20260813|pr135_joint_solve_20260810`.
Stores: research memos/JSONL receipts, docs/runbook/SPEC surfaces, canonical task ledger,
main_hot_state, all twelve research-index/DAG paths, and the canonical-equations CLI (rc 0).
Persisted search results: `RECALL_0.txt`, `RECALL_1.txt`, `INDEX_DAG_RECALL.txt`,
`EQUATIONS_STORAGE_RECALL.txt`, and the per-AP-source recall files. Memory registry search
reinforced the Git object-write limitation; no old memory-derived free-space number was used.

Beyond the charter seeds: SR3 confirms ExFAT routing and the need to separate complete retained
archives from uncustodied bulk; VR3/VR4/VR5 expose historical raw hashes and reproduce-chain
refusals; the two formerly surplus JF2 raws are now absent; actual RLC5/G85 identities change
the dedup selection. Twenty nearby decode-manifest chains were screened: three pass current
archive/runtime verification, while seventeen refuse for changed runtime path sets, missing
receipts, or missing archives. The GB1 memo excludes two initially considered AP raws. The
BO2/RD2 terminal advisory memos support safely externalizing those original data forks, not
inventing a regenerated archive claim. No storage equation superseded the source/destination
hash contract in the searched registry; matched DAG hits concerned training-time memory, not
filesystem reclaim.


## BOUNDARIES AND HANDOFF

No scorer, GPU, Modal dispatch, training, MPS, GT decode, exact evaluation, archive mutation,
publication, upstream edit, PR edit, or pointer write. Live GDC2 and RLC5-run3 stores were excluded
from mutation. Sealed-tree changes were limited to certified byte-identical raw hardlinks. No bulk
or payload deletion. Shared index and unrelated dirty work were preserved; index identity is checked
again around the required serializer. No direct `.omx/state` edit; ddm_sr4's checkpoint lives in this
owned evidence directory. Only the explicitly required serializer may perform its own bookkeeping.
The six scientific integration hooks do not apply to storage custody; consumers are preserved raw
paths, MOVED receipts, #419 and MAIN's commit harvest.

Serializer is LAST, with per-file post-edit hashes, explicit files, no co-author trailer and
`[no-triality] [p0-ledger-ok]`. Non-Python review override is allowed by the charter. Transport logs and
`SERIALIZER_RESULT.json` are emitted after the frozen source manifest and excluded from their own
commit snapshot. Any rc-17 fallback must be verified and handed to MAIN as unlanded custody.

LIVE-HYPOTHESES: Additional cold duplicates may exist because historical and sampled groups remain
unconsumed. They are untested physical-reclaim leads, not a queued cleanup task or credited savings.
DEAD-ENDS: Same-volume moves do not free the tier; already-linked raws offer no new allocation gain;
no snapshot met the expiry rule; partial samples and MOVE labels cannot establish byte custody.

Charter frontier reference unchanged by this work: composition S 0.1372848557085275 @ 180,466 B
[contest-CUDA T4 n600] (move 43). No score was measured by this arm.

<!-- # FORMALIZATION_PENDING: storage hygiene charter; no measured scientific row -->
