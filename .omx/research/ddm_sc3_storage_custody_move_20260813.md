# ddm_sc3 storage custody move

**Disposition:** `MOVE_COMPLETE_VERIFIED_TARGET_REACHED__MEMO_UNCOMMITTED_BLOCKED_GIT__APPARATUS_GAP_RECORDED`  
**Axis:** `[byte/custody apparatus, scorer-free]`  
**Frontier effect:** none; no scorer, training, archive mutation, Modal job, or exact evaluation ran.

## Result

Two inactive cold-store families moved to APDataStore with complete source and destination
path/byte/SHA-256 equality. Both external manifests report `complete_verified`; both original paths
resolve through compatibility symlinks; neither retirement directory remains.

| Measure | Before | After | Delta |
|---|---:|---:|---:|
| Vertigo available (KiB) | 133,929,416 | 245,371,456 | **+111,442,040** |
| Vertigo available (GiB) | 127.725 | 234.004 | **+106.279** |
| Vertigo used (KiB) | 1,819,087,512 | 1,707,645,468 | -111,442,044 |
| APDataStore available (KiB) | 710,605,824 | 571,221,248 | -139,384,576 |
| APDataStore available (GiB) | 677.687 | 544.759 | -132.927 |

The measured volume-level Vertigo recovery is **106.279 GiB**, so the charter's >=100 GiB target is
reached. The selected source trees allocated 132.880 GiB; the smaller volume-level recovery is
reported rather than conflated with that allocation. The second manifest's `df_before` was captured
while the first move was still in flight, so per-family `measured_vertigo_recovered` fields overlap
and are not additive. The result table uses the first manifest's original `df_before` and the second
manifest's terminal `df_after`.

## Selected families

| Original path | APDataStore custody path | Data files | Logical bytes | Allocated KiB | Source tree-manifest SHA-256 | State |
|---|---|---:|---:|---:|---|---|
| `/Volumes/VertigoDataTier/pact/cold_store/experiments` | `/Volumes/APDataStore/pact/cold_store_experiments_20260813/data` | 56 | 84,079,796,268 | 82,109,296 | `e74b4bacb032bb43d5822f0a890c54ad8632b605b52a3e00fde92b143c2a88bf` | `complete_verified`; symlink resolves |
| `/Volumes/VertigoDataTier/pact/cold_store/__external__` | `/Volumes/APDataStore/pact/cold_store_external_dqs1_20260813/data` | 20 | 58,599,267,432 | 57,225,856 | `d7d5b6539fdfe1a7815d00e0f3d7ae478de4d9df116f13dde1806e6638137f57` | `complete_verified`; symlink resolves |

The combined selected allocation is **139,335,152 KiB (132.880 GiB)** and combined logical payload
is **142,679,063,700 bytes (132.880 GiB)**. Those are fixed source-tree denominators, separate from
the measured 106.279 GiB volume-level recovery.

## Selection and consumer boundary

The selected families are already under Vertigo's `cold_store` namespace. Their newest data files
are from 2026-05-26 and 2026-05-24 respectively. The first family comes from the legacy MLX
decoder-Q parent-contract/cache-retention campaigns; its original retention journal marks the
relevant caches `certified_rebuildable=true` and records source/destination data digests. The second
contains four legacy DQS1 first-materialized locality/advisory families whose May lane rows are
terminal historical records.

A bounded current-consumer census found zero exact source/destination path matches in these six
current surfaces: `.omx/state/main_hot_state.md`, `codex_arm_queue.next_if_resumed.jsonl`,
`codex_arm_queue.jsonl`, `canonical_task_status.jsonl`, `durable_daemons.json`, and
`active_lane_dispatch_claims.md`. The arm queue reported 0/4 live at the decision boundary. Process
enumeration through `ps` was unavailable in the managed sandbox; no global process-absence claim is
made. Compatibility is therefore mandatory: each original source path remains present as a symlink
to the verified custody copy.

The following were hard-excluded and untouched: `regen2`; every currently live solve or terminal
directory; the RE1 probability store; PZ4R `direct_v6`; and any path named by the active fire order.
Broader `evidence`, `experiments`, PFS1, CB1, HPRC, public-dataset, and recent DDM roots were not
promoted to inactive whole families without equivalent path-specific consumer and hash proof.

## Certify-or-block protocol

For each family, the apparatus writes a complete pre-move manifest beside the eventual APDataStore
payload. It records relative path, logical bytes, mtime, SHA-256, tree-manifest SHA-256, producing
arm/provenance, rebuildability rationale, destination, symlink plan, command, tool hash, and before
`df`. Source hash stages checkpoint each completed file atomically. Destination checkpointing was
added before the second-family execution, but the already-launched first-family verifier cannot use
that retrofit; its source remains retained through the uninterrupted destination stage, and only the
complete destination snapshot may authorize a swap. The payload is copied with `rsync`, the
APDataStore data tree is fully rehashed, and path/bytes/SHA-256 rows and the tree-manifest digest must
match exactly. A post-copy dry run must also succeed.

Only after that equality certificate does the original directory move to a bounded retirement name,
the original path become a symlink, and a known largest file resolve through that symlink. Only then
is the verified duplicate on Vertigo retired. Any mismatch or ambiguous swap state fails closed with
the Vertigo source retained. ExFAT `._*` AppleDouble metadata is excluded from the data-fork census,
following the prior verified SR2 custody precedent; payload data files are never excluded.

Per-family machine-readable receipts:

- `/Volumes/APDataStore/pact/cold_store_experiments_20260813/SC3_MOVE_MANIFEST.json` — receipt SHA-256
  `5a4e3a5cea677a8b77fdeef2187520a415991560460ca4cef3f5fdf77f57a4c6`
- `/Volumes/APDataStore/pact/cold_store_external_dqs1_20260813/SC3_MOVE_MANIFEST.json` — receipt SHA-256
  `0835b01fb1f3af41b1b00dfb5a3bc4a9f577a64af7ff4e5e8f0078c894530732`

## RECALL EVIDENCE

**Stores searched:** `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, the common contract,
`docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, `.omx/research/**`,
`.omx/state/**`, the canonical research index and sub-0.15 DAG, the canonical-equations registry,
task/lane/fire-order ledgers, the prior SR2 memo/manifest, and the MLX cache-retention journal.
Queries included `storage`, `custody`, `VertigoDataTier`, `APDataStore`, `certify-or-block`,
`MOVED.json`, `cold_store`, `regen2`, `RE1`, `probability_store`, `PZ4R`, `direct_v6`, `DQS1`, and
the exact selected paths.

**Beyond the charter seeds:** SR2 established that ExFAT-generated AppleDouble files make raw
destination counts invalid; this changed SC3 to data-fork path/byte/hash equality before either
certificate began. The retention journal supplied producing-arm and rebuildability evidence for the
MLX cache family. The current fire-order/daemon/path census changed the selection away from broad
high-allocation roots and onto two already-cold May families. The canonical-equations/DAG searches
found no storage-specific equation or newer routing that displaced the charter.

## Boundaries and score status

- This is a scorer-free byte/custody result. It measured storage allocation and cryptographic payload
  identity only; it did not measure score, decoder behavior, or contest-runtime behavior.
- No Modal call was made and no payload was generated or discarded.
- The source-tree allocation, logical data bytes, and volume-level `df` delta are separate
  denominators and will not be conflated.
- The bounded consumer search did not find a current match in the named surfaces; it is not a global
  nonexistence claim.
- Pointer unmoved. Current own-vehicle frontier remains LC2: `S=0.16959899569230852 @ 187,226 B`
  `[contest-CUDA T4, n600]`.

## Serializer landing

`BLOCKED-GIT`: the required serializer targeted only this memo and failed before changing the empty
index:

```text
error: unable to create temporary file: Operation not permitted
error: .omx/research/ddm_sc3_storage_custody_move_20260813.md: failed to insert into database
fatal: adding files failed
```

No partner work was staged. Recompute the post-edit SHA-256 before retrying the same sole-file
serializer commit with message
`storage: record verified SC3 custody moves [no-triality] [p0-ledger-ok]`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: Git history; fire trigger: the managed
  workspace permits Git index/object writes; action: recompute this memo's post-edit SHA-256 and
  serializer-commit only this memo with the recorded `[no-triality] [p0-ledger-ok]` message.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN storage custody; consumer store:
  `/Volumes/APDataStore/pact/<next-family>/SC3_MOVE_MANIFEST.json`; fire trigger: Vertigo available
  space falls below 150 GiB or a governed payload preflight cannot preserve its output; action:
  partition an old narrow family, prove current-consumer absence in the live surfaces, then execute a
  manifest-first hash-verified move with heavy read phases serialized.

## LIVE-HYPOTHESES

- Other old, narrowly named cold/HPRC families may be safe move candidates if Vertigo pressure
  returns, because several have May/June newest-file dates; each still needs its own current-consumer
  census and complete pre-move manifest.
- Broad `evidence` and `experiments` roots likely contain movable subfamilies because they dominate
  allocated space, but only partition-level certification can separate inactive evidence from live
  consumers safely.

## DEAD-ENDS

- Whole-tree moves of broad `evidence`, `experiments`, or other high-allocation roots are closed for
  this charter: age and consumer quiescence were not proven at that scope.
- Any move of `regen2`, a live solve/terminal directory, the RE1 probability store, PZ4R `direct_v6`,
  or an active fire-order path is closed by the charter's hard exclusions.
- Root mtime, a successful `rsync`, sampled hashes, or raw ExFAT file counts are not custody proof;
  only complete data-fork path/byte/SHA-256 equality authorizes source retirement.
- Concurrent full-tree readers on Vertigo are closed as an execution pattern: the attempted overlap
  caused severe I/O contention, so successor moves must serialize the heavy read phases.
- Retrofitting checkpoint code after a detached verifier has launched does not make that live process
  checkpointed. The first-family destination verifier completed uninterrupted but had no per-file
  destination checkpoint; the second-family verifier used the corrected 20-row checkpoint path.
