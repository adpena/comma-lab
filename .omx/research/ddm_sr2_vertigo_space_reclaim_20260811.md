# ddm_sr2 Vertigo certify-or-block space reclaim

**Disposition:** `MOVE_COMPLETE_VERIFIED_STOP_THRESHOLD_REACHED__MEMO_UNCOMMITTED_BLOCKED_GIT`  
**Axis:** `[byte/custody apparatus, scorer-free]`  
**Frontier effect:** none; no scorer or archive evaluation was run.

## Result

| Measure | Before | After |
|---|---:|---:|
| Vertigo available (KiB) | 25,161,592 | 254,450,808 |
| Vertigo used (KiB) | 1,927,855,332 | 1,698,566,116 |
| APDataStore available (KiB) | 967,203,712 | 735,734,272 |
| Certified source allocation (KiB) | 229,393,440 | 0 at original path; it is now a symlink |
| Certified logical data bytes | 234,889,540,091 | 234,889,540,091, full manifest equal |

The move reclaimed **229,289,216 KiB (218.667 GiB) measured by `df`**. Vertigo free space rose from
23.996 GiB to **242.663 GiB**, so the >=150 GiB stop condition is reached. The small difference
between the source tree's 229,393,440 allocated KiB and the volume-level 229,289,216 KiB delta is
reported rather than conflated; volume metadata and concurrent volume activity are different from
the retained tree-allocation census.

## Moved directories

| Original | Destination | Data files | Logical bytes | Source tree-manifest SHA-256 | Verification | Original-path compatibility |
|---|---|---:|---:|---|---|---|
| `/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns` | `/Volumes/APDataStore/pact/vertigo_coldstore_20260811/nerv_long_training_campaigns` | 4,404 | 234,889,540,091 | `81d865cd3d2ddf0ba81a5a9b76423ceac6e65fab8c5d67404f1557dd28e4443a` | Full 4,404-row manifest equal; exact largest 20 equal | Symlink installed and resolves |

Certification records live at
`/Volumes/APDataStore/pact/vertigo_coldstore_20260811/_manifests/nerv_long_training_campaigns/`.
They include the fixed pre-move census, source and destination per-file SHA-256 manifests, manifest
digests, largest-file checks, counts, `rsync` reconciliation output, before/after `df`, and the
validated `MOVED.json`. ExFAT generated 5,161 `._*` AppleDouble metadata files. They are retained but
excluded from the data-fork census and hash comparison, matching the prior verified ai1 move
precedent. The post-copy dry run contains exactly 4,404 file and 758 directory permission-only
differences (`.f...p...` / `.d...p...`), because ExFAT cannot preserve POSIX mode bits; it requests
zero content, size, time, or path transfers.

## Full >=5 GiB census

Denominator: **38/38 top-level directories classified**. Sizes are allocated KiB from the retained
pre-move `du -x` census. `UNKNOWN` means left untouched. Only the selected campaign received the
whole-tree newest-file, consumer, count, and SHA certification needed for a COLD move. Root mtimes
alone were not promoted into whole-tree coldness claims.

| # | Directory | KiB | Root mtime UTC | Classification | Reason / action |
|---:|---|---:|---|---|---|
| 1 | `pact/evidence` | 344,423,580 | 2026-07-26 | UNKNOWN | Broad live/historical evidence namespace; 431 corpus references and no whole-tree quiescence proof. |
| 2 | `pact/experiments` | 287,896,964 | 2026-06-06 | UNKNOWN | Broad results namespace; 515 corpus references and no whole-tree quiescence proof. |
| 3 | `pact/nerv_long_training_campaigns` | 229,393,440 | 2026-06-08 | COLD | Newest file 2026-06-08; later terminal ledger closes the stale active row; moved after full hash certification. |
| 4 | `pact/cold_store` | 158,534,056 | 2026-08-05 | UNKNOWN | Younger than 14 days and already a custody namespace. |
| 5 | `mac-lab` (sibling root) | 129,792,876 | 2026-03-10 | UNKNOWN | Ownership and non-Pact consumers are outside this charter's proof surface. |
| 6 | `experiments` (sibling root) | 98,030,252 | 2026-05-25 | UNKNOWN | Sibling root with 938 corpus references; ownership and live consumers unresolved. |
| 7 | `pact/ddm_pfs1_20260729` | 50,215,720 | 2026-07-29 | UNKNOWN | Not older than 14 days at survey time. |
| 8 | `pact_externalized` (sibling root) | 32,620,532 | 2026-05-31 | UNKNOWN | Externalized sibling custody; no authority to rewrite its original path. |
| 9 | `pact/ddm_cb1_perclass_carrier_byteclose_20260725T203310Z` | 32,204,672 | 2026-07-25 | UNKNOWN | Old enough by root mtime, but whole-tree newest-file/consumer proof was not completed before the stop candidate. |
| 10 | `pact_experiments` (sibling root) | 28,618,992 | 2026-05-25 | UNKNOWN | Sibling root with 107 references; ownership and consumer state unresolved. |
| 11 | `pact/public_datasets` | 26,190,552 | 2026-08-04 | UNKNOWN | Younger than 14 days and a shared dataset namespace. |
| 12 | `pact/hprc_projection_gap_repairs` | 24,915,220 | 2026-06-01 | UNKNOWN | Historical citations remain; whole-tree quiescence not certified. |
| 13 | `pact/hprc_section_value_profiles` | 19,393,856 | 2026-06-01 | UNKNOWN | Historical citations remain; whole-tree quiescence not certified. |
| 14 | `pact/hprc_residual_transform_full600_sweep_20260601T002114Z` | 16,650,784 | 2026-05-31 | UNKNOWN | Historical citations remain; whole-tree quiescence not certified. |
| 15 | `pact/snerv_mistake_b_g1a_20260609T201221Z` | 15,783,996 | 2026-06-09 | UNKNOWN | Root is old, but whole-tree newest-file/consumer proof was not completed. |
| 16 | `pact/ddm_fz1_20260804` | 14,311,484 | 2026-08-04 | UNKNOWN | Younger than 14 days. |
| 17 | `pact/g55_public_task_layered_codec_closure_x264rgb_46k_20260726` | 14,307,132 | 2026-07-26 | UNKNOWN | Root is old, but public-receiver custody and consumer state were not fully certified. |
| 18 | `pact/ddm_hm1_20260810` | 11,391,308 | 2026-08-10 | LIVE | Less than 48 hours old; boundary requires untouched. |
| 19 | `pact/compact_pact_vq_ch48_qat4_int2_mixed_full600_section_value_codex_20260602T002500Z` | 11,062,948 | 2026-06-01 | UNKNOWN | Root is old, but whole-tree newest-file/consumer proof was not completed. |
| 20 | `pact/pact_nerv_selector_v3_int8_section_value_20260601Tlocal` | 11,062,628 | 2026-06-01 | UNKNOWN | Root is old, but whole-tree newest-file/consumer proof was not completed. |
| 21 | `pact/compact_pact_vq_ch48_score_bound_full600_2000ep_int2_section_value_codex_20260601T220100Z` | 11,062,600 | 2026-06-01 | UNKNOWN | Root is old, but whole-tree newest-file/consumer proof was not completed. |
| 22 | `pact/ddm_cu1_repro_20260803` | 10,734,840 | 2026-08-03 | UNKNOWN | Younger than 14 days. |
| 23 | `pact/ddm_pe4_20260805_r2` | 10,733,572 | 2026-08-05 | UNKNOWN | Younger than 14 days. |
| 24 | `pact/ddm_td1_20260804` | 10,733,404 | 2026-08-04 | UNKNOWN | Younger than 14 days. |
| 25 | `pact/ddm_pa2_zero_byte_decode_family_20260724T194836Z` | 10,731,064 | 2026-07-24 | UNKNOWN | Root is old, but whole-tree newest-file/consumer proof was not completed. |
| 26 | `pact/ddm_sv3_20260810` | 7,620,276 | 2026-08-10 | LIVE | Less than 48 hours old; boundary requires untouched. |
| 27 | `pact/ddm_hp3_20260810` | 7,510,864 | 2026-08-09 | LIVE | Less than 48 hours old at survey time; boundary requires untouched. |
| 28 | `pact/ddm_sr1_20260809` | 7,390,580 | 2026-08-09 | LIVE | Less than 48 hours old at survey time; boundary requires untouched. |
| 29 | `pact/ddm_pe2_20260805` | 7,158,136 | 2026-08-04 | UNKNOWN | Younger than 14 days. |
| 30 | `pact/ddm_fd1_20260728` | 7,158,036 | 2026-07-28 | UNKNOWN | Approximately 14 days old; not promoted to COLD without a whole-tree newest-file check. |
| 31 | `pact/ddm_ct1_campaign_telemetry_encode_20260725` | 7,154,988 | 2026-07-25 | UNKNOWN | Root is old, but telemetry consumers were not fully certified quiescent. |
| 32 | `pact/g85_pvsa_public_receiver_20260727_r1` | 7,153,888 | 2026-07-27 | UNKNOWN | Root is old, but public-receiver custody and consumer state were not fully certified. |
| 33 | `pact_cold_store` (sibling root) | 7,153,144 | 2026-05-26 | UNKNOWN | Sibling custody namespace; ownership and consumer state unresolved. |
| 34 | `pact/hprc_mlx_component_neutralization_full600_20260531T235713Z` | 5,548,848 | 2026-05-31 | UNKNOWN | Historical citations remain; whole-tree quiescence not certified. |
| 35 | `pact/hprc_pair_scoped_residual_full600_20260601T011601Z` | 5,544,544 | 2026-05-31 | UNKNOWN | Historical citations remain; whole-tree quiescence not certified. |
| 36 | `pact/hprc_batched_baseline_profile_20260601Tlocal` | 5,544,036 | 2026-05-31 | UNKNOWN | Historical citations remain; whole-tree quiescence not certified. |
| 37 | `pact/ddm_sn1_error_source_tensor_superseded_20260723` | 5,347,420 | 2026-07-23 | UNKNOWN | Superseded label is not deletion authority; whole-tree quiescence not certified. |
| 38 | `pact/ddm_tp1_20260805` | 5,320,980 | 2026-08-05 | UNKNOWN | Younger than 14 days. |

No complete >=5 GiB tree was classified `ALREADY-MIRRORED`. The custody mirror contains selected
subtrees, not a verified byte-complete mirror of any census row. The explicitly protected
`ddm_ps135_20260810`, `pr135_joint_solve_20260810`, and `terminal_watch` trees are below the 5 GiB
denominator and were excluded entirely, as required; they were not touched.

## Consumer and custody decision

The selected campaign has 533 historical citations, so removal without compatibility would break
receipts. Its newest file is `2026-06-08T23:51:03Z`; the only apparent active-lane reference is
superseded by the later `stopped_postrun_process_absent_verified` terminal row. No current consumer
was found in the live board or current task/lane surfaces. The original path now remains as a
symlink to the verified APDataStore copy, and cited paths resolve through it after the swap.

The pre-move source contained 4,404 files, 758 directories, no symlinks, no AppleDouble files, and
234,889,540,091 logical bytes. The full source per-file manifest has digest
`81d865cd3d2ddf0ba81a5a9b76423ceac6e65fab8c5d67404f1557dd28e4443a`. Before source retirement,
the destination data-fork manifest and exact largest-20 subset both matched. The complete source and
destination manifest files each contain 4,404 rows and have the same manifest SHA-256. The original
duplicate was retired only after the verified symlink resolved to a known largest-file path.

## RECALL EVIDENCE

**Stores searched:** `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, the common contract,
`docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`, `.omx/research/**`,
`.omx/state/**`, `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, the canonical-equations registry,
task/lane ledgers, and the AP custody mirror manifests. Content queries included
`VertigoDataTier`, `certify-or-block`, `MOVED.json`, `vertigo_ai1_move`,
`nerv_long_training_campaigns`, `storage`, `custody`, `move`, and `provenance`.

**Beyond the charter seeds:** the prior ai1 move first failed closed because ExFAT generated `._*`
AppleDouble files; its v2 receipt established data-fork-only equality as the correct cross-filesystem
test. That changed this move's manifest and count rules before certification. The custody mirror audit
also showed that mirrored subtrees are not proof that a full census directory is already mirrored,
so no whole-tree `ALREADY-MIRRORED` claim was made. The stale campaign active row had a later terminal
closure, which allowed the old campaign to proceed to copy while retaining its cited path via symlink.
The canonical-equations and DAG searches found no storage-specific equation or newer routing that
changed this plan.

## Boundaries and score status

- No scorer, Modal, training, archive mutation, or exact evaluation ran.
- No file under `ddm_ps135_20260810`, `pr135_joint_solve_20260810`, `terminal_watch`, the AP mirror,
  `upstream/`, or any less-than-48-hour path was changed.
- No source byte was deleted before a destination copy, full manifest, and largest-file check.
- The census is allocated KiB; the move manifest's `bytes` field is logical data-fork bytes. They are
  different denominators and are not conflated.
- The charter's predicted >=300 GB certified-cold set was not tested to exhaustion because the first
  229,393,440 KiB candidate alone reaches the stop threshold. It is therefore neither confirmed nor
  globally falsified. The bounded pass did certify more than the charter's 50 GB falsifier floor.
- Pointer unmoved. Current own-vehicle frontier remains lc2: `S=0.16959899569230852 @ 187,226 B`
  `[contest-CUDA T4, n600]`.

## Serializer landing

`BLOCKED-GIT`: the required serializer attempted to stage only this memo and failed before changing
the empty index:

```text
error: unable to create temporary file: Operation not permitted
error: .omx/research/ddm_sr2_vertigo_space_reclaim_20260811.md: failed to insert into database
fatal: adding files failed
```

Fire order: disposition `QUEUED-WITH-A-FIRE-ORDER`; owner `MAIN`; consumer store `Git history`; fire
trigger `the managed workspace permits Git index/object writes`; action: recompute the memo's
post-edit SHA-256 and rerun `tools/subagent_commit_serializer.py` with only this memo, message
`storage: record verified Vertigo cold move [no-triality] [p0-ledger-ok]`, and the recomputed
`--expected-content-sha256`. Do not add the untracked charter or any partner work.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: Git history; fire trigger: Git
  index/object writes become permitted; action: serializer-commit only this memo with its recomputed
  post-edit SHA-256 and the recorded `[no-triality] [p0-ledger-ok]` message.
- **QUEUED-WITH-FIRE-ORDER** — owner: MAIN storage custody; consumer store:
  `/Volumes/APDataStore/pact/vertigo_coldstore_20260811/_manifests/`; fire trigger: Vertigo available
  space falls below 150 GiB; action: partition `pact/evidence` and `pact/experiments` by newest
  descendant and current consumer, then certify the next whole subtree before moving anything.

## LIVE-HYPOTHESES

- `pact/evidence` and `pact/experiments` may contain additional cold, movable subtrees because their
  roots are old and together allocate about 603 GiB, but whole-tree moves are unsafe until current
  consumers and newest descendants are partitioned.
- Several old, narrowly named HPRC and compact-renderer trees may certify quickly if Vertigo later
  falls below the safety threshold, because their root mtimes predate the 14-day cutoff and historical
  references can be preserved by symlinks.

## DEAD-ENDS

- Treating an AP custody-mirror subtree as proof of a complete mirrored census directory is closed:
  the scopes do not match.
- Treating root mtime alone as whole-tree coldness is closed: it does not prove descendant age or
  consumer quiescence.
- Treating raw destination file counts on ExFAT as equality is closed: AppleDouble sidecars inflate
  the raw count; data-fork counts and hashes are the valid cross-filesystem comparison.
- Treating successful `rsync` completion as deletion authority is closed: full destination hashes and
  largest-file checks remain mandatory.
