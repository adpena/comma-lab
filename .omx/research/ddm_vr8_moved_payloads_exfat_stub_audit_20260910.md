# ddm_vr8 — MOVE destinations deleted later; recovery and custody enforcement

Axis: **[macOS-CPU byte/custody apparatus]**. `research_only=true`; `score_claim=false`.

**RECOVERY VERIFIED; MAIN LANDING STILL REQUIRED.** All **11/11** missing payloads
(**27,468,072,000 B**) were restored at the original destination and match the recorded
complete SHA-256. All **26 original MOVE claim rows / 14 unique destinations /
34,792,891,200 B** now have verified custody; **zero current MOVE failures** remain.
Eleven correction rows and eleven recovery rows were appended without changing the original
MOVE_LOG prefix or any of the 15 source manifests. The separate FE1 reclaim finding remains
unresolved: its recorded old-path SHA differs from the retained replacement. This is not a
blanket clean-reclaim verdict and not a score measurement.

## CENSUS

The exact charter find over both SSD roots at depth <=4 returned **26 files**: 15 current
per-file manifests, one 11-row MOVE_LOG, eight older directory MOVE certificates, and two
incidental filename matches (a copied-manifest provenance receipt and a PYCACHE REMOVED
receipt). The 26 current claim rows reduce to **14 unique destination paths** because the
TC3 retained manifest also points at RP1's raw, and all eleven SJ1 moves appear twice.
`certificate_inventory.json`, `move_rows.json`, and `initial_destination_census.jsonl`
retain paths, source states, full current hashes where bytes exist, errors and denominators.

| Destination (absolute path) | Source state | Initial state | Expected B | Original SHA-256, now verified |
|---|---|---|---:|---|
| `/Volumes/APDataStore/pact/ddm_rp1_round1_bulk/parseback_0.raw` | absent | VERIFIED | 3,662,409,600 | `fd4b08e6aa0e967cc1453b9363c5f5ad243cb3bb57417f0f383cdb4bbfbfb2e8` |
| `/Volumes/APDataStore/pact/ddm_rp1_round1_bulk/overlay_base_odd_frames.u8` | absent | VERIFIED | 1,831,204,800 | `aaf85da8282d090fbc56d0e0af43d8b63ebee6486e26b3b3973ec9504c9219da` |
| `/Volumes/APDataStore/pact/ddm_rp1_round1_bulk/overlay_cand_odd_frames.u8` | absent | VERIFIED | 1,831,204,800 | `8c3fecf0bdb578104bb7a220abebcf568ca50a4fe4d94938b85b1e66bf6d9eca` |
| `/Volumes/APDataStore/pact/cold_store_sj1_bulk_20260910/ddm_sj1_multipass_token_predistortion/candidate/parseback/0.raw` | absent | MISSING | 3,662,409,600 | `5aa5ffe54837f07cce002830b8623b73e185abde49f6a96b2be1bb9d2d448879` |
| `/Volumes/APDataStore/pact/cold_store_sj1_bulk_20260910/ddm_sj1_multipass_token_predistortion/candidate_pass3/parseback/0.raw` | absent | MISSING | 3,662,409,600 | `bfe96bac3066dfec78cf32075582686ac74e6c1756df6008522e589bf0709b4b` |
| `/Volumes/APDataStore/pact/cold_store_sj1_bulk_20260910/ddm_sj1_multipass_token_predistortion/pose/overlay/odd_frames.u8` | absent | MISSING | 1,831,204,800 | `35b001e499232cdda63bc805574df13976553fb3498833feeeefa7f66eb9b253` |
| `/Volumes/APDataStore/pact/cold_store_sj1_bulk_20260910/ddm_sj1_multipass_token_predistortion/pose_pass4/overlay/odd_frames.u8` | absent | MISSING | 1,831,204,800 | `bbbea5058041a39e05ffacfe8d97a70b441f589b761a831570a9878c8903ae18` |
| `/Volumes/APDataStore/pact/cold_store_sj1_bulk_20260910/ddm_sj1_multipass_token_predistortion/pose_pass3/overlay/odd_frames.u8` | absent | MISSING | 1,831,204,800 | `4560927f432a20db598d028dadf90614967d6f11a54125803d87e686306e2859` |
| `/Volumes/APDataStore/pact/cold_store_sj1_bulk_20260910/ddm_sj1_compose39_price/parseback/0.raw` | absent | MISSING | 3,662,409,600 | `c5a7986cf3f16360a0ef5f197ad4b5cc4cdf4c143f0c8a4319460476fc986ea5` |
| `/Volumes/APDataStore/pact/cold_store_sj1_bulk_20260910/ddm_sj1_compose39_price/pose/base_overlay/odd_frames.u8` | absent | MISSING | 1,831,204,800 | `6966a44acf5566ed1894b2e4d4110a00c25313643461458b1d6b80a2d79b3f53` |
| `/Volumes/APDataStore/pact/cold_store_sj1_bulk_20260910/ddm_sj1_compose39_price/pose/cand_overlay/odd_frames.u8` | absent | MISSING | 1,831,204,800 | `37ea3842982a7fe7d1c4c06216fde20e5bc4d1b6c15841d71981085f0fe2c653` |
| `/Volumes/APDataStore/pact/cold_store_sj1_bulk_20260910/ddm_sj1_pass5_price/parseback/0.raw` | absent | MISSING | 3,662,409,600 | `f7cd6a31635fcba5fce0053db19e2543c8b1999fe83d2f6f37be4f666a9a8a12` |
| `/Volumes/APDataStore/pact/cold_store_sj1_bulk_20260910/ddm_sj1_pass5_price/pose/base_overlay/odd_frames.u8` | absent | MISSING | 1,831,204,800 | `aaf85da8282d090fbc56d0e0af43d8b63ebee6486e26b3b3973ec9504c9219da` |
| `/Volumes/APDataStore/pact/cold_store_sj1_bulk_20260910/ddm_sj1_pass5_price/pose/cand_overlay/odd_frames.u8` | absent | MISSING | 1,831,204,800 | `494484d210cc91ac33ff5107ea40616509b3a6536f541aa9845fc0c8164f80cc` |

All eleven initially missing payloads have an adjacent **directory** AppleDouble sidecar;
none has a `._<payload filename>` sibling at the exact missing path. Their payload
subdirectories were empty at the initial census and now contain the restored payloads. The three retained RP1 data files coexist with their own
AppleDouble siblings and are real full-size files. `stub_evidence.json` pins 19 distinct
sidecars without moving/deleting them. A surviving sidecar does not establish failed copy.

The explicit additional filename census (`*reclaim*.json*`, `*delete*log*.json*`, depth <=4)
found eight more SSD certificates. Seven SR3 archives are August records, listed/stat-ed,
not freshly payload-hashed. FE1's certificate has no embedded date; filesystem mtime
2026-09-09 conservatively brings it into scope. Its replacement raw is present and fully
hashed: **3,662,409,600 B**, SHA
`30964f538c9c5a94095ad295919ce82c5a272e109dba7700ea9bc1e63ca4b455`.
Its missing original raw has no recorded SHA in that certificate, and its command contains
placeholders. A subsequent targeted log search recovered a historical full-file SHA for the
old path in all four shard logs (line 171): `bbb0230c09aedb1a2e61ff32d578636a1ecb9de71acfda472d9889907eb2c9ae`.
Those same logs later name the AP replacement and its measured `30964f53...` SHA. Source
`ddm_fe1_frame_embedding_search.py:513` computes the entire input-file hash. The replacement
launch manifest supplies its real argv, retained in `fe1_original_hash_followup.json`.
Verdict: **ORIGINAL_PATH_RECORDED_HASH_DIFFERS_FROM_REPLACEMENT**; the deleted original is
not recovered, and its completion/timing relationship to deletion remains unproven. The
replacement is not certified byte-identical. MAIN owns the registered blocked follow-up. The eight older MOVE directories and seven
older reclaim archives are explicitly outside the fresh bulk-hash denominator.

The repository-side VR3/VR5/VR7 plan/journal census contains **3,887 path-bearing rows**,
including repeated plans and PRE_DELETE/DELETED observations. Recent successful VR3/VR5/VR7
records contain **1,247 unique deletions / 163,219,215,550 logical B**: VR3 18 raws
(applied_at_utc on 2026-09-09), VR5 17 raws, and VR7 1,212 files. This is historical deletion accounting, not bytes reclaimed by VR8 and not
measured free-space gain. DELETED certificates do not assert a present MOVE destination;
they are not counted as missing MOVE payload failures. Every row's current source state is
in `reclaim_row_census.jsonl`. Root-unavailable/error counts are explicit in discovery receipts;
no absence outside these filename/depth scopes is claimed.

## ROOT CAUSE — actor and timing

**SOURCE-VERIFIED and RECORDED:** MAIN's original mover is
`.omx/research/ddm_main_storage_20260910/vertigo_move_sj1.py`, not mv1's helper.
Its launcher names PID **23157**, start **2026-09-10 04:42:39Z**, git
`89295a8233baf8a6d7ec50b35bef874885f95996`. The script hashes source, runs `shutil.copy2`,
hashes destination, compares, writes MOVED, then unlinks source. The done receipt says
rc=0 after **1,610.175214 s**, finishing 05:09:29Z with 11 files / 27,468,072,000 B.

**The deletion actor is MAIN's VR7 apply**, PID **34056**, launched **06:24:05Z** from git
`03efa6682b8dd2ab727f4b7f4beb1069776a9801` by
`.omx/tmp/codex_runs/vr7_apply_20260910/launch_manifest.json`. Its exact argv and pinned
ledger SHA are retained there. The eleven SJ1 deletions span
**06:28:41.777008Z–06:38:08.271142Z**. Each PRE_DELETE records
`raw_sha256_verified_current=true`. The code hashes immediately before `path.unlink()`;
it deletes only the data fork and neither updates the source MOVED redirect nor appends
retirement to MOVE_LOG. The VR7 observation exclusions treated those location manifests as
historical text while they remained consumable redirects. That is the custody lifecycle bug.

For `37ea3842…`, the MOVE manifest records 04:56:40Z; VR7's separate rehash row 6 records
1,831,204,800 real B at **05:54:57.252618Z**; apply journal rows 18/19 record pre-delete
verification and deletion at **06:36:32.622217Z**. The raw bytes therefore existed after
the move. Failed original copy and five-minute reaper explanations are contradicted for
this eleven-path INSTANCE. VR7 completed all 1,212 deletions; rc=3 meant the net `df` gain
failed its target, while its logical deletion target passed. VR5 has no SJ1 cold-store
path in its apply journal. Exact sidecar-creation provenance is unknown and not guessed.

`provenance_pins.json` pins rp1's memo, VR7's memo/rehash/journal, the original mover and
both launcher manifests. mv1 requested landing `0e8e2e420` was verified in Git history.
The original mover source and historical receipts remain unchanged.

## RECOVERY AND APPEND-ONLY CORRECTION

Eleven `MOVED_CERTIFICATE_FALSE` events were appended to the existing AP MOVE_LOG using
VR3's landed `append_fsynced`, with `claim_scope=CURRENT_DESTINATION_CUSTODY_ONLY` and the
VR7 deletion evidence. Historical rows were not edited. `MOVE_LOG_before.jsonl` and
`corrections.json` preserve the exact prefix and its SHA. New `RECOVERED` rows are appended
only after expected size and complete SHA equality; restoring the original destination
repairs every immutable source redirect without editing any sealed tree.

Two direct copies use read-only retained sources: the c5a7986c… raw from
`/Volumes/VertigoDataTier/pact/ddm_dwc1_decode_wall_clock/move40/output/0.raw`, and the
37ea3842… overlay from `/Volumes/VertigoDataTier/pact/ddm_rp1_round2/pose/overlay_base/odd_frames.u8`.
`copy_recovery.jsonl` holds the reread/hash receipts. Failed copies remain retained.

Six other overlays use the exact SJ1 field loader and JG1 renderer from the pinned VR7
per-file configs, writing 600 frame checkpoints. Three raws use the actual sealed F26
receiver input loader and its retained complete token checkpoint, then the exact CPU
batch-one master/carrier operations and actual selected frame-0 mode for every pair.
Original native libraries are pinned/copied; no scorer or token re-decode is invoked.
The frame writer change is a storage adapter, and the recorded whole-file SHA is its
acceptance authority. Reduced-input tests are regression controls, not empirical recovery.
Every frame/pair is persisted; interrupted uncheckpointed suffixes are preserved and refuse
truncation. Restore configs and retained inputs are hash-pinned, seed 20260910, four CPU
threads. Each job checks SSD remaining bytes plus 8 GiB reserve and requires MAIN's GO.
All rebuilt outputs go directly to the missing AP destination; private checkpoints remain
under `/Volumes/VertigoDataTier/pact/ddm_vr8_audit/retained/`. No cleanup deletes are enabled.

## SELF-PROTECTION AND LANDING

First landing: `tac.artifact_moved.verify_payload`/`copy_verified` reject AppleDouble names
and magic, symlinks/nonregular files, size/SHA drift and destination identity changes.
Certification rereads/hashes actual destination bytes; failure retains source and copied
bytes. Existing tiny legitimate payloads remain supported: content and name identify a
stub, not an arbitrary 4-KiB cutoff. All chartered bulk payloads are larger than the stubs.

Second landing: Catalog **#419**, `check_moved_payload_destinations`, dynamically discovers
file MOVE certificates on both SSDs at depth <=4, checks current destination stat/header,
and refuses malformed/empty/unavailable inventories. Corrections alone do not waive a
missing historical destination. SHA checking is an explicit gated census operation;
the regular preflight reads only small headers/stat. Recognized older directory-level
schemas remain a separate tree audit. The final live census measured zero violations across 16 current certificate files, and
`preflight_all()` now calls #419 with `strict=True`. The FE1 replacement-identity finding is
separate and does not get laundered by this file-MOVE gate. The CLI entry guard was moved
after the check definitions so both #417 and #419 are defined before execution.

Current fixture verification: **237 tests passed, two process-visibility tests skipped under sandbox restrictions** across mover, scanner, preflight and
both restoration adapters and VR3/VR5/VR7 deletion controls; Ruff checked owned modules. Every Python landing needs two
current review_tracker passes; receipts are in this arm's directory. The first serializer
returned **rc=17 Git object-store denial** and retained fallback commit
`9eb5c05ae34bb38c9261743cd22f75a55a711421`; bundle verification passes. It is **not a MAIN
landing**. The shared staged index was empty at the first attempt and remains preserved.
The deletion apply now independently refuses any target still named by its descriptor-declared, hash-pinned MOVED manifest, including historical-observation exclusions. This closes the demonstrated deletion path; it is not a global reverse-reference scanner or a manifest-retirement transaction. Final recovery hashes and census are complete; `FINAL_HANDOFF.json` is the post-serialization result authority and supplies the enforcement commit identity.

## RECALL EVIDENCE

Read charter/common contract, PROGRAM, no-fake/storage/review/serializer portions of both
governing files, operating handoff and live hot state. Read all seven 2026-09-09/10 directive
files selected by filename; none changes this custody authorization. MAIN_GO_CUSTODY_AUDIT
was present before hashing/copying/re-rendering. Lane registry had no ddm_vr8 row; no scorer,
GPU or remote lane was claimed or dispatched. Checkpoints use ddm_vr8.

Queries are retained in `recall_0.json`, `recall_1.json`, `task_recall.json` and
`governing_recall.json`: `AppleDouble|stub.only|MOVED.certificate|MOVE_LOG|cold.store.{0,20}reclaim`
over research Markdown; `AppleDouble|MOVED.json|certif.{0,20}reclaim` over docs, canonical
research index, DAG and task ledger. An initial wrong task filename produced a recorded
rc=2 and was corrected to `canonical_task_status.jsonl`; it is not treated as empty coverage.
The equations CLI returned **483 rows**, with no AppleDouble/MOVED/reclaim equation match.

Beyond charter seeds: SA3's source-parse failure from AppleDouble *.py companions motivates
checking content magic as well as names. The SR3 directory certificates and FE1 incomplete
reclaim certificate expanded the inventory and prevented a blanket zero-failures claim.
VR7's actual DELETED rows changed the root-cause conclusion from suspected failed copy to
stale redirects after deliberate reclamation. Named memory laws were read directly; Codex
memory registry reinforced non-destructive custody and scoped negatives, not live byte facts.

The six scientific integration hooks (sensitivity, Pareto, allocator, autopilot, posterior,
probe) are N/A to this storage-only change: it changes no scored representation or scientific
law. Operational consumers are the original MOVE_LOG/redirects, preflight #419, and MAIN's
serialized landing packets. `research_only=true` does not waive actual custody verification.

## BOUNDARIES

No scorer, Modal, GPU, MPS, eval, GT decode, score or candidate promotion. No upstream/PR/
sealed-tree edits; rp1 round2 and sj1 compose39 sealed candidate were read-only. No existing
payload or stub was deleted or overwritten. Restores write only previously missing payloads
and owned append-only checkpoints. SSD routing/reserve checks and GO gate are active; there
was no foreground sleep awaiting GO. No bare Git commit, shared-index edit/stash, Python
review override, co-author trailer, or external message. No /tmp evidence is used. Existing
research logs, manifests, and unrelated work were preserved. Git landing remains permission-
blocked; source fallback is real, MAIN landing is not claimed.

# FORMALIZATION_PENDING: no new canonical law registered; storage-custody implementation and full-file recovery audit, not scientific score progress.

Existing frontier observed live: **composition S 0.1374765052591843 @ 180,238 B
[contest-CUDA T4 n600] (move 42)**. MAIN moved it while VR8 ran; this arm did not move it.

## FINAL CUSTODY RECEIPTS

`final_destination_census.jsonl` carries all 14 full SHA results and their hash times, plus
final unchanged stat identities/header checks. `final_claim_census.jsonl` expands them back
to every original claim and source state. `final_census_summary.json` records 26 original
claims, 22 initially false current-location claims, 11 corrected/recovered payloads, 33
final MOVE_LOG rows (11 original + 11 correction + 11 recovery), and the unchanged prefix.
`discovery_recheck.json` confirms the same 26 inventory files, with no additions/removals.
`stub_preservation_recheck.json` proves all 19 original sidecars unchanged. Full hashes were
computed in this audit; the final census reuses those hashes only when stat identity remains
unchanged, and does not pretend the metadata preflight itself recomputes SHA.

The three initial raw config drafts (`raw_0000_restore.json`, `raw_source_0000.json`,
`raw_source_0006.json`) were abandoned before launch when positional descriptor selection
proved invalid. They are retained as preparation evidence, not authoritative jobs.
`raw_jobs_verified.json` and `overlay_jobs.json` name the actual launch configurations.
All nine render certificates prove original SHA equality; two more payloads were copied.

## NEXT_IF_RESUMED

- QUEUED-WITH-A-FIRE-ORDER; owner MAIN; consumer store `.omx/research/ddm_vr8_20260910/FINAL_HANDOFF.json`; fire trigger: harvest the reviewed packets in a Git-writable MAIN session, land source first then enforcement and verify the live destination census. Canonical task `ddm_vr8_two_landings_20260910`.
- QUEUED-WITH-A-FIRE-ORDER (blocked); owner MAIN; consumer store `.omx/research/ddm_vr8_20260910/fe1_original_hash_followup.json`; fire trigger: harvest this audit, establish the FE1 original render completion/runtime timeline against the four logged old-path hashes, then recover a certified original or append explicit replacement-only retirement. Canonical task `ddm_vr8_fe1_identity_20260910`.

## LIVE-HYPOTHESES

- FE1's removed temporary raw may represent a different render state from the replacement: all four old-path full hashes agree with each other and differ from the retained replacement, while the reclaim note names a stalled run. The precise cause and original byte identity at deletion remain unproven; the registered MAIN task owns this lead.

## DEAD-ENDS

- Failed SJ1 original copy or a five-minute reaper kill, INSTANCE scope: successful move receipts and later independent full hashes prove the payloads existed; VR7's deletion journal names the subsequent actor.
- Recovering by trusting a MOVED label or surviving AppleDouble metadata, INSTANCE scope: neither certifies present data bytes; the restored originals now pass full SHA checks.
- Allowing VR7 to delete a live redirect target because it is rebuildable, INSTANCE scope: that created the missing-byte obligations; the new apply guard refuses it.
- Claiming the FE1 replacement is byte-identical to the removed raw, INSTANCE scope: current replacement SHA differs from all four historical old-path hashes.
