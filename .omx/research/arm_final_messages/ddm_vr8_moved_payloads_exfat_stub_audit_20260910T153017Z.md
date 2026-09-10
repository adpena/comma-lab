Recovered **11/11 missing payloads: 27,468,072,000 bytes**, each matching its original SHA-256.

- **26 original MOVE claims / 14 destinations / 34,792,891,200 bytes verified.** Initially 22 claims pointed to 11 missing files; **zero current MOVE failures remain**.
- Appended 11 corrections and 11 recovery records. Original log rows, 15 manifests, and 19 AppleDouble sidecars remain unchanged.
- Root cause: **MAIN’s VR7 apply, PID 34056**, deleted successfully moved payloads while leaving their redirects active.
- **Gate #419 is strict in the working tree**, with a deletion guard for declared active MOVE targets. **237 tests passed; two process-visibility tests skipped.**
- Eight older MOVE directories and seven older reclaim archives were listed without full hashing. FE1 remains unresolved: its recorded original SHA differs from its verified replacement.

Verified fallback commits: source **`9eb5c05ae34b`**, then enforcement **`dc7399262964`**. **Neither landed in MAIN** because Git object writes were denied; the shared index is unchanged.

[Audit memo](/Users/adpena/Projects/pact/.omx/research/ddm_vr8_moved_payloads_exfat_stub_audit_20260910.md) · [Complete handoff and bundles](/Users/adpena/Projects/pact/.omx/research/ddm_vr8_20260910/FINAL_HANDOFF.json)

All work respected MAIN’s GO gate. No scorer, Modal, GPU/MPS, GT decode, upstream/PR/sealed-tree edits, existing payload/stub deletion or overwrite, shared-index manipulation, review override, co-author trailer, external messages, or `/tmp` evidence. Recovery payloads and checkpoints remain on SSD.

Current frontier: **composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)**. This audit did not move it.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN; consumer store:** `ddm_vr8_20260910/FINAL_HANDOFF.json`; **trigger:** harvest in a Git-writable session, land source then enforcement, and verify current destinations.
- **QUEUED-WITH-A-FIRE-ORDER, blocked — owner MAIN; consumer store:** `ddm_vr8_20260910/fe1_original_hash_followup.json`; **trigger:** harvest the audit and resolve FE1’s original render timeline before recovery or explicit replacement-only retirement.

## LIVE-HYPOTHESES

- FE1’s deleted temporary raw may represent a different render state: four historical hashes agree but differ from the replacement, and the reclaim note describes a stalled run.

## DEAD-ENDS

- **Failed SJ1 copy/five-minute reaper:** contradicted by later full hashes and VR7’s deletion records.
- **Trusting sidecars or MOVE labels:** neither proves present payload custody.
- **Deleting active MOVE targets because rebuildable:** caused this incident; the new guard refuses it.
- **Claiming FE1 replacement equality:** contradicted by the recorded hashes.

