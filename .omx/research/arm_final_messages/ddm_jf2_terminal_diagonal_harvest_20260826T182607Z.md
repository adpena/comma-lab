Executed the JF2 charter in full. The terminal diagonal produced three real byte winners, all with exact production-token decode identity:

| Candidate | Total bytes | Δ vs 127,292 B |
|---|---:|---:|
| `null` | 127,606 | +314 |
| `k002500` | 127,124 | **−168** |
| `k005000` | 127,331 | +39 |
| `k010000` | 127,678 | +386 |
| `k020000` | 128,072 | +780 |
| `k040000` | 126,350 | **−942** |
| `k060000` | 125,716 | **−1,576** |

This is `[macOS-CPU advisory / scorer-free exact model-pack and RC64 measurement]`, n600. No SegNet, PoseNet, or contest evaluator ran, so JF2 measured no `d_seg`, `d_pose`, complete score, or frontier movement.

Artifacts:

- [JF2 research memo](/Users/adpena/Projects/pact/.omx/research/ddm_jf2_terminal_diagonal_harvest_20260826.md), SHA-256 `c070c80e569593debd422cf1eec149f47e3d198790b4f116b86ed669e222c38b`
- [Terminal receipt](/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/BYTE_DIAGONAL_TERMINAL.json), SHA-256 `0a1cc640ad237374fec2bd4ce8d523d37ccf4584d9bd417a4b79b104f4b4dba5`
- [Narrowed scorer fire order](/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/wj1_target_consumer/JF2_FIRE_ORDER.json), SHA-256 `f7607371791441bd6ccc1eeb6d73b5c2bb863f33de75b81e955331b4432af9a3`

All seven complete row directories—including model packs, RC64 streams, archives, checkpoints, decoded token fields, and masks—were retained locally and mirrored byte-identically to `/Volumes/APDataStore/pact/ddm_jf2_terminal_diagonal_harvest/retained/`. No research payload was discarded or moved.

Canonical ledger validation passed with 681 rows. JF2 Items 1 and 3 are green; Item 4 correctly remains pending because byte winners now require the ordered scorer gate.

The required Git landing could not complete because the managed filesystem rejected Git object creation with `Operation not permitted`. The shared index remains empty, and I make no commit claim. A replay-verified fallback is ready:

- [Commit blocker](/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/jf2_commit_fallback/COMMIT_BLOCKER.md)
- [Bundle manifest](/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/jf2_commit_fallback/BUNDLE_MANIFEST.json)
- Patch SHA-256: `10ad39db612d52ae6b3bdd480947a660d0fe7d4f4099c7a99b80034e557551bc`

## NEXT_IF_RESUMED

- **Disposition:** `QUEUED-WITH-A-FIRE-ORDER`; **owner:** MAIN/JF1 byte-model successor; **consumer store:** `.omx/tmp/arm_receipts_local/ddm_jf1_joint_field_model_refit/wj1_target_consumer/`; **fire trigger:** WJ1 is complete, its 28,523-position target list still hashes to `bb1c…`, and the terminal receipt still hashes to `0a1c…`; **action:** run retained WJ1 coarsening followed by real JF1 refitting against the null control.
- **Disposition:** `QUEUED-WITH-A-FIRE-ORDER`; **owner:** MAIN’s exclusive full-n600 scorer custodian; **consumer store:** `/Volumes/APDataStore/pact/ddm_jf2_terminal_diagonal_harvest/scorer/`; **fire trigger:** the scorer lane is exclusively granted, AP custody revalidates, and a fresh storage preflight passes; **action:** score `null`, `k002500`, `k040000`, and `k060000` serially in chunks of at most 120 while retaining every output.
- **Disposition:** `BLOCKED-ENVIRONMENT`; **owner:** MAIN Git-writable custodian; **consumer store:** canonical Git history; **fire trigger:** Git object writes are available and the fallback manifest hashes still match; **action:** apply `COMMIT_INTENT.patch` and land the memo through the mandated serializer command recorded in the bundle.

## LIVE-HYPOTHESES

- The three byte winners may preserve enough task distortion to improve complete score. This is plausible because their gains survived physical model packing, native RC64 encoding, and exact receiver decoding, but scorer behavior remains untested.
- WJ1-targeted coarsening may compound the terminal gains. Its pinned 28,523 modeled-high-cost positions isolate a narrower refit surface, and JF2 now provides retained null and terminal controls.
- `k060000` is the strongest first scorer candidate because it saves 1,576 bytes, but it is not presumed distortion-safe; the fire order keeps all three winners to avoid rate-only selection bias.

## DEAD-ENDS

- The prediction that every terminal candidate would lose bytes is closed: three of seven won.
- Treating the S1E result as duplicate closure is closed: S1E measures a trained-renderer-width family, not JF1’s coupled field-and-model refit.
- Treating entropy-law or equation estimates as the terminal verdict is closed: the real serialized packs and native streams now provide the authority for this byte-only question.
- `null`, `k005000`, `k010000`, and `k020000` are closed as byte winners for this exact instance because they finished at +314, +39, +386, and +780 bytes.
- Immediate promotion is closed: JF2 has no realized `d_seg`, `d_pose`, or complete score.

`[contest-CUDA T4 n600]` own-vehicle frontier remains GB1: `S=0.14811799921260607 @ 180,215 B`; JF2 measured no score and did not move it.

