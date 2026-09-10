# ddm_vr6 — tonight’s reclaim is BLOCKED; no deletion batch

Owner: ddm_vr6. Tokens: `[no-triality] [p0-ledger-ok]`. Axis: **[filesystem bytes measured; scorer-free]**. `score_claim=false`.

| Result | Measured value |
|---|---:|
| Certified DELETABLE / target | **0 / 40 GiB** |
| Reclaimed by this arm | **0 B** |
| Large-file classification | **183 paths: 181 BLOCKED, 2 RETAIN** |
| Largest-ten coverage | **10/10 directories; 171 files ≥1 GiB; zero traversal errors** |
| Fresh payload hashes | **3/3 pass-5 files; 7,324,819,200 B (6.821769 GiB); all match retained receipts** |
| Full decode / rebuild / scorer jobs | **0** |

The requested capacity target was not met. No payload, seal, archive, runtime, receipt, per-pair ledger, live-arm tree, upstream file or pointer was modified, moved or deleted. This is a blocked inventory and certificate handoff, not a successful reclaim. The native `plan-retained` command ran and refused; the delivered classification ledger is explicitly BLOCKED, not a falsely relabeled native deletion plan.

## Largest-ten inventory

`vertigo_du.txt` is the complete direct-directory `du -k -d 1` result (rc=0, empty stderr). Fresh recursive metadata censuses cover every ≥1-GiB regular file in the ten largest directories, excluding symlinks. Directory allocated bytes below are du’s hard-link-aware allocation counts; file logical bytes in the JSONL are a separate denominator and must not be summed as expected capacity gain. The census is a time interval on a concurrently active filesystem, not an atomic snapshot.

| Directory under Vertigo pact | Allocated GiB (du) | Files ≥1 GiB |
|---|---:|---:|
| `evidence` | 328.467224 | 32 |
| `experiments` | 247.264397 | 84 |
| `ddm_pk4_20260813` | 108.151585 | 0 |
| `cold_store` | 66.472218 | 15 |
| `ddm_pfs1_20260729` | 47.886978 | 14 |
| `ddm_qs1_20260813` | 40.938705 | 1 |
| `pr135_joint_solve_20260810` | 31.457458 | 6 |
| `public_datasets` | 24.977257 | 1 |
| `hprc_projection_gap_repairs` | 23.761005 | 18 |
| `ddm_js5_20260812` | 23.345116 | 0 |

The final ledger also covers every ≥1-GiB file in the named tonight stores and protected rp1/tc3/sj1 lineage stores: 20 root censuses, 183 distinct paths, 484,074,805,746 logical B. Twelve predecessor-ranked roots were additionally inspected for native raw receipts; the final top-ten denominator remains distinct. The full owner table is [blocked_owners.md](ddm_vr6_20260910/blocked_owners.md); row-level reasons and owners are in [the ledger](ddm_vr6_reclaim_plan_20260910.jsonl). All protected small files remain retained regardless of the ≥1-GiB reporting threshold. Lack of a large-file row is not a cleanup authorization.

## Tonight’s prediction and the best concrete lead

| Named store | Whole directory logical GiB | Files ≥1 GiB | Disposition |
|---|---:|---:|---|
| `ddm_sj1_pass5_price` | 7.386534 | 3 | BLOCKED: alternate receipt/layout and host gate |
| `ddm_sj1_compose39_price` | 7.570469 | 3 | BLOCKED: current move-40 pointer tree |
| `ddm_bnd2_segment_code` | 1.742102 | 0 | RETAIN payloads/receipts; uncertified intermediate work BLOCKED |
| `ddm_bnd3_address_term` | 6.606948 | 0 | RETAIN payloads/receipts; uncertified intermediate work BLOCKED |
| `ddm_tc2_lane_context_map` | 5.520017 | 0 | RETAIN payloads/receipts; uncertified intermediate work BLOCKED |

These five entire stores total **28.826071 GiB**, already below the ≥45-GiB prediction before protecting a single artifact. Pass-5 + compose-39 total **14.957003 GiB**. Pass-5’s `superseded_binding_20260909` is only **1,605,666 B**; its entire `encode` subtree is **7,049,996 B**. The proposed tens of GiB in those work subtrees are not present in this inventory. Bnd3’s 48 retained twin outputs are original per-candidate payload custody; tc2’s rider and handoff feed tc3. “Negative research verdict” does not close their storage consumers.

**Best lead: pass-5 `parseback/0.raw`, 3,662,409,600 B (3.410885 GiB).** Fresh raw SHA `f7cd6a31635fcba5fce0053db19e2543c8b1999fe83d2f6f37be4f666a9a8a12` matches `PARSEBACK_RESULT.json`. Archive `eae99e0083129a911103bf691bdbf3763b9b7c0e3e365a6e3dc1cd2d2d4eec07` and runtime `bfb233be8e0caa69dbb58700656501b6edb5d0f1168af50dece7e4ee4ee4fb30` remeasure equal to the seal, using the seal’s named digest definition. The recorded argv, cwd, launch git hash, current wrapper hash, expected-field hash and receipt pins are retained in `pass5_seal_reproducer.json`. This is custody verification, not a newly executed rebuild or source-transitive reproducibility proof.

Both 1,831,204,800-B pose overlays also match their fresh hashes and their source-field receipt hashes. They were rendered from the base and full pre-subset edit fields, not the final admitted seal field. Their launch argv and body dependencies are in `overlay_reproducer_inspection.json`; transitive source/body closure is not established. The final seal alone cannot certify those outputs. No body/GT/scorer input was decoded.

## Why native planning and apply are blocked

- The actual planner still admits only regular `0.raw`, rejects every `ddm_sj1*` path, and requires the VR4 `inflated_outputs_manifest.json` / `provenance.json` / `contest_auth_eval.json` layout plus a pinned closed-advisory-instance memo. cd3/vg1 generalized the process gate, not these certificate shapes.
- An honest alternate-layout source status caused the actual `plan-retained` call to return **rc=2**, `MAIN_REHASH_INCOMPLETE:expected=0 actual=1`. Its complete argv and output are in `plan_retained_attempt.json`. The one supplied raw hash is complete; the message arises because the planner selects zero compatible source rows. No source status was forged to bypass admission.
- In the largest ten dirs, **81** legacy-layout `0.raw` files were inspected: **77** lack the required manifest at the expected path, **4** fail runtime exact file-set equality (extra README/bytecode files), **0** pass. Across all twelve predecessor-ranked inspected roots this is 82 raw paths. This is a scoped layout/custody result, not proof no alternate reproducer exists.
- Plan-time `pgrep` returned 3 (`Cannot get process list`); `ps` raised `Operation not permitted`. `lsof +D` returned 1 with no output. Empty lsof cannot establish idleness on an unobservable host. The refreshed and original observations are in `process_plan.json`.
- The stock reference scan returned **21 / 7 / 8** hits for the raw / base overlay / candidate overlay aliases. These are retained receipts, launch records and seal-associated directory references, not proof of active readers; neither were they silently waived as safe. MAIN must resolve actual consumers with host visibility.

The native apply command is supplied verbatim in [main_apply_command.txt](ddm_vr6_20260910/main_apply_command.txt) and structurally in `apply_command.json`, pinned to this ledger’s SHA. **Disposition: BLOCKED_DO_NOT_FIRE.** There is no valid nonempty deletion batch. The current native legacy apply also refuses a 40-GiB target below its 60-GiB floor; generalized apply requires a positive admitted set. The command was not run. MAIN must obtain actual admissible certificates and generate a new pinned command, rather than execute this diagnostic command or hand-edit verdicts. No engine protection was weakened in this inventory-only charter.

## APDataStore and storage routing

The complete tonight scope is direct `ddm_*_t4_*` directories ending `20260909` or `20260910`, no symlink traversal: **8 directories, 247 files, 3,026,894 B, zero files ≥1 GiB, zero traversal errors**. All are RETAIN exact-row result custody. Exact per-directory totals are in `ap_tonight_t4.json`.

At the plan sample (2026-09-10T03:38:16.477298+00:00), Vertigo had **52,292,390,912 B (48.701 GiB)** free; APDataStore had **75,109,892,096 B (69.952 GiB)**. Free-space changes during this run belong to concurrent activity, not VR6 reclaim. With Vertigo’s 40-GiB serializer reserve, route the next bounded bulk arm to **APDataStore**, subject to a fresh per-launch storage waterfall, its own reserve and aggregate demand. Do not reroute existing rp1/tc3 work.

The <20-GiB falsifier cannot conclude that the drive is “genuinely full of reproducers”: admission failed on missing/alternate custody and host observability. It does falsify the ≥45-GiB size prediction in the five named stores. Moving bytes between SSDs does not create combined capacity.

## RECALL EVIDENCE

Read the charter/common contract, PROGRAM, the operating manual, current hot board/pointer, and the applicable CLAUDE/AGENTS no-fake, payload, storage, serializer and checkpoint rules (CLAUDE and AGENTS are byte-identical). Original content recall covered research memos and arm receipts with `certif.{0,20}reclaim|superseded_binding|parse.back.{0,30}(reclaim|delet)`; index/DAG, design docs and task stores additionally used `rebuildable bulk`. Exact scopes, queries and source hits are retained in `recall_evidence.json`, `research_recall.txt`, and `index_design_task_recall.txt`. The canonical equations CLI returned **482** rows; the recorded custody/reclaim query found no relevant replacement for the storage contract.

Beyond charter seeds, DK2 documented the generation-size prediction failure and active-training harm from concurrent movers; this reinforced whole-store measurement before admission and a single payload-hash stream, with no movers. TC2’s current memo identifies a live seal-intake consumer, and BND3 retains the original twin trial payloads and intermediate outputs: those changed terminal-arm cleanup into explicit retention/blocking. The local memory registry’s serializer-denial precedent informed the one-attempt fallback handling; actual landing status is recorded live, never inferred from memory.

## Verification and landing boundary

`validation.json` verifies unique paths, complete top-ten large-file coverage, owners/dispositions, totals, three stable receipt-matching hashes, zero admissions/deletions, the exact rehash-ledger pin and the disabled apply command. Blocked/protected files outside the three hashed pass-5 rows deliberately have null current hashes; historical receipt hashes are separate. **3/183 files freshly hashed, not 183/183.** No Python source was edited and no code-test result is claimed. The artifact self-review is recorded separately; it does not clear the host gate.

The rehash ledger is `/Volumes/VertigoDataTier/pact/ddm_vr6_reclaim_20260910/rehash_ledger.jsonl`, SHA-256 `f44e265de10ccbc7657e33535967bb4b506558ade859a8ecef4e97a58cf61ef4`. Durable evidence is under `.omx/research/ddm_vr6_20260910/`; no evidence path is temporary. The serializer input preflight refused an ignored hash-progress log before any Git write (rc=13); that log is retained on disk and excluded from the Git file set. The corrected Git-write attempt uses post-edit hashes and no attribution trailer. The resulting landing boundary is recorded in `serializer_status.json`; a refused landing is handed off as `landing.patch`, not claimed as committed. Checkpoint owner is `ddm_vr6`.

All score/actuator/equation integration hooks are N/A for storage-only observations. Consumer wiring is this classified ledger plus the explicit fire orders below. No new mathematical or score law is claimed.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN / storage custodian, with sj1 owner; consumer store: `.omx/research/ddm_vr6_reclaim_plan_20260910.jsonl`; fire trigger: harvest in a host-visible session. Resolve the superseded pass-5 raw’s specific path protection, original wrapper/runtime/field reproduction chain and receipt-shaped reference hits; admit it only through supported certificate machinery, then regenerate the ledger and pinned apply command. The 3.410885-GiB lead alone cannot meet 40 GiB. Remaining large-file rows are FOLDED into this inventory as blocked, with no independent deletion fire orders.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: main Git plus `.omx/research/ddm_vr6_20260910/landing.patch` and serializer receipt; fire trigger: the corrected serializer Git-write attempt refuses landing. Land only the reviewed VR6 artifact set in a Git-writable session and verify its hashes; otherwise this item is FOLDED by successful main landing.

## LIVE-HYPOTHESES

- The 3.410885-GiB pass-5 raw is a plausible future reclaim: its fresh hash matches its decoded receipt and its retained archive/runtime match the seal. Supported admission, transitive source closure and host consumer checks remain untested.
- Additional older raw trees may have alternate valid reproducers: 77 top-ten instances lack only the stock manifest at the expected path, which is not an exhaustive search for every historical receipt. This lead is folded into the blocked inventory, not authorized deletion.

## DEAD-ENDS

- ≥45 GiB from the five named tonight stores: closed for this measured snapshot because their entire contents total only 28.826071 GiB.
- Direct native planning of the pass-5 seal layout: closed in the current implementation by its source-status, receipt-layout and blanket sj1 protection rules; the actual planner refused.
- Treating the final pass-5 seal as the complete overlay reproducer: closed because the overlays consume different pre-subset fields and ancestor body inputs.
- Empty lsof as host-idle proof: closed in this sandbox because pgrep/ps cannot observe processes.
- AP tonight T4 result dirs as multi-GiB bulk: closed in the complete eight-directory date scope; they total only 3,026,894 B.
- Deleting original candidate payloads or modifying live rp1/tc3/current-pointer trees: outside this charter’s authorized reclaim class; all retained. Drive-to-drive relocation alone creates no combined capacity.

Existing own-vehicle frontier, unchanged by this arm: **S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600]**, archive `986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857`. Recomputed from the pointer’s retained receipt components; no new score was measured.

<!-- FORMALIZATION_PENDING: storage-custody observation; no new score law or canonical equation. -->
