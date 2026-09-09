# VR4 — reclaim blocked by unavailable process visibility

Owner: ddm_vr4. Date: 2026-09-09. Tokens: `[no-triality] [p0-ledger-ok]`.
Disposition: **BLOCKED; 0 / 120 GiB reclaimed.** `score_claim=false`; `research_only=true`.
Axis: `[filesystem custody / scorer-free]`. No score work performed and no frontier move attributed to this arm.

| Drive | Inventory-end free B | Final free B | Attributable reclaimed B |
|---|---:|---:|---:|
| APDataStore | 40,521,957,376 | 40,527,855,616 | 0 |
| VertigoDataTier | 69,755,052,032 | 69,753,630,720 | 0 |

The before column is the exact free-space sample at the end of each drive's inventory, not an invented start-of-run baseline. The initial `df -h` observation was AP 38 GiB / Vertigo 66 GiB; the final rounded observation is AP 38 GiB / Vertigo 65 GiB. Small changes between the exact samples are concurrent filesystem activity, not reclaim. No mutation batches ran, so there are no batch deletion receipts.

## Binding blocker and boundaries

`pgrep -alf 'ddm_(sj1|rp1|bnd1|cmp2)|inflate|reclaim'` returned 3: `sysmon request failed with error: sysmond service not found` and `pgrep: Cannot get process list`. The independent `ps` launch raised `PermissionError(1, 'Operation not permitted')`. A single-target `lsof` returned 1 with no output, but that is not adequate proof of an idle host when process visibility is unavailable. Exact commands and output are retained in `ddm_vr4_20260909/process_gate.json`.

The charter forbids touching running-process inputs and requires liveness checks before every move. No deletion, move, compression, new payload materialization, training, scorer or paid dispatch was attempted. No existing SSD payload, source, seal, submission, pointer, or cs1 code blob was mutated. Only two small arm-owned storage-access receipts were written to new SSD directories; both root writes succeeded. No permission escalation or safety bypass was attempted. Local disk cleanup was not attempted.

## Census and certificate denominator

The full recursive metadata traversal visited both charter roots, followed no directory symlinks, counted regular files, and recorded all files at least 1 GiB. There were zero traversal/stat errors. This is a full metadata inventory, not a full SHA census or a proof that all smaller files are reclaimable.

| Drive | Regular files | Files ≥1 GiB | Raw/parseback/render candidates ≥1 GiB | Retained reproducer passes |
|---|---:|---:|---:|---:|
| APDataStore | 251,099 | 145 | 107 | 15 |
| Vertigo | 1,219,527 | 290 | 141 | 5 |
| Total | 1,470,626 | 435 | 248 | 20 |

`ddm_vr4_reclaim_ledger_20260909.jsonl` retains the VR3 row schema with explicit successor fields. It covers 435 unique large files, including all 26 queued VR3 paths. **435 BLOCKED; 0 complete certificates; 0 deleted; 0 moved.** Historical VR3 SHA values are separate from the null current `sha256`, preventing a stale hash from becoming deletion authority. Per-row owner suggestions, missing certificate fields, references, trigger and consumer store are explicit. The exhaustive grouped owner table is `ddm_vr4_20260909/blocked_owners.md`; the full top-level byte census is in `inventory.json`.

The existing `experiments/ddm_vr3_certified_raw_reclaim.py:certify_selected` was reused unchanged to verify retained archive bytes/hash, runtime exact file set and hashes, inflate script hash, upstream snapshot digest field, decode raw digest and provenance argv. Its original apply allowlist was never broadened or bypassed. Three of the 26 queued paths pass this retained-reproducer check: NI1 advisory_r1b and RR8 advisory_composed_r1 / advisory_native_r3. Vertigo's other two passes are the surplus JF2 k060000_r2 / null_r2. AP contributes 15 passes, with exact paths and full retained certificates in `ap_certificate_preflight.json`.

These 20 rows total 68.217695 GiB, of which AP contributes 51.163271 GiB. They are **untested reclaim leads**, not DELETABLE: current full raw rehash, current live-seal/reference clearance and reliable host liveness remain owed. No terabyte-scale hash job was launched after the process gate failed. The repo-only reference check covered 133 AP-plus-successor paths; 11 had hits. Other rows explicitly say not scanned. That scan is not a substitute for a live-seal or filesystem-consumer audit.

Of the queued 26, four SG2B rows fail exact runtime path-set equality because additional AppleDouble `._*` entries exist; the rule was not weakened. Nineteen lack the stock certifier's required receipt layout. A separate depth-two neighbor receipt search found alternate receiver proofs for several older HiNeRV/C1/RX1/PR135 paths. Those are leads to reconcile, not proof that the reproducer is absent everywhere.

## Predictions and routing

The 26-row predicted size is confirmed by the current inventory: 88.683003 GiB. The predicted ≥18 newly certifiable rows is **not established**: three retained chains passed, and zero gained complete deletion clearance.

The five actual AP `ddm_*_t4_*_20260909` result directories total **2,257,407 B**, with no ≥1-GiB raw under any of them. Thus the predicted ≥40 GiB in ≥10 such directories is falsified in this date-and-path scope. Those directories hold receipts/archive bytes, not downloaded full raws. Their AppleDouble siblings are separately inventoried.

The charter's <60-GiB falsifier cannot establish that the drives are genuinely full of live payload: admission was blocked by host observability before payload liveness could be determined. Zero admitted bytes is a process-safety result, not a scientific statement that no bytes are rebuildable.

**Routing recommendation:** for the next bounded write, prefer Vertigo's measured 64.963 GiB free over AP's 37.745 GiB, subject to each arm's fresh storage preflight and aggregate demand. Do not reroute or mutate sj1/rp1/bnd1/cmp2's existing live trees. A drive-to-drive move does not increase combined free space; it cannot by itself meet a ≥120-GiB combined reclaim target.

## RECALL EVIDENCE

Read charter/common contract, PROGRAM, the operating manual, current hot state/pointer, relevant CLAUDE/AGENTS storage/no-fake/serializer/checkpoint rules, VR3 and CS1. Checked the lane registry and keeper queue; this arm is live and owns no scorer slot. No predecessor checkpoint existed.

Original content recall searched research memos/arm receipts with `certif.*(raw|reclaim)|reclaim.*certif`, and index/DAG, design docs and task rows with `certify.or.block|certified.reclaim|cold.store|storage.*reclaim`; exact results/argv are in `recall_searches.json`. The canonical equations CLI returned 480 equations; a custody/reclaim search found no applicable replacement for certificate rules. The memory registry's serializer denial precedent (MEMORY.md:118) prompted checking the actual serializer fallback mechanism, not assuming a successful main landing.

Beyond charter seeds, DK2's 2026-09-04 memo showed that inflated advisory trees, not T4 receipts or burn-cell roots, held bulk; it also documented serious live-arm slowdown from concurrent movers. This changed the census to a single-process traversal and the AP certificate search to advisory work directories. SR3's 2026-08-26 memo retained whole-tree/consumer/cold-store protections, ruling out size-only whole-tree cleanup. New source verification found AppleDouble path-set drift and alternate legacy receipt layouts; neither was silently treated as safe.

## Verification and landing boundary

No source code changed, so no Python review waiver or code test claim is made. Artifact validation checks coverage, duplicate paths, totals, the 26-row join, zero admitted/deleted bytes, null current raw hashes, and the required owner/trigger fields. The canonical checkpoint records this as blocked, not complete. The serializer is invoked with post-edit SHA-256 per file, the two charter tokens, and no attribution trailer; main landing status is determined only by its receipt.

All six scientific integration hooks are N/A because this is storage-custody metadata with no score law, allocator, actuator or deployable archive. Consumer wiring is the successor ledger plus the exact fire orders below. `FORMALIZATION_PENDING`: storage observation, no measured score law.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN / storage custodian; consumer store: `.omx/research/ddm_vr4_reclaim_ledger_20260909.jsonl`; fire trigger: this handoff is harvested in a session with working process visibility. Refresh the 20 retained-reproducer leads first, obtain current raw hashes and live-seal/process/reference clearance, then admit only complete certificates and execute the existing certified reclaim discipline toward ≥120 GiB. Reconcile alternate receipts for remaining rows only where needed; do not infer deletion eligibility from this blocked ledger.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: main Git and the serializer receipt under `/Volumes/VertigoDataTier/pact/ddm_vr4_reclaim_20260909/serializer/`; fire trigger: serializer returns a fallback bundle. Verify and consume that exact packet in a Git-writable session; do not mark the ledger committed until main ancestry and content hashes agree.

## LIVE-HYPOTHESES

- Twenty retained-reproducer rows may supply 68.218 GiB once current bytes and liveness are verified; their archive/runtime/decode chains passed the existing certifier.
- Some older queued rows may be certifiable through alternate receipts; the bounded neighbor search located explicit receiver-proof and decode documents outside the stock layout.

## DEAD-ENDS

- Empty lsof output as sufficient clearance is closed for this session: pgrep and ps cannot observe the host.
- The five Sep09 AP T4 result directories as ≥40 GiB of reclaimable raws is closed by their complete 2,257,407-B inventory.
- Silently ignoring SG2B runtime path-set drift is closed: exact current runtime equality failed on four queued rows.
- Size/age-only deletion, source-code reclaim, live-tree mutation and local-disk cleanup remain outside authorization.
- Moving bytes between the two SSDs alone cannot create combined capacity; it only redistributes it.

Existing own-vehicle frontier, unchanged by VR4: S 0.13791730003757818 @ 180,388 B [contest-CUDA T4 n600], read from the current local pointer.

<!-- # FORMALIZATION_PENDING: storage-custody memo with no measured score row of its own; the reclaim ledger is the artifact, no canonical equation applies -->
