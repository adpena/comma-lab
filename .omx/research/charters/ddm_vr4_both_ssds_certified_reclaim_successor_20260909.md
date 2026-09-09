# ddm_vr4 — both SSDs are the binding constraint again (APDataStore 38 GiB free / 98 %; Vertigo 67 GiB / 97 %): the certify-or-block reclaim SUCCESSOR that vr3 queued with a fire order — free ≥ 120 GiB across both drives by certified move/delete, never by an uncertified byte (charter, 2026-09-09)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: codex arm (gpt-6-astra, medium) · Spawned by MAIN 2026-09-09 (operator: "Codex should be available too"). Sources: vr3 (`.omx/research/ddm_vr3_both_ssds_full_certify_or_block_reclaim_20260908.md`, `ff1095d49`: 61.4 GiB freed on Vertigo by certified deletion of 18 raws; 26 unreferenced certificate-incomplete rows QUEUED-WITH-A-FIRE-ORDER for the successor), cs1 (`.omx/research/ddm_cs1_consolidation_debt_20260909.md`, `ce582b160`: 182 BLOCKED authored-code blobs on the SSDs — code is NOT bulk; do not touch it here), CLAUDE.md "Local Disk, SSD Spill, Auto-Cleanup, And Provenance" (certify-or-block; ALWAYS KEEP THE PAYLOAD). Axes: bytes `[measured, df/du]`; `score_claim=false`.

## MANDATE
Three live arms (sj1 gen 3, rp1, bnd1) write renders, encodes and fields to the SSD tier tonight; APDataStore has 38 GiB and Vertigo 67 GiB. The tier fails closed at zero. Reclaim ≥ 120 GiB total by (1) certifying and deleting the 26 vr3 rows where a full certificate can now be completed (original path, bytes, sha256, rebuild command/argv, source archive/runtime hashes, reason rebuildable), (2) a FULL APDataStore pass under the same ledger schema (vr3 covered Vertigo fully; APDataStore only partially), (3) moving certified rebuildable bulk from the fuller drive to the emptier only when it nets space. Never delete an artifact whose certificate is incomplete; never touch anything referenced by a live seal, a live pointer row, a running process (`lsof`/`pgrep` before every move), or the 182 cs1 code blobs.

## PRIOR-LAW PREDICTION (m38 — write measured numbers beside these)
- The 26 vr3 rows are mostly inflated raws and parse-back trees of superseded candidates (predicted ≥ 60 GiB, ≥ 18 certifiable now that their seals are superseded by moves 33–37).
- APDataStore: T4 artifact dirs of promoted moves (`ddm_*_t4_*_20260909/`) hold inflated frames ~3.6 GB each; the receipts are the payload, the raws are rebuildable from the sealed archive + runtime (predicted ≥ 40 GiB certifiable across ≥ 10 dirs).
- **FALSIFIER:** if < 60 GiB is certifiable across both drives, the tier is genuinely full of live payload; report the BLOCKED table with owners and the drive the next arm must be routed to.

## SCOPE
Storage custody only. No code changes beyond the ledger tool if it needs a new field (then two review passes). No score work.

## HARD CONSTRAINTS
- `upstream/` READ-ONLY; never touch `submissions/`, any `SEAL_*.json`, any path named in `.omx/state/canonical_frontier_pointer.json` or in a live arm's tree (`ddm_sj1_multipass_token_predistortion/`, `ddm_rp1_rate_directed_predistortion/`, `ddm_bnd1_boundary_representation/`, `ddm_cmp2_compose/`), nor the 182 cs1 blobs.
- Certify-or-block per CLAUDE.md; machine-readable ledger under `.omx/research/ddm_vr4_reclaim_ledger_20260909.jsonl` (committed) with a per-row certificate; `df -h` before/after each batch, recorded.
- Detached only via `tools/launch_detached_process.py … --nice 10 --nice-best-effort`; no `nohup`/`&`; moves via `rsync --checksum` then sha verify then delete, per file, resumable.
- Serializer commits w/ post-edit `--expected-content-sha256`; tokens `[no-triality] [p0-ledger-ok]`; NEVER a Co-Authored-By or AI-attribution trailer. Checkpoint `tools/subagent_checkpoint.py --subagent-id ddm_vr4`.

## PRIOR NEGATIVE SIGNAL
- Local disk reclaim frees nothing until Time Machine snapshots thin (memory `local_disk_reclaim_frees_nothing_until_tm_snapshots_thin_20260904`) — this charter is SSD-only; do not touch local disk.
- vr3's 26 rows were left BECAUSE their certificates were incomplete; completing a certificate means producing the rebuild command and hashes, not asserting "rebuildable".
- APDataStore is ExFAT: no symlinks; leave a manifest file at every moved path.

## OPTIMAL FORM
- Reference form: vr3 (`ff1095d49`) — the full-certificate ledger, the BLOCKED table with owners, 61.4 GiB freed with zero uncertified bytes. Same schema, same tool. SCOPE reduction: none (both drives, full pass). MECHANISM reductions FORBIDDEN: no size-only deletion, no "old = rebuildable".
- **PRIOR-LAW PREDICTION (falsifiable):** as above. FALSIFIER: < 60 GiB certifiable.

## DELIVERABLE
The ledger, the freed-bytes table per drive (before/after), the BLOCKED table with owners, and the routing recommendation for tonight's arms. Commit via the serializer. End with the live frontier line.
