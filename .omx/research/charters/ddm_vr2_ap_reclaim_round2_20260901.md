# ddm_vr2_ap_reclaim_round2 — APDataStore certify-and-MOVE reclaim: unblock rxc1 gen-3 (AP at 100%, 1.2 GiB free) + the #1165 round-2 residuals (task #1165; memo ddm_x012_crossing_ledger_20260901.md names the consumer chain)

## MANDATE

Operator standing program: certify-and-MOVE reclaim (the #1364 grant precedent: 43/44 rows
certified-moved hash-verified retire-with-symlink; ddm_sr2/sr3 lineage). LIVE BLOCKER this arm
cures FIRST: **APDataStore is at 100% (1,263,927,296 B free)** and the rxc1 gen-2 arm exited
`BLOCKED(storage-reserve)` at 26/32 sealed rows — its typed fire trigger is "APDataStore has at
least 1,400,000,000 free bytes with no concurrent decline" (arm final message
`.omx/research/arm_final_messages/ddm_rxc1_gen2_screen_resume_20260901T143849Z.md`, blocker
receipt `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/BLOCKER.json` sha
`581a0768…d839b`). The gate-1 SCMDL screen (#1374, the campaign's live pricing instrument) is
storage-deadlocked until this arm frees space. MAIN's direct attempt was permission-refused at
the retire (rm) step — the reclaim belongs to this chartered owner, with full certify-or-block
discipline.

## SCOPE

1. **FIRST DELIVERABLE (unblocks rxc1): free ≥10 GB on APDataStore** by certified MOVE to
   VertigoDataTier (measured ~254 GiB free). Pre-identified candidate: the already-cold slice
   `/Volumes/APDataStore/pact/vertigo_coldstore/pact/ddm_pfs1_20260729` (~48 GB — cold-store
   content parked on AP when Vertigo was full; pfs1 is a closed lane, #772/#1078 lineage).
   Verify no live reader before moving; a smaller sufficient certified candidate is legal if
   verified faster. Protocol per #1364: rsync copy → full-checksum verify (rsync -a -c dry-run,
   0 diffs required) → machine-readable MOVE_CERT (schema certified_cold_move.v1: src, dst,
   files, bytes, verify method, utc, reason) → remove source ONLY after verify → symlink at the
   old path. NEVER delete unverified bytes; certify-or-block. MAIN's prepared (never-run) script
   is at `/Volumes/VertigoDataTier/pact/coldstore_returned_from_ap/ap_reclaim_move.sh` —
   reusable as reference, but the arm re-derives and owns its own execution. Any single step
   projected >30 min runs DETACHED via `tools/launch_detached_process.py` (script paths must
   avoid claude/codex tokens — the fleet-reaper argv predicate; measured refusal this session).
2. **Emit the rxc1 unblock receipt**: after the move, record measured AP free bytes; if
   ≥1,400,000,000 with no concurrent decline, write a typed READY note into the memo naming the
   rxc1 gen-3 resume fire order (MAIN spawns it; this arm does NOT touch
   `experiments/ddm_rxc1_restartable_exact_coder.py` or
   `/Volumes/APDataStore/pact/ddm_jc1/restartable_exact_coder/` beyond READ of BLOCKER.json).
3. **#1165 round-2 residuals as budget allows**: wwc1 custody 2.09 GB
   (`/Volumes/APDataStore/pact/` wwc1 store, routed by #1360) certify-and-MOVE · the pk4
   cold-move row (due 08-27, overdue) · record the 2 tool debts as typed rows if not payable
   here. Loop-until-target: continue certified moves of already-cold AP content until AP free
   ≥50 GB OR candidates are exhausted, whichever first; every move gets its own MOVE_CERT.
4. **Standing-hazard note**: AP at 100% is a fleet-wide writer hazard (sg2b stores, keeper arm
   stores, serializer reserves). Report the post-reclaim tier map (free bytes per tier).

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire. NO scorer runs. The local SCORER LANE belongs to MAIN.
- NEVER delete without a verified MOVE_CERT (certify-or-block, ALWAYS KEEP THE PAYLOAD — this
  arm moves bytes, it never discards them). Retire-with-symlink at every moved path.
- Do NOT touch live arm stores: `/Volumes/APDataStore/pact/ddm_jc1/` (rxc1, READ-ONLY),
  `/Volumes/APDataStore/pact/ddm_sg2b_*` (READ-ONLY; its Vertigo fire_main has a LIVE p03 run),
  `/Volumes/APDataStore/pact/ddm_xov1_crossover_pass/` (fresh retained custody).
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- DETACHED >30-MIN COMPUTE per the canonical launcher; pidfile + done-receipt; monitor, never
  in-session multi-hour loops.
- Bulky receipts to `/Volumes/VertigoDataTier/pact/ddm_vr2_ap_reclaim_round2/` (NOT AP — AP is
  the disk being drained).

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- sr3 carve-out deadlock (#1336): protection-by-declaration deadlocked archiving BOTH ways —
  verify every carve-out/reference BEFORE moving, and update references at move time.
- #1302 serializer fat-clone breach: an 8.4 GiB unplanned copy landed on an SSD — measure the
  destination free space before every copy and refuse if projected free < 20 GiB post-copy.
- #1003 permission-error-wearing-a-capacity-mask: if a write fails, distinguish permission vs
  capacity at source (touch-probe both tiers) before diagnosing.
- The #1122 AppleDouble class: ExFAT sidecars (._*) are real bytes — count them in certs and
  never let a sidecar mismatch fail a checksum verify silently; name them explicitly.

## OPTIMAL FORM

- Family exemplar: the #1364 executed grant (43/44 rows, hash-verified, retire-with-symlink,
  manifest per row) is the reference form; ddm_sr2's certify-and-MOVE memo is the sister.
  Provenance pins: rxc1 final message file above · BLOCKER.json sha `581a0768…d839b` · crossing
  ledger memo sha d82090f1c40949fa1af8f570bcd85a9dc70cce7e5d1acb8148eaafd25af30890 (b572096f65).
- SCOPE reductions legal: smallest-sufficient-move-first (unblock rxc1 before the 50 GB
  target); per-directory moves. MECHANISM reductions FORBIDDEN: full-checksum verify on every
  moved tree, real certs, no sampling.
- **PRIOR-LAW PREDICTION (falsifiable):** the pre-identified 48 GB slice is already-certified
  cold content with no live reader, so the move completes with 0 checksum diffs and AP free
  rises to ≥45 GB. FALSIFIER: a live reader or reference into the slice is found — then STOP
  that candidate, record the reference per the #1336 lesson, and take the next candidate.

## DELIVERABLE

`.omx/research/ddm_vr2_ap_reclaim_round2_20260901.md` — typed rows: per-move {src · dst ·
bytes · files · verify result · MOVE_CERT path+sha · symlink} + measured before/after tier map +
the rxc1 gen-3 READY note (or the named blocker) + #1165 residual dispositions + DEAD-ENDS +
denominator (candidates enumerated/moved/refused). Commit via the serializer. End with the
own-vehicle frontier line (S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600], afr1 sha
cbb8d928…d405bf25 — UNMOVED unless a fire order lands).
