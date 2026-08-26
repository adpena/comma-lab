# ddm_sr3_ap_certify_compress_reclaim — lossless certify-and-COMPRESS reclaim on APDataStore closed-lane retention: free ≥25 GiB to green the w96b aligned-run storage gate

## MANDATE

w96b landed implementation-green (commit `3d9e021d07`) with its storage falsifier FIRED:
two-seed aligned-run demand after CAS dedup = 24,979,443,712 B; required (w/ 8 GiB
reserve) = 33,569,378,304 B; LIVE AP free MEASURED 2026-08-26 = 12,942,966,784 B
(12.05 GiB) — shortfall ≈ 20.6 GiB, and bs4 is actively retaining more. Vertigo free is
8.35 GiB (no cross-tier headroom). The cure is NOT deletion and NOT symlinks: it is
LOSSLESS CERTIFY-AND-COMPRESS of CLOSED-lane retention trees on AP — ALWAYS KEEP THE
PAYLOAD is satisfied by content-addressed/compressed storage (a discarded byte is not; a
losslessly-compressed, sha-manifested, round-trip-verified byte IS kept). Target: raise
AP free to ≥ 33,569,378,304 B + working margin (free ≥ 25 GiB total reclaim).

1. SURVEY + ADJUDICATE: enumerate `/Volumes/APDataStore/pact/*` trees; for each,
   adjudicate lane status against the canonical ledgers (harness task rows + lane
   registry + memos) — ONLY trees whose owning lane is CLOSED/terminal/superseded are
   candidates. Rank by (bytes × compressibility × closure-confidence). Emit the
   candidate table with per-tree receipts (the closure citation).
2. CERTIFY-AND-COMPRESS, tree by tree, smallest-risk-first:
   (a) build a per-file manifest: path, bytes, sha256 for EVERY file in the tree;
   (b) pack to a single zstd archive (long-window, high level) alongside the manifest;
   (c) VERIFY round-trip: extract to temp, re-hash every file, all shas must match —
       fail-closed, no removal on any mismatch;
   (d) only then remove the originals, leaving the archive + manifest + a
       machine-readable reclaim certificate (original tree path, total bytes, tree
       hash, archive path/sha256/bytes, verification receipt, closure citation, exact
       reconstruction command) IN PLACE at the tree root (pattern: fb2's cleanup
       certificate + sr2's certify-and-move machinery).
3. STOP CONDITION: stop when live AP free ≥ 36 GiB (target + margin) OR candidates
   exhausted. Report live `df` before/after per tree.
4. VERDICT + HANDOFF: reclaim ledger (per-tree certificates + freed bytes) + final live
   AP free + whether the w96b fire trigger (≥33,569,378,304 B) is GREEN → typed handoff
   for MAIN to fire the aligned seeds per SEALED_FIRE_ORDER_W96B.json. Ledger rows via
   tools/canonical_task_status.py (actor ddm_sr3), consuming #1165's input.

## HARD CONSTRAINTS — ABSOLUTE EXCLUSIONS (violating any = stop immediately)

- NEVER touch these LIVE stores: `/Volumes/APDataStore/pact/ddm_bs3_born_small_resolved/`
  (bs4 scorer arm writing NOW) · `/Volumes/APDataStore/pact/ddm_w96a_aligned_window/`
  (the aligned run's own store) · any tree whose owning lane appears in the live fleet
  (`tools/codex_arm_queue.py status`) · any tree modified in the last 24 h (mtime check
  per tree before touching).
- NEVER delete without the FULL chain (manifest → archive → round-trip verify → cert).
  Round-trip verification is per-file sha equality, not size. Any mismatch = keep the
  originals, mark the tree FAILED, move on.
- NEVER touch `cold_store*` / `vertigo_coldstore` trees without per-tree adjudication —
  they may already BE certified custody for other consumers; re-certifying custody bulk
  requires reading its existing certificates first (compress is allowed only if the
  existing certificate's reconstruction contract survives — i.e., update the cert).
- `upstream/` READ-ONLY. NO Modal, NO scorer. Serializer commits w/ post-edit shas
  (`.py` = 2 review passes); on git-object denial retain the serializer bundle (#1293).
- Compression tool must be deterministic + available at decode (zstd CLI or python
  zstandard; record version in the cert).

## PRIOR NEGATIVE SIGNAL (binding)

- Local-disk fallback / symlinks-as-storage CLOSED (w96a/w96b storage contract).
- Deletion-without-certificate FORBIDDEN (certify-or-block, CLAUDE.md non-negotiable).
- The bs3 fat-clone incident (#1302): a fallback that ALLOCATES on AP during pressure is
  the live hazard class — this arm's temp extraction space must be bounded and cleaned
  per tree (context-managed temp on the LOCAL disk scratch, never on AP/Vertigo).
- Dedup-alone-clears-storage CLOSED by w96b's own measurement (2.66 GB short even
  before the reserve, against a denominator that has since SHRUNK).

## OPTIMAL FORM

- Family REFERENCE exemplars w/ provenance pins: sr2's certify-and-MOVE reclaim
  (#1024, its landed machinery + certificates) · fb2's certify-and-remove cleanup
  certificate (pinned tree 26b27dce163fa2be966b980aa651d8b828e83f1e) · the CLAUDE.md
  "certify or block" rule text (the binding contract) · w96b's
  W96B_BUILD_AND_STORAGE_RECEIPT.json (the demand this serves).
- SCOPE reductions declared: closed-lane trees only this pass; custody cold_store trees
  deferred unless needed to reach target. MECHANISM reductions FORBIDDEN: no
  size-only verification, no sampling — every file hashed both sides.
- **PRIOR-LAW PREDICTION (falsifiable):** closed-lane retention (tv1 47.9 GiB · rx2
  54.6 · sa1 24.6 · wd2 31.5 · ai1 32.0 · b2e 27.9 · wc1 28.9 GiB candidates) is
  dominated by raw u8/float arrays and checkpoints with zstd ratios ≥3×, so compressing
  2–3 trees suffices to free ≥25 GiB. FALSIFIER: measured ratios <1.5× or closure
  adjudication empties the candidate list → storage routes back to #1165's operator
  boundary (pk4 cold-move 08-27) with the measured per-tree table as its input, and the
  aligned-W96 route stays storage-blocked with an honest number.

## DELIVERABLE

`.omx/research/ddm_sr3_ap_certify_compress_reclaim_20260826.md` — candidate table w/
closure citations + per-tree reclaim certificates + freed-bytes ledger + final live AP
free vs the 33,569,378,304 B trigger + typed MAIN handoff + ledger receipts +
GESTALT-DELTA line. Serializer commit (or bundle). End with the own-vehicle frontier
line.
