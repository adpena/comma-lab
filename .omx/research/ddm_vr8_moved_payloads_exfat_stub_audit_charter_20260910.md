# ddm_vr8 — audit every certified MOVE for ExFAT `._`-stub destinations; re-materialize and re-certify (charter, MAIN 2026-09-10)

## The incident (P0: ALWAYS KEEP THE PAYLOAD)
ddm_rp1 (round 2, 2026-09-10) found that sj1's move-40 overlays were cold-stored to destinations that
contain ONLY ExFAT AppleDouble `._` stubs, while the MOVE log records them `MOVED` with shas — five
1.83 GB payloads including `parseback/0.raw`. rp1 recovered by re-rendering and reproduced sj1's
recorded overlay sha `37ea3842…` exactly, so the bytes are REBUILDABLE — but a MOVED record whose
destination is a stub is a false certificate, and the certify-or-block rule was violated silently.
MAIN's own 2026-09-10 certified MOVE of the 25.6 GiB sj1 bulk (MOVED.json) and the vr5/vr7 reclaim
applies are the suspects; treat every MOVED/reclaim certificate written since 2026-09-09 as unverified
until this audit proves each destination.

## Read first
- rp1's note and memo: `/Volumes/VertigoDataTier/pact/ddm_rp1_round2/NOTE_TO_MAIN_base_mismatch_20260910.md`,
  `.omx/research/ddm_rp1_round2_move40_20260910.md` (§ on the custody finding; the exact paths and the
  re-render command that reproduced `37ea3842…`).
- The MOVED apparatus (landed today): `src/tac/artifact_moved.py` (`resolve`, `move_with_manifest`,
  `violations`), `src/tac/artifact_moved_gate.py`, `experiments/ddm_vr3_certified_raw_reclaim.py`
  (`moved_row`, `plan-retained`), the vr5/vr7 memos `.omx/research/ddm_vr5_*`, `ddm_vr7_*`, and every
  `MOVED.json` / `MOVE_LOG.jsonl` under `/Volumes/VertigoDataTier/pact/` and `/Volumes/APDataStore/pact/`
  (`find /Volumes/*/pact -maxdepth 4 \( -name 'MOVED.json' -o -name 'MOVE_LOG.jsonl' -o -name '*MOVED*.json' \)`).
- Memory laws: `landing_an_arm_bundle_guard_on_file_count_and_skip_exfat_dot_underscore_stubs_20260910`
  (the `._` stub class), `always_keep_the_payload_never_run_a_measure_and_discard_20260809`.

## Deliverable
1. A census: every MOVED/reclaim certificate on both SSDs with, per row, the destination path, whether
   the destination is a real file (size + sha256 recomputed) or a `._` stub / missing, and the source
   path state. Hash every destination that claims to be a payload; report counts and bytes.
2. For every false certificate: re-materialize the payload — from a retained source copy if one exists
   (hardlink/copy + sha verify), else by the recorded rebuild path (rp1's re-render reproduced the sha;
   run it ONLY after MAIN's GO if it needs the scorer or > 1 core — see boundaries). Rewrite the log rows
   APPEND-ONLY: never edit a historical row; append a correction row with `status: MOVED_CERTIFICATE_FALSE`
   + the recovery row `RECOVERED` with the new destination sha, using the landed helpers.
3. Root cause: WHY did the copy leave only a stub? (ExFAT + AppleDouble on a large copy that was
   interrupted or rate-limited; `cp`/`shutil` behaviour; the fleet reaper killing the mover at ~5 min;
   a `find` that matched `._` files as the payload.) Name the actor from the logs (`MOVE_LOG` timestamps
   vs launcher manifests vs the reaper's window). Do not guess.
4. Self-protection (two landings, per CLAUDE.md): make the mover REFUSE to write a MOVED certificate
   unless the destination has been re-read and hashed to the source sha (size > stub size, not a `._`
   file), and add a preflight gate (claim a catalog number via `tools/claim_catalog_number.py`) that
   scans MOVED/MOVE_LOG rows for destinations that are stubs or missing. Tests for both.
5. Memo `.omx/research/ddm_vr8_moved_payloads_exfat_stub_audit_20260910.md` with the census table,
   root cause with evidence, every recovered sha, and a `# FORMALIZATION_PENDING:` line if no law is
   registered; serializer commit (two review passes per .py; no co-author trailer; tags
   `[no-triality] [p0-ledger-ok]`).

## Boundaries (binding)
- NEVER delete or overwrite any file, stub included, before the recovery row is written and the new
  destination sha verified; keep stubs as evidence (rename to `*.stub_evidence` only after certification).
- Host-quiet windows: MAIN runs decode-timing windows during this arm's life. Hashing and copying at
  ≥ 25 % CPU for minutes REFUSES those windows. Before any hashing/copy/re-render, wait for the file
  `/Volumes/VertigoDataTier/pact/MAIN_GO_CUSTODY_AUDIT` to exist (background until-loop, never a
  foreground sleep); reads, listing, and small JSON work are fine at any time.
- No scorer runs, no Modal, no edits to `upstream/`, the PR tree, or any sealed candidate tree.
- The rp1 round-2 store `/Volumes/VertigoDataTier/pact/ddm_rp1_round2/` and the sj1 sealed candidate
  `/Volumes/VertigoDataTier/pact/ddm_sj1_compose39_price/candidate/` are read-only.

## OPTIMAL FORM
- Reference form: vr3/vr5/vr7 certified-reclaim ledgers (rows with sha + bytes + rebuild path) and the
  landed MOVED.json helper; this arm audits and corrects, it does not invent a new ledger format.
- Scope: both SSDs, all certificates since 2026-09-09 (earlier ones listed but hashed only if time allows;
  report the denominator either way — vacuity is not a pass).
- Provenance pins: rp1's memo sha, mv1 landing 0e8e2e420, vr7 memo, the MOVE logs' shas.

## Prior negatives accounted (operator 2026-08-15)
- eb1/eb2 bundle landings matched `._` stubs as files (same class, different surface).
- vr5 apply refused 17 rows on self-reference before fixing; vg1's cwd census was malformed — read those
  memos before trusting a reclaim gate.
- rp1 recovered by re-render; the recovery must be CERTIFIED (sha equal to the recorded one), not assumed.

Checkpoint as `ddm_vr8` every ~10 tool uses. Final message: census counts (rows / false / recovered /
bytes), root cause with the actor named, the gate number, commit sha, every boundary, and the frontier
line `composition S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600] (move 40)` (or the new
pointer if MAIN has moved it by then — read `.omx/state/main_hot_state.md` POINTER_LINE).
