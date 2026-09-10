# ddm_mv3 — land mv1's held MOVED.json patch with a CONTEXT-ONLY rebase (charter, MAIN 2026-09-10)

Successor of ddm_mv2 (FINISHED rc=0, correctly STOPPED: `.omx/research/ddm_mv2_20260910/BLOCKED.md`).
mv2 verified the patch sha and the 21-file set, found rp1's GO absent, and found FIVE hunks that
no longer apply at HEAD — three known to the mv2 charter plus two more that are pure context drift:

- `src/tac/candidate_seal.py:1413` — dwc1 (86961e487) inserted decode-timing validation beside it.
- `experiments/ddm_vr3_certified_raw_reclaim.py:19` — vr7 added imports beside it.

Both are inserted CONTEXT, not competing edits to the same lines. This charter grants what mv2's
STOP rule withheld: rebase context-only conflicts by hand and land.

## Inputs (read first)
- `.omx/research/ddm_mv2_land_mv1_moved_manifest_patch_charter_20260910.md` (the whole prior charter
  binds here unchanged except the STOP rule below) and mv2's final message
  `.omx/research/arm_final_messages/ddm_mv2_land_mv1_moved_manifest_patch_20260910T100414Z.md`.
- Patch `.omx/research/ddm_mv1_20260910/landing.patch` sha256
  `ea104296ba676338472bb884b0356bb88c44e1da122d2eb68e4d6e645d2e6c92`; the arm's base head
  `6020c867f82fe1e61cb7373da9d5ba1c5c9ce475`; fallback commit `1076c601ac26029c04546b066ca90d777927e104`.
- Read `git log --oneline 6020c867f..HEAD -- src/tac/candidate_seal.py experiments/ddm_vr3_certified_raw_reclaim.py`
  to see exactly what landed in between (dwc1's decode leg, vr7's reclaim changes, the quiesced
  timing instrument bf467026f) before touching either file.

## Revised STOP rule
- Context-only conflicts (the intervening change touches DIFFERENT lines; `git apply --3way` or a
  hand rebase keeps every line of BOTH sides) → rebase and continue. Record each such hunk in the
  final message with the surrounding intervening commit named.
- Semantic conflicts (the intervening change edits the SAME lines mv1 edits, or mv1's hunk calls a
  function whose signature changed) → STOP and report the exact lines; do not improvise.
- The three hunks mv2's charter already covered (catalog row, `next_catalog_number.txt`, the
  pre-existing memo) are handled as that charter says (claim the number via
  `tools/claim_catalog_number.py`; append the row at the table end; skip the memo if identical).

## OPTIMAL FORM
- Family reference form: mv1's patch as produced (sha above), 171 tests passing in its own run
  (`.omx/research/ddm_mv1_20260910/pytest.log`). Landed mechanism byte-identical; rebase deltas only.
- Scope-vs-mechanism deltas: rebase context and the catalog number only. A mechanism change is a
  TOY-BRACKET violation — STOP instead.
- Provenance pins: as above, plus the HEAD you land on (record `git rev-parse HEAD` before and after).
- Verification form: `src/tac/tests/test_artifact_moved.py`, `test_candidate_seal.py`,
  `test_decode_wall_clock.py`, `test_decode_timing_concurrency.py`, and the vr3 tests
  (`src/tac/tests/test_ddm_vr3*` if present, else `experiments/ddm_vr3_certified_raw_reclaim.py --help`
  must still run); `ruff check` on every changed .py; `import tac.preflight`; the new gate's strict run
  with its live count recorded (0, or warn-only with the count).

## Prior negatives accounted (operator 2026-08-15)
- mv2 itself: a STOP rule scoped to three hunks could not absorb two context-drift hunks — this charter
  scopes the rule by CONFLICT KIND, not by count.
- eb1/eb2 bundle-landing misses (guard on the 21-file count and the ExFAT `._` stubs), pm2's live-arm
  binding drift (land BEFORE `/Volumes/VertigoDataTier/pact/ddm_rp1_round2/MAIN_GO_HEAVY_COMPUTE`
  exists; STOP if it does), d7bf47655's blind commit-after-tests (serializer only on pytest rc 0),
  and mv1's own serializer refusal (read `final_serializer.stderr` first) — all as in the mv2 charter.

## Deliverable
One serializer commit (post-edit shas, zsh arrays, two review passes per .py, no co-author trailer,
message tagged `[no-triality] [p0-ledger-ok]`), then
`tools/codex_arm_queue.py mark --name ddm_mv1_moved_manifest_resolution_readers --status landed`.
Final message: commit sha, catalog number, every rebased hunk with its intervening commit, tests
passed/failed, ruff, and the frontier line
`composition S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600] (move 40)` unchanged.
Checkpoint as `ddm_mv3` every ~10 tool uses.
