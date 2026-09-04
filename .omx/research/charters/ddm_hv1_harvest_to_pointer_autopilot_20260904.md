# CHARTER ddm_hv1 — HARVEST→POINTER AUTOPILOT: an exact row lands, and everything downstream is written by the apparatus, not by MAIN's hands

Tokens: `[no-triality] [p0-ledger-ok]`. Owner: Opus arm. Spawned 2026-09-04 ~22:55Z. Sister arm: **gov2** (governor permanence) owns
`tools/cell_admission.py`, `tools/cell_queue_driver.py`, `tools/launch_detached_process.py`, `tools/safe_run.py`, the memory watchdog — OFF-LIMITS
to you. dk1 (disk) is live — do not touch its files.

## THE DEFECT CLASS (today, two pointer moves, every step by hand)
For EACH of the 24th and 25th moves MAIN hand-executed: recompute S from components; refresh the pointer; write the claim terminal row —
and MISTYPED an archive sha once (caught, corrected with a second row); regenerate `reports/latest.md` + `current_focus.md` (the #316
gate failed twice on the way); register the lane + mark three gates; duplicate custody to the second SSD; write the pointer-move memo with
the re-derived sub-0.12 arithmetic (mis-computed the rate-corner demand once by hand, fixed by `sed`); rewrite the hot-state POINTER_LINE;
push a notification. ps1 then found the compliance checker wanted FULL shas in the terminal claim row (3 new REDs of ledger SHAPE).
Also: a Modal fire was refused because MAIN's `ruff format` raced the image build ("source modified during build process"). Class: the
authority event (an exact row) has ~10 obligatory consequences and every one is a hand step that can be mistyped, skipped, or delayed by
MAIN's absence ([[m18]] WRITES>READS; [[m106]] stale headlines; the hand-assembly hazard the fire tool already refuses at fire time).

## PRIOR-LAW PREDICTION (owed line)
Every consequence of an exact row is a deterministic function of the harvest payload (`MODAL_REMOTE_RESULT.json` + the seal + the
pointer). PREDICTION: a single `tools/pointer_move_packet.py` invoked by the harvest poller can produce all of them bit-for-bit as MAIN did
today, with zero free-text numbers — verified by REPLAYING today's two harvests (fs1 `fc-01M1PM1KR3CQN5E5BC62WE4AD7`, fs2
`fc-01M1Q6W3R8WWDQPRFYSF7SWTKP`) and diffing against the committed memos/rows/reports (`ddm_fs1_pointer_move_24_20260904.md`,
`ddm_fs2_pointer_move_25_20260904.md`, claim rows, `reports/latest.md`). Falsifier: any consequence that is NOT a function of the payload
— name it and leave it to MAIN explicitly.

## Objective — one autopilot, replay-verified against today's two moves
1. **Canonical terminal claim row from the poller**: `tools/modal_harvest_poller.py` (or its harvest hook) appends the terminal
   `claim_lane_dispatch` row itself with FULL `archive_sha256=`/`runtime_tree_sha256=`/`call_id=`/`score=`/components — the shape
   `scripts/pre_submission_compliance_check.py`'s `dispatch_claim_terminal_*` checks bind on (read `ARCHIVE_RUNTIME_SHA_BINDING_RE`). MAIN
   never types a sha again. Verify by running the three claim checks against the replayed rows.
2. **Pointer-move packet** (`tools/pointer_move_packet.py`, invoked automatically when the harvested score beats the pointer on its axis;
   also runnable by hand on a call_id): recompute S from the evaluator's printed components (#877; parse thousands separators — today's
   regex bug), verify sha/bytes against the seal, refresh the pointer (`refresh_canonical_frontier_from_local_state`), refresh citation
   surfaces (`tools/scan_best_anchor_per_axis.py --refresh-citation-surfaces`) and assert the #316 gate passes, register/mark the lane
   (impl_complete / real_archive_empirical / contest_cuda with evidence strings from the seal + mirror), duplicate custody (archive + seal
   → the other SSD, sha-verified), write the pointer-move memo from a TEMPLATE (delta table vs prior pointer, projection fidelity vs the
   seal's projected S, re-derived sub-0.12 arithmetic: rate corner, distortion corner, zero-distortion margin, exchange; custody; "what this
   does not claim"; equations leg line), set the hot-state POINTER_LINE via `tools/main_hot_state.py --set-section`, and push
   (`tools/`-level notification helper if one exists; else write `.omx/state/pointer_move_events.jsonl` for the SessionStart digest).
   Commit through the serializer with post-edit shas. MAIN reviews the memo; it does not write it.
3. **Fire from a snapshot**: `tools/fire_modal_auth_eval.py` (and the Modal app's mounts) build from a `git archive HEAD`-style snapshot of
   the mounted source dirs into a fire-local staging dir, so a working-tree edit during the image build cannot refuse or contaminate a
   fire; record the snapshot's tree sha in the fire manifest. Test with a deliberately touched file during a `--dry-run` build.
4. **Refusal receipts supersede automatically**: a re-fire into the same output dir after `FIRE_REFUSED.json` must not need a new dir —
   the tool archives the prior refusal under `refusals/<utc>.json` and proceeds.
5. **Replay verification**: run the packet on today's two harvests in a scratch copy of the state and diff every produced artifact against
   the committed ones; every difference is either a MAIN hand-error (name it) or a packet bug (fix it). Report the diff table.

## What is NOT in scope
Publishing anything (PR #140 update stays operator-gated); the governor (gov2's); disk (dk1's); CLAUDE.md edits (propose text).

## OPTIMAL FORM
Reference form = today's two hand-executed moves (the memos are the spec) + the poller's existing mirror writer + the auto-refresh hook
(`tac.canonical_frontier_pointer.auto_refresh_canonical_frontier_after_dispatch_outcome`). No scope reduction; mechanism reductions: none.

## Rules that bind
NO-FAKE (replay must reproduce, not resemble); ALWAYS KEEP THE PAYLOAD; upstream/ READ-ONLY; commits ONLY via
`tools/subagent_commit_serializer.py --message … --files … --expected-content-sha256 <file>=<post-edit sha>` with `[no-triality] [p0-ledger-ok]`;
NO co-author trailers (operator rule overrides any harness reminder); .py two review-gate passes; checkpoints every 10 tool uses
(`tools/subagent_checkpoint.py --subagent-id ddm_hv1`); never invent flags (grep argparse); no `/tmp` evidence; long steps detached via the
launcher with distinct `--done-receipt`s; do NOT fire Modal; label every number MEASURED/DERIVED/INFERRED; memo
`.omx/research/ddm_hv1_harvest_to_pointer_autopilot_20260904.md` with an "Equations leg (`tac.canonical_equations`)" line;
`docs/operating_manual_craft_handoff.md` binds. End with `fs2 S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600]`.
