# ddm_mv2 — land mv1's held MOVED.json patch at the rp1 stage boundary (charter, MAIN 2026-09-10)

## Why now
mv1 (`ddm_mv1_moved_manifest_resolution_readers`, FINISHED 2026-09-10T04:00Z) produced a landing
patch that MAIN HELD because it edits scripts a live arm (rp1 round 1) imported
(`experiments/ddm_jg2_tail_reencode.py`, `experiments/ddm_rp1_rate_rank.py`,
`experiments/ddm_sj1_pass5_price.py`, `src/tac/candidate_seal.py`). rp1 round 1 is finished and
rp1 round 2 is respawned but GATED (no heavy compute until MAIN's GO file), so this is the stage
boundary the hold waited for. Land it BEFORE rp1's chain starts.

## Inputs (read first)
- `.omx/research/ddm_mv1_20260910/landing.patch` (21 files, +1229/−13) and
  `.omx/research/ddm_mv1_20260910/final_real_resolution.json`, `pytest.log`, `review_pass*.json`.
- The arm's final message: `.omx/research/arm_final_messages/ddm_mv1_moved_manifest_resolution_readers_20260910T040056Z.md`.
- `git apply --check` on current HEAD FAILS on exactly three hunks: `docs/meta_bug_class_catalog.md:675`
  (a catalog row was appended since), `.omx/state/next_catalog_number.txt` (the number moved on), and
  `.omx/research/ddm_mv1_moved_manifest_resolution_20260910.md` (already exists in the working tree —
  compare, do not clobber; if identical, skip that hunk). Everything else applies.
- Memory law: `landing_an_arm_bundle_guard_on_file_count_and_skip_exfat_dot_underscore_stubs_20260910`
  and `landing_a_gate_into_a_live_arms_script_breaks_its_checkpoint_binding_20260910`.

## Deliverable
1. Apply the patch onto HEAD with the three conflicting hunks resolved by hand: claim the catalog
   number through `tools/claim_catalog_number.py` (read its --help; never hand-edit
   `next_catalog_number.txt`), append the catalog row at the CURRENT end of the table in
   `docs/meta_bug_class_catalog.md` with the claimed number, and make `src/tac/preflight.py` +
   the new gate module reference that number. Keep the arm's code otherwise byte-identical; note
   every deviation in your final message.
2. Run the patch's tests (`src/tac/tests/test_artifact_moved.py` + the modules it touches:
   `src/tac/tests/test_candidate_seal.py`, `src/tac/tests/test_decode_wall_clock.py`,
   `src/tac/tests/test_decode_timing_concurrency.py`) and `ruff check` on every changed .py;
   also `.venv/bin/python -c "import tac.preflight"` and the gate's own strict run against the
   repo (live count must be 0 or the gate lands warn-only with the count recorded).
3. Two visible review passes per changed .py (`tools/review_tracker.py mark-file <f> --status
   reviewed`, twice), then ONE commit via `tools/subagent_commit_serializer.py --message "ddm_mv1
   landed from its patch at the rp1 stage boundary: MOVED.json helper + writer + wired readers +
   preflight gate #<N> … [no-triality] [p0-ledger-ok]" --files … --expected-content-sha256
   <file>=<post-edit sha>` for every file (zsh: use arrays). No co-author trailers. Gitignored
   files (`.omx/state/*`, `*.log`) are NOT committed — the catalog claim tool handles its own state.
4. `tools/codex_arm_queue.py mark --name ddm_mv1_moved_manifest_resolution_readers --status landed`.

## Boundaries
- Do NOT touch `upstream/`, the PR tree `submissions/semantic_joint_ctxmix/`, any
  `/Volumes/*/pact/ddm_rp1_round2/` or `ddm_rlc1_rule118_cure/` artifact, or the running rp1
  agent's files beyond what the patch already changes. Do not run anything heavy (n600 passes).
- If the patch's code no longer matches HEAD beyond those three hunks, STOP and report the exact
  hunks; do not improvise a rewrite of the arm's mechanism.
- Checkpoint: `tools/subagent_checkpoint.py --subagent-id ddm_mv2 …` every ~10 tool uses.

## Final message
Commit sha, the claimed catalog number, test counts (passed/failed), ruff result, every
deviation from the arm's patch, and the frontier line
`composition S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600] (move 40)` unchanged.
