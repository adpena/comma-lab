# ddm_rp1 round 2 — respawn charter (MAIN, 2026-09-10 ~04:40Z)

**Why a respawn.** The rp1 Opus agent (pid 37594) died with the session context after
launching its five n600 K=192 sizing shards. All five shards FINISHED rc=0
(`/Volumes/VertigoDataTier/pact/ddm_rp1_round2/logs/n600_s{0..4}/`, elapsed 17,407–17,792 s).
The chain from the checkpoint's `next_action` is owed: carrier re-solve → frame-0 8-mode
INSIDE the admission → twin price vs move 40's tail → seal → MAIN fires → packet move 42 iff
exact S < 0.13763861019288715.

**Resume surface (read first, in this order).**
1. `tools/subagent_checkpoint.py read --subagent-id ddm_rp1` (step 20, in_progress).
2. `.omx/research/ddm_rp1_round2_move40_20260910.md` (the arm's own working memo).
3. `/Volumes/VertigoDataTier/pact/ddm_rp1_round2/ROUND2_BASE.json` + `n600/shard*/` outputs +
   `sizing/` + `rank/`; the round-1 packet `.omx/research/ddm_frontier_pointer_move_39_20260910.md`
   and move 40's packet `..._move_40_20260910.md` (what composition already banked).
4. Laws banked since the arm died (memory files, read them):
   `pose_resolve_is_mandatory_after_every_field_change_20260910` (admit ONLY on the RESOLVED
   pose; the full pass-5 field was stale +2.48e-2 → resolved +1.33e-4 and STILL failed),
   `composition_of_disjoint_token_edits_is_subadditive_on_seg_by_pair_overlap_20260910`,
   `first_order_token_price_is_a_ranking_never_a_charge_tail_flag_mass_untouchable_20260909`,
   `validator_contract_no_producer_can_satisfy_is_a_forever_refusal_test_the_producer_on_the_pass_path_20260910`.
   Operator verbatim this session: "Remember pose re solve" · "And frame 0" — frame 0 is a
   REPAIR of edit-broken pairs inside the admission, never a standalone door.

**Seal contract (changed since the arm died).** `tools/make_candidate_seal.py` now REQUIRES a
decode_wall_clock leg (dwc1, 86961e487). This candidate keeps move 40's receiver byte-identical
except `archive.zip` and the two pin literals in `inflate.py` (the stage step asserts exactly
that), so it INHERITS move 40's measured leg:
`--inherit-decode-wall-clock /Volumes/VertigoDataTier/pact/ddm_sj1_compose39_price/SEAL_ddm_sj1_compose39_rp1_union_contest_cuda.json.decode_wall_clock.json`
(MAIN writes that sidecar from the quiesced move 40 timing; if it is absent when you reach the
seal, STOP at the seal and report — do not invent a timing). Any other receiver change forfeits
inheritance and needs its own quiesced timing (`tools/quiesced_decode_timing.py`).

**Host-quiet protocol (binding).** MAIN runs quiesced decode-timing windows on this host; a
single process ≥ 25 % CPU outside the window's own tree refuses the whole 25-minute measurement.
Orientation (reads, small scripts) is fine at any time. Do NOT start any n600 scorer pass, encode,
parse-back, or multi-thread job until the GO file exists:
`/Volumes/VertigoDataTier/pact/ddm_rp1_round2/MAIN_GO_HEAVY_COMPUTE`. Wait for it with a
background until-loop (`run_in_background`), never a foreground sleep (rc=144 reaper law:
`harness_monitor_dies_rc144_use_bg_until_loop_20260903`). Launch every heavy step through
`tools/launch_detached_process.py --done-receipt <name>` (the guard hook blocks nohup/disown).

**Standing rules (unchanged).** CLAUDE.md non-negotiables; scores ONLY via `upstream/evaluate.py`
on exact bytes recomputed from components (MAIN fires Modal; you seal, you do not fire);
`upstream/` read-only; PR tree `submissions/semantic_joint_ctxmix/` untouched; commits ONLY via
`tools/subagent_commit_serializer.py --message "… [no-triality] [p0-ledger-ok]" --files …
--expected-content-sha256 <file>=<post-edit sha>`; NO co-author trailers; `.py` files need two
`tools/review_tracker.py mark-file … --status reviewed` passes; `REVIEW_GATE_OVERRIDE=1` only for
non-.py; never edit a script another live arm imports without telling MAIN (binding-drift law);
keep every payload (retained/ on the SSD tier, MOVED.json when moving bulk); checkpoint every
~10 tool uses with `tools/subagent_checkpoint.py --subagent-id ddm_rp1 …`.

**Deliverable.** A sealed candidate (seal JSON path + exact archive sha + projected net ΔS with
the seg/pose/rate decomposition on the RESOLVED pose, twin-priced against move 40's tail) or a
typed blocker. Final message: what you MEASURED, what you did NOT, every boundary, and the
own-vehicle frontier line: `composition S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600]
(move 40)` — a new number only if YOUR measurement moved it.
