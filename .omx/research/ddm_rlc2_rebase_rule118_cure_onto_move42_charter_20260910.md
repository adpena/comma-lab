# ddm_rlc2 — re-base rlc1's rule-118 cure onto pointer move 42 and seal it for its own T4 fire (charter, MAIN 2026-09-10)

## Why
rlc1 (landed ba0110e15; pr9 CLEAR-WITH-CONDITIONS; pr11) cured the receiver-code door: the video-
selected constants of the tc3 lane-boundary context map moved out of free code into a COUNTED 19-byte
rider, with fixed-point geometry (`runtime/rlc1_geometry.{c,py}`, `runtime/rlc1_mixer.py`, edits to
`inflate.py`, `inflate.sh`, `runtime/residual_archive.py`). Its candidate saved 60 B against move 40
(180,173 B) with byte-identical raw output. Two things changed since: (a) the pointer is now move 42
(rp1 round 2: S 0.1374765052591843 @ 180,238 B, archive sha f111ab42…, receiver = move 40's code,
tree `/Volumes/VertigoDataTier/pact/ddm_rp1_round2/candidate/candidate_runtime`), so the cure must be
re-applied to move 42's archive bytes; (b) ddm_pr11 ruled that a receiver that has never run on T4
takes its timing authority from its OWN cold contest-T4 decode (`mode: "t4_direct"`,
`tac.decode_wall_clock.build_t4_direct_leg`), not from local calibration — five local calibration
attempts were refused by host daemons and tool activity and that route is SUSPENDED.

## Read first
- rlc1's memo `.omx/research/ddm_rlc1_rule118_cure_20260910.md`, `FINAL_HANDOFF.json`,
  `PUBLIC_TWINS_AND_TIMING_BLOCK.json`, the counted-config accounting; pr9's review
  `.omx/research/ddm_pr9_second_family_check_rlc1_cure_20260910.md` (conditions 1–2: manifest refreshed by
  MAIN — `MANIFEST_REFRESH.json`; timing = your own T4 fire); pr11's t4_direct contract
  `.omx/research/ddm_pr11_adjudicate_timing_admission_on_daemon_bursts_20260910.md` §"Exact mode t4_direct".
- The move 42 packet `.omx/research/ddm_rp1_round2_rate_directed_predistortion_k192_20260910_pointer_move_42_20260910.md`
  and seal `/Volumes/VertigoDataTier/pact/ddm_rp1_round2/SEAL_ddm_rp1_round2_contest_cuda.json` (its
  `receiver_pins`, `retained_payload_paths`, `falsifiers`, `admit_bar` are the shape you reproduce).
- `tools/make_candidate_seal.py --help`: the seal REQUIRES a decode-wall-clock leg. Your candidate cannot
  inherit (receiver differs: b06e59a6… vs 6726fd77…) and has no T4 receipt yet. Read `src/tac/decode_wall_clock.py`
  and `src/tac/candidate_seal.py` for what a seal needs; if a pre-fire seal for a `t4_direct`-authority
  candidate is genuinely impossible under the current contract, STOP at that exact point with the exact
  refusal text and the smallest contract change that would allow "seal → MAIN fires → the fire's own receipt
  becomes the t4_direct leg → seal re-validated"; MAIN and the second family decide. Do not invent a waiver.
- Laws: `fitted_scalars_in_receiver_code_are_content_move_41_retracted_rule_118_20260910`,
  `receiver_change_needs_a_measured_decode_wall_clock_tc4_timed_out_at_1800s_on_t4_20260910`,
  `second_family_owns_the_admission_rule_freeze_before_run_t4_direct_outranks_local_calibration_20260910`,
  `a_flag_and_a_constant_can_disagree_silently_guard_both_objects_20260910`.

## Deliverable
1. A new candidate tree `/Volumes/VertigoDataTier/pact/ddm_rlc2_cure_on_move42/candidate_runtime/`: move 42's
   tree + rlc1's receiver diff (exactly the seven-path delta pr9 enumerated; rebuild the counted rider
   against move 42's archive; the archive = move 42's bytes with the rider, re-encoded by the real coder,
   priced by real encode). Refresh `MANIFEST.sha256` to the exact tree (49 rows; archive.zip and the manifest
   excluded, `LC_ALL=C sort`, generated from OUTSIDE the tree). Record the runtime digest.
2. Full n600 parse-back on the exact bytes (`[macOS-CPU advisory]`, cold) proving raw identity with move 42's
   retained raw (sha in the packet) — the cure must not change a pixel; the public smoke
   (`experiments/ddm_rlc1_smoke.py` pattern) rebound to the new digest; the offline literal census
   (pr9's acceptance table) re-run on the new tree.
3. Predicted row: bytes (exact), d_seg/d_pose identical to move 42's by construction (raw identical), S
   recomputed from components; expected net ≈ −60 B ≈ −4.0e-5 S vs move 42 (state the bar it clears).
4. The seal via `tools/make_candidate_seal.py` (falsifiers pre-registered; retained paths; admit bar),
   or the exact STOP above. MAIN fires; the fire's T4 receipt (`t4_direct`) is the timing authority and
   the exact score row; packet move 43 iff exact S < 0.1374765052591843.
5. Memo `.omx/research/ddm_rlc2_rebase_rule118_cure_onto_move42_20260910.md` + serializer commits (two review
   passes per .py; no co-author trailer; tags `[no-triality] [p0-ledger-ok]`); checkpoint as `ddm_rlc2`.

## Boundaries
- Never edit `upstream/`, the PR tree, or any sealed tree (`ddm_rp1_round2/candidate`, `ddm_sj1_compose39_price/candidate`);
  copy, never move. Keep every payload (retained/ on the SSD tier; hardlink byte-identical raws with a
  sha-verified certificate rather than duplicating 3.66 GB).
- No Modal; no timing windows (suspended); n600 parse-back is allowed (no window is open) but launch it
  through `tools/launch_detached_process.py --done-receipt …` and stay under the 8 GiB per-store cap.
- Receiver code must carry NO video-selected literal (rule 118): the rider is the only counted content;
  re-run pr9's literal census and quote it.

## OPTIMAL FORM
- Reference form: rlc1's landed cure (exact diff, counted rider, fixed-point geometry) on move 42's tree —
  a re-base, not a redesign; the encoder path is the real coder (twin encode), never a ledger sum.
- Provenance pins: rlc1 landing ba0110e15, pr9 memo sha, pr11 memo sha, move 42 packet d2803c214, archive
  sha f111ab4259c757409e791247d33978a714ceb1cd66e50c149e2e65fbf208756f.

## Prior negatives accounted (operator 2026-08-15)
- pr8: move 41's receiver carried video-selected constants — the census is the door check, quote it.
- pr9 condition 1: a stale manifest is a fire blocker — regenerate from outside the tree (MAIN's first
  regeneration hashed its own in-progress file).
- rp1 r2: a flag and a constant disagreed silently (wrong base field) — bind the base archive/tree by sha
  in every receipt and refuse on lineage mismatch.
- sj1's subset-writer silent revert: an edit spliced onto the wrong base reverts banked tokens — your
  archive must be move 42's bytes + rider, verified by raw identity.

Final message: candidate bytes + sha, raw identity proof, predicted S from components, the seal path (or the
exact STOP), every boundary, and the frontier line
`composition S 0.1374765052591843 @ 180,238 B [contest-CUDA T4 n600] (move 42)` unchanged.
