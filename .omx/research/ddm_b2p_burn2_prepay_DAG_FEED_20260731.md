---
schema: dag_feed.v1
feed_id: FEED-b2p-burn2-blocker-prepay
date_utc: 2026-07-30
arm: ddm_b2p
ledger_row: "#783"
pointer: "0.1910828242 [contest-CPU] UNMOVED"
score_claim: false
tokens: [p0-ledger-ok]
---

# FEED-b2p — burn-2 blocker prepay (during the QA24 burn window): QA75 UNBLOCKED · QA80 BUILT · QA81 TYPED BLOCKER

- **Pointer UNMOVED** (`0.1910828242 [contest-CPU]`). Build/materialization prepay only — NO scorer, NO
  Metal, NO trained byte. The QA24 burn (pid 68621) untouched. All `[macOS-CPU advisory]`, `score_claim=false`.
- **QA75 UNBLOCKED (sg1 §5 / lv1 materialization blocker CLEARED):** decoded the EXACT C1 solve archive
  (`ms2r_r3` 04_candidate, archive sha `e3d0581f…`, v10 receiver, scorer-free) → 600 per-pair frames
  `(2,874,1164,3)` uint8 (frame0 independently described) on SSD `ddm_b2p_20260731/qa75_solve_frames/`
  (3.4 GB, 600 per-pair sha manifest). Determinism **identical=True** (decode-twice spotcheck; integer-only
  realize). Typed loader `tac.witness_dsl.qa75_solve_frame_targets.SolveFrameTargets` landed + n4 tests.
  NO SegNet run (logit/margin distill precompute = post-burn). Guard: refuses residual archives.
- **QA80 PRODUCER BUILT:** `src/tac/boundary_math/margin_budget_field.py` — per-pixel flip-distance
  `d=|m|/‖Δw‖` from CACHED `gt_n{96,600}.npz['margins','lstars']` × MEASURED `HEAD_PAIR_NORMS` (recall-first:
  the `segnet_head_rank4_linear_flipdist_v1` law already existed). `exact_flip_distance_field` (needs
  runner-up) + `conservative_budget_field` (SOUND cache-only lower bound). Demo fields n96/n600 produced
  (q50 ~1.55) to `qa80_margin_budget/` + n4 tests. Exact burn-frame n600 field = named POST-BURN scorer step.
- **QA81 TYPED BLOCKER (not separable):** cb1 branch `2721704ab2` lands the carrier as `encode_static_class_mask_rule`
  IN the WIP `direct_description_carrier_compose.py`; all 3 cb1 code files import that symbol (absent on main).
  The file is dirty with an UNRELATED parallel-session change (`requires_pose6_transport`, disjoint regions).
  Landing needs the WIP file (forbidden) or clobbers uncommitted work; new-files-only would ImportError.
  SEQUENCING blocker → once the parallel session commits, `git cherry-pick 2721704ab2` applies clean.
  Unblock owner: MAIN / parallel session.
- **Hygiene:** venv gate0 = ALREADY canonical on main (`launch_tr1_run.venv_custody_gate0` prefers
  `tools/check_venv_src_custody.py`; verified None/clean). DSL stubs LEFT AS-IS (no flag became real; the
  fold-and-delete contract fires only at spec_tr1 lever landing = burn-2).
- **commit:** `a7968d9a62` (2 producers + 2 n4 test files; ruff-F clean; 12 tests pass; review-gate 2 passes).
- **verdict_scope:** QA75 INSTANCE-cleared · QA80 PRODUCER built (exact burn-frame field = post-burn, not a
  negative) · QA81 SEQUENCING blocker (not a cb1 falsification).
- **next (post-burn boundary):** burn-2 composes QA75 distill (frames ready) + QA80 budget (producer ready,
  runner-up scorer pass owed); QA81 fires after the parallel WIP commits.
