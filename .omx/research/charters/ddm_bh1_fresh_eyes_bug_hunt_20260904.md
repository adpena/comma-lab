# ddm_bh1 — fresh-eyes bug hunt over the surfaces that produced this wave's closures (operator: "likely bugs in a lot of places")

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm (report-first; fix only where a fix is < 30 min and tested) ·
Spawned by MAIN 2026-09-04 · Cost: $0

## The standing law this executes
CLAUDE.md "Confound self-protection": periodically spawn fresh-eyes hunters over orthogonal surfaces to find defaults
that are HARMFUL × SILENT × MEASUREMENT-CORRUPTING before they poison verdicts. This wave already found: the burn's stage
entry restarting τ/λ/EMA (MAIN, 16:10Z — the "cold transition" was a mislabel), the sealed-pin path mismatch, the τ-band
validation hole, the AA lattice misregistration, the silent centerline truncation, the dead pose head (2,014 B), the
`ratified`⊂`stratified` gate token, the venv-symlink identity, the n96 constants live in harnesses. The operator's
expectation is that more remain. Hunt them.

## Surfaces (orthogonal; take each with fresh eyes, verify at source, report with path:line + a reproducing check)
1. **Born trainer stage entry and objective** (`experiments/ddm_qbt1_qbflow_trainer.py`, `ddm_qbr1_born_fairform_burn_prep.py`):
   what else re-initialises at entry (LR warmup? gradient clipping? loss weights? the pose term's weight; `realized_weight`
   100 vs r10's); does `build_initial_state` load r10's EMA shadow into the LIVE weights AND the shadow (or only one)?;
   is the 16-pair chunk the same pairing r10 used; does the dual ascent read the realized error from the live or the
   shadow forward; the STE in `roundtrip_to_camera_uint8_ste` (gradient path sane?); `tau_for_step` off-by-one at step 0.
2. **Milestone evaluator** (`_evaluate_milestone` :600-660, `ema_scope` :432): shadow vs live; HT weights vs the burn's
   `selection_weights`; float16 logits retention (sd1's caveat) — is the argmax retained from float32?; `d_pose_hat`
   composition (PoseNet on rendered pairs vs GT pairs — same preprocessing as the contest? `resize→rgb_to_yuv6` order).
3. **ng2's cap and ng3's τ band wiring**: does the cap's gradient reach the parameters (the 1.25% number — is that a bug in
   the STE area or the intended magnitude?); does ng3's band actually take effect at step 1 (its smoke said so — re-verify
   from the live history's `tau` column of the running cell).
4. **Closure arithmetic**: recompute md1's persistent fraction, lb1's break-even precision, pr1's k_post and the −1.032e-4
   selector projection, and gm1's waste shares from their stored JSONs INDEPENDENTLY (own code, no import of their
   instruments); report every mismatch > 0.1%.
5. **Fire/seal apparatus**: `experiments/ddm_reseal_pins_inside_sealed_tree.py`, the three fire scripts under
   `/Volumes/APDataStore/pact/ddm_ng*/fire/`, claims, done-receipt naming, the memory-guard arithmetic (units, inactive pages
   as reclaimable — is that right on macOS?).
6. **GT lineage and population**: which live surfaces still read `gt_n600.npz` (PyAV) as authority; n32 selection vs
   `selection_weights` (HT) — any consumer that mixes n32 rows into n600 claims.

## Deliver
Memo `.omx/research/ddm_bh1_fresh_eyes_bug_hunt_20260904.md`: a table (surface · finding · path:line · severity
[measurement-corrupting / verdict-changing / hygiene] · reproducing check · fix-or-charter), fixes landed for the cheap
ones (tests + two review passes), and — for every verdict-changing finding — WHICH memo/closure it invalidates and what
must be re-measured. Final message → `.omx/research/arm_final_messages/ddm_bh1_final_<utc>.md`, committed; LAST action
`touch .omx/tmp/codex_runs/ddm_bh1.done`.

## Constraints
- $0 CPU; ng2 + ng3 LIVE on the Metal, fs1/gv1/ng4 on the CPU — never touch their custody/claims; read-only on run dirs;
  `upstream/` and `submissions/semantic_joint_ctxmix/` read-only; no /tmp paths. Commits ONLY via
  `tools/subagent_commit_serializer.py --message … --files … --expected-content-sha256 <file>=<post-edit sha>`; tags
  `[no-triality] [p0-ledger-ok]`; NO co-author trailer; any .py: tests + `tools/review_tracker.py mark-file` twice; never
  REVIEW_GATE_OVERRIDE on .py. EQUATIONS-LEG LAW for any memo stating a measured finding. Reference commit `820db413ea1186ec525629875c7805389d0e9f0c`.
