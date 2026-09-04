# ddm_ng1 — next burn generation, first race: the TRANSITION itself (warm optimizer state, LR at the object's tail)

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm · Spawned by MAIN 2026-09-04 · Cost: $0 (design + seal +
resume smoke; NO LAUNCH — the Metal is held by the QBR1 chain until ~14:00Z; MAIN fires)

## Why (gestalt; the first-order fact of seed 1)
QBR1 seed 20260902, both cells (sealed configs differ ONLY in `objective.native_interface_weight` 100 vs 0): S_hat
0.398768 at step 0 → 0.4857/0.4868 at step 2,000 → 0.4251/0.4274 at 5,000 — an excursion that recovers with the LR
anneal and ENDS ABOVE the warm start, while `seg_expected_flip_realized` fell 0.00502 → 0.00325 (surrogate decoupled
from the exact argmax). ft1 showed the same shape on the shipped renderer (+31% → +6.5%). Three instances in two days
(memory `cold_transition_excursion_open_then_recover_three_instances_20260904`; registered sibling law
`muon_finisher_schedule_warmstart_and_lr_anneal_v1`). The discriminator QBR1 races is second-order (Δ 0.0023) against
this first-order degradation (+0.027). The accuracy half of sub-0.12 needs the born field at d_seg ≤ 1.3647e-4 BY
OPTIMIZATION; a stage that ends above its own warm start cannot be the vehicle. **The next generation's first race is
the transition.**

## Verified at source (VERIFIED-AT-SOURCE LAW — extend this list; every premise you add carries `path:line`)
- Sealed configs: `/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/sealed_configs/seed_20260902_*.json` —
  `learning_rate 0.0002`, `resume_from None`, `initial_state.path = …/initialized/qbr1_from_r10_ema_state.pt`
  (398,687 B; EMA state; verify whether it carries ANY optimizer state), `schedule.chunk_update_counts [2500, 2500]`,
  `checkpoint_every_steps 16`, `ema.value 0.99908 (ema_decay_run_geometry_v1, constant_decay, warmup False)`.
- Trainer (sealed copy the chain runs): `/Volumes/VertigoDataTier/pact/ddm_wc3_qbr1_ema_law_cure/sealed_source_106d0dd0_v2/
  experiments/ddm_qbt1_qbflow_trainer.py` (working-tree twin `experiments/ddm_qbt1_qbflow_trainer.py`, commit
  `d6009074ec75577ee2a37696c65064fbcdb61204`) — find at source: how the optimizer is CONSTRUCTED at stage start (AdamW? state loaded from where?),
  what `resume_from` restores (weights only vs optimizer/EMA/step), how the LR schedule is defined (constant 2e-4?
  cosine? where is the anneal that produces the recovery?), and what `--seg-…` fields the sealed config's
  `objective` block maps to. Burn prep: `experiments/ddm_qbr1_born_fairform_burn_prep.py` (seal/authorize/adjudicate).
- The warm start's provenance: r10 (`.omx/research/ddm_qbt2b_r10_third_doubling_stop_verdict_20260829.md`) — find r10's
  OWN terminal LR and whether an optimizer-state checkpoint of r10 exists on the SSD (its run dir); if it does, the
  warm cell resumes optimizer state from it; if not, derive the least-cold alternative (momentum re-warm from a short
  frozen-LR window; LR = r10's terminal LR, never 2e-4 transferred) and say which you chose and why.
- Milestone receipts to read: `/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/runs/seed_20260902/*/milestones/
  step_*/MILESTONE.json` (S_hat, d_seg_hat, d_pose_hat, archive_bytes_exact).

## OPTIMAL FORM
- Reference form: the sealed QBR1 cell machinery UNCHANGED (same trainer, same 32-pair selection, same seed 20260902,
  same 5,000 updates, same EMA law, same native_interface_weight 100 as the control) — the ONLY intervention is the
  transition: optimizer state carried (or least-cold re-warm), LR at the object's own tail with the SAME anneal
  shape. One lever. SCOPE unchanged. TOY-BRACKET: none.
- Pre-registered falsifier: the warm cell must end BELOW the warm start (S_hat < 0.398768 at step 5,000) AND sit
  below the cold control at every milestone (1k…5k); if it ends above 0.398768 the transition is NOT the cause and
  the schedule/objective itself is (then the next race is the LR magnitude alone, holding the transition warm).
- Second pre-registered read (free): the surrogate-vs-exact decoupling — report `seg_expected_flip_realized` beside
  `d_seg_hat` at every milestone; if the surrogate keeps falling while d_seg_hat rises in the WARM cell too, the
  loss itself is miscalibrated (vr1 rows 1/4 become the next race).

## Deliver (NO LAUNCH)
1. A design memo `.omx/research/ddm_ng1_warm_transition_burn_design_20260904.md` with the source-verified transition
   mechanism, the chosen warm form, the falsifiers, and the per-milestone read plan.
2. A SEALED cell config pair under `/Volumes/APDataStore/pact/ddm_ng1_warm_transition/sealed_configs/` produced by the
   same seal machinery the burn prep uses (never hand-typed JSON), `authorized_configs/` NOT written (MAIN authorizes
   at fire time with fresh claims), and a bounded resume smoke (the burn prep's own `bounded_resume_smoke`, ≤ 10 min
   CPU or a Metal window ONLY if the chain has released it — check `pgrep -f ddm_qbr1_cell_chain` first; if the chain
   is live, run the smoke on CPU with the trainer's CPU path and say so).
3. A one-line MAIN fire command in the memo (the launcher argv exactly as the chain would use it).
4. Final message → `.omx/research/arm_final_messages/ddm_ng1_final_<utc>.md`, committed. LAST action:
   `touch .omx/tmp/codex_runs/ddm_ng1.done`.

## Constraints
- Never write under the live burn's `runs/`, `authorized_configs/` or `CHAIN_LEDGER.jsonl`; never touch its claims.
  `upstream/` and `submissions/semantic_joint_ctxmix/` read-only. No /tmp paths. GT authority DALI
  (`gt_cache_dali.pt`); `gt_n600.npz` is PyAV (the burn's continuity frame only).
- Commits ONLY via `tools/subagent_commit_serializer.py --message … --files … --expected-content-sha256
  <file>=<post-edit sha>`; tags `[no-triality] [p0-ledger-ok]`; NO co-author trailer (operator rule overrides any
  harness reminder). Any .py: tests + `tools/review_tracker.py mark-file` twice with a real second read; never
  REVIEW_GATE_OVERRIDE on .py. EQUATIONS-LEG LAW: the memo cites `tac.canonical_equations`
  `muon_finisher_schedule_warmstart_and_lr_anneal_v1` (and appends the QBR1 seed-1 anchor to it via the helper if it
  fits its domain; otherwise a FORMALIZATION_PENDING line naming the law it would need). Read
  `docs/operating_manual_craft_handoff.md` §labels first.
