# ddm_ng4 — the CONTINUOUS-OBJECTIVE cell: carry every annealed/adapted state r10 ended with (τ, duals, EMA law, batch geometry), not only the weights

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm · Spawned by MAIN 2026-09-04 · Cost: $0 to seal + CPU smoke; MAIN
fires on the Metal under the memory-guard admission (concurrent cells allowed, operator 09-04)

## Why (the bug-class finding, verified at source 2026-09-04 16:10Z)
The QBR1 stage entry restarted the OBJECTIVE while carrying only the weights: τ restarts at 0.15 (`ddm_qbr1_born_fairform_
burn_prep.py:233-234`; trainer `tau_for_step` :685) though r10 ended its 10,000-step anneal at ≈0.05 (`AUTHORIZED_N32_R10_
10020_20260829.json`: `expected_flip_tau_start 0.15, expected_flip_tau_end 0.05, margin_steps 10000`); the duals restart
from 0 (burn step 1 λ ≈ 7e-5); the EMA law changes (r10 0.99954/0.99910 effective → burn 0.99908 constant); the objective
schema differs. gm1 measured that at τ = 0.15, 85% of the seg gradient lands on already-correct pixels; md1 measured the
damage complete within 16 updates. ng1 carried the optimizer moments ALONE and still restarted the objective → lost. Every
cell so far (six QBR1 + ng1 + ng2 + ng3) inherits this restart; ng3 is the closest to continuous in τ (2δ_R = 0.044 ≈ 0.05)
but still resets λ and the EMA law. This cell is the missing CONTROL: "just keep training r10", with nothing restarted
but the seed's batch order — if the excursion vanishes, the whole "cold transition" family was an objective restart.

## Verified at source (VERIFIED-AT-SOURCE LAW — extend with path:line)
- r10 terminal state: `/Volumes/APDataStore/pact/ddm_qbflow_implicit_boundary_flow/qbt1_trainer/governed_n32_r10/`
  (`history_journal.jsonl` rows are `{"kind":"row","payload":{…}}`; the last payload carries `margin_constraint.lambdas`,
  `objective` and `ema_effective_decay`; `stage_03_joint_boundary_interior_birth/checkpoints/stage_03_end.pt` holds
  weights + EMA + AdamW state (ng1 verified) — read the terminal τ (derive from tau_for_step at step 10,020 if not logged),
  λ_Lane/λ_Movable, the EMA decay/warmup state, `chunk_pairs 16`, and the pair-order schedule.
- The burn machinery + seal path (ng1/ng2/ng3 memos), the re-root tool `experiments/ddm_reseal_pins_inside_sealed_tree.py`,
  the seal law memory `seal_validates_only_inside_the_tree_that_fires_it_20260904`, and the memory-guard admission (the ng3
  fire script's inline arithmetic; gv1 is landing the canonical function — use it if it has landed).
- The controls to read against: the cold control (0.398768 / 0.466875 / 0.485677 / 0.475383 / 0.442190 / 0.425149) and
  r10's own terminal d_seg on the n32 selection (its milestone/verdict receipts).

## OPTIMAL FORM
- Reference form: the sealed QBR1 cell UNCHANGED except that EVERY annealed/adapted objective state is CONTINUED from r10's
  terminal state: τ held at r10's terminal value (or continuing its schedule geometry — state which, and why), λ initialised
  to r10's terminal duals, the EMA law and warmup as r10's (its effective decay at the end), batch geometry as r10's
  (`chunk_pairs 16`, same selection). Optimizer: COLD (fresh AdamW) — one lever = the objective continuity; a warm-AND-
  continuous twin is the SECOND cell if the first wins (m164). Seed 20260902, 5,000 updates. If any of these cannot be
  carried through the sealed config/DSL path without a trainer change, land the change as a DSL-held lever in the working
  tree, snapshot a new sealed tree, re-root pins, and validate INSIDE it before declaring SEALED.
- Pre-registered falsifiers: (1) S_hat at 1k ≤ 0.398768 + 0.005 (no excursion) AND S_hat(5k) < 0.398768 — else the
  restart was not the (whole) cause; (2) the live-forward d_seg_hat at step 16 within 1.2× of step 0 (md1's instrument;
  the 16-update damage absent) — else the damage has another source (report the parameter-group displacement);
  (3) λ trajectory continuous (no re-warm from 0).

## Deliver (NO LAUNCH)
Design memo `.omx/research/ddm_ng4_continuous_objective_cell_20260904.md` with the r10 terminal state table (τ, λ, EMA,
batch), the exact carried values with provenance, the seal receipt, the CPU resume smoke (no-op detector: step-1 state ≠
the cold control's; differential: at τ = 0.15 and λ = 0 the loss equals the control's bit-for-bit), the re-rooted config
validated inside its sealed tree, and the MAIN fire command (memory-guard admission; distinct done-receipt
`ng4_continuous_DONE.json`). Final message → `.omx/research/arm_final_messages/ddm_ng4_final_<utc>.md`, committed; LAST
action `touch .omx/tmp/codex_runs/ddm_ng4.done`.

## Constraints
- $0; ng2 + ng3 are LIVE on the Metal and fs1/gv1 on the CPU — never touch their custody/claims; CPU only for smokes;
  no Modal. `upstream/` and `submissions/semantic_joint_ctxmix/` read-only; no /tmp paths; GT authority DALI. Commits ONLY
  via `tools/subagent_commit_serializer.py --message … --files … --expected-content-sha256 <file>=<post-edit sha>`; tags
  `[no-triality] [p0-ledger-ok]`; NO co-author trailer; any .py: tests + `tools/review_tracker.py mark-file` twice; never
  REVIEW_GATE_OVERRIDE on .py. EQUATIONS-LEG LAW: cite `tac.canonical_equations` `muon_finisher_schedule_warmstart_and_
  lr_anneal_v1` (this cell is its objective-side sibling) and `checkpoint_trajectory_error_partition_v1`. Read
  `docs/operating_manual_craft_handoff.md` §labels first. Reference commit `820db413ea1186ec525629875c7805389d0e9f0c`.
