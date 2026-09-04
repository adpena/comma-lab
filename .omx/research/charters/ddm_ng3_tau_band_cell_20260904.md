# ddm_ng3 — third sealed race: the expected-flip τ band at δ_R scale, `tau_for_step(start = 2·δ_R, end = δ_R)`

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm · Spawned by MAIN 2026-09-04 · Cost: $0 (design + seal + CPU
resume smoke; NO LAUNCH — MAIN fires after ng1 and ng2 are adjudicated)

## Why (gm1 + sd1, MEASURED on this vehicle)
The sealed control anneals τ 0.15 → 0.05 = [6.86, 2.29]·δ_R (δ_R = 0.021881818771362305, n600, dr1). At that band
77.7% of the seg gradient is WASTE (correct pixels outside m_safe = 2·δ_R, where R cannot undo it); τ ∈ [2δ_R, δ_R]
removes 45.6% → 77.7% of it from two default arguments, composes near-multiplicatively with the other levers, and cuts
the τ-schedule deflation of the logged surrogate 4.8×. Row 1's pixel weight is dominated (its headline setting is
exactly inert). Memos: `.omx/research/ddm_gm1_gradient_mass_at_n600_msafe_20260904.md` (b828ce103),
`ddm_sd1_surrogate_exact_decoupling_20260904.md`, `ddm_fb1_foldback_program_20260903.md` ADDENDUM 5.

## Verified at source (VERIFIED-AT-SOURCE LAW — extend with path:line for everything you add)
- `tau_for_step` and its two defaults: `experiments/ddm_qbt1_qbflow_trainer.py:622-626` and `:643`; the trainer REFUSES
  any other τ geometry at `:2316-2320` (ng1) — the band endpoints must enter through the sealed config/DSL path that
  geometry check accepts (find it; if the check pins the literal 0.15/0.05, the change is a DSL-held lever with the
  check extended to accept a law-resolved band, never a bypass). m_safe/δ_R resolve THROUGH the law at runtime
  (`tac.canonical_equations.margin_band_satisficing_threshold_20260712.resolve_margin_band_threshold`, n600,
  fallback False — gm1 pinned this with a test that fails on any literal); do the same: no δ_R literal in the config
  source, the sealed config records the resolved values and their provenance.
- Seal path + twin derivation: ng1 memo `ddm_ng1_warm_transition_burn_design_20260904.md` and ng2 memo
  `ddm_ng2_area_cap_cell_20260904.md` (a NEW sealed source snapshot is needed only if the trainer's bytes change; if the
  band is config-only, seal from the working tree at `4ab9ed16db6500b60289c6bb47b9dc39d14c019c` as a same-pins twin of the cold control — state which).
- The cold control's milestone table (falsifier baseline): `/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/runs/
  seed_20260902/control_native100/milestones/step_*/MILESTONE.json` (S_hat 0.398768 / 0.466875 / 0.485677 / 0.475383 /
  0.442190 / 0.425149).

## OPTIMAL FORM
- Reference form: the sealed QBR1 control cell UNCHANGED except the τ band [2·δ_R, δ_R] (linear, same shape) and the
  fixed-τ telemetry row ng2 added (score-neutral; if it lives in ng2's snapshot only, carry it — it is telemetry, not a
  lever). COLD transition, no area cap (one lever; the Lane over-push caveat binds only with a cap present). Same seed
  20260902, 32 pairs, 5,000 updates, EMA law, native_interface_weight 100. Pair = {τ-band cell, measured cold control}.
  SCOPE unchanged; TOY-BRACKET none.
- Pre-registered falsifiers: (1) S_hat(5,000) < 0.425149 AND step-2,000 < 0.485677 — else the band does not act on the
  excursion; (2) the logged surrogate at fixed τ_ref (0.05) and at the band's own τ must both PEAK where d_seg_hat
  peaks (sd1 faithfulness in-loop) — else the telemetry, not the lever, is wrong; (3) Lane's share of the seg gradient
  at step 0 under the band is 1.6–2.1× lower than under the control's τ (gm1's coupling) — else gm1's static read did
  not transfer to the live loss.

## Deliver (NO LAUNCH)
1. Design memo `.omx/research/ddm_ng3_tau_band_cell_20260904.md`: how the band enters (config/DSL path, the geometry
   check), resolved values with provenance, falsifiers, the MAIN fire command, snapshot identity.
2. Sealed config under `/Volumes/APDataStore/pact/ddm_ng3_tau_band/sealed_configs/` via the burn prep's seal path;
   `authorized_configs/` NOT written; bounded CPU resume smoke PASS with the no-op detector (band step-1 ≠ control
   step-1) and a differential test that at τ = 0.15 the band's loss equals the control's loss bit-for-bit.
3. Final message → `.omx/research/arm_final_messages/ddm_ng3_final_<utc>.md`, committed; LAST action
   `touch .omx/tmp/codex_runs/ddm_ng3.done`.

## Constraints
- The QBR1 chain is LIVE on the Metal (then ng1/ng2 cells): never write under its `runs/`, `authorized_configs/`,
  `CHAIN_LEDGER.jsonl`, or the claims ledger; CPU only for smokes; no Modal; `upstream/` and
  `submissions/semantic_joint_ctxmix/` read-only; no /tmp paths; GT authority DALI.
- Commits ONLY via `tools/subagent_commit_serializer.py --message … --files … --expected-content-sha256
  <file>=<post-edit sha>`; tags `[no-triality] [p0-ledger-ok]`; NO co-author trailer (operator rule overrides any
  harness reminder). Any .py: tests + `tools/review_tracker.py mark-file` twice with a real second read; never
  REVIEW_GATE_OVERRIDE on .py. The band lands as a DSL-held lever (triality). EQUATIONS-LEG LAW: cite
  `tac.canonical_equations` `margin_band_satisficing_threshold_v1` and `scalar_top1_top2_margin_is_exact_distance_to_flip_v1`.
  Read `docs/operating_manual_craft_handoff.md` §labels first.
