# ddm_ng2 — second sealed race of the next burn generation: the one-sided AREA CAP (vr1 row 3) + free fixed-τ telemetry

Tokens: `[no-triality] [p0-ledger-ok]` · Owner: Opus arm · Spawned by MAIN 2026-09-04 · Cost: $0 (design + seal + CPU
resume smoke; NO LAUNCH — MAIN fires on the Metal after ng1's warm cell)

## Why (sd1, MEASURED)
The QBR1 excursion (S_hat 0.3988 → 0.4857 @2k → 0.4251 @5k) is **rare-class over-paint**: Lane predicted/GT area
1.0334 → 1.0929, Movable 1.0259 → 1.0580, both maximal at step 2,000, mass-conserving (+7.550e-4 rare / −7.550e-4
majority of the frame). qbt1's dual ascent is RECALL-ONLY (`experiments/ddm_qbt1_qbflow_trainer.py:593`, Lane werr ≤
0.12 / Movable ≤ 0.009) with no cap on over-paint — vr1 row 3's precondition. Row 3 covers 75.9–82.6% of the excursion
mass and acts on the mechanism. Separately, the loss's fall was the τ anneal (−40.54% schedule leg); at fixed τ the
surrogate is faithful — so the free telemetry row (log the surrogate at a fixed reference τ) ships in the same cell.

## Verified at source (VERIFIED-AT-SOURCE LAW — extend with path:line for everything you add)
- sd1 memo `.omx/research/ddm_sd1_surrogate_exact_decoupling_20260904.md` (per-class area table, the τ identity, the
  edge map) · ng1 memo `.omx/research/ddm_ng1_warm_transition_burn_design_20260904.md` (how a cell is derived as a
  same-pins twin of the sealed control by deep copy; the seal machinery; the MAIN fire command; commit d57e49b02).
- Trainer: `experiments/ddm_qbt1_qbflow_trainer.py` at `92cda3296a93f76a368354275131f70f0ea55be2` (working tree compiles again after the 4a7ae5ca0
  re-pin; the SEALED tree at `/Volumes/VertigoDataTier/pact/ddm_wc3_qbr1_ema_law_cure/sealed_source_106d0dd0_v2/` is
  what the chain runs — decide, and state, which tree ng2 seals from; a NEW loss term cannot run from the sealed tree
  unchanged, so this cell needs a NEW sealed source snapshot: follow the burn prep's own snapshot/seal path
  (`experiments/ddm_qbr1_born_fairform_burn_prep.py`), never an ad-hoc copy).
- Constraint set: `dual_ascent_margin_constraints` (:593), per-class block in `joint_objective` (:663-681),
  `derive_balanced_class_weights` bincount (:686-702) — the cap's λ_c = W_birth/(δ·A_GT_c) derives from THAT bincount;
  no hand-typed area constants. Chan-Vese law: equation `chan_vese_area_constraint_birth_balance_v1`, DSL lever
  `AreaConstraintBirth` (find both at source; use the DSL lever if it compiles to this trainer, else register the
  flag through the DSL first — never a hand-added trainer flag).
- Telemetry row 0: emit `seg_expected_flip_realized_tau_ref` at τ_ref = 0.05 beside the annealed value in
  `history.jsonl` every step (read-only, score-neutral, defaults ON per the "off is a tracked queue" law).

## OPTIMAL FORM
- Reference form: the sealed QBR1 control cell machinery UNCHANGED except (a) the one-sided area cap term
  `E = (λ_c/2)·relu(A_c − A_c^GT)²` on the rare classes (Lane, Movable; state whether Undrivable/Road/MyCar get one — the
  law says one-sided rare-class pressure, derive it) and (b) the fixed-τ telemetry (no gradient). Same seed 20260902,
  same 32 pairs, 5,000 updates, same EMA law, native_interface_weight 100, COLD transition (so the pair is
  {area-cap cell, measured cold control} and the lever is ONE; the warm×cap combination is a later cell only if both
  win — m164). SCOPE unchanged; TOY-BRACKET none.
- Pre-registered falsifiers: (1) area-cap cell S_hat(5,000) < cold control 0.425149 AND its step-2,000 milestone
  below the control's 0.485677 (the over-paint peak) — else the cap does not act on the mechanism; (2) Lane and Movable
  predicted/GT area at step 2,000 within 1.03 (the step-0 values) — else the cap is not binding; (3) the fixed-τ
  surrogate must peak where d_seg_hat peaks (sd1's faithfulness holds in-loop) — else the telemetry row is wrong, not
  the lever.

## Deliver (NO LAUNCH)
1. Design memo `.omx/research/ddm_ng2_area_cap_cell_20260904.md`: source-verified constraint set, the cap's derivation
   (λ_c numbers with their bincount receipt), the telemetry emission, falsifiers, the MAIN fire command, and the
   sealed-source snapshot's identity.
2. Sealed cell config under `/Volumes/APDataStore/pact/ddm_ng2_area_cap/sealed_configs/` via the burn prep's seal path;
   `authorized_configs/` NOT written; bounded CPU resume smoke PASS with the no-op detector (cap ON step-1 ≠ control
   step-1) and a differential test that the cap term is zero when A_c ≤ A_c^GT.
3. Final message → `.omx/research/arm_final_messages/ddm_ng2_final_<utc>.md`, committed; LAST action
   `touch .omx/tmp/codex_runs/ddm_ng2.done`.

## Constraints
- The QBR1 chain is LIVE on the Metal until ~14:00Z, then MAIN fires ng1's warm cell (~3 h): never write under the
  chain's `runs/`, `authorized_configs/`, `CHAIN_LEDGER.jsonl`, or the claims ledger; CPU only for smokes; no Modal.
  `upstream/` and `submissions/semantic_joint_ctxmix/` read-only; no /tmp paths; GT authority DALI.
- Commits ONLY via `tools/subagent_commit_serializer.py --message … --files … --expected-content-sha256
  <file>=<post-edit sha>`; tags `[no-triality] [p0-ledger-ok]`; NO co-author trailer (operator rule overrides any
  harness reminder). Any .py: tests + `tools/review_tracker.py mark-file` twice with a real second read; never
  REVIEW_GATE_OVERRIDE on .py. The lever lands as a DSL `Lever` factory (triality; `tools/triality_drift_detector.py`
  is a Stop-hook). EQUATIONS-LEG LAW: cite `tac.canonical_equations` `chan_vese_area_constraint_birth_balance_v1`
  and append sd1's over-paint anchor to it via the helper if it fits. Read `docs/operating_manual_craft_handoff.md`
  §labels first.
