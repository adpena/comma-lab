# Confound Hunt H3 — LEVER-EFFICACY (binding vs INERT) + CONTROL-VALIDITY on the LIVE C0 run

**Surface:** LEVER-EFFICACY + CONTROL-VALIDITY. **Run:** `experiments/results/levelset_n600_witness_20260715T095030Z/` (pid 72377, C0, family `v9_cgauge_ideal_mod19`). **State at hunt:** alive ~9.5 min, RSS 27 GB, epoch ~0 (`mem_probe before_v0_verdict`; NO epoch-loop `lever_engage` rows yet). **$0, report-only, no score claim, pointer 0.19108 UNMOVED.** Sibling surfaces (H1 liveness/guards/schedule, H2 measurement/verdict-authority, H4 config-drift/resume/loss-scale) NOT duplicated.

**Method:** every "fired" count in the activation ledger is SETUP-time registration, NOT proven-binding. Each zero-valued / staged argv smell was traced to the trainer code path + the run.log setup row to decide: binding-with-nonzero-effect / intentionally-open / intentionally-blind-window / SILENT-inert-confound.

**HEADLINE:** NO silent-inert lever confound found — all 5 investigated zero-value smells are EITHER binding, intentionally-open, or intentionally-blind-by-design. The ONE real confound is at the STACK level (#7): C0 bundles ~15 levers, many self-tagged "A/B owed", fired together — its verdict is load-bearing ONLY at stack granularity, never per-lever.

---

## RANKED DATA — [lever · ON-in-name? · actually-binding?(cite) · intentional-vs-silent · poison-scope · L1/L2/L3 fix]

### #7 — STACK-ATTRIBUTION CONFOUND (HIGHEST poison-scope) — REAL
- **lever:** the whole C0 stack (~15 seg/geometry levers ON simultaneously).
- **ON-in-name?** yes. **actually-binding?** individually UNMEASURED. Stage rows in run.log self-tag many as unmeasured: `seg_subpix_boundary` "A/B owed (needs GO)", `seg_margin_satisfice` "A/B owed", `seg_chroma_boundary` "A/B owed (needs GO)", `seg_temporal_screw` "A/B owed", `seg_subpix_edge_weight` "A/B owed", `dseg_aware_taper` "advisory; NON-PROMOTABLE", `seg_phase_advect` "advisory until byte-closed".
- **intentional-vs-silent:** the FULL-STACK config is intentional (C0 = the reference for one-rung Phase-2 ISO A/Bs), but per-lever attribution is SILENTLY absent.
- **poison-scope:** C0's d_seg cannot be attributed to any single constituent lever — the config-orphan / velocity-orphaning meta-bug (`[[config_orphan_confound_permanent_fix_lever_registry_20260706]]` / L27) lifted to STACK scale. C0 is a valid one-rung control for Horizon/StepNative, but "this stack reaches d_seg X" is NOT evidence any single lever moved it.
- **L3 fix:** C0's verdict is admissible ONLY at stack granularity; the campaign must eventually run the per-lever ablations the stage-rows already flag "A/B owed" before any constituent lever earns a binding verdict.

### #5 — weight_entropy_penalty λ=15.0 — BINDING, term_domination WATCH (defer depth → H4)
- **ON-in-name?** yes. **actually-binding?** YES — `train_levelset...py:6639-6644`: `if we_lambda > 0.0: ... terms_out["weight_entropy"] = we_lambda * _we_rate` added to total loss (we_lambda=15.0).
- **intentional-vs-silent:** intentional (rate-in-the-loss MDL term), but LARGE coefficient.
- **poison-scope:** if `terms_out["weight_entropy"]` is structurally >~40% of total loss it swamps every geometry lever's measurable contribution — the exact "dominating term makes lever contribution unmeasurable" concern. No `loss_terms` row exists yet (ep0), so magnitude unverified.
- **L1/H4 fix:** the L1 `term_domination` alarm (>40% of total) is the correct guard; **H4 owns loss-scale depth** — flag: verify weight_entropy fraction on the first `loss_terms` row. (`w-seg=100` dominance is the OBJECTIVE, NOT a confound; weight_entropy is the one regularizer that could rival it.)

### #4 — pose-carrier s_r=0.0 / pitch=0.0 — INTENTIONAL BLIND-WINDOW (H2-adjacent)
- **ON-in-name?** carrier yes (s_t=0.044 nonzero). **actually-binding for pose?** NO for ep 0–726 by design — `pose_finish_armed` (run.log): "pose-blind until d_seg converges (muon switch @726), then terminal joint pose-descent". s_r/pitch=0 = translation-dominant init; residual co-trains.
- **intentional-vs-silent:** intentional (matches L68 pose BANKED R1 dxi).
- **poison-scope:** any early-window (<726) pose reading is NON-load-bearing — not a lever verdict.
- **L3 fix:** pose-axis verdicts only admissible ≥726 (post pose-finish engage); early d_pose ↑ is EMA-shadow lag (L67/DAG), not a lever failure.

### #8 — "17 levers fired" overstates binding-NOW — TIMING confound (mild)
- **ON-in-name?** ledger says 17 fired. **actually-binding NOW?** at ep~0, ZERO epoch-loop `lever_engage "fired"` rows exist (`grep '"status": "fired"'` → empty); the count is SETUP registration only.
- **intentional-vs-silent:** intentional staging, but the count SILENTLY conflates registered-active with firing-now.
- **poison-scope:** armed-not-firing levers gated in the future — `seg_temporal_screw` @450, `seg_chroma_boundary` @450, `lane_render_band` @500, `seg_phase_advect`/`muon`/`pose_finish` @726, `polyak` @2546, `persistence` warmup @275 — have NOT actuated. A reader treating an early-epoch d_seg as "the full 17-lever stack" is wrong; half the stack hasn't fired.
- **L1 fix:** distinguish `state ∈ {registered, armed, fired}` per epoch window (the L31 ACTIVATION-dimension) so no verdict cites unfired levers.

### #1 — dseg_aware_taper scale=0.0 — CLEAN (auto-scale, NOT inert)
- **ON-in-name?** yes (#1 duty-to-measure, 78.9%). **actually-binding?** YES — `train_levelset...py:4085-4096`: `_dat_scale <= 0.0` ⇒ `saliency_from_margins(scale=None)` = **AUTO** (median-|margin| kernel width, `dseg_aware_fourier_taper.py:84`), NOT a no-op. run.log stage row: taper_min 0.9536 / max 1.0462 / mean 1.0 over 80 curvelet cols (±4.6% byte-neutral spectral reweight of `curv_feats_np`).
- **intentional-vs-silent:** intentional (scale=0.0 is the documented AUTO sentinel).
- **poison-scope:** none as a confound; effect is a MILD static ±4.6% basis reweight at setup (NON-PROMOTABLE; re-validate at convergence per its own note). CLEAN.

### #2/#3 — ladder movable/lane lambda-gate 0.0 — CLEAN (UNGATED = documented default)
- **ON-in-name?** yes. **actually-binding?** the LADDER radius-continuation over the amplify birth (amplify weight 1.0) runs; `lambda_gate==0 ⇒ always-open (=1)` (`ladder_homotopy.py:133` + argparse help `:14504` "0 (default) => UNGATED"). The gate multiplier is a constant 1; the ladder itself is NOT disabled.
- **intentional-vs-silent:** intentional documented default (dilation-GO sound independent of lane-share).
- **poison-scope:** none. CLEAN.

### #6 — seg_margin_satisfice "MASK-BY-STAGE at l7" — CLEAN (binding; stale note only)
- **ON-in-name?** yes. **actually-binding?** YES at ep0 — `ms_gate = {"on": ms_start <= 1}` (`:5591`, ms_start=0 ⇒ ON), term applied `terms_out["margin_satisfice"] = ms_w * ms_term` (`:6396`, ms_w=0.2). run.log: annulus_frac 0.0483, active.
- **intentional-vs-silent:** binding; the "MASK-BY-STAGE at l7" phrase is a NOTE-vs-CODE descriptive drift — `--seg-form-unify-tau` DISSOLVES the discrete l7 stage (`:2489-2498`, "l7-start-epoch is likewise inert under unify") and the satisfice gate keys on EPOCH (ms_start) not stage, so no l7-masking path exists; the term simply engages at ep0 as designed. Not inert.
- **poison-scope:** none (binding). Minor: refresh the stale l7-mask note.

### CONTROL-VALIDITY (Phase-2 treatments) — CLEAN
- **HorizonWeightedMargin:** ABSENT — no `horizon` token in argv; `hz_w=0.0 ⇒ gate never engages` (`:11181`). DEFER-tagged in `launch.sh` header.
- **StepNativeActivation:** ABSENT — `--activation hosc` (not step); no step-native token. DEFER-tagged in header.
- **verdict:** C0 is a CLEAN one-rung control for both named ISO A/Bs. No Phase-2 treatment is silently half-on in C0.

---

## L3 VERDICT-CLEARANCE precondition + positive-control sentinel

**Precondition (for any C0 verdict to be load-bearing):** every lever the reader believes active must be provably binding, and C0 must be a clean control for its A/Bs.
- **Satisfied per-lever:** the 5 zero-value smells are all binding (taper, satisfice, weight_entropy), intentionally-open (lambda-gate), or intentionally-blind-window (pose s_r/pitch). **NO silent-inert lever confound.**
- **NOT satisfied at stack granularity (#7):** ~15 levers fired together, individually unmeasured ("A/B owed") — so a C0 verdict is admissible ONLY as "this STACK reaches d_seg X", never "lever L moves d_seg".

**Positive-control sentinels EXIST (apparatus is NOT blind):**
1. **structured_init** — sky IoU 0.976, hood IoU 0.993, direct-argmax-disagree 0.00528 (run.log) ⇒ the seg-forward + R measurement path registers KNOWN structure correctly.
2. **pose_finish gate** — `canary_positive_fired=true`, `canary_negative_fired=false` ⇒ the gate registers its known-effect canary and correctly rejects the negative.

**term_domination (L1):** `w-seg=100` dominance is the objective (not a confound). The single regularizer that could rival it is **weight_entropy λ=15.0** — flagged for H4 to verify on the first `loss_terms` row.

---
_pointer 0.19108 UNMOVED · MEANS only · H3 report-only, no dispatch/train/score claim._
