# Confound Hunt Synthesis — 2026-07-05 (operator: "that confound is poison")

Fresh-eyes adversarial confound hunters on the level-set witness trainer + measurement apparatus.
Signature hunted: **DEFAULT-HARMFUL × SILENT × MEASUREMENT-CORRUPTING**. REPORT-ONLY; central fix pass after all 6 return.
Pointer 0.19110 UNMOVED (means). Accumulate → verify adversarially → fix ALL (Layer-1 alarm + Layer-2 gate) → 3-clean recursive review → Layer-3 verdict-clearance CLAUDE.md non-negotiable.

---

## HUNTER 1 — GUARDS / ACTUATORS / DEFAULT-MODES (agent a305401a) — RETURNED

- **F1 CRITICAL — spike-guard `--spike-guard-mode` default=`legacy` (trainer:6433; loop 5805-5828, accepted-only median append 5894-5897).** The exemplar; confirmed BOTH v5 & v6 ran legacy (neither passed rollback), froze ep114/ep103, eikonal pinned high = frozen artifact. Cure `rollback` (SpikeGuardRollback 5097-5101) built+wired, default never flipped. FIX: default→`rollback` OR autoconfig injects it. GATE: refuse legacy for n600 w/o waiver + runtime `spike_deadlock_ALERT` on ≥K consecutive skips.
- **F2 HIGH — adaptive viscosity ε SILENTLY FLOOR-PINNED (_adaptive_visco_eps 1045-1061; called 5707 with η=lr, λ=0.05 → formula ~0.001 ≪ floor 0.3).** The "adaptive CFL ε" was a CONSTANT 0.3 every epoch (0 change-events; `grep -c eik_stabilizer_adaptive` v6 = 0; distinct visco_eps = {0.3}; visco_c_a DID vary 0.821→0.825 so formula live but output swallowed). **Doubly-confirms the FEED-06h retraction: "viscosity REFUTED" rests on ε being meaningful, but adaptive-ε was NEVER exercised — only constant ε=0.3 ran.** FIX (to actually test adaptive): lower floor toward ~1e-2 or rescale formula. GATE: `adaptive_eps_INERT` warn if clamped w/ 0 change-events >N ep.
- **F3 HIGH (latent) — `--verdict-pairs` default=24 = NON-n600 (trainer:6470; strided subsample 4027-4028).** The DEFAULT best-ckpt selection + ALL d_seg telemetry + closed-loop classifier run on 24/600 pairs — violates the n600 non-negotiable at the number that defines the goal. v6 correctly passed `--verdict-pairs 0` (this session clean) but it's the trap for the next launch. FIX: default→0. GATE: loud `[non-n600 verdict]` tag / launch-refuse when verdict_pairs!=0 at num_pairs=600.
- **F4 MED — SAME absorbing-median deadlock LIVE in BASE trainer (train_witness_realized_through_R_mlx.py:2271-2290, accepted-only append 2290, NO rollback cure).** Sibling surface unfixed (per "6-7× spread"). FIX: port rollback OR deprecate base loop. GATE: preflight scan for accepted-only `recent.append` pattern across BOTH.
- **F5 MED (latent) — `--closed-loop-control` eikonal_bump bumps UP the exploding term (1841, 4511); if fed frozen-verdict "plateau" it accelerates explosion.** v6 enabled but never fired (deadlocked first). FIX: refuse plateau classification when spike-skip fraction high. GATE: closed-loop cannot act while skip-fraction over threshold.
- **F6 MED — eikonal default-on + grad-clip=1.0 uniform: dominant volatile eik gradient hijacks the clip budget, starves seg (v6 eik ~85% of loss).** Matches the "weaken/gate the eikonal penalty" hypothesis; default-on + no-alarm let it masquerade as physics. FIX: cap/normalize eik term (Huber/per-term clip) vs seg. GATE: `loss_term_domination_ALERT` when a term > X× seg for >N ep.
- **CLEAN (honest negatives):** verdict try/except honest (verdict_async_failed, no fake-score); best-selection NO-FAKE-correct (finite+strictly-better, NaN never wins); ep_loss sum-not-mean gated off this session (fixed --curriculum).
- **BOTTOM LINE:** the "viscosity NO-GO" conclusion poisoned on TWO independent axes — F1 (legacy guard froze both runs) + F2 (adaptive-ε provably inert, constant 0.3). Re-run needs rollback mode AND a real (non-floor-pinned) ε or eik cap.

## HUNTER 6 — LOSS-TERM SCALE (agent af1f64bb) — RETURNED
All loss_terms verified POST-weight. 75 accum batches/ep. Score rewards eik at ZERO yet eik=86-91% of loss.
- **F1 CRITICAL — EIKONAL is 86-91% of loss (ep108: eik 235.7 / total 258.7 = 91.1%; seg 6.2%; pose 0.7%).** 86-91% of the gradient budget chases |∇φ|=1 (invisible to scorer) while seg (the d_seg proxy) is 6-10%. IS the guard-trip: eik grows 1.5→235 → batch_loss 19→259 vs median armed at CE level ~17-20 → 124 > 5×20 → 75/75 skip → freeze. Without eik, batch_loss≈17 < 5×median → NO freeze. GATE: alarm when any single reg term >40% of loss for N rows.
- **F2 CRITICAL — grad-clip 1.0 is a SHARED budget hijacked by the eik gradient; seg step starved ~300-900×.** clip_grad_norm scales the WHOLE vector by 1/gnorm; gnorm median 926 (v5) / 317 (v6) is eik-dominated → effective step 1e-3; seg-only gnorm would be O(10-15) → seg direction throttled 20-900×. clip preserves direction, crushes magnitude → d_seg moves ~0. FIX: per-term/per-group grad-clip (clip eik separately BEFORE sum). GATE: alarm when gnorm > 100× grad_clip sustained.
- **F4 HIGH (ROOT of F1) — VISCOUS-RESIDUAL UNIT/π-GROUP MISMATCH: the 0.05 weight was tuned for the LEGACY eik residual; the viscous form (`_eikonal_visco_mlx`, trainer:3305) has raw magnitude ~2490 (124.5/0.05).** Same flag name, silently different units across the form switch → "0.05" is meaningless → THE dimensional bug that produced the 86% domination. So the eik domination is NOT "physics needs it" — it's an uncalibrated unit bug. FIX: normalize each residual form to O(1) per-pixel-mean before weighting; startup-assert raw magnitude band. GATE: alarm on form-switch without weight recal.
- **F3 HIGH — eik-weight ANNEALS UP 0.05→0.10 at tau (wrong direction; should decay post-SDF).** GATE: refuse launch with eikonal_weight_end > eikonal_weight w/o waiver.
- **F7 HIGH (amplifier) — spike guard judges eik-DOMINATED total batch_loss, can never see seg progress; not re-armed at the viscosity onset (only at curriculum transitions 5409).** FIX: feed guard a reg-EXCLUDED (seg+pose) loss or per-term z-score. GATE: alarm skip-rate=100% for ≥2 ep.
- **F5 MED — `length` term DEAD (0.0002% of loss); F6 MED — pose 0.6% of loss vs concave √ score term (set w_pose from score marginal once eik de-dominated; pose HELD/open per memory).**
- **BOTTOM LINE:** the training objective is a de-facto eikonal-SDF regularizer with a small seg/pose passenger; every "what the witness is learning" conclusion from these runs is corrupted. Highest-leverage: knock eik to ≤10% of loss (recalibrate viscous-form weight + anneal down) + give guard/clip a budget separate from scored terms.

## HUNTER 5 — STAGE / SCHEDULE TRIGGERS (agent afc09bf9) — RETURNED
Headline: NO curriculum boundary is the freeze root (all at 275/400/450/726/1001, far from ep102/113). EXONERATES curriculum. Chain = eik runaway (physics) → legacy guard (mechanism) → freeze. But 4 genuine schedule-class confounds:
- **F3 HIGH — `--resume-allow-lever-drift` injects a STIFF term at FULL weight onto an opt state trained WITHOUT it.** ep100 ckpt trained w/o boundary_distance(0.2) AND w/o eikonal-viscosity; both attached full-weight at ep101 = silent loss-composition/level shift at the resume boundary → the added viscosity is exactly what runs away. rewarmup machinery (`--stage-transition-rewarmup-epochs 20`) EXISTS but keyed to curriculum switches, NOT resume-drift. FIX: ramp added terms via rewarmup on resume-drift; refuse full-weight stiff-term add on resume. GATE: loud row enumerating ckpt-cfg vs resume-argv term diffs.
- **F5 CRITICAL META — `--closed-loop-control` classifies the FROZEN run as "converging" (SMOKING GUN: v5 closed_loop ep125 & ep150 "converging" d_seg_slope=-0.00038 action none, BOTH after the ep113 freeze, same epochs as verdict ep_loss:0.0).** The system meant to CATCH the freeze CERTIFIED the dead run healthy — the meta-confound made concrete. FIX: gate every "converging/healthy" classification on liveness (ep_loss>0 AND skip_frac<thresh) else classify FROZEN+STOP. GATE: a "converging" verdict with ep_loss==0.0 is a hard assertion failure.
- **F4 MED (latent) — `--reorient-every 50` periodic basis reset does NOT re-treat the guard (no recent_losses.clear()); inert here (basis converged) but a future guard-trip when unconverged.** GATE: registry of loss-level-shifting events, each MUST pair a guard re-arm.
- **F6 LOW-MED — `--seed-anneal-epochs 101` withdraws the seed crutch exactly at resume start_epoch=101 (loss-composition shift in the freeze window; folds into F3).**

---

## CROSS-HUNTER CONVERGENCE (3 of 6 back) — the "viscosity NO-GO" verdict is poisoned on ≥5 INDEPENDENT axes
1. Legacy spike-guard froze both runs (H1-F1, H5-F1, H6-F7 — 3 hunters agree).
2. Adaptive-ε was INERT (constant 0.3, 0 change-events) — adaptive CFL-viscosity NEVER ran (H1-F2).
3. The eik domination that caused the runaway is a UNIT/π-group BUG (viscous residual weight never recalibrated across the form switch) — NOT "physics needs strong eik" (H6-F4). **Deepest root; reframes the whole eikonal narrative.**
4. grad-clip 1.0 hijack starved the seg step 300-900× — d_seg couldn't move regardless of the guard (H6-F2).
5. The closed-loop controller CERTIFIED the frozen run "converging" — the safety net was itself fooled (H1-F5, H5-F5). The meta-confound, live.
**⟹ FEED-06g/06h/06i "viscosity refuted" and every eikonal verdict this session are NON-LOAD-BEARING.** The v7 rollback-only run (235748Z, live) tests ONLY the guard fix — it will likely ride through the freeze but STILL be eik-dominated (86%) + seg-clip-starved, so a weak d_seg there is NOT a clean viscosity verdict either. The CLEAN test is the post-fix-all run (guard default + eik unit-recal + per-term clip + closed-loop liveness + verdict-pairs default + resume-drift rewarmup).

## HUNTER 3 — RESUME / STATE-CARRY (agent a070743d) — RETURNED
Compound poison LIVE in both runs. Launches used warm-start's COSMETIC side-effects (clear-spike-guard + allow-lever-drift) while keeping the POISON (stale moments), and used NEITHER --warm-start-weights-only NOR --spike-guard-mode rollback.
- **F1 CRITICAL — `--resume-clear-spike-guard` is a ONE-SHOT reset, NOT a cure, in legacy mode (trainer:4928).** Re-arms median once from first accepted batch; re-enters sustained spike → re-freezes in 1-13 ep. FIX: clear-spike-guard IMPLIES rollback mode, OR refuse legacy+clear as fail-closed.
- **F2 CRITICAL — STALE ep100 optimizer moments restored into a DRIFTED loss geometry (trainer:4780 restored_opt:true; optP__step=6837, 40 moment tensors).** Moments fit to ep100 surface, but resume argv changes the surface (adds bd 0.2, viscosity). Docstring literally names this class (4671). FIX: clear-spike-guard/allow-lever-drift IMPLY warm-start-weights-only, OR warn loudly when both palliative flags set but opt restored.
- **F3 HIGH — resume ALWAYS loads `live` into the model, never the clean EMA (trainer:4725); no `--resume-from-ema` flag.** A crash mid-spike writes diverging live weights; resume re-enters divergence; clean EMA in same file never used. FIX: add `--resume-model-from {live,ema}`, default ema for warm-start/re-treatment.
- **F4 MED-HIGH — new loss levers INVISIBLE to the drift guard (key-present-only check, trainer:569 `if key not in resume_cfg: continue`).** ep100 npz predates --boundary-distance-weight → the bd term added at resume can't be flagged even without allow-lever-drift. FIX: persist ckpt git-sha; flag engaged levers absent from ckpt cfg.
- **F5 MED — allow-lever-drift SILENCES material divergence (seed_anneal 300→101, lane_band 350→450), corrupting A/B labeled "continuation" (it's a drifted re-treatment).** FIX: stamp continuation=False/retreatment=True; require --retreatment-reason.
- **F6 MED-LOW — closed-loop pending verdict carries ep_loss=0.0 into the trend history (skews slope).** FIX: NaN-sentinel skipped-epoch ep_loss; drop non-finite from regression.
- **CLEAN:** RNG restore bit-faithful; atomic checkpoint writes (tmp+os.replace); deploy ckpt = EMA shadow; arch-drift guard fails closed.

## HUNTER 2 — MEASUREMENT INTEGRITY (agent ad9bc291) — RETURNED
"No wrong formula anywhere — every number is correctly computed. The confound: 6 telemetry/verdict/control surfaces keep reporting MOTION while the optimizer is FROZEN, and only a per-row `spike_skipped` bool distinguishes the two."
- **F1 HIGH (highest poison) — closed-loop reports "converging" on a FROZEN run (v5 ep125/150 "converging" while 4450 skips + ep_loss:0.0).** [CONVERGES w/ H1-F5, H5-F5]. FIX: gate _cl_decide on liveness FIRST.
- **F2 HIGH — `loss_terms` recomputed on FROZEN live weights, AMPLIFIED BY β-ANNEAL: hosc_beta re-anneals at epoch-top EVERY epoch regardless of skips (1.0→5.134), so the eik "creep" 121→127 during freeze is β-on-frozen-weights, NOT physics (trainer:5614/5624).** THE EXEMPLAR + its driver — the operator's original misread mechanically explained. FIX: when skip=true emit terms=null/frozen:true / weights_stepped:false; record hosc_beta/softmax_temp in the row.
- **F3 HIGH — spike-guard median gate on a non-updating window (the deadlock root at the control surface).** [CONVERGES w/ H1-F1]. FIX: auto re-anchor median on sustained all-skip.
- **F4 MED-HIGH — `ep_loss:0.0` is a SILENT frozen tell that reads as "converged to zero"; the 0.025 gold was sampled on frozen epochs.** FIX: log accepted/skipped_batches in verdict; never let ep_loss==0.0 pass without frozen_epoch:true.
- **F5 MED — verdict d_seg from FROZEN EMA shadow indistinguishable from a genuine plateau.** FIX: attach ema_updates_since_last_verdict.
- **F6 MED — adaptive-ε `c_a` measured on frozen live weights → the LIVE STATUS "|c_a|=0.82" arithmetic is confounded; re-measure on a live-stepping run before load-bearing.**
- **F7 LOW — `resolve_eval_device` passes `--eval-device` verbatim; MPS block is doc-only (decode_memory_tier.py:152).** FIX: hard-reject mps.
- **CLEAN (verified):** async-verdict race-free (main-thread deep-copy); verdict-batch chunking bit-identical (eval-mode BN running stats); verdict device = fp32 numpy CPU one-codepath (no MLX/MPS authority); sum_terms self-check honest.
- **HIGHEST-LEVERAGE SELF-PROTECT (H2's headline):** stamp a run-level LIVENESS signal (accepted-batch fraction) onto EVERY verdict/loss_terms/closed_loop/eik_stabilizer row → no reader or controller can mistake frozen for converging.

## HUNTER 4 — LEVER EFFICACY / NO-FAKE (agent a2cee71c) — RETURNED
- **#1 HIGH — `--eikonal-viscosity-adaptive` CONFIRMED INERT from the log (visco_eps:0.3 fixed while visco_c_a varies; eik_stabilizer_adaptive row NEVER fires).** Needs |c_a|≥80 to beat floor; measured O(1). [CONVERGES w/ H1-F2, H2-F6]. FIX: drop the √(η·λ/8) normalization / reparam ε on |c_a| into [floor,upper]. GATE: assert max(visco_eps)>floor over 20-ep window.
- **#2 HIGH — INIT-TIME weight-shaping levers DISCARDED on resume but print "active:true" (structured-init, lane-prior-phi1/paint, palette-anchor, siren-init).** Run at setup (2364-2496) then OVERWRITTEN by resume model.update (4725). Any A/B toggling them on a resumed run is FAKE; the paint "3× lane-FN" was a FRESH-run measurement (surrogate-vs-authority gap if cited for resumed). FIX: refuse init-levers+resume w/o --resume-reinit-ok OR emit "applied:false, reason:overwritten_by_resume".
- **#3 MED-HIGH — 5 DUPLICATE flags (argparse last-wins) silently shift schedules: eikonal-weight-end 0.1→0.05 (anneal DEAD/flat), tau-start 300→400, lane-band 350→450, persistence 300→275.** Reading top-of-config is wrong. FIX: launcher asserts no duplicate long-flags.
- **#4 MED — `--lane-prior-phi1-mode paint` yields part_frac[lane]=0.0 EVEN FRESH (lane_band_px 1261 but lane_px 0) — the paint did NOT win, contradicting "wins by construction"; likely a sign/band-side bug.** FIX: assert part_frac[lane]>0 immediately after inject_lane_sdf(paint).
- **#5 MED — `--seed-islands` compose weight annealed to 0 by ep101 (seed-anneal-epochs=101, resume start_epoch=101) → off 899/900 epochs (island-FORMATION losses via --witness-alone-island-loss ARE live).** FIX: seed-anneal relative to resume epoch.
- **CONFIRMED-REAL (trustworthy):** eikonal-viscosity 0.3 (the viscosity itself, real; the adaptive WRAPPER is inert), boundary-distance-weight 0.2, amplify/persistence, film-stiefel (per-step, resume-safe), muon (fires ep726), w-seg/w-pose/pose-carrier. NOTE: l7-start 1001 > epochs 1000 → l7 never fires (deliberate).

---

## ═══ MASTER DEDUP MAP — 18 confounds, ranked; the fix-all spec ═══

**TIER 0 — corrupted THIS session's "viscosity NO-GO" verdict (CRITICAL, 4 independent axes):**
- **C1 spike-guard default=legacy → absorbing median-freeze** [H1F1·H5F1·H6F7·H2F3·H3F1]. FIX: default→rollback; clear-spike-guard/eikonal+legacy refuse. ALARM: skip-frac>0.9/ep → spike_deadlock_ALERT.
- **C2 adaptive-ε floor-pinned INERT (constant 0.3, never adapted)** [H1F2·H4#1·H2F6]. FIX: drop √(η·λ/8) norm / reparam into [floor,upper]. GATE: max(visco_eps)>floor over 20ep else "pinned".
- **C3 EIKONAL 86-91% of loss via UNIT/π-GROUP BUG — viscous residual (~2490 raw) shares the 0.05 weight tuned for the LEGACY residual** [H6F1·H6F4·H1F6] — THE DEEPEST ROOT (domination is an uncalibrated unit bug, not physics). FIX: normalize each residual form to O(1) per-pixel-mean pre-weight; startup-assert magnitude band. ALARM: any reg term>40% loss.
- **C4 grad-clip 1.0 shared-budget hijack — seg step starved 300-900×** [H6F2]. FIX: per-term/group grad-clip. ALARM: gnorm>100×grad_clip.

**TIER 1 — META-CONFOUND: the safety systems were fooled (CRITICAL — operator's "meta confound"):**
- **C5 closed-loop certifies FROZEN run "converging" (would've bumped the exploding eik)** [H1F5·H5F5·H2F1]. FIX: gate _cl_decide on liveness FIRST → FROZEN+STOP. GATE: "converging"+ep_loss==0.0 = hard fail.
- **C6 ALL telemetry/verdict on FROZEN state w/ no liveness stamp; β-anneal-on-frozen makes eik "creep" look like physics; ep_loss:0.0 reads as converged; gold sampled on frozen epochs** [H2F2·H2F4·H2F5·H6]. **THE #1 SELF-PROTECT: stamp accepted-batch-fraction (liveness) on EVERY row.** FIX: skip=true → terms=null/weights_stepped:false; log accepted_batches + hosc_beta/softmax_temp.

**TIER 2 — the resume compound that INJECTED the poison (CRITICAL):**
- **C7 clear-spike-guard is one-shot not a cure in legacy** [H3F1] → folds into C1 fix.
- **C8 stale ep100 opt moments restored into drifted geometry (cosmetic warm-start flags, poison kept)** [H3F2]. FIX: clear-spike-guard/allow-lever-drift IMPLY warm-start-weights-only OR loud warn.
- **C9 resume loads LIVE not clean-EMA** [H3F3]. FIX: --resume-model-from {live,ema}, default ema for re-treatment.
- **C10 init-levers discarded on resume but print active:true** [H4#2]. FIX: refuse init-levers+resume w/o ack OR emit applied:false.
- **C11 allow-lever-drift injects stiff term full-weight + new levers invisible to drift guard + mislabels re-treatment as continuation** [H5F3·H3F4·H3F5]. FIX: rewarmup-ramp stiff terms on resume; persist git-sha; stamp retreatment=True.

**TIER 3 — latent traps (didn't fire this session, will next):**
- **C12 --verdict-pairs default=24 = NON-n600 (default best-ckpt+telemetry on 24/600)** [H1F3]. FIX: default→0. GATE: refuse/loud-tag !=0 at n600.
- **C13 5 duplicate last-wins flags shift schedules 100ep + flatten eik-weight** [H4#3]. FIX: launcher asserts no dup long-flags.
- **C14 eik-weight anneals UP (dead here via C13 flatten, but header lies)** [H6F3]. GATE: refuse end>base w/o waiver.
- **C15 reorient-every-50 doesn't re-treat the guard (latent; basis converged)** [H5F4]. FIX: registry of loss-shifting events each pairs a guard re-arm.
- **C16 seed-anneal-epochs=101 withdraws seed at resume start_epoch=101** [H5F6·H4#5]. FIX: seed-anneal relative to resume epoch.
- **C17 BASE trainer has the same median-deadlock, NO rollback cure (sibling surface)** [H1F4]. FIX: port rollback OR deprecate base.
- **C18 misc:** length term dead 0.0002% [H6F5] · pose 0.6% vs concave √ [H6F6] · paint part_frac[lane]=0 even fresh (sign/band bug) [H4#4] · eval-device mps doc-only [H2F7] · cl pending ep_loss=0.0 in trend [H3F6].

**CLEAN (verified honest, do NOT touch):** async-verdict race-free · verdict-batch chunk bit-identical · verdict=fp32-numpy-CPU (no MLX/MPS authority) · best-selection NO-FAKE-correct · RNG bit-faithful · atomic writes · deploy ckpt=EMA shadow · arch-drift fails closed.

**THE 3-LAYER IMMUNE SYSTEM (self-protect, operator "meta confounds"):**
- **L1 runtime alarms** (turn silent→loud): spike_deadlock (skip-frac>0.9) · term-domination (>40% loss) · gnorm-hijack (>100×clip) · adaptive-inert (pinned at floor) · frozen-verdict (ep_loss==0.0) · liveness-stamp (accepted-frac on every row).
- **L2 STRICT preflight gates** (refuse the code anti-pattern): no-guard-defaults-to-deadlock-mode · reject-filter-updates-from-accepted-only-needs-rearm · telemetry-row-tags-state-measured · default-on-lever-has-efficacy-assert · no-duplicate-long-flags · resume-palliative-flags-imply-warm-start · verdict-pairs-default-n600.
- **L3 verdict-clearance** (CLAUDE.md non-negotiable): no load-bearing verdict without an apparatus-validity precondition (liveness + lever-active + metric-on-claimed-state) + a positive-control sentinel + the recurring fresh-eyes confound-hunt cadence.
