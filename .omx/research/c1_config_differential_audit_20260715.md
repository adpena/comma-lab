# C1 CONFIG DIFFERENTIAL AUDIT — is `c1_optimal_form` the latest optimal? (#507 launch gate)

Date: 2026-07-15 (audit executed into 07-16Z) · Arm: C1 CONFIG DIFFERENTIAL AUDIT (P0, launch-gating)
· $0 static audit, nothing launched, live dry-start untouched. Research-only: `true`.
**Pointer: submittable 0.19108 UNMOVED / banked-borrowed 0.18804 UNMOVED — everything here is MEANS.**

**Operator question:** "Are these configs the latest optimal? Also check against the last few configs
to see if anything is missing or wrong or questionable." Sharpened mid-audit to: (1) center the
ORPHANED-CURE recovery (scoped to what is *genuinely* unwired), (2) the COMPLETE config X-ray — all
235 flags / every dimension classified, (3) the ranked fix sequence that becomes `c2_optimal_form`.

**Surfaces audited (recompiled/diffed myself, never trusted prose):** the live compiled argv
(`experiments/results/levelset_n600_witness_20260716T014623Z/launch.sh`, 235 flags,
dsl_compile_hash `58d449cd…`) · a fresh in-process recompile of
`spec_c1_optimal_form_20260715.compile_c1_optimal_form_launch_config()` (23 levers verified) ·
`constants_manifest.json` (14 LawRef constants) · flag-by-flag diffs vs **C0**
(`levelset_n600_witness_20260715T095030Z/launch.sh`) and vs **v752**
(`compile_crucible_v752_launch_config` recompiled live, 157 pairs) · the r2–r6 dry-start logs +
the 07-16 re-verify log · trainer argparse + wiring source (`train_levelset_witness_realized_
through_R_mlx.py`, argparse-verified for every claim below) · consumed findings:
`v9_missing_signal_constants_audit_20260715.md` (FEED-510), `wallclock_burndown_build_20260715.md`
(FEED-509burn, incl. §2b/§6/§7), FEED-509-b3x (maglaw 3-arm MEASURED), FEED-optdyn-followup,
`perparam_normalize_masks_all_norm_clipping_c0_confound_20260715`, `adaptivization_tickets_20260715.py`
(14 tickets), `SPEC_cohesive_v9max_package_20260715.md`.

---

## 0. VERDICT UP FRONT: **HOLD the r5/r6 auto-fire — c1 is NOT the final-optimal config; per operator plan #515, run the §8 A/B battery first, compose c2_optimal_form from the winners, then FIRE once**

`c1_optimal_form` is scientifically sound — nothing measured-poison is live, nothing built-and-proven
is orphaned *by choice*, every OFF is tracked with a cited reason. But it is **not the latest optimal**
on three concrete counts, all fixable in ONE $0 recompile + one dry-start:

1. **It orphans-by-timing the one PROVEN wall-clock cure**: `VerdictParallelWorkers` (built, trainer-wired
   `9d3bfc837b`, **benched 5.686× on the scorer-forward share, values float-identical**, receipt
   `experiments/results/verdict_parallel_bench_20260715T184252Z/receipt.json`) landed ~40 min AFTER the
   c1 compile and is OFF in the frozen argv. The mid-chain-mutation refusal was correct **while the
   chain was running** — but the chain has NOT fired the real launch (dry-start r6 `resume_ok=False`;
   re-verify still producing its receipt), so the "cannot mutate mid-chain" reason has expired: the
   cheapest honest move is recompile, not fire-the-stale-compile.
2. **The wall-clock budget gate is passing on an optimistic input**: L45 gate says 7.46 d (3.58 min/ep)
   within the 8.31 d compiled budget, but the measured composite at the REAL config is **~361 s/ep
   post-lane-band-engage (C0 ep33+, +75 s/ep) + verdict amortization ⇒ ~12.5 d** — ~50% over budget.
   The budget was anchored pre-lane-band. Firing as-is means a mid-run budget breach discovered at
   ep33, not a gate refusal now. (verdict_scope: instance — the gate INPUT is stale, the gate CODE is
   correct.)
3. **Two free Tier-0 flags from the parent audit's own §E are absent**: `--verdict-batch 64` (measured
   never-slower; c1 emits 32) and `--verdict-live-gap-every` (confound-H2: default-ON next launch;
   absent — the EMA-lag-vs-real-regression discriminator for exactly the early rows this run's first
   read depends on).

Everything else measured-suspect (grad-clip magnitude law, EMA/Muon/beta2 ancestor constants, basis
Fourier-vs-curvelet, activation β_end) has a **reasoned, cited, correct** disposition in c1 and is NOT
a launch blocker. The launch should hold for the c2 recompile, not for those A/Bs.

---

## 1. DIFFERENTIAL vs THE LAST FEW CONFIGS

### 1a. v752 → c1 (recompiled both; 157 → 235 pairs; **0 flags dropped**)

| flag | v752 | c1 | provenance rung |
|---|---|---|---|
| `--grad-clip` | 1.0 | **0.5** | INCUMBENT-sealed (`witness_stability.AMBER.grad_clip`); saturation telemetry real but **INERT under `--grad-normalize per-param`** (confound memo; §4 below) — ANCESTOR-SUSPECT, magnitude-law A/B owed |
| `--mod-dim` | 32 | **19** | DERIVED (Whitney law; mod19 matched control = the leg-A parent; #299 arm holds the 32-vs-19 harvest gate) |
| `--stage-transition-rewarmup-epochs` | 8 | **14** | lever `R7_beta2_window_rewarmup` (typed; boundary re-warmup law) |
| `--stage-transition-rewarmup-shape` | linear | **cosine** | same lever |
| +78 net additions | — | event gates, lane-band family, ladder homotopy, persistence stack, structured-init, pose-carrier, telemetry, S_R, phase-tail… | each lever-owned (23 levers verified against `C1_OPTIMAL_FORM_EXPECTED_ADDITIONS` fail-closed accounting) |

Nothing in v752 was silently lost. The v752→v9·CGauge→c1 lineage is strictly additive plus 4 reasoned
value changes.

### 1b. C0 → c1 (the #509/#510 audit substrate → the launch candidate)

Order-normalized diff = exactly:

| change | provenance |
|---|---|
| `--annulus-plateau-dwell-windows` 4 → **7** | **DERIVED** (F10 satisfiability law: dwell = min_epochs/eval_every + 1 = 150/25+1; the prior 4 made the event unsatisfiable — a real bug FIXED, not drift) |
| `--basis legacy_fourier_ab_control` + `--no-self-orient` now EXPLICIT | typed custody (spec_v9_cgauge base: deliberate A/B control + the owed-16-refuted self-orient drop made explicit instead of argparse-default-implicit) — value-identical to C0's behavior |
| + `--margin-saliency-reachability` (S_R) | leg-A sole scientific delta (#510 §A-6: PREPARED_NOT_FIRED → now firing) |
| + `--pose-training-compute-gate`, `--verdict-pose-gate` | leg-C #495 (trunk-phase compute saver; pose UNCHANGED R1 two-phase finisher) |
| + `--head-offset-solver flip_median` + tau 1.0 | leg-C #386 (Hamming-optimal S1 advisory; OT area objective MEASURED-worse; never mutates shipped/EMA weights) |
| + `--component-wallclock-telemetry` + probe-every 1 + `--profile-timing` | leg-B #480 instrument (the 63%→77.4% attribution answer rides THIS run) |
| + `--seg-phase-advect-start-event label_floor` | #507 skeleton-dissolve (the last epoch-scripted transition now event-fired; 726 demoted to loud backstop cap) |

C0→c1 is exactly the three legs + one derived bug-fix. **No unexplained drift.**

---

## 2. THE COMPLETE CONFIG X-RAY — every one of the 235 flags, by dimension

23 levers own ~96 flags (recompile-verified lever→flag map); the remainder is the crucible-v7/v9 base.
Every flag below is classified {live value · optimal? · rung · cure-if-suboptimal}. No dimension omitted.

### 2a. Optimizer / step dimension (the genuinely-open one)

| flag(s) | live | optimal? | rung | cure status |
|---|---|---|---|---|
| `--grad-clip 0.5` + `--per-group-grad-clip` + `--grad-normalize per-param` | fixed clip, saturated (frac_clipped=1.0) **but INERT**: per-param normalize divides out any norm clip downstream — magnitude law = unit-norm × LR (SignSGD-like) | **UNKNOWN — the deepest open question in the config.** Measured: AutoClip arm-B wins −10.35% at ep25 then **REVERSES post-ep25** (ep75 B 0.018644 vs A 0.015325; mechanism T4: AutoClip p10/w1000 = lagged norm-target under decaying gnorm ⇒ 5–18× step overshoot/EoS) | incumbent = ANCESTOR-SUSPECT; the lr/12 poison reading is **REFUTED** (confound memo) | **A/B OWED, correctly not wired**: ≥150-ep magnitude-law A/B + S4 causal-rebase discriminator (~$0, first-to-fire when GPU frees). `AdaptiveGradClip`/`GradNormalizeNone` levers BUILT + compile end-to-end — one `--dsl-lever` away when the verdict lands. c1 keeping incumbent is the right call (verdict_scope: formulation — naive-AutoClip-at-this-window, not the percentile-law family) |
| `--lr 1e-3 → --lr-end 1e-4`, `--lr-anneal-epochs 1000`, `--lr-hold-frac 1.0` | cosine, **event-mode: LR rides the τ-octave fraction** | YES for the mechanism — **CORRECTION to the #510 §B-2 "BUILT, unwired" row: event-LR IS WIRED AND LIVE** (trainer:12216 `_lr_scheduled_event_for_epoch(ep, args, _tau_ctrl.lr_anneal_fraction())`, active because `--tau-advance-mode event`); the 1000 clock is only the fallback denominator | DERIVED (`lr_control_denominator_v1`, `lr_hold_frac_no_hold_v1`) | none owed at launch |
| `--adam-beta2 0.999` | AdamW default | unproven | ANCESTOR-SUSPECT (beta2-window law exists) | ticket; `R7_beta2_window_rewarmup` lever (reset-moments + 14-ep cosine re-warmup at boundaries) already cures the CROSS-STAGE hazard — remaining question is the in-stage value |
| `--ema-decay 0.997` | Quantizr anchor | unproven on witness | ANCESTOR-SUSPECT (L18) | ticket. Partial cure LIVE: `R7_polyak_finisher` ON (`--polyak-finisher-arm`, start 2546 DERIVED) = the stage-dependent-averaging law at the turnpike. `--ema-decay-finisher` (SWA-widen, trainer-built, default None) is the remaining unwired sibling — redundant with Polyak, keep OFF |
| `--muon-lr 0.002 / --muon-momentum 0.95 / --muon-ns-steps 5 / --muon-lr-final-frac 0.1 / --muon-adamw-lr 1e-4 / --muon-warm-start-momentum` | PR95-lineage internals | **QUESTIONABLE — optdyn measured the finisher runs an UNCHOSEN per-layer LR increase ×1.40 (film) / ×1.09–1.22 (hidden)** (η_rel=‖u‖/‖W‖ not pinned; NS update norm weight-independent) | ANCESTOR-SUSPECT, unmapped in registry | ticket-only (η_rel-pin ticket NEW in optdyn; no trainer flag exists — correctly not hand-wired). **The named $0 unlock is the per-tensor ‖W‖ telemetry row (defaults ON) — wire into c2** so the live run MEASURES its own norm drift |
| `--accum-pairs 8` | 75 steps/ep all stages | unvalidated cross-stage | QUEUED (joint sweep with clip/LR owed) | ticket |
| `--micro-batch-pairs 1` | serial | **YES** — receipts 1.0–1.07× (#447 Metal + fresh_eyes B2 1.036×; old 2–4× projection superseded) AND trainer fail-closes S_R under B>1 (batched LEVER-4 twin = named fallen-crack) | MEASURED | none; unlock = batched S_R consumer |
| `--containment-mode shield / --containment-damp 0.1 / --pose-grad-coeff-max 25.0` | anti-collapse guards | inherited, no counter-evidence | sealed | none |
| `--weight-decay 1e-4` | AdamW wd | optdyn: wd channel negligible (−3.3 of 7:1:0.05 decomposition) | ANCESTOR | none binding (radial force owns the norm ODE) |

### 2b. Basis / activation / INR dimension (the OPEN scientific dimension)

| flag(s) | live | optimal? | cure status |
|---|---|---|---|
| `--basis legacy_fourier_ab_control` | Fourier (typed custody, DELIBERATE control) | **OPEN — the whole dimension.** The 3.2× along-tangent deficit + no-Fourier doctrine say curvelet SHOULD win; #502 built genuine curvelet frames; **zero through-R d_seg number exists** | **PAIRED ARM STAGED, not orphaned**: `c1_optimal_form_curvelet_arm` (same seed/levers, delta = `--basis windowed_curvelet` + bank params) is the receipt producer for the owed `curvelet_through_R_dseg_ab`. Memory-blocked from concurrent fire (2×~67 GiB > 89.6 GiB 0.70-ceiling) ⇒ fires immediately-next. Main config keeping the Fourier control is doctrine-correct (curvelet opt-in, never a silent flip) |
| `--no-self-orient` | directional front-end OFF | YES — owed-16 A/B refuted it (it is itself a directional-FOURIER feature) | none |
| `--activation hosc`, `--hosc-beta 1.0 → 3.177`, `--hosc-beta-anneal linear`, `--hosc-omega 1.0`, `--siren-init` | annealed hosc | **stable survivor, optimality UNPROVEN**: β_end 3.177 is a control-preserving rephase (custodied `hosc_beta_fireband_pin_v1`), contested by step_iso β_end=8.0 (34.2% duty, never fired) + FreSh #448 / FINER++ #310 / StepNative isolates (EXPLAINED-ISOLATE: fresh-start basin treatments, stacking would confound the core) | **β co-anneal on the rung fraction IS LIVE** (trainer:12139 `_tau_ctrl.hosc_beta_for_epoch` under event mode — the #510 §B-3 "unwired" row is STALE at HEAD). ISO/basin arms = queued paired arms post-core-read |
| `--mod-dim 19`, `--hidden-dim 96`, `--n-hidden 4`, `--max-bank-freq 64` | capacity spine | mod19 DERIVED (Whitney); 32-vs-19 harvest gate open (#299 arm) | arm queued |
| `--softmax-temp-start 1.0 → --softmax-temp-end 0.31` | τ anneal, event-rung-advanced | endpoint MEASURED (P-TAU2 knee, band [0.191,0.543]); PATH adaptivized by event rungs | none at launch |

### 2c. Curriculum / event dimension (dissolved — verified live)

**All five transitions event-fired with loud backstop caps** (launcher schedule-provenance gate 12/12 OK,
r6 + re-verify logs): muon=`powerlaw_meat`(cap 726) · lane-band=`lane_nucleus`(500; fired ep31–33 on C0
— sensor→transition proven live) · seg-chroma=`annulus_plateau`(450) · temporal-screw=`annulus_plateau`
shared-field(450) · **phase-advect=`label_floor`(726) — NEW this landing, closes the last epoch-scripted
transition**. `--tau-advance-mode event` (rungs DERIVED via `derive_n_octaves`), `--pose-finish-engage-on
sigma_min_plateau` (#383), `--birth-completion-event` (τ-persist 0.8 + area band). Remaining epoch
constants are either caps (correct class) or clock-adjacent constants with tickets:
`--persistence-warmup-epochs 275`, `--curriculum-min-stage-epochs 250` (feeds the rung derivation),
ladder birth constants (`--ladder-movable-r0 0.2252/birth 60/anneal 200`, `--ladder-lane-r0 2.0/80/260`
— §B-8 hand-placed; the saddle-node critical-λ derivation is the ticket, reduced-order model owed),
`--polyak-finisher-start-epoch 2546` (DERIVED fencepost), backstops 450/500/726 (cap-classified,
`cap_fired_before_event` = S5 falsification signal — correct design). `--curriculum-event-triggered`
is INERT under unify_tau (trainer's own loud row; verdict_scope: instance — this flag under the
unify_tau curriculum only, not a dead lever).

### 2d. Loss-term stack (all lever-owned, all reasoned)

LIVE + provenance-clean: persistence/clDice (w 1.0/1.0, iters 5, classes 3) · island amplify (hinge,
inverse_thickness) · `n323_ladder_island_homotopy` (18 flags) · `v75_area_constraint_birth` (classes 1,3)
· `n287_dash_comb` · `temporal_screw_consistency` (0.1 @ annulus_plateau, ξ=ground_gt) ·
`phase_advection_consistency` (0.4 DERIVED blink-back 0.418, @ label_floor) · `tie_locus_displacement`
(subpix 0.3, pa_flipmass edge weights) · `margin_band_satisficing` (0.2, m_safe 0.03918 DERIVED-live) ·
`margin_saliency` (1.0) + **S_R exact through-R reachability** (replaces the AT-CHANCE texture proxy,
L76) · `unified_tau_eikonal_hold` (eikonal 0.01→0.05 rung-coupled, L13 anchor) ·
`n292_closed_loop_eikonal_control` · `dseg_aware_taper` (#121 Fourier-taper base, 4 flags; the
taper-off ISO arm 78.9% duty holds the prove-via-ISO) · `FEED_08a_length_sigma` (fitted-20260707 σ_cc′)
· `seg_form_unify_tau` · chroma-boundary (0.1 @ annulus_plateau) · lane-band family (τ 0.85, ε 0.35,
dash-forward 55 m, w 1.0) · `--weight-entropy-penalty-lambda 15.0` · logit-adjust (τ 1.0, classes 3) ·
`--w-seg 100`/`--w-pose 1.0` (w_seg = exact costate; **w_pose ANCESTOR at one d_pose** — λ_pose law
`5/√(10·d_pose)` registered, no consumer, binds at pose-finish engage — ticket) · `--score-domain-loss`.
The §2b optdyn caveat applies config-wide: under per-param normalize, loss WEIGHTS steer direction
only — every weight above is direction-mixture-valid but magnitude-blind until the magnitude-law A/B.

Adaptive-ε #318/#320: SLOT, correctly — inert without `--eikonal-viscosity>0` (never sealed) AND the
CFL-edge cure is FALSIFIED_MECHANISM at n600 (FEED-06g). Folding it = the #417 counted-but-inert fake.

### 2e. Verdict / eval / wall-clock dimension (where the fixes live)

| flag(s) | live | optimal? | cure |
|---|---|---|---|
| `--eval-every 25` | cadence | the +903 s/verdict submit-block fires here (§A-3) | **instrumented in c1**: `real_verdict_submit_s` (#480, rides `--component-wallclock-telemetry`) decides submit-block vs cadence-25 checkpoint/mdd-ablation coincidence at the FIRST ep25 row; NCDE cadence law = ticket (consumer owed) |
| `--verdict-batch 32` | chunked | **SUBOPTIMAL — vb=64 measured never-slower** (parent audit §C.2/D.3-7, free) | flip in c2 |
| `--verdict-parallel-workers` | **ABSENT (=0)** | **NO — the benched 5.686× scorer-share cure is off** | wire `VerdictParallelWorkers(8)` in c2 (headroom-derived; first-verdict identity self-check; honest scope: divides the ~370 s scorer share ⇒ ~−12 s/ep amortized at cadence 25, NOT the whole 2555.7 s verdict wall) |
| `--verdict-live-gap-every` | ABSENT | NO — confound-H2 says default-ON | add in c2 (score-neutral observability; EMA-lag vs real-regression discriminator) |
| `--verdict-pairs 0`, `--verdict-device cpu`, `--async-verdict`, `--verdict-anchor-every 0` | n600 CPU authority async | YES (n600-scale law; CPU = verdict yardstick) | none |
| `--mod-dim-dynamics` | ABSENT | D18 claims default-ON; absent + registry-unmapped | orphaned SENSOR (score-neutral, k90 rate signal → D18 ~7 KB free-rate lead); add in c2 |
| `--eval-every`-coupled `--ckpt-every 25` + `--stage-checkpoints` | resumability P0 | YES | none |
| `--fused-r-kernel`, `--cache-gt-skeleton`, `--safe-compile-regions hosc_activation`, PERF_ENV grouped-backward+persistence-pool | ~17× speed set ON | YES (relaxed-identity 07-15: functional parity IS the gradient bar; megakernel #356 stays out on MEASURED economics CPU 0.79–0.83×) | none |
| `--lane-band-cache-static` | default-ON (unemitted) | YES but NOT a wall-clock lever (−0.04 s/ep measured; the +75 s/ep lane-band cost is intrinsic θ-dependent forwards) | the real lane-band sec/ep lever is score-affecting (fold `call_margin` into the main forward) — ticket with own A/B |
| `--compute-dtype` | ABSENT (=fp32) | fp32 correct for THIS launch | bf16 seam BUILT + QC ADMIT (median cos 0.9925, p10 below bar) — paired sec/ep bench + stage-boundary QC owed before adoption; separate arm, not c2-core |
| telemetry: `--annulus-telemetry`+5, `--jacobian-basin-*`(7), component-wallclock(3) | ON | YES (off-is-orphan law) | add per-tensor ‖W‖ row (optdyn unlock) in c2 |

### 2f. Render / R / pose / data (settled)

`--render-h 384 --render-w 512 --render-aa ipe --chroma --palette-anchor` (chroma = d_seg lever, doctrine)
· pose-carrier family (7 flags: table residual, generated source, s_t 0.044) + R1 two-phase finisher —
pose UNCHANGED and SOLVED-for-this-vehicle (R1 dxi banked; PoseBlindComputeGate only saves blind-phase
compute) · `structured-init` family (7 flags, lane included) · seed-islands family (5) + shield ·
`tail_k_warm_restart` (7 flags; tail controller) · `--seed 0 --num-pairs 600 --epochs 3000` +
`--gt-cache gt_n600.npz` (n600-scale law honored) · `--mlx-device gpu` + CPU verdict split (authority
discipline). All sealed/derived; no open questions at this launch.

---

## 3. ORPHANED-CURE RECOVERY TABLE (scoped to what is GENUINELY unwired)

**Corrections first (I verified these in trainer source — do NOT call them orphaned):** event-LR
(`lr_anneal_fraction`, trainer:12216) and β-rung co-anneal (trainer:12139) are **LIVE** under
`--tau-advance-mode event` — the #510 §B-2/§B-3 "BUILT, unwired" rows are stale at HEAD (S6-R4 wiring).
Also LIVE: Polyak finisher · #121 dseg-aware taper · beta2-window re-warmup · S_R · dash-comb ·
temporal screw · phase-advection @ label_floor · tie-locus · satisficing · unified-τ-eikonal-hold ·
closed-loop eikonal · ladder homotopy · head-offset flip_median · pose-finish gate · #480 telemetry ·
lane-band static cache (default-ON) · the 5 event gates. c1 wires far more of the built work than the
escalation feared.

The genuinely unwired, each with (a) artifact (b) target constant (c) exact DSL wire-in (d) impact/cost
(e) caveat:

| # | cure (built artifact) | targets | DSL wire-in | impact / cost | caveat / state |
|---|---|---|---|---|---|
| 1 | **VerdictParallelWorkers** — lever + trainer wiring `9d3bfc837b` + bench receipt 5.686× (`verdict_parallel_bench_20260715T184252Z`) | the 2555.7 s n600 CPU verdict wall at `--eval-every 25` | compose `VerdictParallelWorkers(8)` into the c2 factory (headroom-derived default; identity self-check built-in) | ~−12 s/ep amortized (~−3–4% of 361 s/ep) + faster ep25 first-read / **$0, one recompile** | **ORPHANED-BY-TIMING** (law landed 40 min post-compile; chain-freeze reason expired since no real launch fired) |
| 2 | **vb=64** (measured never-slower) | `--verdict-batch 32` | value bump in the c2 base (typed) | small, free / $0 | ORPHANED free flip (parent audit §E Tier-0 named it; c1 kept 32) |
| 3 | **`--verdict-live-gap-every`** (trainer flag exists) | early-row interpretability | score-neutral lever, defaults-ON per off-is-orphan | protects the FIRST decisive read / $0 | ORPHANED (confound-H2 Tier-0 row not picked up) |
| 4 | **`--mod-dim-dynamics`** (D18 k90 sensor, trainer flag exists) | rate-lever lead (~7 KB k90-truncate) | score-neutral sensor lever, defaults-ON | free rate signal for the D18 byte-close / $0 | ORPHANED + registry-unmapped |
| 5 | **per-tensor ‖W‖ telemetry row** (optdyn "the ONE owed logging change, defaults ON") | `--muon-lr` family (×1.40 unchosen per-layer LR increase), radial-norm ODE `inr_weight_norm_radial_ode_v1`, ‖w‖*=min(k_need,k_R)/ω band-edge | small telemetry_producers addition + score-neutral lever | makes the 12.5-d run MEASURE its own norm drift (feeds η_rel-pin + WW-PGD/row-norm-projection tickets, which are ticket-only-unbuilt) / ~1 h | BUILD-SMALL — the only item here needing new code |
| 6 | **AdaptiveGradClip + GradNormalizeNone levers** (built, compile end-to-end, maglaw A/B partially measured) | `--grad-clip 0.5` + `--grad-normalize per-param` | `--dsl-lever` composition — but **only after** the ≥150-ep durability A/B + S4 causal rebase | epochs-to-target (multiplicative; unknown until measured) / A/B ~$0 n24 | **correctly NOT wired**: arm-B REVERSES post-ep25 (lagged-norm-target overshoot, T4); wiring naive AutoClip today would ship a measured regression |
| 7 | **bf16 compute seam** (`ComputeDtype` lever, QC ADMIT) | sec/ep ceiling ~1.5–1.8× (estimate-flagged) | `ComputeDtype("bf16")` after paired sec/ep bench + stage-boundary QC re-runs | potentially the largest sec/ep lever / bench ~$0 | gated, separate arm — not c2-core |
| 8 | **curvelet arm** (#502 frames + staged `c1_optimal_form_curvelet_arm`) | `--basis` dimension (3.2× along-tangent deficit) | already a compiled config; fires immediately-next (memory-blocked from concurrent) | the open scientific dimension / $0 compile done | STAGED, not orphaned — keep the sequencing |
| 9 | ticket-only-unbuilt (named, honest): NCDE verdict-cadence consumer · #341 terminal head GN solve · batched LEVER-4 S_R consumer (unlocks B>1) · `--head-offset-precondition` (#423) · Bregman #504 consumer · saddle-node birth-λ derivation · λ_pose engage-time consumer · η_rel pin / WW-PGD / row-norm projection · verdict-submit FIX (decision-gated on c1's own ep25 row) | various | each has a typed ticket/slot with cited unlock | — | not wireable today without violating #417/no-fake |

---

## 4. WRONG — anything measured-poison still ON?

**No.** The prime suspect resolves clean: `--grad-clip 0.5` saturation (frac_clipped=1.0, the
FEED-510 "lr/12" poison row) is **INERT on this config** — `--grad-normalize per-param` unit-norms
every tensor AFTER the clip, dividing out any uniform norm scale (source-verified;
`perparam_normalize_masks_all_norm_clipping_c0_confound_20260715`). The telemetry was real; the
attribution was false; the incumbent magnitude law (unit-norm × LR) is UNVALIDATED-but-not-measured-worse,
and the one measured alternative (AutoClip) **reverses post-ep25**. Keeping the incumbent pending the
≥150-ep A/B is the correct, cited disposition (SPEC §3 grad-clip row). Likewise: no FEED-06g-falsified
adaptive-ε, no measured-worse OT head-offset, no megakernel, no naive-AutoClip — every measured-negative
surface is correctly OFF/SLOT. The two remaining "wrong-ish" items are **gate inputs, not config
values**: (a) the L45 wall-clock budget (8.31 d) anchored pre-lane-band vs 12.5 d measured; (b)
mem-preflight projected 24.48 GiB vs 41.86 GiB measured pass-1 peak (1.7× under-projection; still
ADMIT-safe at 128 GiB — calibration owed, not blocking).

## 5. QUESTIONABLE — ancestor-suspect + cross-stage-unvalidated (each with the owed validation)

| constant | owed validation |
|---|---|
| `--ema-decay 0.997` (Quantizr/L18) | decay cross-stage sweep ticket; Polyak arm ON partially covers the finisher |
| `--muon-lr 0.002/momentum 0.95/ns-steps 5/final-frac 0.1` | η_rel-pin ticket + the ‖W‖ telemetry row (recovery-table #5) — optdyn measured the unchosen ×1.40 |
| `--adam-beta2 0.999` | beta2-window law; in-stage value sweep (boundary hazard already cured by R7 re-warmup) |
| `--accum-pairs 8` | joint sweep with clip/LR (ticket) |
| `--w-pose 1.0` | λ_pose=5/√(10·d_pose) at pose-finish engage (registered, consumer owed) |
| `--eval-every 25` | NCDE cadence law (ticket); ep25 submit-block row decides the −36 s/ep fix |
| `--hosc-beta-end 3.177` | step_iso 8.0 arm (34.2% duty) — endpoint contested, path adaptivized |
| ladder birth constants | §B-8 saddle-node derivation (reduced-order model) |
| `--persistence-warmup-epochs 275` / `--curriculum-min-stage-epochs 250` | clock constants adjacent to event sensors — would-fire calibration accrues this run |
| the 7 ADAPTIVIZATION-QUEUED manifest constants (per FEED-510 §C.1) | each has a typed ticket; 2 of the 7 (event-LR, β co-anneal) are ALREADY CURED live (stale audit rows) |

## 6. MISSING (built-but-not-wired that SHOULD be in this launch) — recovery-table rows 1–5

`VerdictParallelWorkers(8)` · `--verdict-batch 64` · `--verdict-live-gap-every` · `--mod-dim-dynamics`
· per-tensor ‖W‖ telemetry. All score-neutral-or-value-identical wall-clock/observability items; all
$0; all require the ONE c2 recompile. Nothing score-affecting is missing without a cited gate.

---

## 7. LAUNCH VERDICT: **HOLD — operator plan #515 locked: fix all poison + run the FULL A/B battery FIRST; the next long run must be the best-attempt final optimal end state on all dimensions**

c1 as-frozen is honest but stale by one working session: it orphans-by-timing 5 free cures, its
wall-clock budget gate passes on a pre-lane-band input while the measured projection is **~12.5 d vs
the 8.31 d budget**, and three whole dimensions (magnitude law · basis · ancestor optimizer constants)
have never been resolved by measurement. The dry-start gate is not green yet either (r6
`resume_ok=False`; the 07-16 re-verify is the receipt producer) — the launch is mechanically held
regardless. Per #515 the hold is now the PLAN: execute the §8 battery, compose `c2_optimal_form`
from the winners, and fire ONE long run at the measured optimum.

## 8. THE A/B BATTERY PLAN (#515) — every open dimension → a measured winner → c2_optimal_form

All arms compose through DSL Lever factories / typed configs (never hand flags), launch through the
governed launcher, and carry the standard admission bar: gradient-quality + no-flicker + n600-scale
verdict rows for anything decision-grade (n24 = SCREEN only, never a verdict; verdict_scope discipline
on every negative). All compute is local M5 Max ⇒ **$0 cloud spend for the entire battery**; the cost
currency is GPU-days on the single Metal device (arms are SEQUENTIAL — wall-clock hygiene + the
0.70-concurrent memory ceiling). n24 ≈ 75 s/ep ⇒ 150 ep ≈ 3.2 h/arm; n600 ≈ 300–360 s/ep.

**Dependency spine:** B0 (instruments) → B1 (magnitude law: it sets the DESCENT CLOCK — every
epochs-metric downstream is confounded until it is fixed) → B2 (basis: basis-BEFORE-capacity, L24)
→ B3 (ancestor constants, each on the B1-winner clock) → B4 (wall-clock burn-down, score-neutral,
parallel-in-calendar with B3 since they share arms' telemetry) → compose c2 → dry-start → FIRE.

| id | dimension + arms | pre-registered metric + falsification | scale + wall-cost | order / gates | mode |
|---|---|---|---|---|---|
| **B0** | instruments (no A/B needed — identity/score-neutral by construction): `VerdictParallelWorkers(8)` (identity self-check built-in) · vb 64 · `--verdict-live-gap-every` · `--mod-dim-dynamics` · per-tensor ‖W‖ telemetry row (~1 h build) | first-verdict value-identity check passes; telemetry rows appear | $0, ~2–3 h build+dry-start | FIRST — every later arm inherits the instruments | $0-local |
| **B1** | **magnitude law (the poison dimension):** A = incumbent (per-param normalize + inert clip 0.5) · B = normalize-none + AutoClip(p10,w1000) · C = normalize-none + fixed 0.5 · **+ S4 causal rebase** (fork armB@ep75 onto fixed-0.5) — extends the measured ep1-39 window (A/B/C exist; B won ep25 −10.35% then REVERSED) | d_seg (cadence-25 verdict) at ep150 + monotone-tail requirement; log-slope ep1-150; FALSIFY any arm that reverses or trips flicker/gradient-quality; winner = lowest ep150 d_seg with monotone tail; if S2/S1 discriminators (percentile/window sweep) rescue AutoClip, that variant re-enters | n24 screen: 4 arms × ~3.2 h ≈ **13 h**; winner CONFIRMED at n600 inside the c2 run's first 150 ep (pre-registered checkpoint read, no extra run) | GATES EVERYTHING (descent clock) | $0-local, governed n24 |
| **B2** | **basis (Fourier vs curvelet, the open scientific dimension):** main (`legacy_fourier_ab_control`) vs `c1_optimal_form_curvelet_arm` (`--basis windowed_curvelet`, #502 frames; same seed/levers) — both arms carry the B1 winner | the owed `curvelet_through_R_dseg_ab`: n600 through-R d_seg at matched epoch/bytes; NO-REGRESSION bar for the doctrine strict-flip; pre-register the bounded read as PROVISIONAL (basis may only separate in the terminal band) — decision rule: curvelet ≥ parity at ep500 ⇒ curvelet into c2 (doctrine); curvelet worse ⇒ Fourier stays, curvelet re-queued terminal-band | n24 paired screen ~500 ep ≈ 2×10.4 h ≈ **21 h**; then n600 bounded ep500 paired ≈ 2 × (500×~350 s) ≈ **2.0 d each, 4 d sequential** (the big-ticket item) | after B1; gates capacity/mod-dim follow-ups (basis-before-capacity) | $0-local, governed |
| **B3a** | `--ema-decay`: 0.997 vs {0.99, 0.999} (+ Polyak-on as the finisher control, already in-config) | EMA-verdict d_seg @ep150 n24; falsify if best-vs-worst < noise band (then 0.997 stays as DONT-CARE, re-classed) | 2 extra arms ≈ **6.5 h** | after B1 | $0-local |
| **B3b** | `--adam-beta2`: 0.999 vs beta2-window-law value | same metric; same falsification | 1 arm ≈ **3.2 h** | after B1 | $0-local |
| **B3c** | `--accum-pairs`: 8 vs {4, 16} (joint with the B1 winner — the audit's named joint sweep) | d_seg@ep150 + sec/ep (JOINT objective — accum moves both) | 2 arms ≈ **6.5 h** | after B1 | $0-local |
| **B3d** | Muon internals: incumbent finisher vs η_rel-pinned finisher | requires the η_rel-pin BUILD (ticket-only; ~day) — if not built by battery time, DEGRADE to measure-only: the B0 ‖W‖ rows on the c2 run quantify the ×1.40 drift and the pin A/B forks from the ep726 stage checkpoint LATER (per-stage-checkpoint dividend: a terminal A/B costs only the tail, not a rerun) | fork-from-checkpoint: 2 × tail ≈ deferred | after c2 reaches ep726 (not launch-gating) | $0-local |
| **B3e** | `--w-pose 1.0` vs λ_pose=5/√(10·d_pose) law at pose-finish engage | d_pose at engage + d_seg non-regression; fork both arms from the c2 ep726/sigma_min checkpoint | fork-from-checkpoint tail ≈ deferred | with B3d (terminal-band pair) | $0-local |
| **B3f** | `--hosc-beta-end`: 3.177 vs 8.0 = the built step_iso arm (34.2% duty) | d_seg@ep150 n24 + activation-saturation telemetry (fixed-β divergence is the known failure; annealed 8.0 endpoint is the question) | 1 arm ≈ **3.2 h** | after B1 | $0-local |
| **B3g** | #121 taper: incumbent ON vs taper-off ISO (78.9% duty — proves the incumbent pays at all) vs adaptive-width extension (margin-band-conditional/per-class @ radial-ODE band edge — BUILD-OWED ~1–2 d) | d_seg@ep150 n24; extension enters only if built by battery time, else taper-off ISO alone settles keep/drop | 1–2 arms ≈ **3.2–6.5 h** | after B1 | $0-local |
| **B3h** | `--eval-every 25` vs 50 (+ NCDE cadence law when its consumer lands) | wall-clock/ep + event-sensor satisfiability (F10: dwell-windows RE-DERIVES with eval_every — recompute 7→4 at cadence 50; would-fire calibration must stay green) + d_seg read-latency cost | piggybacks B3a-c arms (cadence is per-arm config) ≈ **+1 arm 3.2 h** | after B0 (needs live-gap + submit-row instruments) | $0-local |
| **B4a** | verdict-submit block: FIX vs no-fix decision | the first n600 verdict-epoch `real_verdict_submit_s` row (from the B2 n600 bounded arms or the c2 dry-start) decides submit-block vs checkpoint/mdd coincidence; then the fix is a build, re-A/B'd by its own before/after row | ~0 (rides B2/c2) | data-gated | $0-local |
| **B4b** | bf16 compute seam: fp32 vs `ComputeDtype("bf16")` | paired sec/ep bench + stage-boundary QC re-runs (cos_min 0.99 / rel-band, QC ADMIT already at trunk); falsify on QC fail at any stage boundary; win = 1.5–1.8× sec/ep at quality | n24: 2 × ~3 h + QC ≈ **8 h** | after B1 (magnitude law interacts with QC direction-grading) | $0-local |
| **B4c** | micro-batch B>1 fp32 fused re-measure (receipts 1.0–1.07× were unfused/fp32-era) | sec/ep only; S_R keeps it code-blocked in the MAIN config regardless — this only prices the batched-S_R-consumer build | 2 × ~1 h bench | any time GPU free; NOT launch-gating | $0-local |
| B5 (tail, optional) | activation-basin isolates: FreSh #448 / FINER++ #310 / StepNative fresh-start arms | pre-registered as ISOLATES (fresh basins — never stacked on the core); post-c2-read | 3 × ~3.2 h | after c2 first read | $0-local |

**Total battery cost (for operator GO):** cloud **$0**; local Metal wall-clock — **n24 battery
(B0+B1+B3a-c,f-h+B4b) ≈ 47–53 h ≈ 2.0–2.2 GPU-days**; **B2 n600 bounded basis pair ≈ +4.0 GPU-days**
(the dominant item; the compressed alternative — n24 basis screen only, n600 curvelet read deferred to
the staged immediately-next arm — saves those 4 days at the price of firing c2 with the basis
dimension PROVISIONAL). So: **full battery ≈ 6.2 GPU-days; compressed ≈ 2.2 GPU-days**; then the c2
long run itself ≈ 10–12.5 d (until B4a/B4b receipts lower it). B3d/B3e cost nothing up front
(fork-from-checkpoint tails). Builds owed inside the battery: ‖W‖ telemetry (~1 h, B0), η_rel pin
(~1 d, optional B3d), adaptive-width extension (~1–2 d, optional B3g), verdict-submit fix (data-gated
B4a).

**c2_optimal_form composition recipe (winner-sets-flag):**
- B0 → `VerdictParallelWorkers(8)` + `--verdict-batch 64` + `--verdict-live-gap-every` +
  `--mod-dim-dynamics` + ‖W‖ telemetry (unconditional, score-neutral).
- B1 winner → {`--grad-normalize`, `--grad-clip`/`--grad-clip-mode`(+percentile/window)} — via
  `AdaptiveGradClip`/`GradNormalizeNone` levers or incumbent-stay.
- B2 → `--basis` {legacy_fourier_ab_control | windowed_curvelet + bank params} per the no-regression
  rule.
- B3a/B3b/B3c/B3f/B3h winners → `--ema-decay` / `--adam-beta2` / `--accum-pairs` / `--hosc-beta-end`
  / `--eval-every` (+ RE-DERIVED `--annulus-plateau-dwell-windows` if cadence changes).
- B3g → keep/drop/extend `dseg_aware_taper`.
- B4b ADMIT → `ComputeDtype("bf16")`, else fp32.
- B3d/B3e run as fork-from-checkpoint tails on the c2 run itself (not compile-time flags; the pin/λ_pose
  levers enter c3 or the fork wins directly).
- Everything else: byte-identical to c1 (the §2 X-ray's settled dimensions), budget re-anchored to the
  measured composite (§7 item 2), dry-start GREEN required as ever.

**verdict_scope: instance — this audit rules on the c1_optimal_form compile of 2026-07-15/16 on this
host; all sec/ep numbers [macOS-MLX advisory] NON-PROMOTABLE; nothing here is a score claim.**
Pointer **0.19108 UNMOVED** — this entire audit is MEANS; the pointer moves only through a byte-closed
n600 `upstream/evaluate.py` exact row.
