# V9·CGauge missing-signal + constants-are-poison audit + wall-clock hotspot ledger — 2026-07-15

**Tasks:** #510 (primary) + #509 (profiling backbone, folded). **Operator directives (verbatim):**
"Spawn fable subagent to search all of our research and related landed code and todos and do follow on
research online to raise signal we are missing for optimal v9 cgauge config and min wall clock complex
optimal…" + "Constants are poison unless truly optimal across all surfaces and all curriculum we can do
much better." + "Need to examine all of our latest and state of our profiling of time per epoch and run
fresh runs with full telemetry and breakdown so we can continue burn down and lowering and hotspots."

**Doctrine:** `constants_are_poison_unless_optimal_across_surfaces_curriculum_20260715` +
`feedback_wallclock_to_target_joint_objective_drift_ok_if_gradient_good_20260715` (the objective is the
JOINT product **epochs-to-target × sec/epoch**; training drift OK if gradient quality + no flicker;
bit-identity binds decode/verdict only).

**Pointer:** 0.19108 submittable / 0.18804 borrowed bank — **UNMOVED**. Everything below is MEANS.

---

## §0 Live state (MEASURED, 2026-07-15 ~14:45 UTC)

- **C0 (`levelset_n600_witness_20260715T095030Z`, config `v9_cgauge_ideal_mod19`) is DEAD at ep39**
  (pid 72377 gone; run.log mtime 13:24Z). Best d_seg **0.045805 @ ep25** (EMA, [macOS-CPU advisory]).
  Stopped for the C0′ relaunch (PoseBlindComputeGate + full telemetry).
- **The C0′ dry-start-3 FAILED at boot, rc=8 in 0.5s** (`levelset_n600_drystart3_c0prime_20260715`):
  admission guard `DSL COMPILE REFUSED: missing TAC_DSL_COMPILE_HASH / provenance / launch.sh custody`.
  **Root cause (verified):** the DSL-compile-hash enforcement landed at `fa5a671330` (13:59Z) but the
  launcher's `build_launch_sh` did not yet emit the three `TAC_DSL_*` exports when the dry-start's
  launch.sh was generated (14:32Z). **The fix already exists as a sister agent's UNCOMMITTED working-tree
  edit to `tools/launch_witness_run.py`** (+285/−95: `dsl_compile_hash` threaded through
  `_identity_header`/`build_launch_sh`/`write_launch_sh` + `write_dsl_bound_launch`). Owed: land that
  edit, re-run the dry-start, then C0′ relaunch. **No fresh full-config telemetry exists from dry-start-3**
  — the C0 ep1–39 telemetry (below) is the freshest real-config data.
- The sealed component-timer ticket launch also crashed at boot
  (`--muon-start-epoch (726) must be in [1, --epochs (4)]` — the sealed 3000-ep validators refuse a
  4-epoch budget); the usable n24 instrumented data lives in
  `experiments/results/throughput_component_timer_async/ce_only_20260713/dry_start/` (4 epochs, GREEN).

---

## §A Missing-signal sweep (top items)

Full trails: costate duty queue (91 owed), `lever_registry.completeness()` (330 mapped / 71 unmapped),
deferral ledger (114 open), activation ledger (20 built-never-fired), July memos.

| # | Missing signal | Where it lives | Why missing | Expected value |
|---|---|---|---|---|
| 1 | **Grad-clip SATURATION** (NEW, this audit) | C0 `grad_clip_activation` rows ep1–39: global `frac_clipped=1.0` every accum step, `norm_mean≈5.9–6.2`, max 17.5 vs threshold **0.5** | telemetry existed; nobody read it against the constant | effective step = lr·0.5/gnorm ≈ **12× below the scheduled LR** for the whole CE stage; the LR cosine is not controlling the descent clock. Epochs-to-target lever; §B-4 law |
| 2 | **Component-timer coverage gap** (NEW) | C0 `witness_component_wallclock` rows: 4 timed components ×600 ≈ **94 s/ep** vs epoch_total **250–333 s/ep** → **~63% of epoch wall UNATTRIBUTED** (loss-term stack, optimizer, mx.eval sync, staging) | the #480 `_measure_component_decomposition` producer is COMMITTED in `tac.witness_control.telemetry_producers` but **trainer wiring DEFERRED** | the instrument that unlocks every further burn-down; $0 wire-in |
| 3 | **Async-verdict submit BLOCKS ~900s** (NEW; falsifies "fully hidden") | C0 ep25: epoch wall **1152.8s** vs 250s steady (+903s) while verdict_s=2555.7s ran "async" | #306 (07-05) concluded "verdict fully async-hidden… no action" from window 5042s ≫ wall 2439s; C0 shows the submit path itself blocks ~35% of the verdict wall | −36 s/ep amortized (−14%) at cadence 25; profile the submit (EMA snapshot + byte-close blob on main thread?) |
| 4 | **`--verdict-live-gap-every`** | confound pass H2 (`P0_campaign_queue_20260715.md`): "observability = score-neutral → default-off IS the orphan bug"; ABSENT from C0 + C1 argv | default-off orphan | distinguishes EMA-lag from real regression on early rows; default-ON next launch |
| 5 | **`--mod-dim-dynamics` (D18 k90 rate sensor)** | deferral D18 claims default-ON; **absent from both configs' argv + unmapped in lever registry** | orphaned wire | free rate signal (k90 → `--ks` truncate-at-export A/B); blocked also on NO FINAL CKPT |
| 6 | **S_R `margin_saliency_reachability`** | sole scientific delta of C1 leg-A; activation ledger: never `fired` (PREPARED_NOT_FIRED) | Phase-2 gated on C0 convergence | replaces the measured-INERT texture proxy (L76) with exact through-R reachability |
| 7 | **ISO arms never fired** | taper-off (**78.9% duty**), horizon (**47.3%**), step β_end=8.0 (**34.2%**) — BUILD_COMPLETE, dry-run PASS | deliberately isolated out of the live core, gated on C0 | the three highest-duty measurements in the queue |
| 8 | **#341 terminal head GN solve** | solve_dont_train inventory row 1: head near-quadratic CONFIRMED (ρ 0.847/0.868), full-P in-trainer GPU solve = the GO; NOT landed | build owed | replaces tail epochs with a ~3h solve (joint-objective win) |
| 9 | **#288 per-class OT offsets** | `laguerre_logit_offset.py:177` — BUILT, <1s, byte-free, **UNWIRED/orphaned** | wire owed | free d_seg offset at decode |
| 10 | **Outside signal (corpus-gap confirmed)** | AutoClip percentile clipping ([arXiv:2007.14469](https://arxiv.org/abs/2007.14469)), ZClip z-score spike variant ([arXiv:2504.02507](https://arxiv.org/pdf/2504.02507)); INR loss **spectral pre-conditioning** (filtered LS, cond ↓1–3 orders, [arXiv:2504.13390](https://arxiv.org/pdf/2504.13390)); structure-guided Gauss-Newton linear/nonlinear split ([arXiv:2404.05064](https://arxiv.org/abs/2404.05064)) — warm-start confirmation of #341/#342 + L77, per PAPER_WARM_START clause | not in papers_checked ledger | feeds §B-4 clip law + a pre-conditioning ticket adjacent to FreSh #448 |

**Registry hygiene (secondary):** 71 trainer flags unmapped (no DSL Lever owner) incl. Muon internals,
`--per-group-grad-clip`, eikonal-viscosity family; `--seg-chroma-boundary-start-event` /
`--lane-band-start-event` consumed-but-unmapped (registry blind spot, not config gap).

**Correction to a sweep-agent claim (verified against C0):** the #315 start-events ARE live in
`v9_cgauge_ideal_mod19` — the compiled argv carries `--muon-start-event powerlaw_meat
--lane-band-start-event lane_nucleus --seg-chroma-boundary-start-event annulus_plateau
--tau-advance-mode event`, and C0's run.log shows `start_event_fired: lane_band via lane_nucleus @ ep33`
(cap 500 was the backstop, not reached). The remaining naked-epoch constants are
`--seg-phase-advect-start-epoch 726` (N7 event hook BUILD-OWED) and the backstop caps themselves.

---

## §B DE/geometry: the config as state-dependent LAWS (workstream B)

The system is ONE modelable object (L10 unified action S_τ read in the frozen-scorer pullback Fisher
metric; Fisher↔margin Pearson 0.978). Constants are **continuation parameters whose optima MOVE along
the path** (`curriculum_is_continuation_instabilities_are_bifurcations_20260714`). Laws, each with its
derivation anchor and the scalar it replaces:

1. **Adaptive-ε CFL edge-tracking (#318/#320, BUILT, dormant).** Eikonal flow linearized symbol
   σ(k) = −k_n² − c_a·k_T², c_a=(|∇m|−1)/|∇m|; flat-margin annulus → backward heat (ill-posed) unless
   `ε ≥ |c_a|·√(η·λ_eik/8)`. Law: `ε(t)=clamp(|c_a|·√(ηλ_eik/8)(1+margin), floor, upper)`
   (`adaptive_eps_cfl_edge_tracking_v1`; trainer `--eikonal-viscosity-adaptive`, default OFF, in NO
   live config). Replaces: fixed ε + `--eikonal-viscosity-anneal` clock. The ep110 runaway = the
   measured bifurcation this law prevents.
2. **Event-driven LR (BUILT, unwired).** With `--tau-advance-mode event` the τ ladder advances by
   per-rung powerlaw_meat sensors, but LR still anneals on the fixed `--lr-anneal-epochs 1000` clock
   (a v6.4 control-reproduction pin). Law: η ∝ octave fraction k/N
   (`TauAdvanceController.lr_anneal_fraction`) — LR follows the continuation parameter, not the wall
   clock ("a clock cannot slow itself", Ch.6 critical-slowing). Replaces: `lr_anneal_epochs=1000`,
   `lr_hold_frac=1.0`.
3. **β co-anneal on the rung fraction (BUILT, unwired).** τ and β share ONE Γ-limit (Ch.4 Deriv-3);
   live config runs β on a linear 3000-ep clock (β_end 3.177 = slope-preserving rephase of the control,
   explicitly contested by the step_iso arm's 8.0). Law: `TauAdvanceController.hosc_beta_for_epoch`
   on k/N. Replaces: `hosc_beta_end`, `hosc_beta_anneal=linear`.
4. **Clip → percentile/trust-region law (NOT built).** Measured saturation (§A-1) means the constant
   0.5 re-parametrizes descent into normalized flow with effective step lr·0.5/gnorm. Cheap law:
   `clip_t = percentile_p(gnorm history)` (AutoClip); principled law: the #500 categorical-Fisher
   trust region `Δθᵀ JᵀFJ Δθ ≤ 2δ` (winner-rival bound `|t|≤√(8δ_KL/C_wr)` per the signal-loss-audit
   correction) which subsumes clipping. Replaces: `--grad-clip 0.5` (+ the silent LR re-scale).
5. **Verdict cadence as measurement law (sensors BUILT, consumer owed).** One n600 CPU verdict =
   2555.7s. Fire a verdict when the #344 NCDE-predicted |Δd_seg| since the last verdict exceeds the
   detection floor (EventBackstopGate pattern; cadence-25 stays as backstop cap). Replaces:
   `--eval-every 25` as primary.
6. **λ_pose costate law (registered, no consumer).** Exact `λ_pose = 5/√(10·d_pose)`
   (`costate_lambda_marginal_ds_v1`); live scalar `--w-pose 1.0` is correct at exactly one d_pose;
   binds at the terminal pose-finish engage.
7. **Phase-advect start as label_floor event (N7 BUILD-OWED).** The constants manifest itself labels
   `seg_phase_advect_start_epoch=726` a STATIC APPROXIMATION of the detector event.
8. **Island-birth at the computed fold point (derivation owed).** Ladder constants
   (movable r0 0.2252/birth 60/anneal 200; lane r0 2.0/80/260) are hand-placed; the continuation
   reframe says compute the critical dilation-λ of the class-occupancy saddle-node from a reduced-order
   model (#318 stability + #344 NCDE + #180 Morse-Smale) and place birth AT it.
9. **Stage-dependent averaging (BUILT, default-off).** EMA 0.997 is an ancestor-vehicle anchor (L18);
   the Polyak-finisher law (uniform tail mean over ~0.2·stage-window at the constant-τ* turnpike)
   is already compiled in (`polyak_finisher_start_epoch=2546`) but `--polyak-finisher-arm` is off.

Typed carriers for all of these: `src/tac/witness_dsl/adaptivization_tickets_20260715.py`
(`AdaptivizationTicketQueue`, research-only, data-only — tickets, never hand flags).

---

## §C Constants classification (constants-are-poison audit)

### C.1 The 13 LawRef-compiled manifest constants (`constants_manifest.json`, v9_cgauge_ideal_mod19)

| Constant | Value | Class | Note |
|---|---|---|---|
| `softmax_temp_end` | 0.31 | **PROVEN-measured** (with caveat) | P-TAU2 knee anchor (band [0.191,0.543]); measured on the mod32cap schedule — endpoint OK, the PATH is already adaptivized by event-mode rungs |
| `hosc_beta_end` | 3.177 | **ADAPTIVIZATION-QUEUED** | control-preservation rephase, not an optimum; contested by step_iso 8.0; law §B-3 |
| `lr_anneal_epochs` | 1000 | **ADAPTIVIZATION-QUEUED (POISON)** | control-reproduction pin; mismatched to event-mode τ; law §B-2 built |
| `lr_hold_frac` | 1.0 | **ADAPTIVIZATION-QUEUED** | rides §B-2 |
| `polyak_finisher_start_epoch` | 2546 | **DERIVED-as-law** | window fencepost formula; `tail_frac 0.2` is a band constant [0.1,0.3] |
| `seg_phase_advect_start_epoch` | 726 | **ADAPTIVIZATION-QUEUED** | self-declared static approximation; N7 event hook owed |
| `dseg_aware_taper` (4: on/strength/scale/floor) | 1/1.0/0.0/0.05 | **PROVE-via-ISO** | taper-off arm (78.9% duty) is the registered measurement; strength/floor never swept |
| `seg_margin_satisfice_msafe` | 0.03918 | **DERIVED-live** | measured ΔR p95 × headroom 2.0 (headroom = smallest covering integer) |
| `eikonal_retention_tau_rung` | 0.01→0.05 | **DERIVED-as-law** | rung-coupled (`eikonal_retention_couples_to_tau_rung_v1`); the viscosity ε partner is the dormant §B-1 law |
| `margin_saliency_reachability` | false | **QUEUED-treatment** | PREPARED_NOT_FIRED; C1 leg-A |

### C.2 Key sealed argv scalars (beyond the manifest)

- **POISON (measured):** `--grad-clip 0.5` (§A-1 saturation — silently caps effective LR ~12×, CE-stage
  measured, never re-validated at l7/Muon/finisher). Corollary: with clipping saturated, the ~20
  loss-term WEIGHTS only set the descent DIRECTION mixture, not magnitude — every weight-sweep verdict
  taken pre-clip-fix conflates the two.
- **POISON-suspect (ancestor anchors, never witness-re-measured):** `--ema-decay 0.997` (Quantizr, L18),
  Muon internals `--muon-lr 0.002 / --muon-momentum 0.95 / --muon-ns-steps 5` (PR95 lineage, unmapped in
  registry), `--adam-beta2 0.999` (beta2-window law exists; re-treatment named in the run purpose).
- **DERIVED (keep):** `--w-seg 100` (exact costate λ_seg), rate/pose exact multipliers, mod-dim 19
  (Whitney law, GATED-ON-HARVEST vs 32 via #299 arm), `--muon-start-event powerlaw_meat` + lane/chroma/
  screw events + `--tau-advance-mode event` (the #315 sensors — LIVE, verified fired ep33).
- **UNVALIDATED-cross-stage (queue):** `--accum-pairs 8` (75 steps/ep fixed across all stages; joint
  sweep with clip/LR owed), `--w-pose 1.0` (§B-6), persistence-warmup 275 / min-stage 250 (clock
  constants adjacent to event sensors), ladder island-birth constants (§B-8), `--eval-every 25` (§B-5),
  `--verdict-batch 32` (vb=64 measured never-slower — free), `--ckpt-every 25`.

**Counts:** 13 manifest constants → 4 PROVEN/DERIVED-keep · 2 DERIVED-live/law · 7 ADAPTIVIZATION-QUEUED
(incl. taper 4-tuple as one). Sealed argv extras → 1 measured-POISON (grad-clip) · 4 ancestor-suspect ·
~8 cross-stage-unvalidated queued. Every QUEUED row has a typed ticket in
`adaptivization_tickets_20260715.py` with law + built-implementation + unlock.

---

## §D Hotspot ledger + burn-down (workstream D, #509)

### D.1 Measured sec/epoch (all [macOS-MLX advisory] NON-PROMOTABLE)

| Surface | s/ep | Source |
|---|---|---|
| seg-only bench closure (not a run) | 43.3 | #306 (07-02 anchor) |
| v4/mod-19 CE stage (07-05) | 169.7 | #306 |
| v4 tau stage (ep300+) | 217–227.6 | #306 |
| throughput_fresh_eyes real n600 (07-13): v7.5.2 / V9 | 295.6 / 283.0 | fresh_eyes memo |
| **C0 live config ep1–32 (07-15)** | **248–259 (median 251.6)** | C0 `witness_component_wallclock` |
| **C0 ep33+ (lane_band event fired @33)** | **325–333 (+75, +30%)** | C0 telemetry |
| C0 verdict epoch (ep25) | 1152.8 (+903 blocked) | C0 telemetry |
| Gate anchor / refuse ceiling | 208.2 (3.47 min/ep) / 239 (3.99) | `scorer_throughput_gate.py` |
| n24 timer (CE-equivalent, async arm) | 72.7–76.9 (n24!) | `throughput_component_timer_async/ce_only_20260713` |

**Joint-objective projection at the live config:** post-ep33 steady 325 s/ep + 36 s/ep verdict
amortization ≈ **361 s/ep ⇒ 3000 ep ≈ 12.5 days** — above the compiled 8.314-day budget and above the
gate's own 3.99 min/ep refuse ceiling. The budget was anchored pre-lane-band. Without burn-down, the
sealed schedule overruns ~50%.

### D.2 Component decomposition (C0, per-pair probes ×600)

teacher_fwd 12.9 + teacher_bwd 30.8 + witness_fwd 13.9 + witness_bwd 36.2 + realized_R 0.4 ≈ **94 s/ep**
of 251.6 (ep<33). **~157 s/ep (63%) unattributed** (loss-term stack incl. persistence/clDice(iters=5)/
island_amplify/margin_saliency/subpix/satisfice/area/weight_entropy, optimizer + clip per 75 accum
steps, mx.eval graph sync, staging, python loop). At n24 the same 4 components ≈ 3.6s of 74.7s — the
unattributed cost has a large weakly-n-dependent component. **The #480 in-loop decomposition producer
(committed, trainer-wiring DEFERRED) is the instrument owed.** Known lever deltas: lane_band +75 s/ep
(C0); tau-stage group +47 s/ep (#306); pose-carrier **−0.71 s/ep (speed-saver)**; verdict 2555.7s per
firing (2189s mean in #306).

### D.3 Ranked burn-down (JOINT objective: Δ(epochs-to-target × sec/ep))

1. **Wire #480 component decomposition into the trainer + one n24 rerun** — $0, unlocks attribution of
   the 63%; nothing else can be ranked honestly without it. (sec/ep instrument)
2. **Profile + fix the async-verdict submit block** (+903s/verdict ⇒ −36 s/ep at cadence 25, −10%
   immediately; then the §B-5 cadence law compounds it). Falsifies #306's "no action" row — supersede it.
3. **Lane-band render-band cost** (+75 s/ep from ep33 ⇒ binds ~99% of the run): the dash-forward band
   geometry from GT ξ is per-pair-static — precompute/cache (same class as `--cache-gt-skeleton`,
   already live). Target −40–60 s/ep.
4. **Grad-clip law (§B-4)** — 0 sec/ep cost, epochs-to-target lever: unclamp the effective LR the
   schedule thinks it is applying (measured 12× gap). n24 A/B on gradient-quality + flicker telemetry
   per the relaxed-identity directive.
5. **Event-mode LR + β co-anneal (§B-2/3)** — built controllers, wire-in only; aligns the descent clock
   with the rung ladder (epochs-to-target).
6. **#341 terminal head GN solve** — replaces tail epochs with a ~3h solve; head quadraticity CONFIRMED
   (L77); the structure-guided GN paper (2404.05064) is the outside warm-start confirmation.
7. **verdict-batch 64** — measured never-slower, free flag flip at next compile.
8. **FreSh #448** — HELD (no Metal device in managed session); epochs-to-target candidate, no number yet;
   pair with the spectral pre-conditioning ticket (2504.13390).
9. **micro-batch B>1 — DOWNGRADED:** newest #447 Metal receipt measures 1.07×/1.0× (bit-identical
   survivor) and fresh_eyes B2 1.036× — the old 2–4× projection is superseded; also blocked by the
   wa-island/subpix/S_R batched consumers. Keep as ticket, not a priority.
10. Measured LOSERS (do not revisit without new receipts): custom Metal conv (#478, 0.65–0.97×),
    whole-step megakernel (#356, CPU 0.79–0.83×/GPU 1.12–1.21×).

### D.4 Contradiction to resolve (carried from the sweep)

Backward-share: #455 diagnostic says backward/costate-VJP = 82% per pair; the 95%-kill campaign assumed
forward = 78%. C0 in-loop probes say teacher_bwd ≈ 2.4× teacher_fwd, witness_bwd ≈ 2.6× witness_fwd
(bwd ≈ 70% of the timed 4-component slice) — closer to the 82% row, but the 63% unattributed gap owns
the answer. #480 wiring (D.3-1) is the resolver.

---

## §E Recommended launch-config delta (typed-diff PROPOSAL — NOT a launch)

For C0′/next compile, in three risk tiers (all through DSL Lever factories / typed configs, never hand
flags; every score-affecting item needs its own A/B receipt before adoption):

**Tier 0 — score-neutral, adopt at next compile:**
- `--component-wallclock-telemetry` + probe-every 1 (already in C1) **+ the #480 trainer wire-in**;
- `--verdict-live-gap-every` default-ON (confound H2);
- `--verdict-batch 64` (measured never-slower);
- land the sister's launcher fix → dry-start green → C0′.

**Tier 1 — bounded n24 A/Bs first (gradient-quality + flicker gate, relaxed-identity directive):**
- grad-clip percentile law vs 0.5 (ticket 1);
- event-mode LR (`lr_anneal_fraction`) vs 1000-clock (ticket 2);
- β co-anneal on rung fraction vs linear clock (ticket 5);
- adaptive-ε ON vs OFF (ticket 6; watch the floor-clamp inertness flag).

**Tier 2 — Phase-2 (gated on C0′ convergence, already registered):**
- S_R treatment (C1 leg-A), taper-off / horizon / step ISO arms (78.9/47.3/34.2% duty);
- `--seg-phase-advect-start-event` (N7 build) + `--polyak-finisher-arm`;
- #341 terminal head solve build.

---

## STORES CONSULTED

C0 + dry-start-3 + n24 timer run dirs (telemetry JSONL, run.log, constants_manifest, launch.sh) ·
`spec_v9_cgauge.py` (compiled argv, 224 flags) · `spec_c1_optimal_form_20260715.py` ·
`spec_throughput_component_timer_20260713.py` · `tools/launch_witness_run.py` (+ uncommitted sister
diff) · `src/tac/admission_guard.py` / `v9_provenance_gates` · costate_digest (91-item duty queue) ·
`lever_registry.completeness()` · deferral ledger (D15/D17/D18/D25/D44) · activation ledger ·
canonical equations registry · #306 per-lever audit + per_epoch_detailed_accounting ·
throughput_fresh_eyes + frontier_math + authority-ladder (#494) + nogo audit (#465; **#489 NOT FOUND —
likely a typo for #490**) · solve_dont_train inventory (#342) · FreSh memo (#448) · Metal conv (#478) ·
JEPA (#485) · memory mine (#295) / fp16 cf-feats (#296, unbuilt) · scorer_throughput_gate.py (3.59 →
superseded by 3.47/3.99) · L10/#318/#344/#315/#500 law modules · doctrine memories · online: AutoClip
2007.14469, ZClip 2504.02507, INR pre-conditioning 2504.13390, SgGN 2404.05064.

**verdict_scope:** measurement/audit rows are INSTANCE-level on this host/config family
([macOS-MLX advisory], NON-PROMOTABLE); the grad-clip saturation finding is FORMULATION-level for the
live config family (every v9_cgauge_* shares `--grad-clip 0.5`). Nothing here is a score claim;
pointer UNMOVED.
