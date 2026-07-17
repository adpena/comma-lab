# Arm B — forces + triggers build (SPEC_v10 §13.1 row 1 / §13.2 / §13.3) — 2026-07-17

Branch `p0_build_forces_triggers_20260717`. SSoT: SPEC_v10 §13 (branch
`claude/p0_521_spec_v10_capstone_20260717`). All levers DEFAULT-OFF; the OFF path is
behavior-identical to the incumbent (verified by the regression suites named per deliverable).
Pointer: **0.19108 UNMOVED** — everything here is MEANS (training-force/trigger apparatus for the
post-v9c2 boundary merge), no score claim.

## Deliverables (each committed the moment its tests passed)

### D1 — event-fallback phase supervision force (`5eb842e0ca`)
- `tac.boundary_math.phase_primitives.event_fallback_ref_and_weight_numpy` — the memo-exact
  composer `t_ref := where(ref_active, advected_prev_tie, own_gt_tie)`,
  `weight := ann ∧ ground ∧ (ref_active ∨ own_active)` ("advect-where-persistent,
  target-where-born"). STATELESS — **NO per-island persistence hold** (FEED-lane-gain anti-scope
  honored: a hold would fight GT's genuine deaths).
- Trainer: `--seg-phase-advect-ref gt_advected_with_own_tie_fallback` (3rd choice; incumbent
  `gt_advected` branch preserved verbatim). Pair 0 gains own-tie coverage (incumbent: no-op).
  Telemetry: `seg_phase_advect` row gains `advected_px_total` / `fallback_px_total` /
  `fallback_px_per_frame` (0/0 in incumbent modes).
- DSL: `PhaseAdvectionConsistency(ref=...)` accepts + validates the new mode.
- MEASURED basis (FEED-lane-gain §4b, n600 T1 audit): T1 is birth-SILENT; **26.3% of candidate
  straddle px (601/frame; 354 lane-adjacent) receive NO T1 supervision**; the c2 phase stage left
  the churning lane subset phase-unsupervised at birth sites. Δd_seg of the force itself:
  **UNMEASURED — duty-to-measure at the c2 per-stage A/B** (SPEC_v10 §13.1 disposition).

### D2 — fire-on-crest pose-gate option (`c06145cc03`)
- `sigma_min_plateau.evaluate_crest` + `SigmaMinPlateauDetector(mode='crest'|'plateau')` +
  `crest_canary_suite` (+`synthetic_rise_then_decline_series`). Crest = a RESOLVED rising phase
  (rel-slope > band with stderr ≤ slope in a pre-crest window) then NON-RISING for `hysteresis`
  consecutive windows; one-sided noise guard (`slope + stderr ≤ band`) → DEGENERATE ⇒ banked-R1
  (the sealed fallback contract unchanged). Same de-noise/window/band machinery as the sealed
  plateau — no new constants (DERIVED-AT-CONFIG reuse).
- Trainer: `--pose-finish-engage-on sigma_min_crest` (choices extended;
  `_pose_gate_sigma_modes` dispatch; per-mode canary; per-mode disengaged-alarm wording).
  Observer path stays plateau-mode ⇒ incumbent observer rows unchanged.
- DSL: `PoseFinishConditioningGate(engage_mode=...)` (default `sigma_min_plateau` unchanged).
- **MEASURED live anchor [live c2 run 20260717T113932Z, advisory]**: smoothed σ_min ep786→810 =
  0.0010→0.0064→0.0064→**0.0097 (peak ~ep802)**→0.0057→0.0034; latest rel-slope −0.1246/ep,
  stderr 0.0187 (≈6.6σ decline). The plateau detector correctly NEVER fired (consecutive_flat
  0/3) — a crest-then-decline trajectory never presents "flat"; the crest IS the event class this
  run needed. Locked as a test (`test_live_c2_crest_hysteresis_discipline_then_fire`: measured 6
  points = hysteresis-disciplined no-fire; measured-decline-extrapolated (labeled) = fire; plateau
  never fires on the shape).

### D3 — event couplings + the fire=unavailable observer fix (`248776749b`)
- (a) **β-anneal-complete → pose-finish-eligible**:
  `--pose-finish-eligible-on-beta-anneal-complete` (BooleanOptionalAction, default False) — with
  the coupling ON, NO engage signal (muon / σ_min gate / backstop) may engage pose-finish before
  `ep >= (--anneal-epochs or --epochs)`; deferral is LOUD once (`pose_finish_coupling_deferred`
  row). Fail-loud inert-arm guard (requires the two-phase arm). DSL:
  `PoseFinishBetaAnnealCoupling()`. Replaces the c2 `anneal-epochs(1000) ==
  pose-finish-start(1000)` two-constants coincidence with the event it encodes.
  **MEASURED sharpening [live c2, advisory]: the σ_min crest landed ~ep802 — BEFORE the ep1000
  constant — so the constant eligibility epoch is measured-SUBOPTIMAL on this run; eligibility
  must be event-derived too (exactly this coupling's direction).**
- (b) **per-force event entry `ncde_dseg`** (#344 consumer):
  `event_wirings.ncde_dseg_event(verdict_rows)` — 1-state LINEAR-d_seg chart (the exponential
  approach obeys the EXACTLY-linear ODE dx/dt = a·x + c; a log chart makes it nonlinear and the
  r2 guard refuses — MEASURED during build: clean saturating series r2 0.0006 in log space vs
  0.92–1.0 linear), LEVEL-NORMALIZED window (the fit ridge is calibrated for O(1) states; raw
  O(1e-3) d_seg had its coefficient crushed → false NO-FIRE, measured), CADENCE-SCALED #315
  hand-off criterion (plateau_slope_eps is per-EPOCH; the NCDE steps in dt = the verdict cadence
  ~25 ep ⇒ per-step threshold = eps·dt — a DERIVED unit conversion, not a new constant).
  Fires on BASIN (remaining descent spent) or HANDOFF (slope-flatten predicted); NO-FAKE fit
  guard (unstable/low-r2 never fires); fail-safe to the backstop cap. Trainer:
  `--seg-phase-advect-start-event ncde_dseg` dispatch beside `label_floor`, reading the SAME
  trainer-own verdict stream (poison-law). DSL: `PhaseAdvectionConsistency(start_event=...)`.
  λ-critical entry: **NOT BUILT** (honest scope) — the trainer has no in-stream λ-critical
  telemetry; the Road-Lane critical-λ prior lives in the offline shadow controller
  (`_event_advisories.morse_smale`). Routing: a λ-critical sensor needs a per-class-λ stream
  emitted by the trainer first (compose with #433 per-class-λ work) — open item.
- (c) **fire=unavailable ROOT CAUSE (MEASURED)**: on the live c2 run
  `build_verdict_path` found < 8 usable `stage:"verdict"` rows (6–7 at 25-ep cadence since
  warm-start) ⇒ returned None ⇒ `run_probe` silently OMITTED `verdict_latest_advisory` ⇒
  `shadow_controller._event_advisories` stored `ncde_344=None` ⇒ `costate_digest` printed the
  diagnostic-free `#344 fire=False reason=unavailable`. FIX (observer-side only, score-neutral):
  `run_probe` now ALWAYS emits a structured `verdict_latest_advisory`
  (`{"available": False, "fire": False, "reason": <measured cause>}` for sparse/degenerate;
  `available: True` on a real advisory) and the shadow controller never stores None. Verified
  read-only against the live run: reason now reads
  `insufficient verdict rows for the NCDE path: 7 usable 'verdict' rows < 8`.

### D4 — w_pose(t) = 5/√(10·d_pose(t)) derived-weight law (`e3ae9bfdc7`)
- `tac.canonical_equations.w_pose_marginal_weight_law_20260717` (eq
  `w_pose_marginal_weight_law_v1`): the weight IS the score's own pose marginal dS/dd_pose
  (DERIVED; finite-difference-verified in tests). **THE CLAMP IS DERIVED**: the marginal diverges
  as d_pose→0; the seg marginal is the constant dS/dd_seg = 100; they cross at d_pose =
  25/(10·100²) = 2.5e-4 ⇒ clamp = 100.0 — the pose weight never exceeds the score's own seg
  exchange rate (sister of the `--pose-grad-coeff-max` divergence guard — same singularity).
- Trainer: `--w-pose-marginal-law` + `--w-pose-marginal-clamp` (default 100.0, DERIVED);
  consumption ONLY at the pose-finish stage (`_w_pose_now` holder); updates at VERDICT cadence
  when a measured d_pose lands (piecewise-constant, never per-step — SPEC_v75 §8
  weights-at-boundaries honored at the measurement boundary); static `--w-pose` fallback until
  the first measured d_pose; fail-loud inert-arm guard; `w_pose_marginal_law` telemetry row on
  each weight change. DSL: `PoseMarginalWeightLaw(clamp=None→derived)` with LawRef custody on
  the clamp (`LADDER_DERIVED_AT_CONFIG`).
- Physics check at real operating points (tests): live-run start d_pose 112 ⇒ w=0.149 (weak
  force early — correct); R1 1.6e-3 ⇒ w≈39.5; ancestor 3.4e-5 ⇒ clamped at 100.

## Value-provenance labels
- MEASURED: the 26.3%/601/354 coverage gap (FEED-lane-gain n600 audit) · live-c2 σ_min crest
  series + plateau-no-fire (coordinator-relayed run.log rows, advisory) · live-c2 7-verdict-row
  probe root cause (re-run read-only in this build) · the log-chart r2 failure + ridge-crush
  false-NO-FIRE (this build's smoke) · ep725 dual-metric phase_advect antagonism (Arm A harness,
  n96 advisory; routing note below).
- DERIVED: w_pose marginal + crossover clamp (score algebra) · crest guard/one-sided bound (flat
  band reuse) · cadence-scaling of plateau_slope_eps (unit conversion) · crest min_points
  (+1 pre-crest window).
- INFERRED: "seg descent eroding pose-legible photometrics" as the crest-decline mechanism
  (coordinator's reading of EMA d_seg 0.004236→0.004164 coinciding with the σ_min decline) —
  consistent, not yet isolated.
- ASSUMED: the anneal-schedule length (`--anneal-epochs` or `--epochs`) is the right
  "β-anneal-complete" epoch even when hosc-β anneal is inactive (the τ anneal shares the same
  denominator; documented in the flag help).

## Composition / antagonism vs existing levers
- D1 composes with the seg-phase-advect family (Force-1 A_ξ ∘ Force-3 tie): it ONLY adds
  supervision on the uncovered complement (adv ∧ fb disjoint by construction — tested); the
  transport channel is verbatim. Antagonism risk: own-tie targets at fast-moved (non-birth)
  sites double-count Force-3's within-pair target where both are active — same target field
  (`gt_tie_targets_numpy`), so it is a WEIGHT increase not a conflict; watch the term-domination
  guard (≤10% of total) at the A/B.
- D2/D3a compose: gate (ENGAGE signal) × coupling (ELIGIBILITY floor) are orthogonal conjuncts;
  banked-R1 fallback contract preserved on every path.
- D3b vs `label_floor`: two sensors on the same stream, dispatch-exclusive per run; ncde_dseg is
  regime-generic (any stage) while label_floor is stage-conditioned — the c2 A/B should stagger
  them (SPEC §13.2 staggered engagement for attribution).
- D4 under any engage mode (muon/plateau/crest); interacts with `--pose-grad-coeff-max` (both cap
  the same divergence; the law caps the WEIGHT at 100 while coeff-max caps the score-domain
  gradient coefficient — compatible, law is upstream).

## ROUTED (per coordinator, measured anchor recorded — NOT built)
**Phase-weight relaxation event coupling**: at live-c2 BEST ep725 (Arm A dual-metric harness,
n96 advisory) phase_advect vs the armed seg base reads Euclid cos −0.149 / Fisher cos −0.118,
rel-norm 0.63/0.48 — a LARGE (~half the seg force) mildly ANTAGONISTIC term post-settlement,
vs +0.238 aligned at ep701: the force flips sign once boundaries reposition. With the σ_min
erosion under resumed seg descent, the constant `--seg-phase-advect-weight 0.4` is a LATE-PHASE
ANNEAL / EVENT-GATE candidate: hold during conditioning build, RELAX on a measured event (seg
slope resumption OR the D2 crest firing). NOT wired here because `pa_w` is closure-captured in
the compiled loss path (a per-epoch weight holder there is a deeper change than this arm's
scope; half-wiring under resume/compile risk is forbidden). Routes to the boundary A/B set
(SPEC §13.5) + the c2 phase-stage config owner.

## Round-1 adversarial self-review (attack own build)
1. **Binding (counted-but-inert audit)**: every lever verified to BIND when enabled — D1 changes
   the provider fields the loss consumes (channel counts tested disjoint + summing to the
   weight); D2 changes the engage decision path (dispatch tested + live-shape test); D3a forces
   the engage boolean off pre-anneal (deferral row); D3b changes `_pa_event_fired`; D4 writes
   `_w_pose_now["v"]` (the value every training-loss call reads). Inert-arm cases fail loud
   (D3a/D4 raise without the two-phase arm).
2. **OFF-path identity**: all four levers default-OFF; the restructured `_w_pose_now` assignment
   computes bit-equal values to the incumbent expression when the law is off; the incumbent
   `gt_advected` branch is texturally preserved; the observer detector stays plateau-mode.
   Full plateau (19), ncde (18), phase-primitives suites green.
3. **Found-and-fixed during the build** (would each have been a shipped false-negative):
   (i) log-chart NCDE fit r2≈0 on clean saturating d_seg (chart mismatch); (ii) ridge crushing
   O(1e-3) linear states (level normalization); (iii) per-epoch vs per-step hand-off threshold
   (cadence scaling). Each is now load-bearing code + test.
4. **Remaining attack surfaces (open, honest)**:
   - Detector `mode` is NOT persisted in the resume sidecar; a resume that flips
     plateau↔crest mid-run reuses the series under the new semantics (config-freshness gates own
     cross-resume config drift, but this is un-guarded at the detector level).
   - `ncde_dseg` BASIN can fire early in a stage when the trailing window still contains the
     steep entry descent (my n=12 synthetic fired BASIN with remaining 0.0016 — correct within
     the window, but window-relative "travelled" is stage-entry sensitive). The backstop cap +
     staggered A/B own this; a stage-boundary window reset would harden it.
   - D4 consumes whatever d_pose the verdict row carries (live vs EMA-shadow source varies by
     `d_pose_source`); the marginal is smooth so the error is second-order, but a
     source-pinned variant is the cleaner form.
   - Fully-flat σ_min AFTER a crest was latched is fine (monotone), but a crest that ends in a
     long degenerate-noise tail without ever meeting hysteresis ships banked-R1 — by contract,
     yet it means the ep802-class event can still be missed if noise spikes exactly at the peak;
     the densify amendment (gate-row cadence) is the mitigation.
5. **Triality**: DSL leg (4 new/amended Lever factories; registry `completeness()` holds all new
   flags) · equations leg (`w_pose_marginal_weight_law_v1` module; the crest/event-fallback/
   ncde-chart laws carry `# FORMALIZATION_PENDING:` rationale below) · DAG leg = this memo (a
   FEED block fold into `sub015_DAG_*` belongs to the post-v9c2 boundary-merge owner, with the
   arm-A/arm-C landings, to keep one consistent FEED).
   `# FORMALIZATION_PENDING:` crest-detector law + event-fallback coverage law + ncde
   linear-chart/cadence law are registered as code-with-tests here; their canonical-equation
   modules are owed with the first MEASURED Δd_seg / engage-epoch rows from the c2 A/B (a
   registration without an empirical anchor row would be a marker without a measurement).

## Duty-to-measure queue (all default-OFF ⇒ tracked, not orphaned)
1. c2 per-stage A/B: event-fallback Δd_seg (the §13.1 disposition) — HIGHEST.
2. Crest-vs-plateau engage-epoch A/B on the c2 σ_min stream (the ep802 anchor predicts crest
   engages ~200 ep earlier than the backstop).
3. ncde_dseg vs label_floor staggered entry attribution.
4. w_pose(t) law vs static w_pose at the pose-finish (d_pose descent rate + d_seg hold).
5. The ROUTED phase-weight relaxation (needs the pa_w holder wire-in first).
