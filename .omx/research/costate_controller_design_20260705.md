---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_dissent:
  - member: Contrarian
    verbatim: "the ep450 below-band horizon miss shows linear λ-extrapolation is already wrong at 25 epochs under decay; do not let Phase B actuate on horizon projections until a decay-aware λ(t) model is measured. Shadow rows yes; argv emission from projections, not yet."
council_assumption_adversary_verdict:
  - assumption: "aggregate verdict rows (d_seg, d_pose, bytes) are a sufficient state for useful costates"
    classification: CARGO-CULTED
    rationale: "per-class λ is the actually-wanted signal (residual is ~98% bulk, islands 1.1%); aggregate λ cannot see class routing. Mitigated: named as the #253/#255 WRAP gap + probe queue, never guessed."
    empirical_verification_status: ASSUMED_AWAITING_VERIFICATION
  - assumption: "the canonical monitor classifier's window=5 / min_sustained=3 thresholds transfer to the costate layer"
    classification: HARD-EARNED
    rationale: "defaults were tuned on the #205 trace itself (creep_eps vs measured +6e-6/ep steady creep); the backtest reproduces WATCH@325/EROSION@350 on that same trace — consistent, though circular on this one run."
    empirical_verification_status: VERIFIED_VIA_EMPIRICAL_ANCHOR
  - assumption: "independence of channel errors in the chained stderr sqrt(Σ(λ_i·se_i)²)"
    classification: CARGO-CULTED
    rationale: "channels share epochs and share the optimizer state; stated openly in the method string as a floor, not gospel."
    empirical_verification_status: INFERRED_FROM_DOMAIN_LITERATURE
council_decisions_recorded:
  - "op-routable 1: Phase A shadow controller LANDED (observe→estimate→recommend→STOP; actuation structurally impossible)"
  - "op-routable 2: Phase B actuation is DESIGN-ONLY here; gated on operator GO + the Contrarian's decay-model condition for horizon-based actions (rollback/stop actions are projection-free and exempt)"
  - "op-routable 3: per-class λ via #253/#255 attribution wiring = the top probe-queue item"
---

# θ* COSTATE CONTROLLER — DESIGN (task #303; Phase A built, Phase B design-only)

**Frame.** Memory `project_meta_layer_above_triality_hamiltonian_control_costate_20260703`:
the campaign triality {DAG=state x(t) · DSL=control u(t) · equations=law S} is the shadow
set of ONE controlled learning dynamics; the missing fourth object is the COSTATE
λ = ∂S/∂x — the measured marginal-ΔS shadow price that flows measurement → decision and
turns the DSL from a passive emitter into an active controller. The sibling curriculum
study (`council_grand_symposium_curriculum_derivation_20260705.md` §B.4) reaches the same
frame independently: a hybrid Bolza problem — continuous controls on a singular arc,
discrete controls as bang-bang impulses firing on switching-function (= costate) zero
crossings. This memo makes it concrete against the real telemetry.

## 1. State, control, costate (concrete)

**State x** (per verdict row + stage telemetry, all real trainer emissions):
`(d_seg, d_pose, blob_bytes, seg_form/stage, epoch, ep_loss, implied_S)` + stage
transitions + closed_loop rows + structured_init/island_seed nucleation facts + the
launch.sh lever vector (paint/seed-islands/eikonal/mod-dim/…). Per-class d_seg and
island metrics are IN the design state but not yet in the telemetry → per-class λ is a
named gap (probe queue), not an estimate.

**Control u** (the actuation vocabulary, Phase B): τ path (anneal shape/end), lr,
seed-anneal/containment mode, lever gates (focal γ, eikonal base/end/bump, persistence,
island-amplify), stage-boundary triggers (CE→tau, lane-band engage, l7, Muon switch),
early-stop/rollback. Today only TWO controls are closed-loop in-run (bounded eikonal
bump + early-stop arming — trainer build-3); everything else is open-loop launch config.

**Costate λ** — the honesty ladder (implemented in `tac.witness_control.costate_estimator`):

| Tier | What | Value (measured where shown) |
|------|------|------------------------------|
| ANALYTIC | λ_seg = ∂S/∂d_seg | 100 exactly (score law) |
| ANALYTIC | λ_pose = 5/√(10·d_pose) | state-dependent: 4.85 at live d_pose 0.106; 31.6 at d_pose 0.0025; → the CLAUDE.md pose/seg marginal crossover falls out of the same partial |
| ANALYTIC | λ_bytes = 25/37 545 489 | 6.659e-7 S/byte (≈ ±0.0007 S per ±1 KB — rate drift within-run is noise at current blob sizes) |
| MEASURED | dS/dep per stage = λ·dx/dep ± √Σ(λ_i se_i)² | windowed OLS over the last 5 same-stage n600 verdicts (λ(t) is time-varying; the full-stage fit over-weights the decayed onset transient) |
| MEASURED | stage-advance jump | first-post − last-pre verdict; noise floor = combined neighboring-fit residual std |
| MEASURED | rollback gain | S_latest − S_best, exact arithmetic on measured rows |
| PARTIAL | probe sweeps (focal γ → grad-share) | direction/ranking evidence; chain to S unmeasured |
| UNIDENTIFIABLE | everything else | honest refusal + the probe that would identify it |

**Key measured insight from the backtest:** the tau stage's cost is a SLOPE CHANGE, not
a jump — the ce→tau boundary jump itself measured −0.0011 S (benign; tau's first verdict
was still the run best) while the post-onset within-stage slope is the erosion
(+3.3e-3 S/ep at ep350 window, decaying to +2.4e-4 S/ep by ep450). The switching function
the sibling's Bolza frame wants is exactly this within-stage chained slope.

## 2. Identifiability (honest ledger)

**Estimable TODAY from existing telemetry:** the whole MEASURED tier above, per stage,
per run, with OLS stderr → every number in §3/§4.

**UNIDENTIFIABLE today (= the probe queue, ranked):**
1. **Per-class λ** (lane vs road vs movable marginal-ΔS): verdict rows are aggregate.
   Probe: wire `tools/erasure_timing_attribution.py` + `witness_per_stage_annulus_attribution.py`
   (#253/#255) to emit per-class rows the estimator ingests (WRAP decision, inventory §6).
2. **l7 costate**: no available run.log contains an l7 stage (the #205 log ends ~ep525 <
   l7@1000). The historical "l7 harmful" verdict is an ANCESTOR-vehicle lesson
   (`feedback_ancestor_vehicle_findings_are_lessons_not_transferable`) — the controller
   correctly refuses to emit a number. Probe: per-stage checkpoint diff via #253 when a
   run reaches l7 (the per-stage checkpoints are preserved by the non-negotiable).
3. **Muon costate**: same status (muon@726 never reached in available logs; the −32%
   d_seg figure is from the muon_deep_dive measurements on other vehicles/traces).
   Probe: same as (2) + the #270 restart A/B when it fires.
4. **Per-lever costates across the 13-change config diff** (#205 vs seed-fix): matched-epoch
   A/B is CONFOUNDED (13 simultaneous diffs) — `cross_run_lever_costate` returns
   UNIDENTIFIABLE with the joint effect and the diff list. The crutch(174257Z)↔fix(015247Z)
   pair is additionally non-comparable: the VERDICT SEMANTICS changed (seed leakage into
   the readout), so their d_seg values measure different observables. Probe: single-lever
   A/B pairs, cheapest-first by |expected ΔS|·(1/cost).
5. **τ-path / lr / seed-anneal continuous costates**: no logged variation to difference.
   Probe: γ-sweep-style calibration probes (the focal harness is the template) + the
   sibling's DERIVED τ-law as the nominal to perturb around.
6. **λ(t) decay model**: the ep450 below-band miss (predicted +0.0060 [0.0018,0.0103] vs
   realized +0.0004 over 25 ep) shows linear local extrapolation still overpredicts under
   deceleration. Probe: fit exponential-decay slope models on the existing tau trace ($0).

## 3. Backtest scorecard (the NO-FAKE validation; #205 log, as-of replay)

| As-of | Controller said | Measured truth | Verdict |
|-------|-----------------|----------------|---------|
| ep300 | PLATEAU (1 tau row; canonical-classifier single-row edge) → advance/stop, ΔS 0 | CE had converged 0.0103→0.00475; tau just fired | ✓ acceptable (cosmetic edge flagged) |
| ep325 | TRANSITION_TRANSIENT → WATCH_NO_ACTION | the spike was indistinguishable from a recoverable boundary transient at 1 post-onset verdict | ✓ REDISCOVERED tau-onset spike handling |
| ep350 | DIVERGING_ERASING → ROLLBACK (ΔS −0.1655) + STOP (ΔS −0.0825 [−0.155,−0.0097]) | sustained erosion confirmed; run in fact crept to ep525+ | ✓ REDISCOVERED the erosion + would have saved ~175+ epochs (~7 h) |
| ep350 horizon check | +0.0825 [0.0097,0.155] next-25ep creep | realized +0.0119 | ~ IN band, central 7× high (transient decay) — honest wide-band hit |
| ep450 | same classification; STOP ΔS −0.0060 [−0.0103,−0.0018] (windowed) | realized next-25ep +0.0004 | ✗ BELOW band — λ(t) decay + pose offset; recorded as equation residual |
| ep525 (full) | rollback +0.176 S recoverable; creep +4.9e-4 S/ep | matches the #205 diagnosis memo (net +40.4% over CE-best) | ✓ quantifies the known diagnosis |
| l7 / Muon | NO costate emitted (UNIDENTIFIABLE) | those stages never ran in this log | ✓ HONEST refusal (the required behavior, and a test asserts it) |
| Crutch run 174257Z (retro) | DIVERGING_ERASING → rollback −0.126, stop −0.046 [−0.078,−0.014] | d_seg frozen/rising 0.0287→0.0294 while ep_loss fell 410→321 — the crutch decoupling | ✓ REDISCOVERED the crutch diagnosis from telemetry alone |

**Rediscovered: 5/7. Honest misses: the ep450 horizon band (λ-decay, probe queued) and
l7/Muon (correctly unidentifiable — cannot be rediscovered from data that does not
contain them; citing the ancestor numbers instead would have been the fake).**

## 4. Live shadow (run 20260705T015247Z, read-only pass 2026-07-05)

State: ep75, CE, d_seg 0.1275, d_pose 0.1065, 73 820 bytes, implied_S 13.83 (best
0.1217@ep50). Classification CONVERGING (d_seg −6.9e-4/ep; note ep_loss ROSE at ep75 —
lever-engagement territory, not decoupling). Costates: λ_pose 4.85 (pose still matters
at this operating point), dS/dep[ce] −7.1e-2 [−1.78e-1, +3.6e-2] (n=3 — band honestly
wide, spans 0). Recommendation set: **CONTINUE_STAGE** (ΔS −1.78 [−4.45, +0.89] over
25 ep) — with the explicit caveat that at n=3 the band includes 0; next verdicts tighten
it. No rollback recommended (best is within noise of latest). Rows NOT written to the
live dir (containment: live run dir treated read-only; the JSONL write path was
demonstrated on the completed #205 dir).

## 5. Phase B — actuation DESIGN (design-only; operator GO required)

- **u\* selection:** `u* = argmax over READY levers of [expected ΔS·effect − cost]`
  subject to **NEVER-REGRESS (POWERPLAY)** — implemented in Phase A already at the
  recommendation layer (any candidate with central predicted ΔS ≥ 0 is refused by
  construction; `powerplay.variant_ii_accept` is the Phase-B acceptance form).
- **Actuation surface = DSL argv emission, never process control:** the controller emits
  a flag-diff compiled + validated through `tac.witness_dsl.curriculum_dsl.compile_trainer_argv`
  (de-orphaning the emission path — inventory §3). The artifact is a RECOMMENDED-CONFIG
  file; launch remains human/governed (`launch_witness_run` + `system_memory_governor`
  own admission). Per #216: never auto-fires heavy/paid GPU — structurally, the package
  still contains no process-spawning surface; emission is a file.
- **In-run actions:** the trainer's build-3 bounded loop stays the ONLY in-run actuator;
  Phase B extends its action set (e.g. rollback-to-best-and-retreat, lever-gate toggles)
  only via new default-OFF flags with byte-identity when OFF.
- **Reference trajectories (sibling drop-in):** Phase B consumes a
  `ReferenceTrajectory` input — {τ(t) nominal, stage-boundary triggers, expected d_seg
  envelope per stage} — exactly the shape of #302's derived laws (§B.4 singular arc +
  hand-off triggers). The controller then acts on DEVIATION from the reference
  (closed-loop correction), which is the Bolza decomposition the sibling derived. No
  gauge.py edits here; `ControllerGauge`/`CurriculumGauge` composition is the named
  follow-up after #302 fully lands.
- **Gating (binding):** (a) operator GO; (b) the Contrarian condition — horizon-projection
  actions (CONTINUE/STOP sizing) require the λ-decay model probe first; projection-FREE
  actions (rollback-to-best, WATCH) are exempt because they are measured arithmetic;
  (c) admission stays with the memory governor; (d) every emitted config carries the
  evidence chain of the costates that selected it (Rudin readback).
- **Nondimensionalization (π-groups, tao memo):** cross-stage lever comparison uses
  λ̂ = (dS/dep)·L_stage / S_current (stage-length-scaled, dimensionless), so a τ-stage
  creep and a Muon-stage drop are commensurable before ranking.

## 6. Observability surface

Inspectable: every costate carries {value, stderr, band, status, method, evidence, n}.
Decomposable: chain terms printed per channel (the method string is the decomposition).
Diff-able: `--as-of-epoch` replays any historical decision. Queryable: JSONL rows in
`costate_shadow.jsonl`. Cite-able: evidence = named verdict rows/epochs in a named
run.log. Counterfactual-able: as-of replay + the never-regress refusals are logged with
reasons.

## Canonical-vs-unique decision per layer

- Classifier: ADOPT_CANONICAL (`witness_control_monitor.classify_trajectory`, imported).
- Stage-row parsing: ADOPT_CANONICAL (`dashboard_control_telemetry.parse_stage_rows`).
- launch.sh flags: FORK_PRINCIPLED (verbatim replica + parity test — the canonical lives
  in a matplotlib-importing module unimportable from library code).
- Estimator: UNIQUE (no prior surface computes λ with propagated uncertainty).
- Actuation: ADOPT_CANONICAL by deferral (governor + trainer build-3 own it; Phase A has none).

Cross-refs: inventory memo (same date) · canonical equation `costate_lambda_marginal_ds_v1`
· sibling `council_grand_symposium_curriculum_derivation_20260705.md` · trainer build-3 ·
focal calibration memo. Axis: all numbers [macOS advisory] NON-PROMOTABLE, implied-S
units from n600 advisory verdicts; **pointer 0.19110 UNMOVED — this is MEANS; it moves
only when the controller's picks land a lower exact byte-closed row.**
