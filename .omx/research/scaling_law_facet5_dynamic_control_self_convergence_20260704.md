# Scaling-law engineering — FACET 5: the dynamic control system (closed-loop self-convergence)

**Task: FACET 5 of the 5-facet geometry-optimal scaling-law pass (2026-07-04).** The other four facets
make the scaling *statically optimal* (the right basis, capacity, mod-dim, schedule SHAPE). This facet
makes it **ACTIVE**: a closed-loop controller that steers the witness training to the optimum and
*knows when to stop* via a mathematical convergence certificate. **MEANS, not ends** — the pointer
(contest-CPU **0.19110**) moves only when a byte-closed `upstream/evaluate.py` n600 row beats it; this
memo designs the *instrument that decides what to run*, not a byte off the archive. `$0` research; no
heavy/paid/GPU; #205 SACRED READ-ONLY. Governing discipline: NO-FAKE (no fabricated number/citation),
CONTAINMENT (the controller EMITS decisions; operator/governor gates actuation), never-invent-flags
(every flag below grep-verified in `experiments/train_levelset_witness_realized_through_R_mlx.py`).

---

## 0. The one-paragraph thesis

Witness training is a **Bolza optimal-control problem with free terminal time**: state `θ` flows under
the τ-annealed gradient field, the *control* is the schedule `u = (τ, η, w_eik, stage-transition events,
seed injections)`, the *terminal cost* is the realized-through-R `d_seg`, and the secondary is
lexicographic training time. Pontryagin's principle says the optimal schedule is set by the **costate
`λ` = the sensitivity of terminal `d_seg` to the current state** — and we *already have* `λ` at two
scales: the cached through-R margin-Jacobian `S_R` (pixel scale) and the trajectory instrument's
per-stage descent-rate ranking (lever scale). The **only PROVEN** Lyapunov certificate in-tree is the
**OT-dual gap** of `damped_newton_ot_offsets` (a constraint-satisfaction certificate for the per-class
area inner solve); the **operational** early-termination trigger is an **EWMA descent-rate monotonicity
test** on the verdict stream whose *diverging* branch is *exactly* the MEASURED #205 τ-creep signature.
The build is a **de-orphaning unification**: wrap the existing `tools/render_witness_trajectory_dynamics.py`
(#188/#216) into a lean, deterministic, resumable **control monitor** that reads #205-style verdict logs,
classifies the trajectory (converging / plateau / diverging-ERASING / volatile), emits a control
decision + a grep-verified config diff, and **never launches** (the meta-layer costate-controller #247 /
the operator / the P0 system-memory governor gate the actuation).

---

## 1. Training as a controlled dynamical system (the precise frame)

**Full state** `x(t) = (θ(t), s(t), θ̄(t))`: decoder+per-pair-code weights `θ`, discrete stage index
`s ∈ {CE, τ, l7, Muon}`, EMA shadow `θ̄` (the inference/verdict state — the EMA non-negotiable). The
"~8-dim manifold state" (the lane-orbit / per-pair FiLM-code manifold, §OPERATOR-PRIORITY dim≈8) is the
**intrinsic reduced state** the dynamics live on; the **observable** used for control is the low-dim
**verdict/attribution vector**
`y_n = (d_seg, d_pose, implied_S, ep_loss, {per-class d_seg attribution})` emitted per eval. This is a
**POMDP**: `θ` is hidden (millions of params), `y_n` is the cheap observation — the controller acts on
`y`, not on `θ`.

**Plant (state dynamics).** Discrete: `θ_{n+1} = θ_n − η_n · P_n · ∇_θ L_{τ_n}(θ_n)`, where `P_n` is the
preconditioner (AdamW second-moment, or Muon Newton–Schulz orthogonalization in the finisher). Continuum
(the paper §3 viscosity level-set PDE): `dθ/dt = −∇_θ L_{τ(t)}(θ)`, a **temperature-annealed
mirror-descent / graduated-non-convexity continuation** along the single dequantization scalar `τ=ε=ħ`
(registered law `tau_eps_hbar_one_dequantization_two_scales_v1`).

**Control** `u(t) = (τ(t), η(t), w_eik(t), w_len(t); {t_τ, t_l7, t_Muon}; seed/lever injections)`.
Two kinds: **continuous** (τ, η, eikonal weight — the smooth knobs) and **impulsive/switching** (stage
transitions, seed injection, lever engage-epochs — the discrete events). This is a **hybrid control
system** (continuous flow + discrete jumps), which is why the *transitions* need their own treatment
(the "different stages need different treatment" non-negotiable = the hybrid-jump discipline).

**Objective (Bolza, free terminal time T).**
```
minimize   J[u,T] = Φ(θ̄(T))              (terminal cost = realized-through-R d_seg of the EMA shadow)
                  + ∫₀ᵀ c_run dt          (running cost = wall-clock; lexicographically SECONDARY)
subject to dθ/dt = −∇_θ L_{τ(t)}(θ),   θ(0)=θ₀,   hybrid jumps at {t_τ,t_l7,t_Muon}.
```
The lexicographic training-time rule (`feedback_training_time_lexicographic_secondary_...`) formalizes as:
**minimize T subject to Φ(θ̄(T)) = Φ\*** (the achievable terminal d_seg) — i.e. take every score-neutral
speed win free, never trade score for time. **Early termination = choosing T** (the free-terminal-time
degree of freedom). #205 fixed `T` (fixed `--epochs`, fixed stage epochs) and thereby *forfeited* this
control axis — the root of the tau-creep waste.

---

## 2. Pontryagin / the costate control law (tie to the meta-layer costate-controller)

**Hamiltonian.** `H(x, u, λ) = ⟨λ, f(x,u)⟩ − c_run`, with `f(x,u) = −∇_θ L_τ(θ)` the flow field and
`λ(t)` the **costate/adjoint** = `∂J/∂θ(t)` = the sensitivity of the *terminal* `d_seg` to the *current*
state. The costate flows **backward** with terminal condition
```
λ(T) = ∇_θ Φ(θ̄(T)) = ∇_θ d_seg_R(θ̄(T)),      dλ/dt = −∂H/∂θ = (∂²L/∂θ²) λ  (adjoint / VJP of the flow).
```

**This is the concrete identification the meta-layer memory names.** The memory
`project_meta_layer_above_triality_hamiltonian_control_costate` says: the triality (DAG=state x(t),
DSL=control u(t), equations=law S) is missing exactly ONE object — the **costate λ = measured
marginal-ΔS-per-lever** — and adding it turns the DSL from a *passive emitter* into an *active
controller* `u* = argmax_{ready levers} [expected ΔS-toward-target − cost]`. **That is discrete
Pontryagin verbatim:** `argmax_u H = argmax_u [⟨λ, f_u⟩ − cost_u]`, with `f_u` = the lever's effect on
the flow and `λ` = the marginal-ΔS field. The meta-layer's POWERPLAY / never-regress frontier-selection
IS the PMP maximum condition; the "de-orphaning" it calls for is precisely **wiring the costate sensors
into the control law**.

**What `λ` is HERE, concretely — at two scales (fractal, per the meta-layer):**

| scale | costate `λ` | already-built artifact | flag |
|---|---|---|---|
| **pixel / field** | the through-R fragility-weighted **margin-Jacobian `S_R`** = how much moving each boundary pixel's realized margin changes terminal `d_seg` (the reachability of the correct answer at the GT target) | `tools/precompute_sR_reachability.py` → `sR` key in the gt-cache | `--margin-saliency-reachability` (verified) |
| **lever / stage** | the **per-stage descent-rate** `improvement_per_epoch_dseg` (marginal Δd_seg per epoch per lever) + the cross-stage config-strength ranking | `tools/render_witness_trajectory_dynamics.py` `compute_dynamics` | (read-only sensor) |

The pixel-scale `λ = S_R` is the **terminal condition** `λ(T)=∇_θ d_seg` made cheap (cached, θ-slowly-
varying); the lever-scale `λ` is the **integrated** costate the discrete controller maximizes over. Both
are ALREADY in-tree — the costate is not a new build, it is an **un-consumed sensor** (the orphan
problem the meta-layer diagnoses).

**How the costate sets the schedule (the switching law).** PMP for a control with a running cost gives a
**bang-singular** structure: hold the current control while the marginal Hamiltonian gain of continuing
exceeds the running-cost rate, switch when it saturates. Concretely, the costate-projected flow velocity
= the descent rate `r̂ = d(d_seg)/dt`; the switching condition is `|r̂| ≤ c_run` (the descent no longer
"pays" its wall-clock) ⟹ **advance τ / trigger the next stage / terminate**. This is the exact adaptive
replacement for #205's *fixed* stage epochs: the stage boundary is a **measured switching surface**, not
a hardcoded epoch. (The singular arc = the τ-anneal itself; the bang = each stage jump.)

---

## 3. Lyapunov convergence certificate → EARLY TERMINATION

**Honest scope first.** There is **no single global Lyapunov function** proven for the whole non-convex
run — claiming one would be a convergence proof we do not have (NO-FAKE). What we have is a **three-tier
certificate**, exactly one tier PROVEN, and the operational tier is the actual trigger:

### Tier A — PROVEN: the OT-dual gap (`damped_newton_ot_offsets`), a *constraint-satisfaction* Lyapunov
`src/tac/boundary_math/laguerre_logit_offset.py::damped_newton_ot_offsets` solves the per-class-area
(auction-MBO) offset `b*` so `soft_cell_masses(φ,b*) = target_masses`. Its concave dual
`Φ(b) = ⟨π,b⟩ − τ·mean_p LSE((φ_p+b)/τ)` is driven by **Armijo backtracking that accepts the largest
step which does NOT decrease Φ** (dual ascent). Therefore
```
V_OT(k) := Φ(b*) − Φ(b_k) ≥ 0   is MONOTONE NON-INCREASING → 0    (proven, Kitagawa–Mérigot–Thibert
                                                                    2019 damped-Newton global convergence,
                                                                    terminal quadratic rate),
```
and `info["max_mass_err"] = max_c |π_c − m_c(b)|` is the **KKT residual** = a computable certificate that
the per-class mass constraint is met. **Scope, honestly: this certifies the per-class-area INNER solve
(the nucleation memo's "per-class area constraint / auction-MBO" lever), NOT `d_seg` itself.** Its role
in early termination: a `d_seg` plateau is only trustworthy as *converged* when `V_OT` (mass-err) is also
small — otherwise the plateau is a **constraint-violation artifact** (e.g. the lane mass collapsed to ~0,
the exact #205 nucleation failure). So Tier A is the **gate that distinguishes a true optimum from an
erased-class false floor.**

### Tier B — CONJECTURAL: Fisher–Rao / KL to the τ-target, for the CE→τ mirror-descent arc
The paper registers `ce_softmax_mirror_descent_natural_gradient_v1` (CE+softmax = Bregman mirror descent
≡ natural gradient in dual coords, Raskutti–Mukherjee 2015). Under *exact* natural-gradient flow the
Fisher–Rao KL to the smoothed target,
`V_FR(t) = D_KL( softmax_{τ(t)}(φ_θ) ‖ p*_{τ(t)} )`, is a **Lyapunov function** (monotone ↓). Two honest
caveats keep this **CONJECTURE, not registered**: (a) the hard target is a simplex vertex (`KL=∞`); use
the τ-smoothed `p*_τ`; (b) **Muon is NOT Fisher–Rao NG** (paper §5 false-friend) so `V_FR` is *not*
monotone under the Muon finisher — Tier B applies to the CE→τ arc only. Value: it says the mirror-descent
arc *should* be certifiable, and the arc-length of `V_FR` is the adiabatic clock of §4.

### Tier C — OPERATIONAL + MEASURED: the EWMA descent-rate monotonicity test (the ACTUAL trigger)
This is what the trajectory instrument already computes and what the stopping rule fires on. Let `d_n` be
realized `d_seg` at verdict `n` (epoch `e_n`). Define the smoothed descent rate (EWMA, `α∈(0,1)`):
```
r̂_n = α · (d_n − d_{n−1})/(e_n − e_{n−1})  +  (1−α) · r̂_{n−1}.     (r̂<0 descending = good)
```
Per current stage `s` (onset `e_s`): `best_s = min` realized `d_seg` in `s`; `K_no` = consecutive
verdicts since `best_s` with no new running-min (the instrument's `trailing_no_improve_streak`);
`net_delta_s = d_last − d_start` (the instrument's `net_delta_dseg`).

### THE STOPPING RULE (mathematically defined)

Stop / advance the current stage when EITHER branch fires:

- **(P) PLATEAU (descent saturated — the PMP switching surface `|r̂| ≤ c_run`):**
  `K_no ≥ K*` **AND** `|r̂_n| ≤ ε_plateau`  **AND** (if the area lever is on) `V_OT small`.
  ⟹ the descent no longer pays its wall-clock → **advance to the next stage, or terminate if final.**

- **(D) DIVERGENCE / ERASING (the τ-creep detector — URGENT, do not wait for K\*):**
  `r̂_n ≥ +δ_creep` for `K_creep` consecutive verdicts **AND** `net_delta_s > 0`
  **AND** `ep_loss` falling (the surrogate↔hard-verdict decoupling).
  ⟹ this is the **MEASURED #205 signature**: the nucleation memo records
  `d_seg 0.004752→0.006568` climbing across the τ stage while `ep_loss 148.5→134.1` falls. **Stop the
  stage/run immediately and re-steer** (§5).

**Terminal STOP-RUN:** (P) fires on the **Muon** stage AND `implied_S` shows no new best for `K_run`
verdicts.

**Threshold honesty.** `K*` is the instrument's existing `plateau_k`; `(α, ε_plateau, δ_creep, K_creep,
K_run)` are *configuration*, **to be calibrated on #205's OWN verdict log** (the instrument already
self-validates against hand-measured anchors — τ time-to-best ≈375, τ dead-tail ≈200, l7/τ descent ratio
≈2.8 — via `_anchor_check`, so the calibration is a $0 replay, not a fabricated constant). I give the
*structure*; I do **not** invent numeric thresholds as if measured.

**What the rule buys against #205:** #205 ran a **fixed 426-epoch τ stage** (ep300→726) that the
nucleation analysis shows was *eroding* the lane (sub-critical nucleus under mean-curvature flow, law
`mcf_minority_erasure_inevitability_v1`). Branch (D) would have fired within a few verdicts of ep300 —
converting a 426-epoch eroding dead-tail into an immediate re-steer.

---

## 4. Adaptive / model-predictive schedule (vs #205's fixed schedule)

**#205 (fixed).** `--tau-anneal-shape` cosine/geometric over a fixed `--epochs` window; stage epochs
hardcoded (`--tau-softplus-start-epoch 300`, `--l7-start-epoch 800`, `--muon-start-epoch` set); the τ
stage ran a fixed length regardless of the creep. Open-loop.

**Adaptive τ (adiabatic / Fisher–Rao arc-length reparametrization).** The paper's Fisher–Rao arc-length
is the *adiabatic rate*: advance τ at **constant arc-length velocity**, not constant epochs. Measure the
per-verdict arc-length increment `ds_n` (proxy: verdict-to-verdict KL of the soft label field from cached
logits, or the normalized `|Δd_seg|+|Δep_loss|`), then set `Δτ_{n} ∝ ds_target / ds_n`: **slow τ where
the field moves fast** (near a transition — the paper's *critical-slowing-down* power-law), speed it up
on the flat interior. This is `--tau-anneal-shape geometric` (equal epochs per octave) made **online and
state-dependent** rather than a fixed curve. It is realized with real flags today by choosing
`--tau-softplus-tau` / the temp-anneal params per *segment* on resumable relaunches (the controller emits
the segment schedule; §6).

**MPC (receding horizon), stated with its honest limit.** At each verdict, fit a local first/second-order
plant model to the last `W` verdicts (the instrument's `descent_rate` = the plant's first-order gain,
the post-best `std` = the noise), predict `d_seg` over a short horizon `H` as a function of `(τ,η,w_eik)`,
choose the control increment minimizing predicted terminal `d_seg + c_run`, apply the first increment
(receding horizon). **Honest caveat:** the plant is strongly **non-stationary at transitions**, so
aggressive MPC is unsafe; the *reliable* online adaptation is (a) **arc-length τ reparam** + (b)
**gain-scheduled LR re-warmup fired only on a DETECTED transition spike** (`r̂` jumps positive right after
a boundary → engage `--stage-transition-rewarmup-epochs` + `--stage-transition-reset-moments`, which
exist and are default-off) + (c) the **creep-triggered stop** (§3-D). MPC is the *frame*; the safe
subset is arc-length + spike-gated re-warmup + creep-stop. Say so.

---

## 5. Recursive self-reflection lifted to the training loop (adapt council #363)

Council #363 (recursive self-reflection): Round-1 deliberate → Round-2 classify each assumption into a
4-value empirical-verification taxonomy → Round-3 resolve (verify / downgrade-to-PROVISIONAL / escalate)
→ 3-clean-pass SEAL, `MAX_SELF_REFLECTION_ROUNDS = 5`. **Lift it to the run:**

Every `M` verdicts (or at each stage boundary) the controller runs a **self-reflection tick**:

- **Round-1 analogue** = the raw verdict stream + per-stage attribution (#188/#216).
- **Round-2 analogue — CLASSIFY the trajectory state** into a 4-value taxonomy mirroring #363:

  | training class | signal | #363 analogue |
  |---|---|---|
  | **CONVERGING** | `r̂ < 0` descending | VERIFIED_DESCENDING |
  | **PLATEAU** | `\|r̂\| ≤ ε`, `K_no ≥ K*` | SATURATED |
  | **DIVERGING / ERASING** | `r̂ ≥ +δ`, `net_delta > 0`, `ep_loss ↓` | **FALSIFIED_REGRESSING** (the τ-creep) |
  | **VOLATILE** | high post-best `std`, oscillating | UNSTABLE_AWAITING_DAMPING |

- **Round-3 analogue — RE-STEER decision** (emit only; CONTAINMENT):
  - CONVERGING → **continue.**
  - PLATEAU → **advance stage / early-terminate** (emit the stop epoch; the run's `--epochs` / kill is
    the resumable actuation).
  - DIVERGING → the **physics-required** re-steer. For the MEASURED lane creep, the nucleation memo says
    **seeding must be at init** ⟹ this is necessarily a **FRESH seeded run**, not a resume — emit
    `--seed-islands --structured-init --lane-prior-phi1` + raised `--eikonal-weight` (interface-width
    control) + per-class-area (auction-MBO / Tier-A) constraint. **Intra-run** softer re-steers that DON'T
    need re-init: raise `--eikonal-weight`, engage `--margin-saliency-weight` / `--lane-thin-weight` /
    `--amplify-weight` at the stage gate, `--stage-transition-rewarmup-epochs` + `--stage-transition-
    reset-moments`.
  - VOLATILE → engage the **wider-EMA finisher** `--ema-decay-finisher` / `--ema-decay-finisher-start-
    epoch` (SWA-style late-oscillation averaging), or damp LR.
- **3-clean-pass analogue** = require the classification to be **STABLE across `K` consecutive ticks**
  before acting (never re-steer on one noisy verdict — mirrors both the streak requirement and #363's
  3-clean-pass), and bound the number of re-steers per run (like `MAX_SELF_REFLECTION_ROUNDS`) to prevent
  thrash / infinite re-launch loops.

**Deterministic + resumable.** The self-reflection state — `(r̂_n, streak counters, classification
history, re-steer count)` — is a small append-only JSON in the run dir (same discipline as the verdict
log). On resume it is re-read; all inputs are the **deterministic verdict log** ⟹ **deterministic
decisions**. The controller **reads, never writes into training** ⟹ training stays **BIT-IDENTICAL**
(the exact property `--async-verdict` already guarantees: "the verdict is never read back").

---

## 6. The bridge + the concrete, buildable controller (CONTAINMENT-safe)

**What makes the scaling ACTIVE:** facets 1–4 pick the statically-optimal schedule *shape*; facet 5
**closes the loop** so the schedule *adapts online and self-terminates* on a certificate. The concrete
deliverable (design only here — `$0`, not built):

**`tools/witness_control_monitor.py`** — a lean, deterministic, resumable monitor that:
1. **Follows the #205-style verdict log** — reuses `parse_verdicts` + `compute_dynamics` from
   `tools/render_witness_trajectory_dynamics.py` (**DE-ORPHAN, do not rebuild** — that instrument IS the
   plant-model / costate-at-lever-scale estimator; §2 table). Reads the `{stage:"verdict", epoch, d_seg,
   d_pose, blob_bytes, implied_S, ep_loss, seg_form, ts}` lines the trainer already emits
   (`_emit_verdict_row`, verified).
2. **Computes the certificate + classification** — Tier-C EWMA descent test (§3), the 4-value class
   (§5), and — when the per-class-area lever is active — the Tier-A `V_OT` / `max_mass_err` gate via
   `damped_newton_ot_offsets` on the cached field (READ-ONLY on cached GT, no GPU).
3. **Emits a control-decision line** (append-only, sister of the verdict log):
   `{stage:"control", epoch, classification, lyapunov:{tier_c_rhat, tier_a_mass_err?}, decision:
   "continue|advance-stage|early-terminate|re-steer", proposed_config_diff:{...real flags...},
   rationale}`.
4. **NEVER launches.** It emits; the **meta-layer costate-controller (#247)** consumes the decisions
   (this monitor is the costate *sensor + classifier* feeding the active DSL), and the **operator / the
   P0 system-memory governor** (`machine_crashing_risk_is_P0_hard_gate`) gates any actuation. This is the
   binding CONTAINMENT contract: closed-loop *decisioning*, human/governor-gated *actuation*.

**Actuation is already expressible with real flags** (all grep-verified in the trainer — never invented):
- *early-terminate / stage retiming*: `--epochs`, `--anneal-epochs` (decoupled schedule length for
  warm-start arms), `--resume-from`, `--tau-softplus-start-epoch`, `--l7-start-epoch`, `--muon-start-epoch`.
- *transition treatment*: `--stage-transition-rewarmup-epochs`, `--stage-transition-rewarmup-floor`,
  `--stage-transition-rewarmup-shape`, `--stage-transition-reset-moments`.
- *interface / topology re-steer*: `--eikonal-weight`, `--length-weight`, `--margin-saliency-weight`,
  `--lane-thin-weight`, `--amplify-weight`.
- *fresh seeded re-steer (init-only, per nucleation memo)*: `--seed-islands`, `--structured-init`,
  `--lane-prior-phi1`, `--island-dilate-px`.
- *volatility damping*: `--ema-decay-finisher`, `--ema-decay-finisher-start-epoch`, `--muon-lr-final-frac`.

**The trainer already supplies the resumable substrate** the controller's decisions require:
`--resume-from` + per-stage preserved checkpoints + `--anneal-epochs` (so a warm-start arm reproduces the
*disease-regime* schedule tail, not the anneal tail). So "early-terminate + resume with a re-steered
config" is a **real, deterministic, resumable operation today** — the controller emits the flag diff; the
operator launches under the governor.

**The one honest boundary (matches the nucleation lesson exactly):** seeding must be at **init** ⟹ the
DIVERGING/ERASING re-steer for a *zero-mass* class (the measured #205 lane creep) is *necessarily* a
FRESH seeded run, not a live patch. The controller must emit **"recommend fresh seeded run"**, never
"patch the live run" — which is precisely why #205's remaining post-creep epochs are low-EV and the
right move was *preserve #205 + fresh seeded run*.

---

## 7. Consistency / triality note

- **DAG:** append a FEED block (state leg) recording this facet + the control-monitor design node.
- **DSL:** the control-decision schema `{decision, proposed_config_diff}` is the DSL leg made *active*
  (passive→active per the meta-layer); it emits argv from real flags, never auto-fires (CONTAINMENT).
- **equations:** the PMP costate law + the Tier-A OT-dual Lyapunov are the equations leg; Tier-A is
  registrable (proven), Tier-B/C are CONJECTURE/OPERATIONAL and **not** registered as laws.

MEANS firewall: this controller decides *what to run*; the pointer (0.19110) moves only when it picks a
lever that, byte-closed, produces an `upstream/evaluate.py` n600 row < 0.19110.

---

## 8. The ONE synthesis claim

> **[DERIVED frame + PROVEN inner certificate + MEASURED trigger]** Witness training is a
> free-terminal-time Bolza control problem whose optimal schedule is the Pontryagin costate `λ` = the
> already-cached through-R margin-Jacobian `S_R` (pixel scale) and the trajectory instrument's per-stage
> descent-rate (lever scale); the **only PROVEN Lyapunov** is the OT-dual gap of `damped_newton_ot_offsets`
> (a *constraint-satisfaction* certificate for the per-class-area inner solve, used as the gate that
> tells a true optimum from an **erased-class false floor**); and the actual early-termination trigger is
> an **operational EWMA descent-rate monotonicity test** whose *diverging* branch `(r̂ > +δ ∧ net_Δd_seg
> > 0 ∧ ep_loss ↓)` is *exactly* the MEASURED #205 τ-creep (surrogate↔verdict decoupling from
> sub-critical lane nucleation) — so **the single highest-value control the system was missing is that
> creep detector, which converts #205's fixed 426-epoch eroding τ stage into an arc-length-adaptive,
> creep-terminated, self-re-steering run.**

**Proposed Lyapunov certificate:** `V_OT(k) = Φ(b*) − Φ(b_k) ≥ 0` (OT concave-dual gap of
`damped_newton_ot_offsets`, monotone ↓ 0, PROVEN) as the *constraint* certificate, gating the operational
`V_C`: the EWMA descent rate `r̂_n` on the verdict `d_seg` stream (MEASURED-calibrated, not proven-global).

**Stopping rule (restated):** advance/terminate a stage on **(P)** `K_no ≥ K* ∧ |r̂| ≤ ε_plateau ∧ V_OT
small`; stop-and-re-steer *urgently* on **(D)** `r̂ ≥ +δ_creep` for `K_creep` verdicts `∧ net_Δd_seg > 0
∧ ep_loss ↓`; stop-run when (P) fires on Muon with no new best `implied_S` for `K_run` verdicts. All
thresholds calibrated by $0 replay on #205's own verdict log (the instrument self-validates against its
hand anchors); none fabricated here.

*Sisters:* `project_meta_layer_above_triality_hamiltonian_control_costate` (the costate identification) ·
`lane_nucleation_failure_seed_above_critical_nucleus` (the MEASURED creep = the DIVERGING trigger) ·
`deepmath_amortizing_argmax_paper_draft_20260704` (Fisher–Rao / τ=ε=ħ / MCF-erasure laws) ·
`tools/render_witness_trajectory_dynamics.py` (#188/#216, the plant-model sensor to DE-ORPHAN) ·
`src/tac/boundary_math/laguerre_logit_offset.py::damped_newton_ot_offsets` (the PROVEN Tier-A Lyapunov).
