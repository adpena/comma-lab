# CT-1 — OPTIMAL CONTROL OF TRAINING + THE CAMPAIGN (T5 crucible deep-research seat, requirement O)

review_status: fresh-research-round-1 (unreviewed)
date: 2026-07-07 · seat: CT-1 · charter: import the THEOREMS of optimal/event/switched/adaptive/
learning control into the witness training run, the v3/v4 schedule, the costate controller, and
the multi-run campaign — positive-design contract (every section ends in a LAW in our notation
with plugged values, a BUILD item, or a $0 probe with predicted band + kill threshold).

STORES CONSULTED: `.omx/research/t5_crucible/ORCHESTRATION_LEDGER.md` (full, incl. requirements
A–O + landings log) · `DRAFT_OPTIMAL_STACK_v3_20260707.md` §0–§5 (stage graph, event exits,
transition law 2.2c, τ_end derivation 2.2d, chain-A terminal fold 2.3, curriculum §3, costate §4,
λ_bytes law §5.0) · `.omx/research/costate_controller_design_20260705.md` §1–§4 (state/control/
costate tiers, identifiability ledger, backtest scorecard, live shadow) ·
`negatives_scale_validity_review_20260707.md` §3–§4 (TAIL_k law, scale-validity table) ·
corpus_query: "adaptive epsilon CFL" (adaptive_eps_cfl_edge_tracking_v1 + witness_config_
differential_equations_derivation_20260705 + confound_hunt F2 floor-pinned finding) ·
"powerlaw meat exit forecast" (S3 position, powerlaw_meat_exit producer, weak-KAM O(1/t) anchor) ·
"event triggered curriculum injection test backtest" (v1/v2 draft req-B contract rows).
NOT consulted (declared): the full S1/S4/S6 position bodies (schedule/control-relevant content
reached via the ledger + v3 fold); no WebFetch of individual papers — the control-theory theorem
statements below are classical (Pontryagin/Trélat-Zuazua/Mayne/Grüne/Tabuada/Heemels/Liberzon/
Hespanha/Krstić-Wang/Feldbaum/Borkar/Arimoto/EWMA-R2R) and are labeled INFERRED-from-literature
where the exact constant matters; every plugged NUMBER is from our own measured corpus.

Measured constants used throughout (all MEASURED unless noted):
S = 100·d_seg + √(10·d_pose) + 25·bytes/37,545,489 · λ_seg = 100 · λ_bytes = 6.6586e-7 S/B ·
crossing margin 0.00178 S · best clean d_seg 0.0033662@ep650 (mod32cap) · decode-gap
Δ = +1.0427e-4 d_seg = +0.010427 S (5.86× margin) · τ*_end = m_q/ln5 = 0.062 (DERIVED) ·
Muon −32% d_seg · control anneal TRUNCATED (β 3.177/4.0, τ 0.216/0.05) · ~42 s/ep · budget
1000–3000 ep · PR95 reference 29,650 ep / 8 stages · verdict cadence 25 ep · TAU-onset erosion
slope +3.3e-3 S/ep@ep350 → +2.4e-4 S/ep@ep450 · co-predicate eps_rel 2e-4/ep (rel), V=4,
backtest first sustained fire ep625 · transition forfeit +5.4e-4 S (restore-EMA-best law) vs
+2.7e-3 S (no restore) · cap_fin = clamp(1.5·τ̂_e, 150, 350), τ̂_e = 305 ep (INFERRED) ·
EMA ρ = 0.997 · attribution floor 0.015 S now (recon gap), 0.00178 S after req-F#6.

---

## §0 — EXECUTIVE MAP (field ↔ our object ↔ verdict)

| # | field | our object | verdict |
|---|---|---|---|
| 1 | Pontryagin / adjoint | the run as ẋ=f(x,u); costate controller's mature form (req M) | **IMPORT-NOW** (terminal costate lives on the EMA shadow; LR-anneal = singular arc; PMP stop-rate law RATIFIES the shipped eps_rel to within 5%) |
| 2 | Turnpike theory | anneal shape under 1000–3000-ep budget | **IMPORT-NOW** (exit-arc is budget-independent → validates cap_fin; extra budget → TAIL_k, never longer transients; derived exit floor 3/ν = 115 ep vs shipped 150) |
| 3 | MPC / receding horizon | TAU→FIN hand-off + adaptive-ε feedforward (req M) | **IMPORT-NOW** (forfeit-matched exit threshold s* = ν·forfeit = 1.4e-5 S/ep → fire ~60 ep later, recover ≈ the +5.4e-4 forfeit; horizon N = 2 cadences) |
| 4 | Event/self-triggered control | co-predicate triggers, verdict cadence | **IMPORT-NOW** (self-triggered cadence law Δt = clamp(floor/|Ŝ′|, 25, 100); Zeno exclusion ≡ req-B caps — already honored) |
| 5 | Switched/hybrid systems | stage graph, TAIL_k cycles, spike-skip | **IMPORT-NOW** (dwell-time law: run-1 satisfied 25×; TAIL_k dwell derived; l7-defect = failure of S as common Lyapunov function — theorem-grounds the measured demotion) |
| 6 | Gain scheduling + adaptive | per-class λ_c, adaptive-ε clamps | **IMPORT-NOW** (1-Lipschitz easing = the LPV slow-variation condition, already satisfied; clamp→projection fix shape for the measured floor-pinned adaptive-ε confound) |
| 7 | Iterative learning control | the campaign v3→v4→v5; decode gap; parity row | **IMPORT-NOW + CAMPAIGN** (decode-gap is ILC-repeatable → feedforward-cancel in v4: train-side bar 0.00100; EWMA gain law; matched-instrument demand = req-N formalized) |
| 8 | Extremum seeking | measure→sweep→derive loop (req M) | **IMPORT-NOW** (attributable-dither law: sweep step ≥ 2·floor/ĝ; byte-sweeps need ≥5.3 KB steps at margin floor, ≥45 KB at today's recon-gap floor → req-F#6 gates fine rate sweeps) |
| 9 | Dual control (Feldbaum) | run-1 instrument-vs-crossing value (req N) | **CAMPAIGN-LAYER** (EVSI arithmetic: run-1's information value ≈ 1 saved run ≫ its 1.6–6.4% crossing value; feeds PowerPlay ordering) |
| 10 | Stochastic approximation / two-timescale | per-step vs per-cadence vs per-stage vs per-run | **IMPORT-NOW** (separation check: settle 3/ν ≈ 115 ep vs co-predicate window 100 ep = 0.87× — MARGINAL; V=4→5 fixes it for one flag value) |
| — | LQR/Riccati direct, continuous HJB solve | — | **DEAD** (no linear plant; dim(θ) ~10⁵ kills grid HJB; value-gradient enters only via §1's λ = ∇V identity) |

The single deepest import (threads §1/§3/§7): **the costate λ(t) is the value-function gradient
∇ₓV(x,t) and obeys a BACKWARD equation — a forward-running controller can only realize it through
an internal MODEL of the future.** The measured ep450 λ-decay miss (predicted +0.0060 [0.0018,
0.0103], realized +0.0004) is exactly the error of extrapolating λ forward linearly. Requirement
M's "feedback + feedforward with fitted forecast models" is therefore not a preference — it is
the ONLY consistent estimator of the adjoint. (DERIVED; anchor: costate_controller_design §3.)

---

## §1 — PONTRYAGIN / ADJOINT: the run as an optimal-control problem

### 1.1 The problem, in our notation (DERIVED framing; every symbol maps to a real emission)

State x = (θ, e, m, v, s): live weights θ, EMA shadow e (per-step e⁺ = ρe + (1−ρ)θ, ρ = 0.997),
optimizer moments (m, v), stage tag s. Control u = (η(t), τ(t), β(t), λ_c(t), σ(t)): lr, softmax
temperature, hosc slope, per-class weights, stage-switch signal. Dynamics (control-affine in η):

  θ̇ = −η(t) · P(m, v) · ∇_θ L(θ; τ, β, λ_c, s),  ė = κ(θ − e), κ = (1−ρ)·steps/ep

Cost is a **Mayer problem with lexicographic time** (L59: wall-clock strictly secondary):

  J = Φ(x(T)) = S( byteclose(e(T)) ),  minimize J first, then T.

### 1.2 Terminal costate lives on the EMA shadow (DERIVED; consequence checkable)

λ(T) = ∇ₓΦ. Since Φ depends only on e(T): **λ_θ(T) = 0, λ_e(T) = ∇S∘byteclose ≠ 0.** The adjoint
recursion through the EMA coupling gives the weight a step at time t has on the terminal read:
(1−ρ)·ρ^(N−n) over steps — a horizon of 1/(1−ρ) = 333 steps ≈ 0.6 ep (at ~600 serial pair-steps/
ep). Consequences: (a) every verdict/probe/spectrum MUST be computed on e, not θ (ratifies the
standing EMA-shadow mandate and the D5 EMA-shadow-at-inference risk from the risk register — now
a theorem consequence, not a convention); (b) each stage/TAIL cycle needs a terminal settle
window ≥ 3/(1−ρ) steps ≈ 2 ep before its exit verdict — trivially satisfied by the 25-ep cadence
(state: SATISFIED, no build).

### 1.3 LR is bang-bang + singular arc; the anneal IS the singular arc (DERIVED)

H = λᵀf is LINEAR in η (dynamics control-affine) ⇒ PMP gives η* = η_max or η_min except where the
switching function σ_η(t) = −λᵀP∇L vanishes on an interval — a **singular arc**, where η_sing is
set by higher-order (generalized Legendre-Clebsch) conditions, i.e. state-dependent. Reading: the
measured-good practice (flat lr → anneal → warm finisher) is bang(η_max) → singular-arc tracking
→ terminal transient. The singular arc is class-(c) SELF-DERIVING by theorem — a fixed cosine is
only an open-loop approximation of it. This is the PMP justification for requirement M(3)'s
"control laws are DEs from the system's own physics", applied to lr.

### 1.4 THE LAW — PMP stop-rate, and it RATIFIES the shipped trigger to 5% (DERIVED, plugged)

With lexicographic time cost ρ_wall = attribution-floor-per-epoch, PMP's transversality says:
leave the stage when the marginal score rate falls below the running cost rate:

  **exit when |dS/dep| < ε_stop, ε_stop = floor_S / cadence = 0.00178 / 25 = 7.1e-5 S/ep.**

Shipped trigger: eps_rel = 5e-3 relative per 25 ep. Converted at the exhaustion operating point
(d_seg = 0.0034): Δd_seg = 5e-3·0.0034/25 ep = 6.8e-7 d_seg/ep → ×100 = **6.8e-5 S/ep**. The
shipped relative trigger and the PMP-derived absolute stop rate coincide within 5% AT THE
OPERATING POINT — an independent ratification the draft did not have. Caveat (stated): the
coincidence is operating-point-dependent (relative trigger scales with d_seg); at d_seg = 0.001
(the crossing target) the shipped trigger = 2e-5 S/ep, i.e. 3.5× FINER than the floor-derived
stop — correct direction (finer scores demand finer exits). LAW ROW: register
`pmp_stop_rate_epsilon_v1`: ε_stop(t) = floor_S(t)/cadence(t), floor_S = 0.015 until req-F#6
then 0.00178. $0 PROBE P-CT3 (§11) backtests it.

---

## §2 — TURNPIKE THEORY: the shape of the optimal schedule under any budget

### 2.1 The theorem, mapped (INFERRED-from-literature: Trélat–Zuazua 2015 exponential turnpike;
Grüne strict-dissipativity characterization)

For long-horizon OCPs that are strictly dissipative w.r.t. an optimal steady state, the optimal
trajectory = entry transient → hug the turnpike (quasi-steady optimal operating point) → exit
transient, with transient lengths set by SYSTEM time constants, NOT by the horizon; extra horizon
is spent ON the turnpike. Our objects: (i) within a stage, the turnpike is the productive plateau
(quasi-steady descent at the stage's operating point); (ii) across the run, τ's geometric anneal
is a slowly-moving turnpike (quasi-static in the two-timescale sense, §10).

### 2.2 Measured verification of the exponential approach (MEASURED + DERIVED)

Within-TAU erosion slope decays +3.3e-3 → +2.4e-4 S/ep over ep350→ep450 ⇒ contraction rate
**ν = ln(13.75)/100 = 0.0262 /ep, settle time 3/ν ≈ 115 ep** — the exponential-approach signature
turnpike theory predicts. Cross-checks that fall out at once: (a) the derived exit-transient
floor 3/ν = 115 ep vs the shipped cap_fin floor 150 — shipped is 1.3× the derived, KEEP; (b) M4's
"τ-stage ran 76–125 ep past meat" is turnpike OVERSTAY (time on the pike after the exit arc
should begin = pure waste) — the event exits are the theorem-correct fix; (c) the M5 finisher
failure (τ_e = 305 > 274 budget) is an exit arc truncated below its OWN time constant — the
theorem says the exit arc length is non-negotiable, the entry point isn't.

### 2.3 THE LAW — budget allocation is turnpike-shaped (DERIVED, plugged)

For ANY epoch budget T:  **schedule(T) = entry(≈250–300 ep measured CE) + turnpike(T − entry −
exit) + exit(clamp(1.5·τ̂_e, 150, 350))**, and marginal budget dT goes to (i) turnpike dwell only
until meat-exhaustion, then (ii) **TAIL_k cycles — each a fresh turnpike-exit pair at τ_k =
max(τ_{k−1}/2, τ*_k)** (negatives-review §3 law). Turnpike depth of TAIL: from τ_end = 0.062 with
halving, τ* is reached in ≤2–3 cycles; each cycle needs ≥ its own settle (≈115 ep) + exit
(≥150 ep floor) ⇒ **TAIL cost ≈ 265–350 ep/cycle ⇒ k_max ≈ 3–7 within the 3000-ep budget after
ep650 entry** — the tail is τ*-limited, not budget-limited, at 3000 ep; PR95's 29,650-ep/8-stage
run is the k→∞ existence proof. BUILD: none new (TAIL_k already the negatives-review adoption;
this section supplies its budget law + cycle-length floor 265 ep, previously unstated).

---

## §3 — MPC / RECEDING HORIZON: the feedforward half of requirement M

### 3.1 The formal MPC problem (DERIVED framing)

At each verdict cadence t_k: internal model = {AIC powerlaw/exponential meat forecast (built:
`powerlaw_meat_exit`), per-class λ_c trajectories (F-rows), response surfaces (#170)}; decision
set = {continue, fire stage-exit (with the §2.2c restore-EMA-best transition), adjust cadence};
constraints = fail-safe caps (req B), spike/liveness guards, memory preflight; terminal cost =
forecast remaining meat at horizon end. Recompute every cadence (receding).

### 3.2 Horizon length from measured forecast validity (MEASURED → DERIVED)

Measured forecast errors at 25-ep horizon: ep350 creep predicted +0.0825 [0.0097, 0.155],
realized +0.0119 (in band, central 7× high); ep450 predicted +0.0060 [0.0018, 0.0103], realized
+0.0004 (BELOW band — linear-λ extrapolation overpredicts under deceleration). Import (Grüne–
Pannek unconstrained-MPC suboptimality, INFERRED-from-literature): closed-loop degree of
suboptimality α(N) improves exponentially in N given exponential cost controllability — which
§2.2 measured (ν = 0.0262). But the model error GROWS with horizon; the binding constraint is
model validity, not α(N). **LAW: N* = 2 cadences (50 ep) with the exponential/powerlaw mixture
model (the linear model is 1-cadence-only — measured); every horizon-N claim carries its band and
the model id.** The v3 spectrum-rate exponential mixture (S3 DECIDE) already replaced linear-λ —
this law pins its horizon.

### 3.3 THE LAW — forfeit-matched exit threshold: fire ~60 ep LATER, recover the +5.4e-4 (DERIVED, plugged — the section's headline)

MPC hand-off principle: exit when forecast remaining stage gain < the measured transition cost.
Remaining gain from current slope s under exponential decay = ∫s·e^(−νt)dt = s/ν. Transition cost
(v3 §2.2c, MEASURED) = +5.4e-4 S. Therefore:

  **fire TAU→FIN when s < s* = ν · forfeit = 0.0262 × 5.4e-4 = 1.41e-5 S/ep.**

Shipped trigger fires at s ≈ 6.8e-5 S/ep (§1.4) — 4.8× coarser ⇒ fires EARLY by
Δt = ln(6.8e-5/1.41e-5)/ν ≈ 60 ep. Backtest agrees: shipped fired ep625; forfeit-matched
predicts ~ep685 (cap 726 still binds). At ~ep685 the TAU-window EMA-best ≈ the ep650 true best ⇒
the +5.4e-4 forfeit → ≈0, at a cost of ~60 ep × 42 s ≈ **42 min wall-clock for +5.4e-4 S = 30% of
the crossing margin** — under L59 (time lexicographically secondary) this trade is mandatory.
Self-consistency: as the transition law improves (restore-best), forfeit shrinks, s* shrinks,
fire moves later — the law is a fixed-point, not a constant. BUILD B-CT1: add the forfeit-matched
threshold as a SECOND co-predicate arm (would-fire audit row first, per req-B: backtest P-CT3 +
injection + cap 726 unchanged). ~10 LOC given the per-epoch-normalized slope already lands (MINOR-9).

### 3.4 Adaptive-ε as MPC (DERIVED, one paragraph)

The #318/#320 law ε(t) = clamp(|c_a|√(ηλ_eik/8)(1+m), 0.3, 0.7) is feedback-only (responds to the
current CFL edge). Its MPC form: replace c_a(t) by the τ-forecast c_a(τ(t+H)) one horizon ahead
(τ's path is KNOWN — it is our own control), so ε arrives at the stability edge BEFORE the anneal
sharpens the fronts. Zero new sensors; the τ schedule is the internal model. BUILD B-CT2 (~5 LOC
inside the adaptive block): evaluate the law at τ(t+25) instead of τ(t). Falsification: if the
clamp-binding check (negatives-review probe, queued) shows the clamps bind >90%, B-CT2 is moot
until the §6 projection fix re-derives the window.

---

## §4 — EVENT-TRIGGERED + SELF-TRIGGERED CONTROL

### 4.1 The trigger theorems, mapped (INFERRED-from-literature: Tabuada 2007 ISS triggering;
Heemels et al. periodic-ETC; self-triggered variants)

ISS event-triggering: fire when the measurement error since the last action exceeds a
STATE-PROPORTIONAL threshold (‖e‖ ≥ σ‖x‖, σ<1 a contraction share) — this preserves a fraction
(1−σ) of the Lyapunov decrease and guarantees a strictly positive minimum inter-event time (Zeno
exclusion) under Lipschitz dynamics. Mapping: (a) our per-epoch-normalized RELATIVE eps_rel is
exactly the state-proportional form (absolute thresholds are the anti-pattern the theorem
excludes; MINOR-9 got this right); (b) Zeno exclusion ≡ req-B's fail-safe caps + min-stage 250 +
25-ep cadence (a built-in dwell) — REQ-B IS THE ZENO THEOREM, already honored; state SATISFIED.
(c) The principled per-class veto threshold (v3 §2.2(1) left eps_c = pooled until calibration):
**eps_c = σ·ν_c·d_seg_c with σ = 0.5** — half the class's own measured contraction rate; run-1's
per-class F-rows identify ν_c (fractional law class (e), formula now supplied).

### 4.2 THE LAW — self-triggered verdict cadence (DERIVED, plugged; the wall-clock win)

Self-triggered control computes the NEXT sampling time from the current state instead of polling:
choose Δt so the maximum foregone improvement per interval ≤ the attribution floor:

  **Δt_next = clamp( floor_S / |Ŝ′(t)| , 25, 100 ) ep,  floor_S = 0.00178 (post req-F#6).**

Plugged: at ep350 (|Ŝ′| = 3.3e-3) → 0.54 ep ⇒ floor 25 binds (verdicts are the cost; n600
chunked-CPU each); at ep450 (2.4e-4) → 7.4 ⇒ floor still binds; near exhaustion (1.4e-5, §3.3) →
127 ⇒ cap 100 binds. Net effect: **verdict cadence stretches 25→100 ep exactly in the late-TAU/
FIN/TAIL region where verdicts are least informative — ~30–40% fewer n600 verdicts per run at
zero score cost** (every skipped verdict is one the floor says could not have changed a decision).
Event-safety: the floor 25 IS today's cadence (degrades to current behavior); alarms (spike,
liveness) remain per-epoch — only the n600 verdict stretches. BUILD B-CT3 (~15 LOC in the
verdict scheduler) + req-B tests; $0 PROBE P-CT2 backtests on the mod32cap 41-row trace
(predicted band: 12–17 of 41 verdicts skipped, zero missed-best beyond one cadence; kill: any
missed best > 1 cadence).

---

## §5 — SWITCHED / HYBRID SYSTEMS: stages as a switching law

### 5.1 Dwell-time theorems, mapped (INFERRED-from-literature: Liberzon–Morse dwell time;
Hespanha–Morse average dwell time)

Switching among individually-contracting modes is stable if the dwell time exceeds
τ_d > ln(μ)/ν, where μ bounds the Lyapunov jump at a switch and ν is the per-mode contraction.
Plug OUR jumps: worst measured switch = cold Muon fire, +27.5% loss quench (M1) ⇒ μ = 1.275,
ν = 0.0262 ⇒ **τ_d > ln(1.275)/0.0262 ≈ 9.3 ep.** Shipped min-stage 250 ep = 27× margin —
run-1 stage graph SATISFIED with no build. Where it binds: **TAIL_k cycles.** With the v3
transition law (restore EMA-best + never-reset moments) the jump is the measured benign kind
(ce→tau boundary −0.0011 S, i.e. μ ≈ 1) ⇒ dwell floor is set by SETTLE not stability:
**dwell_TAIL ≥ 3/ν ≈ 115 ep** — consistent with §2.3's 265–350 ep/cycle (settle + exit arc).
LAW ROW: cap_tail (negatives-review: 2× TAU length, injection-tested) now carries a derived
LOWER bound too: `tail_cycle_dwell ≥ 115 ep`; a TAIL cycle shorter than this is measuring its
own transient (the M-S2 class of confound).

### 5.2 Common Lyapunov function: S itself — and l7 is the measured counterexample (DERIVED)

A switched system needs no dwell restriction at all if a COMMON Lyapunov function exists. Our
candidate is S (via n600 verdicts): each stage's surrogate loss differs, but each stage is
admissible iff it decreases S. Measured record: CE ✓, TAU ✓ (post-onset), warm-FIN ✓ (bet,
instrumented), **l7 ✗ — the measured "l7 RAISES d_seg" defect is precisely the failure of S as a
common Lyapunov function under the l7 mode.** The standing l7 demotion (CLAUDE.md capstone
section) is therefore not a tuning judgment but a switched-stability exclusion: a mode that
climbs the common V may only enter under a dwell + restore contract (which is what the FIN
regression guard implements for Muon). LAW: **mode admission rule — a stage/lever enters the
default graph only with a measured (or would-fire-audited) ΔS ≤ 0 record on the common V; else
it enters ONLY behind a restore-guard.** This is the switched-systems restatement of req-B, now
with the theorem name attached; no new build.

---

## §6 — GAIN SCHEDULING + ADAPTIVE CONTROL: per-class λ_c and the clamp problem

### 6.1 LPV slow-variation condition — already satisfied, now with the bound (DERIVED)

Per-class λ_c(t), the homotopy ramps, and the anneal are gain-scheduled (LPV) controls: the
frozen-time argument that justifies designing each operating point separately requires the
scheduling parameter to vary SLOWLY relative to the loop's contraction: |ṗ|/p ≪ ν. Plug: the
1-Lipschitz easing over 275 ep gives |ṗ|/p ≈ 1/275 = 3.6e-3 /ep ≪ ν = 0.0262 /ep (7.2× margin) —
the shipped easing IS the LPV condition with margin; the 20-ep cosine band-engage ramp gives
1/20 = 0.05 /ep ≈ 1.9× ν — MARGINAL (the measured deconflict law presumably absorbed the
transient; flag, don't change: the ramp is measured-good). LAW ROW: any NEW ramp obeys
ramp_length ≥ 3/ν ≈ 115 ep unless a measured deconflict row (like the band's) licenses faster.

### 6.2 THE FIX SHAPE — projection, not clamp, for adaptive laws (DERIVED; measured anchor)

Measured anchor: the adaptive-ε confound (F2, confound hunt 2026-07-05) — the "adaptive" CFL law
computed ~0.001 ≪ floor 0.3 and was therefore a CONSTANT every epoch (INERT, 0 change-events).
Adaptive-control import: hard clamps break the adaptation Lyapunov argument at the boundary
(chatter or permanent saturation, which is what we measured); the PROJECTION OPERATOR modifies
the update tangentially at the boundary, preserving V̇ ≤ 0 and keeping the law LIVE inside a
re-derived admissible set. Concrete rule (generalizes the confound-hunt L1 alarm):

  **A clamp that binds > 90% of epochs = the law is INERT; the fix is RE-DERIVE THE WINDOW
  (the τ-law re-derivation the negatives review already queued), and where a bound must remain,
  implement it as projection (tangential), not saturation.**

BUILD B-CT4: the clamp-binding fraction is already an F-row candidate; add the projection form to
the adaptive-ε block IF the queued $0 clamp-binding check at fine-τ margin fields shows >90%
binding (pre-registered decision, not a change now). Cost ~10 LOC, gated.

---

## §7 — ITERATIVE LEARNING CONTROL: the campaign as run-to-run control (requirement N + P)

### 7.1 The ILC frame (INFERRED-from-literature: Arimoto ILC; semiconductor EWMA run-to-run)

Campaign iterates: u_{k+1} = u_k + L·e_k, where u_k = run-k's config (lever vector + schedule
params), e_k = measured deviation vector (per-class d_seg residuals vs target, pose term, byte
sections, decode gap). Plant y_k = P(u_k) + d + w_k with d = REPEATABLE disturbance, w = noise.
Two theorems to import: (i) monotone convergence iff ‖I − L·P̂‖ < 1 in the chosen norm;
(ii) **ILC drives the REPEATABLE component d to zero exactly, even under model mismatch — the
non-repeatable w passes through.** EWMA-R2R stability (semiconductor lit): with intercept update
d̂_{k+1} = ω·(y_k − P̂u_k) + (1−ω)·d̂_k, stable iff 0 < ω·ξ < 2, ξ = true/model gain ratio.

### 7.2 THE LAW — the decode gap is ILC-repeatable: cancel it feedforward in v4 (DERIVED, plugged)

The R6 parity row measured Δ = +1.0427e-4 d_seg (training-side 0.0035103 → decoded 0.0036146) —
a per-run repeatable disturbance of the byte-close/decode path, worth +0.010427 S = 5.86×
the crossing margin. ILC treatment:

  **v4 feedforward: train-side d_seg bar = decoded target − Δ̂ = 0.0011 − 1.0427e-4 ≈ 0.00100
  (tighten the crossing triple's train-side leg explicitly).
  Campaign update: Δ̂_{k+1} = ω·Δ_k^measured + (1−ω)·Δ̂_k with ω = 0.5 (stable for gain-ratio
  ξ ∈ (0,4) — safe against even a 4× decode-model error).**

With Newton-ILC gain L = γ·P̂⁻¹ on the identified lever subspace and model error bounded by the
measured ~35% instrument gap (HVP-vs-true as the proxy for our worst model bias): γ = 0.7 gives
per-run contraction |1 − γξ| ∈ [0.055, 0.545] ⇒ **2–3 runs to the identified floor** — the
control-theory form of the v4/v5 campaign arithmetic (Model B's with-repair band).

### 7.3 REQUIREMENT P(a) — the campaign observability/identifiability condition (DERIVED)

The Gramian analog for our setup: identifiability of (P restricted to the controlled lever set A,
plus d) from K runs requires ALL THREE, and each maps to a concrete signal:

1. **EXCITATION RANK:** the stacked config-delta matrix U_K = [Δu_1; …; Δu_K], AUGMENTED by
   within-run single-lever events (F8 rows count as rank-1 excitations at their activation
   epochs), must have rank ≥ |A|. Measured anti-anchor: the 13-simultaneous-diff pair (#205 vs
   seed-fix) is rank-1 over a 13-dim subspace — UNIDENTIFIABLE, and the estimator correctly
   refuses. Consequence: the λ=0 twin + mirror-schedule twin + F8 mid-run activations are not
   nice-to-haves; they are the rank suppliers.
2. **NOISE FLOOR MEASURED, NOT INFERRED:** every threshold/SE in the loop is currently
   denominated in neighboring-fit residuals (inferred). A direct verdict REPLICATE (same
   checkpoint, repeated n600 verdict; plus one across-decode replicate) measures σ_meas so
   rank-deficient/under-floor directions are DECLARED, not guessed. (w-passthrough in 7.1 makes
   σ_meas the campaign's resolution limit: no ILC iterate can attribute below it.)
3. **MATCHED INSTRUMENTS (req-H):** e_k must be the SAME observable each run — the measured
   counterexample is the crutch(174257Z)↔fix(015247Z) pair whose verdict semantics changed (seed
   leakage into the readout), making their d_seg values different observables. The parity row +
   effective-config provenance (F9) are the instrument-integrity checks; a semantics change
   invalidates the pair for ILC, full stop.

**Gap-identifiability statement (the requirement-P sentence):** the S-vs-S_floor gap (0.19110 −
0.118 ≈ 0.073 S) is identifiable iff every S-component slice — 100·d_seg per class per surface,
√(10·d_pose), λ_bytes·bytes per section — carries a per-run measured row with SE < the
attribution floor, AND the lever→component Jacobian estimate has full rank on A (item 1), AND
σ_meas (item 2) < floor. Missing any leg, the campaign converges to an UNATTRIBUTED plateau —
signal completeness is the convergence precondition, exactly as the operator pinned.

**Planned-signal audit against this condition:** per-class F-rows ✓ (slices) · F7/F8 lever
engagement/attribution ✓ (rank) · F9/F10 provenance ✓ (matched instruments) · parity row ✓ (once)
— MISSING: verdict replicates (σ_meas), per-stage-boundary decode parity (Δ̂ per stage, drift
within-run), live margin-quantile row (τ*_k self-derivation input), forecast-residual row (model
validity), signed per-class-pair flip rows (asymmetry re-tests). These five are §12's ADD list.

---

## §8 — EXTREMUM SEEKING: the measure→sweep→derive loop, made well-posed

### 8.1 The averaging theorem, mapped (INFERRED-from-literature: Krstić–Wang 2000)

ES converges to an O(a²) neighborhood of the optimum given timescale separation (plant fast,
dither slower, adaptation slowest) and a dither the output can SEE above noise. Our loop
(requirement M(2): sweep → fitted surface → derived local law → next sweep at the implied
optimum) is finite-difference ES at the campaign timescale; the response surfaces (#170) are the
averaged gradient estimates.

### 8.2 THE LAW — attributable-dither sizing + the minimal signal set (DERIVED, plugged)

Gradient-estimate usability requires the dither's output swing to clear the attribution floor:

  **sweep step a* = max( 2·floor_S / ĝ , knob resolution ),  ĝ = fitted first-order gain.**

Plugged for the rate knobs (ĝ = λ_bytes = 6.6586e-7 S/B exactly): a byte-side sweep step must be
≥ 2×0.00178/6.6586e-7 ≈ **5.3 KB per arm at the margin floor — and ≥ 45 KB at today's 0.015
recon-gap floor, i.e. FINE rate sweeps (waterfill depth steps of a few KB) are unattributable
until req-F#6 lands. The ckpt-fidelity fix is a PRECONDITION of the rate-sweep program**, not
hygiene (requirement P, ES leg). For loss-side knobs, ĝ comes from the surface fit; any arm with
predicted |ΔS| < floor is pre-registered as UNATTRIBUTABLE-BY-CONSTRUCTION and not launched
(formalizes req-J(2) as the ES admissibility rule).

**Minimal signal sets for well-posedness (requirement P(c)):**
- ES: per-arm (u, y) rows under MATCHED conditions (req-H) + σ_meas replicate (else a*
  incomputable) + the F9 effective-config row (else u is not known — the freq_along silent-tool
  confound class).
- MPC (§3): the AIC meat forecast per class (F3) + the forecast-RESIDUAL row (predicted vs
  realized per cadence, per model id — else N* = 2-cadence validity is asserted, never checked) +
  anneal-state row F1 (the known-τ-path internal model) + transition-cost table (measured forfeit,
  §2.2c) — all but the residual row are planned; the residual row is ADD #4 in §12.

---

## §9 — DUAL CONTROL (Feldbaum): pricing run-1's information (requirement N + P(b))

### 9.1 The frame

Optimal control under unknown parameters is DUAL: actions both exploit (score now) and probe
(identify for later). Certainty-equivalence is suboptimal when identification changes future
decisions. Run-1's dual value = EVSI = E[best v5 decision | signal] − E[best v5 decision | prior].
For a binary design decision with prior p on "mechanism works" and per-arm swing |ΔS_arms|
between right and wrong v5 choices, EVSI ≈ min(p, 1−p)·|ΔS_arms| (decision-theoretic lower form).

### 9.2 THE PRICE TABLE — expected v5 ΔS per candidate signal (DERIVED estimates; priors stated)

| run-1 signal | v5 decision it flips | swing |ΔS_arms| | prior p | **EVSI (S)** | status |
|---|---|---|---|---|---|
| pose-through-witness row (F11, w_pose>0) | ship L3 FiLM mechanism vs L1 Jacobian fallback | pose term 0.105 (R1 floor) vs ~0.017 (success 3e-5) ⇒ ~0.088 | 0.5 (never fired) | **~0.044** | planned ✓ — the single most valuable signal in the campaign; dominates everything else by 10× |
| big-3 anneal-completion recovery (F1+F3 split legs, §2.2d) | v5 allocates to schedule-completion vs representation levers | demoted band ~[0, 1e-3] ⇒ ~5e-4 within-run + arm-selection ~2e-3 | 0.5 | **~1e-3** | planned ✓ (instrumented split τ-leg/β-leg) |
| per-stage decode parity Δ̂_k trajectory | trust/repair the byte-close per stage; catches within-run decode drift before it eats a crossing | insurance on Δ = 0.0104 S; p_drift ≈ 0.2 | 0.2 | **~2e-3** | **MISSING → ADD** |
| verdict replicate σ_meas | threshold/floor calibration for every trigger + ILC resolution limit | mis-set trigger costs forfeit-class 5.4e-4–2.7e-3 per run | 0.4 | **~6e-4** | **MISSING → ADD** |
| per-class ν_c (online contraction fits) | eps_c veto calibration (pooled-slope cancellation guard) | misfire cost = forfeit class ~5.4e-4–2.7e-3 | 0.3 | **~4e-4** | planned-partial (F3 per-class tail fits; emit ν_c explicitly) |
| signed per-class-pair flip rows | admit/kill the one-sided-hinge lever family (L-asymmetry; UniWARD flagship re-test) | lever-family ΔS if real ~1e-3 | 0.3 | **~3e-4** | **MISSING → ADD** ($0 from cached fields first) |
| live margin-quantile m_q row | TAIL_k τ*_k self-derivation + adaptive-ε τ-law + τ_end confirm | enables the tail (unmeasured bet, order 1e-4–1e-3/cycle) | enabling | **~3e-4·k** | **MISSING → ADD** |
| forecast-residual row | MPC horizon validity (N* online) + λ(t)-decay model selection | prevents ep450-class band misses steering exits; ~5e-4 | 0.4 | **~2e-4** | **MISSING → ADD** |

Reading (requirement N quantified): summed EVSI of run-1's instrument program ≈ 0.048–0.05 S of
expected v5 decision value, vs run-1's own crossing probability 1.6–6.4% × (0.19110 − S_run1) —
**the instrument value exceeds the direct crossing value by roughly an order of magnitude**, and
one avoided wasted run (12–35 h at 42 s/ep) rides on top. PowerPlay-consistent ordering: the
table IS the duty-to-measure ranking for the signal adds (pose row first — already planned;
decode-parity trajectory second).

---

## §10 — STOCHASTIC APPROXIMATION / TWO-TIMESCALE: do our timescales separate?

### 10.1 The condition (INFERRED-from-literature: Borkar two-timescale SA; ODE method)

Fast loop (per-step SGD) must equilibrate before the slow loop (controller) moves: the slow
actor should act on windows ≥ the fast system's settle time, else it chases transients.

### 10.2 THE CHECK — one marginal window found (DERIVED, plugged)

Fast settle = 3/ν ≈ 115 ep (§2.2). Slow surfaces: per-cadence classifier (25 ep — acts on 1
window: WOULD chase transients, but the TRANSITION_TRANSIENT state already discounts post-switch
verdicts — the backtest ep325 ✓ row is precisely this theorem honored); co-predicate window
V=4 × 25 = 100 ep = **0.87× the settle time — MARGINAL**; per-stage (250+ ep) ✓; per-run ✓.
LAW: any trigger consuming windowed slopes needs window ≥ 3/ν ≈ 115 ep or an explicit
transient-discount state. **Fix: V = 4 → 5 (125 ep ≥ 115), one flag value.** Also ratified:
SGD-as-ODE's asymptotic O(1/t) tail is the weak-KAM anchor already registered for
powerlaw_meat_exit — the meat forecast's model class is theorem-consistent (no change).

---

## §META — Ashby, the review machine, and where the loop is still open

Requisite variety: a regulator must carry at least the variety of the disturbances it must
absorb. The H→P requirement ladder is variety INJECTION into the design regulator — each letter
was an operator catch that the machine now generates internally (requirement N(4)'s maturity
metric). Where the campaign controller is STILL OPEN-LOOP, honestly: (1) **representation/basis
choice** — chain-A proved the wall is the basis; no in-run actuator exists for it; it closes only
at the ILC/campaign timescale (Arm-A arms across runs). (2) **verdict-semantics integrity** — the
one measured sensor-fault class (seed-leakage changed the observable); the parity row + F9 are
fault-DETECTION residuals, but no automatic quarantine exists: a run with a semantics change
should be auto-excluded from ILC pairs (small build, campaign layer). (3) **the operator GO
gate** — deliberately open (CONTAINMENT); the controller's job is to make the GO packet
signal-complete, which is exactly requirement P. (4) **meta-meta:** the design process itself now
has a measured contraction record (v1→v2→v3 finding counts 17→6+13→blocker-free-with-PARTIALs);
if the per-round finding rate stops contracting, requirement N(3)'s inflection rule fires on the
DESIGN family itself — switch families (new representation), don't polish.

---

## §11 — RANKED ADOPTION LIST (predicted band + cost + $0 probe each)

Ranked by expected S-impact × (1/cost), crossing-margin-denominated (0.00178 S = 1 margin):

| rank | import | expected effect | cost | $0 probe (pre-registered band · kill) |
|---|---|---|---|---|
| 1 | **§3.3 forfeit-matched exit s\* = ν·forfeit = 1.41e-5 S/ep** (fire ~60 ep later) | +5.4e-4 S (0.30 margins) at +~42 min wall-clock | ~10 LOC would-fire arm + req-B tests | **P-CT3**: backtest on mod32cap 41-row trace; band: first fire ep670–700 (vs shipped ep625), EMA-best at fire within 1 cadence of ep650; kill: fires <ep650 or >cap 726 |
| 2 | **§7.2 decode-gap ILC feedforward** (train bar 0.00100; EWMA ω=0.5) | protects 5.86 margins of realized-gap allowance; makes the crossing triple honest train-side | 0 LOC (a target-number change + one ledger row) | none needed (arithmetic); falsified if run-1 parity row Δ outside [0, 3e-4] d_seg |
| 3 | **§4.2 self-triggered verdict cadence** Δt = clamp(floor/\|Ŝ′\|, 25, 100) | −30–40% n600 verdicts late-run, 0 score cost | ~15 LOC + req-B tests | **P-CT2**: replay cadence law on the 41-row trace; band: 12–17 verdicts skipped, no missed best >1 cadence; kill: any missed best >1 cadence |
| 4 | **§10.2 co-predicate window V=4→5** (two-timescale separation) | removes transient contamination of the exit trigger (misfire cost 0.3–1.5 margins) | 1 flag value | **P-CT1**: refit ν per stage from the trace; band ν ∈ [0.02, 0.035]/ep; kill: ν <0.01 ⇒ recompute all §2/§4/§5 window laws |
| 5 | **§1.4 PMP stop-rate would-fire row** ε_stop = floor_S/cadence (7.1e-5 S/ep now; operating-point conversion printed) | calibration signal for run-2 trigger re-derivation; 0 behavior change run-1 | ~5 LOC (F4 audit row variant) | rides P-CT3's replay |
| 6 | §3.4 adaptive-ε feedforward (evaluate law at τ(t+25)) | earlier stability-edge arrival; unquantified | ~5 LOC, gated on clamp-binding check | the queued negatives-review clamp-binding probe gates it (>90% binding ⇒ moot until §6.2 re-derivation) |
| 7 | §6.2 clamp→projection re-derivation (gated) | revives an INERT law (adaptive-ε was constant 0.3 — measured) | ~10 LOC, gated same as 6 | same probe |
| 8 | §2.3/§5.1 TAIL_k budget + dwell laws (cycle ≥ 265 ep; k_max 3–7 @3000 ep; dwell ≥115 ep) | prevents TAIL cycles that measure their own transient | doc rows in the TAIL build (already ~40 LOC planned) | none ($0 arithmetic; validated by run-1 TAIL telemetry) |
| 9 | §9.2 EVSI table → duty-to-measure re-ranking | campaign-layer decision quality | 0 LOC (ledger ordering) | — |
| 10 | §7.3 ILC pair-admission rule (matched-instrument quarantine) | protects the campaign from the measured crutch↔fix non-comparability class | ~10 LOC campaign-ledger check | — |

DEAD imports, for the record: direct LQR/Riccati (no linear plant) · grid/NN HJB value solve
(dimension; the value function enters only through λ = ∇V estimation) · continuous ES dither on
θ itself (per-step dither would fight the optimizer; campaign-timescale FD-ES only).

---

## §12 — WHAT v4 / RUN-1 CHANGES NOW (≤5 items) + THE ≤5 SIGNALS TO ADD (req P) + campaign layer

### 12.1 Run-1 / v4 config changes (exact flags/values; all req-B-tested before launch)

1. **Second co-predicate arm (would-fire first):** forfeit-matched threshold
   `--tau-fin-slope-star 1.41e-5` S/ep (per-epoch normalized), shipped arm untouched; promote to
   firing arm iff P-CT3 passes. [§3.3; +0.30 margins expected]
2. **`--copred-verdict-window 5`** (V=4→5; 125 ep ≥ 3/ν=115). [§10.2]
3. **Train-side d_seg bar 0.00100** in the crossing triple (decoded 0.0011 − Δ̂ 1.0427e-4); Δ̂
   EWMA row (ω=0.5) added to the campaign ledger. [§7.2]
4. **Self-triggered verdict cadence** `--verdict-cadence-law "clamp(0.00178/|slope|,25,100)"`
   (floor = today's cadence ⇒ degrades safely; alarms stay per-epoch). Gate on P-CT2. [§4.2]
5. **PMP stop-rate + clamp-binding F-rows** (audit-only): ε_stop would-fire row + adaptive-ε
   binding fraction row. [§1.4, §6.2 — zero behavior change, feeds run-2 re-derivations]

### 12.2 THE ≤5 SIGNALS RUN-1 MUST ADD to be signal-complete for the campaign (requirement P)

(Planned F1–F12 + per-class rows + parity row audited against the §7.3 identifiability
condition; these five are the gaps. Each is score-neutral read-only ⇒ defaults ON.)

1. **Verdict replicate σ_meas row** — once per stage boundary: repeat the n600 verdict on the
   same checkpoint (and once across decode). The campaign's resolution limit; every threshold
   law's denominator. [~10 LOC]
2. **Per-stage-boundary decode-parity row Δ̂_k** — byte-close + decode + verdict at each stage
   checkpoint (pieces all exist; per-stage ckpts mandated). Catches within-run decode drift;
   makes the ILC feedforward per-stage identifiable. EVSI ~2e-3 S. [~20 LOC orchestration]
3. **Live margin-quantile row m_q(t)** — per-verdict flip-annulus margin percentile (the #333
   machinery, emitted per cadence). Input to TAIL τ*_k = m_q(k)/ln5 self-derivation, the τ_end
   confirm, and the adaptive-ε τ-law. [~15 LOC]
4. **Forecast-residual row** — every cadence: (model id, predicted ΔS, realized ΔS). Makes MPC
   horizon validity (N*=2 cadences) measurable online; the ep450-class miss becomes visible in
   flight. [~10 LOC]
5. **Signed per-class-pair flip-mass row** — directional (c_i→c_j vs c_j→c_i) flip counts per
   verdict (first: the queued $0 cached-field probe). Feeds the L-asymmetry re-tests and the
   one-sided-hinge lever family decision. [~15 LOC]

### 12.3 Campaign layer (not run-1)

Newton-ILC gain γ=0.7 update rule + excitation-rank check before each run's config freeze
(§7.1/7.3) · EVSI table as the duty-to-measure ranking (§9.2) · ILC pair-admission
(matched-instrument quarantine) (§7.3(3)) · family-inflection watch on the design process itself
(§META(4)) · run-2 trigger re-derivation from the §12.1(5) audit rows.

---

*Positive-design contract check: §1 law (ε_stop, ratification) · §2 law (budget allocation +
115-ep floor) · §3 law (s\* = ν·forfeit) + builds B-CT1/B-CT2 · §4 law (cadence) + B-CT3 + probe
P-CT2 · §5 laws (dwell + mode admission) · §6 law (ramp bound) + gated B-CT4 · §7 laws
(feedforward + identifiability condition) · §8 law (dither sizing + signal sets) · §9 price
table · §10 law (window) + probe P-CT1 · every claim labeled; every plugged number from the
measured corpus. Wall-clock statements honor L59 (time lexicographically secondary).*
