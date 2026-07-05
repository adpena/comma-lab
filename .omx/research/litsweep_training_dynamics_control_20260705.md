# LITERATURE SWEEP — training dynamics, optimization geometry, and control (the 5 domains the eikonal-runaway incident cracked open)

**Date:** 2026-07-05 · **Agent:** LITSWEEP-DYNAMICS · **Operator directive:** *"explore the related
domains and related papers with fields grade media passion and curiosity and deep math … take as
long as you wish and need."*
**Axis discipline: every internal number cited below is `[n24 advisory]` / `[macOS advisory]`
NON-PROMOTABLE; every literature claim is judged against OUR measured anchors, never adopted from
an abstract. Pointer contest-CPU 0.19110 UNMOVED — this whole memo is MEANS.**

Anchors engaged (proactive recall done first): `stepping_instability_diagnostic_20260705.md`
(eikonal runaway, step-size-gated; restored-moments DAMP, fresh-moments 25.3× worse) ·
DAG FEED-05o/05q (bracket tightened: **lr 5e-4 STABLE-fastest, 9.1e-4 unstable**; v4 live) ·
`council_grand_symposium_curriculum_derivation_20260705.md` (triggers-not-clocks; Γ/nucleus laws) ·
`council_grand_symposium_levelset_loss_geometry_20260705.md` (16-term telemetry; γ*=0 HOLD) ·
`costate_controller_design_20260705.md` (Phase A shadow; λ-decay probe queued) ·
`tao_dimensional_analysis_pi_groups_for_witness_20260705.md` (π_train form registered, constant
pending) · `deepmath_amortizing_argmax_maslov_caustic_tau_eps_hbar_20260704.md` ·
`muon_deep_dive_keep_and_tune_finishing_stage_schedule_not_switch_20260703.md`.

---

## DOMAIN 1 — Neural-SDF eikonal instability (StEik and its lineage)

**The delight:** the literature not only *predicts* our runaway — it proves the continuum object is
ILL-POSED, which sharpens (and partially contradicts) our "step-size-gated" verdict.

**StEik (Yang, Walker, Parkinson et al., NeurIPS 2023, arXiv 2305.18414).** Analyzes the gradient
FLOW of the eikonal penalty `(|∇u|−1)²` in the continuum (infinite-capacity) limit and shows it is
an **unstable PDE**: linearized about a near-SDF state, the flow is anti-diffusive (backward-heat-
like) in the **direction of ∇u — exactly our "divergence along the SDF-slope direction."** The
instability is intrinsic to the LOSS, not to any particular discretization. Their cure ladder:
DiGS-style full Laplacian penalty `(Δu)²` stabilizes but OVER-SMOOTHS (kills fine detail — for us,
the lane dashes); StEik's contribution is a **directional (normal-direction) second-order
penalty** ("directional divergence" / Laplacian-of-normal restricted along ∇u) that damps only the
unstable mode and leaves tangential curvature (fine geometry) free. Bonus: with stability restored
they can afford higher-frequency architectures (quadratic layers).

**ViscoReg (arXiv 2507.00412; outperforms SIREN/DiGS/StEik/HotSpot on ShapeNet/SRB).** Grounds the
same fix in **viscosity-solution theory**: the eikonal equation is ill-posed precisely because it
admits infinitely many weak solutions; the *viscous* eikonal equation `|∇u| = 1 + ε·Δu` selects the
viscosity solution (the true SDF) as ε→0. ViscoReg trains with a **vanishing-viscosity eikonal
term, decaying ε to zero on a schedule** — a *continuation in ε*, i.e. literally our curriculum
philosophy (Γ-convergence/GNC) applied to the eikonal term itself. This answers the memo's hunt
question "why is a soft eikonal penalty the unstable discretization": the penalty targets the
inviscid equation; the stable object is the ε-viscous one, approached from above.

**HotSpot (Wang et al., CVPR 2025).** Replaces eikonal supervision with a **screened-Poisson /
heat-method** loss (asymptotically sufficient condition for SDF-ness): convex-alternative lineage.
Bigger surgery — it swaps the constraint family, not a bolt-on term.

**Cross-check against OUR measurements (the most valuable rows):**
- **CONSISTENT:** runaway registered by the eikonal term, along |∇φ| (raw eikonal ≈2,070 at
  blowup) — the literature's exact unstable mode. Our finding that at lr 5e-4 the eikonal
  *reduces ~5×* in one epoch shows the discrete/finite-capacity flow is conditionally stable —
  also consistent: finite network + finite step act as regularizers of an ill-posed flow.
- **REFINEMENT/TENSION:** literature says the continuum flow is unstable at ANY step size — so
  **lr-reduction delays/gates the discrete onset but is not a structural cure**. Our `low_lr
  STABLE` verdict is a 60-step window (the Contrarian's caveat, now theory-backed): as capacity
  effectively grows (β_hosc anneal sharpening, tau descent) the discrete flow approaches the
  ill-posed continuum and the stability threshold can keep FALLING. Prediction registered: if the
  v4 (5e-4) run shows a SLOW eikonal creep re-emerging near tau@400 / late β, that is the
  continuum instability re-crossing the (lowered) discrete threshold — the cure then is the
  loss-level term (below), not another lr cut.
- Our RECOMMENDED FIX (rewarmup / lr) is correct as the *unblock*; StEik/ViscoReg supply the
  *structural* spare tire.

**Verdicts:** StEik **DRAW-FROM** (lever: default-OFF `--eikonal-directional-div-weight` penalty
on the normal-direction second derivative; exact term form to be lifted from the paper at build
time, never invented). ViscoReg **DRAW-FROM** (lever: vanishing-viscosity eikonal variant,
ε-schedule co-annealed with τ — it composes with our Γ/π_int machinery natively; arguably the more
"us" of the two since it is continuation-theoretic). HotSpot **WATCH** (constraint-family swap =
run-3+/θ* design item only). DiGS full-Laplacian **NOT-RELEVANT as a lever** (over-smoothing is
exactly our MCF-erasure enemy: it would eat the lane dashes).

---

## DOMAIN 2 — Edge of Stability (the ep92 organic death was textbook)

**The delight:** our incident is a *clean field observation* of progressive sharpening + an EoS
crossing under a nearly-constant lr, and the Adam-specific threshold law gives a **derivable
number for our measured bracket**.

**The core results.** Cohen et al. (ICLR 2021): full-batch GD drives sharpness λ_max(H) UP
(progressive sharpening) until it reaches `2/η`, then hovers ("edge of stability") — training
continues non-monotonically via oscillation along the top eigenvector. Damian-Nichani-Lee 2023
("Self-Stabilization"): the mechanism of hovering is a negative feedback — oscillation along the
sharp direction couples to the third-order term and *reduces* sharpness; **the oscillation is
functional, not pathological.** Cohen et al. 2022 (arXiv 2207.14484, "Adaptive Gradient Methods at
the Edge of Stability"): for Adam-family the object that equilibrates is the **preconditioned
sharpness** `λ_max(P^{-1/2} H P^{-1/2})`, at threshold

    λ_pre* ≈ (2/η) · (1+β₁)/(1−β₁)      [= 38/η at β₁ = 0.9]

Central flows (Cohen, Damian et al. 2024/25; "rod flow" for Adam, arXiv 2605.06821): the
time-AVERAGED EoS trajectory follows an ODE ("central flow") that includes an implicit
curvature-penalty — quantitatively predicts loss curves through the oscillatory regime.

**"Adaptive Preconditioners Trigger Loss Spikes in Adam" (arXiv 2506.04805).** THE paper for our
moments finding: spikes occur when the second-moment estimate **v_t decouples from (decays below)
the instantaneous squared gradients**, letting the preconditioned sharpness exceed 2/η for
sustained periods → spike. Five-stage spike anatomy + a gradient-directional curvature predictor.

**Cross-check against OUR measurements:**
- **CONFIRMED (and folklore inverted, jointly):** our measured "restored moments DAMP (6.7×) /
  fresh moments EXPLODE (25.3×)" is exactly 2506.04805's mechanism run in reverse: fresh v = the
  maximal decoupling (v near zero at a converged, sharp basin → huge effective steps along the
  sharp direction); restored v encodes the sharp direction's gradient history → smaller effective
  step there. The "stale moments are dangerous" folklore we falsified per-arm is falsified by the
  literature too. Do NOT reset moments on resume — now doubly grounded.
- **THE DERIVABLE LAW (candidate canonical equation, FORMALIZATION_PENDING):** our bracket
  [5e-4 stable-fastest, 9.1e-4 unstable] + λ_pre* = 38/η predicts the ep100 basin's preconditioned
  sharpness sits in **λ_pre ∈ [38/9.1e-4, 38/5e-4] = [4.2e4, 7.6e4]** — a factor-1.8 window. $0
  probe: power-iteration HVP on the restored snapshot, preconditioned by the restored v̂^{-1/2}
  (n24, CPU). If the measured λ_pre lands in-window, the bracketing upgrades to a LAW
  (`eos_adam_preconditioned_threshold_v1`) and per-stage lr caps become DERIVED:
  `η_max(t) ≈ 38/λ_pre(t) × margin` — consumable by the costate controller as the lr admission
  constraint, and it retro-derives v1's organic ep92 death (progressive sharpening under hosc-β
  anneal raised λ_pre until 38/η crossed at the barely-decayed cosine η ≈ 9.2e-4).
- **π_train closure (Tao memo B.7):** the missing "sharpness trace per stage" IS this probe; the
  dimensionless training group is `π_EoS = η·λ_pre/38` (stability iff π_EoS ≲ 1) — a cleaner form
  than lr×steps×sharpness for the stability half.
- **SPIKE-GUARD TENSION (new insight):** self-stabilization says EoS oscillation is the mechanism
  by which training reduces sharpness. Our guard median-freezes and SKIPS everything above
  5×median → at an EoS crossing it *blocks the self-stabilization feedback* and produces the
  absorbing deadlock we measured three times. The guard was "CORRECT" as an alarm; as an actuator
  it fights the physics. Redesign (default-OFF): tolerate bounded oscillation (skip only on
  sustained MONOTONE growth of the per-term canary, e.g. eikonal trough-ratio > 2 over k epochs),
  and respond with rollback-to-best + lr×0.1 (projection-free costate actions) instead of
  skip-with-frozen-median.

**Warmup (why rewarmup works).** Kalra & Barkeshli (arXiv 2406.09405) + Gilmer et al. 2022: small
initial lr lets the iterate move into FLATTER regions (natural sharpness reduction) so the target
lr becomes admissible; warmup is a controlled descent of `η·λ` toward the EoS line rather than a
jump above it. This is a literature CONFIRMATION of the diagnostic's resume-as-stage-boundary
rewarmup fix (floor 0.1 → ramp): resuming at a sharpened basin with η above 38/λ_pre is precisely
the pathological case warmup exists for. (2509.07972 gives a convergence-acceleration analysis;
2410.23922 shows warmup need shrinks with better init scaling — our per-stage version: rewarmup
need shrinks if the boundary re-treat is gentle.)

**Verdicts:** EoS threshold law **DRAW-FROM #1** (the λ_pre probe, $0). 2506.04805 **DRAW-FROM**
(mechanism cited in the moments-restored law; its curvature spike-predictor is the principled
guard trigger). Central flows **WATCH** (quantitative loss-curve prediction through oscillation —
future forecast instrument for the costate controller's λ(t)-decay gap). Warmup papers
**DRAW-FROM (confirmation)** — no new lever needed; the fix already matches the mechanism.

---

## DOMAIN 3 — Per-term loss balancing (what the 16-term telemetry can feed)

**The landscape.** Three families: (i) **gradient-balancing** — GradNorm (Chen et al., ICML 2018:
tune w_i so per-term gradient norms track relative training rates), PCGrad (Yu et al. 2020:
project out conflicting gradient components), ConFIG (2024: conflict-free combination with
guaranteed positive alignment to every term — built FOR PINNs); (ii) **uncertainty weighting**
(Kendall et al., CVPR 2018: w_i = 1/2σ_i², σ learned — principled but assumes likelihood
semantics our geometric terms don't have); (iii) **NTK-informed** — Wang-Yu-Perdikaris ("When and
why PINNs fail to train: an NTK perspective", JCP 2022): in the linearized regime each loss
component converges at a rate set by its NTK block's eigenvalues; setting `w_i ∝ tr(K)/tr(K_i)`
EQUALIZES convergence rates. Earlier Wang-Teng-Perdikaris "gradient pathologies" (2020) is the
practical version: `ŵ_i = max|∇L_r| / mean|∇L_i|`, EMA-smoothed (learning-rate annealing for loss
weights). Recent: AutoBalance (2510.06684), Dual-Balancing (2505.11117), BRDR — benchmark result
(2606.04125): NTK-weighting typically lowest error on stiff systems. The PINN community fights
exactly our shape of problem: a PDE-residual term (their eikonal-analog) that DOMINATES and
destabilizes.

**Cross-check against OUR measurements (a sharp nuance the literature would get wrong):**
- The generic PINN prescription "down-weight the dominating residual term" would be **WRONG
  here**: our per-arm matrix proved the eikonal term is the **CANARY, not the cause** (no_bd and
  v1_pure explode identically; seg stays flat ~8-9; the diagnostic's own words: "the eikonal term
  is the canary here, not the underdog"). Naively GradNorm-ing our 16 terms would have *reduced*
  the eikonal weight during the runaway — silencing the alarm while the SDF-slope divergence
  continued (it is an OPTIMIZER instability in weight space, visible through whichever term
  measures |∇φ|). Loss balancing addresses gradient-share allocation, NOT EoS crossings. Keep the
  two failure classes separate in the controller's vocabulary.
- Where balancing IS our shape: Shannon's γ-waterfilling (loss-geometry symposium) and the
  measured γ*=0 verdict are already an equalization analysis — the focal calibration measured the
  island gradient share and found no starvation. The NTK/gradient-statistics machinery is the
  PRINCIPLED generalization of that probe to all 16 terms: compute per-term gradient-norm shares
  (we have the telemetry hooks; a no-grad recompute per term) → the costate controller gets
  **measured w_i recommendations with the same honesty ladder** (PARTIAL tier: direction/ranking
  evidence).
- **Tishby constraint (ours, binding):** dynamic weights = continuous objective change; the
  curriculum law says lever/objective changes belong at stage boundaries or in CE, never
  mid-τ-descent. So: slow-EMA quasi-static updates, CE-stage-only, frozen through the anneal —
  a discipline the PINN literature does not have and we do.

**Verdicts:** Wang-Perdikaris gradient-statistics + NTK weighting **DRAW-FROM** (lever: per-term
gradient-share probe on checkpoints ($0, extends the focal harness), feeding *recommended* static
w_i per stage — not per-step GradNorm; fire only at stage boundaries). ConFIG **WATCH** (if
per-term gradient CONFLICT is ever measured between seg and eikonal/bd — measure first; PCGrad
family adds per-step cost and trajectory nondeterminism concerns for our bit-exact resume paths).
Kendall uncertainty weighting **NOT-RELEVANT** (no likelihood semantics for eikonal/length/
persistence terms; would launder geometry into fake σ²). GradNorm-as-per-step-controller
**NOT-RELEVANT-NOW** (violates the stage-boundary discipline + would have silenced the canary).

---

## DOMAIN 4 — Optimal control of training (the costate controller has ancestors)

**The landscape.** (i) **SGD-as-SDE optimal control:** Li-Tai-E (stochastic modified equations,
JMLR): lr schedule = control of an SDE; for quadratic-plus-noise the optimal policy is
**constant-then-decay** (hold lr while signal-dominated, anneal ~1/t when noise-dominated). The
Feb-2026 random-feature treatment (arXiv 2602.04774) computes optimal schedules analytically via
optimal control: same shape — extended constant phase then power-law/exponential tail (the modern
WSD schedule is the empirical twin). (ii) **PMP for deep learning:** Li et al. (JMLR 2018,
method-of-successive-approximations; 1803.01299 discrete-weight): trains networks by iterating
Hamiltonian maximization — validates the Pontryagin frame as more than metaphor. (iii)
**Hypergradients:** Baydin et al. 2018 (lr as learnable via ∂L/∂η), GreedyLR (Amazon 2023,
zeroth-order greedy adaptation): cheap, myopic, no stability guarantee at EoS (a hypergradient
step can push η above 38/λ_pre exactly when the landscape sharpens). (iv) **PBT** (Jaderberg
2017): population-based online hyperparameter evolution — needs a population; our fleet is one
Mac; N/A now, natural fit for the tertiary-sweep tier later. (v) **Bang-bang:** in
control-constrained Bolza problems optimal controls are bang-bang off singular arcs — the
curriculum symposium's §B.4 decomposition (continuous controls on singular arcs, discrete
stage-switches as impulses) is the textbook-correct structure, independently derived.

**Cross-check against OUR design:**
- The costate controller (Phase A shadow, honesty ladder, never-regress acceptance) is *ahead* of
  most of this literature in one respect — measured marginal-ΔS with propagated uncertainty and
  honest UNIDENTIFIABLE refusals — and *behind* it in one: the literature's schedules are derived
  from a MODEL of the dynamics (SDE/central flow), while our λ extrapolation is local-linear
  (the measured ep450 below-band miss). The λ(t)-decay probe already queued is exactly "fit a
  model before projecting" — central flows (Domain 2) is the principled model family to try after
  exponential decay.
- Constant-then-decay optimality + the EoS cap combine into a derived lr policy shape:
  `η(t) = min(η_target, 38/λ_pre(t)·margin)` during descent, then the tail anneal — which is what
  the v4 flag surgery (flat 0.5×) approximates and what Muon's lr-final-frac 0.1 already does for
  the finisher (muon-deep-dive GAP2 = the decay leg, literature-aligned).
- Greedy/hypergradient lr adaptation **contradicts** our incident record: any controller that
  raises η on "loss improving" walks into the ep92 class of death. Only stability-constrained
  schedules (EoS cap) are admissible here.

**Verdicts:** Constant-then-decay optimal-control results **DRAW-FROM (confirmation + form)** —
the lr policy form above goes into the Phase-B reference-trajectory design; no new build. PMP/MSA
**WATCH** (framing already absorbed via #247). Hypergradient/GreedyLR **NOT-RELEVANT** (unstable
at EoS; wrong direction for our failure class). PBT **WATCH** (tertiary/fleet tier only).
Random-feature optimal schedules (2602.04774) **WATCH** (toy model; read if we ever derive the
1e-3 endpoint from first principles instead of the PR95 echo).

---

## DOMAIN 5 — Curriculum = annealing = continuation (what the theory says about timing)

**The landscape.** (i) **GNC (Blake-Zisserman 1987)** — the original: a convexity parameter is
relaxed on a schedule; modern robust-estimation GNC (Yang et al. 2020) uses a GEOMETRIC μ schedule
(×~1.4/iteration) as standard practice. (ii) **Gaussian homotopy:** Mobahi-Fisher 2015 — Gaussian
smoothing continuation relates to the convex envelope (the best convex under-approximation);
optimality of the smoothing path. (iii) **Hazan et al. 2016 (graduated optimization, ICML):**
first PROVABLE guarantee — for "σ-nice" functions, a **geometric (halving) σ-schedule** converges
to the global optimum in poly time; the schedule that theory blesses for continuation is
GEOMETRIC. (iv) **Single-loop Gaussian homotopy (2203.05717):** joint descent in (x, σ) — the
continuation parameter becomes a trained variable (cf. our online τ-ODE idea in θ*). (v)
**Simulated annealing:** Hajek 1988 — convergence to global minima iff Σexp(−d*/T_k) = ∞, i.e.
`T_k = c/log k` with c ≥ d* (the deepest non-global basin depth); geometric cooling forfeits the
guarantee (it can freeze into a local min) but is universal practice. (vi) **Numerical
continuation (Allgower-Georg):** predictor-corrector with **adaptive step control** — the
continuation step Δλ is chosen by whether the corrector CONVERGES (readiness), not by a
prescribed clock; step-size halving on failed correction.

**Cross-check against OUR derivations (strong convergence, one honest tension):**
- **Geometric schedules:** Hazan's provable schedule + GNC practice + our Fisher-Rao
  constant-velocity derivation (geometric τ/β, equal-epochs-per-octave, CV≈0.39 confirmed) all
  land on GEOMETRIC. Three independent routes, one answer — the geometric-β BUILD (run-3 spec
  item 4) is now theory-triangulated.
- **Triggers-not-clocks:** Allgower-Georg adaptive continuation IS the formal ancestor of the
  curriculum symposium's readiness-trigger law (fire CE→tau on plateau + per-class nucleus =
  advance the homotopy parameter only when the corrector has converged at the current parameter).
  The run-3 event-trigger spec is standard numerical-continuation discipline, arrived at
  independently. Naming this lineage strengthens the spec, changes nothing in it.
- **Honest tension (recorded, resolved):** Hajek says the only GUARANTEED annealing schedule is
  logarithmic (c/log t), and geometric cooling can freeze local minima. We knowingly take the
  geometric path because our object is different: we are not doing stochastic global optimization
  over a fixed landscape — we are TRACKING a continuation path (Γ-limit) where the τ-parameterized
  problems are adiabatically connected and the nucleus/plateau triggers detect tracking failure
  (the analog of a failed corrector step). If a future run shows systematic freezing into
  sub-optimal partitions across seeds, the log-schedule tail (slow final octaves) is the
  literature's remedy — WATCH condition registered.
- **ViscoReg cross-link:** the vanishing-viscosity ε-schedule (Domain 1) is a continuation in the
  SAME formal sense; if built, its schedule should be geometric and boundary-relative, inheriting
  the whole §C.ii machinery — one discipline, another parameter.

**Verdicts:** Hazan graduated-optimization + GNC + Allgower-Georg **DRAW-FROM (confirmation +
naming)** — no new lever; they certify geometric shape + trigger-based advancement and give the
freeze-detection WATCH condition. Mobahi-Fisher **WATCH** (envelope theory; relevant only if we
ever derive the CE-stage smoothing kernel formally). Single-loop homotopy **WATCH** (θ*/run-3
online-τ design input). Hajek **DRAW-FROM (guardrail)** — the freeze-vs-track distinction is now
on the record.

---

## LITERATURE-vs-OUR-MEASUREMENT CONTRADICTION LEDGER (the most valuable rows)

| # | our finding | literature position | resolution |
|---|---|---|---|
| 1 | "instability is STEP-SIZE-GATED; low-lr STABLE" (60-step window) | StEik/ViscoReg: continuum eikonal flow ill-posed at ANY step size | Both true at different limits: lr gates the DISCRETE onset; sharpening (β/τ anneal) can lower the threshold under a fixed lr. lr-cut = unblock, loss-term = structural cure. Pre-registered watch: slow eikonal creep near tau@400 at 5e-4 ⇒ build the StEik/ViscoReg term. |
| 2 | "restored moments DAMP (6.7× vs 25.3×)" — folklore inverted | 2506.04805: spikes = v_t decoupling from g²; fresh v = maximal decoupling | CONFIRMED, no tension: literature and our per-arm matrix agree against the folklore. Never reset moments on resume. |
| 3 | spike guard "CORRECT all three times" | Self-stabilization/central flows: EoS oscillation IS the sharpness-reduction mechanism; skipping it blocks recovery | Guard correct as ALARM, counter-productive as ACTUATOR (median-freeze deadlock ×3 measured). Redesign: bounded-oscillation tolerance + rollback-and-lr-cut response. |
| 4 | (hypothetical naive import) down-weight the dominating term | PINN balancing: residual dominance → rebalance | WRONG here: eikonal is the canary, not the cause (per-arm falsification). Balancing ≠ stability; keep the failure classes separate in the controller. |
| 5 | geometric τ/β anneal (Fisher-Rao derived) | Hajek: only log-schedules guarantee global convergence | Different objects (adiabatic tracking vs stochastic global opt). Geometric stands; freeze-detection = the registered fallback trigger. |

## RANKED NEXT-ACTIONS ($0 probes first; expected ΔS-per-effort against S = 100·d_seg + √(10·d_pose) + 25·bytes/37.5M)

| rank | action | cost | expected value | consumer |
|---|---|---|---|---|
| 1 | **λ_pre probe** (power-iteration HVP, preconditioned by restored v̂^{-1/2}, on the ep100 snapshot, n24 CPU): test λ_pre ∈ [4.2e4, 7.6e4] = 38/η bracket | $0, ~1 unit | upgrades the lr bracket to a LAW; derives per-stage lr caps η_max(t)=38/λ_pre(t)·margin; retro-derives ep92; closes the π_train sharpness gap | costate controller (lr admission), run-3 spec, `eos_adam_preconditioned_threshold_v1` |
| 2 | **eikonal-creep watch** on v4 near tau@400 (already-flowing per-term telemetry; zero new code) — pre-registered discriminator for contradiction row 1 | $0 | decides whether the StEik/ViscoReg loss-term build is needed at all | run-forward gates |
| 3 | **guard redesign** (default-OFF): bounded-oscillation tolerance + rollback-to-best + lr×0.1 response (replaces skip-with-frozen-median at EoS crossings) | small build | removes the measured ×3 deadlock absorbing state; aligns the actuator with EoS physics | trainer build-4; costate Phase-B projection-free actions |
| 4 | **StEik/ViscoReg term** as default-OFF flag (directional-divergence OR vanishing-viscosity eikonal; geometric ε-schedule, boundary-relative), byte-identical when OFF; FIRE only on rank-2's creep signal | build + A/B | structural cure for the one measured instability class of this vehicle; protects the whole remaining anneal | run-3 curriculum |
| 5 | **per-term gradient-share probe** (extend the focal harness to all 16 terms on ep-checkpoints; Wang-Perdikaris statistics + NTK-style shares) → *recommended* stage-boundary w_i with costate honesty tiers; Tishby rule binding (CE/boundary only) | $0 probe; opt-in weights | converts static weights into measured ones where equalization actually binds (γ*=0 showed islands don't — the bulk-term shares are unmeasured) | costate controller PARTIAL tier; run-3 |

Non-actions (explicitly): no GradNorm/PCGrad per-step controller (row 4 + determinism), no
hypergradient lr (EoS-unsafe), no PBT (no fleet), no Kendall σ² (semantics), no HotSpot swap
(vehicle surgery out of scope), no log-schedule switch (row 5 resolution).

## CANDIDATE CANONICAL EQUATIONS — FORMALIZATION_PENDING (do NOT register unanchored)

1. `eos_adam_preconditioned_threshold_v1`: stability iff `η · λ_max(P^{-1/2}HP^{-1/2}) ≲
   2(1+β₁)/(1−β₁)` (=38 at β₁=0.9); π-form `π_EoS = η·λ_pre/38 ≲ 1`. ANCHOR PENDING: the rank-1
   probe (measured bracket [4.2e4, 7.6e4] awaiting the direct HVP measurement).
2. `eikonal_continuum_illposedness_v1`: the (|∇φ|−1)² gradient flow is anti-diffusive along ∇φ in
   the continuum limit (StEik); discrete stability is (η, capacity)-gated; cure = normal-direction
   second-order damping or vanishing viscosity. ANCHOR PENDING: the rank-2 creep watch (positive
   or negative, it anchors the discrete-vs-continuum boundary for THIS vehicle).
3. `moments_restore_damping_v1`: on resume at a sharpened basin, restored second moments reduce
   the runaway amplification vs fresh moments (measured 6.7× vs 25.3×; mechanism = v_t/g²
   coupling per 2506.04805). ANCHOR EXISTS (per-arm matrix) — registration blocked only on
   equations-leg cycle + naming review; nearest-to-registrable.

## Sources (external)

- StEik: https://arxiv.org/abs/2305.18414 (NeurIPS 2023 proceedings id 2d6336c1c2987e9d1d9894edd593478d)
- ViscoReg: https://arxiv.org/abs/2507.00412 · HotSpot: Wang et al., CVPR 2025
- EoS: Cohen et al. ICLR 2021 · Adaptive EoS: https://arxiv.org/abs/2207.14484 · Rod flow (Adam EoS): https://arxiv.org/abs/2605.06821 · Adam spikes: https://arxiv.org/abs/2506.04805 · Self-Stabilization: Damian-Nichani-Lee 2023 · Central Flows: Cohen-Damian et al. 2024/25
- Warmup: https://arxiv.org/abs/2406.09405 (Kalra-Barkeshli) · Gilmer et al. 2022 · https://arxiv.org/abs/2509.07972 · https://arxiv.org/abs/2410.23922
- Balancing: GradNorm (Chen et al. ICML 2018) · PCGrad (Yu et al. 2020) · Kendall et al. CVPR 2018 · Wang-Teng-Perdikaris 2020 (gradient pathologies) · Wang-Yu-Perdikaris JCP 2022 (NTK) · ConFIG 2024 · AutoBalance https://arxiv.org/abs/2510.06684 · Dual-Balancing https://arxiv.org/abs/2505.11117 · PNP benchmark https://arxiv.org/abs/2606.04125
- Control: Li-Tai-E JMLR (SME) · Li et al. JMLR 2018 (MSA/PMP) · https://arxiv.org/abs/1803.01299 · https://arxiv.org/abs/2602.04774 (random-feature optimal schedules) · Baydin et al. 2018 · GreedyLR (Amazon Science) · Jaderberg et al. 2017 (PBT)
- Continuation: Blake-Zisserman 1987 · Yang et al. 2020 (GNC) · Mobahi-Fisher 2015 · Hazan et al. ICML 2016 · https://arxiv.org/abs/2203.05717 · Hajek 1988 · Allgower-Georg, *Numerical Continuation Methods*

**HARD GATE restated: pointer 0.19110 UNMOVED. Everything here is MEANS; the first milestone
remains a byte-closed `upstream/evaluate.py` n600 exact row.**
