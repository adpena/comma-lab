# Witness-native schedule derivation — BLINDED first-principles, then compared

- **UTC:** 20260709 · **Agent:** WITNESS-NATIVE SCHEDULE DERIVATION (Opus) · **[no-triality]**
- **review_status:** fresh-derivation-round-1
- **Authority:** `[macOS advisory / pure-math derivation]` — `$0`, no launch, run dirs read-only.
  Pointer UNMOVED contest-CPU **0.19109982**; this is a MEANS (a derivation + a restart recommendation).
- **Mandate:** memory `elementwise-audits-launder-structural-cargocult-pr95-skeleton` (operator fury, 3rd
  recurrence). Element-wise cargo-cult audits laundered PR95's discrete-stage BLUEPRINT (CE→tau_softplus→Muon,
  CE at PR95's 10% position, PR95 stage NAMES) through run-1 (crucible_v6). No element-level question can
  answer "should this SHAPE exist at all" — that needs a STRUCTURE-level derivation from the task's OWN math,
  blinded from the incumbent. This memo is that derivation.

## STORES CONSULTED
- `corpus_query` over research/equations/memory/dag/council/docs (retrieval-first).
- **Read in full (the witness's OWN mathematics — first-principles, blinding-safe):**
  `deepmath_lens_phasefield_gmt_levelset_20260704` (Ch.4 Γ-convergence / Modica-Mortola / level-set),
  `deepmath_lens_dynamics_transition_easing_20260704` (Ch.6 optimization dynamics / continuation / Muon-Stiefel),
  `docs/triality_dag_dsl_equations_deepmath.md`, the mandate memory.
- **Measured-physics constants (harvested from corpus SNIPPETS only, not full incumbent docs):**
  τ\*(knee)≈**0.31** (Kneedle elbow, band ≈[0.19,0.53]); flip-mass CDF heavy-tailed, m_q90 **Lane→Road 0.75 /
  Road→Lane 1.94** (fattest tail); through-R per-stage d_seg **CE 0.01045→0.005443, τ_softplus(0.3)→0.004563**
  (−0.00088); info/octave **CV≈0.39** (geometric = near-constant Fisher-Rao velocity); κ≈19 boundary Hessian;
  annulus ~97% of d_seg in ~4.7% area; MCF d_seg cost ~95.7% Lane, Lane retention 1.00→0.13.
- **DEFERRED to Phase 2 (opening them in Phase 1 would UN-BLIND):** `position_S2_schedule_curriculum`,
  `DRAFT_derived_optimal_next_run_for_council`, `DRAFT_OPTIMAL_STACK_v6`, `t5_crucible/ORCHESTRATION_LEDGER`,
  the v6.4 launch config, the run-1 config. I read ONLY their one-line corpus snippets for MEASURED constants
  (τ\*, CV, per-stage traces = physics); every STRUCTURAL claim in them is quarantined until §Phase-2.

## BLINDING PROTOCOL (the proof is the commit order)
Phase 1 below is committed as its OWN commit BEFORE any incumbent schedule doc is opened or Phase 2 is written.
The commit hash timestamps the blinding. Only after that commit do I read the incumbent and write Phase 2/3.

---

# PHASE 1 — THE DERIVATION (blinded from the incumbent shape)

## §1. The core result: ONE continuous τ-homotopy, not discrete stages

### 1.1 There is exactly one control parameter
The witness renders `soft(x)=softmax(φ(x)/τ)`, scored against the frozen SegNet argmax. By the τ=ε=ħ
two-scale dequantization (Ch.4 §3, proven-cited), the SAME scalar τ is (i) the Maslov/tropical Planck
constant of the pointwise `softmax→argmax` limit, (ii) the Modica–Mortola diffuse-interface width ε of the
spatial phase field, and (iii) the mirror-descent temperature of the training dynamics (softmax = ∇(neg-entropy)\*).
Annealing τ:1→0 **is** graduated-non-convexity homotopy continuation; Γ-convergence (Modica/Baldo +
Sandier-Serfaty flow-convergence) is the theorem that this continuation lands on the hard argmax-partition
minimizer. **The witness's mathematics natively describes a CONTINUOUS FLOW in one parameter τ — not a
sequence of discrete objectives.**

### 1.2 The loss form AS A FUNCTION of τ — "CE stage" IS the τ≈1 arc (the decisive derivation)
Question (a), verbatim: is the "CE stage" anything other than the high-temperature limit of the anneal? Derive
the temperature-τ cross-entropy family directly:

```
L_τ(φ, y) = τ · [ logsumexp_k(φ_k/τ) − φ_y/τ ] = τ·logsumexp(φ/τ) − φ_y
```

- **τ = 1:**  `L_1 = logsumexp(φ) − φ_y` = **standard multi-class cross-entropy** (the "CE stage" objective).
- **τ → 0:**  `τ·logsumexp(φ/τ) → max_k φ_k`, so `L_τ → max_k φ_k − φ_y = ReLU(−m)`, where
  `m = φ_y − max_{k≠y}φ_k` is the top1−top2 margin. This is the **max-margin / perceptron / hinge** loss —
  i.e. exactly the "margin-softplus / tau_softplus" objective. (2-class reduction, identical behaviour:
  `L_τ = τ·softplus(−m/τ)`; τ=1 → softplus/CE, τ→0 → hinge.)

**Therefore CE and margin-softplus are NOT two loss forms. They are the τ=1 and τ→0 endpoints of ONE
continuous family `L_τ = τ·CE(φ/τ)`.** The very same τ that softens the *render* softens the *loss*. The
"CE→tau_softplus stage switch" is an IMPLEMENTATION artifact — two temperatures of one functional coded as two
named functions with a hard epoch dispatch. Γ-convergence (De Giorgi) makes the family's minimizers continuous
in τ automatically. **DERIVED VERDICT (a): the loss is ONE object `L_τ`, τ(t) continuous; the "CE stage" is
literally its τ≈1 arc. No loss-form switch exists in the witness's math.** The stage NAMES are the framing that
smuggled the blueprint.

### 1.3 Even the anneal SHAPE is derived, not chosen
Surface tension and the count of resolvable interface configurations scale with `log ε = log τ`
(scale-space / renormalization-group). Sandier–Serfaty "slow relative to the landscape" ⇒ equal information
per OCTAVE of τ ⇒ **geometric (log-linear) decay** (measured info/octave CV≈0.39 = near-constant Fisher-Rao
velocity confirms it). Cosine is scale-space-WRONG (rushes mid-τ where the interface crosses Nyquist, lingers
at endpoints). Shape falls out of the same one-flow math.

## §2. The finite-τ turnpike — start, cruise, stop (Γ-licensing of finite τ)

Γ-theory licenses a FINITE continuation, NOT τ→0:
- **Floor τ\*** = the resolution/persistence scale. With margin-eikonal `|∇m|≈1` the diffuse half-width is
  `τ/2` px; below the pixel/annulus/R-blur scale (~1px) further sharpening is sub-grid aliasing the
  **hard-argmax verdict cannot even read** (Ch.4 Deriv-2). Measured knee **τ\*≈0.31** sits exactly at this
  floor. The flow is `τ: τ_start → τ*≈0.31`, never →0. ("drive τ→0" is a named FALSE FRIEND, Ch.4 §7.)
- **Turnpike (CT import).** The optimal continuation has a budget-INDEPENDENT entry arc (high-τ soft warmup),
  a turnpike cruise (the geometric octave-march), and a budget-independent exit arc settling at τ\*. **Extra
  budget extends the TAIL at τ\*, it does NOT stretch the transients** — over-descent past τ\* is aliasing,
  under-length starves. Derived answer to "how long": march to meat-exhaustion, then TAIL_k at τ\*.

## §3. The events — what partitions a CONTINUOUS flow (geometric, not temporal)

A continuous flow still has EVENTS, but they are Morse/geometric markers gating the τ-advance RATE and firing
source terms — NOT loss-form switches at fixed epochs.

**(b) Self-triggered τ-advance (critical slowing).** As τ→small the leading relaxation eigenvalue → 0
(critical slowing, Ch.6 I.3): the relaxation time diverges, so the continuation must advance τ ONLY when the
current scale's transient has relaxed. ⇒ event: advance dτ when per-scale d_seg progress PLATEAUS (rel-ε below
floor over a window; measured ν decay). Self-triggered control: `dτ/dt = f(local relaxation)`, not a clock. A
codim-≥2 crossing (two homotopy parameters at once) is ill-conditioned — cross ONE at a time (numerical
continuation principle). So even at a genuine event, superposition is forbidden by the math.

**(c) The optimizer switch is a CONDITIONING event, derived from curvature — not an epoch.** Muon =
Newton–Schulz orthogonalization = Stiefel-manifold flow = a κ-buster (whitens the update spectrum to σ=1). At
high τ the landscape is soft/well-conditioned — mirror-descent/AdamW suffices, orthogonalization buys ≈0. As
τ→τ\* the interface localizes, the boundary Hessian stiffens (measured κ≈19), and the residual becomes
dominated by the LOWEST-persistence (finest-scale, Lane-dash) index directions whose Euclidean escape is
EXPONENTIAL (saddle-to-saddle, Abbe leap). Muon's Stiefel curvature converts exp→poly escape
(Mousavi-Hosseini; Spectral-Flattening-Muon ≈ Stiefel, 2605.13079 — the mechanistic license, not analogy).
⇒ **switch to Stiefel conditioning when a measured anisotropy/conditioning statistic crosses threshold** —
operationally, when d_seg progress enters the power-law saddle-escape regime at small τ (residual =
finest-scale-dominated). **Muon is a CONDITIONER, not a SOURCE: it cannot nucleate a zero-mass class** (the
orthogonalized gradient of an absent phase is still zero) — so it belongs AFTER nucleation, at the finishing
end, OUTSIDE the τ-continuum (it changes the metric, not τ). Its trigger is thus doubly non-temporal: it
requires (i) the finest-scale/ill-conditioned regime AND (ii) nucleation already complete.

**(d)+(e) Per-class staggered nucleation = the LADDER, embedded IN the flow.** The 5 classes are 5 objects
with measured, ORDERED physics:

| class | persistence / scale | margin, IoU | resolves when |
|---|---|---|---|
| MyCar(hood), Undrivable(sky) | high, static | margin≈5.6, IoU≈0.99 | FIRST octaves (coarse τ) |
| Road | high, large-area | IoU 0.955 | early |
| Movable(cars) | mid, sparse | IoU 0.90 | mid — needs nucleation |
| Lane markings | LOWEST (finest scale) | IoU 0.26, m_q90 heavy-tail 0.75 | LAST + LONGEST; sub-margin at coarse τ ⇒ UNBORN without a source |

Modica–Mortola fact: a minority phase absent at coarse ε **cannot spontaneously nucleate** (the diffuse
interface can't represent a sub-ε feature). Lane dashes are "unborn" at coarse τ — no amount of descent on the
existing partition creates them. The mathematically-correct cure is **controlled continuation with a per-class
source term**: a per-class area/volume constraint `λ_k·(area_k − target_k)²` (Esedoğlu–Otto auction-MBO analog;
the theorem-identified cure for MCF minority-erasure) and/or a seed, which NUCLEATES class k exactly when the
continuation reaches k's persistence-scale, after which the flow carries it. **The sequence of nucleation
events ordered by persistence — {MyCar,Undriv,Road}(coarse) → Movable → Lane(finest) — IS the LADDER.** Each
rung is a Morse BIRTH in the persistence diagram, gated on τ reaching that class's scale — NOT a bolt-on stage.
Simultaneously, the multi-phase Baldo Γ-limit prices surface tension PER-PAIR `σ_ij`: high on {Road,Undriv,MyCar}²
(smoothing there is free, saves perimeter), ~0 on Lane/Movable pairs (uniform perimeter-flow is MCF that ERASES
the fragile tail — measured 95.7% of MCF d_seg cost is Lane, retention 1.00→0.13). So per-class `λ_k`
(nucleation/area) AND per-pair `σ_ij` (erasure-aware tension) are the two per-class knobs, both embedded in the
one flow, both ordered by the same measured persistence.

## §4. THE DERIVED STRUCTURE (blinded — write this down before looking)

The witness-native schedule is ONE continuous variational level-set flow with geometric event markers:

1. **ONE loss** `L_τ = τ·CE(φ/τ)`, τ the single dequantization parameter. CE = its τ≈1 arc; margin/tau_softplus
   = its τ→τ\* arc. **NO discrete loss-form stages, NO stage names.**
2. **τ(t) geometric, FINITE:** `τ_start≈1 → τ*≈0.31` (resolution floor), then TAIL at τ\* (turnpike; extra
   budget → tail, never longer transients). NOT →0.
3. **τ-advance SELF-TRIGGERED** on per-scale relaxation (advance when d_seg plateaus), not epoch-clocked; one
   continuation parameter crossed at a time (no codim-≥2 superposition).
4. **Per-class LADDER:** controlled-continuation nucleation events ordered by persistence
   {static → road → movable → lane}, each a Morse birth gated on τ reaching that class's scale, via per-class
   area/volume source `λ_k` + per-pair erasure-aware tension `σ_ij` (Lane/Movable σ≈0).
5. **ONE metric-conditioning event:** switch to Muon/Stiefel when the residual is finest-scale /
   ill-conditioned-dominated (saddle-escape regime at small τ) AND nucleation is complete — a conditioning
   finisher OUTSIDE the τ-continuum that cannot itself nucleate.

**There are NO discrete loss-form stages, NO fixed proportional boundaries, NO PR95 stage names in this
derived structure.** The only "stages" are geometric events (nucleation births; the conditioning switch) read
off the flow's own state. — END PHASE 1 (committed before Phase 2).
