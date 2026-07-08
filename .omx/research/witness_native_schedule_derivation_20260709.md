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
off the flow's own state. — END PHASE 1 (committed at `37a974742` before Phase 2 opened any incumbent doc).

---

# PHASE 2 — COMPARISON vs the incumbent (run-1 = crucible_v6, v6.4 schedule)

**Incumbent read (AFTER the Phase-1 commit):** `.../levelset_n600_crucible_v6_run1_.../launch.sh` +
`experiments/train_levelset_witness_realized_through_R_mlx.py` argparse + `_seg_form_for_epoch`. Incumbent
schedule: `--curriculum` CE(ep0–299) → `--tau-softplus-start-epoch 300 --tau-softplus-tau 0.3`(fixed) →
`--muon-start-epoch 726`; l7 off (@3000). Render anneal SEPARATE: `--softmax-temp-start 1.0 --softmax-temp-end
0.31 --tau-anneal-shape cosine_hold --tau-hold-frac 0.2 --anneal-epochs 3000`. Event modifiers ON:
`--curriculum-event-triggered --curriculum-min-stage-epochs 250 --curriculum-nucleus-guard
--curriculum-reanchor-levers`. LADDER: `--seed-islands --seed-island-eased --island-amplify(=amplify-weight)
--persistence-loss(=persistence-loss-weight) --witness-alone-island-loss --lane-render-band@350
--length-sigma-matrix fitted-20260707 --structured-init(-include-lane)`. Muon: `--muon-warm-start-momentum
--muon-lr-final-frac 0.1`. Transition: `--stage-transition-rewarmup-epochs 8 --...-reset-moments`.

**Code-level confirmation of the skeleton:** `_seg_form_for_epoch` (L1788–1794) is a HARD dispatch
(`if ep < tau_softplus_start_epoch: return 'ce' else 'tau_softplus'`); the comment L1650 reads verbatim
*"Curriculum seg_form by epoch (PR95 d_seg sequence): ce -> tau_softplus -> l7_softplus."*; `_STAGE_TAGS`
carries the PR95 names. The event-trigger `_fire` (L1862) *"mirrors the hardcoded `_seg_form_for_epoch`
schedule EXACTLY"* — it only RELOCATES the hard switch epoch; it does NOT dissolve the discreteness.

**Per-element map (derived structure → incumbent): approximates-with-error, or structurally diverges.**

| # | Derived (Phase 1) | Incumbent | Verdict |
|---|---|---|---|
| **1 loss** | ONE `L_τ=τ·CE(φ/τ)`, τ continuous; CE=τ≈1 arc | TWO hard-switched named forms `ce`→`tau_softplus`(τ fixed 0.3) at ep300; loss-τ QUANTIZED to {1, 0.3}, DECOUPLED from render-τ | **STRUCTURAL DIVERGENCE** — the one residual PR95 skeleton |
| **2 shape/floor/tail** | geometric τ:1→τ\*≈0.31, then TAIL | render `cosine_hold` 1.0→**0.31**, hold-frac 0.2 (=tail) | **floor 0.31 + tail = MATCH (vindicated); shape cosine≠geometric = minor divergence** |
| **3 self-trigger** | advance τ when per-scale relaxes | event-triggered exits + min-stage 250 gate the LOSS-FORM boundary; render-τ itself is clock-driven (cosine over fixed 3000) | **PARTIAL — self-trigger on the discrete boundary, not on the τ-rate** |
| **4 LADDER (per-class λ_k + per-pair σ_ij)** | persistence-ordered nucleation births + σ_ij(Lane/Mov≈0) | seed-islands[1,3] + amplify + persistence[1,3] + **`length-sigma-matrix fitted`(=σ_ij!)** + lane-band@350; structured-init shows lane_px=0 (unborn ⇒ needs source, as derived) | **STRONG MATCH (vindicated + built); ordering coarse (all[1,3] together, band@fixed-350) not per-rung persistence-gated** |
| **5 Muon = conditioning finisher** | switch when finest-scale/ill-conditioned AND nucleation done; outside τ; can't-nucleate | `--muon-start-epoch 726`(fixed) + warm-start + lr-final-frac 0.1; after nucleation; event-scaffold present | **direction+gentleness+placement = MATCH (vindicated); fixed-726 = un-derived knee-transfer residual** |

**HONESTY BOTH WAYS.** The derivation does **NOT** reproduce discrete loss-form stages — it derives them
*away* into one continuous `L_τ`. But it **DOES vindicate** most of the surrounding machinery as
derivable-not-inherited: the render sharpness anneal (continuous, floored at the measured knee τ\*=0.31, with a
turnpike tail), the LADDER (`seed-islands` + the fitted per-pair `σ_ij` = exactly Baldo's multi-phase surface
tension), and Muon as a gentle metric finisher outside the τ-continuum that cannot nucleate. So **four of five
elements are substantially present and now first-principles-justified; the single clean structural divergence
is element 1 — the CE→tau_softplus hard loss-form switch** (the `_seg_form_for_epoch` "PR95 d_seg sequence"),
with two lesser residuals (cosine≠geometric shape; fixed-epoch 350/726 gates instead of event/conditioning
triggers). The incumbent's whole L1–L6 CE→tau transition-easing apparatus (rewarmup, reset-moments, loss-blend)
exists ONLY to soften a discontinuity element 1 dissolves at the source.

**RUN-1 LIVE DATA — there is essentially none.** `run.log` ends at the `before_v0_verdict` mem-probe;
`costate_shadow.jsonl` has only the v0 advisory tick (`as_of_epoch: null`). Run-1 was operator-stopped at the
first verdict and produced **NO d_seg trajectory**. Consequence for Phase 3: the "restart vs run-1 partial
trajectory" cannot use run-1 data — the restart is the first real measurement of EITHER structure. The only
prior baseline is the earlier discrete-stage through-R trace (`n205_joint_nexus...`): CE 0.01045→0.005443,
τ_softplus(0.3)→0.004563, with the FEED-ft ep300 bump 0.0056→0.020 (3.4×) on the amort decoder.

---

# PHASE 3 — RESTART RECOMMENDATION

**VERDICT: HYBRID — adopt-derived on element 1 (the loss form), keep-incumbent-now-justified on 2/4/5, cheap
config-fix on the element-2 shape.** The operator-directed restart (stop run-1; relaunch with TAIL_k + LADDER)
is derivation-CONSISTENT: TAIL_k = the turnpike tail (§2, vindicated), LADDER = the nucleation ledger (§3,
vindicated). My net addition is one BUILD (unify the loss) + one flag flip (geometric), which REMOVES the last
PR95 skeleton rather than easing it.

**The restart schedule (concrete):**

1. **BUILD — unify the loss to `L_τ = τ·CE(φ/τ)` with τ = the render softmax-temp.** Replace the
   `_seg_form_for_epoch` hard dispatch (when a new flag is set) with ONE temperature-parameterized loss
   `L_τ = τ·logsumexp(φ/τ) − φ_y` (τ = `_softmax_temp_for_epoch(ep)`). CE is then its τ≈1 arc and the margin
   objective its τ→0.31 arc — no ep300 switch, no `ce`/`tau_softplus` forms. **New flag `--seg-form-unify-tau`
   (VERIFIED ABSENT from argparse ⇒ BUILD).** Reuses existing `--softmax-temp-start/end --tau-anneal-shape`.
   **LOC ≈ 60–100** (the loss kernel + dispatch bypass + default-off byte-identical guard + a unit test that
   `L_τ→CE` as τ→1 and `→ReLU(−m)` as τ→0). **Risk LOW–MED:** removes the loss-form discontinuity entirely
   (the ep300 3.4× bump cannot occur); the only open question is whether multi-class `τ·CE(φ/τ)` at small τ
   weights already-correct pixels differently than the current `tau_softplus` — settled by the A/B.
   *(Partial fallback already in-code: `--tau-anneal-start/end` (default None) can anneal the tau_softplus τ
   WITHIN the tau stage — run-1 did not set it — but it does NOT cover the CE arc nor couple to the render-τ,
   so it is a weaker approximation, not the unification.)*

2. **CONFIG — `--tau-anneal-shape cosine_hold → geometric`** (EXISTS; keep `--tau-hold-frac 0.2` tail).
   Derived (Ch.4 Deriv-1; measured info/octave CV≈0.39). **LOC 0. Risk ~0.**

3. **KEEP (now derivation-justified, do NOT churn):** `--softmax-temp-end 0.31` (=τ\*), `--tau-hold-frac 0.2`
   (tail), the full LADDER (`--seed-islands --seed-island-eased --amplify-weight --persistence-loss-weight
   --witness-alone-island-loss --lane-render-band --length-sigma-matrix fitted-20260707 --structured-init`),
   Muon finisher (`--muon-start-epoch 726 --muon-warm-start-momentum --muon-lr-final-frac 0.1`),
   event-triggering (`--curriculum-event-triggered --curriculum-min-stage-epochs 250 --curriculum-nucleus-guard`).

4. **SECONDARY (optional, lower priority):** promote the residual fixed-epoch gates (lane-band@350; Muon@726)
   from clock to conditioning/persistence-event triggers. Per the event-trigger scaffolding these are already
   CAPS with event exits, so the marginal value is small — note it, don't block the restart on it. **LOC ≈ 40**
   if pursued (reuse the quadratic-basin ExitEvent / persistence-birth predicates).

**Pre-registered comparison plan (this IS the A/B on the schedule-structure question):**
- Arm U (restart, DERIVED): unified `L_τ` (build 1) + geometric (2) + KEEP (3).
- Arm D (control, INCUMBENT): the run-1 discrete-stage config unchanged (or the prior through-R discrete trace
  as the historical baseline, since run-1 produced no trajectory).
- **Pre-registered success for Arm U:** reach the CE-arc floor (≈0.00475–0.00544) with NO loss-form transition
  bump at the τ≈0.3 crossing (vs the incumbent 0.0056→0.020, 3.4×), and descend monotonically to the τ\*-floor
  (≈0.00456) or below. **Falsification:** if Arm U is WORSE than Arm D at the τ\*-floor, the CE/tau
  discretization was load-bearing (contra the math) — revert to discrete, keep only the geometric shape.
- Measure both n600, byte-closed through R (the AXIS-9 discipline); the schedule question is only SETTLED by
  the measured d_seg trajectory, not this derivation. Pointer 0.19110 UNMOVED until a byte-closed exact row.

---

## FINAL SUMMARY (≤16 lines)

**DISCRETE-OR-CONTINUOUS VERDICT: CONTINUOUS.** The witness's own math gives ONE variational level-set flow in
one parameter τ; discrete loss-form stages are NOT witness-native — they are dissolved, not reproduced.

**The derived structure (5 lines):**
1. ONE loss `L_τ = τ·CE(φ/τ)` — CE is its τ≈1 arc, margin/tau_softplus its τ→τ\* arc; no CE→tau switch exists.
2. τ(t) GEOMETRIC & FINITE: τ_start≈1 → τ\*≈0.31 (resolution floor), then TAIL (turnpike; budget→tail, not transients).
3. τ-advance SELF-TRIGGERED on per-scale relaxation; one continuation parameter crossed at a time.
4. Per-class LADDER: persistence-ordered nucleation births (per-class area source λ_k) + per-pair σ_ij (Lane/Mov≈0), embedded in the flow.
5. ONE metric-conditioning event (Muon/Stiefel) when residual is finest-scale/ill-conditioned AND nucleation done — outside τ, can't nucleate.

**RESTART RECOMMENDATION: HYBRID.** Adopt-derived on element 1 (BUILD `--seg-form-unify-tau`, ~60–100 LOC,
LOW–MED risk) — it removes the last PR95 skeleton (the `_seg_form_for_epoch` "PR95 d_seg sequence") instead of
easing it. Flip `cosine_hold→geometric` (0 LOC). KEEP (now first-principles-justified, NOT inherited): floor
0.31, TAIL, LADDER (incl. fitted per-pair σ_ij), Muon gentle finisher, event-triggering. HONEST vindication:
4 of 5 elements were already derivable-and-present; the clean divergence is only the loss form. Run-1 produced
NO trajectory (stopped at v0) → the restart is the first real measurement; A/B = unified-`L_τ` vs discrete,
pre-registered on the absence of the ep300 3.4× loss-form bump. MEANS — pointer 0.19110 UNMOVED.
