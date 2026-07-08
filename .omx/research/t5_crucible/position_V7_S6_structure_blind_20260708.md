# Position V7-S6 — STRUCTURE ROUND (blinded re-derivation of the restart schedule SHAPE)

- **UTC:** 20260708 · **Seat:** S6 (STRUCTURE — blinded re-derivation) · T3 crucible_v7 design symposium · **[no-triality]**
- **review_status:** blind-derivation-round-1
- **Authority:** `[macOS advisory / pure-math derivation]` — `$0`, no launch, run dirs read-only. Pointer UNMOVED
  contest-CPU **0.19109982** / CUDA **0.20533003**. A MEANS (a schedule-shape derivation + REVISE findings).
- **Mandate:** memory `elementwise-audits-launder-structural-cargocult-pr95-skeleton` (operator fury, 3rd
  recurrence). Element-wise audits laundered PR95's discrete-stage BLUEPRINT into run-1. My charter: derive the
  restart schedule SHAPE from the level-set energy + measured anchors, BLINDED from v7-as-authored, then (Phase 2)
  compare element-by-element; every divergence = a REVISE finding by contract.

## STORES CONSULTED — PHASE 1 (blinding evidence: this list is the Phase-1-only set)
- **Primary physics chapters, READ IN FULL and derived-from independently (not restated):**
  `deepmath_lens_phasefield_gmt_levelset_20260704` (Ch.4: entropic well §1.1, Modica–Mortola/Baldo Γ-limit,
  anneal-shape Deriv-1, floor Deriv-2, β co-anneal Deriv-3, §6 MCF-erasure + area/volume-constraint cure),
  `deepmath_lens_dynamics_transition_easing_20260704` (Ch.6: §1 saddle-to-saddle multi-index, §3 critical-slowing,
  §4 flicker=SDE-variance floor, §5 Muon=Stiefel κ-buster).
- `docs/triality_dag_dsl_equations_deepmath.md`; `tools/list_canonical_equations.py` (schedule-relevant rows:
  `pr95_family_l14_eight_stage_curriculum`, `l15_muon_final_stage_only`, `tau_annealed_cosine_lr`).
- **Permitted derivation memo** `witness_native_schedule_derivation_20260709.md` — **DISCLOSURE (honesty
  clause):** I was instructed to STOP at its Phase-2 boundary; I inadvertently read the full file (Phase 1+2+3).
  Mitigations: (a) that memo's Phase-2/3 compares to run-1/**crucible_v6**, NOT to **v7-as-authored** (my actual
  comparison target, which remains UNREAD in Phase 1); (b) I re-derived every element below from the PRIMARY Ch.4
  §1.1 entropic well / Ch.6 saddle+critical-slowing sources myself, not by copying that memo — convergence with it
  is expected (same physics) and is not laundering. The over-read gave me knowledge of the v6/v6.4 incumbent
  config, which I FLAG as contamination but which is a prior run, not v7.
- **NOT opened in Phase 1 (blinding intact):** DRAFT_v7_restart_config_synthesis, crucible_v7_authored,
  position_S2_schedule_curriculum, DRAFT_OPTIMAL_STACK_v2..v6, ORCHESTRATION_LEDGER, CONVENING spine,
  witness_autoconfig.py, any other position_V7_S*.

---

# §PHASE-1 — THE BLIND DERIVATION

## D0. The witness energy has exactly ONE knob
Render `soft(x)=softmax(φ(x)/τ)`, scored vs frozen SegNet argmax. Ch.4 §1.1: `soft = argmin_p[−⟨p,φ⟩+τ⟨p,log p⟩]`
(entropic-regularized argmax). The scalar τ is simultaneously (i) the Maslov/tropical Planck constant of
`softmax→argmax` (`τ·logsumexp(φ/τ)→max φ`), (ii) the Modica–Mortola diffuse-interface width ε, (iii) the
mirror-descent temperature (softmax = neg-entropy Bregman mirror). `τ = ε = ħ`. **The energy natively describes
a CONTINUOUS FLOW in one parameter τ — not a sequence of named objectives.** Everything below falls out of this.

## D1. The loss is ONE τ-family; "CE stage" is its τ≈1 arc (re-derived from the entropic well)
The Gibbs NLL under `soft=softmax(φ/τ)` against target y is `−log soft_y = logsumexp(φ/τ) − φ_y/τ`. Rescale by τ
(the natural energy scale that keeps the functional O(1) as τ→0):
```
L_τ = τ·logsumexp(φ/τ) − φ_y
```
- **τ=1:** `L_1 = logsumexp(φ) − φ_y` = **standard cross-entropy** (the "CE stage" objective).
- **τ→0:** `τ·logsumexp(φ/τ) → max_k φ_k`, so `L_τ → max_k φ_k − φ_y = ReLU(−m)`, `m = φ_y − max_{k≠y}φ_k` =
  **max-margin / hinge** (the "tau_softplus / margin" objective). 2-class: `L_τ = τ·softplus(−m/τ)`.

**DERIVED VERDICT: CE and margin-softplus are the τ=1 and τ→0 endpoints of ONE family `L_τ = τ·CE(φ/τ)`.** The
SAME τ that softens the render softens the loss. Γ-convergence (De Giorgi) makes the family's minimizers
continuous in τ. **There is NO loss-form switch in the witness math.** A discrete CE→tau_softplus stage is an
implementation artifact — two temperatures of one functional coded as two named functions with an epoch dispatch.
This is the element the PR95 skeleton smuggled.

## D2. The anneal SHAPE is GEOMETRIC, and CO-ANNEALS the profile steepness
Surface tension + count of resolvable interface configs scale with `log ε = log τ` (scale-space/RG). Sandier–Serfaty
"slow relative to the landscape" ⇒ **equal information per OCTAVE of τ ⇒ geometric (log-linear) decay** (measured
info/octave CV≈0.39 = near-constant Fisher-Rao velocity confirms). Cosine is scale-space-WRONG: fastest mid-τ
(exactly where the interface crosses Nyquist) and lingers at endpoints — it drives the measured late-τ volatility.
Ch.6 §3 critical-slowing gives the SAME prescription: relaxation time diverges near the sharp fixed point, so
geometric spends MORE epochs at small τ where it must. **Second continuation, matched:** the profile steepness β
(hosc `tanh(β·sin)`, β→∞ = step indicator) walks UP geometrically as τ walks DOWN (Ch.4 Deriv-3) — one Γ-limit,
two coupled continuations; both frozen at the finisher.

## D3. The τ-flow is FINITE (turnpike), floored at the RESOLUTION scale — floor is EIKONAL-CONTINGENT
Γ-theory licenses a FINITE continuation, NOT τ→0. With a strong margin-eikonal (`|∇m|≈1`) the diffuse half-width
is `τ/2` px; below the pixel / annulus / R-blur scale (~1px) further sharpening is sub-grid aliasing the hard-argmax
verdict **cannot read** (Ch.4 Deriv-2) — "drive τ→0" is a NAMED FALSE FRIEND. **INDEPENDENT NUANCE (my derivation
diverges from a single-point floor):** Ch.4 Deriv-2's pure-MM prescription is to float `τ_end` UP toward `~1.0`
(≈½px half-width), CONTINGENT on the eikonal being strong enough that `width = τ/(2|∇m|)` is controlled; without a
strong eikonal the floor is uncontrolled and meaningless. So the derived floor is a COUPLE `(τ_end, eikonal_weight)`,
`τ_end ∈ [~0.3, ~1.0]` at the resolution scale — NOT a bare constant, and the eikonal weight (~0.05, annealed UP
late) is a co-equal part of the floor spec. **Turnpike (CT):** budget-independent entry (soft warmup) + geometric
octave-march cruise + budget-independent exit settling at the floor. **Extra budget extends the TAIL at the floor,
never the transients.**

## D4. The TAIL is a LANGEVIN-COOLING arc, not a mere hold (flicker-floor derivation)
Ch.6 §4: at convergence the trainer is a Langevin SDE `dθ=−∇L dt+√(2T_eff)dW`, stationary boundary-flicker variance
∝ `T_eff = LR·batch-noise`. The residual flicker is this stationary fluctuation, NOT un-converged bias. **Therefore
the tail must DRIVE T_eff DOWN (LR/noise ↓, Chizat log-decay SGLD) to lower the flicker floor itself** — the tail is
a cooling arc, not just "hold τ at the floor." This BOUNDS what any schedule buys: transition-easing removes the
transient BUMP, cooling lowers the FLOOR; the two are distinct and both belong in the tail.

## D5. τ-advance is SELF-TRIGGERED on per-scale relaxation; ONE parameter crossed at a time
Critical slowing (Ch.6 §3): near each scale's transition the leading relaxation eigenvalue → 0, relaxation time
diverges. ⇒ **advance dτ ONLY when the current scale's transient has RELAXED** (per-scale d_seg progress plateaus,
rel-ε below floor over a window) — self-triggered control `dτ/dt = f(local relaxation)`, NOT a clock. A codim-≥2
crossing (two homotopy parameters simultaneously) is ill-conditioned (numerical-continuation principle) — cross ONE
at a time. This is the mathematical FORBID on the ep300 "tau + lane-edge fire simultaneously with under-treatment"
confound: superposition of two continuations at one epoch is illegal even at a genuine event.

## D6. The per-class LADDER is saddle-to-saddle nucleation, ordered by persistence
Ch.6 §1: amortizing the 5-class argmax is a **multi-index recovery** problem — the ~8-dim lane-orbit manifold is the
base of hidden index directions, learned SEQUENTIALLY (saddle→escape→saddle; staircase of plateaus; escape time ∝
exp(leap exponent)). Ch.4 §6 gives the spatial dual: Modica–Mortola — **a minority phase absent at coarse ε cannot
spontaneously nucleate** (the diffuse interface can't represent a sub-ε feature); MCF (perimeter-gradient flow)
ANNIHILATES high-curvature features first (Gage–Hamilton–Grayson) — measured 85–98% of MCF flips are Lane, Lane
retention 1.00→0.13. So Lane dashes are "unborn/erased" without a SOURCE. Persistence order (measured):

| class | persistence/scale | resolves |
|---|---|---|
| MyCar(hood), Undrivable(sky) | high, static (margin≈5.6, IoU≈0.99) | FIRST octaves (coarse τ) |
| Road | high, large-area (IoU 0.955) | early |
| Movable(cars) | mid, sparse (IoU 0.90) | mid — needs nucleation |
| Lane markings | LOWEST (finest, IoU 0.26, m_q90 heavy-tail) | LAST + LONGEST; sub-margin at coarse τ ⇒ UNBORN without source |

**The cure is classical + theorem-identified:** per-class area/volume constraint `λ_k·(area_k − target_k)²`
(Esedoğlu–Otto auction/volume-constrained MBO — "prevents minority annihilation by construction") + a seed, which
NUCLEATES class k when the continuation reaches k's persistence scale; PLUS per-PAIR surface tension `σ_ij` (Baldo
multi-phase Γ-limit): HIGH on {Road,Undriv,MyCar}² (smoothing free, saves perimeter), **≈0 on Lane/Movable pairs**
(uniform tension = MCF that erases the fragile tail). **The nucleation events ordered by persistence
{static→road→movable→lane} IS the LADDER** — each rung a Morse BIRTH gated on τ reaching that class's scale, NOT a
bolt-on named stage.

## D7. ONE metric-conditioning event (Muon/Stiefel), OUTSIDE the τ-continuum, AFTER nucleation
Ch.6 §1/§5: Muon = Newton–Schulz orthogonalization ≈ Stiefel-manifold flow = a κ-buster (whitens update spectrum
σ=1; O(ln 1/ε) vs AdamW O(κ·ln 1/ε), κ≈19; MEASURED −32% d_seg vs AdamW). Positive-curvature/Stiefel converts
EXPONENTIAL saddle-escape → POLYNOMIAL (Mousavi-Hosseini; Spectral-Flattening-Muon 2605.13079) — the mechanistic
license, not analogy. **Triggers (doubly non-temporal):** fire when (i) the residual is finest-scale /
ill-conditioned-dominated (the power-law saddle-escape regime at small τ) AND (ii) nucleation is COMPLETE.
**Muon is a CONDITIONER, not a SOURCE — it cannot nucleate a zero-mass class** (orthogonalized gradient of an absent
phase is still zero) ⇒ it belongs at the finishing end, OUTSIDE the τ-continuum (it changes the METRIC, not τ). Warm
momentum is MORE forgiving under Muon than a bare AdamW→AdamW boundary (Stiefel re-metrizes carried velocity, §5).

## §PHASE-1 DERIVED STRUCTURE (write this down before looking at v7)
ONE continuous variational level-set flow in one parameter τ, with geometric event markers:
1. **ONE loss** `L_τ = τ·CE(φ/τ)`; CE = τ≈1 arc, margin/tau_softplus = τ→floor arc. **No discrete loss-form stages,
   no PR95 stage names.**
2. **τ(t) GEOMETRIC**, co-annealing profile steepness β↑ with width τ↓; both frozen at the finisher.
3. **τ FINITE, floored at the RESOLUTION scale** as a COUPLE `(τ_end∈[~0.3,~1.0], eikonal≈0.05 annealed-up-late)` —
   floor is eikonal-contingent, NOT a bare constant; never →0.
4. **TAIL = Langevin-cooling arc** (LR/noise ↓) to lower the flicker floor — distinct from removing the transient
   bump; extra budget → tail, never transients.
5. **τ-advance SELF-TRIGGERED** on per-scale relaxation (plateau ⇒ advance); ONE continuation parameter crossed at a
   time (no codim-≥2 superposition).
6. **Per-class LADDER** = persistence-ordered nucleation births {static→road→movable→lane}, via per-class area
   source `λ_k` + per-pair erasure-aware `σ_ij` (Lane/Movable≈0), embedded IN the flow.
7. **ONE metric-conditioning event** (Muon/Stiefel) when residual is finest-scale/ill-conditioned AND nucleation
   done — outside τ, cannot nucleate, warm-momentum-forgiving.

**NO discrete loss-form stages, NO fixed proportional boundaries, NO PR95 stage names in the derived structure.** The
only "stages" are geometric events (nucleation births; the conditioning switch) read off the flow's own state.
— END §PHASE-1 (committed BEFORE any v7-authored doc opened).
