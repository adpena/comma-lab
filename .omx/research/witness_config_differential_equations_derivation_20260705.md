---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_dissent:
  - member: Contrarian
    verbatim: "every optimum below is DERIVED, none is MEASURED — the word 'optimal' is banned until the symposium's n24/n600 arbitration crowns it. The π_eik constant is as unmeasured as the 38 that already failed; ship the FORM and the FOUR directional predictions, not the number."
council_assumption_adversary_verdict:
  - assumption: "the eikonal runaway is a GENERIC optimizer edge-of-stability (π_EoS = η·λ_pre/38)"
    classification: CARGO-CULTED
    rationale: "MEASURED-falsified by the sibling: λ_pre = 3.66e6 ⟹ π_EoS ≈ 94 ≫ 1 while lr 9.1e-5 is measured-STABLE. The instability is a SPECIFIC ill-posed PDE (the eikonal penalty gradient flow), not the generic Hessian-sharpness edge. Its stability group is π_eik (eikonal-residual + viscosity), NOT π_EoS. The DE derivation VINDICATES the 'specific PDE' reading."
  - assumption: "anisotropic normal-only damping (nᵀHn) necessarily beats isotropic viscosity on d_seg-drift"
    classification: CARGO-CULTED
    rationale: "the symbol analysis (§2) leaves the ill-posed DIRECTION genuinely open (normal for |∇m|>1 vs tangential for |∇m|<1); MEASUREMENT (isotropic ViscoReg ε=0.3 WON n24; raw StEik self-amplified) is consistent with BOTH stories. Normalized-nᵀHn at n600 is the DISCRIMINATING experiment, not a foregone ranking."
council_decisions_recorded:
  - "op-routable #1 (analysis, no launch): register π_eik and the eikonal-illposedness law FORMALIZATION_PENDING with the derivation as mechanism + the v5 re-entry as the qualitative anchor; the CONSTANT stays unmeasured"
  - "op-routable #2 (handed to #317 symposium): the DERIVED-ranked v6 cure stack + the single discriminating arm (normalized nᵀHn) + the λ_eik-schedule prediction (stop the up-anneal) — measurement crowns 'optimal'"
related_deliberation_ids: [eikonal_stabilizer_build_20260705, stepping_instability_diagnostic_20260705, litsweep_training_dynamics_control_20260705, tao_dimensional_analysis_pi_groups_for_witness_20260705]
---

# Witness config as a coupled differential-equation system — the eikonal permanent-cure DERIVED, and the general calibrate-by-DE framework

**Task #318 (DE-DERIVATION). Axis discipline: this whole memo is DERIVATION (means/apparatus).
Every "optimum" is PREDICTED / measurement-owed and tagged so; NOTHING here is a score. The
n24/n600 arbitration owned by the v6 symposium (#317) is what earns the word "optimal." If a
measurement below contradicts a derivation, MEASUREMENT WINS and the derivation becomes a corrected
law (honest negative, registered). Pointer contest-CPU 0.19110 UNMOVED.**

**The binding guard, restated as the spine of this memo:** a DE derivation PREDICTS the optimum
with a MECHANISM; only measurement earns "optimal." This is the NO-FAKE surrogate-vs-authority split
applied to mathematics: *derived ≠ measured-optimal.* Every §-header optimum carries a
`[PREDICTED — measurement-owed]` tag and names the arbitration that would confirm or correct it.

Grounded in the measured saga (proactive recall done first): `stepping_instability_diagnostic_20260705.md`
(the 5-arm mechanism matrix; eikonal runaway is step-size-gated; restored moments DAMP) ·
`eikonal_stabilizer_build_20260705.md` (n24 arbitration: **ViscoReg ε=0.3 STABLE + best d_seg**;
raw StEik **self-amplifies** 575–1431×; λ_pre=3.66e6 FALSIFIES the generic 38/η law) ·
`litsweep_training_dynamics_control_20260705.md` (StEik/ViscoReg ill-posedness lineage; EoS) ·
DAG **FEED-05v/05w/05x/05y** (v5 = the ViscoReg arm at n600: banked **best-ever witness d_seg
0.124→0.02517 @ep125** then **RE-ENTERED runaway ~ep110 while ε≈0.27**; gold ckpt preserved) ·
`tao_dimensional_analysis_pi_groups_for_witness_20260705.md` (π-group discipline: DA fixes the
FORM, never the dimensionless constant — the constant is MEASURED) · the τ=ε=ħ unification
(`deepmath_amortizing_argmax_paper_draft_20260704.md`, `council_grand_symposium_curriculum_derivation_20260705.md`).

---

## 1. The governing PDE — the eikonal penalty gradient flow (symbols + first variation)

The witness carries an eikonal regularizer on the **decision margin** `m(x) = φ_top1(x) − φ_top2(x)`
(the field the measured runaway lives on; the sibling's telemetry term). The energy is

    E_eik[m] = (λ_eik / 2) ∫_Ω (|∇m| − 1)² dx           (drive m toward a signed-distance / 1-Lipschitz margin)

The L2 gradient flow is `∂m/∂t = −δE_eik/δm`. First variation (h = test field, n = ∇m/|∇m| the
unit level-set normal, R = |∇m| − 1 the eikonal residual):

    dE_eik[m; h] = λ_eik ∫ R · (∇m·∇h)/|∇m| = λ_eik ∫ R (n·∇h)
                 = −λ_eik ∫ ∇·(R n) h        (IBP)
  ⟹ δE_eik/δm = −λ_eik ∇·(R n)
  ⟹ **∂m/∂t = λ_eik ∇·[(|∇m|−1) n]**                                          (★ the governing PDE)

Expand the divergence into its normal + curvature parts (κ = ∇·n = mean curvature of the level
sets, ∂_n = n·∇):

    ∂m/∂t = λ_eik [ ∂_n|∇m| + (|∇m|−1) κ ]                                     (★′)

This is a **degenerate, quasilinear, second-order** operator: its only second-order content is
`∂_n|∇m|` (a *directional* second derivative along the normal); it carries **NO tangential
second-order term at the SDF state**. That degeneracy is the seed of the ill-posedness.

---

## 2. Principal symbol + the ill-posed direction (why the flow blows up)

Linearize (★) about a locally-uniform-gradient state `m₀` with `|∇m₀| = a` (constant), perturbation
`m = m₀ + ψ`, normal `n₀`. Standard expansions:

    |∇m|   = a + ∂_n ψ + O(ψ²)                    (∂_n ≡ ∂_{n₀})
    n      = n₀ + (1/a) ∇_T ψ + O(ψ²)            (∇_T = tangential gradient = ∇ − n₀∂_n)
    R n    = (a−1)n₀ + [ (∂_n ψ) n₀ + ((a−1)/a) ∇_T ψ ] + O(ψ²)

Take the divergence and use, for locally-planar level sets (κ₀≈0), `∇·(∇_T ψ) = Δ_T ψ` (tangential
Laplacian) and `∇·((∂_nψ)n₀) = ∂²_nn ψ`:

    **∂ψ/∂t = λ_eik [ ∂²_nn ψ  +  c_a · Δ_T ψ ],     c_a := (a−1)/a = (|∇m|−1)/|∇m|**    (LIN)

**Principal symbol** (Fourier ψ ~ e^{ik·x}, k = k_n n₀ + k_T; growth rate σ' = λ_eik·σ):

    **σ(k) = −k_n²  −  c_a · k_T²**                                             (SYMBOL)

Read it directly — this is the entire mechanism:

| direction | symbol | sign | verdict |
|---|---|---|---|
| **normal** (k_T=0) | −k_n² | always < 0 | **forward diffusion — always stable in the normal direction** |
| **tangential** (k_n=0), `a>1` (|∇m|>1, gradient too steep) | −c_a k_T², c_a>0 | < 0 | forward diffusion — stable |
| **tangential**, `a<1` (|∇m|<1, gradient too FLAT) | +\|c_a\|k_T², c_a<0 | **> 0, → +∞ as k_T→∞** | **BACKWARD HEAT — ill-posed; high tangential frequencies blow up unboundedly** |

**The ill-posedness (the StEik/ViscoReg continuum result, re-derived from our field m):** where the
margin is FLAT (a = |∇m| < 1 — i.e. near the decision separatrix, the small-margin annulus, which
is *exactly where d_seg lives*), the eikonal flow is a **backward heat equation in the tangential
direction with unbounded growth rate**. No step size fixes a backward-heat continuum: for any η the
discrete amplification `1 + η_eff λ_eik σ(k) > 1` for the unstable modes, and worse at higher k. The
grid supplies a UV cutoff `k_max ~ π/Δx`, so on the grid the fastest-growing mode is checkerboard-
scale and amplifies each step — **lr reduction slows the onset (smaller per-step amplification) but
is NOT a structural cure.** This is precisely the measured "low_lr STABLE is a 60-step window, not
asymptotic" (Contrarian) + the litsweep contradiction-row-1 resolution.

> **Honest sign caveat (the open discriminator).** The clean symbol above puts the ill-posed mode
> TANGENTIAL for a<1. StEik reports its cure as NORMAL-direction damping. The two are reconciled by
> which linearization state and which field-normalization one uses; I did NOT reproduce StEik's exact
> stencil, so I flag the DIRECTION as genuinely open. This is not a blemish — it is the single
> highest-value prediction this memo makes (§4, arm 4): it is falsified/confirmed by whether the
> normalized normal-only term (nᵀHn) stabilizes at n600. Measurement crowns the direction.

**Where the runaway VALUE |∇m|≈2070 comes from:** the measured blowup value is the *result* of the
tangential backward-heat driving |∇m| far from 1, not the onset state — onset is at a≈1 where c_a≈0
(marginal) and the sharpening curriculum (β_hosc anneal, τ descent) tips a below 1 in the boundary
annulus, switching on the backward-heat. This DERIVES the "descend-then-runaway" signature (trough
then eikonal-alone climb) the sibling measured in every arm.

---

## 3. The viscous cure as parabolic regularization — DERIVING the two-sided ε window

ViscoReg replaces the residual by the **viscous eikonal** `(|∇m| − 1 − ε·Δm)²`, selecting the
viscosity solution as ε→0. Its leading effect on the flow symbol is a **biharmonic (4th-order)
term**: the `εΔm` inside the residual contributes, through the gradient flow, a `−ε²(k_n²+k_T²)²`
piece (isotropic, always ≤ 0, dominant at high k). The regularized tangential symbol is

    **σ_T(k_T) = |c_a| k_T² − ε² k_T⁴   (a<1)**                                (VISCO-SYMBOL)

This is the crux — read both edges off one parabola-in-k_T²:

- **σ_T > 0 (unstable) only for `k_T < √|c_a| / ε`**, with a FINITE max growth `σ_max = |c_a|²/(4ε²)`
  at `k_T² = |c_a|/(2ε²)`. Viscosity does not delete the instability — it **CAPS the growth rate at
  a finite value and moves it to a finite frequency** (this is what "well-posed" means: bounded
  growth, no k→∞ blowup). Larger ε ⟹ smaller σ_max (∝ 1/ε²) ⟹ easier to out-step.
- The DISCRETE explicit-Adam step is then stable against this capped mode iff its CFL holds:
  `η_eff · λ_eik · σ_max ≤ 2`, i.e.

    **π_eik := η · λ_eik · |c_a|² / (8 ε²)  ≤  1**              (LOWER edge → **ε ≥ |c_a|·√(η λ_eik /8)**)

- **Upper edge (measured ε=1.0 explodes):** the biharmonic term has its OWN 4th-order CFL —
  `η_eff λ_eik ε² k_max⁴ ≤ 2`, i.e.

    **π_bih := η · λ_eik · ε² · k_max⁴  ≤  2**                  (UPPER edge → **ε ≤ √(2/(η λ_eik)) / k_max²**)

**⟹ DERIVED two-sided window** `|c_a|√(η λ_eik/8) ≤ ε ≤ √(2/(η λ_eik))/k_max²`, matching the measured
`ε=0.3 STABLE, ε=1.0 EXPLODES`. **The constants (8, and the exact |c_a| definition) are
MEASUREMENT-OWED** — the sibling already showed a plausible constant (the 38 in π_EoS) can be wrong
by ~100× at this state, so I ship the FORM and the DIRECTIONAL content, not the number.

### 3.1 The v5 re-entry, DERIVED (FEED-05y)
v5 banked d_seg 0.124→0.025 then re-entered runaway **~ep110 while ε≈0.27**. π_eik ≤ 1 fails when
its numerator grows or ε shrinks. BOTH happened simultaneously:
1. **ε ANNEALS DOWN** (0.3→0.27→…→0, the `--eikonal-viscosity-anneal 1000` linear schedule) —
   drives ε toward the lower edge.
2. **|c_a(t)| GROWS** (progressive sharpening: β_hosc anneal + τ descent push more of the boundary
   annulus into the flat a<1 regime) — raises the lower edge `ε_lower = |c_a|√(ηλ_eik/8)`.

The margin `ε − ε_lower(t)` is squeezed from both sides → re-crossing. **This is a derivation of the
measured re-entry, not a restatement.** It also names the two fixes precisely (§4).

---

## 4. The v6 cure stack — DERIVED-ranked, each `[PREDICTED — measurement-owed]`

All four map onto the FEED-05y candidate list (a)/(b)/(c)/(d) and are handed to the #317 symposium
for n24/n600 arbitration. **Ranking is by derivation confidence, NOT by measured value.**

**Arm 1 — STOP the λ_eik up-anneal (FEED-05y candidate c). `[PREDICTED — highest derivation confidence, near-zero risk]`**
The eikonal weight currently anneals **UP 0.05→0.10**. Every stability group scales with λ_eik:
`π_eik ∝ λ_eik`, `π_bih ∝ λ_eik`, and σ ∝ λ_eik. Raising λ_eik **during** sharpening is doubly
destabilizing — it lowers BOTH CFL ceilings exactly as |c_a| rises. **DERIVED: hold λ_eik FLAT or
DECAY it** (e.g. 0.10→0.05, or constant 0.05). Once |∇m|≈1 is established the constraint is
*maintained* cheaply; the weight buys enforcement stiffness we no longer need and pay for in
stability. This is a structural, viscosity-independent improvement. *Confirm:* a $0 n24 arbitration
arm (flat vs up-anneal λ_eik, ε fixed) — the sibling's arbitration harness already supports it.

**Arm 2 — DO-NOT-anneal ε to zero; FLOOR it, then make it ADAPTIVE (candidate a). `[PREDICTED]`**
- *Simple (derivation-forced):* the anneal→0 is HALF the re-entry mechanism (§3.1). Hold ε constant
  OR floor it at `ε_floor ≳ |c_a|_max √(η λ_eik/8)` (worst-case-sharpness lower edge). Trade: a
  constant isotropic ε over-damps the tangent in the EASY epochs (some d_seg cost) but never
  re-crosses.
- *Adaptive (the DERIVED optimum):* **closed-loop viscosity tracking the lower edge with minimal
  damping**

    **ε(t) = clamp( |c_a(t)| · √(η(t)·λ_eik(t)/8) · (1+margin),  ε_floor,  ε_upper(t) )**   (ADAPTIVE-ε)

  This is the *least* isotropic damping that keeps π_eik ≤ 1 at every t — a costate/CFL-tracking
  control, sister of the costate-controller design. **Cheap `|c_a(t)|` proxy:** the eikonal-residual
  telemetry already logs `mean((|∇m|−1)²)`; `|c_a| ≈ √(that)/mean|∇m|`, or simply track raw
  `mean|∇m|` per epoch (one no-grad reduction, already in the term). **DERIVED prediction: adaptive-ε
  ≥ fixed-ε on the d_seg-drift axis at the n600 horizon** (minimal tangential biharmonic cost in
  easy epochs), and STRICTLY stable by construction (π_eik ≤ 1 held). This is my **highest-confidence
  ranking claim** because it holds under BOTH symbol stories (it is isotropic → catches a tangential
  OR normal instability). *Confirm:* n600 A/B fixed-ε=0.3 vs adaptive-ε, d_seg-drift + skip-rate gates.

**Arm 3 — WARM-START from the preserved 0.026 gold checkpoint, but MEASURE κ FIRST (candidate d). `[PREDICTED — measurement-gated]`**
Warm-starting from the descended basin MAY have lower sharpness κ = λ_max(H) (larger CFL margin,
higher admissible η/lower required ε). BUT the 0.026 state was banked *at the runaway edge* (ep125,
deadlock ~ep110) — it may be a SHARP wall, not a safe basin. **Do not assume — MEASURE:** run the
sibling's Adam-preconditioned HVP λ_pre probe on the preserved `v5_dseg0026_preserved_20260705/BEST`
snapshot and compare λ_pre(0.026) vs λ_pre(ep100)=3.66e6. **DERIVED prediction: if λ_pre(0.026) <
λ_pre(ep100), warm-start from 0.026 with Arm 1+2; else the basin is NOT safer and the stabilizer must
carry it regardless.** This is a $0 measurement that decides a launch config — highest EV per effort.

**Arm 4 — NORMALIZED nᵀHn (candidate b): the DISCRIMINATING arm, NOT a foregone winner. `[PREDICTED — the direction discriminator]`**
Raw StEik `|∇m^T H ∇m|` self-amplified 575–1431× (measured NO-GO) because it carries a `|∇m|²`
prefactor — at the far-from-SDF state |∇m|≫1 the damping term is itself a runaway. The **normalized**
form removes the quartic scaling exactly:

    **κ_n := nᵀ H n = (∇m^T H(m) ∇m) / |∇m|²,   n = ∇m/|∇m|**            (normal-direction curvature; scale-free)

Add `W · κ_n²` (or `W·|κ_n|`) — damps ONLY the normal-direction second-order mode, tangent (lane
dashes) untouched → **best d_seg-drift IF the ill-posed mode is normal (§2 story H-A).** BUT the
symbol analysis (§2, a<1) puts the onset TANGENTIAL → normal-only damping might NOT stabilize alone
(story H-B). **DERIVED recommendation: build normalized κ_n AND run it COMPOSED with a small ε_floor
(belt+braces).** Then the n600 arm is informative either way:
- if `κ_n + ε_floor` beats adaptive-ε on d_seg-drift with equal stability ⟹ H-A (normal), and the
  anisotropic term is the d_seg-preserving cure — register it as the winner;
- if `κ_n` alone fails to stabilize while isotropic ε holds ⟹ H-B (tangential), isotropic viscosity
  is STRUCTURALLY required — a clean, valuable negative that corrects the StEik-direction reading for
  OUR field.

**The direction of the ill-posed mode is the one thing the derivation cannot settle alone — so it is
the one thing the arbitration MUST measure.** (A ready numpy reference for κ_n is in
`src/tac/boundary_math/eikonal_normal_curvature_reference.py` — §6.)

**Predicted ranking on the d_seg-drift axis at the n600 horizon (all measurement-owed):**
`adaptive-ε(t) ≥ fixed-ε=0.3` is HIGH-confidence (holds under both symbol stories). The
`normalized-nᵀHn` rank is CONDITIONAL: best case (H-A) it tops the list; H-B it fails to stabilize.
This is a *deliberate revision* of the task's suggested "anisotropic > adaptive-ε > fixed-ε" — the
symbol analysis will not license anisotropic-best unconditionally, and MEASUREMENT is the arbiter.

**lr as Δt:** keep **η = 1e-3** (the measured winning arm). Do NOT derive lr from `38/λ_pre` — that
constant is MEASURED-falsified (λ_pre=3.66e6 ⟹ 38-law says η_max≈1e-5 but 9.1e-5 is measured-stable,
~10× off). The correct DERIVED relation is the **coupled CFL** `π_eik = η λ_eik |c_a|²/(8ε²) ≤ 1`:
viscosity BUYS BACK step size because it replaces the grid-scale worst mode (σ~k_max²) with the
capped mode (σ~1/ε²) — this is WHY the winning arm could run at 1e-3 where the inviscid flow died.
η and ε trade off along the single inequality `η ≤ 8ε²/(λ_eik |c_a|²)`.

---

## 5. The general "calibrate-by-DE" framework (the operator's "configs", plural)

The campaign config is ONE coupled gradient-flow system: each knob sets a timescale of the SAME
level-set flow, and the knobs are tied by the **τ = ε = ħ** unification (softmax-temperature =
viscosity = semiclassical smoothing are the SAME vanishing parameter; the whole curriculum is one
vanishing-viscosity/Γ-convergence continuation). The recipe: **for each knob, name its governing DE,
the DERIVED optimum, and the MEASUREMENT that crowns it.** Extends the Tao π-group table
(`π_τ=τ/h`, `π_int`, `π_train`) with the eikonal-subsystem groups derived here.

| knob | governing DE / object | DERIVED optimum `[PREDICTED]` | measurement that crowns it |
|---|---|---|---|
| **ε_visco** | viscous-eikonal biharmonic regularization; two-sided window `ε_lower ≤ ε ≤ ε_upper` (§3) | ADAPTIVE `ε(t)=clamp(|c_a|√(ηλ_eik/8)(1+m), ε_floor, ε_upper)`; do NOT anneal to 0 | n600 A/B fixed vs adaptive; d_seg-drift + skip-rate |
| **λ_eik** | eikonal stiffness; `σ ∝ λ_eik`, both CFL ceilings `∝ 1/λ_eik` | FLAT or DECAY (stop the 0.05→0.10 up-anneal) | $0 n24 arm: up-anneal vs flat |
| **η (lr)** | flow time step Δt; coupled CFL `π_eik ≤ 1` (NOT π_EoS — constant falsified) | η=1e-3 with viscosity (buys the step back); tie to ε via `η ≤ 8ε²/(λ_eik|c_a|²)` | measured bracket (5e-4 stable / 9.1e-4 unstable) + the visco arm |
| **τ (softmax temp)** | margin sharpness ⟹ drives `κ(t), |c_a(t)|` (the progressive-sharpening source); τ=ε=ħ continuation | geometric anneal (Fisher-Rao CV≈0.39; Hazan/GNC triangulated); τ_end≈1 pixel pitch (π_τ=τ/h≈1) | already measured (CV confirmed); τ@400 eikonal-creep watch |
| **curriculum boundaries** | Γ-convergence adiabatic tracking; Allgower-Georg predictor-corrector | READINESS triggers (plateau + per-class nucleus), NOT clocks; re-treat at every boundary (rewarmup) | nucleus/plateau detectors; skip-rate at boundaries |
| **spike guard** | actuator on the flow, not the alarm | legacy alarm (SC1' every-epoch) + rollback ONLY with canary-trend trigger + best-verdict-anchored snapshots | measured: rollback-on-median is trigger-blind (non-cure) |

**The one-line discipline:** every knob's optimum is a *derived hypothesis with a mechanism*; the
config is "calibrated by DE" when each row's MEASUREMENT column is filled with a crowning n24/n600
arbitration. Until then the row is `[PREDICTED]`. This is the durable deliverable beyond the eikonal
cure — the template for calibrating any future knob (fewer blind sweeps, every sweep a pre-registered
DE test).

---

## 6. Reference operator (deliverable 3) — normalized nᵀHn, collision-free numpy reference

`src/tac/boundary_math/eikonal_normal_curvature_reference.py` — a NEW, default-safe, numpy-only
REFERENCE for `κ_n = nᵀHn = (∇mᵀ H ∇m)/|∇m|²` on a grid field (central differences on the interior),
for the #317 symposium to parity-check its MLX trainer term against (the established byte-identity
pattern). It does NOT touch the trainer (that region is #317's; collision avoided per #340). The
key property the reference PROVES numerically (self-test EXECUTED, PASSED, at |∇m| ∈ {1, 100, 2000}):
the **scaling-power law** `κ_n ~ c¹`, `raw StEik ~ c³`, so normalization removes exactly
`c² = |∇m|²`. κ_n is NOT scale-invariant — a normal second-derivative carries the units of m and
scales LINEARLY, the SAME class as the eikonal residual `(|∇m|−1)` it damps. What is removed is the
`|∇m|²` OVER-amplification of the raw form: at the measured runaway state |∇m|≈2000 the raw term is
**4×10⁶ = |∇m|²** times larger than κ_n (self-test row `removed_prefactor_scale_2000 = 4.0e6`) —
that is why raw self-amplified (575–1431×) and κ_n does not. (This corrects an initial "scale-
invariant" mis-statement caught by the reference's own self-test — the honest scaling law is linear,
and it is the |∇m|² *ratio* to raw, not invariance, that removes the blowup.)

---

## 7. Canonical equations — FORMALIZATION_PENDING (derivation is the mechanism; constant unmeasured)

Per the registry discipline (register only with a measured anchor; a clean NEGATIVE also anchors):

1. `eikonal_penalty_flow_illposedness_v1` — the (|∇m|−1)² L2 gradient flow (★) has principal symbol
   `σ(k) = −k_n² − c_a k_T²`, `c_a=(|∇m|−1)/|∇m|`; **backward-heat (ill-posed) in the tangential
   direction where |∇m|<1**; lr gates the discrete onset but is not a structural cure. ANCHOR: the
   v5 re-entry + the 5-arm matrix (qualitative POSITIVE); DIRECTION (normal vs tangential) pending
   the normalized-nᵀHn arm. `[FORMALIZATION_PENDING — mechanism = §1–2; direction measurement-owed]`
2. `viscous_eikonal_two_sided_window_v1` — capped growth `σ_max=|c_a|²/(4ε²)` ⟹ **lower edge
   `ε≥|c_a|√(ηλ_eik/8)`** (π_eik≤1) + **upper edge `ε≤√(2/(ηλ_eik))/k_max²`** (π_bih≤2). ANCHOR:
   measured ε=0.3 STABLE / ε=1.0 EXPLODES (bracket POSITIVE); the v5 re-entry as the lower-edge
   crossing. CONSTANT (8) MEASUREMENT-OWED. `[FORMALIZATION_PENDING]`
3. `eos_adam_preconditioned_threshold_v1` — stays the sibling's clean NEGATIVE (π_EoS=η·λ_pre/38
   FAILS: λ_pre=3.66e6). The DE derivation VINDICATES the negative: the runaway is the SPECIFIC
   eikonal π_eik, not the generic π_EoS. `[FORMALIZATION_PENDING — register with the negative]`

**Do not register unanchored; do not attach a constant a measurement has not crowned.**

---

## 8. Coordination + honest risks

- **Handed to the #317 v6 symposium** (SendMessage): the DERIVED-ranked cure stack (Arms 1–4), the
  λ_eik up-anneal prediction, the ADAPTIVE-ε law + `|c_a|` proxy, the warm-start λ_pre measurement,
  and the normalized-nᵀHn DISCRIMINATOR — so its longer-horizon n24/n600 arbitration CONFIRMS the
  derived optima and tests the ranking, rather than blind-sweeping. Derivation guides measurement;
  measurement crowns "optimal."
- **Risk 1 (symbol direction):** §2 leaves normal-vs-tangential open; Arm 4 is designed to measure it.
- **Risk 2 (constants):** every π-group constant is unmeasured; the 38 already failed. Ship FORM +
  directional predictions only.
- **Risk 3 (n24→n600 transfer):** the disease reproduces on the n24 slice (control 24.2×) — that is
  the transfer evidence; the v5 n600 re-entry is the true horizon; SC1' stays armed every epoch.
- **Risk 4 (isotropic over-damping):** adaptive-ε minimizes it by construction; fixed-ε pays it —
  the d_seg-drift column is the MEASURED check, not an assumption.

**HARD GATE restated: pointer 0.19110 UNMOVED. Everything here is MEANS; the first milestone remains
a byte-closed `upstream/evaluate.py` n600 exact row. No "optimal" without a measured anchor.**
