# Tao's dimensional-analysis formalization → π-groups for the witness campaign

**Date:** 2026-07-05 · **Source:** operator-directed research — Tao, "A mathematical formalisation
of dimensional analysis" (2012-12-29). **Pointer 0.19110 UNMOVED — this is a MEANS lens; its value
is making calibrations TRANSFER and units bugs STRUCTURALLY impossible.**

## What Tao formalizes (compressed)
1. **Parametric approach:** dimensionful quantities are families x_{M,L,T} indexed by unit choices,
   transforming under the structure group (ℝ⁺)³ by weights: x has dimension MᵃLᵇTᶜ iff
   x_{M,L,T} = x̃·M⁻ᵃL⁻ᵇT⁻ᶜ. Dimensionally consistent laws = laws INVARIANT under the group action.
2. **Abstract approach:** dimensionful quantities live in 1-dim vector spaces/torsors (no canonical
   unit); only dimensionally consistent operations exist BY CONSTRUCTION — type-safety for physics.
3. **Buckingham π (the working tool):** any dimensionally consistent law collapses to a relation
   among DIMENSIONLESS groups; DA fixes the FORM (E = α·mc²) but never the dimensionless constant α.
4. **Failure modes:** dimensionless constants must be measured; multiple competing scales
   underdetermine; transcendental functions (exp, sin, tanh) are only legal on DIMENSIONLESS
   arguments; mixed-dimension sums destroy the classification.

## Why this is OUR lens (we already live its successes and its failures)

### We already paid for violating it (the units-bug class = DA failures)
- **The governor KiB/1e6-mislabeled-GiB bug** (mine §1, fixed #294/#298): a raw float crossed a
  boundary without its dimension; the band/admission math silently mixed two unit systems
  (anti-conservative 2.63 GiB). Tao's abstract approach says: this bug class exists BECAUSE we pass
  ℝ instead of a typed 1-dim torsor. Engineering translation: telemetry/ledger fields carry the
  unit IN THE NAME (`_gib`, `_px384`, `_cyc_per_unit`, `_per_pair`) and conversions happen at ONE
  boundary — already our post-fix practice; this memo makes it the stated LAW of the schema.
- **The axis-tag discipline** ([contest-CPU]/[contest-CUDA]/[macOS-MLX]) IS dimensional analysis on
  the evidence space: scores on different axes are different "dimensions" — never equatable, only
  separately measured. Our non-negotiable was independently derived DA.

### We already discovered its deepest instance (τ = ε = ħ)
The #284 deep-math law "τ=ε=ħ" is a DIMENSIONAL IDENTIFICATION: the softmax temperature τ, the
Γ-convergence interface width ε, and the semiclassical parameter are the SAME dimensionful scale
(length, in SDF units where |∇φ|=1 pins φ to length). `softmax(logit/τ)` is Tao's transcendental
rule enforced: the exponential's argument must be dimensionless, so logits are MEASURED IN τ UNITS.
The Γ-optimal τ_end≈1.0 = ONE PIXEL PITCH — the dimensionless group is **π_τ = τ/h** (h = grid
pitch), and the law "anneal τ_end→~1" is really "π_τ → 1". That form transfers to ANY resolution.

## The Buckingham-π ledger (nondimensionalize our measured laws → they TRANSFER)
The campaign's calibrations are currently stated with dimensionful constants pinned to one
configuration (n600, 512×384, 128 GiB, w_lane=6px). π-groups make them portable across n / grid /
tier — exactly what the tertiary 8GB sweep + n24 disambiguators need to feed the n600 truth:

| law (measured) | dimensionful form (as registered) | π-group form (proposed v2) |
|---|---|---|
| nucleation survival | σ=0.8→94%, σ=1.5→49% at lane w=6px | **π₁ = w/σ**: survival(π₁); knee π₁ ≈ 5±1 — resolution-portable critical nucleus |
| Γ-optimal τ anneal | τ_end ≈ 1.0 (at 512-grid) | **π_τ = τ/h → 1** (h = pixel pitch) |
| eikonal/length interface | λ_eik 0.05→0.10, ν=0.001 | φ~length ⟹ eikonal term dimensionless-per-area, length term ~length: **π_int = interface width/h** governed by (λ_eik, ν, τ) jointly — the ramp is holding π_int ≳ 1 as MCF narrows |
| dash-comb blindness | freq_along 8 vs dash 25 cyc/unit | **π_dash = f_basis/f_dash** ≈ 0.32 (deficit 3.2×) — the #277 lever restated: raise π_dash → ≥1 without aliasing (f_max·h ≤ ½, the Nyquist π-group) |
| focal-γ (symposium) | γ dimensionless ✓ | already a π-group — transfers across n by construction; only the SHARE table (dimensionless) needs measuring, at any n |
| verdict throughput | wall 2439s vs window 5042s | **π_duty = wall/window** ≈ 0.48; async-hide iff π_duty < 1 — the WF-F1 correction restated dimensionlessly |
| memory envelope | 67.68 GiB @128, 3.6 GiB smoke @8 | **π_mem = peak/RAM** + per-pair slope (GiB/pair): the config→envelope curve family IS a π-parameterization; tier transfer = same π-curve, different RAM |
| LR/steps (training dynamics) | lr 0.005, 600 steps pretrain | **π_train = lr × steps × sharpness** — why 600 steps under-absorbed the paint (π_train too small for the band's curvature); the scaling-law facet in DA form |

**Buckingham's caveat is our discipline restated:** DA fixes the FORM; the dimensionless constants
(the knee ≈5, γ*, α) must be MEASURED — "calibrate, don't guess" is π-theorem-mandated, not just
house style. And the "multiple scales" failure mode is real for us: grid pitch h, stem stride 2h,
camera pitch h', lane width w, dash period — where several compete (the R chain), DA
underdetermines and the measured knee wins (why the aliasing table had to be computed, not derived).

## Actions (small, concrete — no new apparatus)
1. **π-v2 registration pass** on the affected canonical equations (nucleation, τ-anneal, eikonal
   ramp, dash-deficit, verdict-duty, memory-envelope): APPEND dimensionless forms with the measured
   constants as anchors — via `tools/register_lever_laws` style. Makes the tertiary/n24 data
   formally transferable to n600. (Queued as the next equations-leg sync; ~1 unit.)
2. **Units-in-the-name schema law** (already de facto post-#294): every new telemetry/ledger field
   carries its unit suffix; conversions at ONE read boundary; review checklists check it. Recorded
   here as the standing rationale so the KiB/GiB class never recurs under a new name.
3. **Transcendental-argument audit** (cheap, one grep-pass when convenient): every exp/tanh/sin in
   the trainer takes a dimensionless argument (logit/τ ✓, β·sin ✓ (β dimensionless), check the
   Fourier features 2π·f·x — f·x dimensionless iff coords normalized ✓). Any raw-dimensionful
   argument found = a latent τ=ε=ħ-class bug.

Cross-refs: [[deepmath-amortizing-argmax-maslov-caustic-tau-eps-hbar]] (τ=ε=ħ = the DA
identification) · #294/#298 units fix (the paid failure) · symposium
`council_grand_symposium_levelset_loss_geometry_20260705.md` (γ = already-dimensionless knob) ·
tertiary sweep spec (tier transfer = π_mem curves). Pointer 0.19110 UNMOVED.
