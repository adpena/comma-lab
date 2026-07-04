# Scaling-law FACET 4 — the adiabatic τ-schedule + the seeded-grow nucleation dynamics

**Facet 4 of the 5-facet "geometry-optimal scaling-law engineering" pass (task-adjacent to #284/#285,
2026-07-04). $0 research; NO heavy/paid/GPU; #205 SACRED READ-ONLY.** MEANS, not ends — the pointer
(contest-CPU **0.19110**) is UNMOVED and nothing here moves it. This facet fixes the *temporal
continuation* (the schedule τ(t)) and the *seed/interface* settings so that, given facet 3's
separatrix-seeded SDF, the seeded lane **refines + GROWS** rather than **births-from-zero** (impossible)
or **erodes** (#205's failure). Every schedule/eikonal/length setting below is **derived from the
geometry**, not hand-tuned; the flags are grepped from the live trainer (never invented).

Sisters: paper draft `deepmath_amortizing_argmax_paper_draft_20260704.md` (laws #6 τ=ε=ħ, #7
multiphase Modica–Mortola Γ-limit, #8 MCF minority-erasure) · memory
`lane_nucleation_failure_seed_above_critical_nucleus_20260704` · facets 1
(`scaling_law_facet1_metric_preconditioning`) / 2 (`..._facet2_intrinsic_manifold_parametrization`) /
3 (separatrix-seeded SDF, `scratchpad/facet3_seed/seed_probe.py`, DIL=2.0) · calibration memory
`feedback_calibrate_parametrization_fisher_geometry_common_unit_math_first_tiered`.

---

## 0. Where facet 4 sits (orthogonality to 1/2/3)

- **Facet 1 (metric)** shapes the per-step *direction* (Fisher/NTK preconditioner; Muon = κ-busting
  constant lever in the boundary basin).
- **Facet 2 (parametrization)** shapes the *basis + DOF* (shearlet cartoon exponent; Whitney
  mod_dim≈17–19, not 32).
- **Facet 3 (seed)** shapes the *initial field* (separatrix-seeded SDF, +2px dilation).
- **Facet 4 (this)** shapes the **annealing PATH τ(t)** (which basin the descent *lands in*) and the
  **eikonal/length/area settings** that keep the seeded thin lane **above the critical nucleus** while
  the flow runs. Facets 1/2 set *where the step goes*; facet 4 sets *the continuation the steps trace and
  the endpoint basin they select.*

---

## 1. Adiabatic τ-schedule = Fisher-Rao geodesic in the softmax-temperature coordinate

### 1.1 The metric of the anneal (DERIVED, $0-confirmed)

The curriculum `softmax_τ`, `τ: 1.0→0.05` (`--softmax-temp-start/-end`) IS the Maslov/Modica–Mortola
dequantization (law #6, `τ=ε=ħ`). Treat `τ` as a *coordinate* on the softmax family and compute its
Fisher metric. For a two-class pixel of margin `m` (logits `{0, m}`), `p = σ(m/τ)`:

- `dp_k/dτ = −(p_k/τ²)(z_k − z̄)`, `z̄ = Σ_j z_j p_j`.
- **Fisher metric of τ:** `g_ττ = Σ_k (1/p_k)(dp_k/dτ)² = Var_p(z)/τ⁴ = (m²/4τ⁴) sech²(m/2τ)`.
- **Information velocity:** `ds/dτ = √g_ττ = (m/2τ²) sech(m/2τ)`.

`Var_p(z)` is the *logit variance under p* — precisely facet 1's **margin-caustic** field
(`1−Σp² = tr F`, the exact 0.978 Fisher identity). So the anneal metric is the *same* Fisher metric
facet 1 preconditions; the schedule is **not a separate object** — it is the arc-length reparametrization
of the descent along the caustic.

**$0 confirmations** (`scratchpad/facet4_check.py`, numpy, synthetic — NOT committed):

- Info velocity `ds/dτ` **PEAKS at `τ_c = m/(2w*) = 0.2421·m`**, where `w* = 2.0653` solves
  `w·tanh w = 2` (numerically confirmed: peaks at 0.121, 0.243, 0.486, 0.972 for m = 0.5, 1, 2, 4 —
  exactly `0.242·m`). This `τ_c(m)` is the **per-pixel phase-transition temperature** (where the argmax
  is being decided).
- **Total Fisher-Rao arc length per pixel** over `τ∈[2, 0.05]` = `gd(m/2τ_end) − gd(m/2τ_start)` (gd =
  Gudermannian), **bounded by π/2 ≈ 1.5708** (confirmed 1.43 / 1.32 / 1.09 / 0.71 for m = 0.5/1/2/4).
  Higher-margin pixels travel *less* of the window (they freeze earlier) — the "pixels freeze once
  `τ ≪ m/4`" picture.

### 1.2 Why GEOMETRIC (log-spaced) is the no-wasted-motion schedule (DERIVED)

**Constant-information-velocity** (uniform Fisher-Rao arc-length per epoch = "no wasted motion") requires
`ds/dt = √g_ττ · |dτ/dt| = const`, i.e. **dwell where `g_ττ` is large**. `g_ττ` peaks at each pixel's
transition `τ_c(m)` — so the schedule **slows down near every argmax-flip**. This is *identically* the
**adiabatic theorem**: stay on the instantaneous ground state (the perimeter-minimizing argmax partition
at temperature τ, the Γ-limit of law #7) by changing the "Hamiltonian" slowly relative to the spectral
gap; the gap **→0 at the Maxwell/flip set** (critical slowing), so the dwell time `∝ 1/gap²` **→∞**
there. **Constant-Fisher-velocity and adiabatic-gap-dwell are the same schedule.**

For the **broadband margin spectrum** of an all-class cartoon boundary (a scale-free `p(m)∝1/m` over the
active octaves — the flip mass is spread 50%/19%/13% across Road/Lane/Undrivable at *many* curvature
scales), each octave of `τ` decodes one octave of margin scale `m` (since `τ_c ∝ m`). The `$0` check
confirms the **information-per-octave-of-τ is FLAT in `log τ`** (coefficient of variation ≈ 0.39,
normalized mean ≈ 1.0). ⟹ **equal epochs per octave = `--tau-anneal-shape geometric` IS the
constant-Fisher-velocity (geodesic / adiabatic) schedule.** This *derives* the paper §7 geometric
recommendation from geometry, upgrading the code's own heuristic note ("geometric damps late-τ d_seg
volatility") to "geometric is the τ-Fisher geodesic under a broadband margin spectrum."

Corollary (do-not-over-claim): if a *single* class had a dominant margin scale `m*`, the
constant-velocity schedule would concentrate at `τ_c(m*)` — approximated by `--tau-anneal-shape
cosine_hold` with `--tau-hold-frac` near that octave. Our spectrum is broadband ⟹ **geometric**; a
per-region schedule is a future refinement, not now.

### 1.3 The resolution floor (DERIVED direction; unit-convention caveat — HONEST)

The interface **half-width is `τ/2`** in SDF-margin units (law #6; eikonal-on-margin, paper §3).
Annealing below the render/stem Nyquist sharpens the interface **finer than the grid can represent** —
those epochs refine a boundary the render cannot place and the SegNet stride-2 stem (~2px Nyquist)
cannot read (paper §5 "below the stem Nyquist — needs a *store*, not a finer chart"). ⟹ **floor
`--softmax-temp-end` at the resolution scale** (paper §7: raise `0.05→~`resolution). The `$0` check shows
~27% of the *decision-info* sits below `τ=0.25` for the synthetic spectrum — but that info is
**render-limited (unreachable through R below the grid Nyquist)**, so it is **wasted COMPUTE, not wasted
signal**. **Caveat (NO-FAKE):** the exact `τ_end` depends on the SDF's pixel-unit convention, which I did
NOT verify in code — so `τ_end` is an **A/B to MEASURE** (`{0.05, 0.1, 0.25}`), not a single asserted
value. Direction is derived (raise the floor); magnitude is measurement-gated.

---

## 2. Seeded-grow nucleation dynamics + interface-width control

### 2.1 The failure mode (MEASURED, from the nucleation memory)

The `tau_softplus` stage is **sharp-limit mean-curvature flow (MCF, law #7/#8)**. Allen-Cahn/MCF
**critical-nucleus theorem:** a phase region below a critical size **shrinks to zero** under curvature
flow. #205 seeded the lane at `part_frac = 0.0` (`lane_px=0`, `lane_mean_iou=0.0`) ⟹ maximally
sub-critical ⟹ can *never* nucleate AND the residual lane is actively eroded ⟹ **d_seg CREEPS UP**
(0.004752@ep300 → 0.006568@ep400) while the smooth loss falls (surrogate↔hard-verdict decoupling).
Muon@726 sharpens but **cannot nucleate a zero-mass class** ⟹ #205 predicted NOT to reach the lane goal.

### 2.2 The seeded-grow fix (DERIVED from law #8 + the measured survival knee)

Given facet 3's separatrix-seeded SDF, the lane starts **on the boundary with mass > 0**. To make the
flow **grow** it (not erode), three geometry-fixed settings, each mapped to a live flag:

**(a) Seed ABOVE the critical nucleus** — dilate the seeded lane band `+2px` (native median 6px → ~8px).
Measured survival under the MCF-proxy smoothing: native `σ=0.8→0.941`, `σ=1.5→0.489` (51% erased);
`+1px→0.903`, `+2px→0.979`. **+2px clears the nucleus at both σ.** This is *exactly* facet 3's
`DIL=2.0` (+2px 1-Lipschitz outward shift of the zero-level-set). Flags:
`--structured-init --structured-init-include-lane --seed-islands --island-dilate-px 2 --containment-mode shield`
(default `island-dilate-px` is **1** → set **2**; `--seed-islands` requires `--structured-init`; shield
protects the seed grad from the bulk-CE wash).

**(b) RAISE the eikonal = interface-width control (#286)** — `--eikonal-weight 0.01→0.05`. The eikonal
`|∇φ|→1` pins the SDF unit-gradient so the interface half-width is a *real* `τ/2` and does **not smear
wider then collapse**. A sharp interface = the *low* effective-σ regime (measured `σ=0.8→94%` survival vs
`σ=1.5→49%`): raising the eikonal keeps the lane edge in the 94% regime. Optional triple-junction relax
(`--eikonal-junction-relax`, `--eikonal-junction-tau`) leaves the Herring-angle creases un-over-penalized.

**(c) KEEP the length SMALL — it IS the MCF-erosion driver** — `--length-weight 0.001` (default; do NOT
raise). The Chan–Vese perimeter term is the **surface tension**; its gradient is *precisely* the
curvature flow `V=−κ` that erodes the thin high-curvature lane. Raising it accelerates lane erosion.
Length legitimately smooths the *bulk* regions, so keep the small global value and add **per-class**
holds instead of a bigger global length. **This is the exact inversion of the naive "add more smoothing"
instinct.**

**(d) Add the per-class AREA driving force (auction-MBO surrogate) = the nucleus-basin lift** — the
principled cure to law #8 is a per-class area/volume constraint. Live surrogates, gated to the tau/MCF
stage:
`--lane-thin-weight <small> --lane-thin-start-epoch 300` (LEVER-B realized-margin thin-lane hinge, holds
the dropped-dash mass) + `--persistence-loss-weight <small> --persistence-warmup-epochs <n>`
(clDice + persistence island-recall = the auction-MBO area constraint; "births the finest-scale erasure
tail the CE drops") + optionally `--amplify-weight <small> --amplify-persist inverse_thickness`
(island-birth term, up-weights the thinnest tail). These supply the **bulk driving force f** that raises
the critical-nucleus basin so the seeded lane **grows**.

### 2.3 Contrast with #205 (the exact inversion)

| axis | #205 (erosion) | seeded-grow (this facet) |
|---|---|---|
| lane seed | `part_frac=0.0` (zero mass) | separatrix-seeded, `--island-dilate-px 2` (+2px, above nucleus) |
| eikonal | `0.01` (interface smears) | `0.05` (sharp `τ/2` interface, #286) |
| length | `0.001` (unconstrained MCF) | `0.001` (KEEP small) + per-class holds |
| per-class area | none during tau | `--lane-thin` + `--persistence` (auction-MBO surrogate) |
| result | MCF erodes → d_seg creep 0.00475→0.00657 | flow grows the seeded lane (A/B owed) |

---

## 3. Geodesic coupling to facets 1 & 2 (note the coupling; do NOT re-derive their parts)

- **Same metric.** The anneal's "no wasted motion" is measured in the **same Fisher metric facet 1
  preconditions** (`g_ττ ∝ Var_p(z)` = the margin caustic = the 0.978 identity). The facet-4 schedule is
  the τ-line geodesic; the *full* geodesic is its pullback onto **facet 2's intrinsic ~8-dim manifold** —
  and the **octaves of τ index facet 2's scale-strata** (coarse-to-fine = persistence order = Morse-Smale
  filtration). Geometric-in-τ ≡ coarse-to-fine on the manifold.
- **The schedule ROUTES compute to where facet 1 pays.** Constant-velocity dwells at the transitions,
  where the boundary-basin conditioning (facet 1's `κ≈19`) is worst — exactly where facet-1's **Muon
  κ-busting constant lever** has leverage. Muon (`--muon-start-epoch`) lives *outside* the τ-continuum
  (paper §4): the schedule hands off to it at `τ_end` (the resolution floor). Order: facet-2 DOF (mod_dim
  ≈17–19, not the over-embedded 32 that adds slow non-informative modes the constant-velocity schedule
  would waste epochs annealing) → facet-1 metric → **then** the facet-4 schedule is geodesic. Their
  exponents/constants are theirs; facet 4 only asserts the *path* is the geodesic in their metric on their
  manifold.

---

## 4. The concrete $0 config (real flags, grepped) + the pre-metric owed

Seeded-grow adiabatic config (additive to `witness_autoconfig`; A/B-owed, net-S #205-gated,
operator-GO + governor-gated per CONTAINMENT — this facet dispatches NOTHING):

```
# schedule (facet 4a): Fisher-Rao geodesic = geometric, floor at resolution
--tau-anneal-shape geometric --softmax-temp-start 1.0 --softmax-temp-end 0.1   # (0.1 = A/B vs 0.05/0.25)
# seed above critical nucleus (facet 4b, +2px = facet-3 DIL=2.0)
--structured-init --structured-init-include-lane --seed-islands --island-dilate-px 2 --containment-mode shield
# interface-width control (#286): raise eikonal, KEEP length small
--eikonal-weight 0.05 --length-weight 0.001
# per-class area driving force (auction-MBO surrogate), gated to the tau/MCF stage
--lane-thin-weight <small> --lane-thin-start-epoch 300 --persistence-loss-weight <small> --persistence-warmup-epochs <n>
# ease the ep300 tau+band collision (Ch.6, light touch)
--stage-transition-rewarmup-epochs 20
```

**$0 done (this facet):** the Fisher-Rao / adiabatic derivation numerically confirmed
(`scratchpad/facet4_check.py`: `τ_c=0.242·m`, arc-length `<π/2`, info-per-octave FLAT).
**Pre-metric OWED (T2, warm-started off a per-stage ckpt, per the calibration DAG — NOT this facet):** on
the SAME seed, A/B `geometric` vs `cosine` with the seeded-grow settings, measuring the **realized-
through-R d_seg trajectory** across the tau stage — the decisive test is *does the lane creep REVERSE*
(0.00475→ falling, not →0.00657). This is a "reach-the-floor" Tier-2 measurement, not a knob sweep.

**HONEST tags:** schedule = **DERIVED** (Fisher-Rao geodesic + adiabatic, $0-confirmed). Nucleation
config = **DERIVED** from law #8 + the **MEASURED** 6px/+2px survival knee (memory) + facet-3 DIL=2.0.
`τ_end` magnitude = **MEASUREMENT-GATED** (unit-convention caveat). Net-S impact = **CONJECTURE** (A/B +
byte-close owed, #205-gated). `scratchpad/facet4_check.py` [C] stripe-survival toy **over-smooths**
(whole-stripe erf-center → 99% vs the memory's 49% per-pixel erosion) — I defer to the memory's measured
numbers; the toy is directional-only (dilation monotonically helps), its magnitude is NOT cited.

---

## 5. THE ONE CONTRIBUTED SYNTHESIS CLAIM

> **The witness's temporal continuation has TWO geometry-fixed settings, neither hand-tuned, and they are
> the SCHEDULE and the SEED, decoupled:** (a) the annealing *path* is the **Fisher-Rao geodesic** in the
> softmax-temperature coordinate — `g_ττ = Var_p(z)/τ⁴` makes the information velocity peak at each
> pixel's transition `τ_c = 0.242·m` (`w·tanh w = 2`), so a **broadband cartoon-margin spectrum makes
> `--tau-anneal-shape geometric` (equal-epochs-per-octave) the UNIQUE constant-information-velocity /
> adiabatic schedule** (dwell `∝ 1/gap²` at the argmax-flip set, in the SAME margin-caustic metric facet 1
> preconditions, indexing facet 2's scale-strata); and (b) the *endpoint basin* is selected by
> **NUCLEATION, not the schedule** — under the tau-stage's sharp-limit MCF a phase below the critical
> nucleus erases inevitably (law #8), so the seeded lane must **start above it** (facet-3 +2px separatrix
> seed, `--island-dilate-px 2`, matching the measured 6px→8px survival knee) **and be held above it** by a
> per-class area driving force (`--lane-thin` + `--persistence` = auction-MBO surrogate) with the
> **eikonal RAISED** (`0.01→0.05`, interface-width control #286) and the **length KEPT SMALL** (`0.001`,
> because the perimeter term IS the MCF-erosion driver) — the exact inversion of #205's zero-seed
> unconstrained-MCF erosion (d_seg creep 0.00475→0.00657). **Schedule DERIVED + $0-confirmed; nucleation
> config DERIVED from law #8 + the measured survival knee; net-S CONJECTURE (A/B + byte-close owed,
> #205-gated). Pointer 0.19110 UNMOVED, MEANS.**
