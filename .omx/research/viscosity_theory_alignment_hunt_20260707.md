# Viscosity-solution theory alignment hunt — Barles–Souganidis · selection · homogenization · weak KAM · comparison · WKB · junctions

**Operator directive 2026-07-07: "Research our problem space online against viscosity solution
theory and hunt for gestalt and deep math alignment and eureka." Lineage: #284 (Amortizing the
Argmax, τ=ε=ħ) + #318 (DE-derivation, π_eik CFL window) + #286 (Γ-optimal τ-anneal) + #320
(adaptive-ε cure). Axis discipline: EVERYTHING here is MEANS (theory alignment + config/lever
implications); no number below is a score; every literature claim is cited; every alignment is
checked against a NAMED measured anchor; every verdict names its engineering bridge. Pointer
contest-CPU 0.19110 UNMOVED.**

---

## 0. The measured anchors this hunt is checked against

| anchor | value | provenance |
|---|---|---|
| lane-dash erasure; error ∝ 1/persistence | dashes = binding residual; MBO probe attributes **95.7%** of smoothing cost to Lane | #284 §2.8 (`mcf_minority_erasure_inevitability_v1`) |
| dash-gap FP | FP = **0.00396** = 90% of band recon; range-dependent; dash PHASE = ego-distance (screw ξ) | `project_dashgap_fp_deepdive_range_dependent_ego_phase_fp_as_signal_20260701` |
| residual fragmentation | 441,329 flips in 142,270 comps, mean 3.1 px; **44.6% singletons** (n600, ep425) | `contour_string_flip_coding_n600_20260707.md` (#307 NO-GO) |
| along-tangent frequency deficit | **3.2×** (along 8 vs across 32 effective) | `lane_dash_residual_root_is_along_tangent_freq_deficit_R_allpass_20260703` |
| tau-stage saturation → cold-Muon near-net-zero | live: 0.003366 pre-boundary → transient **0.004351** at the switch → slow recovery 0.004163@ep800; tau saturated ~70 ep before the fixed boundary | live control run + DRAFT council §14 |
| long-budget tail shape | long900: 0.002423@ep450 → 0.002334@500 → 0.002176@800 → 0.002017@900 (slow, non-exponential tail) | DAG (witcap_kd_c1_long900) |
| eikonal ep110 instability | v5 re-entry at ε≈0.27 (two-sided CFL squeeze); checkerboard-scale (k_max) fastest-growing mode | #318 §2–3.1; FEED-05y |
| annulus concentration; lane pinned | ~97% of d_seg in ~4.7%-area annulus; Lane (cls-1) **stuck** | #333 (`costate_loop_closed_annulus_sense_curriculum_event_act_20260706`) |
| junctions weak | CE→tau junctions −14% but triple-junction DOF measured WEAK | 4-lens DOF probe (`lane_dash_residual_...20260703`) |
| lane nucleus sub-critical | LANE survival under σ=1.5 tau/MCF: native 44.6% → +1px 90.0% → +2px 98.3% | DAG FEED (nucleus-knee probe, n24) |
| the discretization | gradient descent on a τ-smoothed level-set energy over a coordinate-INR, rendered through R (bicubic↑ → uint8 → bilinear↓) into a frozen CNN argmax | trainer; #284 |

---

## 1. Barles–Souganidis: is our training discretization "monotone"? — NO, and the predicted artifact classes MATCH

**Theory.** Barles–Souganidis 1991 (*Asymptotic Analysis* 4:271–283): any **monotone, stable,
consistent** approximation scheme converges to the (unique, comparison-principle-protected)
viscosity solution of a fully nonlinear 2nd-order PDE. Monotonicity is the load-bearing leg:
non-monotone schemes can converge to **wrong weak solutions** or develop spurious oscillations
near kinks/shocks — the classical counterexamples are high-order/spectral discretizations at
non-smooth points. The standard repair is the **filtered scheme** (Froese–Oberman, arXiv
1204.5798, SINUM 2013; extended to time-dependent HJB by Bokanowski–Falcone–Sahu, arXiv
1611.04939): blend an accurate non-monotone scheme S_acc with a monotone scheme S_mono via
`S = S_mono + Δx·F((S_acc − S_mono)/Δx)`, where the filter F falls back to S_mono exactly where
the two disagree at O(Δx) (i.e., near singularities) — provably convergent, high-order where
smooth, and "the filter is a singularity detector."

**Our scheme, classified.** A coordinate-INR trained by gradient descent is a **global
spectral/collocation discretization** — the least monotone scheme class there is (every basis
function has global support; a local update moves values non-locally with both signs). Two
levels of failure:
1. **Continuum level:** the inviscid eikonal penalty flow is itself ill-posed (backward heat
   tangentially where |∇m|<1 — #318 §2). BS doesn't even apply until the limit equation has
   comparison; ε-viscosity restores it (§5 below).
2. **Scheme level:** even with the viscous (well-posed) limit, the INR+SGD scheme is
   non-monotone, so BS gives NO convergence guarantee to the viscosity solution; the theory
   predicts (a) **grid/UV-scale spurious oscillation at singular sets** and (b) **wrong
   selection at shocks/kinks/junctions**.

**Alignment vs anchors — the prediction table matches point-for-point:**
- (a) predicted spurious oscillation at the singular set → measured **boundary jitter**: ~97% of
  d_seg in the 4.7% annulus (the kink set of m), and the ep110 runaway's fastest-growing mode is
  literally **checkerboard-scale k_max** (#318 §2) — the textbook non-monotone signature.
- (b) predicted wrong selection at non-smooth structures → measured **triple-junction weakness**
  (junction DOF weak) and the fragmented **44.6%-singleton flip confetti** (no coherent wrong
  region — exactly high-frequency selection noise, not a displaced front).

**The repair already half-exists — name it as what it is.** The Froese–Oberman filter's
structure is: *monotone fallback applied selectively where the accurate scheme misbehaves*. Our
`--eikonal-viscosity-adaptive` with floor 0.3 (#320) is the **global** monotone envelope
(uniform isotropic ε = always-on fallback = pays d_seg drift in easy regions — the exact
"monotone schemes have limited accuracy" cost BS-era numerics pays). The **filtered** form is
the ALREADY-BUILT band-restricted variant `--eikonal-visco-ca-band 0.5` plus, one step further,
per-pixel ε gated on the backward-heat indicator `c_a<0` (fire viscosity ONLY where |∇m|<1 in
the annulus): least damping, provable-class stabilization. This upgrades the ca-band flag from
"escalation option if the floor fails" (#320 §3) to "the theory-preferred form" (a filtered
scheme in the Froese–Oberman sense).

**VERDICT: CONFIRMS-existing (+sharpens).** Confirms `adaptive_eps_cfl_edge_tracking_v1` and the
#320 escalation path; sharpens it: the ca-band/per-pixel-gated ε is not a tweak, it is the
filtered-scheme construction with a convergence pedigree. **Engineering bridge:**
`--eikonal-visco-ca-band 0.5` promoted from escalation to first-line in the next-run config
(council draft §on eikonal); optional new lever spec: per-pixel ε mask `ε(x) = ε·1[c_a(x)<0]`
(DSL `EikonalViscoStabGauge` extension — a `Lever` factory, not a hand flag, per triality).

---

## 2. Vanishing-viscosity/vanishing-discount SELECTION: which partition does the limit pick?

**Theory.** Crandall–Lions: the ε→0 viscous limit selects the unique viscosity solution among
the (generally infinitely many) Lipschitz a.e.-solutions. Two sharper modern selection results:
- **Vanishing discount** (Davini–Fathi–Iturriaga–Zavidovique, *Inventiones* 206 (2016) 29–55):
  the discounted equation `λw_λ + H(x,Dw_λ) = c(H)` has a UNIQUE solution for each λ>0, and
  w_λ → w₀, a *specific* weak-KAM solution (maximal subject to Mather-measure constraints
  ∫w dμ ≤ 0). The discount term restores uniqueness at finite λ and picks a canonical limit.
- **Vanishing noise after long time** (Gao–Liu, arXiv 2208.11860): taking t→∞ FIRST and THEN
  noise→0 selects the weak-KAM solution given by the Freidlin–Wentzell variational formula with
  boundary data on the **Aubry set** (self-consistent at each local attractor).

**Mapping to us.** Multiple weak solutions ↔ multiple argmax partitions with equal (τ-resolved)
loss. Below the Maslov free-energy resolution **τ·ln 5** (#284 law 1), partitions are
loss-indistinguishable and the training limit is decided by *implicit selection*, not by the
objective. Our knobs map onto the two selection mechanisms exactly:
- **Weight decay = the discount λ.** At the field level, WD adds a `λ·(state)` damping term —
  the DFIZ discount. Finite WD ⇒ unique selected limit; WD→0 with no replacement ⇒ selection
  falls to noise order-of-limits. This gives a *mechanism* to the measured "Muon: KEEP the WD"
  (`muon_deep_dive_keep_and_tune_finishing_stage_schedule_not_switch_20260703`): the discount is
  what keeps the finishing-stage limit canonical and reproducible.
- **SGD minibatch noise + long-time-then-anneal = the Gao–Liu order of limits.** We run long at
  finite noise, then cool (LR decay). Theory says this selects boundary data on the Aubry-set
  analog — the *most stable* (max quasi-potential-basin) partition, which is NOT guaranteed to
  be the d_seg-min partition when d_seg differences live below τ_end·ln 5.
- **τ_end floor.** The #285/ch.4 raise of `--softmax-temp-end` 0.05→~1.0 sets the resolution
  scale; the selection risk window is exactly [d_seg gaps smaller than the τ_end free-energy
  gap]. This is the principled meaning of the τ_end floor: below it you are buying selection by
  implicit bias, not by loss.

**Alignment vs anchors.** Consistent with: seed-stable partitions under the deterministic-repro
spine (finite WD + seeded noise = discounted selection = unique); the cold-Muon near-net-zero
(the finishing stage is operating in the selection regime — the loss surface is flat at τ_end
resolution, so big optimizer changes move d_seg little; matches 0.004163@ep800 slow drift).

**VERDICT: CONFIRMS-existing.** Confirms keep-WD (#269/#270 restart config), the τ_end floor
(#285), and the deterministic-repro spine (comparison+discount ⇒ unique limit). **Engineering
bridge:** (i) council draft §finishing-stage: state WD's role as the discount/selection term —
do NOT zero it late; (ii) $0 A/B spec (owed, not run): two seeds × {WD>0, WD=0} at the Muon
stage, metric = Hamming distance between final partitions — prediction: WD>0 pair agrees, WD=0
pair diverges. Registration only after that anchor exists.

---

## 3. HOMOGENIZATION — dash erasure IS a homogenization phenomenon; the #287 dash-comb IS the cell-problem corrector ⇒ **EUREKA candidate #1**

**Theory.** Lions–Papanicolaou–Varadhan (1987, unpublished-but-canonical; see e.g. World
Scientific *Homogenization of Hamilton–Jacobi equations* survey, and Evans's adjoint-method
papers): for `H(x/δ, Du_δ)` with δ-periodic microstructure, u_δ → ū solving `H̄(Dū) = 0` where
the **effective Hamiltonian H̄** is defined through the **cell problem** `H(y, P + Dv(y)) =
H̄(P)`; the microstructure survives ONLY in the **corrector** v(y), via the two-scale expansion
`u_δ(x) ≈ ū(x) + δ·v(x/δ)`. The same architecture holds variationally: periodic perimeter/
Allen–Cahn energies Γ-converge to a homogenized **anisotropic** perimeter (Ansini–Braides–
Chiadò Piat; Braides' Γ-convergence programme; recent: arXiv 2010.05849, 2601.08677). Two
refinements that matter to us: (i) homogenized surface tensions can develop **facets/gradient
discontinuities and ZERO effective mobility** — interfaces in periodic media generically **PIN**
below a forcing threshold (Dirr–Yip; arXiv 2108.00558 "zero mobility for Allen–Cahn and
curvature flows in periodic media"); (ii) the effective H̄ has flat parts tied to the
Aubry–Mather set of the cell dynamics (the D4 bridge).

**The identification.** Lane dashes are literally a 1-parameter periodic microstructure along
the lane tangent: period ≈ dash+gap length (scale δ, in effective 512×384 pixels, shrinking
with range — the measured range-dependence). Three smoothing scales sit above it: τ (interface
half-width τ/2), ε (viscous cutoff √|c_a|/ε), and R's bilinear↓ (Nyquist). Whenever
min(smoothing scales) ≳ δ_along, the training flow can only see the **homogenized** object: a
solid anisotropic band with effective (averaged) surface tension — the dash phase is
*integrated out of the energy*. The INR then renders the homogenized solution. Predictions,
each already measured:

1. **The homogenized band overfills the gaps** → dash-gap FP should be ≈ the band-reconstruction
   FP. MEASURED: FP 0.00396 = **90% of band recon** (dashgap deep-dive). ✔
2. **Erasure order = below-homogenization-scale first** → error ∝ 1/persistence; finest
   along-tangent features die first. MEASURED (MBO 95.7% Lane; erasure ∝ 1/persistence). ✔
3. **Pinning**: below the depinning threshold the homogenized interface has zero effective
   mobility → the lane boundary should be *stuck*, not slowly improving. MEASURED: Lane (cls-1)
   stuck in the annulus while other classes descend (#333); tau-stage saturation on the lane
   long-tail. ✔ (This gives the pinning literature's mechanism to the measured "lane stuck".)
4. **The coarse chart cannot hold the corrector**: the two-scale expansion says the fine
   structure lives in a SEPARATE δ-periodic term, not in ū's chart → raising the coarse chart's
   capacity should NOT recover dashes (capacity-alone measured NOT to pay; bigcap overfits),
   while the along-tangent frequency deficit (3.2×) is exactly "the chart lacks the corrector's
   frequencies". ✔

**The corrector = the #287 tropical dash-comb.** The cell problem's solution for a dashed lane
is a δ-periodic modulation of the band field along the tangent coordinate — parametrized by
(period, duty cycle, **phase**). The #287 max-plus comb is exactly this object, and its phase is
transported by the ego screw ξ (dash PHASE = ego-distance — MEASURED in the dashgap deep-dive).
So the two-scale expansion *is* our architecture split:

    witness field (coarse ū, INR)  +  analytic comb corrector (δ·v(x/δ), phase from ξ)

with the corrector **rule-118 FREE** at decode (generic algorithm; the counted payload is the
per-lane period/duty/phase residuals — bytes we already planned for #287). Homogenization theory
upgrades #287 from "a lever that might help" to "the principled unique repair of a named
erasure mechanism": you cannot train the coarse flow into representing sub-homogenization-scale
structure (pinning + Γ-limit both forbid it); you must add the corrector term.

**Honest boundary.** The identification is mechanism-level, not yet a registered law: (a) the
"homogenization scale" here is set by τ/ε/R jointly and none of our runs has MEASURED the
crossover (d_seg(dash recovery) vs δ_along/τ curve); (b) LPV is for H(x/δ) oscillatory
Hamiltonians — our microstructure is in the TARGET (the data term), not in H; the variational
(Braides/Γ-convergence of the perimeter+data energy) reading is the rigorous route, and pinning
results are for forced MCF, which our #284 law-8 (training flow = MCF in the sharp limit)
justifies importing. (c) The corrector A/B (witness+comb vs witness alone, n600, through R) is
UNMEASURED.

**VERDICT: EUREKA-candidate (the strongest of the hunt).** Upgrade path: register
`dash_erasure_homogenization_corrector_v1` (spec §9.1) once the corrector A/B lands.
**Engineering bridge:** (i) #287 dash-comb moves up the council draft's lever ranking with a
*law-shaped* justification (addendum §15); (ii) the comb MUST be a DSL `Lever` factory
(render-time analytic band × max-plus comb, phase from ξ) — not a hand flag; (iii) the τ_end
floor gets a SECOND meaning: τ_end also sets the homogenization crossover — do not anneal τ
below the dash period unless the corrector is active (otherwise you pay boundary-jitter without
buying dash recovery); (iv) a $0 crossover probe exists: sweep τ on a checkpoint and measure
dash-gap FP vs τ/δ_along (pre-registerable).

---

## 4. Weak KAM / Aubry–Mather / Lax–Oleinik: the lane long-tail as the projected Aubry set; the saturation SHAPE

**Theory.** Fathi's weak KAM theorem: the Lax–Oleinik semigroup `T_t u₀` converges as t→∞ to a
weak-KAM (viscosity) solution; T_t is Legendre–Fenchel conjugation iterated (the just-ledgered
2606.09077 connection). Convergence is obstructed exactly on the **Aubry set** (where the
Peierls barrier vanishes — no strict descent available); the RATE can be genuinely slow:
even with a hyperbolic periodic-orbit Aubry set there are examples where the rate **cannot beat
O(1/t)** (arXiv 1109.3327, Wang–Yan-type Lax–Oleinik rate results). Away from the Aubry set,
convergence is exponential; ON it, polynomial.

**Mapping + alignment.** The lane long-tail is where our flow has (a) zero effective mobility
(pinning, §3), (b) margins pinned at ≈0 (the annulus), and (c) no strict descent below τ·ln 5
resolution — three independent reasons it is the **convergence-obstruction set** of our
semigroup, i.e., the projected-Aubry-set analog. Prediction: the d_seg trajectory should be
exponential early, then cross over to a **polynomial (power-law) tail carried by the lane
class**. Measured shape is consistent: long900's tail (0.002423@450 → 0.002176@800 →
0.002017@900) is far slower than any exponential fit through the early stage, and the live
control's tau stage saturates ~70 ep before the fixed boundary — but an actual `a + b·t^(−α)`
fit has NOT been run, so this stays qualitative.

**VERDICT: CONFIRMS-existing (qualitative; fit owed).** The valuable operational content is the
**detector shape**: every plateau/exit classifier we run (SC1', the costate slope detectors, §14
item-5 exhaustion criteria) implicitly assumes descent shapes; weak-KAM rate theory says the
late stage is power-law on the binding class, so (i) exponential-plateau detectors will fire
EARLY on the lane class (declaring exhaustion while a 1/t tail still pays), and (ii) per-class
exit criteria should fit `a + b·t^(−α)` and exit on the *extrapolated remaining meat*, not on a
window slope. **Engineering bridge:** costate/plateau detector upgrade (power-law fit option,
per-class) — DSL Schedule primitive parameter, feeds §14 item 5; $0 retro-fit on the existing
long900 + live-control trajectories (pre-registerable: α_lane < α_road).

---

## 5. Comparison principle audit of the loss stack: which terms threaten uniqueness?

**Theory.** Uniqueness of viscosity solutions = the comparison principle, which needs (i)
degenerate ellipticity (monotonicity in the Hessian argument) and (ii) properness
(monotonicity in u). Terms that break either lose uniqueness ⇒ the trained partition becomes
selection-dependent (seed/schedule-sensitive) — the deterministic-repro spine's PDE face.

**Audit of our terms:**
| term | operator reading | comparison verdict |
|---|---|---|
| eikonal `(|∇m|−1)²`, inviscid | backward heat where |∇m|<1 | **VIOLATES** degenerate ellipticity — the ONE proven violator; measured (ep110) |
| + adaptive-ε viscosity (#320) | biharmonic/isotropic damping | RESTORES well-posedness (capped growth, #318 §3) |
| focal reweighting | state-dependent mobility w(p)≥0 | preserves (mobility ≥ 0 = time reparametrization); *degenerates* where w→0 — can add pinning, not non-uniqueness |
| per-class λ / island amplification | anisotropic weighting of a proper operator | preserves |
| logit-adjust-τ / AHA offset | constant Hamiltonian shift | preserves |
| length/perimeter term | mean-curvature (degenerate elliptic) | preserves (it IS the classical example) |
| fixed-β hosc (saturating) | vanishing-gradient plateaus | not a comparison issue; an optimizer stall (already measured DIVERGES — separate law) |
| weight decay | +λu properness term | **STRENGTHENS** comparison (the discount, §2) |

**VERDICT: CONFIRMS-existing.** The stack is comparison-safe EXCEPT the inviscid eikonal, which
#320 already cures; WD actively helps. **Engineering bridge:** a one-line discipline for the
council draft: *any new loss term is classified by its second-order symbol sign (degenerate
elliptic? proper?) before wiring* — the D5 table is the template; cheap to keep as a review
checklist row (no new gate needed; the review contract's §5 evidence-tag pass can carry it).

---

## 6. Large deviations / WKB: Hajek's log-schedule does NOT apply to τ (τ is GNC, not temperature) — and the Muon transient is a quench signature

**Theory.** Viscous HJ = Hopf–Cole log-transform of a diffusion; ε→0 is WKB; Freidlin–Wentzell
quasi-potentials govern noise-driven escape with times exp(ΔE/ε); Hajek (Cooling schedules for
optimal annealing, *Math. OR* 1988): T(t) ≥ Γ/log t is necessary AND sufficient for simulated
annealing to reach the global minimum (Γ = deepest local-minimum escape depth). Gao–Liu
2208.11860 re-derives weak-KAM selection from FW theory (§2).

**The alignment — and the refutation.** The tempting import ("replace #286's Γ-geometric τ-anneal
with Hajek-logarithmic") is WRONG, because our τ is **not the escape temperature**: τ is the
smoothing/interface width of a graduated-non-convexity (GNC)/continuation homotopy — #284 §3
(τ=ε=ħ) makes it the Maslov dequantization parameter, and the correct requirement for a
homotopy is **adiabatic tracking** (stay on the continued minimizer branch; no bifurcation
jumps), not logarithmic cooling. The quantity Hajek DOES govern is the **actual noise
temperature = SGD minibatch noise × LR**. Consequences, both matching measurement:
- The measured ep300 stage-boundary bump (d_seg 0.0056→0.020 transient) and the live Muon-switch
  transient (0.003366 → 0.004351) are **quench/adiabaticity failures** (the continuation was
  kicked off its branch), not annealing-schedule failures — cured by re-warmup/easing (#269
  warm-start momentum, rewarmup-epochs; both built). CONFIRMS #269/#270.
- LR decay is the Hajek-governed cooling: cooling LR geometrically FAST late freezes the basin
  (fine when we're in the selection regime and WANT to freeze — §2 — but it means the late stage
  cannot escape a wrong basin; the escape budget must be spent BEFORE τ_end/LR_end).

**VERDICT: REFUTES the naive import (Hajek-log for τ); CONFIRMS-existing** (#286 Γ-geometric =
constant-relative-speed in scale, the scale-space-correct homotopy schedule, measured CV≈0.39;
#269/#270 rewarmup as the adiabatic repair). **Engineering bridge:** none new to build; the
council draft §14 item 2 gets the sharpened principle: *τ-path = homotopy (geometric, event-
gated adiabaticity); LR-path = temperature (spend escape budget early, freeze late; never
re-raise LR after entering the selection regime without also re-raising τ)*.

---

## 7. Junctions/networks: flux-limited solutions say the junction condition is a FREE PARAMETER — fit it from the frozen scorer ⇒ EUREKA candidate #2

**Theory.** Imbert–Monneau (Ann. Sci. ENS 2017; arXiv 1306.2428; multidimensional 1607.03996):
HJ equations on networks/junctions are NOT well-posed by the PDE alone — well-posedness requires
a junction condition from a 1-parameter family indexed by a **flux limiter A**; comparison holds
via their vertex test function; different A = different (all legitimate) viscosity solutions.
Meanwhile the multiphase Modica–Mortola Γ-limit (#284 law 7, Baldo/Sternberg) imposes **Herring
angle conditions** at triple junctions determined by the surface-tension matrix σ_ij; with equal
σ_ij (our current single length weight) that means 120°–120°–120°.

**The mismatch.** The frozen SegNet's partition has junction angles set by a CNN, with NO reason
to satisfy equal-tension Herring conditions. Our perimeter/length regularizer therefore imposes
a *wrong junction boundary condition* — the flow fights the target exactly at triple points.
That is a mechanism for the measured triple-junction weakness (junction DOF WEAK; junctions
improve −14% CE→tau then stall). The theory names the missing degree of freedom: the junction
condition (flux limiter / σ_ij matrix) is a **fittable parameter, not a fixed law**.

**The fit is $0 and data-driven:** measure the empirical junction angles of the frozen SegNet
argmax partition on GT (per class-triple, n600 statistics — the junction inventory already
exists in the 4-lens DOF probe), invert **Young's law** (σ₁₂/sin θ₃ = σ₂₃/sin θ₁ = σ₁₃/sin θ₂)
to get the pairwise tension ratios σ_ij, and weight the length/perimeter term **per class-PAIR**
with those ratios. Per-class λ exists in the DSL; per-class-PAIR σ_ij is the upgrade the theory
demands. (The full Imbert–Monneau flux-limiter machinery is the rigorous umbrella; the σ_ij
Young-angle fit is its cheapest actionable projection.)

**VERDICT: EUREKA-candidate #2** (smaller EV than §3 — junctions are a minority of the residual
vs the lane comb, but the fit is $0 and the lever is a weight matrix we already know how to
wire). **Engineering bridge:** (i) $0 probe: junction-angle histogram per class-triple from the
cached GT argmax → σ_ij table; (ii) DSL: extend the length-term `Lever` to accept a σ_ij matrix
(default all-ones = today's behavior, byte-identical OFF); (iii) council draft §15 carries it as
a treatment arm, NOT in the clean baseline. Registration spec §9.2.

---

## 8. Ranked verdict table (EV order)

| # | direction | verdict | binding evidence | engineering bridge (exact surface) |
|---|---|---|---|---|
| 1 | §3 homogenization | **EUREKA-candidate**: dash erasure = homogenization; #287 comb = cell-problem corrector; lane-stuck = pinning (zero effective mobility) | dash-gap FP 0.00396=90% band recon ✔ · erasure ∝ 1/persistence ✔ · lane stuck ✔ · capacity-doesn't-pay ✔ · corrector A/B OWED | #287 comb as DSL `Lever` (phase from ξ, rule-118 free) · τ_end ≥ dash-period rule · $0 τ-crossover probe · council §15 rank-up |
| 2 | §1 Barles–Souganidis | CONFIRMS+sharpens: INR+SGD is non-monotone; predicted artifacts (UV jitter, junction mis-selection) MATCH; ca-band ε = Froese–Oberman filtered scheme | ep110 checkerboard mode ✔ · 97%-annulus jitter ✔ · 44.6% singleton confetti ✔ | promote `--eikonal-visco-ca-band 0.5` to first-line; per-pixel ε(x)=ε·1[c_a<0] as DSL gauge extension |
| 3 | §7 junctions | **EUREKA-candidate #2**: junction condition is a free parameter (flux limiter/σ_ij); fit from scorer's Young angles | triple-junction WEAK ✔ · fit UNRUN | $0 junction-angle→σ_ij probe; per-class-PAIR length weights (DSL length-Lever matrix arg) |
| 4 | §4 weak KAM | CONFIRMS (qual.): lane long-tail = Aubry-set analog; late convergence power-law O(1/t)-class, not exponential | long900 tail shape ~✔ (fit owed) · tau early-saturation ✔ | plateau/exit detectors: per-class `a+b·t^(−α)` fit; $0 retro-fit on logged trajectories |
| 5 | §2 selection | CONFIRMS: WD = DFIZ discount (canonical unique limit); τ_end = selection resolution τ·ln5; long-time-then-cool = Gao–Liu Aubry-boundary selection | keep-WD measured ✔ · cold-Muon flat drift ✔ | keep WD late (never zero); $0 A/B spec (WD×seed partition-Hamming) |
| 6 | §6 WKB/annealing | REFUTES naive Hajek-log for τ (τ=GNC homotopy, not temperature); CONFIRMS Γ-geometric + rewarmup; LR is the Hajek variable | ep300 bump + Muon transient = quench ✔ · CV 0.39 ✔ | §14-2 principle: τ=homotopy/LR=temperature; escape budget before freeze |
| 7 | §5 comparison | CONFIRMS: only the inviscid eikonal violates comparison (cured #320); WD strengthens it; focal→pinning-risk not uniqueness-risk | ep110 ✔ | review-checklist row: classify new loss terms by symbol sign |

**Refuted/blocked this hunt:** (a) Hajek-logarithmic τ-schedule import (wrong object — §6);
(b) "junction weakness needs a new architecture" (no — it needs the σ_ij/flux-limiter DOF — §7);
(c) "train the dashes back with more capacity/epochs" (homogenization+pinning forbid it below
the crossover scale — §3, consistent with the measured capacity NO-GO).

---

## 9. Canonical-equation REGISTRATION CANDIDATES (specs ONLY — do NOT register without the named anchors)

### 9.1 `dash_erasure_homogenization_corrector_v1`
- **Law:** For a δ-periodic along-tangent microstructure (dash period δ_along) under smoothing
  scales (τ, ε, R-Nyquist), when min-smoothing-scale ≳ δ_along the training flow converges to
  the homogenized (solid anisotropic band) solution with zero effective mobility of the lane
  interface (pinning); sub-δ structure is recoverable ONLY via a corrector term
  `δ·v(x/δ)` = the max-plus dash comb with phase transported by the ego screw ξ (two-scale
  expansion `u ≈ ū + δ·v(x/δ)`), never by coarse-chart capacity.
- **Cited theory:** Lions–Papanicolaou–Varadhan cell problem; Braides/Ansini–Braides–Chiadò Piat
  Γ-homogenized anisotropic perimeter; Dirr–Yip + arXiv 2108.00558 pinning/zero-mobility.
- **Anchors HELD:** dash-gap FP = 90% of band recon (0.00396); erasure ∝ 1/persistence (MBO
  95.7% Lane); lane cls-1 stuck (#333); capacity NO-GO (bigcap overfits); dash phase = ego-dist.
- **Anchors OWED before registration:** (i) corrector A/B — witness+comb vs witness alone,
  n600 through R, Δd_seg with byte-accounting; (ii) the τ-crossover probe (dash-gap FP vs
  τ/δ_along on a fixed checkpoint, $0).
- **Consumers:** DSL comb `Lever`; τ_end floor rule; council §15.

### 9.2 `junction_young_angle_sigma_fit_v1`
- **Law:** The frozen scorer's triple-junction angles define, via Young's law, a pairwise
  surface-tension matrix σ_ij; a perimeter/length regularizer with σ_ij ≠ all-ones (equivalently
  an Imbert–Monneau flux-limiter choice) is the well-posed junction condition matching the
  target partition; the all-ones default imposes Herring 120° and fights the target at
  junctions.
- **Cited theory:** Imbert–Monneau flux-limited solutions (Ann. ENS 2017); Baldo/Sternberg
  multiphase Γ-limit + Herring conditions.
- **Anchors HELD:** triple-junction DOF WEAK (4-lens probe); junctions −14% CE→tau then stall.
- **Anchors OWED:** (i) the $0 junction-angle histogram → fitted σ_ij (must differ from
  all-ones by a measurable margin); (ii) an A/B (σ_ij-weighted length vs uniform) with junction-
  local d_seg attribution.
- **Consumers:** length-term `Lever` σ_ij argument; councils' treatment-arm list.

### 9.3 `filtered_viscosity_bs_monotone_envelope_v1` (smallest; may fold into #320's equation)
- **Law:** The INR+SGD discretization is non-monotone (BS-class no-guarantee); a viscosity term
  applied selectively on the backward-heat indicator set {c_a<0} is a filtered scheme
  (Froese–Oberman): monotone-envelope convergence where singular, un-damped accuracy where
  smooth; predicted ≥ fixed-ε on the d_seg-drift axis at equal stability.
- **Anchors HELD:** ep110 checkerboard instability; #320 byte-identity + n6 parity.
- **Anchors OWED:** n600 A/B floor-ε vs ca-band/per-pixel-ε (stability + d_seg-drift).
- **Consumers:** `EikonalViscoStabGauge` (extend, don't fork).

---

## 10. Sources (external)

- Barles & Souganidis 1991, *Convergence of approximation schemes for fully nonlinear second
  order equations*, Asymptotic Analysis 4:271–283 ([pdf mirror](https://benjaminmoll.com/wp-content/uploads/2021/04/barles-souganidis.pdf); [journal](https://journals.sagepub.com/doi/10.3233/ASY-1991-4305))
- Froese & Oberman, *Convergent filtered schemes for the Monge–Ampère PDE*, [arXiv:1204.5798](https://arxiv.org/abs/1204.5798); Bokanowski–Falcone–Sahu, *High-order filtered schemes for time-dependent second order HJB equations*, [arXiv:1611.04939](https://arxiv.org/abs/1611.04939)
- Calder, *Viscosity solutions — finite difference schemes* (lecture notes, [UMN](https://www-users.cse.umn.edu/~jwcalder/8590F18/lecture_numerics.pdf))
- Davini–Fathi–Iturriaga–Zavidovique, *Convergence of the solutions of the discounted
  Hamilton–Jacobi equation*, Inventiones 206 (2016) 29–55 ([Springer](https://link.springer.com/article/10.1007/s00222-016-0648-6)); discrete case ([arXiv:1607.08295](https://arxiv.org/pdf/1607.08295))
- Gao & Liu, *A selection principle for weak KAM solutions via Freidlin–Wentzell large deviation
  principle of invariant measures*, [arXiv:2208.11860](https://arxiv.org/html/2208.11860)
- Lions–Papanicolaou–Varadhan homogenization lineage: survey [World Scientific](https://www.worldscientific.com/doi/10.1142/S0218202508002978); Evans adjoint methods [arXiv:0904.3094](https://arxiv.org/pdf/0904.3094); variational effective Hamiltonian [SIAM](https://epubs.siam.org/doi/10.1137/S0363012902417620)
- Braides programme: *Anisotropic surface tensions for phase transitions in periodic media*
  [arXiv:2010.05849](https://arxiv.org/pdf/2010.05849); Γ-convergence of periodic energies to local anisotropic perimeter [arXiv:2601.08677](https://arxiv.org/pdf/2601.08677)
- Pinning/zero mobility: [arXiv:2108.00558](https://arxiv.org/pdf/2108.00558) (surface-tension gradient discontinuities + zero mobility, Allen–Cahn/curvature flows in periodic media); Dirr–Yip pinning results (see refs therein); quantitative homogenization through obstacles [arXiv:2603.12179](https://arxiv.org/pdf/2603.12179)
- Weak KAM/Lax–Oleinik rates: Fathi, *Weak KAM theorem in Lagrangian dynamics*; rate O(1/t)
  examples [arXiv:1109.3327](https://arxiv.org/pdf/1109.3327); Zavidovique lecture notes [arXiv:2308.06356](https://arxiv.org/pdf/2308.06356)
- Hajek 1988, *Cooling schedules for optimal annealing*, Math. OR ([record](https://experts.illinois.edu/en/publications/cooling-schedules-for-optimal-annealing/))
- Imbert & Monneau, *Flux-limited solutions for quasi-convex Hamilton–Jacobi equations on
  networks*, Ann. Sci. ENS 2017 ([arXiv:1306.2428](https://arxiv.org/pdf/1306.2428)); multidimensional [arXiv:1607.03996](https://arxiv.org/pdf/1607.03996)
- PINN/eikonal viscosity failures (scheme-level sanity): Neural Eikonal Solver [arXiv:2205.07989](https://arxiv.org/pdf/2205.07989); PINNeik [arXiv:2007.08330](https://arxiv.org/pdf/2007.08330)

**Triality:** DAG = FEED-07s (appended). DSL = no new flags invented here; the three named lever
surfaces (comb `Lever`, ca-band gauge extension, σ_ij length-matrix) are SPEC'd for DSL-first
landings. Equations = §9 specs only (registration gated on the named owed anchors). Pointer
contest-CPU **0.19110 UNMOVED** — everything above is means.
