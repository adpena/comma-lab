# Deep-math lens — CHAPTER 4 of "Amortizing the Argmax": PHASE FIELD / GEOMETRIC MEASURE THEORY / LEVEL-SET / Γ-CONVERGENCE / EIKONAL / MEAN-CURVATURE FLOW

- **UTC:** 20260704
- **Task:** #284 (Ch.4, the variational/constructive realization of Ch.3's hard↔soft duality).
- **Authority:** `[macOS advisory / pure-math review]` — `score_claim=false`, `promotable=false`,
  `ready_for_exact_eval_dispatch=false`. **$0**, no training run, no GPU, no paid dispatch. #205 read-only.
  **Pointer UNMOVED:** contest-CPU **0.19109982** / contest-CUDA **0.20533003**
  (`.omx/state/canonical_frontier_pointer.json`). This is a MEANS (a formalization + a ranked lever list);
  only a byte-closed exact-eval row moves the pointer.
- **Grounding (READ, exact):** `src/tac/boundary_math/lever_b_levelset_generator.py`
  (`signed_distance_fields`, `eikonal_penalty`, `boundary_length_penalty`, softmax(φ/τ) render at L664/700);
  `experiments/train_levelset_witness_realized_through_R_mlx.py` (`_softmax_temp_for_epoch`, argparse:
  `--softmax-temp-start 1.0 --softmax-temp-end 0.05 --tau-anneal-shape {cosine,geometric,cosine_hold}`,
  `--eikonal-weight 0.01 --length-weight 0.001 --hosc-beta 4.0 --hosc-beta-end`);
  `train_witness_realized_through_R_mlx.py` (`_live_margin_weight`: margin `m=srt[...,-1]-srt[...,-2]`).
- **Sisters (build-on, do-not-re-derive):** `[[unified-variational-levelset-flow-everything-is-facets]]` ·
  `[[project_gr_unified_action_full_witness_architecture]]` · `sdf_levelset_dynamical_topology_opt_research_20260702`
  (the L1-L9 sweep + the $0 MBO curvature-flow RD probe) · `eikonal_sdf_dseg_recovery_test_20260629`
  (the structured-SDF R-survival + break-even) · DAG FEED-ew/ey (topology review + β-anneal lever) ·
  `[[theta-star-witness-lever-stack-and-variational-levelset-frame]]`.

> **Ch.3 recap (as stated by the task; no Ch.3 memo exists on disk — honored as a conceptual anchor, NOT
> a fabricated citation):** the witness `softmax(φ/τ)` is the SOFT (smoothed) `argmax`; `τ·logsumexp(φ/τ) →
> max(φ)` as `τ→0` is the **Maslov / tropical dequantization** (`τ = ε = ħ` — the Planck constant of
> idempotent/(max,+)-analysis); the hard↔soft duality is the Legendre/Fenchel pairing between `max` and its
> entropic smoothing. **Ch.4 is the SPATIAL, variational, constructive realization of that pointwise
> algebraic limit.**

---

## §0. THESIS + THE ONE CLAIM

Our witness is a diffuse-interface **multi-phase field**. `soft(x) = softmax(φ(x)/τ) ∈ Δ⁴` (the K=5 probability
simplex) is a smooth phase field that, as `τ→0`, snaps to a pure phase `e_{argmax φ(x)}` almost everywhere,
leaving a thin **diffuse interface** of width `O(τ)` on the codim-1 class boundary. The `τ`-anneal
(`1.0 → 0.05`) is the **Modica-Mortola ε-annealing**; its `τ→0` **Γ-limit** is the **weighted perimeter of the
argmax partition** (Baldo/Sternberg multi-phase theorem); the **length** term (Chan-Vese, weight 0.001) is that
Γ-limit perimeter functional written directly via the coarea mollifier; the **eikonal** term (`(|∇φ|−1)²`,
weight 0.01) is the Hamilton-Jacobi constraint that keeps `φ` a true signed distance so the interface width
and the perimeter are in consistent pixel units; the **training gradient flow** of the perimeter is, in the
sharp-interface limit, **motion by mean curvature** (Allen-Cahn → MCF), which is precisely why thin Lane
structures **erase** (MCF annihilates high-curvature features first) — the measured binding residual.

**THE ONE CLAIM (Ch.4 → Ch.3 duality bridge — the phase-field = Γ-limit = mirror-descent triangle):**

> The single scalar `τ` is the SAME object at two scales. **Pointwise (Ch.3):** `τ·logsumexp(φ/τ) → max(φ)`
> is the Maslov/tropical dequantization `ħ→0`. **Spatial (Ch.4):** the entropic phase-field energy
> `E_τ[softmax(φ/τ)] Γ-converges` (as `τ→0`) to the **perimeter** of the `argmax`-partition
> (Modica-Mortola/Baldo). These are ONE dequantization acting at two scales — the semiring limit of the
> *pointwise* soft-max and the Γ-limit of the *spatial* soft-partition energy coincide. The level-set
> **training** is **mirror-descent** (softmax = the negative-entropy Bregman mirror map) run as a
> **continuation / graduated-non-convexity** homotopy along that dequantization (`τ: 1 → 0`), and
> **Γ-convergence is exactly the theorem that this continuation converges to the hard combinatorial minimizer**
> (the shortest-perimeter partition consistent with the data). Hard↔soft duality (Ch.3, algebraic) is thus
> *realized* (Ch.4, variational): the same `τ=ε=ħ` that Legendre-dualizes `max`↔`logsumexp` pointwise is the
> interface width whose vanishing Γ-converges the soft partition to the hard one. **`τ = ε = ħ = ` the
> diffuse-interface width = the mirror-descent temperature = the tropical Planck constant.**

---

## §1. THE FORMALIZATION — softmax-of-SDF IS an entropic multi-phase field

### 1.1 The order parameter and the entropic well
For each pixel the softmax is the **Gibbs / entropic-regularized argmax**:

```
soft(x) = softmax(φ(x)/τ) = argmin_{p ∈ Δ⁴} [ −⟨p, φ(x)⟩ + τ·⟨p, log p⟩ ].
```

The linear term `−⟨p,φ⟩` is the data drive (assign to the largest logit); the entropic barrier `τ·H(p)`
(`H`=Shannon entropy, here written `⟨p,log p⟩ = −H`) is the **well potential** that keeps `p` in the simplex
interior. As `τ→0` the barrier vanishes and the minimizer is the LP vertex `e_{argmax φ}` (a pure phase). This
is the **exact analog of the Modica-Mortola double/multi-well `W`**: `W`'s wells are the pure phases `{e_k}`;
the entropic barrier's minimizers as `τ→0` are those same vertices. The difference from textbook MM is that
our "well" is the *entropy of a simplex-valued field*, not a polynomial `W(u)` of a free `ℝ^K` field — this is
the **Ginzburg-Landau ↔ entropic-optimal-transport correspondence** (the softmax is the entropic-OT / Sinkhorn
Gibbs kernel; the `τ→0` limit is the un-regularized assignment). Standard in entropic OT; here applied
spatially, pixel-wise, over the SegNet-argmax target.

### 1.2 The three regularizers ARE the three pieces of the level-set/phase-field apparatus
The witness carries exactly the classical level-set variational triple (MEASURED forms, from the READ code):

| term | code form | classical object | role |
|---|---|---|---|
| **softmax(φ/τ) anneal** | `mx.softmax(phi/softmax_temp)`, τ: 1.0→0.05 | **Allen-Cahn / MM diffuse interface**, width ε=τ | the *soft→hard* continuation (the ε-anneal) |
| **eikonal** | `mean((|∇φ|−1)²)` (num.), on margin `m=φ₁−φ₂` (MLX) | **Hamilton-Jacobi `|∇φ|=1`** viscosity soln = distance | metric normalization; **decouples width from slope** |
| **length** | `mean(δ_ε(φ)·|∇φ|)`, `δ_ε=(ε/π)/(ε²+φ²)`, ε=1px | **Chan-Vese / coarea perimeter** `∫δ(φ)|∇φ|=Per{φ=0}` | the **Γ-limit functional itself** (perimeter=rate) |

Two distinct-but-Γ-equivalent routes to the *same* sharp-interface object are BOTH present:
- the **τ-anneal** is the **diffuse-interface (Allen-Cahn/MM) side** — a family whose Γ-limit is the perimeter;
- the **length term** is the **already-sharp mollified perimeter (Chan-Vese coarea) side** — the Γ-limit
  target written directly with a 1px Cauchy mollifier `δ_ε`.
The **eikonal** is the reinitialization that keeps both in the *same* pixel units (so "width τ" and "mollifier
ε=1px" mean the same thing). This is a coherent, non-accidental level-set design — the witness was built with
the right pieces; Ch.4's job is to show they are the classical apparatus and to **calibrate their weights and
the τ-schedule by the Γ-theory.**

### 1.3 Why the MLX eikonal-on-the-margin is MORE correct than per-field
Near an `i↔j` boundary the top-2 logits dominate: `soft_i/soft_j = exp((φ_i−φ_j)/τ) = exp(m'/τ)` where `m'` is
the *signed* gap (`m=|m'|` is the code's `top1−top2`). The two-phase reduction's interface half-width in pixels
is therefore `τ / (2·|∇m|)` (from `soft_i≈soft_j` at `m'≈0`, linearized). So **`|∇m|=1` (eikonal on the margin)
makes the interface width EXACTLY `τ/2`** — the single well-defined width parameter. Per-field eikonal
(`|∇φ_k|=1`) over-constrains the medial-axis interior (irrelevant to the argmax) and fights disconnected
Movable silhouettes; **margin-eikonal is the phase-field-correct choice** (the MLX trainer already does this —
confirm-and-keep).

---

## §2. THE Γ-LIMIT THEOREMS (PROVEN — real theorems, cited, not re-derived)

1. **Scalar Modica-Mortola (1977) + Modica (1987).** `E_ε[u] = ∫ ε|∇u|² + (1/ε)W(u)` with double-well `W`
   (wells 0,1) **Γ-converges** to `c_W·Per(∂{u=1})`, `c_W = ∫₀¹√(2W)` (the surface tension = wall energy).
   Minimizers develop the **optimal profile** `u_ε(x) ≈ q(d(x)/ε)`, `q` the 1-D heteroclinic, `d`=signed
   distance. **This is the theorem that a diffuse interface of width ε costs (surface tension)×(perimeter).**
2. **Multi-phase Baldo (1990) / Sternberg (1988) / Fonseca-Tartar.** The vector functional with a `K`-well `W`
   (wells `{e_1..e_K}`) Γ-converges to the **weighted perimeter of the partition**
   `E₀[{Ω_k}] = ½Σ_{i<j} σ_ij·H^{n−1}(∂Ω_i∩∂Ω_j)`, with `σ_ij = 2·d_W(e_i,e_j)` a **geodesic distance in the
   degenerate metric `√W`** (a Finsler/Riemannian well-metric). **Triple junctions obey Herring's angle
   condition** `σ_ij/sin θ_k` balance (⇒ the natural triple-point handling our K=5 argmax needs at
   Road/Lane/Undrivable/Movable/MyCar meets). **This is exactly the K=5 case we are in.**
3. **Coarea (exact, not a limit).** `∫_Ω δ(φ)|∇φ| dx = H^{n−1}({φ=0})`. Our length term with `δ_ε`, ε→0, is
   the coarea perimeter — the Γ-limit target written directly (Chan-Vese level-set formulation).
4. **Allen-Cahn → motion by mean curvature (Bronsard-Kohn 1991; Ilmanen 1993; Evans-Soner-Souganidis).** The
   `L²` gradient flow of the MM energy has, as `ε→0`, interface velocity `V = −(constant)·κ` (mean curvature).
   **The perimeter-gradient training flow IS curvature flow of the class boundaries.**
5. **Eikonal viscosity (Crandall-Lions; Sethian; Osher).** `|∇φ|=1` is a stationary Hamilton-Jacobi equation
   whose **unique viscosity solution is the distance function**. The eikonal penalty enforces `φ` = the HJ
   viscosity solution = a true SDF. (And the HJ vanishing-viscosity limit is the *same* `ħ→0` structure as
   Ch.3's Maslov dequantization — the level-set PDE and the tropical limit are the one semiclassical limit.)
6. **Γ-convergence of gradient flows (Sandier-Serfaty 2004).** If the ε-family's gradient flows are slow
   relative to the energy landscape, the flow of `E_ε` converges to the flow of the Γ-limit `E₀`. **This is
   the theorem that a correctly-paced ε-anneal tracks the sharp minimizer** (the basis of the schedule
   derivation in §4).

**Honesty:** (1)–(6) are established mathematics. What is *ours* (and NOT a proven theorem for our exact
setup) is the identification "softmax-entropic-well ≈ MM multi-well" and the claim that our *learned-SDF,
realized-through-R* family Γ-converges with these rates. See the §7 ledger.

---

## §3. THE τ = ε = ħ TWO-SCALE DEQUANTIZATION (bridge to Ch.3) + MIRROR DESCENT

**Pointwise (Ch.3):** `τ·logsumexp(φ/τ) → max(φ)`; `softmax(φ/τ) → onehot(argmax φ)`. Legendre pair:
`logsumexp = (entropy)*` (convex conjugate of the negative entropy on the simplex). Maslov dequantization:
`ħ=τ→0` collapses `(+,×)`-analysis to the `(max,+)` tropical semiring.

**Spatial (Ch.4):** the *same* `τ` is the diffuse-interface width `ε`. `E_τ[softmax(φ/τ)]` Γ-converges to the
partition perimeter. **The pointwise semiring limit and the spatial Γ-limit are the SAME `τ→0`.** This is the
constructive realization: Ch.3 says `soft = hard` in the algebraic/Legendre sense at `ħ→0`; Ch.4 says the
*field* `soft-partition = hard-partition` in the variational/Γ sense at `ε→0`, and both are the one dequantization.

**Mirror descent (the dynamics that closes the triangle).** `softmax` is the **mirror map** of the
negative-entropy Bregman potential `Φ(p)=Σp log p` on the simplex: `p = ∇Φ*(φ/τ)`. Training the witness = a
**mirror flow** in `p` with temperature `τ`. Annealing `τ: 1→0` is **graduated non-convexity (GNC) /
homotopy-continuation**: at `τ=1` the objective is smooth/convex-ish (gradients flow, no RGB-Gibbs — the code's
"soft start"); at `τ→0` it is the hard combinatorial argmax-LP (perimeter minimization). **Γ-convergence of the
family + Sandier-Serfaty flow-convergence = the guarantee that the mirror-descent continuation lands in the
sharp minimizer's basin rather than a spurious one.** So the triangle is:

```
   PHASE FIELD  (diffuse interface, width ε=τ)                       [Ch.4, geometry]
        │  Γ-limit (ε→0, Modica-Mortola/Baldo)
        ▼
   PERIMETER of the argmax partition  (= the hard combinatorial target = rate/MDL)   [Ch.3, the "max"/tropical vertex]
        ▲
        │  mirror-descent continuation (τ:1→0, GNC), tracked by Sandier-Serfaty
   MIRROR MAP  (softmax = ∇(neg-entropy)*, temperature τ)            [the training dynamics]
```

The three vertices are the **representation** (phase field), the **objective** (perimeter/MDL = rate), and the
**algorithm** (mirror-descent anneal) — three views of one `τ`, cyclically related. (This is the witness-level
echo of the campaign-level DAG↔DSL↔equations triality.)

---

## §4. THE Γ-OPTIMAL τ-SCHEDULE — the headline actionable derivation

**The measured facts (READ):** `τ` anneals `softmax_temp_start=1.0 → softmax_temp_end=0.05`; shapes are
`cosine` (DEFAULT), `geometric` (`τ = start·(end/start)^prog`, log-spaced), `cosine_hold`. The code's own
docstring notes `geometric` "spends MORE epochs at small τ (slows the near-τ→0 continuation step that drives
the measured late-τ d_seg volatility)."

**Derivation 1 — the shape should be GEOMETRIC (log-linear), not cosine.** The interface width is `ε=τ`; the
relevant structure is a **scale-space / renormalization-group** flow (each octave of interface width is one
"scale"). Sandier-Serfaty requires the continuation to be *slow relative to the energy landscape*, and the
landscape changes by CONSTANT amounts per multiplicative change in `ε` (the surface tension and the number of
resolvable interface configurations are functions of `log ε`, not `ε`). Therefore the **Γ-optimal continuation
spends equal epochs per OCTAVE of `ε` = geometric decay** — exactly `--tau-anneal-shape geometric`. The default
**cosine is scale-space-wrong**: `dτ/dprog ∝ sin(π·prog)` is slowest at the endpoints and *fastest in the
middle*, so it **rushes through the mid-`τ` regime where the interface crosses the pixel/Nyquist scale**
(precisely where careful tracking matters) and lingers at `τ≈1` (over-smooth, low-information) and `τ≈0.05`
(already past the grid). GNC/graduated-non-convexity practice (Blake-Zisserman; Mobahi-Fleet homotopy
continuation) independently prescribes geometric smoothing schedules. **⇒ switch the default to `geometric`.**

**Derivation 2 — the FLOOR `τ_end` should be the RESOLUTION scale (`≈1`, NOT 0.05), CONTINGENT on a strong
eikonal.** With a margin-eikonal enforcing `|∇m|≈1`, the diffuse-interface half-width is `τ/2` px. At
`τ_end=0.05` that is **0.025 px** — ~40× below the pixel grid AND ~40× below the ~1px annulus / R-blur / the
`δ_ε` mollifier (ε=1px). **MM/Γ theory (and every mesh-based phase-field solver) says the optimal `ε` floor is
the mesh size `h`: sharpening `ε` below the grid buys NO Γ-benefit and only adds discretization/aliasing
error.** For us "below the grid" is worse than inert: the sub-pixel step, pushed through the uint8 knife-edge
and bilinear `R`, is exactly the **Gibbs/aliasing "late-τ volatility"** the code observes. And it is *wasted*:
the d_seg VERDICT reads the **hard argmax of the SDF** (not `soft`); the soft render only needs to be a clean
partition paint **at the resolution scale**. **⇒ float `τ_end` up to `≈1` (≈½px interface half-width), matching
the annulus/R-blur/δ_ε scale.** The half-width `τ/2` identity is *only* valid if `|∇m|≈1`, which is why this
lever is **contingent on Lever 2 (eikonal)** — without it, `τ` and `|∇m|` are entangled and the floor is
uncontrolled.

**Derivation 3 — co-anneal the hosc β with τ (two continuations, one Γ-limit).** `hosc = tanh(β·sin(ωu))`;
`β→∞` is the square wave = the sharp phase-indicator (the "optimal profile" `q` going to a step). `β` sets the
transition-profile STEEPNESS; `τ` sets the interface WIDTH. In MM both →sharp is the Γ-limit; the optimal
profile is reached by walking BOTH. The code has `--hosc-beta 4.0 --hosc-beta-end` (anneal 4→8, DAG FEED-ey) —
**co-schedule β↑ with τ↓ geometrically** so the profile steepens as the width narrows (a matched pair), and
freeze both together at the Muon finisher (the code already freezes τ at muon-start — freeze β likewise).

**Net Lever-1 config (all $0 A/B, CONVERGENT with the code's own late-τ note):**
`--tau-anneal-shape geometric --softmax-temp-start 1.0 --softmax-temp-end ~1.0` (sweep 0.5–1.5) `+`
`--eikonal-weight 0.05` (Lever 2, required) `+` co-annealed `--hosc-beta 4 --hosc-beta-end 8`.

---

## §5. ARE THE REGULARIZER WEIGHTS Γ-OPTIMAL? (eikonal 0.01, length 0.001)

### 5.1 Eikonal (0.01 → 0.05–0.1): NOT just a topology bias — REQUIRED for the τ-schedule to be Γ-meaningful
The eikonal `|∇m|=1` is what makes the interface half-width equal `τ/2` (§1.3). At weight 0.01 the DAG measured
it is "only softly enforced while the architecture is oscillatory" — i.e. the network is free to make `|∇m|`
large near boundaries (sharpening for free) so the *effective* width is `τ/(2|∇m|)`, **uncontrolled and
schedule-decoupled**. **The phase-field frame gives the eikonal a NEW, stronger justification than "topology
prior": it is the metric normalization without which `τ` is not an interface width at all, the perimeter is not
in pixel units, and Lever-1's floor is meaningless.** ⇒ raise to **0.05** (converges with the DAG's independent
FEED-ew/ey recommendation), and **anneal it UP late** (when the interface localizes and `|∇m|=1` matters most).

### 5.2 Length (0.001): the RIGHT functional, the WRONG (isotropic) surface tension — the class-pair fix
The length term IS the Γ-limit perimeter (§1.2, §2.3) — a correct, principled object. But at a **single scalar
weight it applies the SAME surface tension to every class pair**, whereas Baldo's multi-phase Γ-limit is
`½Σσ_ij·Per(∂Ω_i∩∂Ω_j)` with a **class-pair-dependent `σ_ij`**. The MEASURED erasure crux (sister MBO probe,
`sdf_levelset..._20260702` §3) says this uniform tension is actively harmful:
- smoothing **{Road, Undrivable, MyCar} mutual** boundaries costs ≈0 d_seg (stable, margin ~5.6, high
  persistence) and *saves* perimeter → **want HIGH `σ_ij`** there;
- smoothing **Lane↔Road & Movable** ERASES the fragile tail (95.7% of the MBO d_seg cost is Lane; Lane
  retention collapses 1.00→0.13 under curvature flow) → **want `σ_ij ≈ 0`** there.
A uniform length weight prices Lane perimeter (thin, huge perimeter-to-area) the same as Road perimeter — the
worst possible allocation. **⇒ replace the scalar `--length-weight` with a class-pair `σ_ij` (5×5 upper-tri,
10 values): high on {Road,Undriv,MyCar}², ~0 on Lane/Movable pairs.** This is the multi-phase-MM-correct form
and directly attacks the binding residual (CONVERGENT with the sister `mbo_decode_regularizer` DSL lever and
the shipped `lane_thin_weight_map` data counterweight).

### 5.3 Is the length weight "pricing rate"? — FALSE FRIEND, be honest
Tempting to call the perimeter penalty the "rate term" (perimeter = curvelet N-term = bytes). **It is NOT the
archive-byte count.** The rendered perimeter is regenerated FREE at decode (rule 118: the SDF rasterizer is
generic code); the RATE term is on the LEARNED weights/codes, not the boundary length. The Γ-limit perimeter is
the **information content / MDL description length of the boundary** — an *intrinsic-complexity* quantity that
**bounds** how many learned bytes the residual needs, not the counted bytes themselves. So the length term is a
**regularizer** (short-perimeter prior → suppresses salt-and-pepper flip islands; and, class-pair-weighted, the
erasure-aware smoother), whose connection to rate is **indirect (a bound), not a Lagrange price on archive
bytes.** Calibrate it as a *regularizer* (small, anisotropic), not as a rate multiplier.

---

## §6. MEAN-CURVATURE FLOW — WHY LANES ERASE, AND THE VOLUME-CONSTRAINT CURE

By theorem (4), the perimeter-gradient training flow is, in the sharp limit, **motion by mean curvature**:
`V = −κ`. MCF **shrinks high-curvature features first** and, for a curve, drives every convex closed component
to a round point and then extinction (Gage-Hamilton-Grayson). **The thin Lane dashes and small Movable blobs
are exactly the high-curvature / high-perimeter-to-area features MCF annihilates first** — a *mathematically
inevitable* consequence of any perimeter-penalizing flow, NOT a training artifact. The sister $0 MBO probe
measured precisely this: curvature flow's flips are 85–98% Lane at every scale; Lane retention 1.00→0.13.

**The cure is classical: an area / volume constraint (Lagrange multiplier on class area).** Volume-constrained
(auction) MBO / Esedoglu-Otto threshold dynamics **prevents minority-class annihilation by construction**. In
our trainer this maps to:
- a **per-class area-preservation penalty** `λ_k·(area_k − target_k)²` (the auction-MBO analog), OR
- (equivalently, already partly present) the **`lane_thin_weight_map`** — a data-fidelity up-weight on thin
  lane pixels (52.7% of GT-lane components are wholesale-missed; <5px are 93% missed) that *counters* the MCF
  shrinkage with restored seg-loss pressure exactly on the dashes.
The phase-field frame **explains why the thin-weight is needed** (perimeter flow ⇒ MCF ⇒ thin-erasure) and
suggests the **area-constraint as its principled sibling** (stops the erasure at the geometry level, not just
the data level). The **curvature↔margin correspondence** (measured: MBO-flipped-pixel margin monotone
0.13→0.55 vs kept 5.6; Fisher-curvature↔(−margin) Pearson **0.978**, sister unified-levelset) makes curvature a
**byte-free, scorer-free fragility prior** for finest-scale capacity allocation — the level-set reading of the
same annulus the `_live_margin_weight` bit-allocator already exploits (~89% of d_seg in the bottom-5%-margin
2.26% annulus).

---

## §7. HONESTY LEDGER (proven / conjectured / false-friend)

**PROVEN (real theorems, cited §2):** MM scalar Γ-limit (Modica-Mortola/Modica); multi-phase Γ-limit to
weighted perimeter with Herring junctions (Baldo/Sternberg/Fonseca-Tartar); coarea identity for the length
term; Allen-Cahn→MCF (Bronsard-Kohn/Ilmanen); eikonal viscosity solution = distance (Crandall-Lions/Sethian);
Γ-convergence of gradient flows (Sandier-Serfaty). GNC/homotopy geometric-schedule practice (Blake-Zisserman;
Mobahi-Fleet).

**CONJECTURED (well-motivated, NOT proven for our exact setup):**
- *"softmax-entropic-well ≡ Modica-Mortola multi-well."* The entropic barrier `τH(p)` and a polynomial `K`-well
  `W` share the `τ→0` vertex-collapse and the diffuse-interface structure (standard entropic-OT ↔
  Ginzburg-Landau analogy), but the exact `σ_ij` surface tensions and the Γ-limit of our *learned-SDF,
  realized-through-R* family are not a theorem we have proven. The correspondence is a lens with real
  predictive content (§4/§5/§6), not a closed proof.
- *the `τ/2` interface-width identity* holds only under `|∇m|≈1` (needs the strong eikonal); it is a
  linearization, exact only in the two-phase reduction near a smooth boundary segment (breaks at triple points
  and disconnected Movable — where Herring/medial-axis corrections enter).

**FALSE FRIENDS (named and killed):**
- *"length term = rate term."* NO — perimeter regenerates free at decode; it is an MDL *bound*, not the archive
  bytes (§5.3). Do not tune it as a rate multiplier.
- *"sharper `τ` is always better; drive `τ_end→0."`* NO — MM says the optimal `ε` floors at the resolution
  scale; `τ_end=0.05` (0.025px half-width) is 40× sub-grid = pure aliasing, and wasted since the verdict reads
  the hard argmax (§4 Deriv-2).
- *"per-field eikonal."* Dominated by margin-eikonal for the argmax boundary (§1.3); per-field over-constrains
  irrelevant medial-axis interior and fights multi-component Movable.
- *"phase-field is a lens, not a knob"* (the sister-memo's L6 verdict, 2026-07-02). Ch.4 OVERTURNS the
  "lens-only" framing at the IMPLEMENTATION level: the τ-schedule shape+floor, the eikonal weight's
  *necessity*, and the class-pair surface tension are **concrete $0 config levers**, not just a validating
  analogy. (The sister memo's own `geometric`-shape note and MBO class-pair finding are the receipts.)

---

## §8. ENGINEERING NEXUS — ranked levers with honest EV (all $0 config A/Bs; net-S #205-gated)

| # | Lever | Change | EV / honest caveat |
|---|---|---|---|
| **1** | **Γ-optimal τ-schedule** | `--tau-anneal-shape geometric` + float `--softmax-temp-end` 0.05→**~1.0** (sweep 0.5–1.5) | **HIGH, cheapest.** Scale-space/GNC-correct + kills the code's own measured "late-τ volatility" + stops sub-grid aliasing. **Contingent on Lever 2.** Verdict reads hard argmax → sub-pixel soft is wasted+harmful through R. |
| **2** | **Raise eikonal** | `--eikonal-weight 0.01→0.05` (anneal UP late) | **HIGH.** *Required* to make Lever-1's `τ/2` width identity hold (decouples width from slope). New MM justification beyond "topology bias." CONVERGENT with DAG FEED-ew/ey independent rec. |
| **3** | **Class-pair surface tension** | scalar `--length-weight` → 10-value `σ_ij`: high {Road,Undriv,MyCar}², ~0 Lane/Movable | **MED-HIGH.** Baldo-correct multi-phase form; directly attacks the erasure crux (95.7% of smoothing d_seg cost is Lane). Needs the loss to expose per-pair boundaries. CONVERGENT with sister `mbo_decode_regularizer`. |
| **4** | **Area/volume constraint** | per-class `λ_k·(area_k−target_k)²` (auction-MBO analog) | **MED.** The theorem-identified cure for MCF minority-erasure; principled sibling of the shipped `lane_thin_weight_map`. Formalizes why the thin-weight works. |
| **5** | **Curvature-as-margin capacity prior** | allocate finest-scale capacity by boundary curvature (byte-free) | **LOW-MED** (a prior, not a direct mover). $0, principled (curvature↔margin 0.978); feeds the θ* finest-scale allocator + reinforces the `_live_margin_weight` annulus focus. |
| **6** | **Co-anneal β with τ** | `--hosc-beta 4 --hosc-beta-end 8` geometric, frozen at Muon finisher | **LOW-MED.** Matched profile-steepness/width pair; the second continuation to the same Γ-limit. Already partly in FEED-ey; Ch.4 gives the profile-`q` rationale. |

**Sequencing:** Levers 2+1 are one coupled $0 config A/B (eikonal enables the τ-floor) — do them together first.
Levers 3+4 are the erasure-crux attack (loss-side, slightly more work). Levers 5+6 are priors/refinements.
NONE moves the pointer until byte-closed through `tools/levelset_byte_close_and_eval.py` (#202) on a converged
n600 decoder — **#205-gated, means≠ends.**

---

## §9. WIRE-IN (Catalog #125) + THE CLAIM TO CH.3

- **Hook #1 sensitivity-map:** ACTIVE — class-pair `σ_ij` + curvature prior route capacity to the fragile
  boundary (Lane/annulus).
- **Hook #2 Pareto:** ACTIVE — the length/perimeter term is the distortion↔rate(MDL-bound) constraint;
  τ-floor is the resolution constraint.
- **Hook #3 bit-allocator:** ACTIVE — curvature-as-margin is a byte-free finest-scale allocation rule
  (sibling of `_live_margin_weight`).
- **Hook #4 cathedral autopilot:** N/A (advisory formalization).
- **Hook #5 continual-learning:** ACTIVE — CONFIRMS+SHARPENS the sister 2026-07-02 phase-field lens from
  "lens-only" to "4 concrete config levers"; candidate canonical equations below.
- **Hook #6 probe-disambiguator:** N/A.

**Candidate canonical equations (`tac.canonical_equations`):**
- `tau_equals_epsilon_equals_hbar_two_scale_dequantization_v1`: the softmax temperature is simultaneously the
  Maslov/tropical `ħ` (pointwise), the Modica-Mortola diffuse-interface width `ε` (spatial), and the
  mirror-descent temperature; `τ→0` is one dequantization at two scales, Γ-converging the soft partition to the
  argmax-partition perimeter. Producers: this ledger. Consumers: the τ-schedule config; Ch.3 duality.
- `gamma_optimal_tau_schedule_geometric_floored_at_resolution_v1`: the Γ/scale-space-optimal anneal is
  geometric (equal epochs per octave of `ε`) floored at `τ_end ≈ 2h` (interface half-width ≈ ½px), contingent
  on `|∇m|≈1`. Producers: this ledger + the code's late-τ-volatility note. Consumers: `--tau-anneal-shape`,
  `--softmax-temp-end`, `--eikonal-weight`.
- `multiphase_surface_tension_length_term_v1`: the Chan-Vese length term should carry a class-pair `σ_ij`
  (Baldo weighted perimeter): high on {Road,Undriv,MyCar}², ~0 on Lane/Movable — else uniform perimeter flow
  is MCF that erases the fragile tail. Sister of `curvature_ranks_segnet_margin_v1` (2026-07-02).

**THE ONE CLAIM contributed to Ch.3's hard↔soft duality (restated for the book):**
Chapter 3's hard↔soft duality — `max` ↔ `logsumexp`, Legendre-paired, dequantized as `ħ=τ→0` (Maslov/tropical)
— is **constructively realized** in Chapter 4 as a **phase-field Γ-convergence**: the witness `softmax(φ/τ)` is
an entropic multi-phase field whose `τ→0` **spatial** Γ-limit (Modica-Mortola/Baldo) is the **perimeter of the
argmax partition** — the very hard combinatorial object Ch.3's pointwise `max` selects. The **same `τ`** is the
tropical Planck constant `ħ`, the diffuse-interface width `ε`, and the mirror-descent temperature; the level-set
training is **mirror-descent continuation (GNC)** along that dequantization, and **Γ-convergence is the theorem
that the soft continuation lands on the hard minimizer.** `τ = ε = ħ`: hard↔soft duality is one dequantization,
and the phase-field is how it is *built*.

**Pointer UNMOVED (0.19110). MEANS — a formalization + 6 ranked $0 config levers; only a byte-closed
converged-n600 exact row moves the score.**
