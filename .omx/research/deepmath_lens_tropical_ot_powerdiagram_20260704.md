# Deep-math lens: TROPICAL / MAX-PLUS · POWER (LAGUERRE) DIAGRAMS · SEMI-DISCRETE OT · POLYHEDRAL GEOMETRY

**Task #284 · 2026-07-04 · $0 research (online + OSS + papers; no GPU, no paid, no heavy).**
**Status: MEANS (deep-math capture). A finding moves no pointer — only a byte-closed n600 exact
row (`upstream/evaluate.py`, CPU/CUDA, MPS-never) does. Pointer 0.19110 UNMOVED. #205 sacred
(read-only; untouched by this pass).** Every candidate lever below is a MEANS with a named $0
pre-metric → n600-realized-through-R authority → #202 byte-close before ANY claim.

---

## TL;DR — 5 headline findings

1. **The thesis is a THEOREM, not an analogy — with one precise caveat.** For piecewise-affine
   (ReLU) nets, Balestriero–Baraniuk (2019) *proved* the input-space partition IS a subdivided
   **power (Laguerre) diagram**; Zhang–Naitzat–Lim (2018) and Alfarra–Bibi et al. (2022) proved the
   **decision boundary IS a tropical hypersurface**. Our SegNet is **smooth** (EfficientNet-B2, SiLU
   + BatchNorm), so its argmax partition is not a *global* power diagram but a **LOCAL / curved /
   "soft" one** — a Fisher-metric anisotropic Laguerre tessellation. Our SIMPLEX PROBE
   (`scalar_top1_top2_margin_is_exact_distance_to_flip_v1`; min diff **0.0** over 118M px) is
   *exactly* the first-order power-diagram fact: the margin `top1−top2` = the transverse distance to
   the nearest cell wall. So the framing is **mathematically exact locally and empirically
   confirmed** — but it licenses **boundary-annulus / local** algorithms, NOT global-power-diagram
   solvers.

2. **d_seg is the symmetric-difference (Hamming) mismatch between two Laguerre labelings — NOT a
   Wasserstein/OT cost.** Semi-discrete OT (Aurenhammer–Hoffmann–Aronov; Lei–Gu geometric OT) is a
   *solver* only when you get to CHOOSE the sites/weights. **We do not** — the scorer is frozen. So
   the OT lens contributes (a) the Kantorovich **dual = margin-weighting we already have** (#141
   margin-saliency), and (b) a class-cell-**mass** budget (the AHA Minkowski mass-prescription), NOT
   a new coupling or preimage solver. Honest verdict: **OT here is FRAMING + one modest lever**, not
   a solver plug-in.

3. **The tropical/polyhedral view is the RIGHT chart for the lane boundary AND it is rate-cheap on
   the COUNTED term — the one genuinely NEW engineering nexus.** The binding residual is a codim-1
   **Road|Lane FACET** (41.4% of all facets; 0.59% area; 35–40% of flips; 55× over-represented) with
   **straight dashes**; the measured root cause is an **along-tangent frequency deficit** (basis ≤8
   vs dash ~25 cyc/unit = 3.2×). A **tropical / piecewise-linear "dash comb"** (max-plus on/off
   modulation, params = phase·period·width, phase = ego-ξ forward distance) is topology-matched to
   straight dashes AND costs a **handful of counted bytes** — attacking the root cause directly.
   Candidate lever **T2**, A/B owed vs the frequency-basis fix (`--n-dir-freqs 2→4`) + the openpilot
   analytic band.

4. **#218's "power-diagram (Laguerre) weights = per-class logit offset" is CONFIRMED and
   FORMALIZED.** The tropical/OT lens adds a *principled* way to SET the offsets: the AHA/semi-
   discrete-OT weight-solve = "shift each class's wall so its cell mass matches the GT class
   frequency," which is exactly LDAM/logit-adjustment but with a derived (not heuristic) target —
   directly countering minority-class (Lane 0.59% / Movable 1.56%) collapse. Byte-free, $0, composes
   with clDice/persistence (#218 FEED-pt).

5. **The correct global object is a Fisher-metric ANISOTROPIC LAGUERRE tessellation** — which
   *unifies* the measured Fisher-curvature↔margin (Pearson **0.978**) + the ~7:1 cross-boundary
   anisotropy + the annulus-localization into ONE named geometric object, confirming the already-
   pinned annulus-localized anisotropic v2 loss. High unification value; **not a new lever.**

**Ranked candidate levers** (detail in §5): **T2** dash-comb (NEW, highest EV) ▸ **T1** AHA
mass-matched Laguerre logit-offset head ($0, byte-free, do-first-as-metric) ▸ **T3** OT class-mass
budget regularizer (modest; dominated by clDice for the tail) ▸ **T4/T5** Fisher-Laguerre framing +
max-affine head (FRAMING / unification; no direct EV).

---

## §1 — Formalization: is the SegNet argmax partition a power/Laguerre diagram?

### 1.1 The exact statement (affine forms → power diagram)

A **power (Laguerre) diagram** of sites `p_k ∈ ℝ^d` with weights `w_k` assigns `x` to
`k* = argmin_k (‖x−p_k‖² − w_k)`. Expanding the square, the `‖x‖²` term is common to all `k`, so the
comparison reduces to a comparison of **affine functions of x**:

```
‖x−p_k‖² − w_k  ≤  ‖x−p_j‖² − w_j
   ⇔   (−2p_k)·x + (‖p_k‖²−w_k)  ≤  (−2p_j)·x + (‖p_j‖²−w_j)
```

Therefore a power diagram = a partition by **min (equivalently max) of affine forms**. Aurenhammer
(1987) proved the converse: **every "affine diagram" (partition into convex polyhedral cells cut by
the upper/lower envelope of affine functions) IS a power diagram** — with the explicit dictionary
`aₖ = −2pₖ`, `bₖ = ‖pₖ‖²−wₖ`, i.e. `pₖ = −aₖ/2`, `wₖ = ‖aₖ‖²/4 − bₖ`.

Now: `argmax_k ℓ_k(x)` where `ℓ_k(x) = a_k·x + b_k` is *precisely* a max-of-affine partition ⇒
**argmax of K affine logits over x IS a power diagram with K sites.** The `max_k(a_k·x + b_k)` value
function is a **tropical polynomial** in the max-plus semiring (`⊕ = max`, `⊙ = +`); its cells are
the regions of linearity (dual to a regular subdivision of the Newton polytope `conv{a_k}` lifted by
heights `b_k`), and the **tropical hypersurface** = the locus where the max is tied by ≥2 terms =
the **decision boundary** = the **power-diagram walls**. Three names, one object:

| view | object | our name for it |
|---|---|---|
| tropical | `p(x)=max_k(a_k·x+b_k)` value fn; hypersurface = ties | the margin field `m`, its zero-set |
| polyhedral | power/Laguerre cells (max-affine partition) | `argmax_k φ_k` partition (K=5) |
| semi-discrete OT | Laguerre cells = OT map to K Diracs; `w_k` = dual potentials | per-class offsets (#218) |

### 1.2 Where our net DEVIATES from the exact statement (honest caveat)

- **ReLU / piecewise-affine nets** realize the exact statement globally. Balestriero–Baraniuk
  (2019, *"The Geometry of Deep Networks: Power Diagram Subdivision"*) proved a max-affine-spline
  (ReLU/leaky/absolute/maxpool) network's input partition is a power diagram, progressively
  subdivided layer-by-layer. Zhang–Naitzat–Lim (2018) proved the same family = tropical rational
  maps; Alfarra–Bibi–Hammoud–Gaafar–Ghanem (2022) proved the (Affine,ReLU,Affine) decision boundary
  is a subset of a tropical hypersurface (a convex hull of two zonotopes).
- **Our SegNet is SMOOTH** — EfficientNet-B2 uses SiLU/Swish + BatchNorm, which are NOT piecewise-
  affine. So the logits `ℓ_k(x)` are smooth, not globally affine ⇒ the partition is a **curved**
  power diagram. The correct model is Balestriero–Baraniuk's *"From Hard to Soft"* (2018): a smooth
  nonlinearity = a **soft / probabilistic power diagram** = an entropy-regularized (Gibbs) vector
  quantization. The **softmax** over logits is *literally* the soft (entropic) Laguerre assignment;
  the **hard argmax** we score is its zero-temperature limit. (Bonus unification: our **softmax-tau
  curriculum = entropic-OT / Sinkhorn regularization annealing** — the temperature IS the OT entropy
  weight. That re-reads our tau schedule as coarse-to-fine entropic-OT.)
- **The LOCAL structure is exact and we MEASURED it.** First-order Taylor of the smooth logits at
  any boundary point gives affine forms ⇒ locally the partition IS a power diagram, and the margin
  `top1−top2` is the transverse coordinate. The SIMPLEX PROBE
  (`scalar_top1_top2_margin_is_exact_distance_to_flip_v1`, n600 exact CPU-torch SegNet, bit-exact):
  `gap13 = top1−top3 ≥ gap12 = top1−top2` at **all 118M px (min diff 0.0)** ⇒ only the runner-up
  logit can cause the first flip; the scalar margin IS the exact distance-to-nearest-wall. This is
  the empirical confirmation that **near the boundary annulus (where 93% of flips live, GT median
  margin 0.22) the tropical/Laguerre picture is EXACT.**

### 1.3 The correct global object: a Fisher-metric anisotropic Laguerre tessellation

Because the logits are smooth, the *effective* metric on the logit-difference space is not Euclidean
but the **Fisher information metric** of the softmax. We already measured that
**Fisher curvature ‖F_x‖ ≈ a monotone function of the cheap `top1−top2` margin (Pearson 0.978,
Spearman 0.908; trace tracks 0.997)**, and the cross-boundary/along-boundary eigen-ratio ≈ **7:1**.
In power-diagram language: the SegNet partition is an **anisotropic Laguerre tessellation in the
Fisher metric** — Voronoi/Laguerre-with-a-Riemannian-metric (cf. Boissonnat–Wormser–Yvinec
anisotropic Delaunay/Laguerre). This is not a new lever; it is the **name** that unifies three
measured facts (Fisher↔margin, 7:1 anisotropy, annulus localization) with the already-pinned
"annulus-localized anisotropic Fisher-natural-gradient v2 loss." The margin field is a byte-faithful
surrogate for the whole metric ⇒ no metric computation needed at train time.

---

## §2 — Semi-discrete OT: is d_seg an OT/assignment mismatch? (mostly FRAMING + one lever)

### 2.1 What d_seg actually is

`d_seg = mean_pixels 𝟙[ argmax_k ℓ_k(x_gen) ≠ argmax_k ℓ_k(x_gt) ]`. This is the **Hamming /
symmetric-difference measure between two Laguerre labelings** (the GT-frame labeling and the
generated-frame labeling) evaluated at the SAME pixels. It is **not** a transport cost: there is no
mass moving between locations, no ground metric on label-swaps, no coupling. So the naive reading
"d_seg = a Wasserstein distance" is a **FALSE FRIEND** — it is a 0-1 label-agreement, the L⁰
mismatch of two partitions.

### 2.2 Where semi-discrete OT legitimately enters

The **effort to FIX** an error is transport-like. To repair a mislabeled pixel, the witness must
push its logit vector back across the nearest wall; the required transverse displacement = the
**margin** = the gradient of a Kantorovich potential. Formalizing:

- **AHA / semi-discrete-OT identity.** For OT from a continuous source `μ` (the pixel/feature
  distribution) to the **discrete target `ν = Σ_k ν_k δ_{p_k}`** (the K=5 classes) under squared-
  Euclidean cost, the optimal map partitions the source into **Laguerre cells** whose weights `w_k`
  are the dual potentials, solved so that `μ(L_k) = ν_k` (mass-matching; Aurenhammer–Hoffmann–Aronov
  1998; Kitagawa–Mérigot–Thibert 2019 give the Newton solve; Mérigot 2011 / Lévy 2015 the numerics;
  Lei–Gu 2019 the Brenier-potential/GAN reading — discriminator computes the Kantorovich potential,
  generator the map, projection of the Brenier potential IS a power diagram).
- **We do not own the sites/weights.** The scorer is frozen — its `p_k, w_k` are fixed by
  EfficientNet-B2. So we **cannot run the AHA weight-solve to DESIGN the partition**; the classic
  semi-discrete-OT algorithm has no free variables here. This is the crucial honest boundary: OT is
  a solver when the diagram is yours to build (mesh generation, blue-noise, generative Brenier maps);
  our diagram is the adversary's.
- **What survives = the DUAL and the MASS-BUDGET.** (a) The dual potential's gradient at the wall =
  the margin ⇒ the OT reading **re-derives our margin-saliency loss** (#141), nothing new. (b) The
  mass-matching condition `μ(L_k)=ν_k` gives a **principled per-class cell-mass target**: choose the
  per-class logit offset so each realized class-cell mass matches the GT class frequency. That is a
  real, cheap regularizer (§5 T1/T3).

**Verdict for §2:** semi-discrete OT is **FRAMING** (it renames margin-saliency as the Kantorovich
dual) plus **one modest lever** (mass-budget offsets). It is emphatically **not** a new coupling loss
or a preimage solver for our problem, because the diagram is frozen and not ours to optimize. Naming
a margin-weighted d_seg surrogate "an OT loss" would be the *search-masquerading-as-a-solver* fake
(NO-FAKE #6) — we avoid it.

---

## §3 — Tropical/polyhedral boundary chart vs curvelets (the genuinely new nexus)

### 3.1 The boundary is a tropical hypersurface; the lane is a PL facet

`argmax_k φ_k` for K=5 has its decision boundary = the zero-set of `m = φ_top1 − φ_top2`; triple
points = unions of pairwise zero-sets. This is *exactly* a tropical curve / power-diagram wall
complex. Our measured decomposition:

- **Lane residual = a codim-1 Road|Lane FACET** — 41.4% of all facets; 0.59% area; 35–40% of flips
  (55× over-represented); lane↔road = 55% of the residual; ~8-dim nonlinear lane-orbit manifold.
- **Triple junctions** (the "5-logit simplex" DOF the probe isolated) = 0.027% of px, ~1–2% flip
  mass, mostly Road|Undrivable|Movable **car-corners** — a *flip-structure* DOF, NOT the lane tail.
- **Root cause (measured, decision-changing):** the ALONG-TANGENT FREQUENCY DEFICIT — the self-orient
  /curvelet basis is sharp *across* the edge (freq→Nyquist) but smooth *along* it (≤8 cyc/unit); the
  lane **dashes** are a ~25 cyc/unit on/off modulation along the tangent → 3.2× deficit → the basis
  cannot represent the dashes → finest-first erasure (`birth_death_persistence`: 1–6px dashes erased
  97%, 100+px runs 25%; error ∝ 1/persistence).

### 3.2 Curvelets vs tropical/PL — which is the right chart?

- **Curvelets** are provably optimal (Candès–Donoho) for **smooth, C² CURVED** singularities under
  parabolic scaling. That is the right prior for the smooth Road|Undrivable arcs.
- **The lane dashes are genuinely PIECEWISE-LINEAR** (straight painted segments, on/off). A **tropical
  / max-plus curve** — a polyhedral chart — is topology-matched to *straight PL* structure with
  **corners and gaps**, which is exactly where curvelets pay their worst constant (a curvelet frame
  needs many atoms to build a hard on/off comb along a line; a PL comb needs a few numbers).
- **RATE is the counted term.** A PL "dash comb" is parameterized by (phase, period, width, per-lane
  polyline vertices) = **tens of bytes**, whereas resolving the same 25 cyc/unit modulation in a
  Fourier/curvelet basis costs many coefficients. On the COUNTED archive term, the polyhedral chart
  is strictly cheaper for this specific structure.
- **R supports it.** R is all-pass to 2px (MTF 0.997 at the 10px dash scale, 0.955 at stem-Nyquist,
  0.00 only at 2px) ⇒ the sharp PL walls the dash comb needs **survive the observation map** — the
  polyhedral chart is not defeated by R (confirmed non-binding at ep200).

**Verdict for §3:** the tropical/polyhedral boundary chart is a **PLUGS-IN** candidate — the one
place the lens yields a concrete NEW lever (T2) that directly attacks the measured root cause AND
wins on the counted term. It must A/B against (i) the pure frequency-basis fix (`--n-dir-freqs 2→4`,
`--bank-n-scales 4→5`) and (ii) the incumbent openpilot analytic polynomial band (which is smooth,
not dashed — so the dash comb is *complementary*, not competing, to the openpilot band: the polynomial
places the lane centerline, the tropical comb places the dash on/off along it).

---

## §4 — Honest verdict table (plugs-in / false-friend / framing-only)

| # | idea | verdict | why (our-numbers-grounded) |
|---|---|---|---|
| A | argmax partition = power/Laguerre diagram | **TRUE-LOCAL / FRAMING** | Exact for PL nets (Balestriero–Baraniuk); our smooth net ⇒ local/curved; simplex probe (min diff 0.0/118M px) confirms margin=distance-to-wall. Already in #218. |
| B | decision boundary = tropical hypersurface | **TRUE / FRAMING** | Zhang–Naitzat–Lim, Alfarra–Bibi. = the zero-set of `m`. Renames what we have. |
| C | softmax = soft power diagram; tau = entropic-OT temp | **TRUE / FRAMING (bonus)** | "From Hard to Soft"; re-reads our softmax-tau curriculum as Sinkhorn annealing. Conceptual, no new lever. |
| D | d_seg = a Wasserstein/OT cost | **FALSE FRIEND** | d_seg is Hamming/symmetric-difference of two labelings, not a transport cost. Don't call a margin loss "OT". |
| E | run semi-discrete OT to design the partition | **FALSE FRIEND (frozen scorer)** | AHA needs free sites/weights; ours are the adversary's. No solver plug-in. |
| F | OT dual (Kantorovich potential) as the loss | **FRAMING (= #141)** | The dual gradient at the wall IS the margin; re-derives margin-saliency. |
| G | AHA mass-matching → per-class logit offsets (#218 Laguerre weights) | **PLUGS-IN (modest, $0)** | Principled target for LDAM/logit-adjust: set offset so cell mass = GT freq; counters Lane/Movable collapse. |
| H | tropical/PL "dash comb" along-tangent lane chart | **PLUGS-IN (NEW, highest EV)** | Attacks measured along-tangent freq deficit (3.2×); rate-cheap on the counted term; R-survivable; complementary to openpilot band. |
| I | max-affine (tropical) logit head for the witness | **FRAMING / MARGINAL** | Another parameterization of argmax-of-SDF; the deficit is basis-richness, not head-form. Marginal over step_basis/SDF. |
| J | Fisher-metric anisotropic Laguerre tessellation | **FRAMING (unification)** | Names + unifies Fisher↔margin 0.978, 7:1 anisotropy, annulus loss. No new lever. |
| K | #218 power-diagram-weighted long-tail | **CONFIRMED** | The lens formalizes an already-queued #218 lever and supplies the offset-setting recipe (G). |

---

## §5 — Engineering nexus: ranked candidate levers (each a MEANS with a $0 pre-metric)

### T2 — Tropical / PL "dash comb" along-tangent lane modulation  ★ highest EV (NEW)
- **What:** parameterize the Road|Lane facet's along-tangent structure as a **max-plus / piecewise-
  linear on/off comb** (params: phase φ, period p, duty/width, per-lane polyline vertices), rendered
  at scorer resolution on the lane facet. Phase φ = **ego-ξ forward distance** (the dash phase = the
  3rd dual-use of the screw twist; `dashgap_fp` range-dependent finding).
- **Why:** directly closes the measured **along-tangent frequency deficit** (basis ≤8 vs dash ~25
  cyc/unit) that the frequency/curvelet basis structurally cannot; **rate-cheap** on the COUNTED
  term (tens of bytes vs many coefficients); R-survivable (all-pass to 2px). Complementary to the
  openpilot analytic band (band = centerline; comb = dash on/off).
- **EV / honesty:** the strongest NEW output of this lens, but it must beat/complement two incumbents
  in a clean A/B: `--n-dir-freqs 2→4` (the pure basis fix) and the openpilot polynomial band. It is a
  representation lever → converges into inflate.py as a deterministic parametric rasterizer (FREE
  generic code per rule-118), storing only the per-frame (φ, p, polyline) sufficient statistic.
- **$0 pre-metric:** on cached `gt_n96/n600` lane facets, fit a PL dash-comb to the GT lane on/off
  profile and measure the fraction of lane-flip mass a K-vertex comb recovers vs a `n_dir_freqs=2`
  Fourier fit at equal byte budget (realized-through-R, numpy-fp32). Falsify if comb recall ≤ Fourier
  recall at equal bytes.

### T1 — AHA mass-matched Laguerre logit-offset head  ($0, byte-free, do-FIRST-as-metric)
- **What:** per-class additive logit offset `w_k` (a length-5 vector, or compiled into inflate) set by
  the **AHA/semi-discrete-OT mass-matching** condition: choose `w_k` so each realized class-cell mass
  ≈ GT class frequency. This is LDAM/logit-adjustment (#218 "Laguerre weights") with a *derived*
  target instead of a heuristic log-frequency.
- **Why:** counters minority-class collapse (Lane 0.59%, Movable 1.56%) — enlarges their Laguerre
  cells by shifting the wall. Byte-free (5 scalars; deterministic ⇒ FREE in inflate). Composes
  orthogonally with clDice/persistence (#218 FEED-pt) and with T2.
- **EV / honesty:** modest and already half-present in #218; the lens's contribution is the *principled
  offset recipe* (mass-matching) + the confirmation. Cheapest thing to try; good as an early $0
  metric arm.
- **$0 pre-metric:** on `gt_n600`, sweep 5-offset vectors {log-freq heuristic vs AHA mass-matched vs
  zero}; measure realized-through-R d_seg on the frozen SegNet. Falsify AHA if it does not beat the
  log-freq heuristic.

### T3 — Semi-discrete-OT class-cell-MASS budget regularizer  (low–medium; likely dominated)
- **What:** a soft training penalty that the generated frame's realized class-cell **masses** match GT
  class frequencies (the global AHA condition as a loss, not just an offset).
- **Why / honesty:** a global class-balance anchor with an OT justification; but the **clDice/
  persistence loss (#218 FEED-pt) is more surgical** for the lane tail (it reads β0/β1 structure, not
  a mean mass), so T3 is likely **dominated** for the residual and offers only a cheap complementary
  global anchor. Keep as a composable low-priority term, not a headline lever.
- **$0 pre-metric:** correlate per-class realized cell-mass error with per-class d_seg on `gt_n96`; if
  the correlation is weak (mass is not the bottleneck — the tail is a *structure* problem), deprioritize.

### T4 / T5 — Fisher-Laguerre framing + max-affine witness head  (FRAMING; no direct EV)
- **T4** names the correct global object (Fisher-metric anisotropic Laguerre) and unifies three
  measured facts with the pinned v2 loss — conceptual, high-clarity, **no new lever**.
- **T5** (make the witness logit head an explicit max-of-affine / tropical polynomial so its argmax is
  a power diagram by construction) is **marginal** over the existing `argmax-of-SDF` chart, because the
  binding gap is basis richness along the tangent (T2), not head form. Bank as a representational
  variant, not a priority.

---

## §6 — Triality mapping (keep DAG ↔ DSL ↔ equations consistent)

**Equations to register** (FORMALIZATION_PENDING; parent consolidates into `tac.canonical_equations`):
- `segnet_argmax_partition_is_local_laguerre_power_diagram_v1` — argmax-of-affine ≡ power diagram
  (Aurenhammer dictionary `p=−a/2`, `w=‖a‖²/4−b`); exact for PL nets, LOCAL for smooth SegNet;
  empirical anchor = simplex probe min-diff 0.0/118M px + Fisher↔margin 0.978.
- `d_seg_is_hamming_symmetric_difference_of_two_laguerre_labelings_v1` — NOT a Wasserstein cost;
  OT dual = margin (= #141); AHA mass-matching = principled per-class offset.
- `tropical_pl_dashcomb_is_rate_optimal_along_tangent_lane_chart_v1` — PL comb beats Fourier/curvelet
  at equal bytes for straight on/off dashes; phase = ego-ξ; closes the 3.2× along-tangent deficit.

**DSL gauge vocabulary to add** (parent registers on land; flag-validated):
`HeadGauge.LAGUERRE_MASS_MATCH(offsets)` · `Lane.TROPICAL_DASH_COMB(phase=ego_xi, period, width,
polyline)` · `Reg.OT_CLASS_MASS_BUDGET` · (bank) `HeadGauge.MAX_AFFINE_TROPICAL`.

**Proposed DAG FEED line** (for the parent to fold into `sub015_DAG`): *"Deep-math lens #284
(tropical/OT/power-diagram): argmax partition = LOCAL Laguerre power diagram (simplex probe = the
first-order fact); d_seg = Hamming mismatch of two Laguerre labelings (NOT Wasserstein — frozen scorer
⇒ OT is framing + mass-budget lever, not a solver); the ONE new nexus = a tropical PL dash-comb that
closes the along-tangent freq deficit at rate-cheap counted bytes (A/B vs `--n-dir-freqs` + openpilot
band). Confirms/formalizes #218 Laguerre logit-offset (AHA mass-matching = the offset recipe). Pointer
0.19110 UNMOVED."*

---

## §7 — Key papers (cite)

- **Aurenhammer (1987)** *"Power diagrams: properties, algorithms and applications"*, SIAM J. Comput.
  16(1) — affine diagram ⇔ power diagram; the `(p,w)↔(a,b)` dictionary.
- **Aurenhammer, Hoffmann, Aronov (1998)** *"Minkowski-type theorems and least-squares clustering"*,
  Algorithmica 20 — semi-discrete OT (squared cost) ⇔ power diagram; mass-matching weight-solve.
- **Balestriero & Baraniuk (2018)** *"A Spline Theory of Deep Networks"* (ICML) + *"Mad Max: Affine
  Spline Insights into Deep Learning"* — deep nets = compositions of max-affine spline operators.
- **Balestriero, Cosentino, Aazhang, Baraniuk (2019)** *"The Geometry of Deep Networks: Power Diagram
  Subdivision"* (NeurIPS; arXiv 1905.08443) — **the input partition IS a subdivided power diagram.**
- **Balestriero & Baraniuk (2018)** *"From Hard to Soft: Understanding Deep Network Nonlinearities via
  Vector Quantization and Statistical Inference"* (arXiv 1810.09274) — **smooth nonlinearity = soft /
  entropic power diagram** (the model for our SiLU/softmax SegNet).
- **Zhang, Naitzat, Lim (2018)** *"Tropical Geometry of Deep Neural Networks"* (ICML; arXiv 1805.07091)
  — ReLU nets ⇔ tropical rational maps; boundary ⊂ tropical hypersurface; linear regions ↔ polytope
  vertices.
- **Alfarra, Bibi, Hammoud, Gaafar, Ghanem (2022)** *"On the Decision Boundaries of Neural Networks: A
  Tropical Geometry Perspective"* (IEEE TPAMI 45(4):5027) — boundary = subset of a tropical
  hypersurface ↔ convex hull of two zonotopes.
- **Lei, Su, Cui, Yau, Gu (2019)** *"A Geometric View of Optimal Transportation and Generative Model"*
  (arXiv 1710.05488) — Brenier potential projects to a power diagram; discriminator = Kantorovich
  potential, generator = OT map (the "we DON'T own the sites" boundary).
- **Kitagawa, Mérigot, Thibert (2019)** *"Convergence of a Newton algorithm for semi-discrete OT"* +
  **Mérigot (2011)** / **Lévy (2015)** — the numerics of the AHA weight-solve (for context; not
  runnable here since the scorer is frozen).
- (our anchors) `birth_death_persistence_dseg_20260630` · `contest_r_operator_mtf_allpass_to_2px_v1`
  · `scalar_top1_top2_margin_is_exact_distance_to_flip_v1` · #218 margin-field / Laguerre-weights ·
  #141 margin-saliency · Fisher↔margin 0.978.

---

## §8 — What this pass does NOT claim (NO-FAKE guardrails)

- **No pointer movement.** Pointer 0.19110 UNMOVED. Nothing here is a score; every number cited is a
  prior measured `[macOS-CPU advisory]` / realized-through-R anchor or a theorem. A lever becomes real
  only through a composed θ* byte-closed #202 exact row (CPU/CUDA, MPS-never).
- **No "OT solver" claim.** The scorer is frozen; we cannot run AHA to design the partition. Calling a
  margin-weighted d_seg surrogate "an optimal-transport loss" would be the search-masquerading-as-a-
  solver fake (NO-FAKE #6). The OT contribution is honestly labeled FRAMING + a mass-budget offset.
- **No global-power-diagram claim for a smooth net.** The identity is exact for PL nets and LOCAL for
  EfficientNet-B2; overstating it as a global partition algorithm would be a fake. Local/annulus is
  where it is exact — and where 93% of flips live.
- **Levers are A/B-owed.** T2 (dash comb) must beat/complement `--n-dir-freqs` and the openpilot band
  at equal bytes realized-through-R before any adoption; T1/T3 have named $0 falsification pre-metrics.
