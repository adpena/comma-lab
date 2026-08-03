# ddm_mf1 — Is the margin field a Morse function? Is Morse–Smale vocabulary licensed?

**Arm:** `ddm_mf1` · **Date:** 2026-08-03 · **Cost:** $0, scorer-free (cached GT fields only)
**Authority:** `[macOS-CPU advisory]` structural measurement on the FROZEN CPU-torch SegNet outputs
cached in `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`. No score claim. Pointer untouched.

**Status:** PRE-REGISTRATION + DERIVATION written BEFORE measurement (§0–§2). Results appended in
§3+ only after §2 was committed to.

---

## §0 The question

`ddm_de1` (`7a0d6f0abc`) established from `upstream/modules.py` alone that **"affine head =
Morse–Smale complex" is NOT an identity**: power/Laguerre and tropical forms ARE exact at the
terminal-feature head; Morse–Smale needs extra potential/flow hypotheses that were not found in
the corpus. `ddm_sx1` verified the power-diagram identity numerically (argmax agreement 1.000,
head `Conv2d(16,5,k=3)`, affine in R^144, rank exactly 4 ⇒ 140 dims invisible to `d_seg`).

**This arm asks whether the missing hypotheses can be SUPPLIED.** The argmax field itself is
piecewise-constant and has no gradient, so it cannot carry a Morse structure. The only credible
candidate Morse function is the **margin field** `m = z_(1) − z_(2)` (top logit minus runner-up),
which the corpus records as correlating with Fisher curvature at Pearson 0.978 and as the UNIWARD
steg-cost read as a cost.

Two hypotheses, tested separately, each with its own pre-registered falsifier:

- **H1 (smooth).** `m` is a Morse function on the image domain (isolated non-degenerate critical
  points, distinct critical values), and its Morse–Smale complex COINCIDES with the argmax
  partition.
- **H2 (discrete).** Forman's discrete Morse theory — which needs no smoothness and is defined on
  cell complexes, which a pixel lattice is — yields a descending-manifold decomposition that
  COINCIDES with the argmax partition.

**A failure of H1 does not kill H2.** INSTANCE < FORMULATION < FAMILY.

---

## §1 Derivation BEFORE reading the corpus (from `upstream/modules.py` only)

`SegNet.preprocess_input` takes `x[:, -1, ...]` (last frame only) and bilinear-resizes to
`segnet_model_input_size = (512, 384)`; the forward emits logits `z ∈ R^(B,5,384,512)`;
`compute_distortion` is the mean of `argmax(z1) != argmax(z2)`. On the model grid Ω define

- `L(x) = argmax_k z_k(x)` — piecewise-constant label field;
- `m(x) = z_(1)(x) − z_(2)(x) ≥ 0` — margin field (cached as `margins`; `clamp_min(0)` is a
  numerical no-op since top-2 order statistics already satisfy `z_(1) ≥ z_(2)`).

**(D1)** `m ≥ 0`, and `m(x) = 0` ⟺ at least two classes tie for the top ⟺ `x` is on the argmax
decision boundary. So the boundary is exactly the zero level set `{m = 0}`.

**(D2)** Near a point where exactly two classes `a,b` tie and `∇(z_a − z_b) ≠ 0`, locally
`m(x) = |z_a(x) − z_b(x)|`. Therefore **`m` is not differentiable on `{m = 0}`**: it is a crease,
with one-sided transverse derivatives `±|∇(z_a − z_b)|`.

**(D3)** `{m = 0}` is generically a codim-1 curve network in 2D. **Every** point of it is a global
minimum of `m` (value 0). So the minimum set of `m` is **1-dimensional**, not a set of isolated
points.

**(D4)** A Morse function requires `f ∈ C²` with all critical points isolated and non-degenerate.
By (D2)+(D3) `m` violates BOTH. It is not even Morse–Bott: Morse–Bott allows a critical
*manifold* but still requires the Hessian to be non-degenerate transverse to it, and `m` is not
even `C¹` transverse to it. **⇒ Predicted: `m` is NOT a Morse function.** This is a derivation,
not a measurement; §3 quantifies HOW it fails and whether anything weaker survives.

**(D5) Steelman ladder — other candidate potentials.**
- `−m`: same object, same crease. Non-Morse for the same reason.
- `z_(1) = max_k z_k`: a max of smooth functions, also creased on `{m=0}`; non-`C¹` on Ω, and its
  basins are not the argmax regions anyway.
- an unnamed smooth potential: the claim is vacuous until named; the strongest NAMED candidate is
  `m`, and that is what is tested.
- a **discrete** Morse function (Forman): needs no smoothness ⇒ survives (D4) and is tested as H2.

**(D6) A hypothesis-INDEPENDENT topological obstruction.** For a Morse function on a surface, the
stable manifold of a critical point of index λ is diffeomorphic to `R^(n−λ)`. For a **minimum**
(λ=0) in 2D the basin is `R²` — an **open disk**. Symmetrically the unstable manifold of a maximum
is an open disk. **Every 2-cell of a Morse–Smale decomposition is therefore simply connected — it
has no holes.** But an argmax region such as `Road` containing a `Movable` (car) region is an
**annulus**. If such components exist, the argmax partition's cells are NOT MS 2-cells, and this
conclusion holds *whatever* potential is proposed. It also directly refutes the weaker corpus
claim "the separatrix IS the argmax boundary": an annular cell would have to be subdivided by
separatrices that are NOT on any argmax boundary.

**(D7) What survives regardless.** Persistent homology / merge trees are defined for an
**arbitrary** filtration via the cubical complex — no Morse hypothesis is needed. So
"birth–death pair", "persistence", "low-persistence feature erased" are licensed by persistence
theory alone. What needs the Morse hypothesis is the specifically-Morse vocabulary:
"Morse–Smale complex", "separatrix as a gradient-flow stable manifold", "index-1 saddle",
"ascending/descending manifold transversality". §4 separates these lists explicitly.

---

## §2 PRE-REGISTERED FALSIFIERS (fixed before measuring; numbers stated in advance)

All measured on **n600** (all 600 scored frames, 384×512 = 117,964,800 pixels) from the cached
frozen-scorer fields. Denominators reported for every statistic (an empty scope is VACUOUS, never
PASS).

| # | Test | Statistic | Pre-registered falsifier |
|---|---|---|---|
| **F0** | MS cells are disks (D6) | fraction of argmax connected components with ≥1 hole; fraction of PIXELS in holed components | **> 0.1% of components OR > 1% of pixels ⇒ argmax cells are not MS 2-cells (hypothesis-independent)** |
| **F1a** | critical set is 0-dimensional | 8-connected component-size distribution of the weak-minimum set of `m`; enrichment of minima on `{m→0}` vs base rate | **> 1% of weak-minimum pixels in components of size ≥ 4 ⇒ critical set is 1-dimensional ⇒ not Morse**; ≥5× boundary enrichment corroborates |
| **F1b** | non-degeneracy / distinct values | fraction of critical pixels with a neighbour TIE (plateau ⇒ singular Hessian direction); fraction of duplicated critical VALUES among maxima within a frame | **> 1% either ⇒ degenerate; MS complex not well-defined without perturbation** |
| **F2** | cell ↔ critical-point bijection | `R = (#local maxima of m) / (#argmax connected components)` | **R > 2.0 ⇒ no bijection ⇒ the complexes do not coincide.** Constructive: report persistence `h*` where R→1 |
| **F3/F5** | partition coincidence (H1 **and** H2) | pixel agreement between the steepest-ascent (discrete-Morse) descending-manifold partition of `m` — at h=0 and at the best-case `h*` — and the argmax partition | **≥99% ⇒ coincide · <95% ⇒ do NOT coincide · 95–99% ⇒ PROVISIONAL** |
| **F4** | directional asymmetry (coordinator lead) | per-directed-class-pair transverse margin depth slope `s(a→b)` over distance bins 1..8 px; `A = \|s(a→b) − s(b→a)\| / max(\|s\|)` | **median `A` > 0.2 over pairs with ≥1e5 boundary px ⇒ the margin field is directionally asymmetric ⇒ every UNSIGNED POOLED statistic over it (`τ_end = m_q/ln5`, pooled UNIWARD cost) is a mixture and is suspect.** `A < 0.2` ⇒ the "pooled to zero" explanation for `msal_uni` at-chance is NOT supported by margin geometry |

**F3/F5 note.** The steepest-ascent basin partition of a grayscale digital image IS the
descending-manifold decomposition of the Forman discrete gradient built from the lower-star
filtration (Robins–Wood–Sheppard 2011). Classified `INFERRED_FROM_DOMAIN_LITERATURE`. One
implementation therefore tests H1's coincidence claim and H2's simultaneously; only H1's
*smoothness* premise is separately falsified by (D4).

**Verdict rule fixed in advance.** H1 is licensed only if F1a AND F1b pass AND F2 passes AND F3
≥99%. H2 is licensed only if F3/F5 ≥99% (it does not depend on F1a/F1b). F0 failing kills the
*coincidence* half of both, independent of everything else.

---

## §3 Results — n600, all 600 scored frames, PX = 117,964,800

Probe: `experiments/ddm_mf1_margin_morse_probe.py` · raw rows:
`.omx/research/ddm_mf1_morse_probe_n600.json`.

**Instrument validation (re-derive, don't recognise).** My independently-written boundary
extractor reproduces `ddm_sx1`'s census EXACTLY from a different cache:
`bnd_px = 2,551,382` (frac 0.021628) and `crack_len = 1,619,917` — both bit-identical to
`ddm_sx1_separatrix_geometry_n600.json`, which was built from
`gt_n600_lstars_slim.npz`, not from `gt_n600.npz`. Two extractors, two caches, same
numbers. The instrument is trusted for what follows.

| # | statistic (n600) | measured | bar | verdict |
|---|---|---|---|---|
| **F0** | argmax components with ≥1 hole | **876 / 21,304 = 4.112%** | >0.1% | **FIRES (41×)** |
| **F0** | pixels in holed components | **51,375,387 = 43.551%** | >1% | **FIRES (44×)** |
| **F1a** | weak-minimum px in components of size ≥4 | **0 / 1,813,112 = 0.00000%** | >1% | **does NOT fire** |
| **F1a** | weak-min enrichment on argmax boundary | **6.20×** | ≥5× | corroborates (D1/D3) |
| **F1b** | critical px with a neighbour tie | **0 / 3,500,060 = 0.00000%** | >1% | **does NOT fire** |
| **F1b** | duplicated critical values among maxima | **1,066 / 1,826,689 = 0.0584%** | >1% | **does NOT fire** |
| **F2** | `R` = #local maxima / #argmax components | **1,826,689 / 21,304 = 85.74** | >2.0 | **FIRES (43×)** |
| **F3/F5** | boundary agreement, h=0 | prec 0.0501 · rec 0.8930 · **min 0.0501** | <0.95 | **FIRES** |
| **F3/F5** | boundary agreement, fixed h\* | prec 0.4197 · rec 0.7058 · **min 0.4197** | <0.95 | **FIRES** |
| **F3/F5** | boundary agreement, **per-frame ORACLE threshold (ceiling)** | prec 0.8825 · rec 0.8331 · **min 0.8326** (best frame 0.8927) | <0.95 | **FIRES** |
| **F4** | median `A_p0` (margin at interface, both sides 100% support) | **0.0206** | >0.2 | **does NOT fire (10× under)** |

Supporting rows: junction degree — **6,702 triple points vs 1 quadruple point**
(ratio 1.5e-4); `h*` mean 3.268 / median 4.431 against margin quantiles
q05/q50/q95 = 2.11 / 5.895 / 7.358; at `h*`, 6.170% of pixels have roots below
threshold and are pooled (reported so the agreement number cannot hide them).

### §3.1 Three things that went the other way, and one steelman that changed the headline

**(a) My own analytic prediction D3 is NOT observable at the lattice scale.** D3 says the
continuum minimum set is a 1-dimensional curve. The measurement says the weak-minimum set
is **1,813,112 pixels in 1,813,086 components — essentially 100% isolated singletons**
(26 components hold 2 px; none holds ≥4). The resolution is not that D3 is wrong: the
lattice samples the continuum crease at 1 px, and the sampled floor values are generically
distinct, so **discretisation regularises the degenerate critical manifold into isolated
points**. Consequence, stated plainly: *the smoothness failure is not a practical
obstruction to building a discrete Morse complex on this field.* The obstruction to the
corpus's claims is the COINCIDENCE failure, not the smoothness failure.

**(b) The discrete margin field is a bona fide discrete Morse function.** 0.00000% ties,
0.0584% duplicated critical values. Forman's construction applies cleanly and the
persistence diagram is well-defined without perturbation. **H2's existence half is
licensed.** This is a positive result and it is the reason persistence language survives
(§4).

**(c) F4 refutes the margin-geometry leg of the asymmetry hypothesis.** All three variants
agree and all are far below the bar: `A_p0` median **0.0206**, `A_near` median 0.0747,
`A_far` median 0.1610. Per pair, `A_p0` = Road|Lane 0.166 · Road|Undriv 0.042 ·
Road|Movable 0.007 · Road|MyCar 0.005 · Undriv|Movable 0.021. The margin wall rises at
essentially the same rate on both sides of every interface — **transversely symmetric to
~2%**. Detail in §4.3, including what IS asymmetric.

**(d) STEELMAN that changed the headline.** My first `h*` rule gave F3 min-agreement 0.42.
Round-2 review flagged that as the most attackable point ("you picked a bad threshold"), so
the probe now sweeps the entire ladder per frame and keeps the **per-frame optimal**
agreement — a threshold chosen *with knowledge of the answer*, i.e. deliberately unfair to
me. The ceiling is **0.8326**, twice my h\* number. That is the honest headline, and it is
still below the 0.95 non-coincidence bar. Two things follow: (i) the failure is a **17%
mismatch, not a total mismatch** — the margin watershed is an ~83%-accurate *approximation*
of the argmax partition; (ii) **the 83% is not attainable in practice** — a decoder cannot
choose the threshold knowing the answer, and at a fixed threshold agreement falls to 42%.

---

## §4 Verdict, and the retire-list

### §4.1 The verdict, by the rule fixed in §2 before measuring

- **H1 (smooth Morse–Smale): NOT LICENSED — and it fails twice, independently.**
  1. *Smoothness*: `m` is non-`C¹` on the codim-1 set `{m=0}` (D2/D4). Derived from
     `upstream/modules.py`, no measurement needed.
  2. *Cell topology*: smooth MS 2-cells are open disks (D6). **43.551% of pixels live in
     argmax components that are annuli.** This is hypothesis-independent — it holds for
     *any* proposed potential, named or unnamed.
  Plus F2 (R = 85.74) and F3 (ceiling 0.8326).

- **H2 (discrete Morse / Forman): SPLIT VERDICT — existence licensed, coincidence NOT.**
  The discrete gradient field exists and is well-behaved (F1a, F1b both pass), so the
  construction is legitimate. But its descending-manifold decomposition **is not the argmax
  partition**: at the oracle ceiling it agrees on 83.3% of boundary, at a fixed threshold
  42.0%, and raw at h=0 it produces a wall network **17.8× longer** than the argmax
  boundary.

### §4.2 The precise relationship — the sharp part

Read the h=0 row again: **precision 5.01%, recall 89.30%.** These are not symmetric and the
asymmetry is the finding.

> **89.3% of the argmax boundary IS a discrete-Morse separatrix. But only 5.0% of the
> discrete-Morse separatrix network is argmax boundary.**

So the argmax boundary is (almost) a **SUBSET** of the margin field's separatrix network —
not equal to it. The separatrix network is 17.8× too long; the argmax partition is a
specific **5.6% selection** out of it. **Nothing in Morse theory tells you which 5.6%.**
That selection is made by the affine head — the power/Laguerre diagram that `ddm_de1`
derived and `ddm_sx1` verified at argmax agreement 1.000.

This is why the vocabulary is not merely imprecise but **inverted in load-bearing
direction**: Morse–Smale is being invoked to explain where the boundary IS, and it is
exactly the part that Morse–Smale does not determine. Corroborating signature: the boundary
network has **6,702 degree-3 junctions and 1 degree-4 junction** — the generic
Voronoi/Laguerre arrangement signature, consistent with the power-diagram identity.
*(Reported as corroboration only. Degree-3 junctions also occur in watershed networks, so
this is NOT used as a falsifier.)*

### §4.3 The asymmetry lead — corroborated in a different place than predicted

The `ORCHESTRATION_LEDGER.md` ASYMMETRY ADDENDUM (2026-07-08, verified at source, L1273–1280)
and `negatives_scale_validity_review_20260707.md` item 7 (verified at source, L331: *"$0
per-class-pair per-DIRECTION ρ from cached fields … kill: |ρ|<0.1 both sides"*, marked
**"THE flagship case"**) were carried to me by the coordinator. I ran the **precondition**
of that probe, and the result splits:

- **The margin-DEPTH asymmetry is REFUTED.** The transverse profile is symmetric to ~2%
  (F4). The proposed mechanism — *"an unsigned pooled estimator of a signed density has
  zero expectation when the two sides carry opposite signs"* — **does not draw support from
  margin geometry.** Whatever cancels in `msal_uni`, it is not asymmetric margin depth.
- **A large asymmetry IS present, and it is DIMENSIONAL, not shape.** Measuring how far the
  profile can be followed at ≥20% support: **Road side reach = 7 px, Lane side reach = 1 px**
  (support collapses 1.00 → 0.238 → 0.067 by d=2). Lane is a 2–3 px ribbon. The two sides of
  the Road|Lane separatrix are not differently-shaped walls — they are **different-dimensional
  objects: a half-space against a ribbon.**
- **Consequence for `τ_end = m_q/ln 5`: the concern SURVIVES, for a different reason.** An
  unsigned |margin| quantile pooled over both sides of Road|Lane averages a population with
  7× reach against one exhausted at 1 px, with comparable seed counts (629,474 vs 514,023).
  The mixture is governed by **class width**, which is measurable, not by signed-depth
  cancellation, which is not supported. Re-ground the suspicion on width.
- **What I did NOT test, stated so it is not mistaken for closed.** Item 7's ρ is between an
  UNIWARD/saliency cost and **flip mass**. Flip mass requires a candidate label field from
  our vehicle. I do not hold the scorer slot; searching `.omx/research/`, `experiments/results/`
  and the GT cache surfaced no cached candidate-label field for the live-best vehicle
  (scope stated — I did not search exhaustively, so this is "did not find in that scope",
  not "does not exist"). **Item 7 remains OPEN.** Its stated mechanism has lost its
  geometric support; its correlate is untested.

### §4.4 The retire-list — claims that must go, and claims that survive

**RETIRE (each requires the coincidence that F0/F2/F3 falsify):**

1. **"the separatrix" used to mean the argmax boundary while licensing flow reasoning.** In
   Morse–Smale a separatrix is the stable manifold of an index-1 saddle. Measured: only
   5.0% of the actual separatrix network is argmax boundary. The identification is false.
   *(The word may survive as an explicit nickname for the codim-1 zero set of `m` — it may
   not survive as a licence for gradient-flow inference.)*
2. **"the Morse–Smale complex of the argmax partition."** 43.6% of pixels sit in annular
   cells; smooth MS 2-cells are disks. No referent.
3. **"index-1 saddle" / "ascending–descending manifold transversality"** applied to the
   argmax structure. No referent.
4. **"lane dashes = birth–death pairs of the margin's Morse–Smale complex."** The
   *phenomenon* (dashes are erased) stands on its own separate evidence. The *mechanism*
   (a dash is a low-persistence CELL of the MS complex of `m`) requires the bijection that
   F2 refutes at 85.74×.
5. **"curriculum = flow = scale = persistence = Morse–Smale persistence order"** (the
   unification chain in the OPERATOR PRIORITY section) — the **Morse–Smale leg** of that
   identification is unlicensed. The annealing/scale legs rest on other evidence and are
   untouched by this arm.

**SURVIVES — and F1b is the measured reason:**

- **Persistent homology, merge trees, birth–death pairs, persistence-based simplification.**
  Persistence is defined for an **arbitrary** filtration on a cubical complex; it never
  needed Morse (D7). And now, additionally, F1b measured that the discrete margin field has
  **0% ties and 0.0584% duplicated critical values**, so its persistence diagram is
  well-defined and stable *without* symbolic perturbation. Persistence language is
  licensed — cite F1b, not Morse–Smale, as its warrant.
- **Discrete Morse theory as a construction** (F1a + F1b). It just does not produce the
  argmax partition.
- **Power/Laguerre + tropical forms at the terminal head** (`ddm_de1`, `ddm_sx1`) — exact,
  untouched, and now additionally supported as the thing that supplies the 5.6% selection
  Morse theory cannot.
- **"margin ≈ Fisher surrogate" (ρ 0.978)** — a correlation claim about a field, not a
  claim about a complex. Untouched by this arm.
- **"the boundary is codim-1"** — plain geometry, true, and F1a's 6.20× enrichment
  corroborates that the margin's minima track it.

---

## §5 The displacement half — does any flow structure earn a carrier?

**Short answer: Morse–Smale does not, and cannot. A registration field does, it is cheap,
and it does not need Morse theory at all.**

`ddm_sx1` §2.5 (verified at source, L182–185, L666): the seg residual splits
**boundary-PRECISION 76.4%** (Road↔Lane, Road↔Undriv, Road↔MyCar; 97–99% ON the GT
boundary) vs **object-DISPLACEMENT 23.3%** (Undriv↔Movable, Road↔Movable; flips/len 2×
denser; **18–20% more than 3 px OFF** the GT boundary), and *"The 23.3% displacement class
has no carrier in this table. It needs a per-pair positional DOF."*

### §5.1 Why Morse–Smale cannot supply it

A Morse–Smale structure is the gradient flow **of a scalar potential on one image**. A
displacement is a **correspondence between two images**. There is no potential whose
gradient is an object's positional error, and §4 shows there is no usable potential here at
all. The correct object is a **registration / deformation field** — a matching, a section of
a deformation groupoid, not a gradient of anything. Flow vocabulary earns its licence here;
*Morse* vocabulary does not travel with it.

### §5.2 The "second endpoint" obstacle dissolves

The stated structural obstacle: `SegNet.preprocess_input` takes `x[:, -1, ...]`, so a
single-frame description gives one endpoint of a displacement, never the displacement
(`ddm_sx1` L42, L547).

**The second endpoint is not at another TIME index. It is the other VERSION of the same
frame.** The displacement that costs us `d_seg` is the registration between **our decoded
`frame_1`** and **the GT `frame_1`** — the same scored frame, two versions, both in hand at
ENCODE time (we hold the GT; we can decode our own archive). The flow is spatial, not
temporal. SegNet's single-frame read is therefore irrelevant to it: SegNet never needs to
see two frames, because the correspondence is computed encoder-side and shipped as a
parameter the decoder applies. **The obstacle is an artifact of reading "flow" as temporal.**

### §5.3 The carrier, priced — MEASURED inputs, DERIVED arithmetic

Scoped to `Movable` because that is where sx1 localises the displacement class.
Per-class component census, n600 (`component_census.per_class`):

| class | components | ≥64 px | ≥64 px per frame | boundary px (≥64) |
|---|---:|---:|---:|---:|
| Road | 1,266 | 933 | 1.55 | 1,249,671 |
| Lane | 16,581 | 2,813 | 4.69 | 389,802 |
| Undrivable | 650 | 600 | 1.00 | 331,524 |
| **Movable** | **2,207** | **1,483** | **2.47** | **129,846** |
| MyCar | 600 | 600 | 1.00 | 307,901 |

Arithmetic (all constants recomputed, never re-typed):
`S/flip = 100/PX = 8.477105e-7` · `S/byte = 25/37,545,489 = 6.658590e-7` ·
`W = 1.2731082153` B/flip ✓ (matches the live invariant).
Live `d_seg = 0.004311790` ⇒ **508,639 flips**; displacement share 23.3% ⇒ **118,513 flips**.

- Movable crack perimeter (converting boundary-px to crack edges by the measured global
  ratio 1,619,917/2,551,382 = 0.634907): **82,441 crack edges**.
- First-order isotropic translation model: a component translated by `δ` flips
  `∫|δ·n| ds ≈ (2/π)·P_crack·|δ|` ⇒ **52,484 flips per 1 px of displacement** across all
  Movable components.
- **Consistency cross-check (independent):** the measured 118,513-flip displacement debt
  implies a mean displacement of **2.26 px**. `ddm_sx1` independently measured that
  **18–20% of displacement-class flips lie more than 3 px off** the GT boundary — exactly
  what a ~2.3 px mean produces. Two unrelated routes agree.
- **Cost:** one 2-vector per Movable component ≥64 px, quantised 0.5 px over ±8 px
  (≈10.1 bits) = 1.25 B ⇒ **1,483 × 1.25 = 1,854 B** total across all 600 frames.
- **Break-even: 1,456 flips = 1.23% of the displacement debt.**
- **At full effectiveness: net −0.0992 S = 13.66% of the 0.7262358 gap.** At 50%
  effectiveness: **−0.0490 S = 6.75% of gap.** Exchange rate **0.01564 B/flip vs
  W = 1.2731 → 81.4× better than the buy threshold.**
- Even the maximally generous variant — a 2-vector for *every* component ≥64 px, all
  classes (6,429) — costs **8,036 B = 0.00535 S**, break-even 6,312 flips.

**These bytes are video-derived and are COUNTED in `archive.zip`** (rule-118); they are
priced as counted above. The *model* (translate a component) is generic algorithm and is
free in `inflate.py`.

### §5.4 What is NOT established here — the named blockers

1. **Rigid-translation adequacy is UNMEASURED.** The residual after per-component
   registration is unknown; the real error may be non-rigid (scale as a car approaches,
   rotation, silhouette error). Break-even is only 1.23%, so the bar is very low — but "low
   bar" is not "cleared". *The experiment:* decode `cx1`, compute per-Movable-component
   argmax label fields for ours and GT, solve the per-component `δ` that maximises overlap,
   and report flips before/after. Scorer-free once the candidate label field exists; needs
   one n600 scorer pass to produce it.
2. **The vehicle needs a per-object HANDLE.** TR1 is 99.0% tokens; applying a per-object
   translation means a decoder-side spatial re-index of that object's tokens. Cheap in
   bytes, unbuilt in mechanism. This, not the byte price, is the real engineering gate.
3. The 23.3% share is **inherited** from `ddm_sx1`, not re-measured here.

---

## §6 Assumption ledger (verdict-scope discipline)

| # | assumption | classification |
|---|---|---|
| A1 | `margins` in the GT cache is `top1 − top2` of the frozen CPU-torch SegNet on `frame_1` at 384×512 | **VERIFIED_VIA_SOURCE_INSPECTION** (`seg_core.py:111`, `diag_disagreement.py:48`, `build_shared_gt_cache_for_mlx_fleet.py:56`) |
| A2 | boundary/crack extraction is correct | **VERIFIED_VIA_EMPIRICAL_ANCHOR** — bit-exact vs `ddm_sx1` from a different cache |
| A3 | smooth MS 2-cells are open disks (D6) | **INFERRED_FROM_DOMAIN_LITERATURE** (stable manifold theorem: `W^s(p) ≅ R^(n−λ)`) |
| A4 | steepest-ascent basins ≡ Forman descending manifolds of the lower-star filtration | **INFERRED_FROM_DOMAIN_LITERATURE** (Robins–Wood–Sheppard 2011) |
| A5 | holes counted with 8-conn background against 4-conn foreground | **VERIFIED_VIA_SOURCE_INSPECTION** — this was a round-1 BUG (4-conn over-counted: 13,306 → 12,906 holes, 43.810% → 43.551% px). Verdict unchanged, number corrected |
| A6 | F4 profiles are not washed out by a source pixel being near several interfaces | **PARTIALLY CONTROLLED** — internal control: profiles differ strongly per pair on the same source class (Road→Lane p1 = 1.640 vs Road→Movable p1 = 0.915), so the probe resolves per-interface structure rather than returning the class mean. Residual mixing could still *under*-detect asymmetry, which is conservative w.r.t. my "symmetric" conclusion — flagged, not eliminated |
| A7 | `A_p0` was selected as the primary F4 statistic *after* seeing the truncation confound | **DISCLOSED** — not cherry-picking: all three variants (`A_p0` 0.021, `A_near` 0.075, `A_far` 0.161) fall below the 0.2 bar |
| A8 | displacement is rigid per component | **ASSUMED_AWAITING_VERIFICATION** — §5.4 blocker 1; this is §5's whole risk |
| A9 | 23.3% displacement share | inherited from `ddm_sx1` (**VERIFIED_VIA_SOURCE_INSPECTION** of its memo, not re-measured) |
| A10 | measurement is on the **GT** margin field | scope statement — the candidate field is a perturbation of it; per-vehicle margin structure untested |

**Verdict scope.** F0/F2/F3 falsify the **FORMULATION** "the argmax partition is the
Morse–Smale (or discrete-Morse) complex of the margin field" — the strongest named candidate
potential, at n600, on the frozen authority's own outputs. They do **not** falsify the FAMILY
"some topological structure organises the argmax partition": persistence survives (§4.4),
and power/Laguerre is exact. INSTANCE < FORMULATION < FAMILY respected.

**Review:** 3 adversarial passes. R1 found the hole-connectivity bug + the h\* selection bug
+ the F4 truncation confound (counter reset). R2 found that a single h\* is not the
hypothesis's best case and added the per-frame-oracle ceiling sweep, which **doubled** the
F3 headline from 0.42 to 0.83 (counter reset). R3 clean.

## NEXT-IF-RESUMED

1. **Item 7 (`msal_uni` per-side ρ) is still OPEN** and is now a *smaller* job: its
   geometric precondition is measured (symmetric depth, asymmetric reach). It needs one
   candidate label field to compute flip mass per direction. Re-ground the τ_end suspicion
   on class **width**, not signed depth.
2. **§5.4 blocker 1 is the highest-value follow-up in this memo** — per-component `δ` fit
   against a decoded `cx1` label field. Break-even is 1.23%; upside 6.7–13.7% of the gap.
3. **§5.4 blocker 2** — does TR1 expose a per-object handle? That is a vehicle question, not
   a geometry question, and it gates §5 entirely.
4. Diff against sister arm `ddm_hg1` (reading the negatives as a geometric shape, same two
   leads). Agreement = corroboration; divergence = the finding.
