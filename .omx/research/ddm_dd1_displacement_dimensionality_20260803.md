# ddm_dd1 — Displacement and dimensionality: settling the 2–3× conflict, and the DOF ladder

**Arm:** `ddm_dd1` · **Date:** 2026-08-03 · **Cost:** $0, scorer-free (no scorer forwards; `ddm_pu2`
holds the slot). **Authority:** `[macOS-CPU advisory]`, `score_claim=false`,
`promotion_eligible=false`, `rank_or_kill_eligible=false`. Substrate: cached GT fields
(`lstars`) + the three sister arms' emitted JSON. No decode, no vehicle, no training.

**Baseline for every ΔS in this memo:** live best `S = 0.7910689` (`ddm_pu2`, archive sha
`c72ef357`, 353,805 B), seg leg `0.4311790`, `d_seg = 0.004311790` ⇒ **508,639 flips**.
Target = PR130 bar `0.172141`. **LIVE gap = 0.6189279.** `W = 1.273108215332` B/flip
(recomputed: `(100/PX)/(25/37,545,489)`, `PX = 512·384·600 = 117,964,800`).

---

## §0 HEADLINE

1. **The `sx2`↔`mf1` displacement conflict is NOT a measurement disagreement.** Both arms
   measure the *same* flip mass to within **1.0009** (118,615 vs 118,513). The 3.39× headline
   gap decomposes exactly as **subset ratio 3.811 ÷ stale-denominator ratio 1.110 = 3.43**
   (observed 3.39). `mf1` prices the **whole** Movable-edge mass; `sx2` prices the **deep
   sub-part** (26.2% of it). They are different subsets of one agreed quantity.
2. **MAIN's reconciliation hypothesis is REFUTED as stated.** It was "`mf1` = full-recovery
   ceiling, `sx2` = measured share." Both are full-recovery **ceilings** of their own subsets.
   **Neither arm measured realization at all** — both say so in their own text. There is no
   measured-share number on this axis from either arm.
3. **NEW — `mf1`'s only independent corroboration is refuted.** `mf1` claims "two unrelated
   routes agree" because its implied 2.26 px mean displacement matches an observed 18–20%
   off-3px share. Tested against `sx2`'s **directly measured** translation profile: 2.26 px
   produces **2.35%** off-3px. 18–20% requires **d ≈ 5 px**. The cross-check never consulted a
   profile; it agrees with nothing.
4. **NEW — the displacement DOF count is set by ASPECT RATIO, and it is 1 not 2 for Lane.** A
   segment/curve displacement has **one** DOF (normal); the tangential component is a **gauge**.
   MEASURED (§3.1): `κ_eff·L ≤ 0.503` for all `L ≤ 32 px` ⇒ tangential/normal `≤ 0.126` on the
   separatrix. MEASURED (§3.2): per-component the gauge strength is `w/Λ` = **0.039 for Lane**
   (aspect 25.5) vs **0.36 for Movable** (aspect 2.76). *(This CORRECTS my own round-1 claim that
   `mf1` over-counted: `mf1`'s 2-vector for Movable is right.)*
5. **NEW — LANE is component-decomposable, more so than Movable, and it is the best-priced
   description we own.** 2,813 components ≥64 px = **4.69/frame** (vs Movable 2.47). A Lane
   component is a **curve**: median minor axis **2.51 px**, aspect **25.5**. One perpendicular
   offset per component = **703 B** against the **250,403-flip** Road↔Lane mass (**34.30% of the
   live gap**) ⇒ **453× better than `W`** — 5.6× better than `mf1`'s 81.4×. Description only.
6. **The `D`-null discount applies to NOTHING here — and the second half of what I first wrote is
   REFUTED by `rs2` (§3.3b).** Correct: `lstars` is scorer-side, so `D`'s nullity never discounted
   my figures; `rs2` then showed `D∘U` is **near-isometric on `range(U)`** (gain [0.6866, 1.0283],
   cond 1.22, **0.0%** attenuated below 1e-3), so it discounts **token-side** figures either.
   **My "≈4.4 effective camera px of realization width for Lane" was wrong** — the blind pixels are
   structurally unreachable from the token lattice, not a resource we lose. **The real limiter is
   the `clip(rint())` amplitude floor, which no linear instrument can see.**
7. **NEW — per-FRAME targeting is REFUTED for every edge that carries mass** (§3.4, live cx1).
   Road↔Lane frame-Gini **0.108**; **256 of 600 frames** needed for 50% of its flips, **517** for
   90%; top-60 frames capture **13.9%** against 10% for perfect uniformity. The four other big
   edges run Gini 0.168–0.358. Only the *negligible* edges are frame-concentrated (Movable↔MyCar
   Gini 0.990, 0.03% of flips). **But this negative is exactly what LICENSES `hs1`'s static cell
   carrier** — a static address is correct precisely because the mass is temporally uniform.
8. **Dimensionality:** the exact seg answer is describable in **216,395 B** (cheapest of three
   independent measurements) against a **647,553 B** buy threshold — **2.99× under**. We pay
   **499,689 B** of tokens, i.e. **2.31× the description of the exact answer**, and still carry
   all 508,639 flips. **The object we must ship is ~2.3× LOWER-dimensional than the object we
   pay for.** Every byte of the excess is realization overhead.

**PRIOR-LAW PREDICTION (anti-re-anchoring, per `[[re-anchor≠discovery]]`).** Before computing
§4 I predicted from the standing corpus law *"description is cheap, realization is the wall"*
that the description price would land **under** the `W`-budget and that no description-side
result would be decision-changing. **That prediction HELD** (2.99× under). §4 is therefore
**corroboration, not discovery**, and I mark it as such. The genuinely new content of this memo
is §1–§3 (the settlement, the refuted cross-check) and §5 (the normal gauge + bits-per-DOF),
not §4.

---

## §1 THE SETTLEMENT — what each arm actually measured

### 1.1 The three positions, stated without equivocation

| arm | quantity it priced | flips | headline | its gap denominator |
|---|---|---:|---|---:|
| `ddm_sx1` | "object-DISPLACEMENT class" = the two **Movable edges** (`Undriv↔Mov` + `Road↔Mov`) | 23.3% | unreachable by a separatrix carrier | — |
| `ddm_mf1` | the **same** Movable-edge mass, addressed by per-component rigid translation | 118,513 | **13.66% of gap** | 0.7262358 |
| `ddm_sx2` | flips **>3 px off** the GT separatrix, all five edges | 31,101 | **4.0–6.0% of gap**, ranks LAST | 0.6543562 |

Verified at source that `sx1`'s "displacement class" **is** the Movable edges
(`ddm_sx1_separatrix_carrier_20260803.md:183` — *"object-DISPLACEMENT | Undriv↔Movable,
Road↔Movable | 23.3%"*), and that `mf1` inherited it without re-measuring (`mf1` assumption
**A9**, self-declared).

### 1.2 The arithmetic, recomputed

```
Movable-edge share = 0.1185 + 0.1147 = 0.2332   (pc2 edge_decomposition)
  × 508,639 flips  = 118,615        vs mf1's 118,513      → ratio 1.0009   ✓ SAME QUANTITY
sx2 model-free off-3px 0.0611 × 508,639 = 31,078   vs sx2's 31,101         ✓
deep fraction on the Movable edges (mean-ref) = 0.2395
  → deep Movable 28,407 · shallow Movable 90,207    (sx2 quoted ~90,200)   ✓
  → deep Movable is 91.4% of sx2's entire 31,101-flip "hole"
```

**Subset ratio** 118,513 / 31,101 = **3.811**. **Denominator ratio** 0.7262358 / 0.6543562 =
**1.110**. Predicted headline ratio **3.433**; observed 13.66/4.03 = **3.390**. The residual
0.6% is `sx2`'s rounding of 0.0611. **The conflict is fully accounted for. Nothing is left over
for a measurement error.**

### 1.3 Where they DO disagree — and it is not the number

- **Flip mass:** no disagreement (1.0009).
- **Depth:** no disagreement. `sx1` reported 18–20% off-3px on those edges; `sx2` measured
  17.9–19.7%. `sx2`'s framing that it *overturned* `sx1`'s split equivocates: `sx1`'s
  "object-level 23.3%" was a **class** label, not a depth claim, and `sx1` printed the 18–20%
  depth figure itself in the same row.
- **Reach — a real disagreement, and `sx2` wins.** `sx1:185` asserts *"A separatrix carrier
  addresses the 76.4%. It cannot address the 23.3%."* That was an **inference from `flips/len`
  density**. `sx2` **measured** that 76–82% of the Movable-edge mass lies within 3 px, i.e. is
  boundary-shaped and therefore reachable by a boundary carrier. *Denser ≠ differently shaped*
  is correct and it stands.
- **Mechanism — a real disagreement, and `sx2` wins.** `mf1`'s per-component **rigid
  translation** is refuted at FORMULATION scope by `sx2`'s 5.8× route disagreement, and
  independently by §2 below.

### 1.4 Index-type check (coordinator's constraint, from `ddm_hs1`)

`hs1` found that a static-vs-per-pair **address** distinction can masquerade as a price
disagreement (its CELL case: `sx2`'s static prior 430× vs `qd1`'s cell-drop 25.5× *worse* than
`W`, same index, because one **ADDS a shared description** and the other **REMOVES per-pair
payload**). I checked whether the `sx2`↔`mf1` conflict is that trap. **It is not:**

| surface | index | address kind |
|---|---|---|
| `mf1` §5 carrier | per-COMPONENT (Movable ≥64 px) | implicit, from the decoder's own `L*` |
| `sx2` sub-pixel-phase carrier | per-COMPONENT (Movable, 2,197) | identical |
| `sx2` object existence+position | per-COMPONENT | identical |
| `sx2` **4.0–6.0% headline** | **none — it is a measured sub-population share** | n/a |

Both arms' *carriers* sit on the **same component index with the same address kind**, so the
`hs1` trap is excluded. But the check surfaces a **third mismatch on top of subset + denominator**:
`sx2`'s headline is not a carrier price at all — it is the **size of a measured sub-population**,
compared against `mf1`'s **carrier full-recovery ceiling**. That is a category mismatch of the
same *family* as `hs1`'s ADD-vs-REMOVE, one level up: **"how big is this population" vs "what
would a carrier addressing it buy."** Both quantities are correct; they are not comparable, and
the memos compare them.

### 1.5 Verdict

> **SETTLED. One flip mass (~118,600 = 23.3% of seg flips), two different subsets of it priced
> against two different stale gaps. `mf1`'s number is right for the whole; `sx2`'s is right for
> the deep sub-part. `sx2`'s "ranks LAST" verdict is correct *for the deep 24%* and does not
> apply to the shallow 76%, which is the larger prize and which `sx2` itself prices separately
> at ~105× `W`. Neither arm is wrong; `mf1`'s LABEL over-reaches and `sx2`'s HEADLINE
> under-scopes.**

Verdict scope: **FORMULATION** for `mf1`'s rigid-translation model (refuted). **INSTANCE** only
for the two headline percentages (both correct within their stated subsets). The FAMILY
"positional/displacement error is a real and buyable component of `d_seg`" **survives and is
strengthened** — it is now agreed by three arms.

---

## §2 THE REFUTED CROSS-CHECK (new)

`mf1` §5.3: *"the measured 118,513-flip displacement debt implies a mean displacement of
**2.26 px**. `ddm_sx1` independently measured that 18–20% of displacement-class flips lie more
than 3 px off — exactly what a ~2.3 px mean produces. **Two unrelated routes agree.**"*

Both halves fail.

**(a) The model that produces 2.26 px undercounts flips by 2.37×.** `mf1` uses the continuum
formula `flips/px = (2/π)·P_crack`, giving 52,484 (reproduced exactly). `sx2` **measured** the
same object: 994,622 changed px over 8 directions at `d = 1` ⇒ **124,328 changed px per px of
translation**. Ratio **2.37×**. The continuum `(2/π)` factor is wrong on a discrete lattice,
where a 1-px translation flips a set closer to the boundary-pixel count than to `(2/π)×`
crack-length. Correcting it: the same 118,600-flip debt implies **d ≈ 0.95 px**, consistent
with `sx2`'s route B (0.860 px on `pc2`'s flip total) and its deep-corrected 0.654 px.

**(b) 2.26 px does not produce 18–20% off-3px.** From `sx2`'s measured profile:

| d (px) | 1 | 2 | 3 | 4 | **5** | 10 |
|---|---:|---:|---:|---:|---:|---:|
| frac >3 px | 0.0233 | **0.0235** | 0.0371 | 0.0849 | **0.1858** | 0.4662 |

2.26 px ⇒ **≈2.4%**, not 18–20%. The observed 18–20% requires **d ≈ 5 px**. That is precisely
`sx2`'s route-A/route-B **5.8× disagreement**, which is the refutation of the single-magnitude
rigid model. `mf1` asserted the agreement without ever evaluating the profile.

> **Consequence:** `mf1`'s §5 carrier retains its *price* (1,854 B, still far under break-even)
> but loses its *only* independent support. Its assumption **A8** ("displacement is rigid per
> component", self-marked `ASSUMED_AWAITING_VERIFICATION` and called *"§5's whole risk"*) is now
> **actively contradicted**, not merely unverified. `mf1`'s §5.4 blocker 1 is therefore not
> optional due diligence — it is the load-bearing measurement.

---

## §3 THE NORMAL GAUGE — the displacement DOF count is set by ASPECT RATIO (new)

*(Heading corrected in R2. It previously read "both arms over-count the DOF by 2×" — that was my
R1 claim and §3.2 refutes it. Recorded rather than silently rewritten.)*

A displacement carrier attached to a **boundary segment** has **one** degree of freedom, not two.

Let a segment have unit normal `n`, tangent `t`, length `L`, curvature `κ`. Translate by
`δ = δ_n n + δ_t t`.

- **Normal:** swept (flipped) area `= δ_n · L`.
- **Tangential:** a straight segment translated along itself is **the same segment** — zero
  flips, exactly. For curvature `κ`, the normal displacement at arc position `s` is `≈ κ s δ_t`,
  so swept area `= κ δ_t L²/4`.
- **Ratio** `tangential/normal = κL/4`.

> **`δ_t` is a gauge whenever `κL ≪ 4`, i.e. while the segment is short compared to its radius
> of curvature `R = 1/κ`.** Under that condition the carrier is **1 scalar per segment**.

**Where it applies — CORRECTED by my own §3.2 measurement.** My first statement of this claim
said the halving corrects *both* `mf1`'s 2-vector and `sx2`'s 2-bits/axis. **That was an
over-reach and the measurement refutes it.** For a *component* carrier the gauge strength is set
by the component's **aspect ratio**: translating along the long axis only sweeps the end caps.
Measured (§3.2): tangential/normal ≈ `w/Λ` = **0.039 for Lane** (aspect 25.5) but **≈0.36 for
Movable** (aspect 2.76).

> **`mf1`'s 2-vector for Movable is CORRECT — I was wrong to call it over-counted.** The halving
> is essentially exact for **Lane**, which `mf1` never priced, and which carries **2.1× more flip
> mass** than Movable. The per-segment family in §5 (which is boundary-segment-indexed, not
> component-indexed) does keep the halving, since `κL/4 ≤ 0.126` there.

### 3.1 The gauge validity condition — MEASURED, n600 (this arm; gate CLOSED)

`experiments/ddm_dd1_contour_coherence.py` → `.omx/research/ddm_dd1_contour_coherence_n600.json`.
GT `lstars` n600, all 600 frames, no decode/vehicle/scorer. Normal estimated from the gradient of
the box-smoothed (5×5) indicator of each pixel's own class; orientation folded to `[0, π)`;
mean `|Δθ|` accumulated over 16 directions per radius.

**Instrument validation:** boundary-pixel total **2,551,382** — reproduces `sx1`'s independently
measured `boundary_pixels_total` to **ratio 1.0000**, from a different code path.

| `r` (px) | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mean \|Δθ\| (°) | 12.95 | 20.00 | 21.63 | 25.20 | 25.05 | 29.66 | 27.58 | 27.72 | 27.87 | 28.81 |
| `κ_eff` (1/px) | .2259 | .1746 | .1258 | .1099 | .0729 | .0647 | .0401 | .0302 | .0203 | .0157 |
| **`κ_eff·L`** | .226 | .349 | .378 | .440 | .437 | .518 | .481 | .484 | .486 | **.503** |
| **tang/norm = `κL/4`** | .057 | .087 | .094 | .110 | .109 | .129 | .120 | .121 | .122 | **.126** |

**The result is a SATURATION, not a coherence length.** Mean `|Δθ|` rises to ≈28° by `r ≈ 8` and
is then **flat to `r = 32`** — it does not random-walk toward the 45° uniform-random value. So
`κ_eff` falls as ≈`1/r`, and **`κ_eff·L` stays bounded at ≈0.5 across the whole range**. The
small-`r` `κ_eff` is dominated by 8-connected staircase quantisation, not by geometry.

> **GATE CLOSED: `tangential/normal ≤ 0.126` for every `L ≤ 32 px`.** The normal gauge holds
> across the entire useful carrier range, so **`L = 32 px` is admissible** and the §5 table
> should be read at its bottom row, not its top. Equivalently: one normal offset describes the
> local displacement over a 32-px segment to ≈12% amplitude accuracy (`cos 28° = 0.88`).

**Direction of the residual error (conservative).** Euclidean separation approximates arc length;
at large `r` some pairs lie on *different* contours and carry ≈random orientation, which **adds**
decoherence. The measured 28° at `r = 32` is therefore an **upper** bound on same-contour
decoherence, so the true usable `L` is **≥** 32 px. The bound errs in the safe direction.

**Not established:** this measures the GT separatrix's own geometry, which bounds how coarsely a
normal-offset carrier can be *parameterised*. It does **not** measure the displacement field
itself (that needs a decoded label field — §7 item 2). A carrier can be well-parameterised and
still describe the wrong thing.

### 3.2 IS LANE COMPONENT-DECOMPOSABLE? — MEASURED, n600 (coordinator's question)

`experiments/ddm_dd1_lane_component_census.py` → `.omx/research/ddm_dd1_lane_component_census_n600.json`.
4-connected components on GT `lstars`, all 600 frames, scorer-free.

**Instrument validation (third independent check):** every component count and area reproduces
`mf1`'s `component_census` **exactly** — Lane 16,581 / 2,813 ≥64 px; Movable 2,207 / 1,483;
Road 1,266 / 933; Undrivable 650 / 600; MyCar 600 / 600; all `area_px_sum` identical.

| class | comps | /frame | ≥64 px | ≥64/frame | area px | median major | median minor | **aspect** | **A/P** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Road | 1,266 | 2.11 | 933 | 1.55 | 27,407,046 | 478.3 | 94.0 | 5.09 | 21.9 |
| **Lane** | **16,581** | **27.64** | **2,813** | **4.69** | 690,639 | **64.0** | **2.51** | **25.48** | **1.41** |
| Undrivable | 650 | 1.08 | 600 | 1.00 | 58,413,281 | 523.6 | 192.1 | 2.73 | 176.2 |
| **Movable** | 2,207 | 3.68 | **1,483** | **2.47** | 1,460,325 | 24.8 | 8.98 | 2.76 | 11.1 |
| MyCar | 600 | 1.00 | 600 | 1.00 | 29,993,509 | 508.1 | 97.8 | 5.20 | 97.4 |

> **YES — and MORE decomposable than Movable. Lane runs 4.69 components ≥64 px per frame vs
> Movable's 2.47 (1.90×).** The `mf1` carrier *form* applies; the *DOF count* does not.

**A Lane component is a CURVE, not a region — measured three ways.** Median minor axis **2.51 px**,
aspect ratio **25.5**, `A/P = 1.41` (mean half-width). Compare Movable: 8.98 px, 2.76, 11.1. This
confirms `rz1`'s "Lane has NO INTERIOR" (6.92% ≥2 px deep vs Road's 63.64%) from an independent
direction — and it **resolves the coordinator's tension**: a *region-paint* primitive indeed
cannot serve Lane, but a *component* primitive can, because the component IS a curve.

**The curve-native variant, and why it is 1 DOF.** For a dash of length `Λ` and width `w`,
translating by `δ`: perpendicular sweeps both long edges (`≈2Λδ` flips); along-axis sweeps only
the two end caps (`≈2wδ`). Ratio = `w/Λ` = **2.51/64.03 = 0.0392**. **The along-axis DOF is worth
3.9% of the perpendicular one — the gauge is 25× more exact for Lane than the generic `κL/4 ≤
0.126` bound of §3.1.** Movable's `w/Λ = 0.362` is why `mf1`'s 2-vector is right there.

**Price (description budget, NOT a gain).** Road↔Lane carries **49.23%** of flips = **250,403
flips** = **0.21227 S = 34.30% of the live 0.6189279 gap** — **2.1× the entire Movable-edge mass**.

| carrier | comps | DOF | bits | bytes | B/flip | vs `W` |
|---|---:|---:|---:|---:|---:|---:|
| **Lane, 1 perpendicular offset** | 2,813 | 1 | 2 | **703** | 0.00281 | **453×** |
| Lane, 1 perpendicular offset | 2,813 | 1 | 4 | 1,406 | 0.00562 | 227× |
| Lane, 2-vector (the wasteful form) | 2,813 | 2 | 2 | 1,406 | 0.00562 | 227× |
| `mf1` Movable, 2-vector | 1,483 | 2 | ~10 | 1,854 | 0.01564 | 81.4× |

**453× is the best-priced description in the corpus — 5.6× better than `mf1`'s 81.4× headline.**
It is also, and this is the whole caveat, **still only a description.**

**Named limits, not hidden.** (a) A perpendicular offset addresses *position* only. Lane error
plausibly also has **width** and **existence** (island-birth) components; with a 2.51 px median
width, a ±1 px width error is a ±40% area error. I did **not** measure the decomposition of Lane
error into position/width/existence — that needs a decoded field (§7 item 2). (b) `Λ = 64 px` is
the median *major axis of the component*, not a straightness certificate; §3.1's 28° saturation
applies. (c) The 49.23% Road↔Lane share is inherited from `pc2`, not re-measured here.

### 3.3 THE `D`-NULL DISCOUNT — where it does and does not apply (coordinator's owed item)

The coordinator flags that `cell-mass × visible-fraction` under `D`'s 80.6742% null is unmeasured,
so capture figures are upper bounds. **That discount does not apply to any number in this memo,
and the reason is a lattice distinction worth stating rather than a factor worth applying.**

`D` maps **camera** pixels (874×1164) → **scorer** pixels (512×384); its 80.6742% nullity and
22.6969% blind fraction are properties of that map. Every quantity in §3.1/§3.2/§5 is measured on
`lstars`, which is **already in the scorer's own 384×512 lattice** — downstream of `D`. Scorer-side
separatrix and component measurements are therefore **not** discounted by `D`'s null; the null
discounts the **camera-side realization**, which is precisely the 2.31× realization multiplier of §4.

> **Applying a 0.77 visibility factor to a scorer-side capture figure would be a unit error.** The
> figures the coordinator flags (cell-mass capture) are camera-side-indexed and *do* need it.

### 3.3b **REFUTED BY `ddm_rs2` (`84367be88e`) — and it was my own paragraph**

The paragraph that stood here claimed: *"a 2.51 scorer-px dash is ≈5.7 camera px; with 22.6969% of
camera pixels blind, the encoder has ≈4.4 effective camera px of width — the class with the
cheapest description has the thinnest realization channel."* **That is wrong and I am striking it.**

`rs2` computed `M = D∘U` in closed form (linear, separable; BLAS vs `einsum` difference **0.0**;
`M` matches the real receiver to **1.7e-07** relative) and measured: full **196,608-dim** gain range
**[0.6866, 1.0283]**, condition **1.22**, **0.0%** of directions attenuated below 1e-3.
**`D∘U` annihilates nothing.** The 80.6742315% is exactly `1 − 196,608/1,017,336` — `D`'s null
fraction on the **camera** plane — and our renderer emits into `range(U)`, where `D` is
**near-isometric**.

> **The null space is real but STRUCTURALLY UNREACHABLE FROM THE TOKEN LATTICE.** The blind camera
> pixels are not pixels we were using and lost; they are pixels we cannot address in the first
> place. **They therefore subtract nothing from a token-mechanism's realization width.** My "4.4
> effective px" was a discount applied to a resource the mechanism never held.

**Which of my claims were leaning on the refuted reading — stated plainly, as asked.** Exactly one:
the struck paragraph above, and its §0 item 6 restatement. **§3.3's main conclusion is unaffected
and is in fact strengthened** — I argued the `D`-null does *not* discount scorer-side figures
because `lstars` is post-`D`; `rs2` shows it does not discount **token-side** figures either, for a
stronger reason (near-isometry on `range(U)`). §3.1, §3.2, §4 and §5 contain no `D`-null term and
are untouched. The 22.6969% blind fraction remains live **only for camera-plane carriers (`#401`)**.

**The replacement limiter, and it is a different KIND of object.** `rs2`'s point that supersedes
mine: the thing that destroys small signals is the `clip(rint())` **dead zone**, which is
**amplitude-dependent, not direction-dependent** — and **a linearisation cannot represent a dead
zone at all**, so every linear instrument (including gradient keys, and including `M` itself)
is blind to it. **The lost directions are not a subspace; they are an amplitude floor.** For a
2.51 px-wide Lane dash the operative question is therefore not "how many camera px can I reach"
but **"does the dash's amplitude clear the quantiser"** — answerable only by a realized finite
difference, never by a rank or nullity argument. That is the correct form of my §7 item 1b.

### 3.4 PER-EDGE × PER-FRAME — MEASURED on the LIVE cx1 vehicle (operator's "individual frames")

`experiments/ddm_dd1_edge_frame_concentration.py` → `.omx/research/ddm_dd1_edge_frame_concentration_n600.json`.
Consumes `ddm_pu2`'s already-materialised per-pair 5×5 directed confusion tensor
(`per_pair_directed.jsonl`, `/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/`).
**No scorer pass, no decode.** Per `m91`: decomposed per **EDGE**, never per class. `as1` owns
asymmetry composition and `hs1` owns cell-level Gini — both **cited, not re-measured**.

**Instrument validation (fourth):** total flips **508,640** vs the charter's 508,639 (ratio
1.000002); implied `d_seg` 0.004311795 vs 0.004311790; every frame's matrix sums to 196,608 and
every off-diagonal reproduces the stored flip count (asserted, not eyeballed).

| edge | flips | share | **frame-Gini** | frames@50% | frames@90% | top-60 capture | median/frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Road↔Lane** | 235,148 | **46.23%** | **0.108** | **256** | **517** | **13.9%** | 388 |
| Road↔Undriv | 89,545 | 17.60% | 0.168 | 230 | 503 | 16.6% | 143 |
| Road↔MyCar | 63,027 | 12.39% | 0.200 | 217 | 501 | 19.5% | 96 |
| Undriv↔Movable | 61,892 | 12.17% | 0.358 | 149 | 452 | 27.5% | 77 |
| Road↔Movable | 57,225 | 11.25% | 0.259 | 194 | 471 | 20.3% | 87 |
| Lane↔MyCar | 903 | 0.18% | 0.787 | 46 | 157 | 58.5% | 0 |
| Lane↔Movable | 681 | 0.13% | 0.746 | 52 | 202 | 53.6% | 0 |
| Movable↔MyCar | 135 | 0.03% | **0.990** | **3** | 7 | 100% | 0 |
| Lane↔Undriv | 84 | 0.02% | 0.981 | 5 | 14 | 100% | 0 |

> **The answer to "can we target INDIVIDUAL FRAMES?" is NO for every edge that carries mass, and
> the anti-correlation is near-perfect: frame-concentration rises exactly as flip share falls.**
> Road↔Lane needs **256 of 600 frames** to reach half its flips (uniform would need 300) — a
> frame-Gini of **0.108** is very nearly flat. The only frame-targetable edges (Gini 0.75–0.99) are
> the four that together carry **0.36%** of all flips. A per-frame address buys essentially nothing
> where the money is.

**But the negative is the license.** `hs1` measured seg × **CELL** Gini **0.8581, STATIC** (static
top-128 cells = **91.22% capture at 62 B**). Put beside my frame-Gini 0.108: the expensive flips
are **spatially concentrated and temporally uniform**. **That combination is what makes a STATIC
address correct** — the same cells are expensive in *every* frame, so one address amortises over
600 frames instead of needing 600 of them. Had the mass been frame-concentrated, `hs1`'s static
carrier would have been wrong for most frames. **Neither arm's number implies this alone; together
they do.**

**Re-priced Lane carrier — this CORRECTS my own §3.2, which used a tb1-vehicle share.**

| source of the Road↔Lane share | flips | ΔS | % of live gap | 703 B carrier |
|---|---:|---:|---:|---:|
| `pc2` / tb1 (what §3.2 used) | 250,403 | 0.212269 | 34.30% | 453.5× `W` |
| **cx1 / LIVE (measured here)** | **235,148** | **0.199337** | **32.21%** | **425.8× `W`** |

**Cross-vehicle transfer check** — `pc2`'s tb1 shares vs cx1 measured: Road↔Lane **0.939**,
Road↔Undriv 1.083, Road↔MyCar **1.138**, Undriv↔Movable 1.027, Road↔Movable 0.981. **They transfer
to within 0.94–1.14×, not exactly.** Inheriting them uncritically is a real ≤14% error source —
which is precisely the `mf1`-inherited-`sx1` failure mode of §1, and I committed the same one in
§3.2. **Corrected here and the mechanism named rather than the number quietly swapped.**

**§1's settlement is unaffected:** the Movable edges sum to **23.42%** on cx1 vs `pc2`'s 23.32%
(ratio **1.004**) — the quantity `mf1` and `sx2` were both pricing transfers essentially exactly.

---

## §4 DESCRIPTION PRICE — corroboration, NOT discovery

*(Flagged per the PRIOR-LAW PREDICTION in §0. Three arms already converged here; I recomputed
rather than re-typed, and `sx1`'s H1 reproduces to the digit.)*

| description of the **exact** GT label field, n600 | bytes | source |
|---|---:|---|
| H0 memoryless | 23,821,632 | `sx1` |
| H1 order-1 | **253,341** | `sx1` (quoted 253,341 ✓) |
| H2 order-2 | 237,136 | `sx1` |
| order-4 unconditional | 238,945 | `sx2` |
| **order-4 + free canny predictor** | **216,395** | `sx2` — cheapest measured |
| practical `lzma9e` ×10 | 428,120 | `sx1` — real codec, 1.69× above bound |

Band concentration: **89.69% of the bits on 2.16% of the pixels** (2,551,382 band px carry
222,729 B; the 115,413,418 interior px carry 25,612 B). Measured entropy **0.6984 bits per
strict-band px**. *(`rz1`'s "0.60 bits/px" is a **different quantity** — a budget over the
**dilated** 4,684,382-px band, not an entropy over the strict band. Not a conflict; noted so the
two are never averaged.)*

**Against the budget:** killing all 508,639 flips is worth `0.4311790 / (25/37,545,489)` =
**647,553 B**. Cheapest exact description **216,395 B** ⇒ **2.99× under**.

**Against what we pay:** TR1 token payload **499,689 B** (99.0% of the 504,736 B packet) =
**2.31×** the description of the exact answer — while still carrying all 508,639 flips.

> **The dimensional statement: the object we must ship is not larger than the object we pay for.
> It is ~2.3× SMALLER. The entire excess is realization overhead.** This does not overturn the
> corpus law; it puts a **single measured multiplier** on it.

---

## §5 THE DOF LADDER — dimensionality proper

| level | DOF | source |
|---|---:|---|
| camera RGB, both frames | 3,662,409,600 | — |
| after `D` (partition; 80.6742% null, 22.6969% blind to both scorers) | 707,788,800 | `rz1` |
| seg-scored: `frame_1` argmax symbols | 117,964,800 | — |
| separatrix band px (2.16%; 89.69% of the bits) | 2,551,382 | `sx1` |
| crack edges (true perimeter length) | 1,619,917 | `sx1`/`mf1` |
| **contour segments × 1 NORMAL offset** | **10⁵ order** | **this arm** |
| pose (`rank(J) ≤ 6`; 6 of 12 PoseNet outputs unscored) | 3,600 | corpus |

Per-segment carrier cost, **1 normal DOF** per segment (the §3 halving applied):

| `L` (px) | segments | @2 bits | B/flip | vs `W` | @3 bits | vs `W` |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 404,979 | 101,245 B | 0.1991 | 6.4× | 151,867 B | 4.3× |
| 8 | 202,490 | 50,622 B | 0.0995 | 12.8× | 75,934 B | 8.5× |
| 16 | 101,245 | 25,311 B | 0.0498 | **25.6×** | 37,967 B | 17.1× |
| 32 | 50,622 | 12,656 B | 0.0249 | 51.2× | 18,983 B | 34.1× |

**Bits per genuine DOF** (token payload 499,689 B against seg segments + 3,600 pose DOF):

| `L` | total DOF | bits/DOF |
|---:|---:|---:|
| 8 px | 206,090 | **19.4** |
| 16 px | 104,845 | **38.1** |
| 32 px | 54,222 | 73.7 |

> **We are paying ~19–38 bits per genuine degree of freedom.** At 8 bits/DOF the same object
> costs ~105 KB. This is the dimensionality answer in one number, and it is independent of the
> description-entropy route in §4 — two different roads to the same conclusion.

**`L` is now MEASURED, not free (§3.1).** `κ_eff·L ≤ 0.503` for every `L ≤ 32 px`, so the
operative row is **`L = 32 px`: 50,622 segments, 12,656 B at 2 bits, 51.2× better than `W`** —
and at `L = 32` the total seg DOF is **54,222**, i.e. **73.7 bits per genuine DOF** at the
current token payload.

**Honest status of the §5 table:** these remain **description budgets, not gains** — identical
in status to `sx2`'s and `mf1`'s tables and for the identical reason: **realization is
unmeasured**. §3.1 closed the geometric gate on `L`; it did not kill a single measured flip. Do
not quote any row of this table as a score result.

---

## §6 ASSUMPTION LEDGER

| # | assumption | classification |
|---|---|---|
| B1 | `sx1`'s "displacement class" = the two Movable edges | **VERIFIED_VIA_SOURCE_INSPECTION** (`ddm_sx1_...:183`) |
| B2 | `mf1` inherited the 23.3% rather than re-measuring | **VERIFIED_VIA_SOURCE_INSPECTION** (`mf1` A9, self-declared) |
| B3 | flip↔ΔS↔byte constants | **VERIFIED_VIA_EMPIRICAL_ANCHOR** — recomputed; `W` matches the live invariant to 12 dp; `sx1` H1 reproduces to the digit |
| B4 | `pc2`'s per-edge `off3` shares (17.9–19.7% Movable) | inherited from `pc2` via both arms; **not** independently re-measured here |
| B5 | tangential displacement is a gauge | **DERIVED** (first-order, validity `κL ≪ 4`) **+ VERIFIED_VIA_EMPIRICAL_ANCHOR** (§3.1, `κ_eff·L ≤ 0.503` n600). Scope **corrected in R2**: per-component the gauge is `w/Λ`, strong for Lane, weak for Movable |
| B8 | Lane component census / aspect ratio | **VERIFIED_VIA_EMPIRICAL_ANCHOR** — reproduces `mf1`'s `component_census` exactly for all 5 classes; boundary total ≡ `sx1` |
| B9 | Euclidean separation ≈ arc length in §3.1 | **PARTIALLY CONTROLLED** — cross-contour pairs add decoherence, so the bound errs safe (true usable `L` ≥ measured) |
| B10 | `w/Λ` end-cap model of the Lane gauge | **DERIVED**, first-order; assumes the dash is locally straight over `Λ`. §3.1's 28° saturation is the honest correction and is not folded in |
| B11 | Lane error is position-dominated | **ASSUMED_AWAITING_VERIFICATION** — §7 item 1b; this is §3.2's whole risk, exactly as A8 was `mf1` §5's |
| B12 | `D`-null does not discount scorer-side figures | **VERIFIED_VIA_SOURCE_INSPECTION** of the lattice definitions (`lstars` is 384×512 = scorer-side; `D` is camera→scorer) |
| B6 | `sx2`'s 8-direction translation profile is representative of the true error | `sx2`'s scope; it is an **isotropic GT-side probe**, not the real residual — the real residual needs a decoded field |
| B7 | live baseline `S = 0.7910689` / 508,639 flips | **VERIFIED_VIA_EMPIRICAL_ANCHOR** (`ddm_pu2` pointer row) |

**Denominator hygiene.** Both arms' headline percentages used stale gaps (0.7262358 = `dc1_fold`
era; 0.6543562 = `cx1` era) against a live 0.6189279. Re-anchored: `mf1` 13.66% → **16.03%**;
`sx2` 4.03% → **4.26%**; `sx2` shallow-Movable carrier → **12.24%**. The re-anchoring does *not*
explain the conflict (it is a common factor) but every one of these numbers was quoted against a
denominator that had already moved.

**Negative-existence scope.** I did **not** find any realization measurement on the displacement
axis in `ddm_sx1`, `ddm_sx2`, `ddm_mf1`, or their emitted JSON — that is the scope I searched,
and I make no claim beyond it.

---

## §7 WHAT IS OWED

1. ~~Contour curvature distribution — gates §3, §5.~~ **CLOSED in this memo (§3.1).**
   `κ_eff·L ≤ 0.503` for all `L ≤ 32 px`; `L = 32 px` admissible.
1b. **[NEW · scorer-free · cheapest now open]** Lane error decomposition into **position vs width
   vs existence**. §3.2 prices *position* at 453×; with a 2.51 px median width a ±1 px width error
   is a ±40% area error, and island-birth is a standing corpus concern. If Lane error is
   width-or-existence-dominated the 453× carrier addresses the wrong DOF. **This is now the
   highest-value open question in the memo** — it gates the best-priced description we own.
2. **[gates the whole axis · needs ONE scorer pass]** `mf1` §5.4 blocker 1, now load-bearing
   because §2 removed its support: decode the live vehicle, compute per-Movable-component `δ`
   against GT, report flips before/after **and the residual after the best rigid fit**. That
   residual is the number that decides between the rigid (2-DOF, object) and phase (1-DOF,
   segment) carriers. `ddm_pu2` holds the scorer slot; this is the request to queue behind it.
3. **[vehicle question, delegated]** Does TR1 expose a per-object spatial handle? See §8.
4. **Re-check the ancestor-scoped dimensionality claim** "intrinsic dim ~8 → Whitney 17–19;
   mod-16 under-embeds" against TR1. It is **not** verified on this vehicle and per `[[L18
   ancestor≠numbers]]` it does not transfer. I did not test it and do not rely on it anywhere
   above.

---

## §8 THE VEHICLE HANDLE

**STATUS: PENDING — delegated, not returned before this memo landed. Handed to MAIN.**

`mf1` §5.4 **blocker 2** — *"does TR1 expose a per-object handle?"* — is a **vehicle** question,
not a geometry question, and it **gates §3.2 entirely**. Every carrier priced in this memo assumes
the decoder can translate one component's contribution independently. If TR1's token grid is
spatial (token `[i,j]` → a fixed output patch) the handle exists by construction and the §3.2
carrier is buildable; if the tokens are global/entangled, **the 453× description is unrealizable
in this vehicle at any price** and the correct next move is a vehicle change, not a carrier.

I dispatched a read-only source investigation for: (1) where the live cx1/pu2/TR1 decoder source
is; (2) whether the token grid is spatial, quoted at source; (3) what the 535 B selector indexes;
(4) the "11 knobs". **I do not have its findings and I make no claim about them.** Nothing in
§3.2/§5 should be built on until this is answered at source.

**Negative-existence scope:** I did not myself search for the decoder source beyond two `grep`
passes over `src/tac/`, `experiments/`, `tools/` for `np.repeat`+`sigmoid` co-occurrence, which
returned only vendored/intake and unrelated hits. That is the only scope I personally searched; I
assert nothing about what exists outside it.

---

## §9 ROUND-1 ADVERSARIAL SELF-REVIEW

**R1 finding — material, in my own headline.** My first pass computed `flips = d_seg /
(100/PX)` and got **5,086**, a 100× error, because `d_seg` is a **rate** (`S_seg = 100·d_seg`)
and I inverted the score-per-flip constant instead of multiplying by `PX`. It was caught only
because the printed value failed the cross-check against `cx1`'s independently cited 508,639.
Fixed; every downstream number recomputed. **The lesson is the corpus's own:** the guard that
worked was carrying a redundant reference value into the same print, not the arithmetic.

**R1 finding — I nearly reproduced the equivocation I am criticising.** My first reading had
`sx2` "refuting" `sx1`. Re-reading `sx1:183` at source showed `sx1` had **printed the 18–20%
off-3px figure itself**. `sx2` and `sx1` do not disagree on depth at all; the disagreement is
narrower (reach) than either memo's framing suggests. Corrected in §1.3.

**R1 finding — MAIN's hypothesis, which I was told to test, is refuted.** "Ceiling vs measured
share" would require one arm to have measured realization. Neither did. Reporting this rather
than adopting the frame I was handed.

**R1 finding — a unit trap I nearly propagated.** `rz1`'s 0.60 bits/px and `sx1`'s 0.6984
bits/px look like a 16% conflict. They are different denominators (dilated vs strict band) and
different numerators (budget vs entropy). Flagged in §4 rather than averaged.

### R2 (after the coordinator's Lane question)

**R2 finding — MATERIAL, and it refutes my own §0 headline item.** R1 claimed *"both arms
over-count the displacement DOF by 2× in the dominant regime."* The §3.2 measurement shows the
gauge strength is `w/Λ`, which is **0.36 for Movable** — so **`mf1`'s 2-vector is correct and my
correction of it was wrong.** The halving is real but belongs to **Lane** (`w/Λ = 0.039`), a class
`mf1` never priced. I inferred a general claim from a first-order derivation and only the
measurement bounded its scope. §0 item 4 and §3 rewritten; the finding is now stronger *and*
narrower.

**R2 finding — the coordinator's `hs1` index trap does NOT apply here, but checking it found a
third mismatch.** Both arms' carriers share the component index (§1.4), so the ADD-vs-REMOVE trap
is excluded. But the check surfaced that `sx2`'s headline is a **sub-population size** and
`mf1`'s is a **carrier ceiling** — a category mismatch one level up from `hs1`'s. I would not have
looked for it without the prompt.

**R2 finding — a unit error I avoided only by checking the lattice.** The owed
`cell-mass × visible-fraction` discount looked applicable to my capture figures. It is not:
`lstars` is scorer-side, `D`'s null is camera-side (§3.3). Applying 0.77 would have silently
deflated every number in §3.1/§3.2/§5. **Same genus as the R1 `d_seg`-is-a-rate error and the
`rz1` 0.60-vs-0.6984 bits/px trap: three unit errors in one memo, two caught by cross-checks
rather than by care.**

**Instrument confidence.** Three independent exact reproductions of prior arms' numbers from
different code paths: boundary px **2,551,382** ≡ `sx1`; all five per-class component counts and
areas ≡ `mf1`; `sx1`'s H1 **253,341 B** ≡ to the digit. The measurements can be trusted more than
my interpretations of them — which is the pattern of both review rounds.

### R3 (after `ddm_rs2` `84367be88e` and the operator's class-interaction directive)

**R3 finding — an external refutation of a claim I had already marked as a careful correction.**
§3.3's *main* point (the `D`-null doesn't discount scorer-side figures) was right, and I flagged
applying 0.77 as a "unit error." Then I **immediately committed a different error in the next
paragraph** — applying the 22.6969% blind fraction as a *realization-width* discount on Lane. `rs2`
refutes it: `D∘U` is near-isometric on `range(U)`, so those pixels were never ours to lose. **The
pattern is the memo's own recurring one: I catch the unit error in the quantity I am examining and
introduce a new one in the sentence justifying the catch.** Struck in place at §3.3b, not rewritten.

**R3 finding — I repeated the exact failure mode I diagnosed in §1.** §1 convicts `mf1` of
inheriting `sx1`'s 23.3% without re-measuring. **§3.2 then priced a Lane carrier on `pc2`'s tb1
49.23%** — an inherited cross-vehicle share — when `pu2`'s cx1 tensor was on disk the whole time.
Measured: **46.23%**, so 453× → **425.8×**. The diagnosis did not immunise me against the disease.

**R3 finding — my best result today is a NEGATIVE, and its value is entirely relational.**
Frame-Gini 0.108 kills per-frame targeting for 99.6% of flip mass. Alone that is a dead end; set
against `hs1`'s static cell-Gini 0.8581 it becomes the *license* for the static carrier. **I would
have reported it as a dead end had the coordinator not named `hs1` in the same message.** Cross-arm
synthesis was not something I derived — it was handed to me, and I should say so.

**Instrument confidence — now four exact reproductions** from independent code paths: boundary px
2,551,382 ≡ `sx1`; all five per-class component censuses ≡ `mf1`; `sx1`'s H1 253,341 B to the digit;
total flips 508,640 ≡ the charter's 508,639 (1.000002). **The measurements remain more trustworthy
than my readings of them — the consistent finding of all three review rounds.**

**Not clean.** **R1: four material findings. R2: three. R3: three, one an external refutation of my
own paragraph and one a repeat of the failure I had just convicted another arm of.** Counter at
**0 of 3**; memo stands **PROVISIONAL-PENDING-VERIFICATION**.
Per the recursive protocol this memo is **PROVISIONAL-PENDING-VERIFICATION**, not sealed. The
settlement in §1 is the most robust part (it is pure arithmetic on two arms' emitted numbers and
reproduces to 1.0009); §3's gauge is derived but its validity condition is unmeasured; §5 is a
parameterised budget, not a result.
