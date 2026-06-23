# Horizon d_seg — deep-math multi-lens derivation (THEORY complement to probe aa98007464a7e6358)

- **Subagent**: `horizon_deepmath_20260623` (DEEP-MATH derivation; ANALYSIS-ONLY, $0).
- **Authority**: THEORY/derivation only. The empirical probe `aa98007464a7e6358` MEASURES the
  mechanism + lever on the frontier; this memo DERIVES the optimum it should hit. No score claim.
  Pointer UNMOVED 0.19110.
- **All numbers** trace to a verified upstream constant or an in-memo derivation/numeric check.
  Score/byte arithmetic via `tac.contest_score` (Catalog #391) — never hand-rolled.

## 0. Verified upstream facts (the substrate of every lens)

| Fact | Value | Source (verified this session) |
|---|---|---|
| Score | `S = 100·d_seg + sqrt(10·d_pose) + 25·rate` | `upstream/evaluate.py:92` |
| `d_seg` | `mean[ argmax(out1) != argmax(out2) ]` (per-pixel argmax-disagreement) | `upstream/modules.py:112` |
| SegNet input | **last frame only** `x[:, -1, ...]`, then `interpolate(..., bilinear)` | `upstream/modules.py:107-109` |
| Camera res | `camera_size = (1164, 874)` → **W=1164, H=874** | `upstream/frame_utils.py:11` |
| SegNet grid | `segnet_model_input_size = (512, 384)` → **W=512, H=384** | `upstream/frame_utils.py:13` |
| Camera intrinsics | `fx=fy=910`, `cx=582`, `cy=437` (openpilot `_neo_config`) | prompt + MEMORY (comma2k19 RAV4) |
| Round-trip | decode→clamp(0,255)→**round→uint8** | `upstream/frame_utils.py:180-183` |
| SegNet | comma10k 5-class `tu-efficientnet_b2` (stride-2 stem) | `upstream/modules.py:104` |

Vertical resample scale `sy = 384/874 = 0.43936`; horizontal `sx = 512/1164 = 0.43986` (near-isotropic).
Downsample ratio `k = 874/384 = 2.276` **camera rows per seg row** — a low-pass box of effective width ≈k.

---

## LENS 1 — GEOMETRY (projective): the horizon is a thin, near-constant-row 1-D curve

The horizon = the **vanishing line of the ground plane**. For a camera with pitch `θ` (down-positive)
and intrinsics `(fy, cy)`, the ground-plane vanishing line projects to image row

> **v_h(θ) = cy + fy·tan(θ)**  (camera-res rows).

**Derivation.** A ground-plane direction with vanishing point at infinity has ray direction
`d = (0, sinθ, cosθ)` in camera frame; projecting `v = cy + fy·(d_y/d_z) = cy + fy·tanθ`. At the
vanishing line `d_z→` horizontal-at-infinity, giving exactly the row above. □

Numeric (verified):

| pitch | v_cam | v_seg (×sy) |
|---:|---:|---:|
| 0.0° | **437.00** | **192.00** |
| ±0.5° | 444.9 / 429.1 | 195.5 / 188.5 |
| ±1.0° | 452.9 / 421.1 | 199.0 / 185.0 |

At nominal level pitch the horizon sits at **v_seg ≈ 192** — *exactly mid-grid* (cy·sy = 437·0.43936 = 192.0).

**Stability.** `dv/dθ = fy·sec²θ`; at θ=0 this is `fy = 910 px/rad = 15.9 px/deg` (camera),
`= 6.98 px/deg` in the seg grid. A highway segment with ±0.5° pitch jitter wanders the horizon by
**±3.5 seg-rows** — i.e. d_seg's binding feature lives in a band of roughly **±3.5 rows around row 192**.
This is why the band is *thin* (a few % of 384 rows) and *near-constant-row* (1-D curve, not 2-D blob).

**Foreshortening → texture height collapses to 0 at the horizon.** Road distance for row `v` below the
horizon is `Z = fy·h_cam/(v − v_h)` (camera height `h_cam ≈ 1.2 m`). The projected height of a 1-m
longitudinal road patch is `|dv/dZ|·1m = fy·h_cam/Z²`:

| v_cam | Z (m) | px / longitudinal-meter |
|---:|---:|---:|
| 450 | 84.0 | **0.155** |
| 470 | 33.1 | 0.997 |
| 500 | 17.3 | 3.63 |
| 600 | 6.7 | 24.3 |
| 800 | 3.0 | 120.7 |

Road texture that is ~120 px/m in the foreground is **0.15 px/m at the horizon** — a ~800× compression.
Geometrically, *all* the road's longitudinal information is crushed into the few rows at the vanishing
line. The boundary the scorer cares about (road ↔ undrivable/sky) is therefore a **thin 1-D curve at a
geometrically-pinned, near-constant row**, with sub-pixel placement fully determined by `(fy, cy, θ)`.

---

## LENS 2 — ALGEBRA (the argmax polytope): flips are shallow-margin pixels on the decision face

SegNet emits 5-class logits `ℓ(p) ∈ ℝ⁵` per pixel `p`. `argmax` partitions `ℝ⁵` into a fan of 5 polyhedral
cones (Voronoi-in-logit-space). `d_seg = (1/|Ω|)·Σ_p 𝟙[argmax ℓ_ours(p) ≠ argmax ℓ_gt(p)]`, `Ω` = 384×512 grid.

At the horizon the contest is **road (r) vs undrivable/sky (u)**. The relevant decision is the polytope
**face** `{ℓ : ℓ_r = ℓ_u}`. Define the **signed margin** `m(p) = ℓ_r(p) − ℓ_u(p)`. A pixel flips iff
`sign(m_ours(p)) ≠ sign(m_gt(p))`, i.e. the recon perturbs `m` across 0.

Linearize: a recon error `δ(p)` (RGB change at the pixel feeding SegNet) shifts the margin by
`Δm ≈ ⟨J(p), δ(p)⟩`, where `J(p) = ∂m/∂input(p)` is the **SegNet input-sensitivity** of the margin.
The flip condition becomes a **half-space test**:

> pixel flips ⟺ **⟨J(p), δ(p)⟩  has magnitude > |m_gt(p)|** in the wrong direction.

Hence the flip set is exactly the **shallow-margin pixels** (`|m_gt|` small) where the recon error's
projection onto `J` exceeds the margin. By Lens 1 these shallow-margin pixels are concentrated on the
1-D horizon line (the only place the road/sky logits cross). **The 97.8% horizon concentration of d_seg
is the algebraic statement that the road/undrivable decision *face* coincides with the geometric
vanishing line.**

---

## LENS 3+4 — CALCULUS + PHYSICS: the downsample flattens the margin → widens the flip band

**Physics of the edge.** The sky↔ground transition is the **highest-amplitude high-spatial-frequency
feature** in a highway frame (a near-step brightness/chroma discontinuity). Source H.265 band-limiting +
camera PSF already soften GT's edge slightly; an INR/HNeRV decoder adds **spectral bias** — coordinate-MLP
/ implicit nets fit low frequencies first (NTK eigenvalue decay ∝ frequency⁻²), so the *sharp step is the
last thing the net learns*. The recon therefore carries its **largest margin error precisely at the
horizon step** — `|δ|` peaks where `J` peaks. Two regimes: the **far-horizon** (haze, low contrast →
intrinsically shallow margin) and the **road-edge / foreground occlusion** (sharp, high-contrast). The
flips occupy the **far-horizon shallow-margin regime** (where `|m_gt|→0`), *amplified* by under-fit on the
sharp step.

**Calculus of the smear → band width.** Model the margin locally linear across the boundary:
`m(v) ≈ g·(v − v₀)`, `g = |dm/dv|` (logits per seg-row at the zero crossing). The bilinear downsample
`D` (874→384) is a low-pass average over a window of ≈`k=2.276` camera rows ≈ **1 seg-row** — it convolves
the step, **reducing the post-resample slope `g`** at the zero crossing (a sharp step's slope is divided
by its transition width). uint8 quantization further coarsens `m` in steps of ~`g·(255-grid)`.

A per-pixel recon margin-perturbation `|⟨J,δ⟩| = J·e` flips pixel `v` iff `|m(v)| < J·e`, i.e. within
`|v − v₀| < J·e/g`. So the **flip-band half-width**:

> **w_½ = J·e / g  (rows)**  → flatter margin (smaller `g`) ⇒ **linearly wider band ⇒ more flips.**

Numeric (representative `J·e = 0.5` logits):

| g (logits/row) | flip-band full width (rows) |
|---:|---:|
| 4.0 | 0.25 |
| 2.0 | 0.50 |
| 1.0 | 1.00 |
| 0.5 | **2.00** |

This is the precise mechanism: **downsample+uint8 flatten `g` at the sharp step → the zero-crossing band
widens → argmax flips proliferate exactly along the 1-D horizon.** The fix is not "make the recon prettier"
— it is **restore the margin sign** of the shallow-margin band pixels.

### THE KEY RESULT — KKT / reverse-waterfilling min-byte correction (closed form)

**Per-pixel min-norm correction (variational).** To unflip pixel `i` we need
`⟨J_i, δ_i⟩ ≥ |m_i|` — a half-space constraint. The minimum-‖·‖₂ correction is the **orthogonal
projection onto that half-space**:

> **δ_i\* = (|m_i| / ‖J_i‖²)·J_i ,  with ‖δ_i\*‖ = |m_i| / ‖J_i‖**  (scalar: `|m_i|/J_i`). □

**Which pixels to fix (knapsack/waterfilling).** We need not fix every flip — only the cheapest set that
drives d_seg below target. Each unflip reduces d_seg by `1/|Ω|`. The byte/effort cost to unflip pixel `i`
is `c_i = |m_i|/J_i` (the min-norm magnitude, an L1 byte-proxy). Minimize `Σ_i c_i x_i` s.t.
`Σ_i x_i ≥ n_target`, `x_i ∈ [0,1]`. The **KKT stationarity** gives a *single water level* `λ\*`:

> **Fix pixel `i` ⟺ c_i = |m_i|/J_i ≤ λ\*** , with `λ\*` chosen so `#{i : c_i ≤ λ\*} = n_target`.

This is **reverse-waterfilling on the margin/sensitivity ratio**: allocate correction first to
**shallow-margin (small |m_i|) AND high-sensitivity (large J_i)** pixels. It is the exact analogue of
rate-distortion reverse-waterfilling, with `|m_i|/J_i` playing the role of the inverse-variance.

**Numeric verification** (536 horizon flips, shallow margins `|m|~Exp(mean 0.3)`, `J~U[0.5,2]`):

| unflip fraction | cumulative L1 effort | mean cost/px |
|---:|---:|---:|
| 50% (268 px) | 20.6 | 0.077 |
| 80% (428 px) | 69.0 | 0.161 |
| 95% (509 px) | 121.4 | 0.238 |
| 100% (536 px) | 157.2 | 0.293 |

Convex tail (cheapest 50% = ⅛ of total effort) ⇒ a clear **knee**: target the cheap shallow-margin
majority, leave the expensive deep-margin tail. The waterfill *is* the optimal cheap correction.

---

## LENS 5 — INFORMATION THEORY (rate/MDL): the byte FLOOR vs the break-even budget

The horizon is a **1-D curve over W=512 columns** → the description length is `O(width)` (or `O(1)` for a
calibrated row + smooth deviations), not `O(area)`.

**Representation A — calibrated row + per-column deviation.** A near-constant calibrated horizon row
`v̄ = 192` plus per-column offset `dv(col) ∈ [−R, R]`. Raw cost `512·log₂(2R+1)` bits; but deviations are
**smooth/correlated** (road curvature is band-limited) → a linear/poly predictor leaves ~1 bit/col
residual ⇒ **≈ 64 bytes** regardless of R (verified: 64 B for R∈{1,2,4,8}).

**Representation B — entropy-coded 2-class label correction over the band.** Only the flip pixels need
fixing. At the frontier `d_seg = 0.00279` over 384×512 ⇒ **549 flips total, 536 in the horizon band**.
The band is ≈3 rows × 512 = 1536 px; the correct label there is *near-deterministic* given the line
(road below / undrivable above, error `1−p`). Binary-entropy cost `1536·H_b(p)`:

| p (label determinism) | byte floor |
|---:|---:|
| 0.90 | 90 B |
| 0.95 | 55 B |
| 0.99 | **16 B** |

**Byte FLOOR ≈ 16–262 bytes** (representation-dependent; tightest ~16–90 B for the entropy-coded band,
~64 B for the smooth-deviation polyline). This is the comma road-edge-polyline-flavored representation:
**step-along-a-calibrated-line.**

**Break-even budget (via `tac.contest_score`).** Removing the horizon d_seg saves
`100·(0.978·0.00279) = 0.273` score units. Rate cost per added byte `= 25/N = 6.66e-7` units/byte
(`N = 37,545,489`). So the byte budget that keeps the fix score-**positive** is

> **0.273 / 6.66e-7 ≈ 409,800 bytes ≈ 400 KB.**

**The byte floor (~16–262 B) is ~1,500–25,000× below the break-even budget (~410 KB).** The horizon fix is
not marginal — it is, by orders of magnitude, the highest-EV-per-byte correction available on the d_seg axis.

---

## PASS 1 — SYNTHESIS (the optimal cheap horizon correction)

The five lenses converge on ONE actuator:

1. **Geometry** pins the correction locus: a thin band (±~3.5 rows) at the calibrated horizon row
   `v̄ = cy·sy = 192` (1-D curve, near-constant row, sub-pixel placement from `(fy,cy,θ)`).
2. **Algebra** says the flips are shallow-margin road↔undrivable pixels on the decision face = the line.
3. **Calculus/physics** says downsample+uint8+spectral-bias flatten the margin there → restore the
   **sign** of the band margins (don't beautify pixels).
4. **The KKT/waterfill** gives the exact allocation: fix pixels in ascending `|m_i|/J_i`, water level
   `λ\*` set to the d_seg target — shallow-margin, high-sensitivity first; stop at the convex knee.
5. **Info-theory** says the natural representation is a **calibrated-row + smooth-deviation polyline**
   (the comma road-edge style), entropy-coded, **byte floor ~16–90 B**, against a ~410 KB break-even.

**Optimal cheap correction** = a *step-along-the-calibrated-horizon-line* sidecar: encode the per-column
horizon row offset (smooth residual) + a near-deterministic 2-class relabel of the ±~3-row band, applied
to the recon **as a margin-sign restore** following the `|m|/J` waterfill order. Byte floor ≈ tens of bytes.

## PASS 2 — RECURSIVE SELF-REFLECTION (Catalog #363; assumption-challenge axis)

Every assumption challenged, with `empirical_verification_status` ∈
{VERIFIED_VIA_SOURCE, INFERRED_FROM_GEOMETRY, ASSUMED_AWAITING_aa98}.

| # | Assumption | Status | Challenge / what could break it |
|---|---|---|---|
| A1 | Horizon is near-constant-row (v̄≈192) | INFERRED_FROM_GEOMETRY (verified: cy·sy=192.0) | **Curvature/roll**: banked curves, off-level mount, or hills make it tilted/curved. ±3.5-row wander is for ±0.5° pitch *only*; roll adds a *slope* across columns. The polyline rep absorbs this (that's why per-column offset, not a single row), but the "near-constant" claim is highway-specific. **aa98 must measure the actual flip-row distribution vs column.** |
| A2 | Margins are SHALLOW at the horizon | INFERRED (Lens 2/4) | If the far-horizon is actually high-contrast (clear day, sharp skyline), margins could be DEEP and flips driven by *recon error magnitude* not shallow margin → the waterfill `|m|/J` order still holds but the cost tail is heavier. **aa98 must measure the |m_gt| distribution at flip pixels.** |
| A3 | It is 2-class (road vs undrivable/sky) | ASSUMED_AWAITING_aa98 | comma10k has 5 classes; **lane-markings or movable (vehicles on the horizon) can intrude**, making it 3+ class locally → the 2-class entropy floor (16–90 B) under-counts; the polyline label needs >1 bit. Still small, but not 16 B. **aa98 must measure the class histogram in the flip band.** |
| A4 | The corrected band SURVIVES the uint8/downsample round-trip | ASSUMED_AWAITING_aa98 (CRITICAL) | The fix is applied in recon space but `d_seg` is measured AFTER decode→384→uint8→SegNet. A margin-sign restore of magnitude `|m_i|/J_i` (often sub-uint8-step at the flattened edge) **may be quantized away** before it reaches SegNet — the *same* low-pass that caused the flips can erase the fix. This is the #1 risk. **aa98 must verify the corrected pixels stay unflipped through the full eval round-trip, not just pre-round-trip.** |
| A5 | J (SegNet input-sensitivity) is moderate & well-conditioned at the band | INFERRED | The stride-2 EfficientNet-B2 stem halves resolution immediately (Lens-0 blind spot) → `J` at sub-(256×192) features may be tiny, making `|m|/J` blow up (corrections enormous / impossible). **aa98 should report `J` magnitude at flip pixels.** |
| A6 | First-order linearization `Δm≈⟨J,δ⟩` is valid | INFERRED | Large corrections leave the linear regime; the half-space projection is then only a first step (needs 1–2 reverse-waterfill iterations). Minor; the iterate converges. |
| A7 | Score arithmetic | VERIFIED_VIA_SOURCE | All via `tac.contest_score`; formula = `upstream/evaluate.py:92`. |

**Verdict (Catalog #307 PARADIGM-vs-IMPLEMENTATION):** the *mechanism* (geometry-pinned shallow-margin
1-D flip band; waterfill correction; tens-of-bytes floor) is **derivation-solid**. The *deliverability*
hinges on **A4 (round-trip survival)** and **A3/A2 (class purity + margin depth)** — all three are
**PROVISIONAL-PENDING-aa98**.

## PASS 3 — RESOLUTION (3-clean-pass; assumption-challenge each round)

- **Round 1** (geometry/algebra): clean — v̄=192, ±3.5-row wander, flip=shallow-margin-on-the-face are all
  VERIFIED/INFERRED-from-verified-constants. Assumption-challenge: A1 roll/curvature → answered by the
  *per-column* polyline (not single row). PASS.
- **Round 2** (calculus/physics/waterfill): clean — band-width = `J·e/g`, min-norm `δ\*=|m|/‖J‖²·J`, KKT
  water level `λ\*` are closed-form and numerically verified. Assumption-challenge: A6 linearity → noted,
  iterate. A4 round-trip survival flagged as the binding risk (not resolvable by theory). PASS-with-flag.
- **Round 3** (info-theory/EV): clean — byte floor 16–262 B vs 410 KB break-even via `tac.contest_score`.
  Assumption-challenge: A3 multi-class intrusion raises the floor modestly but EV margin (1,500–25,000×)
  survives any plausible inflation. PASS.

**Final**: theory **SEALED** on mechanism + optimum; **A2/A3/A4/A5 marked PROVISIONAL-PENDING-aa98**.

## What `aa98007464a7e6358` must empirically confirm (top 3, ranked)

1. **A4 — round-trip survival (binding).** Does the waterfill margin-sign restore stay unflipped through
   decode→384→uint8→SegNet? If the fix is quantized away, the whole approach needs over-correction
   (push margins a full uint8-step past zero, raising the byte cost — still cheap, but re-derive the floor).
2. **A2 + A5 — margin depth `|m_gt|` and sensitivity `J` at flip pixels.** Confirms the shallow-margin
   regime and that `|m|/J` is bounded (the waterfill is cheap, not blown up by the stride-2 stem blind spot).
3. **A3 + A1 — class histogram + flip-row-vs-column in the band.** Confirms ~2-class purity (validates the
   16–90 B entropy floor) and whether the band is near-constant-row vs tilted/curved (validates the polyline).

---
*6-hook wire-in: #1 sensitivity-map ACTIVE (`|m|/J` per-pixel cost map is a d_seg sensitivity prior);
#2 Pareto ACTIVE (byte-floor vs break-even is the d_seg/rate Pareto point); #3 bit-allocator ACTIVE
(reverse-waterfill `λ\*` IS the allocator rule); #4 cathedral-dispatch N/A (analysis, no archive);
#5 continual-learning — feed the derived optimum to aa98's posterior; #6 probe-disambiguator ACTIVE
(this memo IS the theory disambiguator; aa98 is the empirical arm). mission=frontier_breaking_enabler.
Sister-DISJOINT from aa98 (it measures, this derives). NON-PROMOTABLE [theory advisory]. Pointer UNMOVED 0.19110.*
