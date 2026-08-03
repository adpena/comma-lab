# ddm_pb3 — the blind-set actuator is a SIX-dimensional problem, and bp2 solved it in 692,712 dimensions

**UTC** 2026-08-03 · **arm** `ddm_pb3_parametric_blind_set` · **axis** `[macOS-CPU advisory]`
frozen CPU-torch scorers on decoded camera rasters — **NOT** `upstream/evaluate.py` on an archive.
`score_claim=false`, `promotion_eligible=false`. **Pointer UNMOVED.**

**Vehicle:** `/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_pb2_bestof_archive.zip`,
360,339 B, sha256 `6e1b80e90109edd3c06f29fdfe37937dfb78eac7480c4e65adbc364a10e1e764` — the same
archive bp2 measured, so the two arms price the same object.

**Denominator (every ΔS below is against it):** total gap to the PR130 demonstrated floor
**0.7262358**; 1% of the gap = **10,907 B** *(MAIN correction 2026-08-03: this arm used 0.7263025 /
10,908 B, which predate `ddm_na1`'s PR130 byte fix — the floor is 191,052 B, not 190,952. The
shift is 6.7e-5 S and changes no verdict here; every ΔS below is unaffected at quoted precision.)*
(`tac.canonical_equations.gap_decomposition_against_floor_20260802`).

---

## The answer first

1. **bp2's failure has a mechanism, and it is a wrong-coordinates failure — MEASURED from bp2's
   own n600 receipts, no new scorer time.** `d_pose = ‖e‖²/6` with `e` a **6-vector**. bp2's
   pixel-space step direction is only **2.7%–5.9% aligned** with `-e`, so **>99.6% of its
   enormous reach lands in the five pose dimensions that were already correct.** That is why it
   overshoots at k≈400 of 692,712 coordinates, and why its median per-pair gain is **0.198%**.
2. **The channel's linearized ceiling is enormous: ΔS_pose = −0.22775.** Falsification-tested —
   **0 violations in 6,000 (pair,arm) cells**, on 278 pairs where the bound is non-vacuous. The
   blind set alone has enough 1-LSB reach to more than cover the **entire 0.2120 pose gap**.
3. **The parametric shape moves the price by 95.5×.** bp2 shipped 401,285 B of signs
   (ΔS_rate **+0.2672**). Seven scalars per pair is **4,200 B** (ΔS_rate **+0.0028**).
   Break-even falls from "reduce d_pose 99.89%" to **"reduce mean d_pose 2.01%"**.
4. **And the measured realization does not reach it.** See §5 — the decisive arm is the
   d_pose *actually re-scored through the real receiver*, not a first-order statistic.

---

## 1. The reframe (DERIVED from bp2's receipts; the load-bearing derivation)

`d_pose = MSE(PoseNet(ours)[:6], PoseNet(gt)[:6]) = ‖e‖²/6`. A blind-set perturbation `δ` acts
through `J = ∂p/∂δ` (6 × 692,712):

```
d_pose(δ) = ‖ e + Jδ ‖² / 6
```

**`rank(J) ≤ 6`.** No perturbation family can use more than six output dimensions regardless of
how many of the 692,712 coordinates it touches. Steepest descent in the pixel space is not a
descent method for this problem — it is a descent method for a scalar whose curvature it ignores.

bp2's step is `δ_k = -sign(g)` on the top-k by `|g|`, with `g = (1/3)Jᵀe`, and its `k` is chosen so
the cumulative gradient mass predicts a first-order decrease of `t·d_pose`. Under the quadratic,
that pins two of the three unknowns exactly:

```
⟨e, Jδ⟩ = -(t/2)‖e‖²        d_pose(δ)/d_pose(0) = 1 - t + ρ²,  ρ = ‖Jδ‖/‖e‖
⇒  cos(e, Jδ) = -(t/2)/ρ
```

so `cos` is **recoverable per pair, per step size, from numbers bp2 already recorded**:

| grad-mass target t | mean k | ρ = ‖Jδ‖/‖e‖ | **cos(e, Jδ)** | pairs model-consistent |
|---:|---:|---:|---:|---:|
| 0.002 | 6 | 0.0371 | **−0.02698** | 404 |
| 0.01 | 43 | 0.1253 | −0.03991 | 503 |
| 0.05 | 397 | 0.5554 | −0.04502 | 543 |
| 0.15 | 2,315 | 1.5519 | −0.04833 | 576 |
| 0.35 | 19,330 | 3.4035 | −0.05142 | 600 |
| 0.7 | 96,172 | 6.4639 | −0.05415 | 600 |
| 1.0 | 181,337 | 8.5021 | **−0.05882** | 600 |

**`cos` drifts by only 2.2× across 4.5 decades of k.** That stability is the evidence the
quadratic model describes the data — the model has one free parameter per (pair, t), so *fitting*
is trivial; a recovered constant across four and a half decades is not.

Two consequences fall straight out:

- **The best a scaled version of bp2's own direction can ever do is `1 - cos² ≈ 0.1%–0.35%`
  reduction.** bp2's MEASURED median per-pair reduction is **0.198%**. The derivation predicts
  the measurement it was derived from a different way round; that is the cross-check.
- **bp2's headline 65.9% mean reduction is a tail artifact, not a typical gain.** The top 1% of
  pairs (6 of 600) carry **62.1%** of its total d_pose reduction; the top 5% carry **92.4%**.

## 2. The ceiling — what ANY 1-LSB blind perturbation can reach (MEASURED n600, bound)

`Z = {Jδ : ‖δ‖∞ ≤ 1}` is symmetric and convex, so for any unit `v`,
`min_{z∈Z} ‖e+z‖ ≥ ⟨v,e⟩ - h_Z(v)`. Taking `v = e/‖e‖` gives `h_Z(v) = ‖(Jᵀe)|blind‖₁/‖e‖ = 3·g1/‖e‖`
where `g1` is exactly bp2's recorded `grad_blind_l1`. With `‖e‖² = 6d`:

```
d_floor / d  ≥  (1 - g1/(2d))²      when g1 < 2d,   else 0
```

| | n600 |
|---|---:|
| mean d_pose base | 0.00764247 |
| mean d_pose, bp2 per-pair argmin (MEASURED) | 0.00260760 |
| **mean d_pose, linearized floor** | **0.00023713** |
| pairs where full cancellation is not excluded (γ ≥ 1) | 53.7% |
| pose contribution base → floor | 0.276450 → **0.048696** |
| **ΔS_pose ceiling** | **−0.22775** |

**Falsification test (the reason this is a bound and not a hope):** no measured arm may sit below
the floor. Across **6,000 (pair, arm) cells** — seven top-k arms plus full-descent, full-ascent
and the random-sign control, on all 600 pairs — **violations: 0**, with the floor non-vacuous on
**278** pairs. Landed as a regression test against the live receipt.

Scope: valid under linearization of `J`. It is *not* contradicted by any measurement we have, and
it is derived from a straight-through gradient while every arm it is checked against went through
the real rounding receiver.

## 3. The parametrization, derived rather than chosen

Every weight in the composition `D∘(a·M)` is **non-negative** — `D` is bilinear downsampling
(`1-frac`, `frac`; `ddm_ll1_window_solve._bilinear_taps`), `M` is bilinear resampling times a
rolling-shutter row blend `α ∈ [0,1]` (`pfs1_warp_receiver.warp_rgb` + `v4d_pair_taps`). So for
`a > 0` the **sign** of `g` at a blind pixel is the sign of the backprop image `h = Jᵀ_pose e` at
that pixel's warped location. And `h = Σᵢ eᵢ·(∂pᵢ/∂f0)` lives in a **6-dimensional span**. Hence

> **`δ* = -sign( a 6-coefficient combination of six image fields, pulled back through the warp
> the receiver already builds ).`**

That is the shape. **Six coefficients + one density (the scale knob) = 7 scalars/pair.** DOF is
*justified*: six because `rank(J) ≤ 6`; one more because the ∞-norm box needs a scale. More DOF
buy nothing in the linearized problem.

**The receiver-computable basis (rule-118 FREE).** `∂pᵢ/∂f0` needs PoseNet weights, which never
ship. The direct-VO ansatz supplies a generic stand-in: `ψᵢ(x) = ∇I(x)·Lᵢ(x)`, `L` the standard
2×6 interaction matrix at normalized image coordinates, inverse depth from the ground plane the
v4d selector already carries. **Only the SPAN has to be right**, because the encoder fits the six
coefficients freely — any invertible mixing (the estimator's unknown Hessian inverse) is absorbed
exactly by the fit. Inputs: the render the receiver makes, frozen intrinsics, the shipped plane.
Zero counted bytes.

## 4. The price — this is what the parametric shape actually buys

| | bp2 (per-coordinate) | **pb3 (parametric)** |
|---|---:|---:|
| payload | 401,285 B (signs only, index free) | **4,200 B** (7 scalars × 8 bits × 600) |
| ΔS_rate | **+0.26720** | **+0.00280** |
| mean d_pose reduction to break even | 99.89% | **2.01%** |
| ratio | — | **95.5× cheaper** |

Break-even in ceiling-capture terms: **η = 0.0047**, where `η` is the fraction of the maximum
first-order descent `‖g_blind‖₁` a field realizes (`η = 1` is the unconstrained optimum
`-sign(g)`, `η = 0` inert).

Because the ceiling gain is tail-concentrated, a **subset** payload is cheaper still — 30 pairs
plus a colex index is **231 B** and covers 83.9% of the ceiling gain. The payload is not the
binding constraint anywhere in this family.

**Substituting `η` into the floor is an UPPER BOUND on the gain, never a prediction** — it assumes
the realized `Jδ` is antiparallel to `e`, i.e. it assumes away exactly the orthogonal waste that
killed bp2. (Formally: for `γ = η·g1/(2d) ≤ 1` and `‖Jδ‖ ≥ ‖e‖` — measured `ρ = 8.5` — one has
`d(1-γ)² ≤ d(1-cos²θ)`, so the capture form lower-bounds the achievable d_pose.) The number that
decides the arm is therefore §5, not this table.

## 5. The decisive measurement — d_pose re-scored through the real receiver

**NOT TAKEN. The arm was terminated by a provider weekly usage limit mid-flight (2026-08-03
~04:12Z, last checkpoint step=2 `harvest alignment_n16 applied measurement; write memo`).**

This section is **OWED, not empty-because-null**. Nothing here is a verdict. §1–§4 stand on their
own — they are derivations and bounds re-derived from bp2's landed n600 receipts plus one
scorer-free stratified alignment pass — but **the arm's own framing says §5 is the number that
decides it**, and §5 does not exist. Do not promote, price, or compose this family on §1–§4 alone.

**Exactly what is owed:** the realized `d_pose` of the 7-scalar parametric field, re-scored
**through the real receiver + composite-R + uint8 + frozen PoseNet**, n600, against the same
vehicle SHA `6e1b80e9…e764`. The artifact `reports/ddm_pb3/alignment_n16_stratified.json` is the
harvested stratified alignment input; `reports/ddm_pb3/ceiling_n600.json` is the landed §2 bound
with its 0-violation falsification test.

**The falsifier is already pre-registered and must not be softened on resume:** the family pays
iff realized capture `η ≥ 0.0047`. §4's capture arithmetic is an **UPPER BOUND, never a
prediction** — it assumes away exactly the orthogonal waste that killed bp2 (measured `ρ = 8.5`,
`cos ≈ −0.03…−0.06`). A resumed arm that reports the bound as the result would be repeating bp2's
error one level up.

## 6. Verdict scope and what is owed

**VERDICT: NONE. `verdict_scope: incomplete_arm`.** Pointer UNMOVED.

What is MEASURED and durable regardless of §5:

1. **bp2's failure has a mechanism** — wrong coordinates, not wrong idea. `rank(J) ≤ 6`; bp2's
   pixel-space direction is 2.7%–5.9% aligned with `-e`, so >99.6% of its reach lands in pose
   dimensions that were already correct. Derived from bp2's own receipts, no new scorer time.
2. **bp2's headline 65.9% is a TAIL ARTIFACT** — top 1% of pairs (6 of 600) carry 62.1% of the
   total reduction; top 5% carry 92.4%. The median per-pair gain is 0.198%. Any future arm quoting
   a mean over this distribution is quoting the tail.
3. **The ceiling is real and falsification-tested** — 0 violations in 6,000 (pair,arm) cells,
   non-vacuous on 278 pairs. Landed as a regression test against the live receipt.
4. **The price collapses 95.5×** — 401,285 B → 4,200 B, ΔS_rate +0.26720 → +0.00280, break-even
   from "reduce d_pose 99.89%" to "reduce mean d_pose 2.01%". **This is the operator's 2026-08-02
   directive measured**: the pose win does not have to be bought with rate; the actuator's byte
   cost is a DESIGN variable, and bp2 paid for a shape, not for physics.
5. **And the index compacts further** — 30 pairs + a **colex index** is **231 B** for 83.9% of the
   ceiling gain. The combinatorial number system hits `log₂ C(n,k)` exactly; this is a solved
   problem, and our own precedent is PR101's 3-byte `SIDECAR_NOOP_INFER_RANK_LEN`. **The payload is
   not the binding constraint anywhere in this family** — which is why §5, the realized capture, is
   the only thing left that can decide it.

---

*STORES CONSULTED:* `ddm_bp2_blind_set_pose_actuator_20260802.md` + all 8 receipts under
`reports/ddm_bp2/` (the direct input; every number in §1–§2 is re-derived from
`reach_n600.jsonl`, not recalled), `ddm_cv1_seven_surface_convocation_20260802.md` §3/§12/§13
(the third-currency framing and directive D2 that made this arm P0),
`ddm_ll1_window_solve.py` (the blind mask and the disjoint-2×2 geometry),
`pfs1_warp_receiver.py` + `inflate_runner_v4d.py` (the vehicle; weight non-negativity verified at
source), `upstream/modules.py` (scorer authority),
`tac.canonical_equations.gap_decomposition_against_floor_20260802` (the denominator).
