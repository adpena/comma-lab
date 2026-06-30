# SCREW / TWIST (se(3)/SE(3)) literature + OSS enrichment for the v2 witness vehicle

**UTC** 2026-06-30T22:00Z · **authority** `[research-signal / literature enrichment — advisory]` · **pointer UNMOVED 0.19110**
**score_claim** false · **promotable** false · **ready_for_exact_eval_dispatch** false
**Scope** CPU-only desk research (no GPU, no launch, no live-run touch). Survey of screw/twist theory across robotics,
SLAM/VO, graphics, simulation, control, neural/vision, and coding — distilled into ACTIONABLE enrichments + a
buildable MLX se(3)/SE(3) library spec for the v2 Stratified Screw-Warped Level-Set (S²WL) witness.

**means≠ends.** Everything below is a MEANS (it enriches understanding + engineering). The END is a byte-closed exact
row below 0.19110. NO-FAKE: every formula that is load-bearing is standard textbook math I've transcribed; the few I am
not 100% certain of (Barfoot Q-matrix in particular) are FLAGGED "verify by finite-difference before trusting." Every
literature claim carries a citation (paper + arXiv/DOI or repo URL). Applicability is tagged
**[MEASURED-APPLICABLE]** (we already have probe evidence) vs **[ASPIRATIONAL]** (geometrically grounded hypothesis,
unmeasured — must be validated through R before any verdict).

---

## 0. TL;DR — the five things to take away

**The headline deliverable is a buildable, parity-gated `tac` MLX se(3)/SE(3) library** (§1). The MLX ecosystem has NO
Lie-group primitives (jaxlie=JAX, Sophus/manif=C++, liegroups=numpy/torch) — building one is original, directly unlocks
a differentiable on-manifold ego-screw on our gradient device, and composes with our existing custom Metal kernels.

**Top-5 ranked actionable enrichments** (full table §4):
1. **MLX se(3)/SE(3) primitives library** (exp/log/Adjoint/J_r) — the foundation everything else needs. Effort M.
2. **Continuous-time SE(3) cumulative B-spline for ξ_ego(t)** (Sommer–Usenko) — collapse 600 per-pair poses into
   ~24–48 control floats + a C² smoothness prior. The biggest counted-byte win on the temporal factor. Effort M.
3. **Dual-quaternion screw-blend (DLB/ScLERP) at the per-class warp seam** — the annulus is exactly where d_seg is
   scored; a screw-blend across the class boundary removes the seam-tear a hard per-class warp creates. Likely
   top-EV-per-d_seg. Effort M. [ASPIRATIONAL — must measure through R.]
4. **SE(3) Lie-group symplectic/variational integration of the worldline exp(t·ξ)** — drift-free, deterministic
   worldline reconstruction; serves the deterministic-reproducibility non-negotiable. Effort S.
5. **Geometric trajectory-fit on SE(3) (exponential-coordinate optimization)** — fit the smoothest minimal-screw
   ξ_ego(t) to the PoseNet 6-vectors ON the manifold (vs an ad-hoc Euclidean spline); the principled
   canonicalize-to-ground-frame estimator. Effort M.

**MLX-port shortlist (what to implement natively, no MLX equivalent exists):** SO(3)/SE(3) `exp`/`log` (Rodrigues +
left-Jacobian V, with small-angle Taylor branches), `Adjoint`, right/left Jacobians `J_r`/`J_l` + inverses, hat/vee,
dual-quaternion `DLB`/`ScLERP`, and the SE(3) cumulative B-spline recursion. Reference oracle = numpy-fp32 (bit-identical
authority); cross-check vs jaxlie / spatialmath-python / pytransform3d / Sophus (all permissive licenses).

**Must-read refs:** Solà–Deray–Atchuthan "A micro Lie theory for state estimation in robotics" (arXiv 1812.01537) ·
Barfoot "State Estimation for Robotics" (textbook; the canonical J_l/Q-matrix source) · Sommer et al. "Efficient
Derivative Computation for Cumulative B-Splines on Lie Groups" (arXiv 1911.08860, CVPR 2020) · Kavan et al. "Geometric
Skinning with Approximate Dual Quaternion Blending" (ACM TOG 2008) · Park & Lynch "Modern Robotics" (product-of-
exponentials, screw Jacobian) · Pumarola/Park et al. "Nerfies" (arXiv 2011.12948, SE(3) per-point screw deformation
field).

---

## 1. ⭐ HEADLINE — the buildable `tac` MLX se(3)/SE(3) library spec (parity-gated)

> **Why this is the headline.** Our gradient device is MLX (M5 Max, MLX-first non-negotiable). MLX has no Lie-group
> primitives. The v2 witness needs a DIFFERENTIABLE, on-manifold ego-screw ξ_ego(t): to fit it to PoseNet, to warp the
> per-class partition, and to blend warps at boundaries — all with gradients flowing on MLX. This section is a
> complete spec the design-refine step can build from directly: exact numerically-stable formulas (with the
> small-angle branches), the numpy-fp32 reference oracle, MLX implementation notes, parity-test design,
> differentiability notes, and the Metal-acceleration decision per op.

### 1.0 Convention (fix once, test forever — this is a classic bug source)

- **Twist ordering:** `ξ = (ρ, ω) ∈ ℝ⁶`, **translation-first** (ρ = upper 3, ω = lower 3). This matches
  Solà micro-Lie, manif, and liegroups. **Lynch–Park / Murray–Li–Sastry use the OPPOSITE** screw ordering
  `𝒱 = (ω, v)` (rotation-first) — when porting any formula from those textbooks, swap the blocks. Pin the convention in
  a module docstring + a `CONVENTION` constant; assert it in every parity test.
- `θ = ‖ω‖`. `ω̂ = ω/θ`. `[·]_×` = skew/hat on ℝ³. Small-angle threshold `ε ≈ 1e-6` (fp32) / `1e-8` (fp64).

### 1.1 Core SO(3)

**hat / vee** (exact, trivial): `hat([a,b,c]) = [[0,-c,b],[c,0,-a],[-b,a,0]]`; `vee` = inverse.

**`exp_SO3(ω) → R`** (Rodrigues):
```
R = I + A·[ω]_× + B·[ω]_×²,   A = sinθ/θ,   B = (1-cosθ)/θ²
small-θ:  A → 1 - θ²/6 + θ⁴/120,   B → 1/2 - θ²/24 + θ⁴/720
```

**`log_SO3(R) → ω`**:
```
θ = arccos(clip((tr(R)-1)/2, -1, 1))
ω = (θ / (2 sinθ)) · vee(R - Rᵀ)
small-θ:  θ/(2 sinθ) → 1/2 + θ²/12 + 7θ⁴/720   (so ω ≈ ½·vee(R-Rᵀ) near 0)
θ→π guard: 2sinθ→0; use the eigen/diagonal branch (extract axis from R+I).  [Out of our regime — inter-frame
           ego-rotation is ≪π — but guard or raise rather than emit NaN.]
```

**`J_l_SO3(ω)` (left Jacobian = the SE(3) translation V-matrix):**
```
J_l = I + B·[ω]_× + C·[ω]_×²,   B = (1-cosθ)/θ²,   C = (θ-sinθ)/θ³
small-θ:  B → 1/2 - θ²/24 + θ⁴/720,   C → 1/6 - θ²/120 + θ⁴/5040
J_r(ω) = J_l(-ω)   (right Jacobian; flip sign of the linear term)
```

**`J_l_inv_SO3(ω)` and `J_r_inv_SO3(ω)`:**
```
J_l⁻¹ = I - ½[ω]_× + D·[ω]_×²
J_r⁻¹ = I + ½[ω]_× + D·[ω]_×²
D = 1/θ² - (1+cosθ)/(2θ sinθ)
small-θ:  D → 1/12 + θ²/720 + θ⁴/30240
```
(Source: Barfoot §7; Solà micro-Lie 1812.01537. The `(φ²)` coefficient `D` is identical for left/right; only the
linear-term sign flips. Confidence HIGH.)

### 1.2 Core SE(3)

**`exp_SE3(ξ=(ρ,ω)) → T=[[R, V·ρ],[0,1]]`** where `R = exp_SO3(ω)`, `V = J_l_SO3(ω)` (§1.1). The same Taylor branch in
`J_l` makes the translation block analytic at θ=0.

**`log_SE3(T=(R,t)) → ξ=(ρ,ω)`:** `ω = log_SO3(R)`; `ρ = J_l_SO3(ω)⁻¹ · t` (use `J_l_inv_SO3`, §1.1).

**`Adjoint(T=(R,t)) → 6×6`** (translation-first ordering — DERIVED + verified in this memo, §1.6):
```
Ad_T = [[ R,        [t]_× R ],
        [ 0,        R       ]]
```
(For rotation-first `(ω,v)` the block structure is `[[R,0],[[t]_×R, R]]` — do NOT mix.)

**`adjoint_se3(ξ=(ρ,ω)) → 6×6`** (the Lie-bracket matrix `ad`, needed for the full SE(3) Jacobian / BCH):
```
ad_ξ = [[ [ω]_×,  [ρ]_× ],
        [ 0,      [ω]_× ]]
```

**Full SE(3) left Jacobian 𝒥(ξ) (6×6)** — ONLY needed for exact Gauss–Newton on SE(3) directly (aspirational; our
B-spline path doesn't need it):
```
𝒥(ξ) = [[ J_l(ω),  Q(ρ,ω) ],
        [ 0,        J_l(ω) ]]
```
where `Q` is the **Barfoot Q-matrix** (a degree-(ρ,ω) polynomial in [ρ]_×,[ω]_× with the
`(θ-sinθ)/θ³`, `(1-θ²/2-cosθ)/θ⁴`, `(θ-sinθ-θ³/6)/θ⁵` coefficients). **⚠️ FLAG — transcribe Q carefully from Barfoot
Eq. (7.86) and VERIFY by finite-difference against `log_SE3`; it is the single most error-prone formula here.** For our
witness we can defer Q entirely (the B-spline path uses only exp/log/Adjoint/J_r), so Q is OPTIONAL/aspirational.

### 1.3 Screw-blend layer (the per-class-warp-seam fix)

**Dual quaternion `q̂ = q_r + ε q_d`:** `q_r` = unit rotation quaternion, `q_d = ½·(0,t)⊗q_r` (t the translation vector
as a pure quaternion). Conversions `SE3↔DQ` are algebraic + branch-free except the `θ→π` quaternion-from-R edge (same
guard as log_SO3).

**`DLB(weights w_i, dual-quats q̂_i) → q̂` (Kavan 2008 approximate screw-blend):**
```
1. antipodality fix: for each i, if dot(q_{r,i}, q_{r,0}) < 0 → negate q̂_i  (pick the short-arc hemisphere)
2. b̂ = Σ_i w_i · q̂_i                          (linear combo of dual quaternions)
3. normalize: q̂ = b̂ / ‖b̂_r‖   (divide BOTH parts by the real-part norm — keeps it a valid unit DQ to 1st order)
```
This is the *approximate* dual-quaternion linear blend — smooth, coordinate-invariant, GPU/Metal-friendly, no seam
tear. For the EXACT 2-transform screw-interpolation use **ScLERP**: `q̂(u) = q̂_A ⊗ (q̂_A* ⊗ q̂_B)^u` via the
dual-quaternion power (screw `exp`/`log` on the dual angle). pytransform3d's `dual_quaternion_sclerp` (BSD-3) is the
reference oracle. (Sources: Kavan–Collins–Žára–O'Sullivan ACM TOG 2008; pytransform3d docs.)

**Application to the witness:** at a pixel near a class boundary, instead of a hard `regime(class)` switch (current
probes), assign soft membership weights `w_c(pixel)` (from SegNet softmax or signed distance to the boundary) and
`DLB`-blend the per-class warps `{T_Road, T_sky, T_hood, …}`. The warp field becomes continuous across the annulus →
no discontinuity-induced spurious argmax flips on the exact pixels d_seg scores. **[ASPIRATIONAL]** — geometrically the
right tool, but UNMEASURED through R; the current screw probes used hard regime + persist-fallback, so whether the
blend reduces the through-R d_seg on the annulus is an open measurement (see §4 item 3).

### 1.4 Continuous-time ξ_ego(t): cumulative SE(3) B-spline (Sommer–Usenko)

Represent the whole-clip ego trajectory as a uniform cubic (k=4) cumulative B-spline on SE(3) with control poses
`{T_i}`. For local time `u ∈ [0,1)` in the segment governed by `{T_i, T_{i+1}, T_{i+2}, T_{i+3}}`:
```
T(u) = T_i · ∏_{j=1}^{3} exp( B̃_j(u) · Ω_j ),   Ω_j = log_SE3( T_{i+j-1}⁻¹ · T_{i+j} ) ∈ se(3)
cumulative cubic basis (uniform):
  B̃_1(u) = (5 + 3u - 3u² +  u³)/6
  B̃_2(u) = (1 + 3u + 3u² - 2u³)/6
  B̃_3(u) = (u³)/6
```
(Cumulative blending matrix M̃ = rows-summed-from-bottom of the uniform cubic B-spline matrix; Kim–Kim–Shin 1995 for
SO(3), generalized to Lie groups by Sommer et al.) **Velocity / time-derivative:** because each factor is a
one-parameter subgroup in `u` with *constant* twist `Ω_j`, `d/du exp(B̃_j Ω_j) = B̃_j'(u)·Ω_j` acts cleanly — Sommer
et al.'s O(k) recurrence. **For us, autodiff handles this:** the spline pose is a product of `exp` of constant twists
scaled by scalar bases, so MLX autodiff through `exp` gives both `dT/du` (for d_pose velocity consistency) AND the
control-point gradients — **no custom VJP for the spline layer; only `exp`/`log` need stable implementations.**

**Byte win:** 600 per-pair 6-vectors (3600 floats) → a cubic spline with ~8–16 control poses ≈ **48–96 floats** for the
whole clip + a C² smoothness prior that regularizes the d_pose stream. (Counts as COUNTED video-derived payload; the
spline-eval algorithm is FREE/rule-118 generic in inflate.py.) See §4 item 2.

### 1.5 Differentiability notes (the autodiff gotchas — concrete)

- **The `where(θ<ε, taylor, exact)` NaN-gradient trap.** `mx.where` evaluates BOTH branches; the exact branch contains
  `1/θ`, `1/sinθ`, `1/θ³`, which produce `inf/NaN` at θ=0, and `0·NaN = NaN` poisons the gradient even though the value
  is selected from the Taylor branch. **Fix (mandatory):** clamp inside the exact branch — compute the exact branch
  with `θ_safe = maximum(θ, ε)` so it never divides by exactly 0; the `where` then selects the Taylor value at small θ
  and a finite (unused) number from the exact branch, and the gradient stays finite. (Known JAX/MLX idiom; document it
  in-code.)
- **`exp` needs no custom VJP** once the Taylor branches + θ-clamp are in place — it is analytic at 0; MLX autodiff is
  correct and cheap.
- **`log` is smooth in our regime** (inter-frame ego-rotation ≪ π). The θ→π branch cut is OUT of regime; guard it
  (clamp or raise) — a custom VJP would only be needed if we ever optimize near 180° rotations (we don't).
- **`DLB` is smooth** except the antipodality `if dot<0 negate` (a sign flip at the hemisphere boundary) — measure-zero,
  but use `sign(dot)` multiplication (not Python `if`) so it's vectorized + the gradient is defined a.e.
- MLX exposes `mx.custom_function` (we already use it — §1.7) if any op later needs a hand-written VJP; the spec above
  predicts none of the CORE ops do, which keeps the build small.

### 1.6 Parity-test design (numpy-fp32 = bit-identical authority)

Per the deterministic-reproducibility non-negotiable, the **numpy-fp32 reference is the verdict authority; MLX must
match it (parity ≥ 0.9997 / max|Δ| under tol).** Test battery:
1. **Sampling:** N random twists with `θ` log-spaced over `[1e-9, π-1e-3]` (deliberately hammer the small-angle
   branch) + random ρ; also the exact-zero twist.
2. **Three-tier oracle:** numpy-float64 "golden" → numpy-float32 "authority" → MLX "fast path." Assert
   `max|MLX - numpy_fp32| < 1e-5` and `max|numpy_fp32 - numpy_fp64| < 1e-4` (fp32 expected drift).
3. **Algebraic identities (device-agnostic):** `exp(log(T)) == T`; `log(exp(ξ)) == ξ`; `Ad_{AB} == Ad_A·Ad_B`;
   `J_r(ξ)·J_r_inv(ξ) == I`; `exp(ξ)⁻¹ == exp(-ξ)`; `T·exp(ξ)·T⁻¹ == exp(Ad_T·ξ)` (the adjoint defining identity —
   this independently re-verifies the §1.2 Adjoint block structure).
4. **Jacobian finite-difference:** analytic `J_r`/`Q` vs central finite-difference of `log` — the canonical Q-matrix
   verification (§1.2 flag).
5. **External cross-check (CI-optional, dev-only):** vs **jaxlie** (MIT), **spatialmath-python** (MIT),
   **pytransform3d** (BSD-3), **liegroups** (MIT), **Sophus** via `sophuspy` (MIT/BSD), and **scipy.spatial.transform**'s
   new `RigidTransform` SE(3) support (BSD; see arXiv 2511.18157). Same random twists through both → max|Δ|. These are
   sanity cross-checks, NOT the authority (numpy-fp32 is).

### 1.7 Metal-acceleration analysis (profile-then-accelerate — DO NOT premature-Metal)

We already ship custom Metal via `@mx.custom_function` + `mx.fast.metal_kernel` with a numpy reference oracle +
forward/VJP + per-chip parity guard (`src/tac/local_acceleration/metal_fused_r_operator.py` ships
`fused_r_forward_numpy`/`fused_r_vjp_numpy`; `metal_grouped_conv_backward.py` is the grouped-conv exemplar). Classify
each Lie op:

| Op class | Scale | Verdict |
|---|---|---|
| `exp`/`log`/`Adjoint`/`J_r` on a few transforms (the ego-twist, control poses) | O(10²) tiny mats | **MLX-Python is fine.** Negligible cost; do NOT Metal-ize. |
| Per-class SE(3) warp applied to the **pixel grid** (384×512×N-pairs) | O(10⁸) pixel-ops | **HOT candidate** — profile first. |
| SE(3) B-spline eval + dual-quat screw-blend **per-pixel along the annulus** | O(annulus·N) | **HOT candidate** — profile first. |

**Build order discipline:** build the MLX-Python library with the numpy-fp32 parity oracle FIRST, profile the warp/blend
hot path, and Metal-accelerate ONLY the profiled-worthwhile kernel (the per-pixel warp/blend), following the fused-R
pattern: forward kernel + hand-written VJP kernel + the per-chip parity guard (cf. MLX issue #2205 numerical-variation
across chips — the fused-R kernel already documents this). The warp itself is mostly a grid-sample (homography per
class) which may already be fast via MLX `grid_sample`-equivalent gather; the screw-blend per-pixel is the more likely
genuine Metal candidate. **NO premature Metal** — the tiny Lie ops never warrant it.

### 1.8 Licensing / derivation posture for the library

- **The Lie math is standard textbook** (Rodrigues, V/J_l, J_r, J_l⁻¹, Adjoint, ad, BCH, cumulative B-spline basis,
  DLB). Free to implement from Barfoot (textbook), Solà micro-Lie (arXiv 1812.01537; manif is MIT), Lynch–Park
  (textbook), Sommer et al. (arXiv 1911.08860), Kavan 2008 (ACM TOG). No license encumbrance on the formulas.
- **Reference repos to cross-check / borrow idioms (all permissive):** jaxlie (MIT), liegroups (MIT), spatialmath-python
  (MIT), pytransform3d (BSD-3), Sophus (MIT/BSD), dqrobotics (LGPL — link-only, do NOT copy code), pytransform3d's
  `dual_quaternion_sclerp` (BSD-3). **⚠️ basalt** (the Sommer/Usenko B-spline impl) is **MPL-2.0** (weak file-level
  copyleft) — implement the B-spline FRESH from the paper formulas (§1.4) and merely cross-check numerically; do NOT
  copy basalt source. **⚠️ dqrobotics is LGPL** — implement DLB fresh from Kavan, don't copy.
- **Our-original = the composition** (MLX/Metal-native differentiable Lie + dual-quat + SE(3) B-spline, parity-gated to
  numpy-fp32) — see §6.

---

## 2. Survey — canonical references organized by domain (cited)

### 2.1 Foundations (19th–20th c. kinematics + Lie theory)
- **Chasles' theorem (1830):** every rigid displacement = a screw (rotation about + translation along one axis). The
  native statement of "ego-motion = one twist." (Historical; standard.)
- **Ball, "A Treatise on the Theory of Screws" (1876):** the canonical screw-theory text; twist (motion) ↔ wrench
  (force) duality. The native language of the ego-motion temporal factor.
- **Plücker line coordinates (1865):** the screw AXIS is a line in ℙ³; Plücker coords give a coordinate-free 6-vector.
- **Lie groups/algebras se(3)/SE(3), exp/log/adjoint/right-Jacobian, twists & wrenches:**
  **Solà, Deray, Atchuthan, "A micro Lie theory for state estimation in robotics" (arXiv 1812.01537)** — THE practical
  intro; ⊕/⊖ operators, J_l/J_r, the on-manifold chain rule. **Barfoot, "State Estimation for Robotics" (Cambridge,
  2017)** — the canonical J_l/Q-matrix source.
- **Dual quaternions / motor algebra / PGA:** **Selig, "Geometric Fundamentals of Robotics"**; **Gunn/Dorst PGA**
  ("Projective geometric algebra: a new framework for doing Euclidean geometry," arXiv 1901.05873) — the even subalgebra
  of 3D PGA ≅ biquaternions ≅ screw algebra; motors interpolate smoothly. (See §3 OSS for Klein.)

### 2.2 Robotics (product-of-exponentials, screw Jacobian)
- **Lynch & Park, "Modern Robotics: Mechanics, Planning, and Control" (Cambridge, 2017)** — product-of-exponentials
  (space & body frame), the spatial/body screw Jacobian, twists/wrenches; rotation-first `(ω,v)` convention (mind §1.0).
- **Murray, Li, Sastry, "A Mathematical Introduction to Robotic Manipulation" (1994)** — the foundational PoE +
  matrix-exponential-of-a-twist treatment.
- **Pinocchio** (Carpentier et al.) — SOTA rigid-body dynamics; analytical derivatives on Lie groups (§3).

### 2.3 SLAM / VO / continuous-time trajectories (closest analog to ξ_ego(t))
- **Sommer, Usenko, Schubert, Demmel, Cremers, "Efficient Derivative Computation for Cumulative B-Splines on Lie
  Groups" (arXiv 1911.08860, CVPR 2020)** — the O(k) recurrence for cumulative SE(3) B-spline derivatives; the basis of
  basalt. **THE reference for our temporal factor.**
- **Continuous-time state estimation survey (arXiv 2411.03951, 2024)** — broad map of spline + Gaussian-process
  continuous-time trajectories in robotics.
- **Mueggler et al., "Continuous-Time Visual-Inertial Odometry for Event Cameras" (TRO 2018)** — SE(3) spline VIO.
- **IMU preintegration (Forster et al., "On-Manifold Preintegration," TRO 2017)** — the on-manifold integration of
  inter-frame motion increments; the J_r appears exactly in the bias/covariance propagation (relevant to how we
  accumulate ξ between scored frames). (Not surfaced verbatim in search, but canonical; flag for direct read.)
- **Ego-motion / deep-VO** — what PoseNet does: a learned 6-DOF ego-motion regressed from image pairs. (Our PoseNet
  6-vector = a learned twist estimate; §1 fits a spline to it.)

### 2.4 Neural / vision (SE(3) deformation fields + equivariance)
- **Park et al., "Nerfies: Deformable Neural Radiance Fields" (arXiv 2011.12948)** — encodes a per-point rigid
  transform as a **screw axis S=(r;v)∈ℝ⁶** (exact se(3)!), MLP predicts (r,v) per point, + an **elastic
  regularization** penalizing deviation from rigidity. Direct prior art for "warp the canonical field by a screw" — and
  for the rigidity regularizer we could apply to the per-class warp.
- **D-NeRF / NSFF / Deformable-3DGS (arXiv 2309.13101) / 4DGS (2310.08528)** — per-primitive SE(3) motion fields; how
  to parameterize + regularize a time-varying screw field.
- **Deng et al., "Vector Neurons: A General Framework for SO(3)-Equivariant Networks" (ICCV 2021, arXiv 2104.12229)** —
  lightweight SO(3)-equivariance by lifting scalar neurons to 3-vectors; **SE(3)-equivariant Vector Neurons** (arXiv
  2204.01159). Relevance: if the residual INR should be equivariant to the ego-screw (so the learned lane-residual
  doesn't have to relearn every viewpoint), VN is the cheapest equivariant primitive. [ASPIRATIONAL.]
- **Frame Averaging (Puny et al., arXiv 2110.03336)** — the canonicalization-for-accuracy framework our
  gauge-as-codec-canonicalization (RATE-instead-of-accuracy) repurposes (already in our originality accounting).

### 2.5 Other domains + coding (screw as the minimal-rate motion descriptor)
- **Parametric/global motion compensation in video coding** — VVC's 4-/6-parameter **affine** motion model and the
  6-parameter **perspective** model ("An efficient six-parameter perspective motion model for VVC,"
  ScienceDirect S104732032200061X; "Perspective Affine Motion Compensation for VVC," Springer / GitHub smu-ivpl/PAMC).
  Establishes the rate-coding pedigree: a COMPACT parametric motion field (6–8 params) beats dense per-pixel flow for
  global motion. Our screw is the *physically-correct* 6-DOF parametric model — a strict specialization that throws
  away nothing the ego-motion contains.
- **SF2SE3 (arXiv 2209.08532)** — cluster dense scene flow into a few SE(3) motions (rigid-body proposals + selection).
  Directly relevant to GAP-1 (decompose movables into per-object twists). **RAFT-3D (arXiv 2012.00726)** — dense
  pixelwise SE(3) field + rigid-motion embeddings (soft grouping into rigid objects). The learned analog of our
  per-class regime + movable residual.
- **LieFlow (arXiv 2602.21645) / GNVC-VD / OT-NFM** — flow-matching on Lie algebra for video dynamics (already in our
  borrowed-substrate accounting as the generator family).
- **Quotienting camera kinematics for video stabilization (arXiv 1903.09073)** — quotient out the ego SE(3) path
  (≈ our canonicalize-to-ground-frame, applied to stabilization rather than coding).

### 2.6 Graphics (screw-blending — the seam fix) + simulation + control [coordinator-added]
- **GRAPHICS:** **Kavan, Collins, Žára, O'Sullivan, "Geometric Skinning with Approximate Dual Quaternion Blending"
  (ACM TOG 27(4):105, 2008)** + the 2007 I3D "Skinning with Dual Quaternions." The thesis verbatim: *"the problems of
  linear blending do not stem from incorrect rigging, but from incorrect blending"* — i.e. **a naive (linear) blend of
  rigid transforms collapses/tears; a dual-quaternion (screw) blend doesn't.** This is EXACTLY our per-class-warp-seam
  problem (§1.3). Also **Dual-Quaternion Interpolation (arXiv 2303.13395)** for the math; camera-path SE(3) splines
  (keyframe interpolation) reuse §1.4.
- **SIMULATION (deterministic screw integration):** **Lee, Leok, McClamroch, "A Lie Group Variational Integrator for
  Rigid Body Motion in SE(3)" (IEEE CDC; arXiv-adjacent)** + **Leok, "Lie group variational integrators" (overview)** —
  symplectic, momentum-preserving, **chart-free** integration that stays exactly on SE(3) (no reprojection drift). The
  structure-preserving way to reconstruct the worldline exp(t·ξ) deterministically. **Featherstone spatial algebra**
  (the 6-D spatial-vector formulation underlying MuJoCo/Drake/Bullet). **Dual-Quaternion Variational Integrator (arXiv
  1611.00616)** — same, in DQ form.
- **CONTROL (optimal screw-trajectory fit):** **Lee, Leok, McClamroch, "Geometric Tracking Control of a Quadrotor UAV
  on SE(3)" (CDC 2010; mathweb.ucsd.edu/~mleok/pdf/LeLeMc2010_quadrotor.pdf)** — the canonical
  control-on-SE(3)-avoids-chart-singularities lineage. **Teng et al., "Constrained Trajectory Optimization on Matrix
  Lie Groups" (arXiv 2301.02018)** — Lie-algebraic trajectory optimization with constraints; the principled
  on-manifold spline-fit (vs Euclidean least-squares on raw poses). **"Exponentially Stable First Order Control on
  Matrix Lie Groups" (arXiv 2004.00239)** — exponential-coordinate control/estimation.

---

## 3. OSS table (what we can DRAW FROM / PORT)

| Library | License | Provides | MLX/numpy-portability |
|---|---|---|---|
| **jaxlie** (brentyi) | MIT | SO2/3, SE2/3 exp/log/adjoint/apply/multiply/inverse, AD-friendly | JAX (not MLX). **Closest design template** — mirror its API in MLX; cross-check oracle. |
| **liegroups** (utiasSTARS) | MIT | SO2/3, SE2/3 in **numpy** + pytorch; exp/log/adj/J_l | numpy path = a near-drop-in reference oracle; port idioms to MLX. |
| **spatialmath-python** (Corke / bdaiinstitute) | MIT | SO/SE(2/3), **twist + product-of-exp**, adjoint `tr2adjoint`, closed-form log; numpy | numpy reference oracle + screw/twist utilities. |
| **pytransform3d** (DFKI) | BSD-3 | SE(3) exp/log (Lynch–Park se(3) convention), **`dual_quaternion_sclerp`**, screw axis; numpy | numpy reference oracle for SE(3) **and ScLERP** (the only listed lib with ScLERP built-in). |
| **Sophus** (`sophuspy`) | MIT/BSD | C++ SO/SE(2/3); Python bindings | C++; use `sophuspy` only as an external cross-check, not a port target. |
| **scipy.spatial.transform** | BSD-3 | `Rotation` + new **`RigidTransform`** SE(3) (arXiv 2511.18157, framework-agnostic, differentiable) | numpy/array-API; cross-check + possibly a thin oracle. |
| **manif** (artivis) | MIT | C++ micro-Lie (the 1812.01537 companion) | C++; conceptual template (the ⊕/⊖ API). |
| **Pinocchio** | BSD-2 | rigid-body dynamics + analytical Lie derivatives | C++/Python; heavy; reference for analytic Jacobians only. |
| **dqrobotics** | **LGPL** | dual-quaternion robot modeling/control | **link-only / don't copy** (LGPL); implement DLB fresh from Kavan. |
| **basalt** (Usenko) | **MPL-2.0** | the SE(3) cumulative B-spline impl (Sommer et al.) | **weak copyleft — implement fresh from arXiv 1911.08860**, cross-check numerically only. |
| **Klein** (jeremyong) | MIT | 3D PGA motors (SIMD); motor interpolation | C++/SIMD; PGA-motor reference if we go the PGA route (§4 item 8). |
| **gafro** (arXiv 2310.19090) | (check) | geometric algebra for robotics | reference for GA-motor robotics idioms. |
| **pymanopt** | BSD-3 | optimization on manifolds (incl. SO/SE) | optimizer reference for on-manifold trajectory-fit (§4 item 5). |

**Bottom line:** NO MLX-native Lie library exists. Port the math fresh (textbook), use **liegroups(numpy)** +
**pytransform3d** + **spatialmath-python** + **jaxlie** as the parity oracles, and avoid copying LGPL/MPL code.

---

## 4. RANKED actionable enrichments to OUR v2 engineering (highest-EV first)

| # | Enrichment | v2 application | Expected benefit | Effort | Tag |
|---|---|---|---|---|---|
| **1** | **MLX se(3)/SE(3) primitives** (§1.1–1.2): exp/log/Adjoint/J_r + small-angle branches, parity-gated to numpy-fp32 | The foundation: differentiable on-manifold ego-screw on our gradient device; everything below depends on it | Unblocks fitting/warping/blending the twist with MLX gradients; composes with our Metal kernels | **M** | MEASURED-APPLICABLE (the screw probes already used these formulas in numpy; this makes them differentiable + MLX) |
| **2** | **Cumulative SE(3) B-spline for ξ_ego(t)** (§1.4, Sommer–Usenko) | Replace 600 independent per-pair poses with ~8–16 control poses (~48–96 floats) + C² smoothness prior | Biggest COUNTED-byte win on the temporal factor; smooths/regularizes d_pose; autodiff-clean (no custom VJP) | **M** | ASPIRATIONAL (byte/d_pose win unmeasured; spline-fit residual to PoseNet must be checked) |
| **3** | **Dual-quat screw-blend (DLB/ScLERP) at the per-class warp seam** (§1.3, Kavan 2008) | Soft-blend per-class warps across the class boundary = the **annulus where d_seg is scored**; removes the hard-regime seam-tear | Directly targets the scored annulus; likely **top-EV-per-d_seg** if it reduces boundary flips through R | **M** | ASPIRATIONAL — must measure through R (current probes used hard regime; this is the untested upgrade) |
| **4** | **SE(3) Lie-group variational/symplectic integration of exp(t·ξ)** (§2.6 sim) | Drift-free, chart-free deterministic reconstruction of the worldline from control twists | Serves deterministic-reproducibility non-negotiable; no reprojection drift over the clip | **S** | MEASURED-APPLICABLE for determinism (exp/log are exact one-parameter subgroups; integrator is standard) |
| **5** | **Geometric trajectory-fit on SE(3)** (exponential-coordinate optimization, §2.6 control; arXiv 2301.02018) | Fit the smoothest minimal-screw ξ_ego(t) to PoseNet 6-vectors ON the manifold (vs Euclidean spline); the canonicalize-to-ground-frame estimator | Lower residual + fewer control points than naive Euclidean fitting; principled smoothness | **M** | ASPIRATIONAL (fit-quality vs Euclidean unmeasured) |
| **6** | **Nerfies elastic/rigidity regularizer** on the per-class warp (§2.4, arXiv 2011.12948) | Penalize the residual INR's deviation from the screw-predicted rigid warp | Keeps the learned residual small (it only carries non-rigid lane/movable parts) → fewer COUNTED bytes | **S** | ASPIRATIONAL |
| **7** | **SF2SE3 / RAFT-3D per-object twist decomposition for movables** (§2.5) | GAP-1: decompose independently-moving objects into a few extra per-object SE(3) twists | Captures the genuine movable residual the ego-screw can't (measured ~0.0005 d_seg) at a few streams | **M** | ASPIRATIONAL (the screw probes flagged movables as the genuine residual) |
| **8** | **PGA motors / dual-quaternion representation** for the warp algebra (§2.1, Klein) | Branch-free, numerically-robust composition/interpolation of the per-class warps (vs 4×4 matrices) | Robustness + speed; unifies rotation+translation+blend in one algebra | **M** | ASPIRATIONAL (engineering nicety; matrix form works) |
| **9** | **SE(3)-equivariant residual INR (Vector Neurons)** (§2.4, arXiv 2104.12229) | Make the lane-residual net equivariant to the ego-screw so it generalizes across viewpoints | Smaller/more-data-efficient residual; better generalization (the "generalize the implementation" directive) | **L** | ASPIRATIONAL (research-grade; only if residual capacity becomes the bottleneck) |
| **10** | **6-param parametric-motion rate pedigree** (§2.5, VVC affine/perspective) | Frames the screw as the physically-correct compact global-motion model; informs the rate accounting | Justifies the rate-optimality argument (6-DOF ≪ dense flow) in the codec writeup | **S** | MEASURED-APPLICABLE (rate argument; standard coding result) |

**The decisive ordering note:** #1 is the prerequisite for #2/#3/#5. #3 (boundary screw-blend) is the highest-EV-per-d_seg
*hypothesis* because it touches the scored annulus directly — but it is ASPIRATIONAL until measured through R, and the
existing screw-through-R probe (`tools/measure_screw_warp_through_R.py`) is the natural harness to extend (add the
soft-membership DLB blend and re-measure the annulus d_seg). Build #1, then immediately test #3 on that harness.

---

## 5. MLX-port shortlist (concrete — what to implement, no MLX equivalent exists)

Build order (each parity-gated to numpy-fp32 before the next):
1. **`tac/geometry/lie_mlx.py` core:** `hat/vee`, `exp_SO3`/`log_SO3`, `J_l_SO3`/`J_r_SO3` + inverses (with small-angle
   Taylor + θ-clamp, §1.1); `exp_SE3`/`log_SE3`, `Adjoint`, `adjoint_se3` (§1.2). numpy-fp32 oracle in the same module
   (`*_numpy`), parity test (§1.6). [Q-matrix optional/flagged.]
2. **`tac/geometry/dual_quat_mlx.py`:** `se3↔dq`, `DLB`, `ScLERP` (§1.3). numpy oracle cross-checked vs pytransform3d
   `dual_quaternion_sclerp`.
3. **`tac/geometry/se3_spline_mlx.py`:** cumulative cubic B-spline `T(u)` + autodiff velocity (§1.4). numpy oracle
   cross-checked vs the Sommer arXiv example (implemented fresh; basalt is MPL — don't copy).
4. **Profile** the per-pixel warp/blend; **Metal-accelerate** only if profiled-worthwhile (§1.7), following
   `metal_fused_r_operator.py` (fwd + VJP + per-chip parity guard).

---

## 6. Contribution / give-back framing (a durable MEANS that compounds)

Genuinely novel + OSS/science-worthy (honest borrowed-substrate accounting, NO-FAKE #7 — *composition* novel,
*primitives* borrowed):

- **(a) An MLX/Metal-native differentiable Lie-group library** (se(3)/SE(3) + dual-quaternion + SE(3) cumulative
  B-spline, parity-gated to a numpy-fp32 oracle, optional Metal warp kernel). **The MLX ecosystem lacks one**
  (jaxlie=JAX, Sophus/manif=C++, liegroups=numpy/torch). The *math* is standard textbook (no novelty claim there); the
  *contribution* is the first MLX/Metal-native, autodiff-clean, parity-gated implementation — directly useful to the
  Apple-silicon ML community. **License posture: Apache-2.0 or MIT** (permissive give-back; matches our cross-check
  deps). Cleanly separable from contest IP (pure geometry).
- **(b) The screw-as-task-space-motion-codec paradigm** — compressing a frozen-scorer level-set witness by a *physical
  ego-screw* (indirect-RD / coding-for-machines, the unoccupied A∩B intersection per our originality accounting). This
  is the *research* contribution (a paper/method), not a library. **Honest claim: a novel COMPOSITION of known prior
  art** (screw theory + level-set + indirect-RD + canonicalization-for-MDL) — every primitive is borrowed; the binding
  is ours. **UNVALIDATED until a byte-closed exact row beats 0.19110** — do not frame as an achievement until measured.

Both are MEANS that compound the 10-year program (a reusable MLX-Lie substrate; a publishable task-space-codec method);
neither moves the pointer. The pointer moves only on the exact row.

---

## 7. Wire-in hooks (Catalog #125) + NO-FAKE caveats

1. **sensitivity-map** ACTIVE — the boundary screw-blend (#3) adds a per-annulus warp-continuity sensitivity row.
2. **Pareto** ACTIVE — #2 (spline control-point count ↔ d_pose) and #3 (blend ↔ annulus d_seg) are measured rate↔distortion arms once probed.
3. **bit-allocator** ACTIVE — #2 reallocates the temporal-factor budget (600 poses → ~12 control poses); #7 adds per-object movable twists.
4. **cathedral autopilot** N/A (literature enrichment; not archive-deployable yet).
5. **continual-learning** this memo + the DAG FEED; updates the v2 originality accounting (§6 confirms the give-back posture).
6. **probe-disambiguator** — `tools/measure_screw_warp_through_R.py` is the existing disambiguator; the named next probe (DLB-blend the annulus + re-measure through R) is the #3 disambiguator.

**NO-FAKE caveats (binding):**
- All §1 formulas except the Barfoot Q-matrix are standard textbook math transcribed at HIGH confidence; **Q is FLAGGED
  — verify by finite-difference before trusting**, and it is OPTIONAL for the witness path.
- Items #2/#3/#5/#6/#7/#8/#9 are **ASPIRATIONAL** — geometrically grounded but UNMEASURED through R. No d_seg/d_pose/byte
  claim is real until a through-R or byte-closed measurement lands. The screw-warp d_seg numbers cited in §0/§2 are from
  prior `[macOS advisory / research-signal]` probes (PRE-R label-space and one through-R CPU-torch run), NOT contest
  exact rows.
- This is a literature/engineering enrichment (a MEANS). **Pointer 0.19110 UNMOVED.** The END is a byte-closed exact
  row below it (the lane-survival GPU run).

## 8. Primary citations
- Solà, Deray, Atchuthan. *A micro Lie theory for state estimation in robotics.* arXiv 1812.01537.
- Barfoot. *State Estimation for Robotics.* Cambridge, 2017 (J_l / Q-matrix canonical source).
- Sommer, Usenko, Schubert, Demmel, Cremers. *Efficient Derivative Computation for Cumulative B-Splines on Lie Groups.* CVPR 2020, arXiv 1911.08860.
- Kim, Kim, Shin. *A general construction scheme for unit quaternion curves with simple high order derivatives.* SIGGRAPH 1995 (cumulative B-spline on SO(3)).
- Kavan, Collins, Žára, O'Sullivan. *Geometric Skinning with Approximate Dual Quaternion Blending.* ACM TOG 27(4):105, 2008 (+ I3D 2007 *Skinning with Dual Quaternions*).
- Lynch, Park. *Modern Robotics.* Cambridge, 2017 (product-of-exponentials, screw Jacobian). Murray, Li, Sastry. *A Mathematical Introduction to Robotic Manipulation.* 1994.
- Park, Sinha, Barron, et al. *Nerfies: Deformable Neural Radiance Fields.* arXiv 2011.12948 (SE(3) screw deformation field + elastic regularizer).
- Lee, Leok, McClamroch. *A Lie Group Variational Integrator for Rigid Body Motion in SE(3)* + *Geometric Tracking Control of a Quadrotor UAV on SE(3)* (CDC 2010).
- Teng et al. *Constrained Trajectory Optimization on Matrix Lie Groups.* arXiv 2301.02018.
- Deng et al. *Vector Neurons: A General Framework for SO(3)-Equivariant Networks.* ICCV 2021, arXiv 2104.12229 (+ SE(3) VN arXiv 2204.01159).
- Teed, Deng. *RAFT-3D: Scene Flow using Rigid-Motion Embeddings.* arXiv 2012.00726. SF2SE3, arXiv 2209.08532.
- Gunn, Dorst, et al. *Projective geometric algebra.* arXiv 1901.05873; Klein PGA library (github.com/jeremyong/klein).
- VVC parametric motion: *An efficient six-parameter perspective motion model for VVC* (Elsevier S104732032200061X); *Perspective Affine Motion Compensation for VVC* (Springer / github smu-ivpl/PAMC).
- OSS: jaxlie (github.com/brentyi/jaxlie), liegroups (github.com/utiasSTARS/liegroups), spatialmath-python (github.com/bdaiinstitute/spatialmath-python), pytransform3d (dfki-ric.github.io/pytransform3d), Sophus, manif (github.com/artivis/manif), Pinocchio, dqrobotics, pymanopt. scipy.spatial.transform RigidTransform: arXiv 2511.18157.
- Internal grounding: `screw_twist_warp_dseg_probe_20260629T192609Z.md`, `screw_warp_through_R_gap2_20260629T195829Z.md`, `project_gr_unified_action_full_witness_architecture_20260629.md`, `project_v2_novel_contribution_originality_accounting_20260629.md`.
