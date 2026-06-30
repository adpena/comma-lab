# Factorized 4D fields, se(3) screw factorization, intrinsic-dimensionality — SOTA/OSS brief for the witness level-set field φ(x,y,t,class)

`[macOS research-signal / advisory only]` · pointer 0.19110 UNMOVED · $0 (online+CPU) · NO GPU · 2026-06-30T18:07:53Z

> NO-FAKE: every arXiv id + github repo below is real and verified against web sources at the URLs cited; uncertainty is flagged inline. This is an advisory research brief — no score claim, no exact-eval row. It feeds (i) the n600 v2 codec **rate-half spatial factor** and (ii) **compressed observability** of the partition manifold (measured eff-dim ~26 @ n600, off-pose residual ~83%).

## 0. The object we are factoring (frame)

Our witness is a 4D implicit field **φ(x, y, t, class) → ℝ** (signed-distance / level-set form), over a single driving clip. Its `argmax_class` (or sign structure) is the SegNet partition; inter-class boundaries are codim-1 **1D curves**; lane markings are thin 1D ridges (dashed = birth/death pairs in the Morse–Smale persistence diagram). Already in-tree: per-pair code SVD/participation-ratio, the Morse–Smale persistence codec (topo rate floor ~6/23 dashes/frame, Lane PH⁰-dim 0.83), the curvelet directional basis (the −48% d_seg lever), the se(3) screw-warp (Road causal lever, GREEN through R). Corroborating: partition-manifold eff-dim **~26** (held-out generalization probe), off-pose residual **~83%**.

The factorization question: write φ as a low-token product/sum of axis factors so that (a) the manifold's dimension + structure is *observable* in compressed form, and (b) the spatial factor is a counted-byte-cheap rate term for v2.

---

## A. FACTORIZED NEURAL FIELDS (how each factors a multi-D field; fit to our 4D (x,y,t,class) SDF)

The common move: replace a dense d-D feature grid (storage O(N^d)) with a **sum of low-rank axis factors** (storage O(d·N) for vectors, O(d·N^2) for planes). All are *linear* tensor factorizations followed by a tiny shared decoder. Ranked by relevance to our field at the end (§Synthesis).

| Method | arXiv | repo | factorization | storage | fit to φ(x,y,t,class) |
|---|---|---|---|---|---|
| **TensoRF** (Chen et al., ECCV'22) | [2203.09517](https://arxiv.org/abs/2203.09517) | [apchenstu/TensoRF](https://github.com/apchenstu/TensoRF) | **CP**: sum of rank-1 outer products of 3 vectors (v_x⊗v_y⊗v_z); **VM**: sum of vector⊗matrix (v_x⊗M_yz + …) | CP <4MB, VM <75MB (3D) | CP is the *most token-efficient* 3D form; VM gives the quality knob. Direct template for "v_t ⊗ M_xy" temporal split (see §Synthesis). |
| **K-Planes** (Fridovich-Keil & Meanti et al., CVPR'23) | [2301.10241](https://arxiv.org/abs/2301.10241) | [sarafridov/K-Planes](https://github.com/sarafridov/K-Planes) | **(d choose 2) planes**: for d=4 → 6 planes {xy,xz,yz,xt,yt,zt}; features **multiplied** (Hadamard) across planes; linear decoder w/ learned color basis | 1000× vs dense 4D grid | **Strongest direct fit.** White-box, arbitrary-d, *induces a natural static/dynamic split* (space-only planes vs time-coupled planes) — exactly our static-field × motion separation. d=3 here is (x,y,t); class is a 4th axis or a per-class plane bank. |
| **HexPlane** (Cao & Johnson, CVPR'23) | [2301.09632](https://arxiv.org/abs/2301.09632) | [caoang327.github.io/HexPlane](https://caoang327.github.io/HexPlane/) | 6 planes for spacetime: 3 space {xy,xz,yz} + 3 space-time {xt,yt,zt}; features **fused** (concat/multiply) + tiny MLP | 100× faster train vs dynamic NeRF | Twin of K-Planes (parallel discovery). The 3-space/3-time partition is the canonical observability split: "what is static (xy)" vs "what moves (xt,yt)". |
| **Tensor4D** (Shao et al., CVPR'23) | [2211.11610](https://arxiv.org/abs/2211.11610) | [DSaurus/Tensor4D](https://github.com/DSaurus/Tensor4D) | hierarchical 4D: 4D tensor → 3 time-aware volumes → **9 compact feature planes** | monocular/sparse-view capable | Heavier (9 planes); its hierarchical time-aware-volume idea matters only if our motion is non-rigid+high-rank. Our motion is ~1 screw → overkill; keep as upper bound. |
| **Factor Fields / DiF** (Chen et al., SIGGRAPH'23) | [2302.01226](https://arxiv.org/abs/2302.01226) | [autonomousvision/factor-fields](https://github.com/autonomousvision/factor-fields) | **unifying formula**: signal = Σ basis(coeff-transformed coords) × coeff-field; subsumes NeRF/Plenoxels/EG3D/Instant-NGP/TensoRF; **Dictionary Fields (DiF)** = learned shared basis + per-location coefficients | best compactness for SDF in-paper | **The meta-framework to phrase our whole codec in.** DiF's "learned dictionary basis + sparse coefficients" is the exact shape of our curvelet-basis + per-frame-coefficient rate split. Best reported geometry quality for SDF among fast methods. |
| **tri-plane / EG3D** (Chan et al., CVPR'22) | [2112.07945](https://arxiv.org/abs/2112.07945) | [NVlabs/eg3d](https://github.com/NVlabs/eg3d) | 3 axis-aligned planes {xy,xz,yz}, features summed → tiny MLP; the canonical 3D hybrid explicit-implicit | photoreal 3D GAN | The ancestor of all plane factorizations. For us: a single (x,y) plane + per-class channel is the degenerate "1-plane" tri-plane = our static spatial factor. |
| **DeepSDF** (Park et al., CVPR'19) | [1901.05103](https://arxiv.org/abs/1901.05103) | [facebookresearch/DeepSDF](https://github.com/facebookresearch/DeepSDF) | **auto-decoder**: one MLP φ(z, x)→sdf shared across a *class* of shapes; per-shape **256-dim latent code z** optimized at inference; PCA/dictionary structure emerges in z-space | entire shape *class* in one net + tiny per-shape code | **This is our manifold-observability primitive.** A per-frame (or per-pair) latent code z_t indexing one shared partition-decoder → the z-space *is* the partition manifold; its PCA/eff-dim is directly our measured ~26. The rate-half then ships only {z_t}. |
| **Canonical Factors for Hybrid Neural Fields** (Tang et al., ICCV'23) | [2308.15461](https://arxiv.org/abs/2308.15461) | [brentyi/canonical-factors](https://github.com/brentyi/canonical-factors) (verify) | adds learned canonicalizing transforms to remove *axis-aligned bias* of plane factorizations | quality bump on planes | Relevant because our boundaries are *directional* (curvelet lever): canonicalizing the plane axes to the lane-tangent frame is the learned version of "orient the basis to the boundary." |

**SDF-specific note.** Factor Fields/DiF reports the best geometry (SDF) quality among the fast factorized methods; DeepSDF is the latent-space (dictionary-of-SDFs) primitive. For a *piecewise-constant argmax* target (not a smooth surface), the plane factorizations' linear+tiny-MLP decoder has the same Gibbs/spectral-bias risk we already measured against HNeRV — so the factorization buys *token efficiency of the carrier*, while the **step/curvelet basis** still has to supply the topology-matched decoder. The two compose: K-plane factors → curvelet/step decoder.

---

## B. 4D SPACETIME FACTORIZATION (temporal-axis token efficiency)

The question: how does each represent the **time axis**, and how cheap is the temporal factor vs our se(3) screw worldline?

| Method | arXiv | repo | temporal factor | token cost of time |
|---|---|---|---|---|
| **4D-GS** (Wu et al., CVPR'24) | [2310.08528](https://arxiv.org/abs/2310.08528) | [hustvl/4DGaussians](https://github.com/hustvl/4DGaussians) | canonical 3D Gaussians + a **HexPlane-encoded deformation field** + tiny MLP predicting per-Gaussian (dx, dr, ds) at time t | O(HexPlane planes) shared; deformation is *learned*, not parametric. Motion model is generic (any deformation). |
| **4D-GS (native 4D)** (Yang et al., ICLR'24) | [2310.10642](https://arxiv.org/abs/2310.10642) | [fudan-zvg/4d-gaussian-splatting](https://github.com/fudan-zvg/4d-gaussian-splatting) | **native 4D Gaussians**: 4D mean+4D covariance (a 4D rotor); marginalizing t gives a time-conditioned 3D Gaussian | each Gaussian carries its own (t-center, t-scale, 4D rotation) — *explicit* but per-primitive; rigid global motion is NOT factored out. |
| **K-Planes (d=4)** | [2301.10241](https://arxiv.org/abs/2301.10241) | [sarafridov/K-Planes](https://github.com/sarafridov/K-Planes) | the 3 time-coupled planes {xt, yt, zt}; the 3 space planes {xy,xz,yz} are the static field. Time-smoothness prior on time planes. | O(N·T) per time plane (×3). Static/dynamic split is *free* (which planes a feature lives on). |
| **HexPlane** | [2301.09632](https://arxiv.org/abs/2301.09632) | [caoang327.github.io/HexPlane](https://caoang327.github.io/HexPlane/) | 3 space-time planes {xt,yt,zt} fused with 3 space planes | same O(N·T) as K-Planes; the canonical "spacetime tri-plane." |
| **Tensor4D** | [2211.11610](https://arxiv.org/abs/2211.11610) | [DSaurus/Tensor4D](https://github.com/DSaurus/Tensor4D) | 3 time-aware volumes → 9 planes; hierarchical | highest time cost; only needed for high-rank non-rigid motion. |

**The se(3) screw temporal factor beats all of the above for OUR motion.** Our scene is (per the GROK-confirmed vehicle frame) **one rigid trajectory of the ego through a near-static world** + a few movables. For rigid motion, the worldline of any point is the *geodesic* `g(t) = exp(t·ξ)·g(0)` with **a single twist ξ∈se(3)≅ℝ⁶** (constant-velocity) or a low-order spline of twists. That is **~6 numbers for the entire temporal axis**, versus O(3·N·T) plane features. The plane methods *learn* an unconstrained deformation because they cannot assume rigidity; we *can* assume it (it is the contest's physical truth), so we collapse the temporal factor to its Lie-algebra generator. This is the single biggest token win available on the t-axis, and it is already GREEN through R on the Road class.

> Transfer: replace the {xt,yt,zt} time-planes of a K-Plane/HexPlane factorization with the parametric screw worldline exp(t·ξ). The static field stays a learned (x,y)-plane/curvelet factor; the temporal axis becomes a 6-number generator + per-class depth (§C). Movables get their own residual twists (a handful more).

---

## C. se(3) SCREW / TWIST LIE-FACTORIZATION (the one-twist × per-class-depth warp)

Foundations (canonical, not arXiv-bound):
- **Chasles' theorem**: any rigid 3D displacement = a rotation about an axis + a translation along it = a **screw motion**. ([Modern Robotics §3.3](https://modernrobotics.northwestern.edu/), Lynch & Park — the canonical text.)
- **Twist** ξ = (ω, v) ∈ se(3) ≅ ℝ⁶ is the Lie-algebra generator; **exp(ξ̂) ∈ SE(3)** is the rigid motion; a *constant* twist traces a smooth exponential curve (geodesic worldline) in SE(3).
- **Plücker coordinates** (4 numbers) define the screw axis (a line in 3D); pitch sets rotation-vs-translation mix. ([Mecharithm screw tutorial](https://mecharithm.com/learning/lesson/screws-a-geometric-description-of-twists-in-robotics-9).)
- Lie-algebra gradient-descent / state-estimation reference: [arXiv 2205.12572](https://arxiv.org/abs/2205.12572) (SO(3)/SE(3) for discrete integration + optimization).

In dynamic neural fields:

| Method | arXiv | repo | Lie factorization of motion |
|---|---|---|---|
| **Nerfies** (Park et al., ICCV'21) | [2011.12948](https://arxiv.org/abs/2011.12948) | [google/nerfies](https://github.com/google/nerfies) | deformation field outputs a **per-point SE(3) screw axis S=(r;v)∈ℝ⁶**, applied via exp-map; elastic regularization. *This is the canonical "warp = SE(3) field" paper.* |
| **D-NeRF** (Pumarola et al., CVPR'21) | [2011.13961](https://arxiv.org/abs/2011.13961) | [albertpumarola/D-NeRF](https://github.com/albertpumarola/D-NeRF) | canonical space + time-conditioned displacement field (translation-only deformation; less structured than Nerfies' SE(3)). |
| **NSFF** (Li et al., CVPR'21) | (CVPR'21) | [zhengqili/Neural-Scene-Flow-Fields](https://github.com/zhengqili/Neural-Scene-Flow-Fields) | explicit forward/backward 3D **scene-flow** (per-point displacement) + static/dynamic blending; flow ≈ first-order twist readout. |

**The unification for us (the key result): the per-class warp is ONE twist read out at per-class depth — not three separate transforms.**
The ego camera moves by a single screw `exp(t·ξ_ego)`. A world point at depth `d` induces an image-plane motion that is the *projection* of that screw at depth `d`. So the three measured per-class warps are ONE ξ_ego read out at three depths:
- **Road** (ground plane, finite depth, normal n): the screw + ground constraint → a **planar homography** H(ξ_ego, n, h) — the +15% lever, GREEN through R.
- **MyCar/hood** (rigidly attached, depth≈0 in ego frame): identity warp — the camera and hood move together, zero relative motion.
- **Undrivable/sky** (depth→∞): translation produces no parallax → **rotation-only warp** (the homography of the plane at infinity = the rotational part exp(ω) of the screw).
This is exactly the depth×staticness gradient we MEASURED (FEED-ja). Token cost: **6 numbers for ξ_ego per frame** (or a spline of twists over the clip) **+ one depth/normal scalar per class** (5 numbers), and the entire per-class warp bank is *derived*, not stored. Movables = a few extra residual twists. This is the rate-half's temporal+motion factor in its most compressed honest form, and it is physically exact (Chasles), not a learned approximation.

---

## D. INTRINSIC DIMENSIONALITY estimation (corroborate the measured eff-dim ~26 @ n600)

The partition family {argmax_class over frames} lives on a manifold; its ID is the honest token-budget lower bound for the rate-half. Two ID *flavors* matter and they differ a lot:

| Estimator | source | repo | flavor | property |
|---|---|---|---|---|
| **Participation ratio (PR)** | PR = (Σλ_i)²/Σλ_i² from PCA eigenvalues λ | (in-tree; numpy) | **LINEAR / global** | counts effective *linear* directions. Cheap, but **upper-bounds** the nonlinear ID — a curved 8-D manifold can need ~26 linear directions. *This is almost certainly what our "~26" is.* |
| **TwoNN** (Facco, d'Errico, Rodriguez, Laio, *Sci. Rep.* 2017) | [scirep 2017](https://www.nature.com/articles/s41598-017-11873-y) | [fmottes/TWO-NN](https://github.com/fmottes/TWO-NN), [jmmanley/...](https://github.com/jmmanley/two-nn-dimensionality-estimator) | **NONLINEAR / local** | uses only μ=r₂/r₁ (1st & 2nd NN); minimal-scale → robust to curvature & density variation. The default modern estimator. |
| **MLE-ID** (Levina & Bickel, NeurIPS 2004) | [NeurIPS 2004](https://papers.nips.cc/paper/2577-maximum-likelihood-estimation-of-intrinsic-dimension) | [Tikquuss/intrinsics_dimension](https://github.com/Tikquuss/intrinsics_dimension) | **NONLINEAR** | Poisson process on k-NN distances; ID = inverse-mean of log distance ratios; O(N²D). Pairs with TwoNN as a cross-check. |
| **PH-dim** (Birdal, Lou, Guibas, Şimşekli, NeurIPS 2021) | [2111.13171](https://arxiv.org/abs/2111.13171) | [tolgabirdal/PHDimGeneralization](https://github.com/tolgabirdal/PHDimGeneralization) | **TOPOLOGICAL / fractal** | dimension from the power-law scaling of the 0-/1-dim persistence-lengths sum; built on Schweinhart / Jaquette-Schweinhart ([1907.11182](https://arxiv.org/abs/1907.11182)). *We already use a PH⁰-dim (Lane 0.83) — this is the same family, formalized.* |
| **DANCo / FisherS / MiND / …** | — | [scikit-dimension](https://github.com/scikit-learn-contrib/scikit-dimension) ([arXiv 2109.02596](https://arxiv.org/abs/2109.02596)) | mixed | the canonical Python ID toolbox (scikit-learn API); use for an ensemble of estimators in one call. |
| (toolbox) **DADApy** | Glielmo et al., *Patterns* 2022 | [sissa-data-science/DADApy](https://github.com/sissa-data-science/DADApy) ([2205.03373](https://arxiv.org/abs/2205.03373)) | TwoNN + Gride + density | the Laio-group library (TwoNN authors); also density estimation + clustering on the manifold. |

**The cross-check (falsifiable, $0, do next):** run TwoNN + MLE-ID on the SAME n600 partition-code matrix that gave PR-eff-dim ~26. **Prediction: the nonlinear ID will be markedly *below* 26** (toward the **~8** lane-orbit manifold + the **~6** screw worldline + a small movables count), because PR counts curved directions linearly. If TwoNN ≈ 26 too, the manifold is genuinely high-rank (rules out the rigid-screw factorization and forces a learned residual); if TwoNN ≈ 8–14, the screw×depth×lane-orbit factorization is the right token budget and ~26 was a linear-counting artifact. Either way it is a *measurement*, not an interpretation — exactly the kind the standing discipline demands. Tooling already exists (`scikit-dimension` / `DADApy`, pip-installable, CPU, seconds on 600 points).

> Reconciliation hypothesis (to be measured): **ID_PR ≈ 26 (linear) ⊃ ID_TwoNN ≈ 8–14 (nonlinear) = lane-orbit (~8) + screw (~6) − shared**; off-pose residual ~83% is the fraction NOT captured by the single ego-screw → it is the lane-survival + movables residual, the part that genuinely needs a learned/stored factor.

---

## E. RANKED ADOPTION — opinionated (each: paper + repo + transfer + rough cost)

Grounding from OUR code (read this session): `src/tac/se3.py` (full SE(3) exp/log/geodesic, Sola-2018 [arXiv 1812.01537] convention, (ω,v) order matching PoseNet's 6-vec); `src/tac/boundary_math/lane_sdf_component.py` (class-1 lane = ~7-float/line ground-frame polynomial+dash-gate SDF; FEED-dj: shape captured to false-neg d_seg **0.00046** < target 0.00087, residual **0.00396** = dash-gap false-positives); `src/tac/boundary_math/contour_codec.py` (partition → LZMA-of-labels, size ~ **boundary entropy**, the honest rate axis); `lever_b_levelset_generator.py` (softmax-of-SDF K=5 level set); `hood_static_component.py` / `road_horizon_component.py` (the other per-class strata).

### (i) For the n600 v2 RATE-HALF spatial factor

1. **se(3) screw worldline as the temporal+motion factor** — `src/tac/se3.py` (ours) + Nerfies [2011.12948](https://arxiv.org/abs/2011.12948)/[google/nerfies](https://github.com/google/nerfies) + Modern Robotics screw theory. **THE highest-leverage adoption.** Transfer: store ξ_ego(t) as a short SE(3) geodesic (constant-velocity = **6 floats**; cubic B-spline of control twists = ~**24–48 floats** for the whole 600-frame clip). Already GREEN through R on Road. Cost: ~tens of bytes for the entire t-axis vs O(3·N·T) for time-planes. **This is the single biggest counted-byte win.**
2. **Factor Fields / DiF** [2302.01226](https://arxiv.org/abs/2302.01226) / [autonomousvision/factor-fields](https://github.com/autonomousvision/factor-fields) — the *framework* to phrase the static canonical field in: learned dictionary basis × sparse coefficients = our curvelet basis × per-frame coefficients. Best reported SDF compactness. Transfer: the canonical-frame static field per class becomes a DiF (shared basis FREE in inflate, coefficients counted). Cost: the basis is rule-118 free; coefficients are the counted spatial factor.
3. **Our own contour codec** `contour_codec.py` — for the *bulk* partition (Road/MyCar/Undrivable, piecewise-constant), boundary-entropy LZMA **beats** a plane factor because the target is an argmax partition, not a smooth field. Transfer: ship ONE canonical-frame partition (~1–2 KB) + the screw, not 600 partitions. Cost: measured by `partition_description_bytes`.
4. **K-Planes / HexPlane** [2301.10241](https://arxiv.org/abs/2301.10241)/[2301.09632](https://arxiv.org/abs/2301.09632) — adopt **only the space planes {xy} as a per-class static factor, with the time-planes deleted and replaced by #1**. The white-box static/dynamic split is the value; the learned unconstrained deformation is the part we *replace* with rigid screw physics. Cost: rank-r (x,y) plane per class.
5. **TensoRF-VM / CP** [2203.09517](https://arxiv.org/abs/2203.09517)/[apchenstu/TensoRF](https://github.com/apchenstu/TensoRF) — the explicit rank knob for the per-class canonical matrix M_xy (VM) or rank-1 vectors (CP, cheapest). Use as the quality/byte dial on #4.

### (ii) For compressed MANIFOLD OBSERVABILITY

1. **DADApy + scikit-dimension (TwoNN, MLE-ID, Gride)** [2205.03373](https://arxiv.org/abs/2205.03373)/[sissa-data-science/DADApy](https://github.com/sissa-data-science/DADApy), [2109.02596](https://arxiv.org/abs/2109.02596)/[scikit-dimension](https://github.com/scikit-learn-contrib/scikit-dimension) — run on the n600 partition codes; **the falsifiable cross-check of our PR eff-dim ~26**. $0, CPU, seconds. Predict nonlinear ID ≪ 26.
2. **se(3) log worldline** `log_map_se3` (ours) — the screw trajectory IS the observable ego-motion; `log_map_se3(R_t, t_t)` reads the per-frame twist; its spectrum (is it ~constant? low-order spline?) directly observes motion rank.
3. **PH-dim** [2111.13171](https://arxiv.org/abs/2111.13171)/[tolgabirdal/PHDimGeneralization](https://github.com/tolgabirdal/PHDimGeneralization) — formalizes our Lane PH⁰-dim 0.83; the persistence diagram of the lane field is the dash birth-death observable (6/23 dashes/frame topo floor).
4. **K-Planes static/dynamic plane split** — diff feature energy on {xy} (static) vs {xt,yt} (moving) = direct readout of "what is static vs what moves" without re-running.

---

## F. THE DESIGN — Stratified Screw-Warped Level-Set (S²WL) factorization (original; opinionated)

**Where the SOTA does NOT fit our field (the gap we fill).** Every method in §A–B (K-Planes, HexPlane, TensoRF, Tensor4D, 4DGS) factors a **smooth, dense** field and learns an **unconstrained** temporal deformation. Our field violates all three of their fit assumptions:
1. **It is a PARTITION, not a smooth field.** The authority is `argmax_k φ_k` (codim-1 boundaries); a plane factorization's linear+tiny-MLP decoder has the spectral-bias/Gibbs failure we already measured against HNeRV. Token efficiency of a *carrier* ≠ topology-match of the *decoder*.
2. **The binding structure is 1D (thin lanes), not 2D area.** A finite-resolution plane bank smears a dashed lane ridge; the honest representation is a **1D polynomial-SDF manifold + a birth-death persistence diagram** (our `lane_sdf_component` + Morse-Smale codec), which the plane methods have no term for.
3. **The motion is ONE rigid ego screw, not a generic deformation.** Spending a learned {xt,yt,zt} plane bank to represent what is physically `exp(t·ξ_ego)` is the central waste. The contest's physical truth (GROK-confirmed: one rigid trajectory through a near-static world) *lets us constrain* the temporal factor to its Lie-algebra generator — which the dynamic-NeRF methods cannot, because their scenes are genuinely non-rigid.

So the optimal factorization is **not a plane bank**. It is a **stratified, Lie-warped, topology-matched** factorization:

```
                ┌─ TEMPORAL/MOTION ─┐   ┌──── STATIC (canonical frame) ────┐   ┌── RESIDUAL ──┐
 φ_k(x,y,t)  =  W_k( ξ_ego(t), d_k ) ∘  Φ_k^canon(x, y)                     ⊕  ρ_lane ⊕ ρ_mov
                   (one screw,            (per-class topology-matched           (persistence
                    per-class depth)       spatial factor)                       + sparse)
   partition = argmax_k φ_k        (softmax-of-SDF level set, lever_b_levelset_generator)
```

### F.1 The four factors (and their honest token budgets)

**(1) Temporal/motion factor — ONE screw worldline ξ_ego(t) ∈ SE(3).**
A cubic B-spline of control twists on SE(3) (de Casteljau on the Lie group, or `exp` of a spline in se(3)≅ℝ⁶). Constant-velocity = **6 floats**; a smooth 600-frame ego trajectory needs only a handful of control twists (~**4–8 × 6 = 24–48 floats**). Implemented on top of `src/tac/se3.py` (`exp_map_se3`, `log_map_se3`, `left_jacobian_so3` already present + tested). The worldline is the *geodesic* in SE(3), so interpolation between control twists is exact rigid motion, not a learned approximation.

**(2) Per-class depth/plane readout W_k — the stratification.** This is the §C unification made into a representation. Each class sits at a depth/plane and its image-plane warp is the *projection of the one ego screw at that depth*:
- **Road (k=0):** ground homography H(ξ_ego, n_ground, h_cam) — already an IPM in `lane_sdf_component` (`_FX,_FY,_CX,_CAM_H,_V_HORIZON`) and `road_horizon_component`. (+15% lever, GREEN through R.) ~3 plane params (normal + height).
- **MyCar/hood (k=4):** identity warp (rigidly attached; the #139 static core). 0 params.
- **Undrivable/sky (k=2):** rotation-only warp = homography of the plane at infinity = `exp_map_so3(ω_ego)` (translation has no parallax). 0 extra params (reuses ξ_ego's ω).
- **Movable/cars (k=3):** ego screw + a per-object residual twist (§F factor 4).
Total stratification cost: **~5 floats** for the whole bank; every W_k is *derived* from ξ_ego, not stored. (Physically exact — Chasles + projective geometry — not learned.)

**(3) Static canonical spatial factor Φ_k^canon(x,y) — topology-matched, per class.** After un-warping every frame by W_k⁻¹ into the canonical (t=0) frame, the bulk classes are *nearly time-invariant* → we store ONE canonical field, not 600. The right per-class representation:
- **Bulk (Road/MyCar/Undrivable):** the **contour codec** (`contour_codec.partition_description_bytes`) on the canonical-frame partition — boundary-entropy LZMA, ~**1–2 KB once**. (Beats a plane factor: the target is piecewise-constant.) Optional quality dial: a rank-r DiF/TensoRF-VM matrix M_xy per class for sub-boundary refinement (rule-118-free basis, counted coefficients).
- **Lane (k=1):** the `lane_sdf_component` ~**7-float/line** ground-frame polynomial+dash manifold (centerline poly + half-width poly + dash period/phase). Captures lane SHAPE to false-neg d_seg 0.00046 (FEED-dj, measured). The IPM rasterizer is FREE inflate-time (rule 118); only the ~35 floats/frame coefficients are counted (→ ~1–2 KB; and most of that is *also* screw-predictable from the canonical lane × ego motion, leaving a tiny per-frame coefficient residual).

**(4) Residual factors — where the irreducible video-derived signal lives.**
- **ρ_lane (Morse-Smale persistence diagram):** the dash birth-death pairs (the 0.00396 dash-gap false-positive residual; topo floor 6/23 dashes/frame, PH⁰-dim 0.83). Stored as a persistence diagram (birth, death, position) per frame — the *minimal* sufficient statistic for the thin-ridge residual the smooth factors cannot carry.
- **ρ_mov (movable residual twists):** a handful of extra se(3) twists for the few movers.
- **hard-pixel sidecar:** the through-R survival residual (the LEVER-4/UNIWARD-targeted flips) as a sparse bitmap, allocated by Δd_seg-per-byte (the per-stage/per-pixel surgical-repair toolbox).

### F.2 Why this is the right factorization (the variational/observability argument)

- **It is the indirect-RD optimum in disguise.** The contest is coding-for-machines (indirect RD on the argmax-edge manifold). S²WL puts every byte where the SegNet argmax actually moves: the screw (motion the scorer reads through pose + boundary advection), the canonical boundaries (the partition), the lane manifold + persistence (the unstable d_seg gate). No byte is spent on full-RGB or on a smooth field the argmax ignores.
- **It makes the manifold dimension OBSERVABLE and ADDITIVE:** `eff-dim = dim(ξ_ego spline) + Σ_k dim(Φ_k^canon) + dim(ρ_lane) + dim(ρ_mov)`. Our measured PR eff-dim ~26 should decompose as ≈ **6 (screw) + ~8 (lane orbit) + ~few (bulk boundary low-rank) + movables**. The §D TwoNN/MLE cross-check tests exactly this; off-pose residual ~83% ↔ the fraction NOT captured by the single ego-screw = ρ_lane + ρ_mov (the genuinely learned/stored part).
- **It is gauge-canonical for MDL.** Un-warping to the canonical frame removes the ego-motion gauge orbit before coding (cf. `witness_dsl/gauge.py`); the canonical-frame partition is the MDL-minimal description, and the screw is the orbit coordinate. This is the originality claim of v2 (the warp→SDF→openpilot-WZ-residual chain), honestly a novel *composition* (NO-FAKE #7), closest prior INVC [2112.11312]; the primitives (SE(3) exp, SDF, persistence, IPM) are borrowed and cited.

### F.3 Concrete build sketch (on our real modules; $0-smokeable first)

```
v2 spatial/temporal factor (counted bytes) =
  twists.npz         # ξ_ego cubic-spline control twists  (se3.exp_map_se3)         ~24–48 floats
  depths.npz         # per-class d_k / ground normal+height                          ~5 floats
  canon_partition    # contour_codec.encode_partition(canonical-frame argmax)        ~1–2 KB
  lane_manifold.npz  # lane_sdf_component coeffs (poly+dash) per frame (screw-resid)  ~1–2 KB
  persistence.npz    # Morse-Smale dash birth-death diagram                          ~hundreds B
  movables.npz       # residual twists + sparse hard-pixel sidecar                   ~hundreds B
inflate.py (FREE, rule 118): exp-map warp (se3) + IPM rasterizer + softmax-of-SDF (lever_b_levelset_generator) → argmax partition
```

**Step 0 ($0, do first — pure measurement, no training):**
1. Run `DADApy`/`scikit-dimension` TwoNN+MLE on the n600 partition codes → does nonlinear ID land near 6+8 ≪ 26? (Validates the factor budget; falsifies if ≈26.)
2. Fit ONE ξ_ego spline to the existing per-frame poses (`log_map_se3` frame-to-frame) → residual of `W_k(ξ_ego)·canon vs actual argmax` per class = the measured ρ per class. This *is* the off-pose 83% decomposed by class, and it tells us exactly how many bytes each residual factor needs **before** any GPU.
3. `partition_description_bytes(canonical-frame argmax)` vs `mean_t partition_description_bytes(frame_t argmax)` → the measured rate win of canonicalize-then-code (the screw's byte value).

**Step 1 (build):** wire `--screw-temporal-factor` + `--canonical-frame-coding` into the witness trainer; the per-class SDF components already exist (`lane_sdf_component`, `hood_static_component`, `road_horizon_component`); compose them under one ξ_ego; byte-close via `contour_codec` + the existing rate path; exact-eval. Resumable + per-stage ckpts per the launch non-negotiables.

### F.4 Authoritative recommendation (passionate, but means≠ends)

Adopt the **screw temporal factor (#E-i-1) immediately** — it is the largest counted-byte win, it is physics-exact, it is already GREEN through R, and the SE(3) machinery is already built and tested in `se3.py`. Phrase the static field in **Factor Fields/DiF** terms but realize the bulk via our **contour codec** (topology-matched) and the lane via the **polynomial-SDF + persistence** residual — do **not** drop a uniform K-plane bank on a partition target (it re-imports the HNeRV Gibbs failure). Use **TwoNN/MLE/PH-dim (DADApy)** as the $0 observability cross-check on eff-dim ~26 *this session-class* — it is a measurement, not an interpretation, and it gates whether the whole factor budget is honest. The novelty is the **composition** (Lie-geodesic time × projective-depth stratified readout × topology-matched per-class spatial × persistence residual), not any single primitive — claim it as such, only after a byte-closed exact row beats 0.19110.

---

## Provenance + honesty

- **Axis:** `[macOS research-signal / advisory only]`. No score claim; pointer **0.19110 UNMOVED**. This brief is a MEANS (design + SOTA map); the END is a lower exact score — not achieved here, by construction (no eval run).
- **NO-FAKE:** every arXiv id + repo verified via WebSearch this session (URLs inline). In-tree file paths read this session (`se3.py`, `lane_sdf_component.py`, `contour_codec.py`, directory listing of `boundary_math/`). The S²WL factorization is an original *composition* of cited primitives; no measured result is asserted for it — the Step-0 measurements are proposed, not run.
- **Flagged uncertainty:** `brentyi/canonical-factors` repo URL for [2308.15461] not directly opened (paper id verified via search surface). The "~8 lane-orbit / ~6 screw / ~26 PR" decomposition is a *hypothesis to be measured* (Step 0.1–0.2), not a measured fact for S²WL. DiF's reported "best SDF compactness" is for smooth SDFs, not argmax partitions — transfer is by analogy, to be measured.
- **Next $0 unit:** the three Step-0 measurements (TwoNN/MLE ID; one-screw residual-by-class; canonicalize-then-code rate win). Each is a measured row, CPU, seconds–minutes, and each directly sizes a v2 factor before any GPU.
