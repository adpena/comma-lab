# Information-Geometric Foundations of the Task-Space Witness

*Draft section, 2026-07-10. Provenance discipline: every quantitative claim below is tagged
**[MEASURED]** (an n600 through-R number we produced) or **[FRAMING]** (a classical result we invoke to
name/justify the measured geometry). The information geometry itself (Amari, Chentsov, Nielsen, Goldman) is
**classical mathematics we apply**; our contribution is (i) recognizing that the contest's frozen-scorer
distortion IS this geometry, and (ii) the measured correspondences that confirm it. Pointer 0.19108282
[contest-CPU] — this section justifies the approach; it does not itself move the score.*

## 1. The task-space thesis

The comma video-compression scorer does not measure pixel fidelity. It measures
`S = 100·d_seg + √(10·d_pose) + 25·|archive|/N`, where `d_seg` is the disagreement rate of a frozen SegNet's
last-frame class **argmax** and `d_pose` is the MSE of a frozen PoseNet's six output scalars. This is an
**indirect rate–distortion** problem (the CEO / remote-source problem of Berger–Yeung): we do not code the
video, we code the minimal statistic sufficient to reproduce a downstream machine's decision. Full-RGB
reconstruction spends its bit budget on detail the scorer is invariant to. The witness instead codes the
**task-sufficient statistic** — the argmax partition and the ego-pose — and the natural question is: *in
what geometry should its distortion be measured?* The answer, which this section develops, is the frozen
scorer's own **information geometry**, and it is not a metaphor: our measured boundary statistics are the
applied shadow of dually-flat statistical geometry.

## 2. The frozen scorer defines a dually-flat statistical manifold

The SegNet head is a per-pixel softmax over five classes. A softmax is an exponential family: with logits
`θ ∈ ℝ⁵` as **natural coordinates**, the log-partition is `F(θ) = logsumexp(θ)`, the class probabilities are
the **dual (expectation) coordinates** `η = ∇F(θ)`, and the two are related by the Legendre–Fenchel
transform `F* = ⟨η, θ⟩ − F(θ)` (the negative entropy on the simplex). This is a **dually flat space** in
the sense of Amari: it carries a pair of flat, torsion-free affine connections (the e-connection on θ and
the m-connection on η) whose midpoint is the Levi-Civita connection of the Fisher metric.

The Fisher information matrix is the Hessian of the log-partition, `I(θ) = ∇²F(θ)`, and — the fact the whole
approach rests on — it equals the **curvature of the log-likelihood at the argmax** (Nielsen 2020, 2022):
near the maximizer the log-posterior is a downward paraboloid whose curvature *is* the Fisher information.
A sharp peak (large Fisher) is a confident, stable argmax; a flat peak (small Fisher) is a near-tie whose
argmax is fragile under perturbation, with Cramér–Rao variance `I⁻¹`.

**[MEASURED]** On the contest video, the Fisher curvature of the SegNet head correlates with the class
**margin** (the top-two logit gap) at Pearson **0.978**, and the local logit landscape near the boundary is
quadratic with Levenberg–Marquardt goodness-of-fit ρ ≈ **0.85**. The margin field is therefore the
first-order surrogate for the observed Fisher (our scorer is deterministic, so it is the *observed* Hessian,
not the expected one; the residual 0.022 in the correlation is exactly the gap between first-order margin
and second-order curvature).

## 3. Why this metric and no other

The choice of the Fisher/margin metric over, say, RGB-L², is not an engineering preference — it is forced.
**Chentsov's theorem** [FRAMING] states that the Fisher metric is the *unique* Riemannian metric (up to
scale) invariant under Markov morphisms, i.e. under sufficient-statistic reductions. The SegNet argmax *is*
a sufficient-statistic reduction of the frame; RGB-L² is not invariant under it, while the Fisher metric is.
Coding distortion in the margin metric is therefore descending in the canonically correct geometry for a
sufficient-statistic codec. The dual statement is the **f-divergence data-processing inequality**: the
reconstruction operator `R` (bicubic↑ → uint8 → bilinear↓) is a near-deterministic Markov kernel, and
information about the argmax can only be *lost* through it, with equality iff the coded statistic is
sufficient.

**[MEASURED]** ~97% of `d_seg` error localizes to a boundary annulus of ~4.7% frame area — exactly the
small-Fisher / small-margin locus — and it is boundary *jitter* (the argmax crossing under `R`), not
region-level misclassification. The flips are texture-dependent because a small Fisher makes the Cramér–Rao
argmax variance exceed the margin precisely where the frame is high-frequency.

## 4. The witness is a curved submanifold; distortion is a curved Bregman divergence

The witness does not parametrize the full `m`-dimensional logit field. It parametrizes an intrinsically
low-dimensional (≈8-dimensional [MEASURED]) **non-affine submanifold** capturing the lane/partition
trajectory. Nielsen's **curved Bregman divergences** (arXiv 2504.05654) [FRAMING] are exactly Bregman
divergences of a Legendre-type generator restricted to a non-affine subspace of dimension `k < m` — his
Figure 1 (a `k`-dimensional curved exponential family embedded in an `m`-dimensional one) is a picture of
the witness embedding. The `d_seg`-relevant distortion of the witness is thus a **curved Bregman divergence
of `logsumexp`**, and the margin is the natural-coordinate distance to the decision boundary. This is the
precise form of "spend bytes on the scorer-relevant manifold, not on RGB."

## 5. The solvers are projections

Placing the witness optimally at a boundary pixel is a projection onto the non-affine chart, and three
classical results name the three solves we use:

- **Bregman projection** (Nielsen 2504.05654, Theorem 1) [FRAMING]: the barycenter under a curved Bregman
  divergence is the right Bregman projection of the full-space target onto the non-affine subspace. Our
  semi-discrete-OT damped-Newton head-offset solve and the Dykstra alternating-projection feasibility solve
  are both this projection, computable in dual (η) coordinates.
- **Information-radius centroid** (Nielsen 2102.09728) [FRAMING]: for a *set* of per-pixel argmax
  distributions, the divergence-correct representative is the Jensen–Shannon / information-radius centroid,
  not the Euclidean mean — the correct codebook-update rule should a vector-quantized witness be built.
- **Hessian-preconditioned conjugation** (Plus-Gourdon & Nielsen, arXiv 2606.09077) [FRAMING]: because the
  boundary Hessian is anisotropic (flat-interior, sharp-along-boundary), a naive Newton step is
  ill-conditioned. Affine-deforming the local problem so its second-order Taylor coincides with the
  canonical paraboloid (whose conjugation is the identity), solving the well-conditioned residual, then
  undoing the deformation, is the correct preconditioner — one eigendecomposition plus two matrix-vector
  products per query. This directly targets the anisotropic annulus measured in §3.

## 6. The chart and the pose: an affine (G,X)-structure

The witness is an atlas of charts over the argmax partition, and Goldman's *Geometric Structures on
Manifolds* [FRAMING] supplies the classical language: a locally-homogeneous **(G,X)-structure** with a
**developing map** and a **holonomy homomorphism**. Amari's dually-flat space (§2) *is* a flat affine
structure, so the statistical and differential-topological pictures coincide. Under this identification the
canonicalize-to-ground-frame construction is a developing map of a (G,X)-structure with `G = SE(3)` acting
on the ground plane, and the ego-motion twist `ξ` is its **holonomy**.

This is why the *same* `ξ` warps the partition (for `d_seg`) and yields the pose (for `d_pose`) — the
dual-use screw is the holonomy of the chart, not a coincidence. **[MEASURED]** a single stored se(3) twist,
FiLM-conditioned into the render, drives the pose contribution to a sidecar-priced byte cost; the partition
warp and the pose read out of one `ξ`.

## 7. Two geometries, one witness (the caveat)

The Fisher metric is not the only information metric. The **Wasserstein information metric** (Li 2021,
flagged in Nielsen 2022) is distinct, and the witness lives at the interface: `d_seg` distortion is
**Fisher/margin** (statistical distinguishability of the argmax), while the transport-flavored levers —
semi-discrete-OT head offset, area-mass matching, and the v8 Laguerre power-diagram generator — are
**Wasserstein/transport**. Conflating them is a real hazard: **[MEASURED]** an optimal-transport
area-mass-match move *hurt* `d_seg` (Wasserstein-optimal, Fisher-suboptimal), which is why the boundary
objective is flip-weighted rather than mass-matched. The correct discipline is: descend `d_seg` in the
Fisher metric; use Wasserstein only for the generator/warp geometry.

## 8. Correspondence (measured ↔ named)

| Measured fact (n600, through-R) | Classical name |
|---|---|
| Fisher curvature ↔ margin, Pearson **0.978** | Fisher = curvature of the log-likelihood at the argmax (Nielsen 2020) |
| Boundary logit landscape quadratic, LM ρ ≈ **0.85** | Fisher = Hessian ∇²F of `logsumexp` (Amari; Nielsen 2022) |
| "Code the argmax statistic, not RGB" | Chentsov: Fisher is the unique sufficiency-invariant metric |
| `R`-roundtrip erases the margin → flips | f-divergence data-processing inequality |
| ~97% of `d_seg` in a ~4.7%-area annulus; jitter not miss | small-Fisher / Cramér–Rao boundary locus |
| Witness on a ~8-dim non-affine manifold | curved Bregman divergence (Nielsen 2504.05654) |
| head-offset / feasibility solve | Bregman projection; information-radius centroid |
| anisotropic annulus ill-conditioning | Hessian-preconditioned conjugation (Plus-Gourdon & Nielsen 2606.09077) |
| dual-use `ξ`: one twist warps partition *and* gives pose | holonomy of the affine (G,X)-structure (Goldman) |
| OT area-mass-match hurt `d_seg` | Wasserstein-optimal ≠ Fisher-optimal (Li 2021) |

## 9. Honest scope

This section is a **justification**, not a result. It gives our empirically-measured boundary geometry its
canonical names and an independent cross-validation trail; it converts a set of separately-discovered facts
(Fisher=margin, the quadratic head, the boundary annulus, the dual-use screw) into one object — a curved
submanifold of a dually-flat affine (G,X)-structured statistical manifold. It surfaces exactly one
buildable solver improvement (§5, Hessian preconditioning) and two design principles (the codebook centroid
for a future VQ-witness; the affine-structure design of the perspective chart), each gated on a measured
check that it beats the current method. The exact through-R measurement remains the sole authority for any
score claim; the information geometry is the framing, never the number.

## References

- S. Amari, *Information Geometry and Its Applications*, 2016.
- N. N. Chentsov, *Statistical Decision Rules and Optimal Inference*, 1982.
- F. Nielsen, "An Elementary Introduction to Information Geometry," *Entropy* 22(10):1100, 2020.
- F. Nielsen, "The Many Faces of Information Geometry," *Notices of the AMS* 69(1):36, 2022.
- F. Nielsen, "Curved representational Bregman divergences and their applications," arXiv:2504.05654.
- F. Nielsen, "On a Variational Definition for the Jensen-Shannon Symmetrization of Distances based on the
  Information Radius," arXiv:2102.09728, 2021.
- B. Plus-Gourdon and F. Nielsen, "Neural Legendre-Fenchel transform with Hessian Preconditioning,"
  arXiv:2606.09077, 2026.
- W. M. Goldman, *Geometric Structures on Manifolds*, 2021.
- W. Li, "Transport information geometry," 2021.
- H. Cramér; C. R. Rao (Cramér–Rao bound); I. Csiszár (f-divergences); R. Sibson (information radius).
