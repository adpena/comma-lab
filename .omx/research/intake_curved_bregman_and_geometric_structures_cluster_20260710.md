# Intake: Curved Bregman divergences (Nielsen 2504.05654) + Geometric Structures (Goldman) — 2026-07-10

**Source:** operator-dropped cluster (info-geometry deep-dive, 4 links):
1. Nielsen, *An Elementary Introduction to Information Geometry* (Entropy 2020) — ALREADY banked
   ([[intake_fisher_is_loglik_curvature_at_argmax_20260710]] slide source).
2. Nielsen, *The Many Faces of Information Geometry* (Notices AMS 2022) — ALREADY banked
   ([[intake_nielsen_many_faces_infogeom_chentsov_wasserstein_20260710]]).
3. **Nielsen, *Curved representational Bregman divergences* (arXiv 2504.05654 v5, 2026)** — NEW.
4. **Goldman, *Geometric Structures on Manifolds* (UMD, 2021)** — NEW.
Framing/theory intake (rigorous SPINE of our measured geometry), NOT a score-mover. Pointer 0.19108282
UNMOVED. PDFs held in session scratchpad/tool-results only — ephemeral, NOT durable evidence paths.

## Paper #3 — Curved representational Bregman divergences (Nielsen)
- Bregman divergence of a Legendre-type convex generator F: `B_F(θ:θ')=F(θ)−F(θ')−⟨θ−θ',∇F(θ')⟩`; = KL
  between exp-family members when F = log-partition. Legendre duality: primal θ ↔ dual η=∇F(θ); reference
  duality `B_F(θ1:θ2)=B_{F*}(η1:η2)`.
- **Curved Bregman = B_F restricted to a NON-AFFINE parameter subspace (dim k<m)** (his Fig.1: curved
  exponential family, U dim k embedded in Θ dim m).
- **Theorem 1: barycenter under a curved Bregman divergence = the right Bregman PROJECTION onto the
  non-affine subspace** (of the full-divergence barycenter).
- α-divergences = representational curved Bregman via α-embeddings of the simplex.

## Paper #4 — Geometric Structures on Manifolds (Goldman)
- Locally-homogeneous (G,X)-structures (Ehresmann 1936 → Thurston geometrization): atlas of charts into a
  homogeneous space X=G/H with transitions in G; NOT necessarily Riemannian. Affine + projective geometry,
  flat torsion-free connection, coordinate atlas, developing map, holonomy homomorphism, parallel transport.

## THE ONE-OBJECT STATEMENT the cluster assembles (our witness geometry, named rigorously)
The witness = a **CURVED (non-affine, dim ~8) submanifold of a DUALLY-FLAT = AFFINE (G,X)-structured
statistical manifold.** Amari's dually-flat space (statistics) IS Goldman's affine (G,X)-structure
(diff-topology) — a pair of flat torsion-free connections, same geometry two traditions.
- **Ambient dually-flat space** = SegNet per-pixel softmax exp-family: natural coords θ=logits, log-partition
  F=logsumexp, dual coords η=∇F=class probs; Legendre θ↔η = e/m-connection pair; τ-anneal moves along it.
  **Quadratic-head chart (#341) = Hessian ∇²F = Fisher** (the AMS-survey "FIM=Hessian of log-partition,"
  generator now named logsumexp). [[L1]] Fisher=margin 0.978.
- **d_seg = a CURVED Bregman divergence**: B_logsumexp restricted to the witness's non-affine dim~8 chart.
  Margin = natural-coordinate distance to the decision boundary. Rigorous form of "spend bytes on the
  scorer-relevant manifold, not full-RGB." [[L17]] task-sufficient statistic; #155 quotient codec.
- **Head-offset solve = a Bregman PROJECTION** (Thm 1): #288 damped-Newton semi-discrete-OT head offset +
  #73 Dykstra alternating-projection feasibility are BOTH the right Bregman projection onto the non-affine
  chart, computable in dual (η) coords by his exact iteration.
- **Pose = the HOLONOMY of the affine structure** (Goldman): canonicalize-to-ground-frame + se(3) screw ξ
  (#193/#194) = a developing map of a (G,X)-structure with G=SE(3) on the ground plane; ego-twist ξ = the
  holonomy. WHY the SAME ξ warps the partition (d_seg) AND gives the pose (d_pose) — the dual-use screw is
  the chart's holonomy, not a coincidence. [[L10]] GR action; L-pose screw dual-use.

## Two GATED threads (framing/solver — NOT builds)
1. **Head-offset as a Bregman projection in dual coords** (Nielsen Thm 1) — an exact projection algorithm
   that MAY beat our generic damped-Newton (#288). $0 check: does the Bregman-projection form beat the
   current solver on the MEASURED head-offset problem? Gated — exact through-R measurement is the authority.
   Owner: #288/#341.
2. **Design #185 perspective-aware chart as an affine (G,X)-structure** (Goldman) — developing map +
   holonomy=ξ, not an ad-hoc warp. Design principle for #185/#191 when that work fires. Held.

## Paper #5 — JS-symmetrization via the Information Radius (Nielsen, arXiv 2102.09728, 2021)
Variational generalization of Jensen-Shannon divergence via Sibson's **information radius** (min average
divergence to a centroid) → information PROJECTIONS + relative-JS divergences; explicitly aimed at
**clustering and quantization**. Connection to us:
- **The information-radius centroid IS the divergence-correct quantizer centroid.** If/when a VQ/codebook
  witness is built (#78 capstone: our own small learned basis / VQ-NeRV-class), the codebook-update rule
  must be the **JS / Bregman / information-radius centroid over the class-logit distributions, NOT Euclidean
  k-means** — because the distortion is a divergence on the softmax simplex, not L2. This is the correct
  centroid for the "represent a set of per-pixel argmax-distributions by one code" problem. Sister of the
  Bregman-projection head-offset (paper #3): projection = 1 point onto the chart; information radius =
  centroid of a set.
- **Symmetric divergence for a symmetric target.** d_seg is a SYMMETRIC argmax-disagreement; JS
  (symmetrized) is the natural symmetric divergence, vs the asymmetric KL/Bregman. Relevant when the
  distortion must be symmetric (e.g. codebook assignment).
- **GATED thread:** JS/information-radius centroid as the codebook-update rule, gated on #78 (a VQ-witness)
  actually being built. Held — no VQ-witness in flight; do not build speculatively.

## Paper #6 — Neural Legendre-Fenchel transform with Hessian Preconditioning (Plus-Gourdon & Nielsen, arXiv 2606.09077, 2026) — MOST ACTIONABLE
The LF transform F↔F* IS the θ↔η duality of our dually-flat space (F=logsumexp on logits ↔ F*=neg-entropy
on class probs). The paper computes it with a **Hessian preconditioner**: affine-deform around the minimizer
so the 2nd-order Taylor coincides with the canonical paraboloid (conjugation = identity), learn/solve the
easy residual near identity, recover via the inverse deformation. Cost: 1 eigendecomposition at init + 2
matvecs/query. Big gains on ILL-CONDITIONED problems. Affine invariance = the Goldman affine (G,X)-structure
(paper #4) → this is the COMPUTATIONAL realization of the cluster's framing.
- **Directly targets OUR measured ill-conditioning.** The anisotropic annulus (#333, flat-interior +
  sharp-along-boundary) IS an anisotropic-Hessian problem. Our head-offset (#288 damped-Newton semi-discrete
  OT), quadratic-basin TerminalSolve (#341), and terminal MC finisher (#396) all Newton/project in the
  dually-flat coords; Hessian preconditioning is the canonical fix for anisotropic-Hessian Newton.
- **GATED thread (most actionable in cluster):** Hessian-preconditioned Bregman-projection for the
  head-offset / TerminalSolve — eigendecompose the local Fisher/Hessian once, deform to the canonical
  paraboloid, solve, undo. Gated on a **$0 measured check** it beats our current damped-Newton on the real
  anisotropic head-offset; exact through-R measurement stays authority. Do NOT build without the check +
  operator GO. Owner: #288/#341/#396.

## Papers #7 + #8 — Fisher-Rao distance bounds + Bregman MATRIX divergences (metrics on the Fisher/Hessian)
- **#7 Nielsen, "Approximation and bounding techniques for the Fisher-Rao distances" (arXiv 2403.10089, 2024).**
  The Fisher-Rao distance = geodesic length in the Fisher metric = the INTRINSIC distance-to-boundary of
  which our first-order **margin** is the linearization. Key for us: the paper gives **tight computable
  upper bounds WHEN the Fisher is a Hessian metric** — ours IS (logsumexp), so a principled distance-to-flip
  is actually calculable. Also introduces the **Birkhoff/Hilbert projective-cone distance** (a metric on the
  positive cone).
- **#8 (ResearchGate, 403'd — topic-only) "Bregman MATRIX divergences" (LogDet/von Neumann/Burg on PSD
  matrices).** The divergence geometry ON the space of Fisher/Hessian matrices (all PSD). Same object as
  #7's Hilbert-cone metric.
- **Connection:** both are metric geometry ON the Fisher/Hessian matrices themselves — relevant to (a) the
  anisotropic-Hessian preconditioning (paper #6, the build in flight), (b) the low-rank pose-section codec
  (#140, a matrix), (c) a refined distance-to-boundary. Reinforce §5/§7 of
  `docs/paper/information_geometric_foundations.md` WITHOUT changing the thesis.
- **GATED thread:** Fisher-Rao geodesic distance-to-flip (via the Hessian-metric tight bounds) as a
  refinement of the first-order margin in margin-saliency (#141) / S_R (#268). Held, gated on a $0 measured
  check it beats the plain margin. Same disposition as the Cramér-Rao flip-risk thread
  ([[intake_fisher_is_loglik_curvature_at_argmax_20260710]]). Do NOT build speculatively.

## Paper #9 (Amari, "Information Geometry and Its Applications", Thm 6.12) — the PARENT theorem
The **generalized Pythagorean theorem**: `D[P:Q] = D[P:R_PQ] + D[R_PQ:Q]` when the legs meet dual-orthogonally
(e-geodesic ⊥ m-geodesic) at R_PQ = the Bregman projection. This is the PARENT of the whole cluster's
projection machinery:
- **Waterfilling additivity** (#157 reverse-waterfill) is EXACT iff components are dual-orthogonal; the
  cross-term when they are NOT = the meta-Lagrangian interaction/Volterra term. A measurable check on the
  existing bit-allocation (are the independent-component d_seg allocations dual-orthogonal?).
- **Dykstra alternating-projection** (#73) converges by iterated Bregman projections (Pythagorean monotone
  descent); the semi-discrete-OT head-offset (#288) is well-posed for the same reason.
- Nielsen's curved-Bregman projection (paper #3) = Thm 6.12 restricted to a non-affine submanifold = witness.
- **GATED diagnostic:** measure whether our per-component d_seg allocations are dual-orthogonal; the additive
  waterfill's systematic error = the cross-term. Not a build — a check on existing #157. Folded into
  `docs/paper/information_geometric_foundations.md` §5.

## Paper #10 (Nielsen, "Non-Euclidean Computational Geometry for ML") — the ALGORITHMIC side
Slide deck (WebFetch could not OCR; content known from Nielsen's corpus): **Bregman Voronoi diagrams,
Bregman POWER diagrams (= Laguerre diagrams), Bregman balls, smallest-enclosing-balls, k-means under Bregman
divergences.** Direct connection: **the SegNet argmax partition IS a Bregman power/Laguerre diagram** (#284:
"argmax = Laguerre power-diagram"; v8 stores the GENERATORS not the boundaries). Nielsen's non-Euclidean CG
gives the algorithms for computing + coding these diagrams → the v8 rate-half (store power-diagram generators,
AR-code them). GATED: informs v8 (#377 build-wave / #380 crucible-3); no new build — reinforces the existing
v8 Laguerre-generator design.

## Already-covered / low-signal (honest, no re-bank)
- `entropy-22-01100-v2.pdf` = Nielsen 2020 Entropy "Elementary Introduction to Information Geometry" = the
  SOURCE of paper #1 / the Fisher slide — ALREADY banked
  ([[intake_fisher_is_loglik_curvature_at_argmax_20260710]]). No new content.
- `HPC4DS/index.html` = Nielsen's "High-Performance Computing for Data Science" course — compute-facet
  adjacent (our MLX/Metal program #252) but a pedagogical course, not new witness math. Noted, not banked.

## Honest verdict + routing
- **Bank** (this note): the cluster is the rigorous SPINE — cite Nielsen (curved Bregman = witness on a
  non-affine chart; head-offset = Bregman projection) + Goldman (dually-flat = affine (G,X)-structure;
  pose = holonomy) alongside the measured Fisher=margin 0.978 + quadratic-head + se3 anchors in the
  writeup/paper. No new lever, no DSL/equation change, no launch — measured content already captured; this
  is rigor/naming + two gated threads.
- **No speculative build.** Same disposition as the Fisher-slide + Many-Faces + Hoel-EI intakes: exact
  through-R measurement is authority; info-geometry is the framing/justification, never the score.
Pointer 0.19108282 UNMOVED.
