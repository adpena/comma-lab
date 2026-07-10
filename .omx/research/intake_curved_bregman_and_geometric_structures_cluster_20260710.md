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

## Honest verdict + routing
- **Bank** (this note): the cluster is the rigorous SPINE — cite Nielsen (curved Bregman = witness on a
  non-affine chart; head-offset = Bregman projection) + Goldman (dually-flat = affine (G,X)-structure;
  pose = holonomy) alongside the measured Fisher=margin 0.978 + quadratic-head + se3 anchors in the
  writeup/paper. No new lever, no DSL/equation change, no launch — measured content already captured; this
  is rigor/naming + two gated threads.
- **No speculative build.** Same disposition as the Fisher-slide + Many-Faces + Hoel-EI intakes: exact
  through-R measurement is authority; info-geometry is the framing/justification, never the score.
Pointer 0.19108282 UNMOVED.
