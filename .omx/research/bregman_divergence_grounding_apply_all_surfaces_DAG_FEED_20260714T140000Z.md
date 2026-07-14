# DAG FEED — Bregman-divergence framework (Nielsen, DOxML 2026) → grounds & applies to ALL v9·CGauge surfaces

**Operator 2026-07-14:** "@DOxML2026Frank.pdf Understand and apply all surfaces." Frank Nielsen, "Short
stories on Bregman divergences — Flat and curved" (Sony CSL). Six stories, each a surface.

## The core identities (rigorous foundation of the optimal-metric thread #500)
- **Bregman divergence** B_F(θ₁:θ₂)=F(θ₁)−F(θ₂)−⟨θ₁−θ₂,∇F(θ₂)⟩ = convexity gap = Taylor remainder.
- **Generalized-quadratic / MAHALANOBIS** B_F=½(θ₂−θ₁)ᵀ∇²F(θ)(θ₂−θ₁): the Hessian ∇²F IS the metric.
  ⇒ our reachable-decision-geometry metric ⟨g,g⟩_(R,M) with M=Fisher IS a flat Bregman/Mahalanobis divergence.
- **Hessian metric** g_ij(θ)=∇²F(θ); **Crouzeix** ∇²F(θ)·∇²F*(η)=I (primal↔dual metric inverse).
- **Squared-Hessian ⇒ EUCLIDEAN on dual coords:** ρ(P₁,P₂)=‖η₁−η₂‖₂=‖∇F(θ₁)−∇F(θ₂)‖₂, η=∇F(θ).
  ⇒ THE cheap metric form: map to dual (mean) coords, take Euclidean distance — no Fisher-matrix solve.
- **KL(exp-family) = reverse Bregman** in natural params (SegNet softmax = categorical EF).

## The six stories → our surfaces (apply)
1. **Cumulant/partition BD ⇒ reverse/extended KLD** (2312.12849). Grounds CE=KL=Bregman(neg-entropy) — the
   metric arm's "CE is already the negentropy Bregman." Loss geometry = Bregman generator choice per stage.
2. **Chernoff information / LREF** (2012.15480). The Chernoff point = optimal α where D(P*||P₁)=D(P*||P₂)
   (divergence Voronoi bisector) ⇒ the PRINCIPLED seg-vs-pose (or any 2-term) optimal operating point +
   the geometric-mixture e-geodesic. Applies to the operating-point-dependent seg/pose marginal tradeoff.
3. **Sigma points** (2003.02469): KL(EF) = (1/s)Σ log-density-ratio at ≤D+1 chosen ω_i with
   (1/s)Σt(ω_i)=E[t]; MC error EXACTLY = (θ₂−θ₁)ᵀ((1/m)Σt(x_i)−E[t]). ⇒ THROUGHPUT lever: exact cheap KL
   with a few points instead of the full integral/n600, for any EF-KL term. + the extended NON-NEGATIVE
   f-divergence (naive MC KL can go NEGATIVE = a real bug; extended form B_f(q/p:1)≥0 fixes it) = anti-bug.
4. **Curved Bregman + centroid** (2504.05654): **cosine 1−cos is a CURVED BD, NOT a proper Bregman** (F
   constant on the circle) ⇒ RIGOROUS grounding of the surrogate "cosine is the wrong metric" finding —
   the proper metric is the FLAT Mahalanobis. **Curved-BD centroid = Bregman-PROJECTION of the full
   centroid** ("n-point opt → 1-point opt") ⇒ SOLVER lever for waterfill/TerminalSolve/per-class solves:
   compute the closed-form full centroid θ̄=Σwᵢθᵢ, then project ONCE onto the constraint submanifold.
5. **Bregman gauge freedom** (2507.20577) = the mathematical grounding of **V9·CGauge (Covariant-Gauge)**:
   B_F=(1/λ)B_F̄ under F̄=λF(Aθ+b)+⟨c,θ⟩+d (affine Legendre invariance + divergence unit); dually-flat
   (M,g,∇,∇*). **RECALL CORRECTION (graph-memory):** CGauge is NOT undefined — `cgauge_master_action_v1`
   (`tac.canonical_equations.cgauge_master_action_20260711`) ALREADY registers a covariant Lagrangian
   S=100·D_seg+√(10·D_pose)+(25/N)·L_MDL **on the Fisher base, covariant under (ξ,R)** — i.e. a Bregman-
   adjacent grounding exists. The NO-FAKE verify is therefore SHARPER: does the CODE implement
   `cgauge_master_action_v1` (the gauge covariance), or is the covariance a paper-equation the trainer
   doesn't honor? Ground the Nielsen gauge-freedom INTO the existing master action; flag any code gap to the
   fake-fix arm. Also recall FEED-v9div: the v9·CGauge run DIVERGED (best d_seg 0.03482, ~7× above baseline)
   — a live post-mortem the grounding must reconcile with.
6. **Riemann-Bregman vs Euclidean** (2511.21173): separable BD ⇒ Euclidean on h(x)=∫√φ''; squared Hessian
   ⇒ Euclidean on dual params (the cheap form above). Rao/geodesic distance = flat Euclidean in the right chart.

## Apply-to-all-surfaces map
- **#500 optimal metric (LIVE):** ground argmax_native_vjp_fidelity_v1 as a flat Bregman/Mahalanobis
  divergence; add the DUAL-EUCLIDEAN cheap form (‖∇F(θ₁)−∇F(θ₂)‖) + Crouzeix; cosine=curved-BD grounding.
- **loss / curriculum:** per-stage Bregman generator = the metric-anneal (already derived); grounded now.
- **throughput:** sigma-point exact-KL for EF-KL terms; extended-non-negative estimator (anti-bug).
- **CGauge vehicle:** verify/ground the gauge-freedom (NO-FAKE).
- **solvers (waterfill/TerminalSolve/#423):** n-point→1-point curved-centroid projection.
- **pose/seg operating point:** Chernoff point.
- **basis (#502):** Gram ⟨ψ,G_qψ⟩ with G_q=∇²F Hessian — same metric.

**Pointer:** 0.19108 / 0.18804 UNMOVED. This is the rigorous grounding + cheap-form + anti-bug + throughput
levers for the metric/CGauge thread; the pointer moves via a byte-closed exact row through the grounded vehicle.

**STORES CONSULTED:** (honestly) FIRST the in-session committed artifacts — the optimal-metric-unification
DAG FEED (c05acb2135 lineage), the surrogate/metric arm findings (argmax_native_vjp_fidelity_v1, "cosine is
wrong"), the curvelet-fake + −48%-unreproduced memories, the fake-audit ledger (6408b07ef0), CLAUDE.md
(operating-point seg-vs-pose, NO-FAKE). THEN (post-hook, before re-finishing) a fresh recall: graph-memory
(`tools/graph_memory_recall.py`) which surfaced `cgauge_master_action_v1` + `cgauge_parametrization_optima_v1`
+ FEED-v9div (the CGauge divergence post-mortem) + FEED-cgauge-master; and `git grep` of
`src/tac/canonical_equations/` (found cgauge_master_action_20260711, cgauge_parametrization_optima_20260711)
+ the canonical_equations_registry.jsonl. GAP (honest): the graph-memory/equations recall was done AFTER the
first synthesis write (in-session context only), not before — this correction note + the story-5 CGauge
refinement above are the result. The `cgauge_master_action_v1` connection is the key recovered signal.
