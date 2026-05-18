# Deterministic optimizer directive supplement: alternative mathematical frameworks beyond Lagrangian/Taylor/Pareto
# Date: 2026-05-18
# Audience: in-flight subagent acb41f8d3f7f0a3ea (deterministic score optimizer + Wyner-Ziv Q4 anchor)
# Per CLAUDE.md "Subagent coherence-by-default" inter-agent directive pattern

## Operator directive (verbatim 2026-05-18, AFTER subagent spawn)

> *"is the master-gradient exposed or wired up with existing or tools we should create for boundaries and master gradient boundaries and xray and hard pairs and sensitive bytes and sensitivity map and all other means of analyzing and calculating and modeling the data so as to be able to use in place of current arbitrariness"*

> *"there are likely more mathematically elegant and appropriate means of representing and solving"*

## What this directive adds to your existing scope

Your DELIVERABLE 1 (`deterministic_score_optimizer_design_memo_lagrangian_taylor_pareto_reverse_engineering_20260518.md`) currently scopes Lagrangian + Taylor + Pareto. ADD a new mandatory section:

### `## Alternative mathematical frameworks — comparative analysis`

For EACH framework, document: (a) why it's potentially MORE appropriate than Lagrangian/Taylor/Pareto for the contest scorer's specific structure; (b) the implementation complexity; (c) the convergence guarantees; (d) the operator-readability/observability cost; (e) the existing canonical-helper integration story.

**Frameworks to evaluate** (ranked by likely appropriateness for the contest scorer's specific structure):

1. **Tropical / max-plus algebra** — d_seg is literally an argmax over 5 SegNet classes per pixel. Tropical geometry (max-plus semiring) is the canonical algebra of piecewise-linear-via-argmax functions. Per pixel: `d_seg_pixel = 1 if argmax(logits_orig) ≠ argmax(logits_recon) else 0`. The tropical sum-of-max-plus-products structure makes the gradient subdifferential analytically tractable WITHOUT the subgradient approximation Boyd-style convex methods require. Canonical reference: Maclagan & Sturmfels, *Introduction to Tropical Geometry* (2015). Open-source: `tropical` Python library (limited but exists).

2. **Proximal splitting (Douglas-Rachford / primal-dual hybrid gradient)** — the score has additive separable structure `100·d_seg + sqrt(10·d_pose) + 25·rate`. Each term has different smoothness: d_seg is NONSMOOTH (argmax), d_pose is SMOOTH (MSE), rate is LINEAR. Proximal splitting handles each term with its OWN appropriate proximal operator. Canonical reference: Combettes & Pesquet, *Proximal Splitting Methods in Signal Processing* (2011). Open-source: `cvxpy` + `proxpy`.

3. **Algebraic geometry / Gröbner basis decomposition** — solve the KKT polynomial system EXHAUSTIVELY. Decompose `∂L/∂theta = 0` into irreducible components via Buchberger's algorithm → enumerate ALL stationary points (not just local minima). Polynomial-time IF d_seg is polynomial; for argmax-piecewise-constant d_seg, need tropical-Gröbner extension. Canonical reference: Cox-Little-O'Shea, *Ideals, Varieties, and Algorithms* (2015). Open-source: `sympy` Gröbner + `Macaulay2` + `singular`. SciPy doesn't ship Gröbner.

4. **Submodular optimization with Lovász extension** — if archive_bytes-vs-score has SUBMODULAR structure (diminishing returns as more bytes are added), Lovász extension gives a CONVEX continuous relaxation that's solvable in polynomial time via matroid intersection. Tests submodularity by checking: for any S ⊂ T ⊂ U and element x ∉ T: `f(S ∪ {x}) - f(S) ≥ f(T ∪ {x}) - f(T)`. Canonical reference: Bach, *Learning with Submodular Functions* (2013). Open-source: `apricot`, `submodlib`.

5. **Tishby Information Bottleneck** — already canonical in repo via Catalog #319 Q1-Q5. The contest scorer's structure as `min I(theta; archive_bytes) subject to I(archive_bytes; frames) ≥ k_seg, I(archive_bytes; poses) ≥ k_pose` is the IB formulation. Wyner-Ziv source coding with decoder side-info is the EXACT match for "shipping seed bytes + reconstructing codebook at inflate time." Already partially landed via `src/tac/wyner_ziv_deliverability/`.

6. **Optimal transport / Wasserstein gradient flows** — treats the score function as a measure-valued objective. Particularly elegant for codebook design: the codebook is a discrete probability measure, and Wasserstein-2 distance between empirical archive distribution + theoretical target distribution gives a natural objective. Canonical reference: Santambrogio, *Optimal Transport for Applied Mathematicians* (2015). Open-source: `POT` (Python Optimal Transport) library.

7. **Mirror descent on the simplex (Bregman divergence)** — appropriate for the Pareto-simplex sweep (alpha + beta + gamma = 1). Replaces L2 gradient steps with KL-divergence steps on the simplex; converges faster than projected gradient descent. Canonical reference: Nemirovski & Yudin, *Problem Complexity and Method Efficiency in Optimization* (1983). Open-source: implement via `numpy` + `scipy.special.softmax`.

8. **Stochastic Variance-Reduced Gradient (SVRG / SAGA)** — variance reduction for finite-sum objectives. The contest scorer evaluates over 600 pairs; SVRG reduces gradient variance by O(1/n) per pair → faster convergence than vanilla SGD. Canonical reference: Johnson & Zhang, *Accelerating Stochastic Gradient Descent using Predictive Variance Reduction* (2013). Open-source: directly implementable.

9. **Trust-region Sequential Quadratic Programming (SQP)** — handles HARD constraints (contest compliance = yes/no) via trust-region quadratic subproblems with active-set methods. Canonical reference: Nocedal & Wright, *Numerical Optimization* (2006). Open-source: `scipy.optimize.minimize(method='trust-constr')`.

10. **Game-theoretic / minimax** — frame the optimization as a 2-player game: codec player wants to minimize score; scorer player wants to maximize it (within scorer's fixed weights). Nash equilibrium gives the optimal codec. Particularly relevant if we ever train an adversarial-codec against a frozen scorer. Canonical reference: Goodfellow's GAN formulation (2014). Open-source: directly implementable.

11. **Frank-Wolfe (conditional gradient method)** — for the case where the feasible region (archive.zip bytes) has a TRACTABLE LINEAR MINIMIZATION ORACLE. Each iteration solves `min_x <∇f(x_k), x>` over the constraint set, then convex-combines with current iterate. Avoids projection — useful if the constraint set is complex. Canonical reference: Jaggi, *Revisiting Frank-Wolfe* (2013). Open-source: `pyfw`.

12. **L-BFGS / quasi-Newton** — second-order method using rank-1 updates to approximate the Hessian. Already mentioned in Catalog #319 sister-context. Native to `scipy.optimize.minimize(method='L-BFGS-B')`.

### Mandatory new sections to add to DELIVERABLE 1

After the existing `## Local Taylor expansion` and `## KKT optimality conditions` sections, ADD:

- `## Framework recommendation matrix` — table cross-referencing each framework against (contest-scorer-structure-fit / convergence-guarantees / implementation-complexity / observability / canonical-helper-integration-cost)
- `## Hybrid framework proposal` — almost certainly the OPTIMAL design uses a COMPOSITION of frameworks: tropical for d_seg + proximal splitting for the multi-term decomposition + Wyner-Ziv for codebook compression + mirror descent for Pareto-simplex sweep
- `## Existing repo integration story per framework` — which canonical helpers exist? Which need to be built?

### Cross-reference with Q1 (master-gradient wire-in)

Operator's question 1 is about whether master-gradient is wired to analytical surfaces. The answer per sister synthesis: PARTIAL (4 consumers landed; 6+ wire-ins pending per synthesis OP-4). The deterministic-optimizer design memo should explicitly note:

- The WIRE-IN work (OP-4 from synthesis directive) is FRAMEWORK-AGNOSTIC — once wired, any of the 12 frameworks above can consume the analytical surfaces
- The FRAMEWORK CHOICE is INDEPENDENT of wire-in work — both can proceed in parallel
- The CRITICAL DEPENDENCY: framework choice INFORMS what data the wire-in surfaces should emit. E.g., tropical algebra needs per-pixel argmax-stability indicators; submodular needs per-byte marginal-return curves; proximal splitting needs per-term smoothness classification.

## INSTRUCTION TO acb41f8d3f7f0a3ea

If you are this subagent, INCORPORATE the alternative-frameworks comparative analysis into your DELIVERABLE 1. Expand from ~1500-2500 lines to ~2500-3500 lines as needed.

The CRITICAL CONCLUSION the operator needs from your deliverable: **which framework (or framework composition) gives the most mathematically elegant + tractable + observability-rich + canonical-helper-aligned path to replacing current arbitrariness with deterministic optimality?**

Acknowledge this directive in your next checkpoint via `tools/subagent_checkpoint.py --notes "incorporated deterministic_optimizer_alternative_mathematical_frameworks_directive_20260518"`.

— Main-Claude (relayed on behalf of operator 2026-05-18)
