# Intake: Nielsen, "The Many Faces of Information Geometry" (Notices AMS Jan 2022) — 2026-07-10

**Source:** operator-dropped `https://www.ams.org/journals/notices/202201/rnoti-p36.pdf` — Frank Nielsen,
*The Many Faces of Information Geometry*, Notices of the AMS, Jan 2022, Vol 69(1):36. SAME author as the
Fisher-curvature slide dropped earlier today ([[intake_fisher_is_loglik_curvature_at_argmax_20260710]]) —
the operator is building the rigorous info-geometry SPINE of our measured geometry. This is a FRAMING/theory
intake (the deep survey behind the slide), NOT a score-mover. Pointer 0.19108282 UNMOVED; honest verdict
at bottom. (30MB PDF held in session scratchpad only — ephemeral, NOT a durable evidence path.)

## What the survey covers (the parts new beyond the slide)
- **Fisher metric** = FIM = covariance of the score `I_X(θ)=E[s_θ s_θᵀ]`, covariant under reparameterization;
  Fisher-Rao distance = geodesic length, parameterization-invariant. Constant sectional curvature for some
  families (normal κ=−½, categorical κ=+¼, zero-centered normal κ=0).
- **KL 2nd-order Taylor = ½dθᵀ I_θ dθ = ½ds²** — the earlier slide, stated exactly. Cramér-Rao
  `Var[θ̂] ≥ (1/n)I⁻¹` — also the slide.
- **f-divergences + monotonicity under Markov kernels** (the data-processing inequality), equality iff
  sufficient statistic; f-div = the only separable invariant divergences (n>2).
- **Exponential/mixture families = dually flat spaces**; FIM = Hessian of the log-partition `I(θ)=∇²F(θ)`
  (Hessian/Shima geometry); e-connection ∇ / m-connection ∇* / Levi-Civita = midpoint; Amari-Chentsov
  skewness tensor; dual α-geometry.
- **Chentsov's theorem**: the Fisher metric is the UNIQUE invariant metric under Markov morphisms /
  sufficient statistics (up to scale).
- **Wasserstein information metric (Li21)** flagged as a DISTINCT alternative to the FIM.

## The FOUR pieces load-bearing for us
1. **Chentsov = the rigorous ground under "code the task-sufficient statistic."** Fisher/margin is the ONLY
   distortion metric invariant under the scorer's sufficient-statistic reduction. RGB-L2 is NOT invariant
   under the SegNet argmax reduction; the Fisher metric IS. ⇒ spending bytes on the scorer-relevant manifold
   in the margin metric (not full-RGB L2) is descending in the CANONICALLY-CORRECT geometry for a
   sufficient-statistic codec — a theorem, not a heuristic. **Writeup/paper citation for the task-space
   capstone** ([[L17]] task-sufficient statistic; [[L74]] indirect-RD; #155 quotient codec).
2. **f-div monotonicity = DPI through R.** R (bicubic↑→uint8→bilinear↓) is a near-deterministic Markov
   kernel; d_seg measured AFTER R ⇒ info can only be LOST through R = WHY boundary flips happen (margin
   distinguishability erased). Same statement as Hoel degeneracy
   ([[intake_causal_emergence_effective_information_20260710]]): coarse-graining that preserves the
   sufficient statistic loses no EI; the witness IS that coarse-graining. Three vocabularies, one theorem.
3. **FIM = Hessian of log-partition** makes the quadratic-head chart (#341
   `quadratic_head_chart_subset_solve_gap_v1`) rigorous: near the argmax the logit landscape is the Hessian
   of a convex potential ⇒ damped-Newton / semi-discrete-OT head-offset (#288) is Newton in this Hessian
   geometry; e/m Legendre duality is the structure τ-anneal moves through (natural ↔ expectation params)
   ([[L10]] GR action; [[L75]] τ=ε=ħ).
4. **Wasserstein vs Fisher (the one ACTIONABLE distinction).** d_seg distortion lives in the Fisher/margin
   metric (statistical distinguishability of the argmax); our OT levers — semi-discrete-OT head offset
   (#288), area-mass-match (#382), v8 Laguerre power-diagram generator (#284/v8) — live in the
   Wasserstein/transport metric. The witness sits at the INTERFACE of two distinct geometries; conflating
   them is a real trap — the MEASURED "OT area-mass-match HURTS d_seg → flip-weighting instead"
   ([[lane-groundframe-xi-transport-no-collapse-chart-selection-law]]) is exactly a Wasserstein-optimal /
   Fisher-suboptimal case.

## Honest verdict + routing
- **Bank** (this note): the info-geometry survey is the rigorous spine — cite Nielsen 2022 (Chentsov
  uniqueness + dually-flat + Wasserstein-vs-Fisher) alongside the measured Fisher=margin 0.978 anchor in
  the writeup/paper. No new lever, no DSL/equation change, no launch — the measured content is already
  captured by the Fisher/margin + quadratic-head + annulus anchors; this is rigor/naming/cross-reference.
- **Framing distinction to HOLD (not a build):** which metric governs which term — Fisher for d_seg
  (distinguishability), Wasserstein for the OT/warp/Laguerre-generator geometry (mass transport). Guards
  against Wasserstein-optimal / Fisher-suboptimal moves. Owner: whoever next touches #288/#382/v8 OT levers.
- **No speculative build.** Same disposition as the Fisher-slide + Hoel-EI intakes: our exact through-R
  measurement is the authority; info-geometry is the framing/justification, never the score.
Pointer 0.19108282 UNMOVED.
