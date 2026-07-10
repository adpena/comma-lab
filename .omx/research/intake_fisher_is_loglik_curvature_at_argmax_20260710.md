# Intake: Fisher information = log-likelihood curvature at the MLE (info-geometry slide) — 2026-07-10

**Source:** operator-dropped slide, "Geometry of the likelihood curve: Likelihood curvature at MLE =
Fisher information = observed Fisher" (from Nielsen, *An Elementary Introduction to Information Geometry*,
Entropy 2020, 22(10):1100). Operator prompt: "Understand." This is a FRAMING/comprehension intake — the
GENERATING EXPLANATION for a fact we already MEASURED, not a new mechanism. Pointer 0.19108282 UNMOVED;
means/theory, honest verdict at bottom.

## The slide
Near the MLE: `l_x(θ) ≈ l_x(θ̂) − ½(θ−θ̂)² I(θ̂)` — the log-likelihood is a downward parabola whose
curvature IS the Fisher information `I(θ̂) = κ = 1/r` (osculating-circle radius r at the peak). Sharp peak
→ large Fisher → small Cramér–Rao variance `Var[θ̂]=I⁻¹` → stable estimate. Flat peak → small Fisher →
large variance → fragile estimate. `−l''(θ̂) = I(θ̂)` (observed Fisher = the Hessian at the MLE).

## Why this IS our geometry (exact identities, not analogy) — cross-validates measured anchors
The "likelihood" = frozen SegNet per-pixel class log-posterior `log p_θ(class|x)`; the "MLE" = the argmax;
so **Fisher at the argmax = margin curvature**. This is the MECHANISM behind the measured
**Fisher-curvature ↔ (−margin) Pearson 0.978** anchor ([[L1]] unified level-set flow):
- **Flat-interior (dark) = large Fisher** — confident argmax, sharp peak, won't flip, ~0 d_seg.
- **Boundary annulus = small Fisher** — near-tie, flat peak, high argmax variance → ALL d_seg lives on
  the codim-1 separatrix where Fisher is smallest (#333 annulus = boundary-jitter; ~97% d_seg in ~4.7% area).
- **R-roundtrip flips = Cramér–Rao** — small Fisher = large `I⁻¹` = argmax variance under the bicubic↑/
  uint8/bilinear↓ R-perturbation exceeds the margin → class crosses the decision boundary → flip. WHY
  round-trip-survival (R_surv) is a real lever and flips are texture-dependent.
- **UNIWARD = same metric, cost-reading** — "embed where the detector can't see" = "perturb where Fisher
  is low" = the flat-peak region (Fridrich inverse-steg).
- **Quadratic-head chart IS this Taylor bowl** — measured `quadratic_head_chart_subset_solve_gap_v1`
  (#341, LM ρ 0.85) = the per-pixel logit landscape near the boundary is a 2nd-order Taylor bowl with
  Hessian = observed Fisher → WHY damped-Newton / semi-discrete-OT head-offset (#288) is the right solver.
- **Degeneracy = small Fisher** — ties to [[intake_causal_emergence_effective_information_20260710]]:
  EI = Determinism − Degeneracy; degeneracy (distinct classes indistinguishable) = small margin = small
  Fisher. Same object, three vocabularies (info-geometry ↔ margin/Fisher ↔ causal-emergence).
- **GR unified action** ([[L10]]): the slide is the LOCAL (2nd-order) picture of the "ONE action S_τ in
  the Fisher metric"; τ-anneal raises the effective curvature (coarse→fine on the separatrix), consistent
  with τ=ε=ħ ([[L75]] #284 Amortizing-the-Argmax: the Laplace/WKB Gaussian width ~ √(τ/I)).

## Precision refinement (a real understanding-sharpening)
Our scorer is DETERMINISTIC → we use the **observed** Fisher (per-pixel Hessian of the frozen log-softmax
at the actual argmax), not the expected `−E[l'']`. They coincide at the MLE as n→∞; for us observed is the
correct object and the **margin field is its FIRST-order proxy** — the residual 0.022 in the 0.978
correlation is the gap between first-order margin and true second-order curvature.

## The one actionable thread (GATED — not build-now)
Cramér–Rao gives an analytic per-pixel R-flip-risk `≈ Φ(−margin / √(vᵀ I⁻¹ v))` (v = R-perturbation
direction) — a 2nd-order sharpening of margin-saliency (#141) / S_R reachability (#268) that could rank
WHICH boundary pixels flip under R → feed the which-flips-to-fix waterfill (#391) or the margin-aware
reweight (#274). **Honest verdict (same as the Hoel-EI thread):** our EXACT through-R S_R measurement IS
the ground truth for "does this pixel flip"; a Fisher/Cramér–Rao model is at most a **$0 analytic PRIOR /
ranking, never authority** (surrogate ≠ authority). It earns a slot in the #154 rate-probe / #268 queue
ONLY if it (a) predicts flip-structure the exact through-R S_R does NOT already capture, or (b) gives a
closed form that saves the through-R eval cost — measured, not asserted. Do NOT build speculatively.

## Routing
- **Bank** (this note): the info-geometry lens is the generating explanation for Fisher=margin=UNIWARD=
  curvelet-singularity=quadratic-head-bowl — cite Nielsen 2020 alongside the measured 0.978 anchor in the
  writeup/paper. No new lever, no DSL/equation change, no launch. The measured content is already captured
  by the Fisher/margin + quadratic-head + annulus anchors; this is naming/cross-reference.
- **Queue candidate** (NOT fired): Cramér–Rao per-pixel flip-risk as a $0 analytic prior in the #268/#154
  queue, gated on "beats exact through-R S_R at finding flips OR saves eval cost." Owner: whoever next
  works #268/#391.
Pointer 0.19108282 UNMOVED.
