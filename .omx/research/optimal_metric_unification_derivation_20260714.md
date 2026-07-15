# The OPTIMAL METRIC unification — one `g = ∇²F` for fidelity, loss, curriculum (2026-07-14)

**Mode:** `research_only=true`, `$0` design/derivation, no training, no dispatch.
**Authority:** `[macOS-CPU advisory / NumPy-fp32; no score authority]`.
**Pointer UNMOVED:** 0.19108 submittable / 0.18804 borrowed bank. All below is MEANS —
this is apparatus/design (a unified law), not a score-mover.

**Task (operator #500):** unify — do NOT re-derive from scratch — the already-landed
information-geometry pieces into ONE reachable-decision-geometry metric that is
simultaneously the fidelity predicate, the training-loss geometry, and a
curriculum-varying metric.

## The single object

`F(θ) = logsumexp(θ)` is the convex Bregman generator of the categorical (softmax)
exponential family. Its Hessian **is** the metric:

    g(p) = ∇²F = diag(p) − p pᵀ,   p = softmax(θ)

This is the categorical Fisher information, equivalently the Bregman Hessian of the
log-partition (Nielsen). Everything below is a *reduction of this ONE object*,
computed (not asserted) in `src/tac/information_geometry/optimal_metric.py`.

## Role 1 — fidelity (surrogate admission at the argmax boundary) — EXACT reduction

The reachable trust region at the winner↔runner-up boundary is the Fisher ball. Its
directional curvature along `u = e_w − e_r` is a *primal / tangent* quadratic form of
`g`:

    C_wr = (e_w − e_r)ᵀ g (e_w − e_r)
         = [diag term: p_w + p_r] − [ppᵀ term: (p_w − p_r)²]
         = p_w + p_r − (p_w − p_r)²

so the directional trust radius is `|t| ≤ √(8·δ_KL / C_wr)` (RIPO). **This reduction is
EXACT and bit-verified**: `metric_directional_quadratic(p, e_w−e_r)` equals the
independently-implemented `tac.optimization.ripo_fisher_trust_region.winner_rival_curvature`
to fp64 (test `test_fidelity_reduction_bit_equal_to_ripo`). This is a *tangent-space*
quadratic — **no `H⁻¹` solve** — so it stays Fisher-natural in logit coordinates. It
**replaces raw cosine** (D42): cosine has zero authority here — the FALSE binary
transfer `√(δ/p_w)` orders pixels almost perfectly OPPOSITELY to the correct radius
(MEASURED Spearman **−0.96**, real SegNet K=5, 18.87M px). Canonical eq:
`categorical_fisher_trust_region_winner_rival_v1` (the OWED RIPO registration — DONE here).

## Role 2 — training-loss (what the witness descends) — PARTIAL / honest surrogate

Specialize `g` to the two-class annulus (winner vs rival; the other K−2 are ≈0 at the
boundary). Its trace is a monotone function of the logit margin `m`:

    tr g|_{2-class} = 2 p (1−p) = ½ sech²(m/2),   p = σ(m)

This is *why* the measured curvature↔(−margin) Pearson is **0.978** (Spearman 0.908) —
the caustic anchor `curvature_neg_margin_pearson_0978_spearman_0908_caustic_20260704`
(deepmath_amortizing_argmax_laws). The witness descends the margin field
(UNIWARD steg-cost = the same metric read as cost), so the training-loss geometry IS a
reading of `g`.

**HONEST GAP (named, not papered over):** this is the *only* reduction that is NOT
exact. The witness descends a **scalar** margin surrogate for **tr(g)** (a two-class
band quantity), not the full K=5 directional metric `g`. The link is a **measured 0.978
band calibration**, not a global identity. So the unification is EXACT on fidelity +
curriculum and a **measured surrogate** on the training-loss leg — a partial unification
with a named gap, which is the honest state. (A future exact loss would descend the full
directional `C_wr` field rather than the scalar margin — that is a design direction, not
a claim.)

## Role 3 — curriculum-varying (g changes with τ) — DERIVED

At softmax temperature τ, in fixed-logit coordinates:

    g(τ) = (1/τ²) (diag(p_τ) − p_τ p_τᵀ),   p_τ = softmax(θ/τ)

- the **1/τ² prefactor** is the chain-rule pullback of the natural-coordinate Fisher
  metric through `η = θ/τ` (DERIVED, not asserted);
- the **operating point** `p_τ` also varies with τ, so `C_wr(τ)` and `p_w(τ)` evolve.

As τ ↓ the metric **concentrates onto the separatrix**: interior mass
`p(1−p) ~ e^{−m/τ}` decays faster than the `1/τ²` blow-up, while the boundary `p≈0.5`
grows like `1/τ²`. MEASURED in the test (`test_curriculum_concentrates...`): at
logits `[4,1,−1,−2,−3]`, `C_wr_natural` = 1.87e-1 → 9.9e-3 → 3.7e-13 as τ = 1.0 → 0.5 →
0.1 (p_w 0.943 → 0.997 → 1.000). This is the curvelet coarse→fine = temperature-anneal
facet: the CE→τ→l7 curriculum sharpens the metric onto the boundary annulus.

## NO-FAKE honesty — squared-Hessian is NOT Fisher-natural (landed guard honored)

All three roles use `g` in its **primal / tangent** quadratic form `Δθᵀ g Δθ` — that IS
Fisher-natural in logit coordinates, **no `H⁻¹` solve**. The DUAL raw-mean *no-solve*
length is `Δθᵀ g² Δθ` (the **squared** Hessian), which is NOT the Fisher-natural
cotangent length `Δηᵀ g⁻¹ Δη` (that one needs an `H⁻¹` solve). This module never
conflates them; `squared_metric_quadratic` exists only to make the distinction testable
(`test_squared_hessian_differs_from_primal_quadratic`) and cross-references the landed
guard `bregman_dual_metric_squared_hessian_v1` as the single source of that truth.

## Covariant (CGauge) extension

The seg-head categorical-Fisher unified here is the piece unified EXACTLY. The full
covariant witness metric adds the pose/gauge fibre read through the YUV6 pullback (the
margin-0.978 calibration is CGauge assumption A2 in `cgauge_master_action_20260711`).
The covariant version is the registered sister — the unification's `canonical_consumers`
link to it; a full covariant merge is a design direction, not claimed here.

## Value-provenance ladder

| quantity | rung |
|---|---|
| `C_wr = p_w + p_r − (p_w − p_r)²` | DERIVED (directional quadratic form of `g`; registry-evaluable via the callable) |
| `|t| = √(8·δ_KL / C_wr)` | DERIVED (KL 2nd-order expansion) |
| `tr g|_{2-class} = ½ sech²(m/2)` | DERIVED (exact identity) |
| `g(τ) = τ⁻² (diag(p_τ) − p_τ p_τᵀ)` | DERIVED (chain-rule prefactor + softmax(θ/τ)) |
| Pearson **0.978** / Spearman **0.908** | MEASURED-ANCHOR (deepmath caustic anchor, cited) |
| RIPO Spearman **−0.9601**, ratio median **16.34×** | MEASURED-ANCHOR (RIPO receipt JSON, cited) |

## What is REGISTERED (this landing)

- `optimal_metric_unification_v1` — the single metric + three reductions.
- `categorical_fisher_trust_region_winner_rival_v1` — the OWED RIPO directional law.

Both: producers = the measurement/derivation modules; consumers = the surrogate-admission
path (RIPO radius), the witness loss (margin field), the curriculum/covariant sister
(cgauge). No orphan. Callables actually compute (NO-FAKE); tests prove the fidelity
reduction is bit-real.

## OWED (serialized follow-ups — NOT done here)

- **DSL / curriculum trainer wire** of the τ-varying metric is OWED (the `witness_dsl`
  arm owns `src/tac/witness_dsl/` right now — merge collision). The equation-level
  `canonical_consumers` declare the surfaces; the actual DSL `Lever` wire is a serialized
  follow-up.
- **D42 surrogate re-admission** under the correct directional locus stays OPEN-CUSTODY
  (re-capture task, not a $0 recompute) per the RIPO memo.

## Stores consulted

`[[ripo_categorical_fisher_trust_region_falsification_20260714]]`,
`[[dual_metric_no_solve_is_squared_hessian_not_fisher_natural_20260714]]`,
`.omx/research/ripo_categorical_fisher_binary_vs_directional_MEASURED_20260714.md`,
`src/tac/canonical_equations/bregman_v9_surfaces_20260714.py`,
`src/tac/canonical_equations/deepmath_amortizing_argmax_laws_20260704.py`,
`src/tac/canonical_equations/cgauge_master_action_20260711.py`,
`src/tac/optimization/ripo_fisher_trust_region.py`.

Pointer delta: 0.0000000000.
