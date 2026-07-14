# Codex premise falsification: RIPO binary ratio law is not a K=5 logit-radius law

**UTC date:** 2026-07-14  
**lane:** `lane_ripo_fisher_isometric_trust_region_500_20260714`  
**task:** `500_ripo_fisher_isometric_trust_region_20260714`  
**status:** `PREMISE_FALSIFIED_BEFORE_ACTUATION`  
**verdict scope:** categorical-softmax geometry and the proposed direct transfer from a sampled-action
importance-ratio clip to an absolute witness-head logit radius; not a verdict on Fisher/KL trust
regions as a family  
**pointer:** `UNCHANGED`; no archive-closed exact score row

## Finding first

The intake claim

```text
||Delta logit_pixel|| <= sqrt(delta / p1_pixel)
```

is not the multiclass categorical Fisher trust region implied by RIPO. RIPO Eq. 10 is a bound on
one sampled action's probability ratio. A five-class softmax logit step is coupled by the simplex,
has a gauge null direction, and depends on all five probabilities and the proposed direction.

The related claim that a low-confidence argmax annulus receives a wider **absolute logit** region is
also false under the categorical Fisher metric. RIPO widens the **importance-ratio** budget for a
low-probability action. A near-tie winner and rival are normally the two highest-probability actions.

## What the paper actually derives

For old policy probabilities `p_a` and importance ratios `r_a = pi_new(a)/pi_old(a)`, RIPO writes

```text
D_KL(pi_old || pi_new)
  approximately 0.5 * sum_a p_a * (r_a - 1)^2.
```

It then allocates a one-coordinate contribution

```text
0.5 * p_a * (r_a - 1)^2 <= delta_10
```

and obtains Eq. 10

```text
|r_a - 1| <= sqrt(2 * delta_10 / p_a).
```

Eq. 11 renames the hyperparameter by absorbing the factor two:

```text
delta_11 = 2 * delta_10,
|r_a - 1| <= sqrt(delta_11 / p_a).
```

The paper's homoscedasticity proposition concerns the variance contribution of the importance-
weighted policy-gradient term at this clipped ratio boundary. It is not a proof that witness-head
logit updates have confidence-independent variance. The paper has no separate appendix in the
retrieved v1; the derivation and proposition appear in the main text.

## Exact K=5 derivation

At one pixel let `p = softmax(z)` have five strictly positive coordinates and let `u` be a proposed
logit change. The exact finite change is

```text
D_KL(p || softmax(z + u))
  = log(sum_k p_k * exp(u_k)) - sum_k p_k * u_k.                 (1)
```

Equation (1) is invariant under `u -> u + c*1`. Its second-order expansion is

```text
D_KL = 0.5 * u^T F(p) u + O(||P u||^3),
F(p) = diag(p) - p p^T,
P = I - 11^T/5.                                                  (2)
```

The quadratic can be written

```text
q_F(u; p)
  = Var_{k~p}(u_k)
  = sum_{i<j} p_i p_j (u_i-u_j)^2.                               (3)
```

Therefore a direction-dependent local trust-region clip is

```text
alpha_quad
  = min(1, sqrt(delta_11 / q_F(v; p)))
  = min(1, sqrt(2*delta_10 / q_F(v; p))),
u = alpha_quad * P v.                                            (4)
```

For finite moves the exact mode solves, separately for the supplied direction/sign,

```text
log(sum_k p_k exp(alpha*v_k)) - alpha*p^T v = delta_10,
alpha in [0,1].                                                   (5)
```

Positive and negative finite roots need not agree.

The largest gauge-fixed Euclidean ball contained in the local Fisher ellipsoid is

```text
||P u||_2 <= sqrt(delta_11 / lambda_max(F(p))).                  (6)
```

It still requires the full probability vector. Since the categorical Fisher spectral norm is at
most `1/2`, the finite bound

```text
D_KL <= ||P u||_2^2 / 4
```

gives a globally conservative, probability-independent ball
`||P u||_2 <= sqrt(2*delta_11) = 2*sqrt(delta_10)`.

## Winner-rival margin law

Let `w` and `r` be the current top two classes. For the minimum-Euclidean symmetric change of their
logit margin,

```text
u = t * (e_w - e_r) / 2,
Delta(z_w-z_r) = t,
C_wr = p_w + p_r - (p_w-p_r)^2.                                  (7)
```

Then

```text
q_F = t^2 * C_wr / 4,
|t|_quad <= 2 * sqrt(delta_11 / C_wr)
          = sqrt(8*delta_10 / C_wr).                              (8)
```

The exact finite KL along this direction is

```text
D(t) = log(1-s + p_w exp(t/2) + p_r exp(-t/2))
       - t*(p_w-p_r)/2,
s = p_w+p_r.                                                      (9)
```

This depends on both top probabilities and tail mass, not on `p1` alone.

### Exact versus approximate reductions

- **Winner versus rest:** exact only on the submanifold preserving every nonwinner conditional
  proportion. Its curvature is `p1*(1-p1)` and its local margin radius is
  `sqrt(delta_11 / (p1*(1-p1)))`. The intake law is missing `1-p1` even here.
- **Top two:** exact only when `s=p_w+p_r` and every tail probability remain fixed. With
  `p_tilde=p_w/s`, full KL equals `s` times the corresponding Bernoulli KL. A raw symmetric
  softmax-logit update generally changes `s`, so treating this as exact is invalid.
- **One independent RIPO ratio per class:** simplex-infeasible without coupling because
  `sum_k p_k r_k = 1`. Saturating five separately allocated action-coordinate budgets would also
  allow about `5*delta_10`, not the one-pixel budget `delta_10`.

## Constructive reversal of the annulus claim

For a near-tie pixel

```text
p_A = (0.45, 0.45, 0.05, 0.03, 0.02),
C_wr = 0.9,
|t|/sqrt(delta_11) = 2/sqrt(0.9) = 2.108185.
```

For a confident pixel

```text
p_I = (0.98, 0.01, 0.005, 0.003, 0.002),
C_wr = 0.0491,
|t|/sqrt(delta_11) = 2/sqrt(0.0491) = 9.025874.
```

The confident interior receives a `4.28x` wider Fisher-isometric absolute winner-rival logit
radius. The scalar `sqrt(delta/p1)` heuristic predicts the opposite: the near-tie pixel is `1.48x`
wider. This is a structural counterexample, not a tuning outcome.

Annulus pixels remain easier to flip because their starting margin magnitude is smaller. That
comes from the objective/required displacement, not from a larger Fisher radius. Preferentially
correcting the annulus therefore needs an annulus-selective direction, mask, or objective in
addition to the trust region.

## Receiver/head pullback boundary

The frozen SegNet probabilities and the witness `out_sdf` logits are not the same coordinates.
Between them lie palette rendering, texture, uint8/resize `R`, and frozen SegNet. For head
parameters `theta`, the exact local metric is

```text
G_head = sum_i J_i^T F(p_i) J_i,
J_i = d SegNet_logits_i(after R) / d theta.                       (10)
```

Using `p_i` to clip a pixel-aligned witness-head logit change without `J_i` is a measurable
cross-space approximation, not a Fisher-isometric head trust region. It must be compared against
controls through real `R`; a negative result scopes only that approximation. The live canonical
selection surface remains `argmax_native_vjp_fidelity_v1` with per-state receipt
`reachable_decision_geometry_fidelity.v1`.

## Delta provenance

Paper default delta is not transferable. For a measured pixel/direction, the authority value is

```text
delta_10,flip_i
  = D_KL(p_i || softmax(z_i + t_flip_i d_i)),                     (11)
```

where `t_flip_i` is the smallest measured through-`R` scale producing the intended frozen-SegNet
argmax flip. In the local winner-rival approximation,

```text
delta_11,flip_i approximately t_flip_i^2 C_wr_i / 4.             (12)
```

The sweep must come from quantiles of desired-correction thresholds subject to protected-pixel
spill thresholds, then be checked with exact KL and realized flips. A scalar margin is insufficient
unless direction, tail mass, and tie-crossing assumptions are declared.

The eikonal epsilon is an SDF spatial-gradient scale and is dimensionally not a SegNet-logit KL
budget. It cannot set delta without a measured Jacobian calibration.

The exact rate conversion is

```text
one Seg pixel improvement = 100/(600*384*512) = 8.4771050e-7 score,
one archive byte          = 25/37,545,489      = 6.6585895e-7 score,
break-even                = 1.2731082 bytes/net realized flip.
```

This ranks already measured candidates; it does not determine delta, curvature, flip direction,
spill, Pose harm, or byte effect.

## Resulting action

Build the full-K output-space clip and exact-KL audit. Test the cross-space witness-head
reprojection only under an explicit formulation label. Queue a true `G_head` block-Fisher/KL-
proximal reformulation if the approximation is not favorable. Do not encode the falsified p1 law
as a canonical Lever or equation.

## Stores consulted

- RIPO, *Beyond Euclidean Clipping*, arXiv:2607.10169, full v1 PDF/main methodology and proof text.
- `.omx/research/paper_warm_start_designs_recent_intake_20260714T160000Z.md`.
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`.
- `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`.
- `reports/latest.md` and `.omx/state/canonical_frontier_pointer.json`.
- `.omx/state/lane_registry.json`, `.omx/state/subagent_progress.jsonl`, and live inbox directives.
- Current EMA-best/checkpoint manifests and existing exact through-`R` frozen-replay receipts.

