# Task #504 — Bregman application to V9·CGauge surfaces

UTC: `2026-07-15T02:00:00Z`  
Scope: `$0` equations, invariants, and registry housekeeping only  
Pointer: UNMOVED (`0.1910828242 [contest-CPU Linux x86_64]`; the local
`0.1880443979880752` archive remains non-submission evidence)  
Canonical metric id: `argmax_native_vjp_fidelity_v1`

## 1. Definition and categorical ground metric

Use Nielsen's orientation

`B_F(x || y) = F(x) - F(y) - <grad F(y), x-y>`.

For categorical logits `z`, let `F(z)=log sum_k exp(z_k)` and
`p=softmax(z)`. Direct differentiation gives

`grad F(z)=p`,

`[H_F(z)]_ij = p_i delta_ij - p_i p_j`, hence

`H_F(z)=diag(p)-p p^T=Cov_p[e_Y]`.

Therefore the categorical output metric already selected by
`optimal_metric_unification_v1` is exactly the Bregman Hessian, not merely an
analogy. It is positive semidefinite on ambient logits and positive definite
only after quotienting the additive-logit gauge. The CGauge master action's A2
names the frozen-scorer pullback Fisher metric, so this result exactly supplies
the categorical output factor. It does **not** by itself supply custody that the
live V9 model computes the full scorer-VJP pullback, nor that the model factors
through the proposed affine/Legendre `(xi,R)` chart. That remaining claim is
`IMPLEMENTATION_CUSTODY_GAP_ONLY`, not a mathematical-family rejection.

For an affine-Legendre generator

`Fbar(theta)=lambda F(A theta+b)+<c,theta>+d`, `lambda>0`,

the chain rule gives

`grad Fbar=lambda A^T grad F(A theta+b)+c`,

`H_Fbar=lambda A^T H_F(A theta+b) A`, and substitution in the definition
cancels the affine gauge terms:

`B_Fbar(theta1||theta2)=lambda B_F(A theta1+b||A theta2+b)`.

This is a covariance law and a deterministic fixture in the implementation. It
is not evidence that the live witness realizes `A`, `b`, or the proposed chart.

## 2. Exact solve-free forms vs local metric forms

Substitution of `p=softmax(z)` into the definition yields the exact finite law

`B_logsumexp(z_S||z_T)=KL(p_T||p_S)`.

Legendre duality reverses the arguments:

`B_F(x||y)=B_F*(grad F(y)||grad F(x))`,

and adding both orientations cancels the generators:

`B_F(x||y)+B_F(y||x)=<grad F(x)-grad F(y),x-y>`.

These finite divergence identities are exactly solve-free. They must not be
renamed Fisher-natural distances. Their local expansion is

`B_F(y+delta||y)=1/2 delta^T H_F(y) delta + O(||delta||^3)`,

which is a local primal/tangent Fisher quadratic, not the finite Fisher–Rao
geodesic distance.

For dual displacement `delta_eta=H delta_theta`, the three local objects are

`delta_theta^T H delta_theta`
` = delta_eta^T H^{-1} delta_eta` (Fisher-natural; inverse/solve required),

while

`||delta_eta||_2^2`
` = delta_theta^T H^2 delta_theta` (no-solve raw dual; squared Hessian).

The existing `bregman_dual_metric_squared_hessian_v1` remains the owning guard.
The new closed-form equation references it and never aliases the two forms.

## 3. NO-FAKE invariants

For differentiable `F`, the first-order convexity inequality is exactly
`B_F(x||y)>=0` for every in-domain pair. Strict convexity gives equality only
at `x=y`; `logsumexp` needs the precise quotient statement: equality holds for
logits differing only by an additive constant. The implementation refuses
material negative values and clamps only sub-tolerance floating residue.

With positive normalized weights and samples in the **right** argument,

`J(c)=sum_i w_i B_F(c||theta_i)`,

differentiate:

`grad J(c)=grad F(c)-sum_i w_i grad F(theta_i)`.

Thus the unique right-data centroid is

`c=(grad F)^-1(sum_i w_i grad F(theta_i))`.

For `logsumexp`, the inverse is gauge-nonunique, so the implementation returns
the zero-mean representative of `log(sum_i w_i softmax(theta_i))` and checks the
first-order residual. The opposite orientation,
`argmin_c sum_i w_i B_F(theta_i||c)`, is the weighted arithmetic mean in primal
coordinates modulo gauge. Both are exposed so the asymmetric conventions
cannot be silently swapped.

## 4. Positive sigma rule and its honest endpoint

For dimension `D`, covariance `Sigma=L L^T`, and `kappa>0`, define

`chi_0=mu`, `w_0=kappa/(D+kappa)`,

`chi_i^±=mu ± sqrt(D+kappa) L_:i`,
`w_i^±=1/(2(D+kappa))`.

Pair symmetry proves `sum w chi=mu`; the paired outer products prove
`sum w(chi-mu)(chi-mu)^T=Sigma`. All weights are positive, so the right-data
Bregman centroid remains in its uniqueness domain.

After a nonlinear map, only the input mean/covariance match is exact. The
transformed Bregman centroid and its finite divergence dispersion are exact for
the chosen sigma support, but the support is an approximation to the full
nonlinear output distribution. In an exponential family the KL quadrature
error reduces exactly to

`(theta_p-theta_q)^T(eta_hat-eta_p)`;

therefore KL is exact only when the sufficient-statistic expectation is matched.

## 5. Lever classification and owed wires

All additions are canonical equations and implementation invariants, not DSL
levers. The divergence orientation, centroid convention, and sigma `kappa` are
generic mathematical API choices with no currently evidenced trainer consumer
or swept admission surface. Creating a DSL lever would fake a live actuator.

OWED, serialized as domain metadata:

- a trainer-consumed Bregman trust-region policy, only if it uses the exact
  finite divergence or the correctly typed local metric;
- a trainer-consumed sigma/centroid policy with a real sweep and acceptance
  receipt;
- curvelet and shearlet basis DSL wires after their owning arm lands a real
  consumer; no edit to `src/tac/witness_dsl/` is made here;
- live V9 affine/Legendre transform and full scorer-VJP pullback custody.

## STORES CONSULTED

- `src/tac/canonical_equations/optimal_metric_unification_20260714.py`
- `src/tac/information_geometry/optimal_metric.py`
- `src/tac/canonical_equations/bregman_v9_surfaces_20260714.py`
- `src/tac/canonical_equations/cgauge_master_action_20260711.py`
- `.omx/state/canonical_equations_registry.jsonl`
- `.omx/research/genuine_curvelet_shearlet_structural_proof_v2_polar_frequency_wedge_20260714.json`
- `.omx/research/bregman_v9_all_surfaces_build_spec_20260714.md`
- live per-arm and fleet inboxes through `2026-07-14T20:32:37Z`

## VERDICT-SCOPE

`DERIVED`: categorical ground metric, affine covariance, finite dual/cancellation
identities, convexity invariant, right-data centroid, positive sigma input
moments, and exponential-family expectation-error law.

`IMPLEMENTATION_CUSTODY_GAP_ONLY`: the live V9 affine/Legendre chart and full
scorer-pullback receipt are not present. `NO_VERDICT_DATA_CUSTODY`: no real-n600
metric/family selection follows. No score, promotion, training, or dispatch
claim is made.
