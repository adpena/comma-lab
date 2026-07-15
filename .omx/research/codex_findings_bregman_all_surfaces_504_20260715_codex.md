# Codex findings — task #504 Bregman all surfaces

UTC: `2026-07-15T02:08:00Z`

Review scope: owned round-1 adversarial review

Pointer moved: `false`

## F1 — fixed: finite extreme logits could break the dual closed form

The initial dual-generator implementation converted logits to probabilities
and then required strict positivity. Finite logits such as `[1000,-1000,0]`
can underflow a softmax coordinate to numeric zero, causing a false domain
failure even though the categorical KL is finite. Fixed by adding a stable
logit-domain KL evaluation (`z-logsumexp(z)`) as the numerical-boundary fallback.
The direct negative-entropy evaluation remains the interior path so registered
fixture bytes stay unchanged. Regression: extreme opposed logits yield finite
`2000`-nat divergence and preserve the dual identity.

## F2 — confirmed: centroid orientation is explicit and correct

Under this repository's definition `B_F(point||reference)`, samples in the
right argument give

`argmin_c sum_i w_i B_F(c||theta_i)`
`=(grad F)^-1(sum_i w_i grad F(theta_i))`.

The opposite orientation gives the primal arithmetic mean. Both callables are
present and tests prove they are generally different. For `logsumexp`, unique
means unique on the additive-logit quotient; the implementation fixes a
zero-mean logit representative.

## F3 — confirmed: Fisher-natural vs squared-Hessian is not conflated

- exact finite Bregman/KL and dual/cancellation forms: solve-free;
- local primal tangent: `dtheta^T H dtheta`;
- Fisher-natural cotangent: `deta^T solve(H,deta)`;
- raw no-solve dual Euclidean: `||deta||^2=dtheta^T H^2 dtheta`.

No callable or equation aliases the final line to Fisher-natural geometry.

## F4 — confirmed: sigma-point exactness stops at the evidenced boundary

The positive `2D+1` sigma rule exactly reconstructs input mean/covariance.
After a nonlinear map, the transformed support and its Bregman centroid are
computed exactly for those points, but the full output integral remains
approximate. Exponential-family KL is exact only if the sufficient-statistic
expectation matches; its exact error is
`(theta_p-theta_q)^T(eta_hat-eta_p)`.

## F5 — confirmed: equation ledger was mutated through the canonical API

Pre-registration query counts were zero for curvelet, compact shearlet, and
five new Bregman application equations. Population used the module-owned
helpers, each routing through locked `register_canonical_equation`; registry
diff is exactly `+7/-0`. Post-registration `query_equations()` and the canonical
listing CLI surface each requested target exactly once.

## F6 — confirmed: shearlet custody lines remain separate

The landed `compact_shearlet_frame.py` and the newer compiled-frame structural
proof cite different source SHA-256 values. The canonical equation records both
and explicitly sets `source_equivalence_claim=false`. The primitive's own
executable swap-test supplies its anchor. The newer proof remains separate
structural context with `NO_VERDICT_DATA_CUSTODY`; no invalidated rank, equal-byte
claim, frame-tightness claim, score, or family verdict enters the equation.

## Sources checked

- Frank Nielsen, *A note on the Artstein-Avidan-Milman's generalized Legendre
  transforms*, arXiv:2507.20577v2: <https://arxiv.org/abs/2507.20577>
- Frank Nielsen and Richard Nock, *On the Centroids of Symmetrized Bregman
  Divergences*, arXiv:0711.3242: <https://arxiv.org/abs/0711.3242>

## Verdict

`ROUND1_PASS_AFTER_FIX`. The only implementation defect found was F1 and it is
regression-protected. CGauge categorical metric identity is `DERIVED_EXACT`;
live affine/Legendre and full scorer-pullback status remains
`IMPLEMENTATION_CUSTODY_GAP_ONLY`. No real DSL lever exists in this landing;
all such wires remain explicitly OWED.
