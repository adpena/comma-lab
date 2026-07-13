# Jacobian-drift direct-full-costate certificate — implementation specification

Date: 2026-07-13 UTC  
Lane: `lane_jacobian_drift_certificate_95kill_20260713`  
Authority: local theorem, synthetic rigorous canaries, and real `[macOS-CPU advisory; torch-fp32; training-signal]` measurement only. `research_only=true`; `score_claim=false`; `pointer_moved=false`; no paid/cloud/heavy dispatch.

## Outcome sought

Extend the completed task-454 mechanism without changing its current-prefix-VJP/banked-suffix behavior. Add a second, explicit provider mode for direct reuse of a full scorer-input costate. The new mode may emit `CERTIFIED_REUSE` only when every neighborhood constant is a custody-bearing upper bound. Real fitted/HVP observations remain `PROXY_REUSE` or `BLOCKED`; they never acquire evaluator or score authority.

## Derived theorem to implement

Let `F(x)` be the frozen-SegNet training-signal output, `J(x)=DF(x)`, `q(x)` its upstream adjoint, and

`p(x)=J(x)^T q(x)`.

At anchor `a`, write `d=x-a`, `q_a=q(a)`, and `delta_q=q(x)-q_a`. The exact decomposition is

`p(x)-p(a) = [J(x)^T-J(a)^T]q(x) + J(a)^T delta_q`.

Use the anchor-adjoint first-order correction

`c_a(d)=(DJ(a)[d])^T q_a`

and an implementation `c_tilde_a(d)` satisfying

`||c_tilde_a(d)-c_a(d)|| <= L_c ||d||`.

The direct estimator is

`p_hat(x)=p(a)+c_tilde_a(d)`.

On a custody-bounded ball, require actual upper bounds

- `||J(a)|| <= B_J`;
- `||DJ(a)[v]|| <= B_H ||v||`;
- `||DJ(a+u)-DJ(a)|| <= L_H ||u||` as an operator from input directions to output-Jacobians;
- `||q(a)|| <= Q_a`;
- `||q(a+u)-q(a)|| <= L_q ||u||`;
- `||c_tilde_a(u)-c_a(u)|| <= L_c ||u||`.

`B_H` is a local Jacobian-Lipschitz upper bound. The first-order corrected remainder additionally needs `L_H=Lip(DJ)`, a Hessian-Lipschitz/third-derivative bound. A point HVP or power iteration does not supply this upper bound.

Taylor's integral remainder and the adjoint split give the DERIVED monotone envelope

`E(r) = (B_J L_q + L_c) r + (B_H L_q + 0.5 L_H Q_a) r^2 + 0.5 L_H L_q r^3`.

Indeed, the four residual terms are `J(a)^T delta_q`, `(DJ(a)[d])^T delta_q`, the Taylor remainder applied to `q(x)`, and correction error. Therefore `||p(x)-p_hat(x)|| <= E(||d||)`.

When composing directly with task 454's already-handled adjoint envelope, use the tighter exact split

`p(x)-p_hat(x) = J(x)^T delta_q + [J(x)-J(a)-DJ(a)[d]]^T q_a + [c_a(d)-c_tilde_a(d)]`.

Its DERIVED bound is

`E_454(r) = A_adj(r) + 0.5 L_H Q_a r^2 + L_c r`,

where task 454 owns

`A_adj(r)=(J0+beta r) kappa (J0 r + 0.5 beta r^2)`.

The general `L_q` cubic and this task-454-composed envelope are compatible alternatives, not terms to sum. Code must not double-count adjoint drift.

Let `tau_p` be the costate-error tolerance. If descent is the owner, derive it as `tau_p=gamma_theta/B_R`, where `B_R` upper-bounds the current renderer VJP operator norm and `gamma_theta>0` lower-bounds the **current corrected reused renderer-gradient throughout the admitted ball**, not just its anchor value. Do not accept a literal radius or literal tolerance. This proves a strict descent direction. A finite-step loss-decrease theorem additionally needs an objective-gradient smoothness bound and an admitted step-size law; do not claim it from the direction inequality alone.

For a custody/calibration cap `r_cap`, define

`r_error = sup { r in [0,r_cap] : E(r) < tau_p }`.

Compute this with a monotone representable-float bisection; do not add an iteration-count knob. The self-adjusting direct-reuse radius is

`r_star = min(r_error, r_geometry, r_cap)`,

where `r_geometry` is the already-derived margin/Fisher or rigorous margin radius. Its authority is the weakest authority among all inputs. Strict membership is required; equality refreshes.

The cheaper alternative using `q(x)` in the correction would also require current `q(x)` and `J(a)^T delta_q`; obtaining them from the frozen network destroys the forward-replacement premise. It is a diagnostic identity, not the operational path.

## Exact drift term and empirical boundary

For Torch measurement, hold `q_a` detached and compute the exact directional term at the anchor:

1. `logits_a=F(a)`;
2. `p_a=grad(logits_a, a, grad_outputs=q_a, create_graph=True)`;
3. `c_a(d)=grad(sum(p_a*d), a)`.

This is `(DJ(a)[d])^T q_a`. The full CE Hessian-vector additionally contains `J(a)^T(Dq(a)[d])`; record that difference as the adjoint-drift diagnostic rather than misnaming the full Hessian as Jacobian drift.

An exact HVP in floating-point is a measured directional correction, not a segment-wide `L_H` upper-bound artifact. Power iteration estimates large curvature directions but is a lower estimate of an operator supremum unless accompanied by a certified residual bound. Consequently the real arm is expected to remain non-authoritative unless existing custody surfaces unexpectedly contain all required bounds.

The rigorous real arm also owes a certified smooth cell or a semismooth replacement: frozen SegNet and through-R contain activation/clamp/rounding branch surfaces, so a classical `C^{2,1}` ball cannot cross one silently. All `J`, `DJ`, `q`, renderer, and tolerance bounds must use the same controlling norm and dual norm. The measured margin/Fisher RMS is a weighted seminorm/proxy; it cannot control the theorem without a same-domain coercivity bound. These are fail-closed certificate requirements, not reasons to suppress the empirical characterization.

## Owned implementation surface

- Extend, do not rewrite, `src/tac/scorer_surrogate/costate_trust_region.py` with the scalar error envelope, self-adjusting radius, typed direct certificate/decision, correction application, and optional Torch fixed-adjoint HVP helper. Preserve every existing current-prefix API and test.
- Extend `src/tac/witness_dsl/costate_trust_region_policy.py` with a default-off typed direct-full-costate policy. No trainer argv and no radius/tolerance knob.
- Add `src/tac/canonical_equations/jacobian_drift_full_costate_20260713.py`; do not mutate a shared registry/initializer directly.
- Add `tools/probe_jacobian_drift_certificate.py`, atomic per-regime checkpoints, terminal byte-stable resume, exact source bundle, source/input hashes, and no `/tmp` evidence.
- Add focused tests under `src/tac/tests/` and canonical-equation tests. Do not edit #455 surrogate or #456 exact-forward surfaces.
- Main agent owns the final memo and DAG FEED. Implementation worker must not edit them.

## Real probe contract

Reuse the three sealed task-454 pair-0 regimes, exact differentiable `_render_chart -> contest_r` scorer inputs, fixed CPU Torch SegNet, GT cache, and registered candidate ladder. Do not use the detached NumPy camera path as an HVP graph.

For every real candidate, record:

- scorer-input displacement and source hashes;
- uncorrected banked-costate error, corrected error, cosine, and the exact decomposition residual;
- fixed-`q_a` Jacobian HVP time, full-loss HVP diagnostic time where measured, exact validation-forward time, and exact fresh input-costate shadow time;
- renderer-gradient dot/cosine versus the fresh exact control;
- fresh exact CE and `d_seg`, with `d_seg` remaining NumPy-fp32 authority and all surrogate quantities training-signal-only.

The real safe-ball characterization must report separately:

- rigorous certified reuses (expected zero if bound custody is missing);
- empirical/oracle-characterized corrected reuses and their exact-shadow safety;
- the previous margin/Fisher `1/64` comparator;
- correction cost in exact-validation-forward equivalents;
- operational validations per anchor including drift cost, plus per admitted reuse if different;
- whether the total beats the measured `402/48=8.375` baseline meaningfully.

Fresh shadows are measurement controls and must remain visible in actual probe work. They are excluded only from a clearly named hypothetical operational path.

## Canaries and falsifiers

- Analytic affine `F`: drift correction and remainder are exactly zero.
- Analytic quadratic/cubic `F`: correction matches the known directional term and the cubic envelope is tight enough to accept an interior point and refresh at equality/outside.
- Nonzero adjoint drift: the full Hessian minus fixed-adjoint HVP equals the independently computed adjoint term.
- A rigorous request without `L_H`, bound custody, correction-error custody, or descent tolerance fails closed.
- Empirical calibration cannot emit `CERTIFIED_REUSE`.
- Source/anchor/correction custody mismatch fails closed.
- Terminal resume rechecks custody and preserves completed receipt bytes.

## Verdict rules

- `GO` only if a real custody-bearing certificate admits more than the inherited `1/64`, every admitted real step preserves fresh exact descent and exact `d_seg`, and drift-inclusive operational cost is meaningfully below `8.375` validation-forward equivalents per anchor.
- `BLOCKED` if the theorem is valid but real upper-bound custody is absent, even if an oracle/empirical shadow suggests reuse.
- `NO-GO` for the measured formulation if the correction is faithful but reuse remains at or below `1/64`, or if drift-inclusive cost consumes the validation win. Scope every negative to pair0, the three saved regimes, fixed-adjoint first-order HVP, current Torch/CPU substrate, and registered ladder/window; do not kill the direct-costate family.
