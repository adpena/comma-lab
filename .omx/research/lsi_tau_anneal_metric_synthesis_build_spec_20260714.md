# LSI tau-anneal and task-#500 synthesis build specification

`research_only=true` · `$0` · no evaluator, training, paid dispatch, or V9 hot-file edit.

## Executable surfaces

- `log_sobolev_tau_anneal_20260714.py` derives the normalization-explicit adiabatic rate, evaluates
  #318's two static CFL groups, exposes NumPy-fp32 authority plus optional MLX parity, and refuses a
  rate comparison without full source, identical state IDs, identical units, and a genuine DE
  tau-rate derivation.
- `metric_unification_synthesis_20260714.py` consumes specialist receipts. It cannot implement or
  select a second metric, basis, or trust-region law. It emits the one compositional specification
  and an activation-readiness result.

## The single #500 specification

Let `q_i(theta)=Q_i C z(R(theta))`, `J_i=dq_i/dtheta`, and
`F_i=diag(p_i)-p_i p_i^T`. The canonical reachable decision metric is

`G_dec = sum_i J_i^T F_i J_i + lambda_damp I`.

For basis coefficients `a_B` with `Psi_B=dtheta/da_B`, the basis geometry is the coordinate
pullback `G_B=Psi_B^T G_dec Psi_B`; it is not an additive rival metric. The finite trust region is

`D_KL(p_i || softmax(z_i+J_i Delta_theta)) <= delta_i`,

with local quadratic form `Delta_theta^T J_i^T F_i J_i Delta_theta <= 2 delta_i`. The curriculum
coordinate adds `g_tau_tau d tau^2` and the independently custodied adiabatic cap. The existing
measured `0.978` Fisher/margin Pearson value, bound to canonical-equation source SHA-256
`72d5b5ae...`, motivates the winner-rival candidate but is explicitly not an algebraic identity or
a substitute for full probabilities/Jacobians.

## Current activation blockers

1. Full-n600 canonical M selection is `NO_VERDICT_DATA_CUSTODY`.
2. Corrected full-K RIPO implementation/measurement is still in progress.
3. Basis pullback Gram custody is absent.
4. Corrected deploy-int8 localized-basis matched-byte n600 ranking is pending.
5. LSI/DE same-state numerical rate comparison is source/data blocked.
6. Stable basis source is frozen, but the V9 provenance reseal is pending.

Every blocker is admission/readiness scoped. No family is closed.
