---
title: "Candidate canonical equations: VR-GHAL applicability gates for #455/#454"
date_utc: "2026-07-13"
research_only: true
registration_status: "DEFERRED_SHARED_REGISTRY_HOT"
---

# Candidate canonical equations: VR-GHAL applicability gates

These are durable candidate equations for main review. They are **DERIVED**, not transcriptions of
the inaccessible paper body. Registration into the shared canonical equation JSONL is deferred
because that surface is live-sibling-held.

## `EQ-VRGHAL-455-MOVING-OPERATOR-DEBT-v1`

Let `T_t` be the population update map under the on-policy distribution at witness step `t`, `T_*` a
fixed reference map, and `T_hat_t` the stochastic sample update. Define

`zeta_t(w) = T_hat_t(w)-T_t(w)`, with `E[zeta_t(w)|F_{t-1}]=0`,

`b_t(w) = T_t(w)-T_*(w)`.

Then

`T_hat_t(w)-T_*(w) = zeta_t(w)+b_t(w)`,

`||T_*(w)-w|| <= ||T_t(w)-w|| + ||b_t(w)||`,

and, for `omega_j=sup_{w in C}||T_{j+1}(w)-T_j(w)||`,

`sup_{w in C}||T_t(w)-T_0(w)|| <= sum_{j<t} omega_j`.

**Use:** a stationary stochastic fixed-point theorem cannot certify the live on-policy residual until
the operator-drift debt is zero by construction or is explicitly bounded by a tracking theorem.

## `EQ-VRGHAL-455-QUERY-TO-TEACHER-v1`

Let `A` be fresh-anchor samples, `D` paired-difference samples, and `c_label` exact teacher labels per
difference. Then

`Q_oracle = A+2D`,

`C_teacher = A+c_label D`,

`saving_fraction = max{0,1-C_teacher/N}`.

For frozen-state squared-loss regression with `g_s(w)=Phi_s^T(Phi_s w-y_s)`,

`g_s(w)-g_s(v)=Phi_s^T Phi_s(w-v)`,

so the teacher label cancels and a cached state has `c_label=0` for parameter-space differences.

**Use:** never report an oracle-query exponent as a teacher-forward saving without an explicit cache
and state-definition map.

## `EQ-VRGHAL-454-CLIPPED-TAIL-DEBT-v1`

For `Clip_R(z)=z min{1,R/||z||}` and a conditional second-moment bound
`E[||Z||^2|F]<=s^2`,

`||E[Clip_R(Z)-Z|F]|| <= s^2/R`,

`P(||Z||>R|F) <= s^2/R^2`,

and across at most `M` valid adaptive decisions,

`P(max_{t<=M}||Z_t||>R) <= M s^2/R^2`.

Thus a total failure budget `delta` requires `R>=s sqrt(M/delta)` or an equivalent explicit tail
allowance.

**Use:** a clipped drift estimate without its tail debt is not a safe stale-forward reuse
certificate.

## Triality routing

- DAG consumer: `vrghal_95kill_fixedpoint_DAG_FEED_20260713.md`.
- DSL consumer: deferred to #455/#454 owners after admission gates pass.
- Equation registry: deferred to main because the shared registry is hot.

