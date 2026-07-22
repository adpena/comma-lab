# Canonical-equations note — DDM G2 solve-minus-predict differential instrument

**Status:** measured advisory telemetry plus derived identities.  No new equation
is registered by this isolated branch because receiver delta-d-seg and the inner
encoder Jacobian are absent.  MAIN landing review is required.

## E1 — exact resize range/kernel split

For receiver scorer plane `y`, exact factor-2 realization `R^-1`, and full solved
and predictor endpoints,

`Delta = R^-1(y_solve) - R^-1(y_predict)`,

`Delta_range = A^+ A Delta`,

`Delta_ker = Delta - Delta_range`.

The measured n600 weighted fractions are `0.6168064528841017` and
`0.38319354711589826`.  This energy split is distinct from bounded uint8
reachability and from the real-linear nullity dimension.

## E2 — head/Fisher differential coordinates

For target/rival classes `(c,c')`, cached top-two margin `m`, and registered head
normal `w_c-w_c'`,

`kappa_F(m) = 1/2 sech^2(m/2)`,

`d_flip = |m| / ||w_c-w_c'||_2`.

The quotient of the five-class gauge-null head operator has rank four.  The
factorized camera-space costate used here is

`lambda_camera = A^T Q4^T lambda_head(residual, target, m)`.

`A^T` and the rank-four quotient are exact.  The frozen SegNet inner encoder
Jacobian is absent, so `lambda_camera` is not an exact input gradient.

## E3 — held-out xi transport versus persistence

For compact endpoint feature `z_t`, translation-first SE(3) twist feature
`xi_t`, and each held-out window `W`, fit on the complement:

`Beta_-W = argmin_B ||Z_-W - Xi_-W B||_F^2 + ridge ||B||_F^2`,

`z_hat_t = xi_t Beta_-W`, for `t in W`.

Report `1 - ||z_W-z_hat_W||^2/||z_W||^2` separately from the predecessor
baseline `z_hat_t = z_(t-1)`.  Measured Lane/Movable predecessor persistence is
high, while held-out xi-only explained energy is negative; the two quantities
must not be conflated.

## E4 — endpoint sensitivity

START sensitivity is copied only from SHA-bound V12 receipt rows.  A per-class
change is measured, but attribution to a predictor component is undefined when
the receipt exposes no such intervention label.

END rungs are deterministic transforms satisfying

`||Delta_tau||_2^2 = tau ||Delta||_2^2`, `tau in {1,.75,.5,.25,0}`.

They are labeled derived and do not imply receiver/evaluator behavior.

## E5 — KKT admission remains blocked

For exact candidate bytes `Delta B`, the registered rate price is

`lambda_B = 25/37545489 = 6.658589531221714e-7`.

Admission needs a receiver-measured numerator:

`-Delta S_dist / Delta B > lambda_B`.

The instrument measures candidate bytes but not receiver `Delta d_seg`; all
candidate KKT rows therefore remain `BLOCKED_NO_RECEIVER_DELTA_DSEG`.  The
byte-minimal measured chart is the compact parabolic shearlet (`92,544 B`), a
next-probe priority rather than a promotion verdict.

## Registry disposition

`NO_REGISTRATION`: E1--E5 instantiate already-settled resize, Fisher/head,
SE(3), and rate laws.  The novel empirical anchor is the n600 ledger; promotion
to a solve law waits for receiver-closed coefficient perturbations and exact
delta-S/byte.

STORES CONSULTED: C1 solved-plane and V12 endpoint receipts; GT cache; exact
resize and adjoint implementations; factorized head/Fisher equations; G2G2 and
describe-line DAG rows; #233 jitter surfaces; lane/subagent/probe/posterior
state; CLAUDE.md, AGENTS.md, PROGRAM, v7.5 and v8 specs.

`0.1910828242 [contest-CPU]` is unchanged; `score_claim=false`.
