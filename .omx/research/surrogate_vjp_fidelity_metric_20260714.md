# Surrogate VJP Fidelity: Argmax-Native Metric Truthing — 2026-07-14

`research_only=true`  
Lane: `lane_surrogate_vjp_fidelity_metric_20260714`  
Authority: `[macOS-CPU advisory; NumPy-fp32 training-gradient evidence; no score authority]`  
Verdict scope: `INSTANCE-OF-FIRST-CUT-FORMULATION x REAL-N600-SOURCE-HELDOUT-120`  
Pointer delta: **NONE**

## Pointer status

The exact `0.19108` submittable pointer and the `0.18804` borrowed defensive
bank are unchanged.  This work is a throughput MEANS.  It can move a pointer
only after a faster epoch produces a score-moving, receiver-closed archive row
under exact evaluation.

## Executive verdict

**The raw-cosine headline is a measurement-locus artifact.  The retained
first-cut instances remain below the static license gate, but the wall for the
optimal decision/Fisher/functional formulation is unresolved and unmeasured.**

Moving from the ambient 589,824-element RGB input costate to the actual
19-dimensional renderer pullback raises aggregate alignment by `12.5x` for the
round-2 convex head and about `51x` for both round-3 pre-SE heads.  That proves
that the verdict is strongly locus-sensitive and that the reachable geometry
differs materially from the ambient RGB geometry.  Because `J_R^T` is
anisotropic, this observation does **not** isolate null-space removal from
reachable-direction reweighting.

The corrected magnitude-sensitive first-order gain remains only `0.2304%` for
round 2 and `0.4648%` for round 3.  Relative L2 remains approximately one.
Those observations withhold a license from these first-cut instances; they are
not a negative verdict on the technique, family, or optimal formulation.

The centered-logit whole-teacher/Jacobian student remains in the explicit
optimal-form reformulation queue with measurement status `UNMEASURED`: its
receipt says `n_pairs=0`, `teacher_calls=0`, and its required n600 manifest does
not exist.

## The top discovery: use the reachable decision geometry

Let `x=R(theta)` be the differentiable witness renderer and `q(x)=P H(z(x))`
the scorer's centered-logit quotient.  For exact and surrogate input costates
`g_T` and `g_S`, define

`h_T = J_R^T g_T`, `h_S = J_R^T g_S`.

For a positive-semidefinite optimizer/preconditioner `M`, the operational
semi-inner product is

`<g_T,g_S>_(R,M) = h_T^T M h_S = g_T^T J_R M J_R^T g_S`.

The required decomposition is

`rho = <g_T,g_S>_(R,M) / (||h_T||_M ||h_S||_M)`,

`r = ||h_S||_M / ||h_T||_M`,

`eta = <g_T,g_S>_(R,M) / ||h_T||_M^2 = rho r`.

`eta` is the same-learning-rate first-order exact-loss descent delivered by
the surrogate, normalized by the exact gradient step.  It exposes the failure
that cosine hides: a perfectly aligned but vanishing vector still does almost
nothing.

The prompt's proposed `P H` insight is correct for the **forward** decision
quotient and incorrect as a direct projection of an RGB costate.  `P H` maps
head features to centered logits; the RGB costate lives in the input dual.
The typed decision map for witness update `u` is instead

`A_theta = W^(1/2) C D_x T D_theta R`,

where `C` forms active winner–rival contrasts and `W` carries preregistered
margin/Fisher/class-pair weights.  Its decision metric is

`G_theta = A_theta^T A_theta`.

Comparing `A_theta u_S` with `A_theta u_T`, or equivalently comparing updates
under `G_theta`, removes the common-logit gauge and directions the actual
renderer cannot reach.  Existing receipts lack the logits/Jacobians needed to
compute it.

## Ranked fidelity metrics

| Rank | Metric | Current retained measurement | Can it license replacement? | Why it is the right/wrong geometry |
|---:|---|---|---|---|
| 1 | Identical-state, trust-region-matched one-step/short-trajectory exact functional parity | `UNMEASURED_MISSING_CUSTODY`: no paired perturbation outcomes | **Yes**, with exact-anchor fallback and recurring on-policy n600 validation | Directly measures exact CE, beneficial/harmful argmax flips, d_seg, d_pose, and holistic class facets after the update. |
| 2 | Low-margin winner–rival directional derivative in reachable decision metric `G_theta` | `UNMEASURED_MISSING_CUSTODY`: no logits, active rivals, or decision Jacobian | Necessary first-order gate; not sufficient alone | Minimal argmax-native tangent: it measures movement toward/through the actual separatrix, not unrelated logit order. |
| 3 | Renderer/optimizer pullback `rho`, norm ratio, `eta`, relative L2 | **Measured-field reaggregation below** with `M=I` | **No** static metric alone; strongest retained early diagnostic | Pulls the comparison into the actual reachable geometry, nulling the kernel and anisotropically reweighting surviving directions while preserving magnitude. |
| 4 | Finite-step softmax KL/Bregman on centered logits | `UNMEASURED_MISSING_CUSTODY`: no logits/probabilities | Secondary gate only | Canonical exponential-family divergence, but soft agreement can miss low-margin argmax harm. |
| 5 | Fisher/natural alignment | `UNMEASURED_MISSING_CUSTODY`: no probabilities/Jacobian | Secondary gate only | Use `F(p)=diag(p)-pp^T` on quotient tangents and `F^+` on quotient cotangents; ordinary `g^T F g` can mix tangent/cotangent roles. |
| 6 | Ordinal/recos/sign concordance | Round-2 coordinate sign `0.5272`; pairwise ordinal `0.5166`; round-3 predicted 19-vectors absent | **No** | Basis-dependent and stricter than the winner–rival fact d_seg needs. Useful sanity diagnostic only. |
| 7 | Raw RGB input-costate cosine | `0.001416–0.001679` | **No** | Ambient Cauchy–Schwarz angle can count unreachable directions, weights reachable directions differently from the trainer, and discards magnitude. |

No threshold is guessed.  A future governed run must seal its deterministic
repeat floor, trust-region radius, non-inferiority band, per-class/worst-pair
facets, and exact-anchor fallback before candidate inspection.

## $0 remeasurement of retained real-source evidence

Each row below is derived from measured per-state dot/norm reductions for the
120 heldout indices `0,5,...,595` of a real-n600-source population.  It is not a
claim that all 600 states were re-evaluated in this pass.

Canonical receipt:
`.omx/research/surrogate_vjp_fidelity_metric_remeasurement_20260714.json`,
SHA-256 `c4116ff0b9af3284b00e90980f693f98be3c11b30eada0ac13bb395cf50c3753`.
All 360 evaluated row files match their original campaign path/SHA/byte seals;
teacher and Metal call counts are both zero.

| Formulation instance | Locus | Cosine `rho` | Relative L2 | Norm ratio `r` | Exact-descent fraction `eta` | Positive-dot states | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| round-2 convex head | raw RGB costate | 0.0014157934 | 1.00000187 | 0.00381280 | 0.0000053981 | 82.500% | baseline only |
| round-2 convex head | renderer pullback, `M=I` | 0.0176974146 | 1.00614916 | 0.13016663 | 0.0023036129 | 60.833% | **FIRST-CUT INSTANCE BELOW STATIC GATE** |
| round-3 pre-SE linear | raw RGB costate | 0.0016650256 | 1.00000044 | 0.00357471 | 0.0000059520 | 91.667% | baseline only |
| round-3 pre-SE linear | renderer pullback, `M=I` | 0.0856220782 | 0.99682046 | 0.05428250 | 0.0046477802 | 65.833% | **FIRST-CUT INSTANCE BELOW STATIC GATE** |
| round-3 pre-SE RFF | raw RGB costate | 0.0016791964 | 1.00000039 | 0.00357496 | 0.0000060031 | 91.667% | baseline only |
| round-3 pre-SE RFF | renderer pullback, `M=I` | 0.0857091120 | 0.99681737 | 0.05423029 | 0.0046480302 | 65.833% | **FIRST-CUT INSTANCE BELOW STATIC GATE** |

Round 2 retains both exact and predicted 19-vectors.  Round 3 retains the exact
19-vector and scalar candidate dot/norm/error reductions, not the predicted
19-vector; round-3 sign/ordinal metrics are therefore
`UNMEASURED_MISSING_PREDICTED_VECTOR`.

For round 2, optimal scalar rescaling still leaves relative error `0.99984339`
and explains only `rho^2 = 0.00031320` (`0.0313%`) of exact renderer-gradient
energy.  The result is weak positive aggregate signal, not useful fidelity.

## Right metric on the right distribution

The operator's on-policy-distillation amendment adds an orthogonal requirement.
Correct geometry on cached states is not fidelity on live visited states.  With
cache distribution `mu`, live distribution `pi`, support mask `m_i`, and sealed
clip `w_max`, use

`w_i = m_i clip(pi(s_i)/mu(s_i), 0, w_max)`,

`L_pi_hat = sum_i w_i L_fid(s_i) / sum_i w_i`.

Every receipt must include density-ratio custody, support violations, clipped
fraction, effective sample size, and time-ordered split.  The current artifacts
contain no density ratios.  This row is `BLOCKED_DISTRIBUTION_CUSTODY`; uniform
weights are not importance-corrected evidence.

## P0 centered-logit student re-spec

The surviving formulation predicts the four-dimensional centered-logit
quotient and is trained jointly on value and directional Jacobian fidelity:

`L = lambda_q L_value(q_S,q_T)`
`  + lambda_J E_v[L_G(J_q,S v, J_q,T v)]`
`  + lambda_flip L_winner-rival`
`  + lambda_KL KL(p_T || p_S)`,

with the entire per-state loss importance-corrected only when sealed on-policy
density ratios exist.  Directional probes avoid materializing the full
Jacobian; their seed, count, distribution, state assignment, and hashes are
checkpointed.  Stage boundaries preserve value-only, Jacobian, flip/KL, and
on-policy-refresh checkpoints independently.

Admission sequence:

1. seal a real n600 cache containing post-R rendered input, centered logits or
   logits/probabilities, labels/margins/active rivals, full teacher costate,
   student VJP, renderer pullbacks, and perturbation outcomes;
2. pass deterministic NumPy-fp32 parity and exact cache custody;
3. pass `eta/rho/relative-L2`, low-margin decision, Fisher/KL, aggregate,
   worst-pair, per-class, and checkpoint-regime gates;
4. seal a chart-appropriate terminality guard before interpreting a low row:
   fixed quadratics require terminal-gradient/curvature error bounds,
   range-space/null custody, and an exact heldout ridge ladder; nonlinear
   students require a separate nonconvex convergence certificate;
5. pass identical-state exact one-step n600 non-inferiority;
6. pass recurring on-policy n600 A/B with exact-anchor fallback;
7. only then measure Metal timing and evaluate amortized economics.

## Economics

Using the settled diagnostic `C_T=3009.070 ms` and inclusive budget
`150.453 ms/step`, the combined student plus amortized update allowance is

`C_S,VJP + U/K < 150.453 - 3009.070/K`.

| Anchor K | Teacher amortized (ms/step) | Maximum `C_S,VJP + U/K` (ms/step) | Status |
|---:|---:|---:|---|
| 20 | 150.4535 | -0.0005 | algebraically impossible |
| 24 | 125.3779 | 25.0751 | algebraically eligible only |
| 32 | 94.0334 | 56.4196 | algebraically eligible only |
| 48 | 62.6890 | 87.7640 | algebraically eligible only |
| 64 | 47.0167 | 103.4363 | algebraically eligible only |
| 96 | 31.3445 | 119.1085 | algebraically eligible only |
| 128 | 23.5084 | 126.9446 | algebraically eligible only |

No row pays yet: `C_S,VJP`, `U`, corrected fidelity, and functional improvement
are unmeasured.  A future functional gate should compare the one-sided lower
confidence bound of exact improvement ratio against the measured cost ratio;
mere positive alignment is insufficient.

### Host handoff and implementation blocker

`tools/run_surrogate_vjp_fidelity_metric_host.command` runs the authenticated
`$0` remeasurement by default and exposes `MODE=preflight-n600` for a future
cache.  `MODE=refit-measure` intentionally returns
`BLOCKED_IMPLEMENTATION` after that preflight and requires both sealed
functional-gate and terminality receipts.

The existing whole-teacher Metal driver optimizes a boundary-masked ambient RGB
Sobolev value/cosine/L2 objective.  Its manifest has no renderer state/Jacobian,
active winner–rival decision Jacobian, applied-step outcome, or on-policy
density ratio.  Calling it would measure the wrong training target under the
new name.  MAIN can run the `$0`/preflight modes now; corrected Metal timing is
owed only after the new resumable joint value+Jacobian fit driver and expanded
cache schema land on an approved SSD tier.

## Custody blocker and exact unblock

`BLOCKED_DATA_CUSTODY`: no retained artifact contains paired real-n600 post-R
RGB, teacher logits/quotient, full teacher input costate, labels, and comparable
student VJP.  The strongest partial custody is the 120-row heldout renderer
gradient evidence reaggregated above.  Fisher/KL/decision/flip/one-step rows
cannot be reconstructed from hashes or scalar reductions.

Exact unblock: a governed host collection must preserve the missing tensors or
sufficient directional statistics with content hashes, real source/runtime
custody, replay assignment, deterministic repeat proof, stage checkpoints, SSD
storage plan, and certify-or-block cleanup.  That collection requires teacher
work and is outside this arm's `$0` / no-Metal containment.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, operating manual, v7.5/v8 specs
- `reports/latest.md`
- `.omx/state/lane_registry.json`
- `.omx/state/subagent_progress.jsonl`
- `.omx/state/master_gradient_anchors.jsonl`
- `.omx/state/modal_call_id_ledger.jsonl`
- `.omx/state/cost_band_posterior.jsonl`
- `.omx/state/continual_learning_posterior.jsonl`
- latest Codex findings/session, council, design, and directive memos
- whole-teacher, on-policy, JEPA, frozen-replay, round-3, sparse-adjoint, and
  pre-SE experiment receipts and cleanup manifests

## Triality and apparatus wire-in

- Equation: `argmax_native_vjp_fidelity_20260714`
- DSL: `surrogate_vjp_fidelity_policy` (default off, research-only, no trainer argv)
- DAG: `surrogate_vjp_fidelity_metric_DAG_FEED_20260714.md`
- Sensitivity contribution: active winner–rival margin/Jacobian weights
- Pareto constraint: exact d_seg/d_pose/bytes safety under functional step
- Bit allocator: none until exact functional improvement per byte exists
- Autopilot: refuses activation without sealed full-n600 and gate receipts
- Continual learning: the first-cut static-gate rows and custody blocker are durable
- Disambiguator: raw vs reachable vs decision/Fisher vs functional modes remain
  callable and separately labeled; measurement, not prose, arbitrates

## Verdict ladder

- `INSTANCE-OF-FIRST-CUT-FORMULATION`: the three retained students are below
  the static diagnostic gate; this is not a technique verdict.
- `REFORMULATION QUEUE`: optimal centered-logit value+Jacobian student under
  decision/Fisher/functional geometry — unmeasured and live.
- `FAMILY`: distilled scorer surrogate — live, gated by custody and functional
  on-policy evidence.
- `PARADIGM`: scorer-forward cheapening — untouched.
