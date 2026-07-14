# Surrogate VJP Fidelity Metric Build Specification — 2026-07-14

Status: **BUILD + local-$0 verification only; no Metal, teacher, paid, heavy, or live-run actuation**  
Lane: `lane_surrogate_vjp_fidelity_metric_20260714`  
Research authority: `research_only=true`  
Verdict scope: `INSTANCE-OF-FIRST-CUT-FORMULATION x REAL-N600-SOURCE-HELDOUT-120`

## 1. Premise truthing

The inherited raw input-costate cosine `0.0014–0.0017` is real measured
macOS-CPU/NumPy-fp32 training-gradient evidence from heldout states of campaigns
whose source population contained 600 real `n600-real-0.mkv` assignments.  It is
not a measurement of the unfit centered-logit whole-teacher student: that
campaign has `n_pairs=0`, `teacher_calls=0`, and no n600 cache manifest.

The full RGB costates, rendered frames, logits/quotients, probabilities, and
perturbation outcomes were not retained.  Cleanup manifests certify that the
heldout costate scratch was deleted.  The retained evidence is:

- scalar input-costate dot/norm/relative-L2 reductions for 120 heldout states;
- paired exact and predicted 19-dimensional renderer gradients for the round-2
  120 states; round-3 retains the exact 19-vector plus scalar candidate
  dot/norm/error reductions, but not the predicted 19-vector; and
- content hashes and real-source assignment metadata.

Therefore this landing MUST NOT claim that it remeasured decision-logit,
Fisher, KL, flip, or exact one-step fidelity.  It may reaggregate the retained
renderer-gradient evidence and issue explicit custody blockers for the rest.
The retained frozen-replay students are first-cut forms.  Their rows may
withhold an instance license but cannot support a technique or family verdict.
The optimal centered-logit decision/Fisher/functional formulation remains in
the explicit reformulation queue.

## 2. Correct geometry

Let the frozen scorer's centered-logit quotient be

`q(x) = P H(z(x))`, where `P = I - 11^T/C`.

`P H` acts in penultimate/logit space.  It is dimensionally invalid to apply it
directly to an RGB input costate.  The exact and surrogate input costates are
already pullbacks, `g_T = J_q,T^T r_T` and `g_S = J_q,S^T r_S`.

For the actual renderer/trainer `x = R(theta)`, define the reachable pullbacks

`h_T = J_R^T g_T`, `h_S = J_R^T g_S`.

For the positive-semidefinite optimizer/preconditioner operator `M`, the
argmax-native functional semi-inner product is

`<g_T,g_S>_(R,M) = h_T^T M h_S = g_T^T J_R M J_R^T g_S`.

Its required magnitude-and-angle decomposition is

`rho_(R,M) = <g_T,g_S>_(R,M) / (||h_T||_M ||h_S||_M)`,

`r_(R,M) = ||h_S||_M / ||h_T||_M`,

`eta_(R,M) = <g_T,g_S>_(R,M) / ||h_T||_M^2 = rho_(R,M) r_(R,M)`.

`eta` is the first-order teacher-loss descent delivered by a surrogate step,
normalized by the equally scaled teacher-gradient step.  Unlike cosine, it
retains the step magnitude.  `M=I` is the only configuration supported by the
retained 19-D receipts; any Adam/Muon/other optimizer claim requires a sealed
per-state `M` or applied step.

This is a necessary training-fidelity diagnostic, not the final admission
authority.  The ultimate gate is deterministic one-step/short-trajectory
functional parity: apply teacher and surrogate updates from the identical
state, rerun the exact frozen scorer through actual `R`, and compare exact CE,
argmax flips/d_seg, d_pose, and bytes.  No static similarity metric alone may
license replacement.

## 3. Ranked metric contract

1. **Exact one-step/trajectory functional parity** — final admission authority.
2. **Reachable optimizer-pullback gain `eta_(R,M)` plus `rho`, norm ratio, and
   relative L2** — primary differentiable target and early gate.
3. **Winner–runner-up low-margin directional/flip preservation** — minimal
   argmax-native diagnostic; compare margin directional derivatives and
   predicted crossings on a preregistered annulus.
4. **Categorical Fisher / KL geometry on the centered-logit quotient** — use
   `F(p)=diag(p)-pp^T`; compare quotient tangents with `F` or quotient
   cotangents with the Moore–Penrose dual `F^+`.  Finite-step KL is the
   exponential-family Bregman check.  It is secondary to actual flips.
5. **Centered-logit value plus directional-Jacobian fit** — the student target:
   fit `q` and sampled `J_q v`, never value-only and never the ill-typed
   projection of RGB costates by `P H`.
6. **Ordinal/recos/sign concordance** — diagnostic only.  Coordinate ranking is
   basis-dependent and stronger than winner–runner-up argmax preservation.
7. **Raw ambient RGB cosine** — retained baseline only; never an admission gate.

No numerical admission threshold is invented here.  A host run must
pre-register a deterministic repeat/no-op floor and a non-inferiority band,
then seal it before inspecting candidate results.

## 4. Right metric on the right distribution

Cached-state fidelity does not imply live on-policy fidelity.  If `mu` is the
cached collection distribution and `pi` the live visited-state distribution,
the on-policy objective is

`L_pi = sum_i m_i clip(pi(s_i)/mu(s_i), 0, w_max) L_fid(s_i) / sum_i m_i w_i`.

The receipt must report weight source/custody, clip, support mask, effective
sample size, clipped fraction, and train/heldout time ordering.  Missing density
ratios or support is `BLOCKED_DISTRIBUTION_CUSTODY`; uniform weights MUST NOT be
reported as importance-corrected evidence.

## 5. Required implementation

New isolated files only:

- `src/tac/scorer_surrogate/vjp_fidelity.py`
  - strict finite-vector validation;
  - weighted semi-inner-product summary (`rho`, norm ratio, `eta`, relative L2);
  - sign and pairwise ordinal diagnostics, explicitly non-authoritative;
  - Fisher primal/dual and finite-step KL utilities;
  - winner–runner-up directional/flip summary;
  - self-normalized clipped-importance objective and ESS;
  - economics sweep `C_eff(K)=C_S,VJP+(C_T+U)/K`.
- `tools/probe_surrogate_vjp_fidelity_metric.py`
  - `remeasure-retained`: read-only authentication and aggregation of exactly
  120 unique heldout indices `0,5,...,595` for the frozen-replay round-2 and
    round-3 linear/RFF artifacts; round-2 may compute vector diagnostics while
    round-3 may compute only quantities determined by its authenticated scalar
    dot/norm reductions; write one small atomic JSON receipt;
  - `preflight-n600`: validate the future whole-teacher n600 manifest through
    the existing strict cache validator and enumerate missing advanced-metric
    fields; never generate teacher data;
  - every unavailable metric is `NOT_MEASURED_<reason>`, never zero/NaN/pass;
  - output states source-population `n=600` separately from evaluated heldout
    `n=120`.
- `src/tac/witness_dsl/surrogate_vjp_fidelity_policy.py`
  - typed, default-off, research-only policy; no live trainer argv;
  - knobs only for metric mode, low-margin annulus, importance clip,
    directional-probe seed/count, anchor K, and sealed measurement,
    functional-gate, and terminality receipt paths.
- `src/tac/canonical_equations/argmax_native_vjp_fidelity_20260714.py`
  - standalone equation registration; no shared registry mutation.
- `tools/run_surrogate_vjp_fidelity_metric_host.command`
  - defaults to the read-only `$0` remeasurement;
  - `MODE=preflight-n600` authenticates a future base cache without teacher or
    device work;
  - `MODE=refit-measure` MUST refuse after preflight until a separately reviewed
    corrected fit driver owns renderer/decision sufficient statistics and the
    joint value+Jacobian objective.  It MUST NOT call the legacy whole-teacher
    fitter, whose boundary RGB Sobolev objective is not the chosen metric.
- focused tests for every new surface.

## 6. Acceptance

- Current retained probe reproduces raw aggregate cosine/relative-L2 from the
  stored reductions; it derives round-2 renderer aggregates and diagnostics
  from paired vectors, and round-3 renderer aggregates only from authenticated
  scalar dot/norm reductions.
- It records source and row hashes, exact index set, schemas, authority axis,
  and verdict scope.
- It fails closed for 119/121 rows, duplicate/missing indices, nonfinite/zero
  exact vectors, schema drift, a missing predicted renderer gradient where the
  schema promises one, missing scalar reductions where round-3 does not retain
  that vector, missing n600 arrays, or fake uniform IS weights.
- Unit tests cover exact vectors, magnitude mismatch with perfect cosine,
  unreachable-null perturbations, Fisher singular gauge direction, KL, flip
  crossings, importance clipping/ESS, economics, and custody failures.
- No Metal/teacher/heavy call occurs locally.
- Host refit refuses with `BLOCKED_IMPLEMENTATION` instead of timing or
  admitting the legacy raw-cosine objective under a corrected-metric label.
- No negative interpretation of a feature chart is admissible without a
  sealed terminality receipt: for a fixed quadratic, terminal-gradient/strong-
  curvature error bound, range-space/null custody, and an exact heldout ridge
  ladder; a nonlinear chart requires its own nonconvex convergence certificate.
- The measured memo, standalone DAG feed, `_codex` findings memo, and session
  summary preserve the verdict and precise unblock.

## 7. P0 re-spec and economics

The still-live family is a centered-logit quotient student trained jointly on:

`L = lambda_q L_value(q_S,q_T) + lambda_J E_v[L_RM(J_q,S v,J_q,T v)] + lambda_flip L_winner-rival + lambda_KL L_KL`,

with the entire per-state fidelity loss importance-corrected when legitimate
on-policy density ratios exist.  The final gate remains exact functional parity.

Sweep `K` without guessing student/update time.  With settled diagnostic
`C_T=3009.070 ms` and inclusive budget `150.453 ms/step`, report the maximum
allowable `C_S,VJP+U/K` for each K.  `K=20` leaves no positive budget;
larger K is only algebraically eligible until measured student timing and the
corrected fidelity gate both pass.

This lane is a throughput MEANS.  It does not move the `0.19108` submittable
pointer or the `0.18804` borrowed bank.  A pointer moves only after cheaper
epochs produce a score-moving, receiver-closed, exact-evaluated archive row.
