---
title: "VR-GHAL applicability to the frozen-SegNet 95%-kill prongs"
date_utc: "2026-07-13"
research_only: true
authority: "DESIGN / MEANS only"
verdict: "FEED-455-454"
verdict_scope: "current live #455 moving nonlinear on-policy costate learner and #454 clipped-difference-as-reuse-certificate formulation"
pointer_delta: "NONE"
heavy_or_paid_launch: false
live_sibling_edits: false
---

# VR-GHAL applicability to the frozen-SegNet 95%-kill prongs

## Executive answer

**DERIVED verdict: `FEED-455-454`. DERIVED theorem-certified teacher-forward saving for the live
#455 formulation: `0 calls`, hence `0%`. DERIVED conditional target, not a theorem result: one exact
teacher call per `20` witness steps saves `19/20 = 95%` of calls. Live `T` fits the verified VR-GHAL
hypotheses: `NO`.**

The failure is prior to constants. **DERIVED:** the live #455 learner can be represented either as
(i) a time-varying stochastic-gradient map, which is not a fixed operator and has no established
nonexpansiveness, (ii) a population best-response map, for which one teacher forward is not an
unbiased oracle evaluation of that map, or (iii) a coupled witness/surrogate map, for which the
teacher forward evaluates only one component and no nonexpansive geometry is established. The
on-policy distribution shift is predictable operator drift/bias relative to a fixed target; it is
not, by itself, zero-mean oracle noise.

**DERIVED:** smoothing the target from argmax pixels to logits or the exact CE input-costate removes
the argmax discontinuity from the regression label, but does not make the nonlinear parameter-update
map nonexpansive. It also does not remove frozen-network activation boundaries or the through-R
round/resize boundaries. The current live target is the exact through-R CE input-costate, not an
argmax table.

**DERIVED:** for #454, clipped stochastic differences are not a safer replacement for the direct
whole-ball Jacobian-drift envelope. Clipping suppresses precisely the large drift that must force a
refresh. A high-probability tail allowance can make the estimate honest, but no theorem implies that
the resulting chance-constrained radius is larger than the deterministic trust region. Halpern's
algorithmic anchor is not a stale-forward cache certificate.

This is deliberately a feed, not a new arm. The nearest honest use is a prior for a **frozen-replay,
convex-head** variant of #455 and, separately, a monitoring-only concentration layer for #454. It is
not authority to change either live sibling surface.

## Source authority and an explicit limitation

- **MEASURED (primary-source read):** the official arXiv abstract states the problem as finding
  `||T(x)-x|| <= epsilon` with probability at least `1-delta` for nonexpansive or contractive `T` on a
  normed space, using unbiased stochastic evaluations with bounded second central moment. It states
  that VR-GHAL applies to quadratically smoothable Banach spaces, clips stochastic differences at a
  Lipschitz scale proportional to `gamma ||x-y||`, and obtains anytime high-probability residual
  control. [Official abstract](https://arxiv.org/abs/2607.09097)
- **MEASURED (primary-source read):** the abstract reports `epsilon^-2` oracle complexity under
  samplewise nonexpansiveness, `epsilon^-3` under Lipschitz-in-expectation, and, under bounded variance
  alone, the displayed dependence `min{epsilon^-5, (1-gamma)^-3 epsilon^-2}` with other dependencies
  suppressed in the abstract. [Official abstract](https://arxiv.org/abs/2607.09097)
- **MEASURED access blocker:** the official PDF, HTML, and source endpoints were not retrievable in
  this environment; arXiv returned cache misses, no in-app browser was available, local browser
  control was not approved, and no local copy existed. The intended primary source is the
  [official PDF](https://arxiv.org/pdf/2607.09097).
- **UNKNOWN:** the paper's exact VR-GHAL recursion, batch schedule, theorem numbering, constants,
  logarithmic factors, precise definition of the Lipschitz-in-expectation oracle class, and the exact
  smoothability constant dependence. They are not reconstructed or attributed below.

Consequently, the rate statements above are paper-verified at abstract resolution. The explicit
clipping and concentration equations below are a **DERIVED generic realization of the abstract's
mechanism**, not a claimed transcription of the unavailable theorem body. This limitation blocks a
positive theorem-based call budget, but it does not weaken the negative hypothesis test: the live map
already fails the fixed-operator and nonexpansiveness gates.

## Objects and norms

Let:

- `theta` be the witness parameters;
- `mu_theta` be the distribution of states produced by the current witness trajectory;
- `s ~ mu_theta` be one rendered/on-policy state;
- `g(s)` be the deterministic exact frozen-SegNet teacher target for that state, currently the
  through-R CE input-costate;
- `f_w(s)` be the amortized surrogate with parameters `w`;
- `ell(f_w(s),g(s))` be the teacher-matching loss;
- `L_theta(w) = E_{s~mu_theta}[ell(f_w(s),g(s))]`.

**MEASURED (repository read):** the live #455 implementation is a nonlinear amortized costate model
trained on states generated by the moving witness; it is not an explicit fixed-point solver with a
registered stationary operator.

For a norm `||.||_X`, the paper-level gates, as stated by the abstract, are:

`||T(x)-T(y)||_X <= gamma ||x-y||_X`, with `gamma <= 1`,

`E[T_hat(x;xi) | x] = T(x)`,

`E[||T_hat(x;xi)-T(x)||_X^2 | x] <= sigma^2`.

**DERIVED:** Euclidean/Frobenius parameter space is a quadratically smooth normed space, so the
finite-dimensional geometry is not the principal blocker. A learned weighting, quotient norm, or
function-space norm would need its own smoothability and operator-bound proof. No such alternative is
needed to reach the present negative result.

## 1. Hypothesis fit for #455

### 1.1 Three possible definitions of `T`; none establishes the live analogy

#### A. One-step optimization map

The most literal update map for a frozen state distribution is

`T_theta(w) = w - eta grad L_theta(w)`.

A same-distribution sample map is

`T_hat_theta(w;s) = w - eta grad_w ell(f_w(s),g(s))`, `s ~ mu_theta`.

For fixed `theta` and conditionally iid `s`, differentiation/interchange assumptions give

`E[T_hat_theta(w;s) | w,theta] = T_theta(w)`.

**DERIVED:** this conditional unbiasedness is only with respect to the frozen map `T_theta`. During
witness training, `theta=theta_t` and hence `T_t=T_{theta_t}` changes. Relative to any fixed reference
map `T_*`, the oracle decomposition is

`T_hat_t(w;s_t) - T_*(w) = zeta_t(w) + b_t(w)`,

where

`E[zeta_t(w) | F_{t-1}] = 0`,

`b_t(w) = T_t(w) - T_*(w)`.

The moving-distribution term `b_t` is predictable bias, not a martingale difference. The fixed-map
residual obeys the drift-debt law

`||T_*(w)-w|| <= ||T_t(w)-w|| + ||b_t(w)||`.

If `omega_j = sup_{w in C} ||T_{j+1}(w)-T_j(w)||`, then

`sup_{w in C} ||T_t(w)-T_0(w)|| <= sum_{j=0}^{t-1} omega_j`.

**DERIVED:** no amount of clipping of `zeta_t` removes this sum. A valid VR-GHAL reduction would have
to freeze `mu_theta` for a solve epoch or explicitly prove a dynamic-regret/tracking theorem with an
operator-drift term. The verified abstract is a fixed-point result, not such a tracking result.

Nonexpansiveness also fails to follow. Where the Hessian exists,

`D T_theta(w) = I - eta Hessian L_theta(w)`.

In a Euclidean norm, a sufficient global condition is that every Hessian eigenvalue lies in
`[0,2/eta]`; for a convex `L`-smooth objective, `0 < eta <= 2/L` makes the gradient step
nonexpansive. Strong convexity with curvature in `[mu,L]` gives contraction factor

`gamma = max{|1-eta mu|, |1-eta L|} < 1`

for a compatible step size.

**MEASURED (repository read):** the live model is a nonlinear neural learner. **UNKNOWN:** a global
convexity, smoothness, positive-curvature, or spectral certificate. **DERIVED:** absent such a
certificate, the map cannot be admitted to the paper's nonexpansive class.

#### B. Population best-response map

One could instead write

`B(theta) = argmin_u L_theta(u)`.

If this is single-valued, `w -> B(theta)` is constant in `w` and therefore a `0`-contraction. That
formal trick does not create a valid oracle. One teacher forward returns `g(s)` for one state; it does
not return an unbiased parameter vector whose expectation is `B(theta)`. Solving the sampled
optimization problem and averaging its minimizers is generally not the population minimizer:

`E[argmin_u ell_s(u)] != argmin_u E[ell_s(u)]`.

**DERIVED:** this definition makes the contraction hypothesis vacuous by moving the hard work inside
one oracle evaluation. It is a forced analogy and is rejected.

#### C. Coupled witness/surrogate map

To make the moving distribution stationary, augment the state `z=(theta,w)` and define schematically

`G(theta,w) = (W(theta,w), S(theta,w))`,

where `W` is one witness update using the surrogate and `S` is one teacher-matching update under
`mu_theta`. A joint fixed point would represent a self-consistent witness and surrogate.

**DERIVED:** one frozen-SegNet teacher forward evaluates only a label needed by `S`; it is not an
unbiased evaluation of the full pair `G`. **UNKNOWN:** a norm under which `G` is nonexpansive. The
feedback loop can amplify perturbations because `w` changes `theta`, which changes `mu_theta`, which
changes the regression target. Thus the augmentation restores a formal fixed map only at the price of
losing the oracle and geometry assumptions.

### 1.2 Argmax discontinuity versus the smooth target

For class logits `a(x)` and a hard label map `h(x)=argmax_c a_c(x)`, `h` is discontinuous on every
tie surface. There is no finite global Lipschitz constant in a norm that separates different class
labels. Therefore an oracle that literally targets hard argmax pixels does not satisfy a global
Lipschitz-difference assumption.

For temperature `tau>0`, the soft target

`pi_tau(x) = softmax(a(x)/tau)`

is locally Lipschitz wherever `a` is locally Lipschitz. Likewise, the live CE input-costate is smooth
inside a fixed activation/rounding cell. **DERIVED:** this repairs target regularity locally, not the
nonexpansiveness of `w -> w-eta grad L_theta(w)`. Frozen ReLU boundaries and the exact through-R
uint8/resize chain still require a fixed-cell, margin, semismooth, or generalized-Jacobian argument.
The current #454b memo independently records the same whole-ball blocker.

### 1.3 Second moment and adaptive sampling

**ASSUMED sufficient condition:** if rendered states, teacher targets, surrogate features, and the
iterate set are all bounded, then a finite uniform central second-moment bound can be derived for a
fixed replay distribution. **UNKNOWN for the live map:** no custody-bearing `sigma^2` bound is
registered, and on-policy adaptive states are not conditionally iid from a single fixed `mu`.

An empirical variance estimate would not repair the logical order. Under this repository's authority
contract, an empirical assertion would require the real `n600` witness surface; no such measurement
was launched in this design pass. Even an `n600` estimate would support a statistical model, not by
itself prove global nonexpansiveness.

### 1.4 Residual is not teacher fidelity

If `T` is a `gamma`-contraction with fixed point `w_*`, then

`||w-w_*|| <= ||w-T(w)||/(1-gamma)`.

For merely nonexpansive `T`, a small residual need not imply proximity to the desired fixed point
without an error-bound/metric-subregularity condition. For the gradient map,

`||T_theta(w)-w|| = eta ||grad L_theta(w)||`.

**DERIVED:** a small residual can be a bad stationary point of the nonlinear regression objective. It
does not imply small costate error, exact joint-controller fidelity, or preserved through-R
`d_seg`/`d_pose`. A positive #455 result therefore needs both the fixed-point theorem and a calibrated
map from residual to teacher-error and then to the exact controller gate.

## 2. The clipped-difference mechanism and concentration

This section is explicit but intentionally does not claim the unavailable paper's exact recursion.

Define norm clipping at radius `R>0` by

`Clip_R(z) = z min{1, R/||z||}`.

For paired oracle evaluations with common randomness `xi`, define

`Delta(x,y;xi) = T_hat(x;xi) - T_hat(y;xi)`.

A generic variance-reduced difference batch at step `k` is

`D_k = (1/b_k) sum_{i=1}^{b_k} Clip_{R_k}(Delta(x_k,y_k;xi_{k,i}))`,

`R_k = c gamma ||x_k-y_k||`,

and a generic recursive estimator is

`v_k = v_{k-1} + D_k`,

with periodic fresh anchor estimates. A Halpern scaffold then has the form

`x_{k+1} = alpha_k x_0 + (1-alpha_k) v_k`.

**INFERRED:** the exact VR-GHAL anchor motion, `alpha_k`, `b_k`, constant `c`, centering, and restart
schedule may differ; they are UNKNOWN until the PDF is available. The equations above encode only the
abstract-verified ideas: stochastic differences, clipping at the pair-distance scale, variance
reduction, and Halpern anchoring.

Clipping is generally biased. Conditional on the past, let `Z=Delta(x,y;xi)` and `m=E[Z|F]`. Then

`D_k-m = (D_k-E[D_k|F]) + (E[D_k|F]-m)`.

The first term is a bounded martingale term. For one sample,

`||Clip_R(Z)-E[Clip_R(Z)|F]|| <= 2R`.

The clipping bias satisfies the elementary second-moment bound

`||E[Clip_R(Z)-Z|F]|| <= E[(||Z||-R)_+|F] <= E[||Z||^2|F]/R`.

If the conditional quadratic variation across samples is `V`, a 2-smooth-Banach
Freedman/Pinelis-type inequality has the schematic form

`||sum martingale_increments|| <= C_X (sqrt(V log(1/delta)) + R_max log(1/delta))`

with probability at least `1-delta`, where `C_X` depends on the norm geometry. Splitting the failure
budget over epochs, for example `delta_e` with `sum_e delta_e <= delta`, yields simultaneous control
over all registered epochs by a union bound. This is the concentration logic behind an anytime
statement: bounded increments from clipping, quadratic-variation control from second moments, plus an
explicit clipping-bias term.

**UNKNOWN:** the exact inequality, constants, self-normalization, and failure-budget schedule used by
the paper. **DERIVED:** for the live on-policy map, the complete error decomposition needs one more
term:

`estimation error <= martingale radius + clipping bias + operator-drift debt`.

The last term is absent from a stationary fixed-map theorem and can dominate the first two.

## 3. Teacher-forward call budget for #455

### 3.1 The theorem's oracle query is not automatically one teacher forward

Let:

- `N` be the naive number of exact teacher forwards over the comparison horizon;
- `A` be fresh-anchor stochastic-operator samples;
- `D` be paired difference samples;
- `Q=A+2D` be the usual count if both sides of every difference are counted as oracle evaluations;
- `c_label` be exact teacher labels required per paired difference.

Then the teacher-forward count is

`C_teacher = A + c_label D`,

not automatically `Q`.

For squared loss with a frozen linear head `f_w(s)=Phi(s)w`, a sample gradient is

`g_s(w) = Phi(s)^T(Phi(s)w-y_s)`.

The paired gradient difference is

`g_s(w)-g_s(v) = Phi(s)^T Phi(s)(w-v)`.

The exact teacher label `y_s` cancels. **DERIVED:** after a state has been labeled once, paired
surrogate-parameter differences need no further teacher forward. If `x` and `y` encode two different
on-policy witness states, however, both states may require labels. Thus `c_label` can be `0`, `1`, or
`2` depending on the operator definition and cache custody. A theorem in oracle evaluations cannot
be converted into teacher calls until this mapping is fixed.

This cancellation also weakens the case for a separate VR-GHAL arm: ordinary exact-label replay
caching already captures the teacher-call saving in the clean convex-head formulation.

### 3.2 Honest rate translation

At abstract resolution, write the paper's bounds as

`Q_SW(epsilon,delta) <= C_SW(problem geometry, variance, delta) epsilon^-2`,

`Q_LIE(epsilon,delta) <= C_LIE(problem geometry, variance, delta) epsilon^-3`,

with logarithmic factors and exact constants absorbed into the `C` terms.

**MEASURED (paper abstract):** only the exponents are available here. **UNKNOWN:** `C_SW`, `C_LIE`,
their `delta` dependence, and which live quantities instantiate them. **DERIVED:** therefore no
integer teacher-call budget follows for a stated `epsilon,delta`; the most concrete honest gate is

`C_teacher(epsilon,delta) < N` for any strict saving,

`saving_calls = max{0,N-C_teacher}`,

`saving_fraction = max{0,1-C_teacher/N}`.

Because the live hypotheses fail and `C_teacher` is uninstantiated, the **DERIVED fail-closed
theorem-certified saving is `0%`**. This is not a claim that the surrogate cannot save calls; it says
this paper does not yet certify one call less on the live formulation.

### 3.3 Conditional `K=20` target and Amdahl translation

**DERIVED from the declared `95%` skip target:** one exact anchor per `20` witness steps gives

`N=20`, `C_teacher=1`, `saving_calls=19`, `saving_fraction=19/20=95%`.

This is the existing #455 target cadence, not a consequence of VR-GHAL. To justify it with the paper,
the fully instantiated theorem and query-to-teacher accounting would need to prove

`C_teacher(epsilon,delta) <= N/20`.

**OPERATOR-SUPPLIED:** frozen forward share `f=0.78`. If every other cost is unchanged and the
teacher-forward call fraction is `r=C_teacher/N`, Amdahl's law gives

`total_speedup(r) = 1 / ((1-f)+f r) = 1/(0.22+0.78r)`,

`wall_reduction(r) = f(1-r)`.

At the conditional `r=0.05` target:

`total_speedup = 1/0.259 = 3.861003861... x`,

`wall_reduction = 0.78*0.95 = 0.741 = 74.1%`.

All four numeric values in this paragraph are **DERIVED from the OPERATOR-SUPPLIED `78%` share and
the ASSUMED realization of the `95%` call-skip target**. They are not measured. Surrogate inference,
anchor fitting, validation, and renderer VJP costs would reduce the realized benefit.

No empirical claim is made here. In particular, no subset smoke is promoted to `n600` authority.

## 4. #454 REUSE: clipped differences versus the Jacobian-drift trust region

Let `p_x=J_x^T q_x` and let `a` be the exact anchor. The exact decomposition is

`p_x-p_a = (J_x^T-J_a^T)q_x + J_a^T(q_x-q_a)`.

If `||J_x-J_a||_op <= L_J ||x-a||`, then

`||p_x-p_a|| <= L_J ||x-a|| ||q_x|| + ||J_a||_op ||q_x-q_a||`.

The live #454b derivation strengthens this with a corrected, whole-ball cubic envelope

`E(r) = (B_J L_q + L_c)r + (B_H L_q + L_H Q_a/2)r^2 + (L_H L_q/2)r^3`,

where `r=||x-a||`, and admits reuse only when the renderer-composed error is below the guarded descent
margin. **MEASURED (repository read):** custody-bearing whole-ball constants are not yet available;
rigorous direct reuse is therefore `0`, not a positive certificate.

For a stochastic drift sample `Z`, clipping gives

`Z = Clip_R(Z) + (Z-Clip_R(Z))`.

If only `E[||Z||^2|F] <= s^2` is known, then

`P(||Z||>R | F) <= s^2/R^2`.

Across at most `M` adaptive reuse decisions with valid conditional bounds, a union bound gives

`P(max_{t<=M} ||Z_t||>R) <= M s^2/R^2`.

To make this tail at most `delta`, one must take

`R >= s sqrt(M/delta)`

or pay an equivalent explicit tail allowance. **DERIVED:** clipping without that allowance is
anti-conservative; clipping with it may be looser than the deterministic whole-ball envelope. There
is no monotone theorem implication that it admits a larger radius or more skipped forwards.

More fundamentally, for a fixed current frame and frozen network, `p_x-p_a` is deterministic. The
randomness enters only through a distribution over future on-policy states. A high-probability
population statement is not a pathwise certificate for the current state unless the controller is
explicitly changed to accept a declared failure probability and accounts for adaptive dependence.
That would be a new chance-constrained formulation, not a drop-in replacement for #454b.

Halpern anchoring mixes an iterate with a designated algorithmic anchor to obtain residual
convergence for a nonexpansive operator. It may keep algorithm iterates near that anchor, but it does
not upper-bound the error of a stale SegNet forward. **DERIVED verdict for #454:** retain the direct
Jacobian/adjoint drift trust region as the safety surface. At most, feed clipped-difference
concentration into an observability-only refresh prior; do not let it authorize reuse.

## 5. The one formulation that could fit

Freeze an exact-labeled replay distribution `nu` for a solve epoch and restrict the learner to a
convex linear/ridge head:

`F(w) = E_{s~nu}[0.5 ||Phi(s)w-y_s||^2] + (lambda/2)||w||^2`,

`T(w) = w - eta grad F(w)`.

If the population Hessian satisfies

`mu I <= H = E[Phi^T Phi] + lambda I <= L I`

with `mu>0`, and `eta` is chosen so

`gamma=max{|1-eta mu|,|1-eta L|}<1`,

then `T` is a Euclidean contraction. With fixed iid replay, the sample gradient update is unbiased.
Bounded replay features plus a proved iterate ball can supply a second-moment bound.

**DERIVED:** this repaired formulation fits in principle. **DERIVED scope:** it is not the current
nonlinear, moving-distribution #455 learner. **DERIVED economic point:** the label-cancellation law
above means replay caching, not VR-GHAL itself, performs the direct teacher-call elimination. VR-GHAL
could reduce cheap surrogate-gradient queries or give a high-probability residual guarantee, but no
strictly smaller number of exact teacher labels follows without a complete accounting.

## 6. Pre-registered decision rule and next action

No new arm should be built from this paper alone. Feed the following gate to the #455 owner; all
conditions are conjunctive:

1. **DERIVED gate `FIXED-MAP`:** freeze and hash the replay distribution for each solve epoch, or
   supply a tracking theorem with an explicit operator-drift budget.
2. **DERIVED gate `GEOMETRY`:** register the exact norm and a proof/custody artifact for
   nonexpansiveness or contraction of the actual update map.
3. **DERIVED gate `ORACLE`:** prove conditional unbiasedness relative to that same fixed map and
   register a bounded central second-moment constant.
4. **DERIVED gate `RESIDUAL-TO-FIDELITY`:** provide a contraction/error-bound constant and a mapping
   from fixed-point residual to costate error and the joint exact-controller gate.
5. **DERIVED gate `CALL-ACCOUNTING`:** write `C_teacher=A+c_label D` from actual cache behavior and
   prove `C_teacher<N`; for the declared `95%` target, prove `C_teacher<=N/20`.
6. **MEASUREMENT gate:** on the real `n600` witness, measure teacher calls, exact joint-gate fidelity,
   and complete wall-clock cost. This is the first artifact allowed to claim a 95%-kill movement.
7. **POINTER gate:** only a byte-closed exact evaluator row on the shipping archive may move the
   frontier pointer.

For #454, a monitoring-only clipped-difference feed is admissible only if its risk budget is typed,
its conditional dependence assumptions are proved, and its upper confidence bound is compared
against the existing whole-ball envelope on `n600`. It may adjust refresh priority; it must not be
the safety certificate unless the operator explicitly adopts chance-constrained reuse.

The shortest-EV path remains: finish the live #455 exact-anchor/fidelity work and the #454b
whole-ball custody work. Do not divert them into a VR-GHAL implementation.

## 7. Triality and cathedral invariant

- **Equations leg:** durable candidate equations are staged in
  `vrghal_95kill_fixedpoint_equations_20260713.md`: moving-operator residual debt,
  query-to-teacher accounting, and clipped-drift tail debt.
- **DAG leg:** a research-only FEED graph is staged in
  `vrghal_95kill_fixedpoint_DAG_FEED_20260713.md` with no actuation authority.
- **DSL leg:** **DEFERRED to main/live owners.** The relevant trainer and certificate surfaces are
  live-sibling-owned, and the result is a feed rather than an admitted arm. A future typed gate would
  need fields for fixed-operator epoch/hash, norm/proof references, operator-drift budget,
  oracle-variance custody, failure probability, and teacher-call accounting. No flags are invented
  here.
- **Shared equation registry:** **DEFERRED to main.** The shared JSONL is hot with sibling activity;
  this lane writes only isolated new files.
- **Pointer delta:** `NONE`.
- **Triality status:** equations and DAG feed landed locally; executable DSL integration deliberately
  absent and explicitly scoped.

## STORES CONSULTED

- `CLAUDE.md` — full binding contract.
- `AGENTS.md` — full binding contract and anti-collision list supplied by the operator.
- `docs/operating_manual_craft_handoff.md` — full craft/authority handoff.
- `PROGRAM.md` — current goal framing.
- `.omx/research/onpolicy_surrogate_95kill_20260713.md` — live #455 formulation and authority state.
- `.omx/research/jacobian_drift_certificate_95kill_20260713.md` — live #454b identity, envelope, and
  blockers.
- `src/tac/scorer_surrogate/onpolicy_costate.py` and
  `src/tac/scorer_surrogate/amortized_onpolicy_costate.py` — live learner shape.
- `.omx/state/lane_registry.json` and `.omx/state/subagent_progress.jsonl` — collision/ownership state.
- `reports/latest.md` and canonical pointer surfaces — pointer separation.
- [arXiv abstract 2607.09097](https://arxiv.org/abs/2607.09097) — primary source successfully read.
- [arXiv PDF 2607.09097](https://arxiv.org/pdf/2607.09097) — primary source requested but inaccessible
  in this environment; exact theorem body remains UNKNOWN.

## Final authority caveat

**DESIGN / MEANS only.** Only a **MEASURED** teacher-call reduction on the real `n600` witness can
move the 95%-kill program. Only a byte-closed, exact evaluator row on the shipping bytes can move the
frontier pointer. NumPy-fp32 remains the portable reference authority; MPS is never a score. No
training, heavy job, paid dispatch, live-run actuation, or sibling-surface edit occurred in this pass.

