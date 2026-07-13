---
title: "P0 backward elimination: costate reuse, forward-only gradient-free crossover, and trajectory adjoints"
date_utc: "2026-07-13"
checkpoint_id: "p0_costate_reuse"
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
authority: "local cached pair-0 training-gradient evidence; NumPy-fp32 reduction; macOS-CPU advisory"
---

# P0 costate reuse + gradient-free training

## Answer first

The input costate is **locally reusable but not slowly varying enough to justify a blind long
cadence**. On the three cached task-455 saved regimes, the exact input-costate field after five
student updates has NumPy-fp32 cosine `[0.85924369, 0.84703696, 0.86481547]` against its anchor and
relative L2 error `[0.58128232, 0.57854396, 0.57641286]` for `[early, boundary, late]`. The rendered
frame moved only `3.05%`, `5.00%`, and `5.23%` in relative L2, so the costate is about `19.03x`,
`11.57x`, and `11.03x` more sensitive than the state under this finite displacement. This is
**MEASURED** on `3 x 589,824` float32 field pairs, not an inference from scalar losses.

The companion task-449 YOPO receipt is more favorable locally: its fresh-prefix reconstruction of a
banked first-block costate has non-refresh age-1 cosine median `0.9999823195` and minimum
`0.9998774505`; the one observed age-3 field still has cosine `0.9999982044`. But exact candidate
non-descent occurs on `3/8` non-refresh rows, including the age-3 row. Thus **high field cosine is not
a sufficient receiver-cell/descent certificate**.

Disposition:

| Angle | Scoped verdict | What survives |
|---|---|---|
| Costate reuse | **WORTH-BUILD**, probe scope only | Event-controlled **`K_max=2`**: one zero-order reuse attempt after each exact anchor, exact forward-only post-step guard, rollback+refresh on any CE/`d_seg`/`d_pose` failure, and unconditional refresh at stage/event boundaries. No fixed cadence is admitted. |
| Forward-only ES/SPSA | **NO-GO** for bulk training | Under an optimistic Rademacher-SPSA law, forward-only beats one exact fwd+bwd only when the active dimension is at most `2`. The cached bulk state has `111,095` numeric fp32 degrees of freedom and the #396 terminal head has `791`; neither is close. #396's sparse exact-metric accept/reject finisher remains **WORTH-BUILD/already routed** because it does not try to estimate the full bulk gradient. |
| One integrated adjoint trajectory | **NO-GO** as a per-step SegNet-VJP replacement; **FEED-454** for low-dimensional schedule control | A horizon adjoint differentiates all controls of a fixed trajectory in one reverse sweep, but the online state equation itself contains `J_R^T J_S^T q` at every step. The outer adjoint nests, rather than deletes, the scorer VJP. It is useful for a small stage/schedule controller, not for eliminating the inner teacher backward. |

**Overall scoped verdict: `WORTH-BUILD` exactly one changed formulation — a fail-closed,
event-conditioned `K_max=2` direct-costate replay probe. Fixed open-loop `K>=2`, generalized bulk
gradient-free training, and “one adjoint sweep removes all teacher VJPs” are not admitted.** The
smooth-loss economic model has a conditional nominal optimum `K=6` over the measured five-step
support, but receiver-discrete failures falsify promoting that number into an execution cadence.

This memo is MEANS. It created no archive, ran no evaluator, changed no trainer/costate/live file,
and moved no pointer.

## Scope and authority

- `MEASURED` below means re-reduced from preserved local receipt/checkpoint bytes in this turn.
- `DERIVED` means algebra from those measurements or a stated smoothness model.
- `ASSUMED` marks a model closure that is not an empirical certificate.
- Axis: `[macOS-CPU advisory training-gradient]`; pair `0`; seed `455`; local cached data only.
- The three task-455 states are epochs `299`, `726`, and `925`. They are distinct regimes, not three
  independent seeds or pairs. Across-pair, across-seed, contest-CPU, contest-CUDA, and live-V9
  transfer are **UNKNOWN**.
- The `82.1524%` backward / `17.8476%` forward split comes from the task-455 diagnostic harness.
  The per-epoch accounting explicitly says its transfer to the in-loop trainer is **UNRESOLVED**;
  absolute diagnostic times are about `12x` heavier than the in-loop path. Every speed number here
  is therefore a diagnostic conditional, not a launch promise.
- Exact score authority remains `upstream/evaluate.py` on exact archive bytes. No score surface was
  exercised.

## Stores consulted

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, and `docs/operating_manual_craft_handoff.md`.
- The v7.5/v8 canonical specs, especially the settled table, operating contract, and measurement
  authority.
- `reports/latest.md`, `tac.frontier_scan.build_frontier_scan_payload`, lane/task/subagent state,
  latest sister findings/session summaries, current council/design memo, and recent directives.
- `.omx/research/per_epoch_detailed_accounting_20260713.md`.
- `.omx/research/codex_findings_master_oss_reconciliation_20260713_codex.md` and task-449 YOPO
  receipt `experiments/results/yopo_first_layer_costate_probe_20260713T003635Z/receipt.json`,
  SHA-256 `a89585cd70b9630c90468f3a502e1efc778836cffc56ca7fb71e997fff2e6fa3`.
- `.omx/research/onpolicy_surrogate_95kill_20260713.md`; final task-455 campaign receipt
  `experiments/results/onpolicy_costate_matched_campaign_final_20260713T043000Z.json`, SHA-256
  `5b73396f4990a0d7d44fd358d64fc87d4b3e442dc7ac7a34f1264013fae5aff8`; its final
  early/boundary/late receipts and preserved stage checkpoints.
- `.omx/research/jacobian_drift_certificate_95kill_20260713.md` and its terminal receipt.
- `.omx/research/frozen_replay_convex_head_95kill_20260713.md`, receipt SHA-256
  `067ce197d30fa9e2c7c4bda48ac671af550e0a00f126289ba5b30946d44fc4b1`.
- `.omx/research/mc_finisher_396_design_20260710.md` and the local papers-checked note for
  arXiv `2607.08406`.

No web, cloud, paid provider, GPU, live run, scorer launch, or trainer actuation was used.

---

## Angle 1 — costate reuse

### 1. What is being reused

For rendered receiver state `x_n=R(theta_n)`, frozen SegNet `S`, and smooth training loss `ell`,

```text
lambda_n = grad_x ell(S(x_n), y),
g_n      = J_R(theta_n)^T lambda_n.
```

Zero-order hold (ZOH) reuses `lambda_a` for steps after anchor `a`:

```text
lambda_hat_(a+j) = lambda_a,
g_hat_(a+j)      = J_R(theta_(a+j))^T lambda_a.
```

YOPO instead banks an internal costate `p1_a` and recomputes the current prefix pullback:

```text
lambda_hat_(a+j) = J_prefix(x_(a+j))^T p1_a.
```

These are not the same estimator. The direct task-455 measurement below tests raw input-costate
ZOH across five updates. The task-449 receipt tests a better transported internal costate and is used
as an upper-quality local comparator, not laundered into a raw-ZOH result.

The frozen-replay `12x` teacher-call amortization is also a different fact: it reuses exact labels on
the **same fixed states** for `7,200` convex-head uses after `600` teacher state calls. Its heldout
costate cosine is only `0.0014157934`. It proves same-state label-cache economics; it does not prove
temporal costate stationarity.

### 2. Direct temporal autocorrelation — MEASURED from cached task-455 fields

The task-455 collection loop sets its stored anchor exactly once at collection step `0`, advances the
student for five contiguous collection updates, and the deployment branch then refreshes with an exact
teacher costate at the post-collection state. Thus the preserved collection and surrogate-stage
`anchor_costate` tensors are exact `lambda_0` and exact `lambda_5`; neither is a learned prediction.

Reducer authority is NumPy-fp32:

```text
dot  = sum_fp32(lambda_0 * lambda_5)
n0   = sqrt(sum_fp32(lambda_0^2))
n5   = sqrt(sum_fp32(lambda_5^2))
rho5 = dot/(n0*n5)
e5   = ||lambda_0-lambda_5||_2 / ||lambda_5||_2
r5   = ||lambda_0||_2 / ||lambda_5||_2
```

Each field has shape `(1,3,384,512)`, or `589,824` fp32 elements. Here `rho_lambda(5)` is the
uncentered normalized lag-5 field autocorrelation. The corresponding mean-centered fp32
correlations are `0.85924339`, `0.84703684`, and `0.86481553`, so a field-wide DC component is not
creating the result. A separate float64 accumulation audit differs from fp32 cosine by at most
`2.151e-7` and from relative L2 by at most `8.798e-8`.

| Regime / saved epoch | `rho_lambda(5)` NumPy-fp32 | stale/current norm | relative L2 `e5` | frame relative L2 | `e5/e_x` |
|---|---:|---:|---:|---:|---:|
| early / 299 | `0.85924369097` | `1.13526678085` | `0.58128231764` | `0.03054558672` | `19.02999x` |
| boundary / 726 | `0.84703695774` | `1.07547795773` | `0.57854396105` | `0.05000939220` | `11.56871x` |
| late / 925 | `0.86481547356` | `1.14793634415` | `0.57641285658` | `0.05226626247` | `11.02839x` |
| min / median / max | `0.84703696 / 0.85924369 / 0.86481547` | `1.07548 / 1.13527 / 1.14794` | `0.57641 / 0.57854 / 0.58128` | — | — |

Checkpoint custody:

| Regime | exact anchor `lambda_0` | exact post-5-update `lambda_5` |
|---|---|---|
| early | collection SHA `4e42aa74674e...` | surrogate-stage SHA `fbacf92cde22...` |
| boundary | collection SHA `d8f15f89bacb...` | surrogate-stage SHA `30ea70668a3f...` |
| late | collection SHA `f166c7661755...` | surrogate-stage SHA `53971f88a5a2...` |

**MEASURED conclusion:** the costate retains a useful descent-scale direction at lag 5, but calling it
“slowly varying” without qualification would be false. A `~0.85` cosine coexists with `~0.58`
relative error, and a small state displacement is magnified by more than `11x`. This looks like a
smooth but ill-conditioned adjoint map, not a nearly constant field.

### 3. The stronger local transport result — MEASURED task-449 comparator

Across the eight non-refresh YOPO rows in the final receipt:

| bank age | rows | global cosine min / median / max | relative L2 min / median / max | exact candidate non-descent |
|---:|---:|---:|---:|---:|
| 1 | 6 | `0.9998774505 / 0.9999823195 / 0.9999996796` | `0.0008016354 / 0.0052982006 / 0.0157259833` | `2/6` |
| 2 | 1 | `0.9999996229` | `0.0008733230` | `0/1` |
| 3 | 1 | `0.9999982044` | `0.0019119858` | `1/1` |

The minimum boundary-annulus cosine is `0.9998451426`; minimum renderer-gradient cosine is
`0.9999437091`. Yet `3/8` non-refresh candidates fail the exact event-conditioned descent surface.
The receipt's registered all-regime verdict is already `NO-GO` for its first-block split and
`K in {1,2,4}` because validation and non-descent close the formulation.

This is the adversarial discriminator:

```text
costate cosine high  !=  J_R-conditioned direction certified
J_R direction high   !=  finite step preserves discrete receiver cells
CE descent           !=  d_seg/d_pose nonworsening
```

### 4. Derived cadence/error/economics law

Let `alpha=T_fwd/(T_fwd+T_bwd)`. The final task-455 diagnostic samples give

```text
alpha = 0.5370454629 / 3.0090696110 = 0.1784755863  [DERIVED from MEASURED]
1-alpha = 0.8215244137.
```

If every `K`-step cycle pays one exact fwd+bwd anchor and one forward-only receiver guard on each
reused step, the optimistic normalized teacher cost and ceiling speedup are

```text
C_K = alpha + (1-alpha)/K,
speedup_K = 1/C_K.
```

This omits PoseNet, renderer, rollback, IO, and optimizer cost and uses the unresolved diagnostic
ratio. It is an optimistic economic screen only.

For local `L`-smooth loss, let `rho_g(j)=cos(g_(a+j),g_hat_(a+j))`, after renderer projection. The
best scalar step along `g_hat` has modeled progress fraction `rho_g(j)^2` relative to exact steepest
descent. Therefore the canonical progress-per-wall functional is

```text
Q_K = (1/K) * sum_(j=0)^(K-1) rho_g(j)^2,
E_K = Q_K / [alpha + (1-alpha)/K],
K*  = argmax_K E_K,
```

subject to exact receiver rollback, stage-boundary refresh, and a measured renderer-conditioning
certificate. Input-costate cosine `rho_lambda` may replace `rho_g` only as an explicitly optimistic
proxy: if `A=J_R^T`, then

```text
||A(lambda_hat-lambda)|| / ||A lambda|| <= kappa_A *
    ||lambda_hat-lambda||/||lambda||,
```

and `kappa_A` is **UNKNOWN** for the direct lag-5 fields.

To make the sparse lag data interpretable, fit an **ASSUMED exponential interpolation only inside the
measured interval** to the worst measured raw-field endpoint:

```text
rho_lambda(j) = r^j,
r = 0.84703695774^(1/5) = 0.96734295244,
tau = -1/log(r) = 30.1185 update units.
```

This produces the following **DERIVED-CONDITIONAL, non-admission** curve:

| exact cadence `K` | max reused age | endpoint `rho` | normalized direction error | `Q_K` | `C_K` | teacher ceiling | smooth proxy `E_K` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | `1.000000` | `0.0000` | `1.000000` | `1.000000` | `1.000x` | `1.000x` |
| 2 | 1 | `0.967343` | `0.2556` | `0.967876` | `0.589238` | `1.697x` | `1.643x` |
| 3 | 2 | `0.935752` | `0.3585` | `0.937128` | `0.452317` | `2.211x` | `2.072x` |
| 4 | 3 | `0.905193` | `0.4354` | `0.907690` | `0.383857` | `2.605x` | `2.365x` |
| 5 | 4 | `0.875633` | `0.4987` | `0.879498` | `0.342780` | `2.917x` | `2.566x` |
| 6 | 5 | `0.847037` | `0.5531` | `0.852494` | `0.315396` | `3.171x` | `2.703x` |

So the smooth identity-conditioned model prefers `K=6` over the support actually measured. That is
the Pontryagin opportunity. It is **not** the execution answer: task-449 observes finite exact-metric
failure already at ages `1` and `3`, and raw `kappa_A` is absent.

### 5. Why the next build is `K_max=2`, not fixed `K=6`

`K_max=2` is **DERIVED as the smallest decisive changed-formulation probe**, not claimed as an
admitted optimum:

1. It makes exactly one reuse attempt, so failure localizes to the stale costate rather than a chain
   of stale updates.
2. Age `1` is the only non-refresh age observed across all three regimes. Ages `2-3` exist only in
   one late arm; raw ZOH lag `1` is not measured at all.
3. `K=4` already has a measured age-3 exact-metric failure despite cosine `0.9999982`; open-loop
   continuation is falsified for that formulation.
4. No task-449 `K>1` passes the strict all-regime admission gate. Therefore `K_max=2` must be an
   event-controlled **probe with rollback**, not a fixed live policy.

Required probe law:

```text
anchor: exact lambda_a; preserve complete anchor/stage checkpoint
attempt: one candidate using stop_gradient(lambda_a)
guard: exact forward CE + through-R d_seg + d_pose on the same candidate
accept: CE strictly decreases AND d_seg,d_pose do not worsen
else: rollback bytes exactly; force exact backward refresh
always refresh: stage boundary, curriculum event, topology/birth event, or custody drift
```

The guard cost must be measured in-loop. The task-449 validation path used `402` forwards for `48`
teacher fwd+bwd calls (`8.375` forwards/call) because it included fractional recession and controls;
that formulation was slower than exact. The new one-candidate law is worth a bounded probe only if it
actually emits approximately one guard forward per reuse and if the in-loop timer confirms backward
dominance.

### 6. Extrapolation: mathematically attractive, empirically premature

For update spacing `h`, if `||d lambda/dt|| <= M1`, ZOH has

```text
||lambda_(a+j)-lambda_a|| <= M1*j*h.
```

If two exact anchors are available and `||d^2 lambda/dt^2|| <= M2`, linear extrapolation

```text
lambda_hat_(a+j) = lambda_a + j*(lambda_a-lambda_(a-1))
```

has the Taylor bound

```text
||lambda_(a+j)-lambda_hat_(a+j)|| <= (M2*h^2/2)*j*(j+1).
```

But extrapolation amplifies independent anchor noise by variance factor `(1+j)^2+j^2`, and the
cached raw fields contain only two exact temporal anchors per regime. `M2`, costate repeat noise, and
renderer conditioning are all **UNKNOWN**. Therefore first-order extrapolation is **FEED-454**, not
the first build. ZOH `K_max=2` is the only clean next discriminator.

### Angle-1 verdict scope

- **WORTH-BUILD:** one local, bounded, resumable, event-controlled raw-input-costate `K_max=2` probe
  with full-facet measurement and exact rollback.
- **NO-GO:** treating `K=6` as a live cadence, treating cosine as a certificate, or reopening the
  measured task-449 first-block `K={1,2,4}` formulation unchanged.
- **FEED-454:** curvature-bounded extrapolation, renderer-condition conversion, and a cheap event
  certificate.

---

## Angle 2 — forward-only ES / SPSA / score-function training

### 1. Best-case variance law

Work in an `r`-dimensional active subspace and locally linearize the deterministic objective. For a
Rademacher direction `Delta in {-1,+1}^r`, the antithetic SPSA estimate is

```text
g_hat_1 = [(f(theta+c Delta)-f(theta-c Delta))/(2c)] Delta
        = (Delta Delta^T) g + O(c^2).
```

For `m` independent direction pairs and their mean:

```text
E[g_hat_m] = g + O(c^2),
E||g_hat_m-g||^2 = ((r-1)/m)||g||^2 + O(c^2),
E||g_hat_m||^2 = [1+(r-1)/m]||g||^2 + O(c^2).
```

Under an `L`-smooth objective, optimizing the scalar learning rate gives the optimistic expected
progress fraction

```text
q_ES(r,m) = 1/[1+(r-1)/m] = m/(m+r-1).
```

Gaussian ES is worse by two dimensions in this calculation: its trace variance is
`(r+1)||g||^2/m`. The Rademacher law is therefore the favorable adversarial comparator.

### 2. Forward-cost crossover

Each antithetic direction consumes two SegNet forwards. With diagnostic `alpha=0.1784755863`,

```text
C_ES(m) = 2*alpha*m = 0.3569511725*m
```

in exact fwd+bwd equivalents, optimistically excluding renderer and update costs. Forward-only wins
in progress per wall time only if

```text
q_ES/C_ES > 1
<=> 1/[2*alpha*(m+r-1)] > 1
<=> m+r-1 < 1/(2*alpha) = 2.8015036146.
```

The integer crossover is brutal:

- `m=1`: win only for `r<=2`;
- `m=2`: win only for `r=1`;
- `m>=3`: no active dimension wins against one exact fwd+bwd under this model.

Even granting a permanently free cached one-sided baseline improves the right-hand side only to
`1/alpha=5.603`; it does not rescue tens, hundreds, or thousands of active directions. A counted
baseline removes most of even that advantage.

### 3. Our dimensions and implied penalty

The three sealed task-455 renderer checkpoints each contain **MEASURED `111,095` numeric fp32
values**: `38,400` pair/frame code values plus `72,695` shared network/head/palette values. The
current probe chart optimizes only one pair's `32`-vector, but bulk training updates the larger
coupled state. The #396 named terminal tensors contain **DERIVED `791` values**:

```text
out_sdf: 5*96 + 5 = 485
palette: 5*3       = 15
out_tex: 3*96 + 3 = 291
total               = 791.
```

For the best one-pair SPSA estimator (`m=1`):

| active dimension `r` | optimistic progress fraction `q` | ES cost | progress/time vs exact | slowdown |
|---:|---:|---:|---:|---:|
| 1 | `1.0` | `0.35695` | `2.8015x` | wins |
| 2 | `0.5` | `0.35695` | `1.4008x` | wins |
| 3 | `0.3333` | `0.35695` | `0.9338x` | `1.071x` |
| 32 | `0.03125` | `0.35695` | `0.08755x` | `11.42x` |
| 791 | `0.0012642` | `0.35695` | `0.003542x` | `282.35x` |
| 111,095 | `9.0013e-6` | `0.35695` | `2.5217e-5x` | `39,655.5x` |

Reaching signal-to-noise `>=1` requires `m>=r-1`: at `r=111,095`, at least `111,094`
antithetic pairs, costing approximately `39,655` exact-gradient equivalents before renderer cost.

These are favorable local-linear numbers. The true exact contest metric is piecewise constant under
uint8/resize/argmax until a receiver-cell boundary is crossed, which adds finite-difference bias and
sparse heavy-tailed returns; ordinary small-`c` ES is worse, not better.

### 4. Why #396 still makes sense at the terminal band

#396 does not estimate a dense unbiased gradient. It uses a guided proposal prior, mutates a sparse
element batch or discrete code, and ratchets only after exact through-R improvement. It is a search
for **one beneficial move**, not a reconstruction of all `791` gradient coordinates. That changes the
sample-complexity question from dimension-wide MSE to proposal hit rate.

Therefore the regime split is:

- **Bulk CE/tau/l7 descent:** `NO-GO`. Active dimension is not measured anywhere near `<=2`; exact
  backprop wins by orders of magnitude.
- **Terminal discrete band:** retain #396's exact-metric `(1+1)`/guided-coordinate finisher. It can
  win when the proposal prior collapses the actionable support to one or two moves and gradients are
  blind to argmax crossings. Its own paired guided-vs-blind hit-rate falsifier remains binding.
- **Potential bridge:** residual ES around a reused exact costate could estimate only a low-rank
  correction. It becomes worth testing only after a measured residual active rank `r_eff<=2` (or at
  most `<=5` under the unrealistic free-baseline bound). No current receipt establishes that.

### Angle-2 verdict scope

**NO-GO** for ES/SPSA/score-function as a replacement for bulk SegNet backprop on the measured witness
states. **WORTH-BUILD/already routed** remains the narrow #396 terminal exact-metric accept/reject
finisher. This negative is not a family kill for low-dimensional black-box controllers or discrete
terminal search.

---

## Angle 3 — the adjoint as a smooth flow

### 1. The correct optimal-control formulation

For a training trajectory treated as controlled dynamics,

```text
dot(theta) = F(theta,u,t),
J[u] = Phi(theta(T)) + integral_0^T l(theta,u,t) dt,
H(theta,p,u,t) = l(theta,u,t) + p^T F(theta,u,t).
```

The outer Pontryagin costate obeys

```text
-dot(p) = grad_theta l + (dF/dtheta)^T p,
p(T) = grad_theta Phi,
delta J/delta u = grad_u l + (dF/du)^T p.
```

For a **fixed differentiable forward trajectory**, one reverse integration computes gradients with
respect to every low-dimensional control `u(t)`. This is a real amortization when `u` is a schedule:
stage times, a handful of loss weights, or event-controller parameters.

### 2. Why it does not remove the SegNet backward

The online witness dynamics contain the teacher gradient:

```text
F_n(theta_n,u_n) = Optimizer_n(J_R(theta_n)^T lambda_n, u_n),
lambda_n = J_S(R(theta_n))^T q_n.
```

Evaluating `F_n` at each forward state already requires `lambda_n`; differentiating `F_n` for the
outer adjoint additionally requires Hessian/Jacobian-vector terms through this map. The resulting
calculation is an adjoint of an optimizer whose vector field contains an inner adjoint. A single
outer forward/backward sweep changes memory scheduling and exposes all control gradients, but it
does not algebraically reduce `N` inner `J_S^T q_n` evaluations to one.

Task-454 supplies the empirical adversarial check. Its exact first drift correction is

```text
c_a(h) = (D J_a[h])^T q_a,
```

but the faithful fixed-adjoint HVP costs `3.350555354 s` median versus `1.688966708 s` for a fresh
exact input-costate shadow and at least `2.552848896` matched validation equivalents per corrected
step. Early/boundary prefix costs `[16.3212, 26.4305]` exceed the inherited `8.375` baseline. The
mathematically faithful “integrate the changing adjoint” correction is more expensive than refreshing
it on this substrate.

### 3. Smoothness is local and event-conditioned

Inside a fixed smooth scorer/receiver cell, `lambda(x)` can be locally Lipschitz and an ODE view is
useful. Across the actual training trajectory, three mechanisms create jumps or high curvature:

1. uint8/resize/round and argmax receiver-cell crossings;
2. stage and optimizer changes (CE -> tau -> l7 -> Muon) and event-fired loss composition;
3. topology/birth/separatrix events, precisely where the costate matters most.

The cached lag-5 result (`rho≈0.85`, `e≈0.58`) is consistent with a locally smooth but strongly
conditioned field. It is not evidence for one globally smooth adjoint ODE across stage boundaries.

### 4. What the flow view is good for

Route the outer adjoint to **FEED-454** as a schedule/controller tool:

- freeze a short, checkpointed trajectory segment;
- choose a genuinely low-dimensional `u` (stage-boundary weights/events, not all witness params);
- compute one outer adjoint over that fixed segment;
- keep the inner costates exact or supplied by a separately admitted reuse provider;
- compare its total HVP/VJP cost against finite-difference schedule probes;
- reset at every nonsmooth stage/topology event.

This can amortize gradients across many control coordinates. It cannot be described as eliminating
the SegNet backward unless a separate low-rank/reuse/surrogate theorem removes the inner VJP.

### Angle-3 verdict scope

- **NO-GO:** “integrate one costate trajectory once” as a replacement for per-step frozen-SegNet
  input VJPs in online witness training.
- **FEED-454:** one outer adjoint for a small fixed-horizon stage/event controller, with exact inner
  costates and explicit HVP economics.

---

## Canonical equation candidate

Candidate id: `event_conditioned_costate_cadence_efficiency_v1`.

```text
lambda_n = grad_x ell(S(R(theta_n)),y)
g_n = J_R(theta_n)^T lambda_n
g_hat_(a+j) = J_R(theta_(a+j))^T P_j(lambda_a,lambda_(a-1))

rho_g(j) = <g_n,g_hat_n>/(||g_n|| ||g_hat_n||)
Q_K = K^-1 sum_(j=0)^(K-1) rho_g(j)^2
C_K = alpha + (1-alpha)/K
K* = argmax_K Q_K/C_K

admit(K) iff:
  exact receiver guard is monotone on CE,d_seg,d_pose for every reused step
  AND stage/event/custody boundaries force refresh
  AND measured total in-loop wall speedup > 1
  AND complete rollback/resume bytes are preserved.
```

Empirical anchor proposed:

```text
pair0, seed455, epochs {299,726,925}, lag5 raw input-costate:
rho={0.85924369,0.84703696,0.86481547},
relative_l2={0.58128232,0.57854396,0.57641286};
task449 transported-costate nonrefresh min cosine=0.99987745,
but exact non-descent=3/8 and no admitted common K>1.
```

Registration is **DEFERRED_MAIN**. The canonical equation registry is a held shared file with current
uncommitted edits, and the user constrained this lane to new files only. The equation must remain a
candidate until a real `K_max=2` raw-ZOH probe measures renderer-gradient direction, full-facet guard,
and in-loop wall time. Registering the smooth proxy now would overstate its authority.

## DAG FEED handoff

Target existing node: `FEED-p0-backward-wave-20260713` / arm `p0_costate_reuse`.

Proposed append:

```text
FEED-p0-costate-reuse-gradfree-20260713 — MEASURED from cached task-455 exact fp32
input-costates: lag-5 rho early/boundary/late = 0.85924369/0.84703696/0.86481547 and
relative-L2 = 0.58128232/0.57854396/0.57641286. Task-449 transported first-block
costates remain >0.9998 cosine but exact candidate non-descent occurs on 3/8 nonrefresh
rows; cosine is not a receiver-cell certificate. DERIVED diagnostic alpha=0.178475586;
smooth-only K*=6 over measured support is NON-ADMISSION. Route only an event-controlled
raw-ZOH Kmax=2 bounded probe with exact rollback; fixed cadence remains unadmitted.
Forward-only unbiased SPSA wins only for active r<=2; bulk r=111095 and terminal-head
r=791 imply NO-GO for bulk, while #396 terminal accept/reject remains open. A horizon
outer adjoint does not delete inner SegNet VJPs; FEED-454 only for low-dimensional schedule
control. score_claim=false; pointer_moved=false; memo=
.omx/research/p0_costate_reuse_gradfree_20260713.md.
```

DAG append is **DEFERRED_MAIN** because the canonical DAG is a hot shared file and this mission permits
new files only. No patch was applied.

## Triality and cathedral close

- **DSL:** N/A for this analysis-only unit. A future build needs a typed, default-off
  `CostateCadencePolicy`; no flags or trainer wiring were invented here.
- **Equation:** candidate `event_conditioned_costate_cadence_efficiency_v1`, held from registration
  until the direct `K_max=2` empirical anchor exists.
- **DAG:** exact FEED text above, deferred to main serializer/hot-file custody.
- **Durable artifact:** this memo, uncommitted as requested.
- **Resumability:** no run was launched. The proposed probe requires atomic anchor, candidate,
  rollback, stage, RNG, optimizer, EMA, and event-state checkpoints before actuation.
- **Pointer delta:** none. Canonical frontier scan still supplies the defensive existing
  `[contest-CPU]` anchor; this unit produced no new archive or score.

## Exact reactivation and kill criteria

`WORTH-BUILD` reactivates only after the in-loop component timer confirms that teacher backward is
the dominant transferable share. Then run one bounded `K_max=2` probe across early/boundary/late
saved states with raw exact `lambda`, not the already-closed YOPO split.

Promote beyond probe only if all are MEASURED:

1. every non-anchor candidate preserves strict CE descent and nonworsening exact through-R `d_seg`
   and `d_pose`, with deterministic repeat floors;
2. renderer-gradient cosine/dot remains positive under a content-bound provider;
3. rollback and resume are byte-identical with zero hidden teacher calls;
4. complete in-loop cost, including the guard and fallback, beats exact by a preregistered margin;
5. at least early/boundary/late, multiple pairs, and multiple seeds share a nontrivial cadence;
6. stage/topology events reset the anchor.

Kill the direct fixed-cadence formulation if `K_max=2` has any reproducible stale-only full-facet
failure while the matched exact direction passes, or if complete wall time is not below exact. Scope
that kill to direct raw-ZOH under the tested policy; do not kill internal-cut transport, learned
residual providers, low-dimensional controller ES, or terminal exact-metric search.
