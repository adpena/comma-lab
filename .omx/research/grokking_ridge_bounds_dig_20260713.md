---
title: "Ridge-grokking bounds dig: Round-2 autopsy and witness-stage transfer audit"
date: 2026-07-13
lane_id: lane_grokking_ridge_bounds_20260713
subagent_id: grokking_ridge_reader
research_only: true
authority: "[macOS-CPU advisory; NumPy-fp32 fit evidence; frozen CPU SegNet heldout targets; no evaluator-score authority]"
review_status: "SELF-AUDITED; UNREVIEWED BY MAIN"
pointer_delta: "NONE"
source_run_mutated: false
verdict: "FEATURE_POVERTY_FORMULATION_NOT_UNDERTRAINED"
verdict_scope: "fixed 31-feature linear Round-2 chart, registered real-n600 replay, and measured seven-point ridge ladder"
---

# Ridge-grokking bounds dig — Round-2 autopsy first

## ROUND-3 ADDRESSED FIRST — do not reject another feature set before this guard

**Round-2 verdict: FEATURE-POVERTY, not UNDERTRAINED, at formulation × instance
scope.** The Xu–Vardi–Safran grokking lower bound is not directly applicable to
this fit, but the actual fixed quadratic admits a stronger exact terminality
certificate. Applied separately to each RGB costate coordinate, the committed
head has `m=31`, `n=1,474,560` sampled feature rows per coordinate, zero
initialization, and spectral ridge. Consequently:

- the paper's overparameterized condition fails: `m-n=-1,474,529`;
- the random empirical-null initialization responsible for its delayed phase is
  absent: `nu^2=0` and `P_null(X) W_0=0` exactly;
- the actual null-space multiplier is
  `|1-eta*lambda|=0.3333333168094761`, so 15 steps retain only
  `6.969166755535331e-08` of even a hypothetical nonzero null component;
- the measured global contraction bound is likewise
  `gamma^15=6.969175963477342e-08`;
- the terminal-gradient certificate gives
  `||W_15-W*||_F <= 2.6011187e-15`, and the measured residual is
  `2.2703187e-15`.

The `$0` real-heldout refit then removes the remaining ambiguity. Extending the
same fit from 15 to 150 deterministic full-batch steps changes heldout cosine
from `0.001415793417951615` to `0.0014157934642280926`, a delta of only
`4.627647760226061e-11`; its maximum weight change is
`8.881784197001252e-16`. Exact optima over seven ridge scales improve the best
cosine only to `0.007690592649965529`, with relative L2
`1.0007586082750441`. More steps and removal of the original ridge choice do
not recover the target. Richer/nonlinear features are therefore the live cure;
this result strengthens the premise of `replace_round3_fidelity_wall` without
editing or judging that sibling's artifacts.

### Required Round-3 fitting protocol

Xu–Vardi–Safran prove the direction **larger weight decay and smaller
initialization suppress the delay**; the numerical constants below are our
derived fixed-quadratic protocol, not a quotation from the paper.

For each fixed Round-3 feature matrix `X`, define

```text
A = X^T X / n
H_lambda = A + lambda I
mu = lambda_min(A) + lambda
L  = lambda_max(A) + lambda
eta = 2 / (L + mu)
gamma = (L - mu) / (L + mu)
W_0 = 0                      # nu^2 = 0; removes the paper's slow null mode
```

Use a declared ladder
`lambda/lambda_max(A) in {0, 1e-6, 1e-4, 1e-2, 1e-1, 1, 10}`. For every
positive-ridge arm, run the above deterministic GD until

```text
||grad F_lambda(W_t)||_F / mu <= epsilon_W
```

or compute the exact/CG optimum. For the zero-ridge arm, use a declared
rank-thresholded minimum-norm solve; do not describe an underdetermined
zero-ridge iterate as terminal without its range-space residual. Select or
reject features only from heldout fidelity after this optimization certificate,
not from a fixed step count.

For the original Round-2 spectral arm this specializes to the explicit
delay-eliminating control:

```yaml
lambda: 3.2247040271759033
learning_rate: 0.20673732459545135
steps: 15
initialization_scale_nu: 0.0
eta_times_lambda: 0.6666666831905239
contraction_gamma: 0.3333333461703458
terminal_parameter_error_bound: 2.6011187e-15
```

The 150-step replay is a measured no-movement control, not the recommended
default. For a new feature chart, recompute `mu`, `L`, and the terminality
certificate; do not cargo-cult these Round-2 scalar constants.

## 1. Paper law and actual Round-2 autopsy

### What the paper actually proves

Xu, Vardi, and Safran analyze deterministic full-batch GD with weight decay in
overparameterized fixed-feature linear regression. Their Theorem 4.2 assumes,
among other conditions, a realizable teacher, sufficiently large `n`,
`m>n`, Gaussian initialization `theta(0) ~ N(0, nu^2 I_m)`, sufficiently small
weight decay, and a stable step size. They decompose the parameters into the
empirical data span and its orthogonal complement. The latter obeys the exact
recursion

```math
\theta_\perp(t)=(1-\eta\lambda)^t\theta_\perp(0).
```

The empirical-span component fits on a curvature-controlled timescale, while
the randomly initialized orthogonal component can dominate population error
until weight decay removes it. Their quantitative late-generalization bound is
of order

```math
t_2 \;\gtrsim\;
\frac{1}{4\eta\lambda}
\log\!\left[
\frac{(m-n)\nu^2}{2}
\left(\sqrt{c/\lambda_{\min}(\Sigma)}+\|\theta^*\|\right)^{-2}
\right],
```

subject to the theorem's stated constants and probability conditions. Thus the
delay scales as `1/(eta*lambda)` and only logarithmically in the nuisance
initialization energy. The paper also stresses that smooth regression loss does
not itself require a visible flat plateau: thresholding into accuracy can make
smooth generalization look abrupt.

Sources: [official arXiv abstract and versions](https://arxiv.org/abs/2601.19791),
[official paper PDF](https://arxiv.org/pdf/2601.19791).

### Applicability table

| Structural quantity | Paper theorem | Actual Round 2 | Consequence |
|---|---:|---:|---|
| feature dimension | `m` | `31` | measured from registered chart |
| training rows per output coordinate | `n` | `1,474,560` | `480` states × `3,072` sampled spatial positions |
| independent output coordinates | — | `3` | the same `X` defines three separate ridge regressions |
| overparameterization | `m>n` | `m-n=-1,474,529` | theorem domain fails |
| numerical rank | — | `28` | three chart redundancies under fp32 rank floor |
| initialization | `nu^2>0` Gaussian | `W_0=0`, `nu^2=0` | paper slow amplitude is exactly zero |
| realizable teacher | assumed | unproved and empirically contradicted by heldout wall | theorem domain fails |
| ridge `lambda` | sufficiently small | `3.2247040271759033` | deliberately strong spectral ridge |
| step size `eta` | stable/small-product regime | `0.20673732459545135` | `eta*lambda≈2/3`; small-product simplification fails |
| fixed-feature convexity | yes | yes | exact quadratic terminality law does transfer |

**Computed paper-bound verdict:** Equation 7 is **NOT DEFINED FOR THIS
INSTANCE**, rather than “large” or “small”: its overparameterization term uses
`m-n`, which is negative here, and its nuisance initialization energy is zero.
Substituting those values into the logarithm would be fake mathematics. This is
a theorem-domain refusal, not a failure of the theorem.

**Computed actual-delay verdict:** the relevant slow-component amplitude is
zero for every step. Even under a counterfactual nonzero amplitude, its 15-step
retention is `6.9691668e-08` (more than `1e7` shrink), with a half-life of
`0.6309297251` steps. The fixed objective should have generalized within 15
steps if the information were present in this linear chart at the chosen
ridge—and the exact ridge-ladder control shows it is not merely hidden by that
ridge choice.

### `$0` refit receipt

Trackable compact receipt:
`.omx/research/grokking_ridge_round2_refit_receipt_20260713.json`.
Durable receipt:
`experiments/results/grokking_ridge_round2_refit_20260713/measurement_receipt.json`
(SHA-256
`fc8c79ef82d829f05cee79890c9b5d237e12d84e92ec83982f182de15ecb6b4d`).
Run directory:
`experiments/results/grokking_ridge_round2_refit_20260713/`.

The probe reused all `480` content-addressed training sufficient-statistic
labels and made `120` new real heldout frozen-CPU-SegNet teacher calls. No
synthetic labels were used. The committed source receipt hash was
`067ce197d30fa9e2c7c4bda48ac671af550e0a00f126289ba5b30946d44fc4b1`
both before and after the run, so the source run was not mutated. A resume
replay returned success without repeating any completed teacher call and left
the final receipt hash unchanged.

| Candidate | Steps | `lambda/lambda_max(A)` | Heldout cosine | Heldout rel-L2 |
|---|---:|---:|---:|---:|
| committed/reproduced GD | 15 | `1` | `0.001415793418` | `1.000001870578` |
| long GD control | 150 | `1` | `0.001415793464` | `1.000001870577` |
| exact ridge optimum | 0 | `0` | `0.005952922309` | `1.000772054350` |
| exact ridge optimum, best cosine | 0 | `1e-6` | `0.007690592650` | `1.000758608275` |
| exact ridge optimum | 0 | `1e-4` | `0.005802692855` | `1.000658280812` |
| exact ridge optimum | 0 | `1e-2` | `0.002832259422` | `1.000187949791` |
| exact ridge optimum | 0 | `1e-1` | `0.001987580976` | `1.000059358047` |
| exact ridge optimum | 0 | `1` | `0.001415793454` | `1.000001870578` |
| exact ridge optimum, best rel-L2 | 0 | `10` | `0.001150614702` | `0.999999528140` |

The unregularized and weak-ridge heads have slightly larger directional
cosine, but their relative L2 remains worse than the essentially zero predictor.
This is not evidence for a useful costate surrogate.

### Autopsy verdict and scope

`UNDERTRAINED` is **FALSIFIED** for the registered fixed objective.
`FEATURE-POVERTY` is **CONFIRMED** only for the fixed 31-feature linear chart,
the registered replay distribution/targets, and the measured ridge ladder. It
does **not** kill frozen-stem features, tileable local pre-SE features, RFF
lifts, margin-field targets, nonlinear heads, on-policy replay, or richer
representations. Any such family receives a fresh terminality certificate.

## 2. Plateau semantics for stage advance

### Structural pieces that transfer

One mechanism transfers algebraically to any update whose weight decay is
actually decoupled and whose local parameter subspace receives no data force:

```math
P_{\mathcal N}\theta_{t+1}
=(1-\eta_t\lambda_t)P_{\mathcal N}\theta_t,
\qquad
\mathcal T_\lambda(t)
=-\sum_{s<t}\log|1-\eta_s\lambda_s|.
```

If—and only if—a fixed-stage measurement establishes a stable projector
`P_N`, a nuisance amplitude `a_0=||P_N theta_0||`, and a
heldout/evaluator-calibrated harmlessness threshold `a_*`, then a remaining
decay-time telemetry estimate closes:

```math
\Delta\mathcal T_{\lambda,\mathrm{remaining}}
=\max\{0,\log(a_0/a_*)-\mathcal T_\lambda(t)\}.
```

This can be logged alongside event #315, the #344 NCDE hit detector, and the
intrinsic-time clock. It is a mechanistic covariate, not a stage transition.

### Structural pieces that do not transfer

The witness trainer is a nonconvex level-set system with changing stage losses,
changing loss weights, evaluator-through-`R` discontinuities, EMA state,
adaptive optimizers/Muon, and features/Jacobians that move with the parameters.
Its plateau can be a saddle-to-saddle transition, a class-birth barrier, a
rounding/receiver barrier, an optimizer-preconditioner effect, or genuine
convergence. There is no fixed empirical span, fixed population covariance,
realizable teacher, or theorem-certified generalization error corresponding to
the ridge proof. Standard coupled weight decay, AdamW's decoupled shrink, Muon,
and EMA also do not share one scalar `(1-eta*lambda)` dynamics without an
optimizer-specific derivation.

**Plateau-criterion verdict: NO-GO as stage-advance authority.** The theorem
does not distinguish “delayed witness generalization in progress” from
“converged witness stage.” A decay clock is admissible as `TELEMETRY_ONLY`
after the projector/amplitude/threshold measurements above; it may not fire or
veto #315/#344 on its own. Reactivate this path only with a fixed-stage local
linearization and an evaluator-calibrated heldout threshold. This negative is
scoped to theorem-to-witness transfer, not to every future local spectral
diagnostic.

## 3. Steps dimension and SPS economics

**Measured witness-stage steps verdict: NOT MEASURED; no shortening claim.**
The `$0` experiment is a surrogate-head autopsy, not witness training. On that
head, delay-eliminating tuning cannot save useful teacher calls because the
15-step fit is already terminal and 150 steps do not improve heldout fidelity.

For a future witness-stage intervention, let `r_grok=N_0/N_grok` be the
measured stage-step speedup and let the SPS memo use teacher fraction `f_T`,
teacher-cost multiplier `k_T`, and witness/update multiplier `k_W`. If the
ridge-derived guard adds no per-step scorer work, its standalone break-even is
simply

```math
r_{grok}>1.
```

Composed with SPS, the conditional wall-clock law is

```math
\frac{\mathrm{Wall}_{grok+SPS}}{\mathrm{Wall}_0}
=\frac{k_T f_T+k_W(1-f_T)}{r_{grok}r_{SPS}},
\qquad
r_{grok}r_{SPS}>k_T f_T+k_W(1-f_T).
```

Under the SPS memo's explicitly **ASSUMED**, not measured,
`f_T=0.95, k_T=1, k_W=2`, the product must exceed `1.05`—at least `4.7619%`
fewer total steps at unchanged per-step composition. If SPS requires a second
scorer VJP, the right-hand side approaches `1.95–2.00`. Since the present-wall
teacher fraction and witness-stage `r_grok` are not measured, no numeric
teacher-call saving is admissible. When measured, teacher-call savings are
`Delta C_T = c_T N_0 (1-1/r_grok)` only if calls per step remain constant.

No witness lever falls out of this audit, so **no curriculum pool row is
registered**. Admitting one would promote a convex-head guard into an unmeasured
nonconvex actuator.

## Canonical law, DAG feed, and triality

The closed law is registered as
`grokking_ridge_undertraining_disambiguation_v1` and implemented in
`src/tac/canonical_equations/grokking_ridge_undertraining_disambiguation_20260713.py`.
Its anchor is the real-heldout receipt above. The law is deliberately scoped to
fixed deterministic quadratic heads:

```math
W_t-W^*=(I-\eta H)^t(W_0-W^*),\quad
H=X^TX/n+\lambda I;
```

```math
P_{\ker X}W_t=(1-\eta\lambda)^tP_{\ker X}W_0,
\quad
P_{\ker X}W_0=0\Rightarrow P_{\ker X}W_t=0,
\quad
\|W_t-W^*\|_F\le\|\nabla F(W_t)\|_F/\mu.
```

Triality:

- **DAG:** `.omx/research/grokking_ridge_bounds_DAG_FEED_20260713.md`.
- **Equation:** the canonical module and locked-registry row named above.
- **Receipt:** `.omx/research/grokking_ridge_round2_refit_receipt_20260713.json`
  points at the full resumable-run measurement receipt and hashes.
- **DSL:** `N/A` with rationale—this is a research-only Round-3 admission guard,
  not a live trainer flag or stage actuator.
- **Pool:** no row; no measured witness lever emerged.
- **Probe outcome:** canonical locked row
  `grokking_ridge_round2_fixed_chart_autopsy_20260713` records formulation-scoped
  `KILL`, preventing the settled 31-feature arm from being re-dispatched while
  leaving every richer-feature family open.

## Claim ledger

- **MEASURED:** Round-2 geometry, rank, ridge/step size, terminal gradient,
  15-vs-150 weights and real-heldout fidelity, exact ridge-ladder optima,
  120 teacher calls, receipt hashes, and source-run immutability.
- **DERIVED:** exact null-mode retention, contraction factor, terminal parameter
  bound, Round-3 terminality protocol, and conditional SPS composition law.
- **INFERRED:** richer features are the required next Round-3 direction after
  optimizer delay and ridge-choice confounds fail to explain the wall.
- **ASSUMED:** the SPS economics example (`f_T=0.95`, `k_T=1`, `k_W=2`), copied
  as a conditional from its memo; it is not current-wall measurement.
- **REFUSED:** direct substitution into paper Equation 7 and direct witness
  plateau/stage-advance authority.

## Stores consulted and artifact custody

`CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`;
`PROGRAM.md`; v7.5/v8 canonical specs; current directive files;
latest Codex findings/session and current design/council memos;
`reports/latest.md`; lane/subagent/canonical state; the committed Round-2 memo,
module, run contract, sufficient statistics, and receipt; the current SPS dig;
and arXiv:2601.19791v3.

Probe implementation:
`tools/probe_grokking_ridge_round2.py`. Durable run:
`experiments/results/grokking_ridge_round2_refit_20260713/` (`548 KiB`, `124`
files at completion). An initial non-finite zero-ridge pseudoinverse attempt
failed before writing a measurement record and was preserved, not erased, at
`experiments/results/grokking_ridge_round2_refit_20260713_FAILED_nonfinite_pinv/`;
the repaired run uses an explicit binary32 rank threshold and finite checks.
No provider, GPU, evaluator, archive, frontier pointer, or live run was touched.

## Pointer-delta honesty

Frontier pointer: **unchanged**. Score: **not measured**. Promotion authority:
**none**. The durable movement is a real-heldout Round-2 falsification receipt,
a Round-3 terminality guard, a canonical fixed-quadratic equation, and a scoped
NO-GO on using ridge-grokking time as a witness-stage trigger.
