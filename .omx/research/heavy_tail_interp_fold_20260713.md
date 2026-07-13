# Heavy-tail interpolator reliability fold — Zhu & Lu (arXiv:2607.09547)

**Date:** 2026-07-13  
**Mode:** `THEORY FOLD = MEANS`; `research_only=true`; design/analysis plus cached-read-only
diagnostics  
**Authority:** paper theorem + local artifact audit; no training, scorer forward, paid dispatch,
live-run mutation, archive mutation, evaluator run, or pointer mutation  
**Score authority:** `NONE`. Only a receiver-closed, byte-closed exact row on the exact archive can
move a contest pointer.  
**Numerical authority:** any future operational backtest must use the deterministic NumPy-fp32
reference for inference/decision parity. Existing float64 convex certificates are read-only
optimization evidence, not permission to promote a different numerical path.  
**Pointer delta:** `NONE`  
**Number labels:** `SOURCE`, `MEASURED`, `DERIVED`, `INFERRED`, `ASSUMED`  
**Paper:** Youheng Zhu and Yiping Lu, *High-Dimensional Interpolators Can Be Fragile: Heavy
Tails and High-Dimensional Large Deviations*, [arXiv abstract](https://arxiv.org/abs/2607.09547),
[full paper v1](https://arxiv.org/pdf/2607.09547).

## Executive answer — lead with the honest MEANS caveat

The paper does **not** move the pointer and its `n log n` versus `n^2` rates do **not** apply
literally to any audited Pact interpolator. The clean fold is nevertheless operationally useful:
an interpolator that drives a verdict or control should be selected by downstream tail loss as
well as average fit, and a fixed positive ridge is the fail-closed default over an unregularized
inverse when the two have comparable mean performance.

| Surface | Scoped one-line verdict | Concrete action |
|---|---|---|
| Paper-to-Pact rate transfer | **NO-GO-regime-mismatch** — `FORMULATION x LITERAL-RATE-TRANSFER`; req-R evidence: non-i.i.d. clustered designs, finite/non-proportional samples, different loss, and an explicit MP spectral cutoff | Do not write `exp(-Theta(n log n))` or `exp(-Theta(n^2))` on a Pact result without a new theorem-compatible experiment |
| `pre_se` convex rung | **REPORT-tail-quantile + ADOPT-ridge-default** — `FORMULATION x FIXED-REPLAY x HELDOUT-RELIABILITY` | Keep MP as an optimization/capacity certificate; use a preregistered ridge rung as the load-bearing reliability row and report lower retained-mass quantiles across costates **and** refit splits/seeds |
| Costate organ `A_ridge_solve` | **ADOPT-ridge-default + TUNE-lambda-for-tail** — `CONTROL-DRIVING-INTERPOLATOR x CACHED-TRAJECTORY` | Retain fixed-positive ridge over any ridgeless solve; select its strength on downstream control-regret tail subject to a mean/bias constraint, not MSE alone |
| `#433` per-class lambda | **TUNE-lambda-for-tail** — `INSTANCE x ONE-VEHICLE x NINE-INTERVAL-TRAJECTORY`; the `-18.46%` mean gain is tail-consistent but **tail-unconfirmed** | A cached-only child backtest exists; the present seven fold errors do not show a tail win |
| Confound immune system | **REINFORCE-confound-story** — conceptual fold only | Any estimator feeding an L1 alarm, L3 verdict, gradient, or controller gets a tail field; no rebuild is warranted |

## 1. What the paper actually proves

### Model and regime

**SOURCE-PAPER.** The base model is

```text
y = X beta + epsilon,
X in R^(n x p),  epsilon ~ N(0, I_n),
X_ij iid with E[X_ij]=0 and E[X_ij^2]=1,
p/n -> gamma in (0, infinity) \ {1}.
```

The main min-norm risk calculation uses a dense isotropic random-effects coefficient,
`E[beta beta^T] = alpha^2 I_p / p`, and population covariates with identity covariance. The risk
is conditional on the realized design `X` and averages over `beta`, training noise, and a fresh
test covariate. In the overparameterized case `gamma>1`, Moore-Penrose minimum-norm least squares
interpolates continuous-design data almost surely. The risk decomposes as

```text
R_n = b_n + H_n,
H_n = (1/n) sum_i lambda_i,n^(-1),
```

where the sum is over the positive eigenvalues of the appropriate normalized sample covariance.
The singular inverse moment is the load-bearing object.

**SOURCE-PAPER, Assumption B.** The general-entry right-tail theorem requires i.i.d. entries with:

1. mean zero and variance one;
2. a bounded continuous density;
3. density bounded below in a neighborhood of zero; and
4. a dimension-free log-Sobolev inequality.

Gaussian entries are Assumption A. Discrete or locally degenerate designs are outside the stated
right-tail proof. `gamma=1` is excluded; the theorem keeps a nonzero proportional distance from the
interpolation threshold.

### Rates, with their exact scope

**SOURCE-PAPER, Theorem 1 (fixed dimension).** With `N>=q` fixed, `r=N-q`, and tail threshold
`x -> infinity`, the ridgeless inverse-trace risk obeys

```text
c x^(-(r+1)/2) <= P(R > b + x) <= C x^(-(r+1)/2).
```

This is the literal polynomial heavy tail. Its mechanism is one column approaching the span of
the others, equivalently one eigenvalue approaching the hard edge.

**SOURCE-PAPER, Theorem 2 (proportional dimension).** For every fixed `delta>0`, under Assumption B,

```text
lim_(n->infinity) [1/(n log n)] log P(R_n > r_star + delta)
  = -|1-gamma|/2.
```

Thus the fixed right-tail event has probability
`exp(-(|1-gamma|/2 + o(1)) n log n)`. The exponent is independent of the fixed threshold offset
`delta`; one positive eigenvalue at microscopic scale `1/n` is the cheapest event.

**SOURCE-PAPER, Theorem 4 and Proposition 9 (ridge).** For every **fixed** `lambda>0`, the Gaussian
ridge risk is a bounded continuous spectral functional and has an `n^2`-speed LDP. Under the more
general normalized/log-Sobolev entries, the paper proves the right-tail upper bound

```text
P(R_n,lambda > r_lambda,star + delta) <= exp(-c_lambda,delta n^2).
```

The general-entry result is an upper bound, not a matched universal rate formula. The constant
depends on `lambda` and `delta`.

**SOURCE-PAPER, Theorem 6.** In the overparameterized regime `p/n->gamma>1`, every exact linear
interpolating algorithm inherits an inverse-trace lower bound from fitting training noise in the
row space. The `n log n` right-tail large-deviation mechanism is therefore not merely an artifact of choosing
minimum Euclidean norm, although the matching upper bound remains min-norm-specific.

### What the theorem does not say

- **DERIVED:** It does not identify the best positive ridge strength. Every fixed positive
  `lambda` removes the hard-edge singularity, while bias and the rate constant still depend on
  `lambda`. “Tune lambda for the `n^2` rate” means tune the finite-threshold rate/decision tail
  *within the fixed-positive family*; it does not mean maximize `lambda`.
- **DERIVED:** It does not cover `lambda=lambda_n -> 0`; that limit can restore hard-edge
  sensitivity.
- **DERIVED:** Its probability is over random fitted designs. A lower quantile across examples
  under one already-fitted model is a different random object.
- **DERIVED:** It is a squared prediction-risk theorem. Retained costate mass, marginal-Delta-S
  forecast error, and downstream controller regret require an explicit loss map before transfer.
- **DERIVED:** Rank-truncated Moore-Penrose is spectral regularization. By discarding the smallest
  numerical eigendirections, the Pact solver removes part of the exact paper mechanism; it is not
  theorem-identical to an all-positive-eigenvalue ridgeless inverse.

## 2. Regime audit: does the theorem apply to our interpolators?

### `pre_se` RankRLS

**MEASURED-CODE/ARTIFACT.** The sealed `pre_se_locus_20260713` fit uses `480` training states and
`120` untouched heldout states from an `n600` exact-custody campaign. Its two feature widths are
`d=188` and `d=332`. Treating one costate state as one independent cluster gives the **DERIVED
proxy** ratios

```text
d/n_train = 188/480 = 0.3917; 332/480 = 0.6917.
d/n_campaign = 188/600 = 0.3133; 332/600 = 0.5533.
n_train/d = 480/188 = 2.5532; 480/332 = 1.4458.
n_campaign/d = 600/188 = 3.1915; 600/332 = 1.8072.
```

Those are only bookkeeping ratios. The actual RankRLS normal equations aggregate many correlated
pixel-pair rows within each state, split into 20 highly unequal class-pair heads. Active-state
counts range from **MEASURED `0` to `480`**, so there is no single honest `gamma`. All **MEASURED
`40/40`** certificates are numerically rank-deficient relative to widths `188/332`; the solver
hard-truncates eigendirections below `eps * width * lambda_max`. The frozen feature rows are neither
i.i.d. entries nor isotropic light-tailed designs.

**Verdict:** direct theorem applicability **N**. The hard-edge warning is **INFERRED-suggestive**,
not a licensed `n log n` rate claim.

### Costate organ and `#433`

**MEASURED-CODE.** `A_ridge_solve` has a design width
`p = 1 + STATE_DIM + PHI_DIM = 17`, fixed code-default `ridge=1e-2`, and walk-forward folds with
only `n_train=2,...,8` sequential intervals from one vehicle. The proxy ratios are therefore
`p/n_train=8.5,...,2.125`, equivalently `n_train/p=0.1176,...,0.4706`: overparameterized, but
tiny, time-correlated, heteroskedastic, and far
from a proportional random-design limit. `#433` uses the same one-trajectory geometry.

**Verdict:** direct theorem applicability **N**. The control-amplification framing is a strong
**INFERRED** transfer; the exact `n^2` claim is not.

## 3. Fold 1 — `pre_se` retained mass and the convex rung

### Current-state correction

**MEASURED-CANONICAL-ARTIFACT.** The completed protected receipt has 20 pair-specific RankRLS
heads solved by rank-truncated Moore-Penrose and explicitly says `no regularization or width
sweep`. It does **not** contain a Tikhonov/ridge comparison. The separate `pre_se_reopen_a` sibling
is in flight and, at this audit point, has not emitted a ridge receipt. Therefore every ridge
statement below is a harvest recommendation, not a claim about a measured current rung.

### What a cached lower-tail read says — and does not say

The sealed headline is a mass-weighted aggregate over the untouched heldout set. A read-only,
deterministic NumPy-fp32 reaggregation of the already-emitted `120` heldout state JSONs gives the
following; `lower q10` uses the conservative observed order statistic (`method=lower`):

| MP locus | Aggregate retained mass | Per-state mean | Lower q10 | Lower-tail mean, worst 12/120 | Minimum |
|---|---:|---:|---:|---:|---:|
| block2 PRE-SE | **MEASURED-CACHED `0.202330`** | **MEASURED-CACHED `0.207222`** | **MEASURED-CACHED `0.164945`** | **MEASURED-CACHED `0.145802`** | **MEASURED-CACHED `0.055684`** |
| block3 PRE-SE | **MEASURED-CACHED `0.093147`** | **MEASURED-CACHED `0.099876`** | **MEASURED-CACHED `0.046867`** | **MEASURED-CACHED `0.038012`** | **MEASURED-CACHED `0.021137`** |

This is a **heldout `n=120` diagnostic nested inside an exact-custody `n600` campaign**, not an
`n600` tail sample and not an estimator-level large-deviation estimate. It shows that the failure
at the `0.47` bar is not carried by only one or two low-mass costates. It does **not** answer whether
this fitted MP head was itself an unlucky draw: there is only one fixed train split and no convex
refit-seed distribution.

### Required harvest report

For retained mass `M` (higher is better), map the paper's upper risk tail to shortfall
`L=(m_bar-M)_+`. Report both levels:

1. **Within-fit costate tail:** aggregate mass-weighted `M`, arithmetic per-state mean, lower
   quantiles `Q_0.10(M)` and `Q_0.05(M)`, lower-tail conditional mean, and checkpoint/class-pair
   strata. This detects rare deployment costates.
2. **Across-fit estimator tail:** for each preregistered split/refit seed `s`, compute the heldout
   mass-weighted `M_s`, then report the lower quantile and lower-tail mean across `s`. This is the
   closer analogue of randomness over fitted estimators in the paper. Reusing the same `n600`
   pool is a finite cross-fit diagnostic, not independent asymptotics; do not fit a Hill exponent
   or claim `n log n` from a handful of seeds.
3. **Spectral context:** report retained numerical rank, minimum retained eigenvalue, cutoff,
   condition number, and discarded-RHS fraction per head. A near-cutoff head is a stress-test
   priority, not proof of the Zhu-Lu law.

**DERIVED recommendation.** Prefer a predeclared fixed-positive RankRLS ridge rung over MP as the
load-bearing *reliability* row when it satisfies the same mean gate and has no worse lower-tail
shortfall. Preserve MP as the exact optimum/capacity certificate for the declared rank-truncated
objective. The two answer different questions; ridge must not replace the MP certificate, and MP
must not alone certify reliable deployment.

**Negative-scope honesty.** The current fixed-replay negative remains valid: both MP aggregates
are far below `0.47`, and the independent tileability gate also fails. The paper neither rescues
nor broadens that verdict. It blocks promoting one fixed MP fit into a split-distribution or
interpolator-family reliability claim.

**One-line verdict:** **REPORT-tail-quantile + ADOPT-ridge-default**, `verdict_scope = FORMULATION x
FIXED-REPLAY x HELDOUT-RELIABILITY`; literal rate transfer is **NO-GO-regime-mismatch**.

## 4. Fold 2 — costate organ `A_ridge_solve`

### Is ridge selected for the right reason today?

**MEASURED-CODE/LEDGER.** The architecture has always used a fixed positive ridge solve
(`ridge=1e-2`, scaled by the mean Gram diagonal), so it already removes a singular normal-equation
inverse. But the current organ summary recommends `A_ridge_solve` because no incumbent has a
passing accumulated record: the literal basis is `no passing records yet — incumbent default
(solve)`. Tail risk is not currently the selection objective.

**MEASURED-CACHED.** On the one-vehicle `#433` walk-forward record, `A_ridge_solve` loses average
forecast MAE to persistence (`0.003902` versus `0.002792`). Thus “ridge default” cannot override
the existing meta-lambda persistence fallback. Its correct role is the default *parametric solve
inside the interpolator family*, not unconditional control authority.

### Derived control-tail law

Let `g_hat_lambda(D)` be the estimated marginal-Delta-S field, let `pi` map that field to a control
decision, and let

```text
Z_lambda(D, xi)
  = [ S(next_state under pi(g_hat_lambda); xi)
      - S(next_state under oracle action; xi) ]_+
```

be downstream one-step control regret. `xi` indexes heldout regimes/costates. The right selection
is

```text
lambda_tail in argmin_(lambda in Lambda_fixed_positive)
  empirical_CVaR_alpha(Z_lambda)

subject to
  mean(Z_lambda) <= best_mean + epsilon_mean,
  all custody / walk-forward / action-stability gates pass.
```

`alpha` and `epsilon_mean` must come from the controller's accepted catastrophic-error budget and
value-provenance ladder; they are not guessed in this fold. In the theorem-compatible asymptotic
case, the equivalent lens is to maximize the finite-threshold large-deviation rate constant among
fixed-positive lambdas. In Pact's finite correlated case, empirical downstream CVaR/chance loss is
the honest quantity.

**DERIVED recommendation.** `A_ridge_solve` should remain the default over any ridgeless normal-
equation variant because its output drives curriculum/control decisions and a singular inverse can
turn one design degeneracy into a disproportionate action. Choose its ridge strength on nested
past-only control-regret tails, while keeping persistence as the fail-closed action when the organ's
meta-lambda says the model is losing.

**One-line verdict:** **ADOPT-ridge-default + TUNE-lambda-for-tail**, `verdict_scope = CONTROL-
DRIVING-INTERPOLATOR x CACHED-TRAJECTORY`; literal `n^2` attribution remains
**NO-GO-regime-mismatch**.

## 5. Fold 3 — `#433` prior-mean per-class lambda

### What is measured

**MEASURED-CACHED, `[macOS advisory] NON-PROMOTABLE`, score_claim=false.** On seven walk-forward
folds from one nine-interval trajectory:

| Arm | Mean WF MAE | Delta vs `A_ridge_solve` | Early-fold MAE | Maximum fold error | Worst-two fold mean |
|---|---:|---:|---:|---:|---:|
| `A_ridge_solve` | `0.003902384` | baseline | `0.005200781` | `0.006107693` | `0.005804615` |
| `P_priormean_aniso` | `0.003182024` | **`-18.4595%`** | `0.004265427` | `0.006727353` | `0.005811968` |
| `Q_priormean_iso` | `0.003066559` | **`-21.4183%`** | `0.004023521` | `0.006859039` | `0.005890645` |

The anisotropic direction remains neutral at this `n`: `Q` is better than `P` on mean. More
importantly for this paper fold, both prior-mean arms improve the mean and middle folds but do not
improve the observed maximum; their worst-two mean is tied-to-worse than plain ridge.

### Mechanism verdict

**INFERRED-CONSISTENT:** yes, shrinkage toward a physics prior is the right *kind* of mechanism for
removing unstable null-space freedom and reducing sensitivity to an ill-conditioned solve.

**MEASURED-TAIL-CONFIRMED:** no. The `-18.46%` number is an average walk-forward improvement, not a
tail measurement, and the cached worst-fold diagnostics do not show tail suppression. With seven
folds, no rare-event exponent is estimable. This is req-R evidence for the scoped negative: two
prior-mean formulations (`P`, `Q`) and two tail summaries (maximum, worst-two mean) fail to beat
`A` on this instance.

### Clean `$0` successor backtest — proposed, not run

Use only the sealed trajectory/cached physics tensors; do not run a scorer, trainer, live process,
or the `#433` tournament again. A new standalone read-only child analysis should:

1. freeze an **ASSUMED-design** log grid around the current `1e-2` ridge, for example
   `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1}` plus an MP diagnostic at zero;
2. evaluate `A`, `P`, and `Q` on identical past-only folds and record score-law-weighted forecast
   error **and** replayed control regret;
3. report every lambda's mean, maximum, worst-two mean, and action-flip count—no post-hoc single
   winner from the same seven folds;
4. select no default until at least the existing reactivation condition of three organ records
   includes a transient-rich regime; then use nested record-level selection and a preregistered
   tail budget;
5. retain the current persistence decision whenever every parametric arm violates the mean or
   tail gate.

The reaggregation in the table above is already a clean `$0` cached diagnostic and its result is
negative for current tail improvement. The lambda sweep is **more-available Y as a diagnostic**,
not a claimed gain and not promotion evidence.

**One-line verdict:** **TUNE-lambda-for-tail**, `verdict_scope = INSTANCE x ONE-VEHICLE x NINE-
INTERVAL-TRAJECTORY`; `-18.46%` is tail-consistent **Y**, tail-confirmed **N**, more-available
**Y (cached-only, underpowered until record accrual)**.

## 6. Fold 4 — confound immune system

**INFERRED conceptual reinforcement, not a rebuild.** Spike guards, median-freeze, satisficing,
L1 alarms, and L3 verdict clearance already encode the fact that a rare measurement error can
corrupt a control or verdict far more than its contribution to average error suggests. Zhu and Lu
add a specific warning: when that measurement is produced by an exact or near-ridgeless linear
interpolator, a hard-edge direction can make rare severe errors qualitatively cheaper than under a
fixed-positive ridge. Therefore every interpolator feeding a verdict, gradient, or controller should
carry a tail field and a spectral-stability field. This does not prove that Pact's existing spikes
come from the theorem's random-matrix mechanism; the confounds are structured, temporal, and often
deterministic.

**One-line verdict:** **REINFORCE-confound-story**, `verdict_scope = CONCEPTUAL-MECHANISM`; no code,
gate, or live-arm mutation is authorized by this fold.

## 7. Canonical law, triality, and wire-in

The clean law is:

> **Control-driving interpolators select fixed-positive regularization by downstream tail regret,
> subject to an average-risk/bias budget; MSE alone is not the authority.**

- **DSL leg:** `DEFERRED_MAIN`. This fold changes no live configuration and invents no flag. Any
  adoption must add a typed tail criterion and provenance-owned thresholds at a stage boundary.
- **Equation leg:** `.omx/research/heavy_tail_interp_fold_equation_feed_20260713.md`.
- **DAG leg:** `.omx/research/heavy_tail_interp_fold_DAG_FEED_20260713.md`; shared canonical DAG
  append is `DEFERRED_MAIN` because the canonical surface is sibling-held/dirty.
- **Sensitivity-map contribution:** per-head minimum retained eigenvalue/rank/cutoff; per-costate
  retained-mass shortfall; per-regime marginal-Delta-S control regret.
- **Pareto constraint:** tail regret x mean regret x action stability. Archive bytes and exact score
  remain unmeasured/non-binding.
- **Bit allocator:** non-binding; no archive or payload is produced.
- **Cathedral/autopilot:** `REFUSE` actuation. A future consumer may prefer ridge within an
  interpolator family only after the typed tail/mean gates pass.
- **Continual learning:** this memo plus isolated equation/DAG feeds; no hot shared ledger append.
- **Probe disambiguator:** ridge versus MP and MSE versus tail-CVaR are both retained as explicit
  interpretations. Cached evidence arbitrates the present instance; no intuition-only collapse.
- **Numerical parity:** a future operational consumer must produce its predictions/actions through
  the deterministic NumPy-fp32 reference and verify any accelerated path against it.

## 8. Value-provenance ladder and stores consulted

- **SOURCE:** Zhu-Lu model, assumptions, Theorems 1/2/4/6, and Proposition 9.
- **MEASURED:** current code defaults, feature widths, train/heldout counts, certificates, receipt
  aggregates, seven stored `#433` fold errors, organ recommendation basis.
- **DERIVED:** dimension ratios, cached quantiles/reaggregations, relative improvements, shortfall
  orientation, and the downstream tail-regret selection law.
- **INFERRED:** relevance of hard-edge fragility to Pact reliability and the physics-prior
  shrinkage mechanism.
- **ASSUMED:** only the proposed diagnostic lambda grid; it has no adoption authority.

**STORES CONSULTED:** `CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; latest
design/council/Codex memos and 2026-07-13 directives; `reports/latest.md`; canonical lane/task/
subagent surfaces; `.omx/research/pre_se_locus_20260713.md`; the sealed pre-SE run contract,
certificates, heldout rows, and receipt; `.omx/research/aniso_perclass_lambda_433_20260711.md`;
`aniso_perclass_lambda_backtest_20260711T144317Z.json`; costate organ envelope/trajectory ledger;
`lambda_net.py`; `aniso_perclass_lambda.py`; and the full arXiv paper.
