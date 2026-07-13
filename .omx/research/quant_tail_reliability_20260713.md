# Quantitative tail reliability for control-driving interpolators — 2026-07-13

> **MEANS, not the frontier pointer.** This unit makes interpolator/solver reliability
> quantitative on cached Pact designs. It performs no training, scorer call, live-run
> mutation, paid/remote dispatch, archive mutation, or pointer mutation. Only a byte-closed
> exact evaluator row can move the score.

## Executive result

**MEASURED-CACHED:** the RankRLS tail curve exists on a real, exact-mass, train-only
60-state development partition with three inherited fit-resample seeds (180
costate-by-seed observations per locus). Block2 selects `lambda=1`; its retained-mass
shortfall p95/p99/CVaR95 changes `0.932984 / 0.953625 / 0.946703` to
`0.868733 / 0.885778 / 0.880559`. Block3 selects `lambda=0.3`; the corresponding
change is `0.959005 / 0.967821 / 0.966219` to
`0.919665 / 0.932909 / 0.929105`.

**NO-FAKE scope:** those are paired development-surface measurements, not an official
held-out-n120 ridge result. The sealed n120 receipts preserve MP selections, hashes, and
retained masses, but not the raw held-out feature/costate arrays needed to reconstruct a
new-lambda ranking. The official n120 surface therefore remains an MP-only anchor.

**DERIVED:** an explicit finite fixed-design correlated-Gaussian quadratic tail law closes
symbolically, with all constants shown below. It does **not** close numerically against the
measured retained-mass curve: the cache lacks residual innovations/covariance custody and
the discontinuous top-k receiver requires a measured boundary-margin envelope.

**MEASURED-CACHED organ proxy:** current `A_ridge_solve` at `lambda=0.01` is neither the
mean nor CVaR optimum on the 16-point diagnostic grid. Both objectives continue improving
to the upper boundary `lambda=1000`; therefore no finite optimum is bracketed. This is a
zero-response/suppression diagnostic, not authority to actuate `lambda=1000`. Keep the
incumbent persistence fallback until transient-rich folds close the bracket and provide
counterfactual downstream-control regret.

## Authority, custody, and method

- **Authority:** `[macOS-CPU advisory; NumPy-fp32 score/rank/selection authority;
  float64 eigensolve optimization evidence]`.
- **Receipt:** `.omx/research/quant_tail_reliability_receipt_20260713.json`,
  664,977 bytes, SHA-256
  `4a91e701895639c49c745de0e2a01e8739609f16a2f0604729529c31cae3be4d`.
  It contains every per-state/per-seed value and the complete lambda curves.
- **PRE-SE n600 custody:** 480 inherited training states + 120 untouched held-out states,
  from sealed receipt `660a5763831539715d8593df0ba40a0f50f660af93c0e5bcd1d399ea340d1abb`.
- **Curve split:** cached stage chunks retain exact-mass arrays for a preregistered
  420-core/60-train-only-dev partition. Each of seeds 455/456/457 performs a
  support/pair-stratified row bootstrap of the 420-core sufficient arrays, fits the
  imported RankRLS solver, and evaluates all 60 untouched-with-respect-to-that-refit real
  states. State-cluster bootstrap is unavailable because the core arrays did not preserve
  state boundaries.
- **Lambda grid:** `0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10,
  30, 100, 300, 1000`. Zero is the Moore-Penrose diagnostic. Positive-ridge solves use
  the existing scaled RankRLS formulation by import.
- **Loss:** `L=1-retained_mass`; `CVaR95` is the exact empirical upper-tail integral,
  retaining fractional boundary mass. With 180 observations it averages nine worst-tail
  observations. With seven organ folds it necessarily equals the observed worst fold.
- **Numerical cross-check:** maximum absolute score drift between the authoritative
  NumPy-fp32 score path and the float64-eigensolve shadow is `.00172033` for block2 and
  `.000448643` for block3. Rankings and all decisions use the fp32 path; no parity claim
  beyond these measured maxima is made.
- **Containment:** all inputs were read-only cached files. No running PRE-SE, round-5,
  heavy-tail, VRGHAL, trainer, witness-control, curriculum, or v9 run file was edited.

### Official n120 MP anchor (not the ridge curve)

| locus | n | retained mean | median | q10 | q05 | q01 | worst | shortfall p95 | p99 | CVaR95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| block2 | 120 | .207222 | .206556 | .167960 | .155007 | .126348 | .055684 | .844993 | .873652 | .869518 |
| block3 | 120 | .099876 | .098030 | .049723 | .040945 | .030260 | .021137 | .959055 | .969740 | .968032 |

`verdict_scope=MEASUREMENT x OFFICIAL-N120 x MP-ONLY`: **NO-GO for an official-n120
lambda comparison from present cache.** `req-R`: preserve raw held-out feature rows,
exact costate-mass arrays, state boundaries, and the sealed split, then run the same
NumPy-fp32 lambda grid without touching training or choosing lambda on n120.

## 1. MEASURED lambda-for-tail curves

The table reports beneficial retained-mass mean and the harmful shortfall tail. Thus
shortfall p95/p99 are exactly the high-severity counterparts of retained-mass q05/q01.

| lambda | B2 retained mean | B2 p95 | B2 p99 | B2 CVaR95 | B3 retained mean | B3 p95 | B3 p99 | B3 CVaR95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .161348 | .932984 | .953625 | .946703 | .113792 | .959005 | .967821 | .966219 |
| .0001 | .096487 | .966303 | .977182 | .974123 | .109878 | .943113 | .956744 | .951485 |
| .0003 | .093574 | .966205 | .978450 | .974139 | .131115 | .929856 | .945256 | .939643 |
| .001 | .113732 | .947378 | .957347 | .953884 | .123300 | .942246 | .953055 | .949661 |
| .003 | .144201 | .912830 | .926335 | .919864 | .131187 | .939912 | .953444 | .949934 |
| .01 | .157206 | .908428 | .925166 | .920132 | .133775 | .939316 | .958588 | .952516 |
| .03 | .162820 | .905861 | .921268 | .916628 | .135768 | .932883 | .956382 | .947082 |
| .1 | .173671 | .898682 | .907448 | .905204 | .146447 | .924533 | .938736 | .934734 |
| .3 | .196327 | .880612 | .898095 | .893684 | **.138857** | **.919665** | **.932909** | **.929105** |
| 1 | **.214268** | **.868733** | **.885778** | **.880559** | .118290 | .926296 | .936911 | .932813 |
| 3 | .205009 | .873280 | .902913 | .891557 | .104477 | .936573 | .945504 | .942957 |
| 10 | .183097 | .884034 | .924477 | .908817 | .083246 | .948940 | .955393 | .953139 |
| 30 | .162181 | .906734 | .932026 | .921704 | .064846 | .961164 | .970133 | .967160 |
| 100 | .145328 | .915362 | .938130 | .930920 | .055102 | .971134 | .978182 | .976967 |
| 300 | .138823 | .919290 | .940051 | .934647 | .051396 | .977045 | .979978 | .978908 |
| 1000 | .136396 | .924620 | .940566 | .935032 | .049938 | .977430 | .980039 | .979134 |

The selected-lambda retained-mass summaries are:

| locus | lambda* | median | q10 | q05 | q01 | worst | shortfall p90 | p95 | p99 | worst | CVaR95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| block2 | 1 | .208581 | .154450 | .131267 | .114222 | .100088 | .845550 | .868733 | .885778 | .899912 | .880559 |
| block3 | .3 | .134172 | .097592 | .080335 | .067091 | .060013 | .902408 | .919665 | .932909 | .939987 | .929105 |

Both minima turn inside the grid, so the **diagnostic development brackets close**.
Block2 CVaR falls 6.99% relative to MP; block3 CVaR falls 3.84%. This is the requested
quantitative evidence for the heavy-tail paper's qualitative direction on Pact's actual
finite correlated design. It does not borrow the paper's asymptotic rate.

`verdict_scope=FORMULATION x CACHED-420/60-CROSS-FIT x RANKRLS`: **ADOPT the
positive-ridge tail direction and the lambda-selection discipline; do not adopt either
numeric lambda as an operational constant yet.** `req-R`: official n120 raw-array custody,
state-cluster uncertainty, and a receiver/control outcome that consumes the ranking.

## 2. DERIVED finite-sample non-asymptotic law

### 2.1 Fixed correlated design and explicit Gaussian constants

Condition on finite matrices `X` (fit design), `Z` (prediction design), positive
semidefinite loss weight `W`, and ridge scale `D`. Let

```
y = X beta + epsilon,       epsilon = Sigma^(1/2) u,       u ~ N(0,I)
G = X'X,                    R_lambda = (G + lambda D)^(-1)
s_lambda = Z R_lambda X' y
```

For prediction error against `Z beta`, define

```
b_lambda = Z(R_lambda G - I) beta
H_lambda = Z R_lambda X'
Q_lambda = (b_lambda + H_lambda epsilon)' W
           (b_lambda + H_lambda epsilon) / m.
```

The same formula bounds ridge-versus-MP score displacement by replacing
`b_lambda,H_lambda` with the deterministic and stochastic parts of
`s_lambda-s_0`, where `R_0=G^+`. Put

```
A = Sigma^(1/2) H' W H Sigma^(1/2) / m   (A positive semidefinite)
c = Sigma^(1/2) H' W b / m
E Q = b' W b / m + tr(A).
```

For every `delta in (0,1)`, with `t=log(1/delta)`, the Gaussian MGF gives the
finite, design-conditional bound

```
P[ Q - E Q >= 2 sqrt((||A||_F^2 + 2||c||_2^2)t)
                  + 2||A||_op t ] <= delta.                 (1)
```

There is no hidden asymptotic constant in (1). The coefficients `2,2,2` follow by
bounding the exact noncentral Gaussian quadratic MGF:

```
log E exp(theta(Q-EQ))
 <= theta^2 (||A||_F^2 + 2||c||_2^2) / (1-2 theta ||A||_op).
```

For independent, centered, unit-variance, `K`-sub-Gaussian whitened coordinates, the
same decomposition gives a Hanson-Wright bound with universal constants rather than
the explicit Gaussian constants:

```
P(|u'Au-Eu'Au| >= z)
 <= 2 exp[-c min(z^2/(K^4||A||_F^2), z/(K^2||A||_op))],
P(2|c'u| >= z) <= 2 exp[-c z^2/(K^2||c||_2^2)].             (2)
```

Equation (2) is honest only after whitening makes the coordinates independent; mere
finite-design row correlation is not enough.

### 2.2 Transfer to discontinuous retained mass

The top-k retained-mass receiver is not a quadratic and is discontinuous at score ties.
Let `tau_0` be the MP kth-order threshold and let the normalized exact costate mass in
the boundary band be

```
F_0(r) = sum_i w_i 1{|s_0,i - tau_0| <= 2r} / sum_i w_i.
```

If `||s_lambda-s_0||_infinity <= r`, only entries in that band can exchange top-k
membership, hence

```
|M_lambda-M_0| <= F_0(r).                                  (3)
```

For diagonal `W` with minimum positive weight `w_min`, a bound
`Q_delta` on the weighted mean-square displacement implies
`r <= sqrt(m Q_delta / w_min)`. Combining (1) and (3) therefore yields a genuine
finite-sample retained-mass bound—but only after `Sigma`, `b`, and the empirical
boundary-margin envelope `F_0` are custodied. With a strict top-k gap `gamma`, the
strong special case is exact stability: if `r < gamma/2`, then `M_lambda=M_0`.

### 2.3 Matrix-Bernstein covariance leg

If residual innovations are available as `n` independent state clusters and
`Y_j = xi_j xi_j' - Sigma` obeys `||Y_j||_op <= L`, with
`v = ||sum_j E[Y_j^2]||_op`, then the self-adjoint matrix-Bernstein law is

```
P(||sum_j Y_j||_op >= z)
 <= 2d exp[-z^2 / (2(v + Lz/3))].                           (4)
```

Consequently the covariance uncertainty can be propagated through `A` and `c` in (1).
The present cached row bootstrap does not establish the cluster independence or bounded
innovation hypotheses required by (4).

### 2.4 Does the bound close against the curve?

**No.** `verdict_scope=BOUND x PRESENT-CACHED-PRE-SE-CUSTODY`: **SYMBOLIC-CLOSE /
NUMERIC-NO-CLOSE.** `req-R`: cache per-state residual innovation vectors and state
boundaries; estimate or upper-bound `Sigma` with cluster custody; preserve score-boundary
gaps/near-boundary exact mass for every held-out state; declare a defensible Gaussian or
independent-whitened-sub-Gaussian noise model. Until then, inserting guessed covariance or
anti-concentration constants would be fake. The measured curve is substantially tighter
than any unevaluated worst-case bound, but no numeric tightness ratio is claimed.

## 3. Expansion across control surfaces

### 3.1 Costate organ A: lambda-for-tail backtest

The cached trajectory has nine intervals and seven deployment-faithful walk-forward folds.
The loss is the score-law-weighted absolute forecast error under the observed combined
control. It is a **proxy** for downstream control regret: the cache has only one realized
action per interval and no counterfactual next states.

| arm | lambda | mean | p95 | p99 | CVaR95/worst | relative to own .01 mean | relative to own .01 CVaR |
|---|---:|---:|---:|---:|---:|---:|---:|
| A current | .01 | .003902 | .005926 | .006071 | .006108 | — | — |
| A grid minimum | 1000 boundary | .001900 | .004012 | .004530 | .004660 | −51.32% | −23.71% |
| P current | .01 | .003182 | .006178 | .006618 | .006727 | — | — |
| P tail-tuned | .1 | .002633 | .005364 | .006233 | .006451 | −17.26% | −4.11% |
| Q current | .01 | .003067 | .006278 | .006743 | .006859 | — | — |
| Q tail-tuned | .1 | .002512 | .005392 | .006340 | .006577 | −18.09% | −4.11% |

The A curve's mean and CVaR both continue downward through the largest tested lambda.
Therefore today's `lambda=.01` is **neither tail-optimal nor mean-optimal on this grid**,
but `1000` is not a closed finite optimum. It is evidence that suppressing the learned
response is favored in this plateau-dominated seven-fold cache. The inherited persistence
mean is `.002792`; no persistence tail vector was cached, so it cannot enter the CVaR
tournament honestly.

`verdict_scope=INSTANCE x ORGAN-A x SEVEN-CACHED-WALKFORWARD-FOLDS`: **NO-GO for an
operational fixed-positive A lambda today.** Diagnostic argmin=`1000` at an open upper
boundary; operational recommendation=`prefer_persistence`. `req-R`: transient-rich new
folds, a closed lambda bracket, persistence per-fold losses, and counterfactual action-to-
next-state outcomes. Heavy governed launches remain operator-GO.

### 3.2 #433 per-class prior means: how much “MORE” exists?

At the existing `.01`, P/Q improve mean error versus A by 18.46%/21.42%, but worsen
CVaR by 10.15%/12.30%. The prior-mean gain is therefore **not a tail-suppression effect**.
Tail tuning to `.1` supplies a measured additional 4.11% CVaR reduction within both P and
Q, while their means improve another 17.26%/18.09%. Against A's current mean this totals
32.54%/35.63% lower mean, yet their tuned CVaRs remain 5.62%/7.69% worse than A.

`verdict_scope=INSTANCE x #433-P/Q x SEVEN-CACHED-WALKFORWARD-FOLDS`: **MORE is
quantified but not controller-admissible:** `.1` is the closed per-arm diagnostic optimum,
with ≈4.11% proxy-CVaR slack recovered; it does not dominate A in the tail and all three
learned arms lack persistence-tail/counterfactual custody. `req-R`: accrue the preregistered
transient-rich folds and record per-action next states plus persistence per-fold errors.

### 3.3 VRGHAL-named frozen witness-SGD locus

No quantitative convergence-residual row exists. The only candidate locus remains a
frozen-stage/frozen-replay/fixed-loss update window, but no fixed operator, native-norm
nonexpansiveness/contraction factor, unbiased stochastic oracle, variance, smoothability
constant, or residual trace is custodied.

`verdict_scope=FORMULATION x FROZEN-STAGE/FROZEN-REPLAY/FIXED-LOSS WITNESS-SGD
WINDOW`: **CONDITIONAL-NOT-THEOREM-ADMITTED.** `req-R`: freeze the update map; prove and
measure `gamma`, `sigma`, `kappa_E`, oracle unbiasedness, native-norm residuals, and the
trust region. This does not reopen the direct-solve-dominated PRE-SE rung.

### 3.4 Other least-squares/interpolator surfaces

| surface | control role | quantitative tail disposition |
|---|---|---|
| A reweighted G/H/I/J/K/N/O | flow-lens/controller evidence | inherit CVaR+mean-gate law; measure only when distinct folds accrue |
| P/Q/R/S | per-class forecast/control | P/Q measured; R/S owed on record accrual |
| `LinearNCDE` sliding-window ridge | trajectory/equilibrium forecast | fixed `1e-3` is uncalibrated for tails; add fold CVaR and bracket |
| prototype-router local ridge | regime router | fixed `1e-2`; require per-regime CVaR and support count |
| Transient Forge ridge | synthetic disambiguator | research-only; tail gate before any real control use |
| rate-law ladder cross-fit ridge | receiver/rate planner | mean-selected alpha; require pair-block byte-regret CVaR |
| OOF scorer-response ridge | spend/candidate triage | require OOF p95/p99 sign/regret before planner consumption |
| Jacobian MP terminal inverse | inverse research surface | MP-only; singular-value stress and tail receipt before promotion |
| Round-3 RFF ridge | fixed-replay surrogate | settled formulation; do not reopen absent its recorded req-R |

## 4. Unified tail-reliability discipline

For every control-driving interpolator and a declared loss `L_lambda`, use

```
lambda* = argmin_{lambda in Lambda, lambda>0} CVaR_alpha(L_lambda)
          subject to mean(L_lambda) <= mean(L_reference) + epsilon_mean.   (5)
```

Tie-break by p99, mean, then smaller lambda. A load-bearing choice additionally requires
a closed bracket, state/trajectory-block holdout, declared numerical axis, and a tail
quantile beside the mean. Use retained-mass shortfall for RankRLS, true counterfactual
downstream control regret when available, and only the explicitly labeled forecast-regret
proxy otherwise. Lambda zero remains a diagnostic, never an implicitly selected fallback.

Equation (5) is implemented and regression-tested as
`control_interpolator_tail_cvar_mean_gate_v1`; equation (1) as
`fixed_design_correlated_gaussian_ridge_tail_v1`. The typed DSL is
`ControlTailReliabilityPolicy`, default-off and cached-read-only. A standalone DAG FEED
and equation FEED accompany this memo because the shared append-only registries were
already sibling-modified; absorbing those unrelated rows would violate collision and
serializer discipline.

## Triality, stores consulted, and pointer delta

- **DSL:** `src/tac/witness_dsl/control_tail_reliability_policy_20260713.py`.
- **DAG:** `.omx/research/quant_tail_reliability_DAG_FEED_20260713.md`.
- **Equations:** `src/tac/canonical_equations/control_interpolator_tail_reliability_20260713.py`
  plus `.omx/research/quant_tail_reliability_equation_feed_20260713.md`.
- **Measurement:** `tools/measure_quant_tail_reliability_20260713.py` and the durable receipt.
- **STORES CONSULTED:** `reports/latest.md`; `.omx/state/lane_registry.json`;
  `.omx/state/subagent_progress.jsonl`; `.omx/state/master_gradient_anchors.jsonl`;
  `.omx/state/modal_call_id_ledger.jsonl`; `.omx/state/cost_band_posterior.jsonl`;
  `.omx/state/continual_learning_posterior.jsonl`; latest council/design/Codex memos;
  sealed PRE-SE receipt and stage chunks; cached #433 receipt and organ trajectory;
  VRGHAL and heavy-tail landed memos; source grep over `RankRLS`, `pinv`, `lstsq`,
  `ridge`, and normal-equation surfaces.
- **Sensitivity-map contribution:** tail quantiles/CVaR become reliability metadata on
  each control response; no score sensitivity is fabricated.
- **Pareto constraint:** mean gate prevents a tail-only choice from worsening declared
  average loss; persistence remains an explicit comparator.
- **Bit allocator:** no direct bit actuation; rate-law/scorer-response predictors must
  expose tail regret before influencing allocation.
- **Cathedral/autopilot hook:** consume a lambda only after mean gate + closed bracket +
  block holdout; otherwise fail closed to reference/persistence.
- **Continual-learning update:** complete receipt and FEED are the empirical anchor;
  shared registry append is deferred, not silently omitted.
- **Probe disambiguator:** the full lambda grid arbitrates MP versus finite ridge and mean
  versus CVaR on one callable measurement surface.
- **Pointer delta:** `NONE`.

## One-line scoped verdicts

1. `FORMULATION x CACHED-420/60-CROSS-FIT x RANKRLS`: **ADOPT positive ridge for
   tail reliability; numeric lambda remains development-only.** `req-R`: official n120
   raw arrays + state clusters + downstream receiver/control outcome.
2. `BOUND x FIXED-DESIGN-CORRELATED-GAUSSIAN`: **DERIVED-CLOSE symbolically.**
   `req-R` for numeric closure: covariance innovations + boundary-margin envelope.
3. `INSTANCE x ORGAN-A x SEVEN-FOLD-CACHE`: **NO finite tail-optimal lambda bracketed.**
   `req-R`: transient folds + persistence tail + counterfactual regret; use persistence.
4. `INSTANCE x #433-P/Q x SEVEN-FOLD-CACHE`: **4.11% within-arm tail slack measured,
   but no tail dominance over A.** `req-R`: same as organ A.
5. `FORMULATION x FROZEN-WITNESS-SGD`: **CONDITIONAL-NOT-THEOREM-ADMITTED.**
   `req-R`: fixed map + gamma/sigma/kappa/unbiasedness/native residual custody.
6. `SCORE x BYTE-CLOSED-EXACT-EVALUATOR`: **UNMEASURED; pointer unmoved.** `req-R`:
   an exact receiver-closed archive row on a contest-compliant CPU/CUDA axis.
