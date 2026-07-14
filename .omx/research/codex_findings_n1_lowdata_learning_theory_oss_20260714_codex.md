# Codex findings — n=1 learning theory and OSS for the costate organ (2026-07-14)

**Pointer status:** UNCHANGED. Submittable pointer `0.1910828242 [contest-CPU]`; the
`0.1880443979880752 [contest-CPU]` PR128 artifact is a borrowed defensive bank, not a
submission-authority row. This is `research_only=true`, MEANS-only apparatus research. It creates no
score, launch, adoption, or promotion claim.

**Lane:** `lane_n1_lowdata_learning_theory_oss_20260714` · **verdict scope:**
`INSTANCE(real #205 trajectory) < FORMULATION < FAMILY < PARADIGM` · **data custody:** one trajectory,
10 verdicts, 9 intervals, 7 deployment-faithful walk-forward folds. Literature results are evidence
about their stated models and datasets, never measurements on this organ.

## Executive verdict

1. **[DERIVED] What is learnable now:** a low-effective-rank correction to a strong, externally
   specified prior. Nine serially dependent intervals do not identify a free nonlinear response field.
   They can update a conjugate linear residual, a fixed-kernel GP posterior, a fixed prototype policy,
   or a tiny readout on frozen features. In each case the prior/structure does most of the work.
2. **[MEASURED—this instance] The repo already resolves the naive architecture question:** trained
   MLP/GRU/DeepONet arms have walk-forward MAE `0.010–0.086`; the closed-form GP arm has aggregate
   walk-forward MAE `0.001852` versus persistence `0.002792`, but wins only `4/7` folds (`p=0.50`), has
   worse per-class MAE (`0.040533` versus prototype `0.011716`), and has no per-lever response field.
   The regime dispatcher reaches `0.001596`, but its policy remains instance-scoped and
   provisional-until-accrual.
3. **[DERIVED] Architecture implication:** n=1 does not permanently cap representational capacity; it
   caps the capacity that may be learned from this trajectory. A large frozen prior plus a very small
   Bayesian/ridge adapter is defensible. Training a GRU/DeepONet/Gated-DeltaNet from these nine
   intervals is not supported by either the evidence or a non-vacuous tiny-n theory argument.
4. **[SPECULATIVE—backtest owed] Highest-value next arm:**
   `U_hierarchical_physics_residual`, a closed-form hierarchical empirical-Bayes residual around
   the existing physics prior-mean family, with shared block precisions and posterior covariance.
   Because `Q_priormean_iso` beat `P_priormean_aniso` on the current instance, U must ship both fixed
   prior modes as a preregistered disambiguator rather than assuming anisotropy wins. It is
   deterministic NumPy, directly portable to MLX, and $0-testable through the existing real-only
   LOO/walk-forward harness. **No adoption without that backtest.**

## 1. What the theory actually proves—and does not

### Shrinkage and ridge

- **[MEASURED—in-literature model]** James–Stein shrinkage dominates the unshrunk estimator for the
  Gaussian normal-means problem in dimension at least three under squared-error loss. This is not a
  distribution-free theorem about arbitrary trajectory regression. [Stein's original paper](https://digicoll.lib.berkeley.edu/record/112831)
- **[MEASURED—in-literature model]** Ridge is asymptotically minimax for dense signals over
  `l2`-balls under Gaussian-design linear models; when `p/n` is non-negligible, risk also depends on
  the design spectrum. Therefore “ridge is minimax whenever `n≈p`” is false without the model and
  parameter class. [Dicker, *Optimal Estimation and Prediction for Dense Signals*](https://arxiv.org/abs/1203.4572)
- **[DERIVED—organ]** `A_ridge_solve` has more structural parameters than nine independent interval
  equations can identify without shrinkage. Its value is finite, deterministic regularization and a
  low-variance comparator—not universal optimality. The existing physics-prior formulation improves
  the target of shrinkage, which is the right direction.

### Interpolation, double descent, and bigger models

- **[MEASURED—in-literature model]** Double descent can occur past the interpolation threshold, and
  minimum-norm interpolation can generalize under specific random-design/spectrum conditions. It does
  not say every overparameterized model generalizes. [Belkin et al.](https://arxiv.org/abs/1812.11118),
  [Hastie et al.](https://arxiv.org/abs/1903.08560)
- **[DERIVED—organ]** With seven walk-forward decisions, the required covariance/spectrum conditions
  cannot be established from the organ trajectory. The observed neural-arm losses are
  `INSTANCE × FORMULATION` evidence against the current trained B/C/D arms, not a FAMILY-level closure.
  The optimal-form reformulation queue is: frozen/meta-trained features; physics-residual adapters;
  independent-trajectory pretraining; #434 simulator pretraining; then a real-only backtest.

### PAC-Bayes and MDL

- **[MEASURED—in-literature model]** Representative PAC-Bayes bounds pay a complexity term containing
  `KL(Q||P)/n` plus confidence/lower-order terms. In small-data experiments even optimized PAC-Bayes
  bounds can trail tight test-set bounds; they become useful when the posterior stays close to a
  meaningful prior. [Pérez-Ortiz et al.](https://arxiv.org/abs/2106.03542)
- **[DERIVED—organ]** At `n≈9`, a diffuse posterior over a neural net makes the certificate unhelpful.
  PAC-Bayes supports a physics/meta prior plus a tiny posterior adapter; it does not manufacture
  information. MDL gives the same design pressure: complexity depends on the singular spectrum and
  signal-to-noise, not only parameter count. [Dwivedi et al.](https://jmlr.org/papers/v24/21-1133.html)
  For scale only, the common penalty
  `sqrt((KL + log(2 sqrt(n)/delta))/(2n))` is about `0.516` at `n=9`, `delta=0.05`, and `KL=0`;
  this is a **DERIVED illustration under that bound**, not an organ risk measurement.

### Conformal uncertainty

- **[MEASURED—in-literature model]** Classical exact conformal coverage assumes exchangeability.
  Time-series extensions use blocks or dependence assumptions and are only approximately valid in the
  general dependent case. [Chernozhukov et al.](https://proceedings.mlr.press/v75/chernozhukov18a.html)
- **[DERIVED—organ]** Even under exchangeability, `n_cal` calibration scores give tail-probability
  resolution in increments of `1/(n_cal+1)`; blocking nine serial intervals makes it coarser. Conformal
  prediction can wrap a forecaster as a conservative observatory signal, but cannot select or rescue an
  architecture here. It must not replace posterior covariance or walk-forward evaluation.

### Meta/transfer learning

- **[MEASURED—in-literature model]** Meta-learning transfers a prior learned from a distribution of
  related tasks; weighted biased regularization can match transfer-risk lower bounds in a Gaussian
  task model when the task covariance is available/learned. [Konobeev et al.](https://proceedings.mlr.press/v139/konobeev21a.html)
- **[DERIVED—organ]** One real trajectory is not a meta-set. #433 comma10k/openpilot features can define
  a prior or frozen feature map, but do not provide independent costate-labeled tasks. Independent
  Pact trajectories, causal scorer counterfactuals, or #434 source-faithful simulated trajectories are
  the required meta-set; the final arbiter remains real-only.

## 2. Ranked technique × OSS × wire-in table

| rank | technique | learning-theory basis | OSS to lift/draw from | fit at 1 trajectory / 9 intervals | architecture implication | MLX-portable? | exact wire-in point |
|---:|---|---|---|---|---|---|---|
| **1** | **Hierarchical physics-residual conjugate shrinkage** | Gaussian prior + conjugate posterior; partial pooling lowers variance and effective degrees of freedom; physics mean carries identifiable structure | NumPy/SciPy closed form; compare API/results with [`BayesianRidge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html); [PyMC](https://www.pymc.io/projects/docs/en/stable/), [NumPyro](https://github.com/pyro-ppl/numpyro), or [BlackJAX](https://github.com/blackjax-devs/blackjax) only as offline posterior oracles | **Highest.** Fits only residual blocks around an existing prior; past-only marginal evidence chooses among a tiny preregistered precision grid | **prior-unlocks-a-richer residual**, not a bigger free net | **Yes**—`solve`, Cholesky, matmul; keep NumPy authority | add `HierarchicalPhysicsResidualAdjoint` beside `RidgeSolveAdjoint`; expose fixed `Q_iso` and `P_aniso` prior modes; register `U_hierarchical_physics_residual` in `ARCHITECTURES`/`make_model`; existing `backtest()` supplies LOO/WF |
| **2** | Regime/change-point multi-output GP posterior | kernel prior defines smoothness/regime sharing; posterior is finite at tiny n because the kernel, not data, supplies capacity | existing NumPy `T_gp_costate_posterior`; kernel ideas from [tinygp derivative observations](https://tinygp.readthedocs.io/en/latest/tutorials/derivative.html), [GPyTorch](https://docs.gpytorch.ai/en/latest/), [GPflow](https://gpflow.github.io/GPflow/develop/) | **High for total forcing, partial for organ.** Existing T wins aggregate mean but not per-class/lever field/significance | **prior-unlocks nonlinear forecast**; do not duplicate existing RBF arm | Yes if implemented in NumPy/MLX; named OSS is JAX/Torch/TF | reformulate `GPCostatePosteriorAdjoint`: fixed additive kernel `k_time + k_regime + k_class`, past-only hyperparameters, multi-output/coregionalized response; compare to T, not just A |
| **3** | Physics prior mean / analytic residualization | known score law and level-set energy reduce target from full lambda to unknown response residual | existing `P_priormean_aniso` and `Q_priormean_iso`; NumPy/SciPy | **High and already partly built.** P improves A on one instance, but Q beats P and both lose persistence; full Pontryagin response is not known | **keep solve; strengthen and disambiguate the prior** | Yes | extend `PhysicsPriorMeanAdjoint.fit`, not scorer features: retain `m0_unit`, expose posterior residual covariance/block precisions, and test fixed P/Q modes on identical folds |
| **4** | Plain ridge / James–Stein-style shrinkage | bias–variance trade; minimax only under stated dense Gaussian/l2 classes | [`RidgeCV`](https://scikit-learn.org/stable/modules/linear_model.html), `BayesianRidge`; `GraphicalLassoCV` is a covariance-precision tool and is not justified for this 9-row response problem | **Strong baseline; insufficient as universal winner.** Keep A as incumbent comparator | **keep ridge** | Yes; NumPy closed form preferred | `RidgeSolveAdjoint.fit`; if tuning alpha, nest it inside each past-only fold—never select on held-out targets |
| **5** | Prototype/regime partial pooling | discrete strong prior; pooling within named regimes trades bias for variance and remains interpretable | existing `PrototypeRouterLens`; no external dependency needed | **High for routing/transients**, but rare regimes have one or zero examples and policy transfer is unmeasured | **keep small/interpretable** | Yes | pool prototype deltas/covariances by existing regime labels; posterior-predictive abstention in `prototype_router.py`; evaluate through current dispatcher backtest |
| **6** | PAC-Bayes / MDL selection | bound or code length penalizes posterior movement/effective spectrum; useful only with informative prior at tiny n | lift equations/tests, not a heavy runtime; no mature drop-in is needed | **Medium as a guard, low as a learner.** Likely vacuous for free nets; useful to compare tiny residual posteriors | **prior-unlocks-small adapter** | Yes for closed-form Gaussian KL/MDL | add diagnostics to backtest JSON: Gaussian `KL(Q||P)`, effective df, log evidence/code length; never replace real WF MAE |
| **7** | Dependence-aware conformal wrapper | finite-sample marginal coverage under exchangeability; block/randomization extensions under dependence | [MAPIE](https://github.com/scikit-learn-contrib/MAPIE), [crepes](https://github.com/henrikbostrom/crepes) | **Low resolution.** Too few serial blocks for a useful nominal 90/95% certificate; valuable only as coarse abstention telemetry after more intervals | **uncertainty wrapper only** | Core quantiles yes; packages are NumPy/sklearn | wrap per-fold residuals after `backtest`; emit interval/abstain telemetry, never train `lambda_net` with conformal scores |
| **8** | Meta-/transfer-learned prior | task-distribution covariance or initialization supplies information absent in target task | [learn2learn](https://learn2learn.net/); [`higher`](https://github.com/facebookresearch/higher) is archived/Torch-only; NumPyro/PyMC/BlackJAX for hierarchical offline analysis | **Not currently identified.** #433 features are priors, not independent lambda-labeled tasks | **needs independent trajectories or #434**, then tiny adaptation | Meta OSS: no. Frozen outputs/readout: yes | future pretraining outside `lambda_net.py`; import only frozen prior/features, then fit a Bayesian/ridge readout in a new arm; ≥3 independent records remain the graduation gate |
| **9** | #434 physics simulator / trajectory distillation | source-faithful causal counterfactuals expand state-action coverage; validity depends on simulator fidelity | existing Transient Forge design/engine; generic diffusion/few-shot packages are not the authority | **Potentially high, presently research-only.** Existing synthetic fixture did not beat real baselines and tested the wrong plateau-heavy regime | **needs #434 data**, but family remains open through optimal transient-rich simulation | NumPy reference + MLX generation possible | train candidate externally; `lambda_net.py` receives frozen candidate; admission only through unchanged real-only LOO/WF gate |
| **10** | GRU/DeepONet/Gated-DeltaNet trained on target trajectory | high capacity can express state tracking, but target data do not constrain it; benign interpolation conditions unverified | existing B/C/D; [GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet), [OLMo-core](https://github.com/allenai/OLMo-core) | **Lowest in current form.** Current trained arms lose; 7B language-model evidence is non-transferable | **needs #434/independent tasks**, or freeze almost all weights and learn a tiny prior-regularized readout | Reference repos are Torch/CUDA; no direct MLX lift | future optimal form replaces `GRULambdaNet` path encoder only after pretraining; keep state tracker frozen and fit `U`-style residual/readout; compare against A/P/T/dispatcher |

## 3. Top-1 next technique: `U_hierarchical_physics_residual`

### Canonical equation candidate (real conjugate law; not registered until a producer exists)

Let the interval-rate target be `y`, one preregistered existing prior prediction
`f_phys in {f_Q_iso, f_P_aniso}`, and `Z` contain only preregistered residual features. Partition
coefficients into shared drift, class-response,
and fixed-regime residual blocks. With observation covariance `Sigma` and block precision
`P = blockdiag(tau_g^-2 I)`:

```text
y = f_phys + Z beta + epsilon,       epsilon ~ N(0, Sigma)
beta ~ N(0, P^-1)

P_N      = P + Z^T Sigma^-1 Z
beta_hat = P_N^-1 Z^T Sigma^-1 (y - f_phys)
Cov(beta | y) = P_N^-1
Lambda_hat(x, phi) = Lambda_phys(x, phi) + B(x, phi) beta_hat
```

This is a genuine conjugate posterior identity, not an empirical organ law. It guarantees a unique
posterior solve for positive-definite `P` even when `p>n`; it does **not** guarantee out-of-sample
accuracy. The latter remains entirely with the backtest.

### Minimum honest experiment ($0)

1. Reuse the exact #205 trajectory extraction and fold definitions; never fit hyperparameters on a
   held-out interval.
2. Prior modes = fixed `Q_priormean_iso` and `P_priormean_aniso` siblings; residual blocks and a
   small finite precision grid are preregistered. Select precision by marginal likelihood using only
   each fold's prefix. Report effective degrees of freedom and posterior variance; do not silently
   optimize dozens of taus or select P/Q on the held-out target.
3. Compare identical folds against persistence, `A_ridge_solve`, `Q_priormean_iso`, `P_priormean_aniso`,
   `T_gp_costate_posterior`, prototype/Bregman, and the dispatcher. Primary gate: class-weighted
   walk-forward MAE. Secondary: per-class MAE, fold wins/sign test, sign/direction accuracy, binding
   AUROC where defined, calibration, regime-stratified errors.
4. **Adoption rule:** no adoption on a mean-only win. It must beat the preregistered incumbent on the
   existing real-only gate, retain deterministic NumPy parity, and remain provisional until independent
   trajectory accrual satisfies `GRADUATION_MIN_RECORDS=3`.

## 4. Direct answers

### Olmo Hybrid / Gated-DeltaNet

**Verdict:** `FORMULATION(current target-trained DeltaNet) = NOT TESTABLE AS LEARNABLE FROM n=9;
FAMILY INTACT`. The paper establishes expressivity/scaling in a controlled 7B language-model setting,
not sample efficiency for this costate organ. [Olmo Hybrid](https://arxiv.org/abs/2604.03444)

It is learnable at n=1 **only in the precise sense that a prior supplies nearly all weights**: freeze a
physics/#434/meta-trained sequence encoder and learn a tiny conjugate/ridge adapter or readout. Scratch or
full fine-tuning must wait for #434 and/or independent real trajectories. Even then, admission is a
real-only backtest against A/P/T and the dispatcher, not a borrowed benchmark.

### Is the physics prior the cheapest n=1 unlock?

**Yes for residualization; no for a data-free full organ.** The score-law factor `dS/dx` and useful
level-set prior directions are analytic/cached, so subtracting a preregistered prior mean is the cheapest way to lower
the residual variance. But the organ needs the response `dx/du` (or the full state/control Jacobian and
terminal/boundary conditions) to form the actual Pontryagin adjoint. Those are not supplied by the score
law alone. Therefore “physics prior needs no data” is true for the prior mean, false for the calibrated
lever field. A source-faithful simulator can supply response structure, but returns to #434 and the
real-only gate.

## 5. Round-1 adversarial review

- **Attack: is top-1 just P with a new name?** No: P uses a single ridge precision and emits no
  posterior covariance; Q is its isotropic sibling and currently beats it. U's testable delta is fixed
  P/Q modes plus block partial pooling, past-only evidence selection, and posterior predictive
  uncertainty. If those do not improve the real gate, U is only an
  `INSTANCE × FORMULATION` negative; queued reforms are regime change-point blocks, structured GP
  residuals, and #434/meta-prior accrual.
- **Attack: can empirical Bayes overfit nine points?** Yes. Hence the finite preregistered grid, prefix-
  only selection, few shared precisions, and no held-out target access. A full per-coefficient ARD or
  horseshoe sampler is not the next experiment.
- **Attack: is LOO honest for a trajectory?** LOO diagnoses influence but is not deployment-faithful.
  Walk-forward is primary. Leave-one-**interval**-out must not be narrated as independent-sample
  generalization; leave-one-regime/block-out is an additional stress test when enough blocks exist.
- **Attack: does the current GP already solve this?** No. T forecasts total forcing with an RBF time
  prior; `response()` is deliberately zero. U targets the missing per-lever residual field.
- **Attack: is a negative against current neural arms a family closure?** No. It is current-arm/current-
  instance evidence. The optimal-form queue above remains open under the operator's 2026-07-14
  discipline.

## 6. Triality, stores, and pointer delta

- **DSL:** no mutation. Research-only candidate name `U_hierarchical_physics_residual`; register only
  with an implemented model, deterministic tests, and typed architecture enumeration.
- **DAG:** standalone feed `n1_lowdata_learning_theory_oss_DAG_FEED_20260714.md`; no shared hot-DAG
  append.
- **Equations:** the conjugate residual posterior above is derived and recorded as an equation
  candidate. **No canonical registry mutation** until an implemented producer/consumer exists; this
  prevents an orphan law.
- **Six hooks:** sensitivity map = posterior mean/covariance over per-lever lambda; Pareto = unchanged
  exact score-law readout; bit allocator/autopilot = no actuation until backtest; continual learning =
  one result per independent trajectory; disambiguator = identical-fold U-vs-P-vs-T backtest.
- **STORES CONSULTED:** `CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; craft handoff; `reports/latest.md`;
  canonical frontier pointer; lane/subagent registries; latest Codex/Claude memos; #433, #434, GP,
  router, envelope, and Olmo DAG memos; `lambda_net.py`, `aniso_perclass_lambda.py`,
  `continual_costate.py`, and `tools/lambda_net_backtest.py`; primary papers and official OSS pages
  linked above.
- **Pointer delta:** none. **Launch/dispatch/eval delta:** none. **Run/process signal delta:** none.
  **Adoption delta:** none.
