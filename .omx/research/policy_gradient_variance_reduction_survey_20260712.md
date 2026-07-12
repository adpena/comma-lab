# Policy Gradients Part 2: variance reduction routed to terminal exact-argmax polish and the costate controller

**Date:** 2026-07-12  
**Lane:** `lane_policygrad_part2_research_20260712` (`research_only=true`)  
**Status:** durable research/routing memo; no training, no paid dispatch, no live-run mutation  
**Pointer:** `0.18804` UNMOVED — this memo changes means/apparatus only  
**Verdict scope:** literature survey plus design routing. No estimator has yet been measured on Pact's frozen scorer.

## Answer first

1. **Surface A — ACTIONABLE:** **UGC (unbiased gradient variance clipping)** is the best single unbiased stochastic estimator for tasks #396/#400 **after** reparameterizing each preselected edit direction as a Bernoulli apply/not-apply bit. It combines DisARM's two-evaluation antithetic estimator in the probability interior with one-coordinate `bitflip-1` near the deterministic boundary. That boundary behavior is decisive for terminal polish, whose useful flip policy should converge toward probabilities 0 or 1. Use the exact frozen-scorer objective for every function evaluation and keep the existing exact monotone acceptance gate. [FROM-LITERATURE] UGC is unbiased and is proved to have uniformly lower variance than DisARM under the paper's assumptions. [DERIVED] This is a stronger match than the seed row's generic “REINFORCE with baseline.” [INFERRED] It remains an unmeasured Pact hypothesis until a paired exact-eval estimator-variance receipt exists.

2. **Surface B — WATCH, DO NOT WIRE:** stay on the **adjoint/costate path**. Baselines, GAE, actor-critic, RUDDER, hindsight/counterfactual critics, TRPO, and PPO can reduce or stabilize parts of the estimator, but none creates the missing independent trajectories needed to identify a critic or action effect from one expensive video/training environment. Reward-to-go removes exactly no variance when the only reward is terminal. [MEASURED] The present synthetic-prior costate controller failed held-out adoption on the available plateau trajectory, so a learned policy-gradient controller has still less evidentiary support. [INFERRED] Policy gradient becomes testable only after #434 supplies a pre-registered multi-trajectory corpus and a critic/credit model wins held-out walk-forward checks; until then #426/#319/#433 remain adjoint-led.

3. **Adjoint versus likelihood ratio:** use pathwise/adjoint derivatives wherever the training rollout is differentiable and known; use likelihood-ratio estimators only at genuinely discrete/non-differentiable interfaces; use exact enumeration when the active support is small enough; use exact evaluator confirmation for every accepted archive change. This is a division of labor, not a claim that one estimator dominates in every dynamical system.

## Authority labels and stores consulted

- **[MEASURED]** means read from a local Pact artifact or current repository/OSS metadata in this pass; it does not mean a new scorer experiment was run.
- **[FROM-LITERATURE]** means stated by a cited primary paper or the named project's own repository/docs.
- **[DERIVED]** means a mathematical consequence shown here from stated assumptions.
- **[INFERRED]** means a routing judgment applying literature to Pact; it is explicitly not measured.

**STORES CONSULTED:** `CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; `reports/latest.md`; `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` including `FEED-policy-gradient-pedregosa` and `FEED-cuda-smoke-moneysafety-hold`; `.omx/research/mc_finisher_396_design_20260710.md`; `.omx/research/mc_finisher_diagonal_build_20260710.md`; `.omx/research/feasibility_clickpolish_design_20260710.md`; `.omx/research/synthetic_data_for_costate_organ_supercharge_20260711.md`; current canonical task/lane/subagent ledgers; the primary papers and official OSS repositories linked below.

[FROM-LITERATURE] The seed is Fabian Pedregosa's [Policy Gradients Part 1: REINFORCE](https://fa.bianp.net/blog/2026/policy-gradient/). Its finite-horizon calculation identifies cubic-in-horizon variance growth in the analyzed score-function construction and explicitly defers variance reduction. [INFERRED] That result is a warning about Surface B's long horizon, not a universal theorem that every controlled estimator below has exactly the same asymptotic variance.

## The two Pact surfaces

### Surface A: terminal exact-argmax polish

[MEASURED] Tasks #396/#400 already define a short-horizon, pair-local, exact-gated finisher over discrete per-pixel edits and a small continuous `dxi`/pose component. The authority objective is

\[
S = 100 d_{seg} + \sqrt{10d_{pose}} + 25\,\frac{\text{archive bytes}}{37{,}545{,}489},
\]

measured through the exact frozen scorer and archive bytes. `d_seg` includes a discrete argmax. The current #396 design is a resumable monotone `(1+1)`-ES-style exact finisher; #400 provides pair-local diagonal batching/candidate structure.

[DERIVED] A useful candidate generator can make the stochastic decision binary without weakening the exact verdict: choose a direction/sign/magnitude from the existing margin or diagonal proposal apparatus, then let bit `b_i=1` mean “apply candidate edit `i`” and `b_i=0` mean “leave it off.” The stochastic objective is

\[
J(\phi)=\mathbb E_{b_i\sim\mathrm{Bernoulli}(\sigma(\phi_i))}\left[S(R(x\oplus b,\xi))\right],
\]

where `R` is the actual uint8/resize/archive receiver chain and `S` is the exact contest metric. No differentiability of `S`, `R`, or argmax is assumed.

### Surface B: controller as a policy

[MEASURED] Tasks #426/#319/#433 concern long-horizon controller credit and a learned costate `lambda = dS/dx`; each real trajectory is a training run, and the environment is one video. #434 is the explicit corpus/starvation cure. The transient synthetic-prior controller did not beat its held-out adoption baseline on the available plateau trajectory; this is an **instance/test-regime negative**, not a family-dead verdict.

[DERIVED] For terminal reward `R_T=-S_T`, the causal reward-to-go at every preterminal time is the same scalar:

\[
G_t=\sum_{k=t}^{T} r_k = R_T \quad\text{when }r_k=0\text{ for }k<T.
\]

Therefore the causality trick removes past rewards but removes no return noise here. Any further temporal credit must come from a learned baseline/critic, reward redistribution, a model, or pathwise derivatives.

## 1. Classic variance-reduction toolkit

The common score-function identity is

\[
\nabla_\theta \mathbb E_{\tau\sim p_\theta}[F(\tau)]
=\mathbb E[F(\tau)\nabla_\theta\log p_\theta(\tau)].
\]

[FROM-LITERATURE] Mohamed et al. survey score-function, pathwise, and measure-valued estimators and their combinations: [JMLR 2020](https://www.jmlr.org/papers/v21/19-346.html). [FROM-LITERATURE] Greensmith, Bartlett, and Baxter explicitly analyze baselines and actor-critic as additive control variates: [JMLR 2004](https://jmlr.org/papers/v5/greensmith04a.html).

| Technique | Assumptions and variance removed | Bias status | Surface A fit | Surface B fit |
|---|---|---|---|---|
| Constant baseline | [FROM-LITERATURE] Subtract `c` from the return; because the expected score is zero, this removes global return level/shift without changing the mean gradient. The variance-minimizing constant is score-weighted and need not equal the average reward. | Unbiased if independent of sampled action. | Useful but weaker than coupled binary estimators; exact objective remains intact. | Insufficient: cannot assign terminal credit across time. |
| State-dependent value baseline | [DERIVED] `E_a[b(s) grad log pi(a|s) | s]=b(s) grad sum_a pi(a|s)=0`; it removes state/prefix-predictable return variation. | Unbiased when the baseline excludes the current action; action-dependent baselines need a correction. | Little value for a terminal one-step mask unless there is meaningful state/context variation. | Potentially strong only after enough trajectories exist to learn and validate `V(s)`; current `n=1` makes this unidentified/overfit-prone. |
| General control variate | [FROM-LITERATURE] Use a correlated zero-mean random variable `c` and subtract an optimally scaled version; baselines are a special score-multiplied case. RELAX/REBAR are richer instances. | Unbiased only if the correction preserves known expectation/gradient exactly. | Strong when a trustworthy relaxed surrogate correlates with exact `S`; not guaranteed for argmax/receiver discontinuities. | Requires a learned model/control variate and held-out data; does not solve trajectory starvation. |
| Reward-to-go / causality | [FROM-LITERATURE] Removes rewards preceding an action because those rewards cannot depend on that action. | Unbiased for the same objective. | Horizon is already terminal/small; nearly no gain. | [DERIVED] Zero gain for a single terminal reward: all preterminal `G_t` equal `R_T`. |
| Advantage and GAE | [FROM-LITERATURE] GAE exponentially mixes TD residuals; `lambda` trades bias against variance and relies on a value function: [Schulman et al.](https://arxiv.org/abs/1506.02438). | Exact Monte Carlo advantage with action-independent baseline is unbiased; bootstrapping/approximate `V` and `lambda<1` introduce bias relative to the full undiscounted objective. | Wrong abstraction for a short terminal mask; UGC/exact enumeration is closer to the action structure. | Could be a later option after #434, but current value-function error dominates and terminal reward makes short-lambda credit model-dependent. |
| Actor-critic / A2C/A3C | [FROM-LITERATURE] A critic supplies an action-independent learned control variate and bootstrapped credit; asynchronous actors decorrelate experience: [Mnih et al.](https://proceedings.mlr.press/v48/mniha16.html). | Actor gradient can be unbiased with an exact/action-independent critic; bootstrapping and approximation create bias/error. | Heavy and unnecessary for cheap one-step black-box polish. | Not viable at current rollout count; parallel actors would mean parallel multi-hour training runs, not free decorrelation. |
| Natural policy gradient | [FROM-LITERATURE] Precondition by the policy Fisher so steps respect distribution geometry: [Kakade](https://papers.nips.cc/paper_files/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html). | Does not itself bias the score estimate; does not reduce return/credit noise. | Possible conditioning aid after a useful estimator exists, but unnecessary before UGC is tested. | Stabilizes geometry, not missing information. |
| TRPO | [FROM-LITERATURE] Constrains KL movement and approximately optimizes a monotonic-improvement surrogate: [Schulman et al.](https://arxiv.org/abs/1502.05477). | Practical approximation/surrogate; not a variance cure. | Exact monotone scorer acceptance is stronger and cheaper for this terminal surface. | No rescue: a trust region cannot make one trajectory identify a policy gradient. |
| PPO clipping | [FROM-LITERATURE] Reuses on-policy samples for multiple surrogate epochs and clips probability-ratio incentives: [Schulman et al.](https://arxiv.org/abs/1707.06347). | Clipped surrogate is biased relative to the exact policy-gradient objective. | Admissible only as search heuristic behind exact evaluation; no advantage over the exact ratchet is established. | Sample reuse/stability helps only after representative on-policy samples exist; current `n=1` remains fatal. |

**Classic-toolkit verdict.** [DERIVED] A baseline is unbiased because it multiplies a conditional zero-mean score, not because it approximates the return accurately. [FROM-LITERATURE] Greensmith et al. also show that the average reward baseline and even the true value function need not be variance-optimal. [INFERRED] “Add a critic” is therefore not a sufficient Surface-B plan; critic fit and gradient-variance reduction must be measured separately on held-out trajectories.

## 2. Discrete estimators ranked for Surface A

Ranking is for **exact discrete scorer calls**, not generic discrete latent-variable benchmarks.

| Rank | Estimator | Literature result and assumptions | Exact/unbiased status | Pact Surface-A verdict |
|---:|---|---|---|---|
| **1** | **UGC = DisARM interior + bitflip-1 boundary** | [FROM-LITERATURE] DisARM can explode near Bernoulli boundaries; bitflip-1 has complementary boundary behavior; UGC switches coordinatewise and is proved uniformly lower variance than DisARM under the paper's assumption: [Kunes et al.](https://arxiv.org/abs/2208.06124). | **Unbiased**; evaluates only discrete `f`. | **Best fit.** Terminal polish should become deterministic, exactly where boundary robustness matters. Cheap batched exact calls and pair-local active sets make sparse coordinate updates tolerable. |
| 2 | Exact conditional enumeration / Rao-Blackwellization | [FROM-LITERATURE] Conditioning/marginalizing selected high-probability states reduces variance without changing bias: [Liu et al.](https://proceedings.mlr.press/v97/liu19c.html). | Exact/unbiased; zero conditional action variance on enumerated support. | First choice whenever a local block has small enumerable support. It is a deterministic subroutine/gate, not the scalable global-mask estimator. |
| 3 | DisARM | [FROM-LITERATURE] Antithetic Bernoulli pair plus analytic integration of augmentation noise; unbiased, two black-box evaluations, lower variance than ARM: [Dong et al.](https://arxiv.org/abs/2006.10680). | **Unbiased**; no differentiable relaxation. | Excellent interior estimator and permissive reference code exists; loses to UGC near deterministic probability boundaries. |
| 4 | ARSM / categorical couplings | [FROM-LITERATURE] ARSM uses Dirichlet augmentation, Rao-Blackwellization, swaps, common random numbers, and pseudo-actions for unbiased categorical gradients: [Yin et al.](https://proceedings.mlr.press/v97/yin19c.html). Later categorical coupling work reports that simpler RLOO can beat ARSM in some tested regimes: [Dong et al.](https://arxiv.org/abs/2106.08056). | **Unbiased**. | Best fallback if actions must remain native multiway `{-2,-1,0,+1,+2}` rather than direction-pinned binary bits. More pseudo-actions/evaluations and less direct code fit. |
| 5 | ARMS / multi-sample antithetic REINFORCE | [FROM-LITERATURE] Generalizes binary antithetic coupling to multiple samples and connects DisARM with leave-one-out estimators: [Dimitriev and Zhou](https://proceedings.mlr.press/v139/dimitriev21a.html). | **Unbiased**. | Useful if exact scorer batching makes many coupled masks cheap; not first because UGC directly handles boundary failure. |
| 6 | ARM | [FROM-LITERATURE] Binary, unbiased, common-random-number/antithetic estimator: [Yin and Zhou](https://arxiv.org/abs/1807.11143). | **Unbiased**. | Superseded by DisARM's Rao-Blackwellization at equal black-box evaluation count. |
| 7 | REBAR | [FROM-LITERATURE] Uses a Concrete relaxation as a control variate while retaining an unbiased discrete gradient: [Tucker et al.](https://arxiv.org/abs/1703.07370). | **Unbiased estimator**, despite using a biased relaxation inside a corrected control variate. | Weak fit: must invent and differentiate a relaxed receiver/scorer whose correlation with exact argmax `S` is unproven. |
| 8 | RELAX | [FROM-LITERATURE] Learns a differentiable control variate and retains an unbiased estimator: [Grathwohl et al.](https://arxiv.org/abs/1711.00123). | **Unbiased estimator** if correction is implemented exactly. | More flexible than REBAR but adds another model/optimization problem and can overfit cheap local samples; lower EV than UGC before any measured need for a learned surrogate. |
| 9 | REINFORCE + baseline / RLOO | [FROM-LITERATURE] Universal score-function estimator; baselines or leave-one-out samples center its learning signal. | **Unbiased** with action-independent/leave-one-out baseline. | Minimum control arm, not recommendation. Ignores binary coupling and boundary structure. |
| 10 | Gumbel-Softmax / Concrete | [FROM-LITERATURE] Replaces categorical samples with a differentiable continuous relaxation: [Jang et al.](https://arxiv.org/abs/1611.01144), [Maddison et al.](https://arxiv.org/abs/1611.00712). | **Biased for the exact discrete objective** at finite temperature. | Search heuristic only; every proposal must pass exact `S`. It cannot be the NO-FAKE estimator. |
| 11 | Straight-through / ST-Gumbel | [FROM-LITERATURE] Uses a discrete forward sample and a surrogate backward derivative; biased gradients can be poorly aligned: [Pervez et al.](https://proceedings.mlr.press/v119/pervez20a.html). Rao-Blackwellizing ST lowers MSE but not its underlying bias: [Paulus et al.](https://arxiv.org/abs/2010.04838). | **Biased**. | Last-resort search heuristic only; no score authority. |

[DERIVED] Antithetic sampling, coupling, and common random numbers are not standalone objectives: they reduce variance only when the induced covariance is favorable while preserving each sample's marginal law. In Pact's deterministic exact scorer, their main role is to compare deliberately coupled edit masks under identical receiver/configuration state, not to average away scorer randomness.

### Why UGC, concretely

[FROM-LITERATURE] For a factorial Bernoulli objective, DisARM forms an antithetic pair and estimates all active-coordinate gradients from the difference of two exact function values. When a Bernoulli probability approaches 0 or 1, DisARM's variance can grow; `bitflip-1` instead chooses one coordinate and compares matched configurations differing only at that bit, holding the others fixed. UGC uses DisARM away from the boundary and `bitflip-1` near it, preserving unbiasedness.

[DERIVED] Terminal exact polish has the same geometry:

- the proposal apparatus supplies a finite edit direction per candidate;
- the decision is apply/not-apply;
- the exact objective is a black-box function of the resulting bit mask;
- the useful terminal policy should saturate toward deterministic decisions;
- evaluator calls are cheap enough to batch, but the full `2^K` mask space is not enumerable.

[INFERRED] These properties make UGC a better first experiment than RELAX/REBAR or Gaussian ES. The key caveat is that `bitflip-1` updates one coordinate at a time; if the active mask is enormous, sparse updates become evaluation-inefficient. #400's pair-local diagonal/candidate screening is therefore part of the fit, not optional.

### Proposed #396/#400 algorithm — design only, not executed

1. **Candidate compression.** Reuse #400 to produce a small active set of exact receiver-surviving edit directions. Each candidate becomes a Bernoulli apply bit. Do not ask UGC to discover edit sign and membership simultaneously.
2. **Exact objective.** For every sampled mask, evaluate the actual through-`R` n600 scorer and current archive bytes. Cache only deterministically keyed exact results; use identical scorer settings and common deterministic decoding for paired masks.
3. **Interior update.** Draw one shared uniform vector, construct the DisARM antithetic masks, batch both exact scorer calls, and compute the per-logit DisARM estimate.
4. **Boundary update.** For coordinates meeting the UGC paper's variance-switch condition, sample a coordinate and form the matched `bitflip-1` pair differing only there. Batch the exact pair. The threshold must come from the paper/implementation and be logged; do not tune it on claimed score.
5. **Rao-Blackwellize small blocks.** If a selected local support is enumerable, replace its Monte Carlo contribution with exact conditional enumeration. This cannot add bias and may eliminate conditional action variance.
6. **Continuous `xi`.** Keep the small continuous pose variable on the existing symmetric diagonal/antithetic finite-difference or `(1+1)`-ES path; do not force it through a Bernoulli estimator. Mixed optimization alternates the UGC mask step and continuous exact step.
7. **Authority gate.** Convert logits to a deterministic mask only as a proposal. Accept an edit solely when the exact monotone #396/#400 gate improves `S` with no component/byte policy violation. Checkpoint logits, RNG state, active set, exact-eval cache keys, and accepted mask after every stage.
8. **Required comparison receipt.** Same exact-evaluation budget and candidate set for: existing `(1+1)`-ES, REINFORCE-RLOO, DisARM, and UGC. Record estimator sample variance, exact scorer evaluations per accepted edit, accepted `Delta d_seg`, `Delta d_pose`, `Delta bytes`, `Delta S`, and wall time. No estimator promotion without this receipt.

**Verdict scope:** [INFERRED] UGC is the highest-priority estimator to test, not a claim that it already beats the existing finisher. If exact conditional enumeration covers the whole active support, enumeration wins and no stochastic gradient is needed. If scorer calls cease to be cheap or the active set cannot be compressed, the UGC route should be rejected on evaluation cost.

## 3. Black-box / zeroth-order alternatives and the honest crossover

| Method | What it does | When it beats action-space REINFORCE | Pact routing |
|---|---|---|---|
| OpenAI-ES / antithetic ES | [FROM-LITERATURE] Optimizes a smoothed parameter-space objective; common random numbers make distributed rollouts cheap to communicate and it is insensitive to delayed reward/horizon: [Salimans et al.](https://arxiv.org/abs/1703.03864). | Low/moderate parameter dimension, long or delayed rewards, no useful critic/relaxation, cheap parallel rollouts, and a reasonably smooth perturbation distribution. | Existing #396 `(1+1)`-ES is a strong exact baseline. Generic Gaussian ES is poor for a high-dimensional sparse bit mask but reasonable for small continuous `xi`. |
| Natural ES | [FROM-LITERATURE] Takes natural-gradient steps on a parameterized search distribution and includes separable/high-dimensional variants: [Wierstra et al.](https://www.jmlr.org/papers/v15/wierstra14a.html). | When distribution geometry matters and population evaluation is affordable. | Useful baseline for `xi` or a very low-dimensional edit basis, not primary for pixel membership. |
| CMA-ES | [FROM-LITERATURE] Adapts a full covariance for rugged continuous black-box objectives; recommended for roughly small-to-moderate continuous dimensions and can be evaluation-hungry: [Hansen](https://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaesintro.html). | Ill-conditioned, nonseparable continuous search where covariance learning repays its population cost. | Good only after compressing `xi`/edit basis to low dimension. It does not natively exploit exact binary flips. |

**Crossover.** [DERIVED] Both ES and policy gradients are likelihood-ratio gradients of different sampling distributions; “ES versus REINFORCE” is therefore a choice of perturbation space and credit structure, not gradient-free versus gradient-based magic. [INFERRED] ES wins when horizon/credit dominates and parameter-space populations are cheap. UGC/structured score-function estimation wins when the discrete action structure is known, exact evaluations are the bottleneck, and antithetic bit couplings reuse each call across many coordinates. Exact enumeration wins when support is small. At Surface A, use UGC for discrete membership and ES/finite differences for tiny continuous `xi`; at Surface B, ES removes temporal credit assignment but still needs a population of multi-hour full runs, so it does not cure the present economics.

## 4. Terminal/sparse credit and Surface-B verdict

| Method | Literature mechanism | Why it does not presently rescue Surface B |
|---|---|---|
| RUDDER | [FROM-LITERATURE] Learns a return decomposition and redistributes delayed reward so expected future rewards approach zero: [Arjona-Medina et al.](https://papers.nips.cc/paper_files/paper/2019/hash/16105fb9cc614fc29e1bda00dab60d41-Abstract.html). | Needs diverse trajectories to learn which subsequences change return; one trajectory makes attribution and memorization indistinguishable. |
| Hindsight credit assignment | [FROM-LITERATURE] Rewrites value/credit using likelihood of past decisions given observed outcomes: [Harutyunyan et al.](https://papers.nips.cc/paper/2019/hash/195f15384c2a79cedf293e4a847ce85c-Abstract.html). | Requires outcome variation and learned conditional probabilities; current environment has neither at adequate support. |
| Counterfactual credit assignment | [FROM-LITERATURE] Uses future-conditioned baselines while constraining hindsight information to exclude the agent's action, yielding low-variance policy gradients: [Mesnard et al.](https://proceedings.mlr.press/v139/mesnard21a.html). | A counterfactual critic still needs data/model support for alternative outcomes; one real trajectory cannot validate it. |
| Model-based stochastic value gradients | [FROM-LITERATURE] Uses differentiable world-model rollouts for pathwise policy gradients, often short-horizon plus a terminal value estimate: [Amos et al.](https://proceedings.mlr.press/v144/amos21a.html). | This is closer to Pact's costate/adjoint path than to model-free REINFORCE. It supports staying differentiable where the training dynamics are known. |

### Honest verdict

**Stay adjoint.** [DERIVED] Variance reduction cannot compensate for absent counterfactual information. A baseline estimated from one trajectory can subtract that trajectory's return but cannot establish how different controller actions would have changed final `S`. GAE and actor-critic replace variance with critic/model error; PPO/TRPO constrain updates after a gradient exists; ES replaces temporal credit with a population cost; RUDDER/HCA/CCA learn credit models that need trajectories.

[INFERRED] The adjoint is preferred here because the training update equations and intermediate state are available and differentiable over large parts of the rollout. It supplies pathwise sensitivity from one trajectory instead of estimating an expectation over rare terminal returns. It is **not universally strictly lower variance**: long differentiable rollouts can have exploding/vanishing derivatives, discontinuities, and model error. Pact should handle those limitations with checkpointed truncation, costate diagnostics, and terminal exact search—not by declaring bare policy gradient viable.

**Reopen policy gradient only when all are true:**

1. #434 yields a pre-registered set of independent real or empirically calibrated trajectories spanning action/regime changes;
2. a state baseline, RUDDER-style return decomposition, or counterfactual critic beats persistence on held-out walk-forward trajectories without leakage;
3. paired/common-seed rollouts produce a measurable gradient signal-to-noise receipt and action-effect sign stability;
4. the number and wall time of full trajectories per statistically useful update beat the governed adjoint alternative;
5. every proposed controller change remains gated by final exact `S` and provenance.

Until then: #426 costate NN continues as a supervised/differentiable organ; #319 MC-return credit is a later comparator/diagnostic; #433 RL-deepen stays WATCH; #434 is the prerequisite data lane.

## 5. Ranked OSS draw and contribution plan

[MEASURED] Pact is MIT-licensed. License compatibility therefore changes “draw from” into three different actions: import permitted code, study/run out-of-tree, or contribute upstream without importing.

| Rank | Repository | License / contents from official repo | Import versus reimplement | Exact fit and contribution opportunity |
|---:|---|---|---|---|
| **1** | [Storchastic](https://github.com/HEmile/storchastic) | [FROM-LITERATURE] PyTorch stochastic-node framework with REINFORCE moving-average/RLOO, exact enumeration, Gumbel, LAX/RELAX, REBAR, ARM, and Rao-Blackwellized REINFORCE. GitHub labels GPL-3.0 while [PyPI metadata](https://pypi.org/project/storchastic/) says AGPL-3.0. | **Do not import into Pact without a license audit.** Run/read out-of-tree; rederive the small UGC estimator from papers and permissive references. | Best executable estimator catalogue and test oracle. High-value upstream contribution: UGC/DisARM method plus black-box tests, if maintainers accept it under their license. |
| **2** | [Google Research DisARM](https://github.com/google-research/google-research/tree/master/disarm) | [FROM-LITERATURE] Reference DisARM/categorical-coupling experiments; Google Research states source is Apache-2.0. | Draw formulas, coupling tests, and deterministic sampling patterns; port only the minimal Bernoulli estimator to Pact's existing Torch/NumPy style. | Closest permissive reference for UGC's interior estimator. Add Pact-specific exact-objective tests locally; contribute generic bug fixes upstream if found. |
| **3** | [EvoTorch](https://github.com/nnaisense/evotorch) | [FROM-LITERATURE] Apache-2.0 PyTorch library with PGPE, XNES, SNES, CMA-ES, CEM, and evolutionary algorithms. | Use as an out-of-tree reference/baseline; do not pull its trainer stack into terminal polish unless a thin adapter is demonstrably cheaper. | Best backend-aligned ES comparison for #396's existing exact search and small continuous `xi`. |
| 4 | [Pyro](https://github.com/pyro-ppl/pyro) | [FROM-LITERATURE] Apache-2.0 PyTorch probabilistic programming framework; its [official discrete-enumeration tutorial](https://pyro.ai/examples/enumeration.html) supports exact/parallel enumeration subject to dependency/treewidth constraints. Storchastic itself depends on Pyro. | Prefer its enumeration/score-parts ideas or a small permissive utility; avoid a full dependency for one estimator. | Strong exact-enumeration/Rao-Blackwellization reference when local support is small. |
| 5 | [RLax](https://github.com/google-deepmind/rlax) | [FROM-LITERATURE] Apache-2.0 JAX mathematical building blocks for values, returns, and discrete/continuous policy gradients; official docs warn that sampling constraints are not enforceable by the library. | Reference equations/tests; do not add JAX solely for this lane. | Good Surface-B GAE/value-loss comparator after #434, low direct Surface-A value. |
| 6 | [TorchRL](https://github.com/pytorch/rl) | [FROM-LITERATURE] MIT, modular PyTorch returns/objectives including GAE and PPO. | Reuse only if Surface B reopens and TensorDict integration is justified; otherwise copy equations/tests, not framework. | Best native future actor-critic harness, but current verdict says do not wire. |
| 7 | [evosax](https://github.com/RobertTLange/evosax) | [FROM-LITERATURE] Apache-2.0 JAX ask/eval/tell library with CMA-ES, OpenAI-ES, and many other ES methods. | Reference algorithm matrix; EvoTorch fits current backend better. | Useful cross-check of ES defaults and low-dimensional crossover. |
| 8 | [OpenAI ES starter](https://github.com/openai/evolution-strategies-starter) | [FROM-LITERATURE] MIT but archived; distributed master-worker reference tied to the 2017 ES paper. | Read common-random-number/distribution logic; do not adopt the old AWS/Redis stack. | Historical oracle only. |
| 9 | [CleanRL](https://github.com/vwxyzjn/cleanrl) | [FROM-LITERATURE] Single-file benchmarked PPO/A2C-style references; the project explicitly says it is not meant to be imported. | Read/compare; no dependency. License must be checked at the exact revision before copying. | Clear PPO implementation audit if Surface B later reopens; no discrete-estimator depth. |
| 10 | [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | [FROM-LITERATURE] MIT; reliable A2C/PPO plus contrib TRPO/maskable PPO. | Use only as a generic behavior/control baseline, not Pact controller infrastructure. | Mature but mismatched to exact scorer masks and one-video economics. |
| 11 | [Tianshou](https://github.com/thu-ml/tianshou) | [FROM-LITERATURE] MIT; REINFORCE/A2C/NPG/TRPO/PPO/GAE and vectorized environments. | Reference only unless #434 creates a real vectorized environment. | Breadth is useful for validation, but its advantage appears only when many environments/rollouts exist. |
| 12 | [RELAX reference repo](https://github.com/duvenaud/relax), [TensorFlow REBAR](https://github.com/tensorflow/models/tree/master/research/rebar) | [FROM-LITERATURE] Original research-state RELAX code has no visible license in its repo; TensorFlow Models is Apache-2.0. | Do not copy unlicensed RELAX code. TensorFlow REBAR is permissive but framework-mismatched; use as equation/test reference. | Historical reproduction only; Storchastic covers these estimators more coherently out-of-tree. |

## Routing and stop conditions

### ACTIONABLE route: #396 + #400

- Implement only a small, deterministic, resumable UGC estimator prototype behind the existing exact terminal-polish interface.
- Keep `(1+1)`-ES and exact enumeration as registered comparison arms.
- Draw algorithm/test ideas from Storchastic out-of-tree and Apache-2.0 Google DisARM; no copyleft dependency enters Pact.
- Promotion gate is a same-budget exact scorer receipt, not proxy gradient loss.
- Stop if the active set cannot be compressed, the bitflip boundary updates require more evaluator calls per accepted edit than the existing finisher, or exact enumeration already covers the active support.

### WATCH route: #426 + #319 + #433; prerequisite #434

- Do not add PPO/A2C/GAE to the controller now.
- Continue adjoint/costate diagnostics and use exact terminal search at the non-differentiable receiver boundary.
- Let #434's real-trajectory corpus and held-out walk-forward gate decide whether a policy-gradient comparator becomes admissible.
- If reopened, first comparator is REINFORCE with an action-independent state baseline and common-seed paired rollouts; GAE/RUDDER/CCA follow only after their critic/credit model wins held-out checks.

## Final falsifiers

- **Surface-A recommendation falsifier:** same-budget exact evaluation shows UGC has worse exact `S` improvement per scorer call and per wall second than `(1+1)`-ES or exact enumeration on the registered pair-local candidate sets.
- **UGC applicability falsifier:** candidate edits cannot be represented as a factorial/direction-pinned Bernoulli mask without destroying important coupled constraints, or the paper's switching assumptions do not hold for the chosen parameterization.
- **Surface-B adjoint preference falsifier:** after #434, a leakage-free held-out policy-gradient controller yields stable action-effect signs and superior final exact `S` per governed compute cost versus the adjoint controller.
- **NO-FAKE guard:** Gumbel/Concrete, straight-through, learned critics, PPO surrogates, and any relaxed scorer remain search heuristics; none is score authority.

## Triality disposition

`[no-triality]` — pure literature/reference routing with derived equations, no new measured witness result, no DSL lever, and no code/DAG actuator beyond this FEED route. The durable unit is this cited memo plus `FEED-policygrad-part2-research`; pointer remains `0.18804`.
