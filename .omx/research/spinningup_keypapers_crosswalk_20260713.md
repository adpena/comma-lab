---
title: "OpenAI Spinning Up key-papers crosswalk to the five live Pact control surfaces"
date_utc: "2026-07-13"
lane_id: "lane_spinningup_crosswalk_20260713"
research_only: true
review_status: "UNREVIEWED_MAIN_REVIEW_OWED"
authority: "literature synthesis and controller design only; no launch, evaluator, score, or pointer authority"
pointer_delta: "NONE"
corpus_revision: "OpenAI Spinning Up docs revision 038665d6"
papers_enumerated: 105
papers_mapped: 47
papers_discarded: 58
---

# Spinning Up key papers: adversarial crosswalk to S1--S5

## Answer first

**Corpus result:** `105` papers enumerated exactly from the official page; `47` bear on at least one
live Pact surface and `58` are discarded for this mission. The count excludes every off-list supplement.
The source is unusually useful as a mechanism vocabulary, but it is not the corpus described in the
prompt: the published revision ends at paper 105, has no offline-RL section, and omits BCQ, BEAR, CQL,
FQE, MBPO, PETS, Dyna, Bootstrapped DQN, Option-Critic, automatic/reverse curriculum, PEARL, and
Reptile. Those missing papers are handled in a separately marked supplement below; none is silently
inserted into the 105-paper census.

The build-changing conclusions are:

1. **S1 -- branch, do not free-run (`INFERRED`).** The transferable MBRL object is a short surrogate
   branch rooted at an exact teacher-labeled optimizer state, followed by a real-teacher audit. MVE gives
   the fixed-horizon baseline; STEVE supplies per-state horizon mixing; MBPO supplies short branches
   from real data; PETS supplies epistemic-disagreement custody. None supplies an evaluator-cell bound
   for discontinuous `d_seg`.
2. **S2 -- support before value (`DERIVED` from current data plus `INFERRED` literature mapping).**
   BCQ/BEAR/CQL can make an organ conservative inside logged support; they cannot identify an unlogged
   deterministic arm. Such arms remain `NOT_IDENTIFIED`. FORE/FQE/CQL become additional gates only
   after transition-complete state, action, propensity/support, reward, successor, and run-stratum logs.
3. **S3 -- finite experimental design, not Atari curiosity (`INFERRED`).** The 72 owed rows are arms
   with missing outcomes, not novel states. VIME-style information gain and bootstrapped posterior
   sampling transfer; raw RND/ICM novelty does not. Descriptor pseudo-counts may prevent a lever family
   from starving, but exact receiver-realized improvement remains the reward.
4. **S4 -- stage as an option with transactional termination (`INFERRED`).** Option termination is a
   useful typing, not a license to learn a free-running `beta(s)`. Advance requires the existing
   nucleus/topology/pose/rate eligibility gates, minimum dwell/hysteresis, a preserved common checkpoint,
   and rollback. A loss-slope sensor alone is specifically falsified on the current trace.
5. **S5 -- meta-init needs tasks, not checkpoints (`DERIVED` from corpus custody).** MAML/Reptile/PEARL
   require a real distribution of independent videos or clips with compatible optimized witnesses.
   Multiple epochs from the same contest clip are not a meta-training corpus. Reptile is the lowest-risk
   first baseline once the corpus exists; recurrent RL2 is not.

## Scope, labels, and source custody

- `MEASURED`: read from a current custody-bearing Pact artifact. No measurement was executed here.
- `DERIVED`: algebra or a direct logical consequence of named source facts.
- `INFERRED`: a cross-domain transfer from an RL mechanism to a Pact object; it still needs a gate.
- `ASSUMED`: a bridge that is neither proved nor measured and may not be used as authority.
- `DISCARD`: no build-changing transfer survives the present hypothesis mismatch. This is a mission-
  scoped triage, never a verdict on the paper or method family.
- Negative verdict ladder: `INSTANCE < FORMULATION < FAMILY < PARADIGM`. Every negative below is at
  most `FORMULATION x CURRENT PACT DATA`, unless a narrower scope is stated.

Primary corpus: [OpenAI Spinning Up, Key Papers in Deep RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html),
docs revision `038665d6`; source file:
[openai/spinningup keypapers.rst](https://github.com/openai/spinningup/blob/master/docs/spinningup/keypapers.rst).
The corpus has 13 sections with counts `31,11,8,3,5,9,4,5,4,6,6,7,6`, summing to `105`.

Current live-surface facts were re-derived from the stores listed near the end. Most load-bearing:

- **S1 `MEASURED-INHERITED`:** the round-2 fixed replay head used 600 unique states, a 480/120 split,
  12x inclusive teacher-call amortization, heldout aggregate costate cosine
  `0.0014157933865487525`, and relative L2 `1.0000018705777456`. It is a narrow fixed-distribution
  directional `GO`, not live-surrogate authority. The current round-3 sibling owns frozen-stem/RFF/
  margin-field fidelity work.
- **S1 `DERIVED`:** exact-teacher economics remain `C_teacher=A+c_label*D`; new on-policy anchors or
  target-policy validation are new `A`, not free merely because cached-label differences have
  `c_label=0`.
- **S1/S2 `DERIVED`:** FORE's discounted KL contraction survives deterministic optimizer dynamics
  for `gamma<1`, but the present cache has no transition-sufficient `(Z,A,Z')` tuples or action
  coverage. The current organ's unlogged arms are causally unidentified.
- **S3 `MEASURED-INHERITED`:** `duty_to_measure_ranked()` presently exposes 77 significance rows,
  exactly 72 of them owed lever measurements. The separate 28-row curriculum pool is not silently
  merged into the action set.
- **S4 `MEASURED-INHERITED`:** the default event trigger would have fired near epoch 151 while
  `d_seg` was still descending; a slope plateau is not a sufficient exit certificate. Under unified
  tau, the old CE-to-tau boundary is also not the live transactional surface.
- **S5 `MEASURED-INHERITED`:** task #211 is unstarted and corpus-gated. The current artifact explicitly
  refuses to treat checkpoints from one contest clip as fictitious independent tasks.

## Full official-corpus enumeration

Disposition is counted once per paper: any `S*` row is one mapped paper even when it touches several
surfaces. Mechanism text is deliberately one line and Pact-specific.

| # | Official section | Paper (year) | Surface or discard | One-line mechanism / scoped reason |
|---:|---|---|---|---|
| 1 | Model-free / DQN | Playing Atari with Deep Reinforcement Learning (2013) | DISCARD | Replay plus target-network Q-learning is a generic control baseline; Pact has neither Atari rewards nor a value-learning need on S1--S5. |
| 2 | Model-free / DQN | Deep Recurrent Q-Learning for Partially Observable MDPs (2015) | DISCARD | Recurrent observation memory does not solve missing counterfactual duty-row outcomes or deterministic organ action support. |
| 3 | Model-free / DQN | Dueling Network Architectures for Deep Reinforcement Learning (2015) | DISCARD | Value/advantage factorization has no identified consumer in teacher calls, costates, duty rows, or stage exits. |
| 4 | Model-free / DQN | Deep Reinforcement Learning with Double Q-learning (2015) | DISCARD | Reduces max-Q overestimation but not offline support failure; CQL/BCQ are the more direct S2 lineage. |
| 5 | Model-free / DQN | Prioritized Experience Replay (2015) | S3 (`INFERRED`, weak) | Priority by surprise can order already-measured rows; importance correction warns that selective replay must not masquerade as an unbiased response model. |
| 6 | Model-free / DQN | Rainbow: Combining Improvements in Deep Reinforcement Learning (2017) | DISCARD | A confounded bundle of DQN improvements offers no isolatable Pact mechanism. |
| 7 | Model-free / policy gradient | Asynchronous Methods for Deep Reinforcement Learning (2016) | DISCARD | Distributed actor-learners conflict with single-box MLX custody and add stale-gradient complexity without a live need. |
| 8 | Model-free / policy gradient | Trust Region Policy Optimization (2015) | S1 (`INFERRED`) | Constrain a surrogate-driven optimizer update to a locally audited region; KL policy bounds do not become `d_seg` guarantees. |
| 9 | Model-free / policy gradient | High-Dimensional Continuous Control Using Generalized Advantage Estimation (2015) | DISCARD | Lambda trades bias and variance across return horizons, but STEVE/MVE give the closer model-rollout transfer. |
| 10 | Model-free / policy gradient | Proximal Policy Optimization Algorithms (2017) | DISCARD | Ratio clipping is generic and cannot repair surrogate model bias or unlogged-arm support. |
| 11 | Model-free / policy gradient | Emergence of Locomotion Behaviours in Rich Environments (2017) | DISCARD | Locomotion curriculum and environment scale are task-specific and reset-dependent. |
| 12 | Model-free / policy gradient | Scalable Trust-Region Method using Kronecker-Factored Approximation (2017) | DISCARD | K-FAC/distributed natural-gradient machinery does not change the current single-box teacher bottleneck. |
| 13 | Model-free / policy gradient | Sample Efficient Actor-Critic with Experience Replay (2016) | S2 (`INFERRED`, caution) | Truncated off-policy correction demonstrates the bias/variance bill of replay; it still assumes behavior support absent for unlogged organ arms. |
| 14 | Model-free / policy gradient | Soft Actor-Critic (2018) | DISCARD | Entropy-regularized continuous control has no justified translation to typed deterministic schedule arms. |
| 15 | Model-free / deterministic PG | Deterministic Policy Gradient Algorithms (2014) | DISCARD | A deterministic actor gradient is not a causal estimator from deterministic logs. |
| 16 | Model-free / deterministic PG | Continuous Control with Deep Reinforcement Learning (2015) | DISCARD | DDPG's actor-critic machinery adds extrapolation risk and no live control benefit. |
| 17 | Model-free / deterministic PG | Addressing Function Approximation Error in Actor-Critic Methods (2018) | DISCARD | Twin critics/delays reduce overestimation but do not solve the S2 coverage failure. |
| 18 | Model-free / distributional | A Distributional Perspective on Reinforcement Learning (2017) | DISCARD | Return distributions are aleatoric/value uncertainty, not calibrated epistemic uncertainty over teacher errors or duty rows. |
| 19 | Model-free / distributional | Distributional RL with Quantile Regression (2017) | DISCARD | Quantile return fitting has no identified Pact target. |
| 20 | Model-free / distributional | Implicit Quantile Networks (2018) | DISCARD | Flexible return quantiles remain the wrong uncertainty object for S1/S3. |
| 21 | Model-free / distributional | Dopamine: A Research Framework for Deep RL (2018) | DISCARD | Useful historical reproducibility infrastructure, superseded by Pact's stricter custody apparatus. |
| 22 | Model-free / baselines | Q-Prop (2016) | S1 (`INFERRED`) | Use an approximate critic/model as a control variate while retaining an exact residual correction; the teacher must remain in the loop. |
| 23 | Model-free / baselines | Action-dependent Control Variates via Stein's Identity (2017) | S1 (`INFERRED`, weak) | Structured control variates suggest cheap local gradient correction, but admissibility depends on the exact estimator hypotheses. |
| 24 | Model-free / baselines | The Mirage of Action-Dependent Baselines in RL (2018) | S1 (`INFERRED`, adversarial) | Re-evaluation found methodological errors in claimed control-variate gains; require same-budget exact-teacher baselines and no proxy-only promotion. |
| 25 | Model-free / path consistency | Bridging the Gap Between Value and Policy Based RL (2017) | DISCARD | Multi-step soft consistency does not map to exact receiver debt or stage termination. |
| 26 | Model-free / path consistency | Trust-PCL (2017) | DISCARD | Adds an off-policy trust region to an otherwise non-transferable value-consistency objective. |
| 27 | Model-free / hybrid | Combining Policy Gradient and Q-learning (2016) | DISCARD | Hybrid gradient estimators lack a clearer role than Q-Prop/MVE on the live surfaces. |
| 28 | Model-free / hybrid | The Reactor (2017) | DISCARD | Replay, distributional critics, and prioritized sequences are an Atari agent bundle, not a Pact controller primitive. |
| 29 | Model-free / hybrid | Interpolated Policy Gradient (2017) | S1/S2 (`INFERRED`, caution) | Mix exact/on-policy and replay/off-policy gradients with an explicit bias/variance dial; support and teacher-call custody remain mandatory. |
| 30 | Model-free / hybrid | Equivalence Between Policy Gradients and Soft Q-Learning (2017) | DISCARD | Theoretical equivalence between two irrelevant objective parameterizations does not change a live gate. |
| 31 | Model-free / evolutionary | Evolution Strategies as a Scalable Alternative to RL (2017) | DISCARD | Black-box perturbations multiply exact objective calls, the opposite of the 95%-kill objective. |
| 32 | Exploration | VIME (2016) | S3 (`INFERRED`, high) | Rank a measurement by expected information gain about the lever-response model per certified cost, not by visual novelty. |
| 33 | Exploration | Unifying Count-Based Exploration and Intrinsic Motivation (2016) | S3 (`INFERRED`) | Descriptor pseudo-counts can expose under-measured lever families; they are an anti-starvation term, not expected pointer descent. |
| 34 | Exploration | Count-Based Exploration with Neural Density Models (2017) | S3 (`INFERRED`, weak) | A learned density generalizes coverage across related duty descriptors, but is excessive until descriptor semantics and real outcomes exist. |
| 35 | Exploration | #Exploration: Count-Based Exploration for Deep RL (2016) | S3 (`INFERRED`, weak) | Hashing offers a cheap coarse coverage audit for duty-row descriptors; collisions forbid authority. |
| 36 | Exploration | EX2: Exploration with Exemplar Models (2017) | S3 (`INFERRED`, weak) | Exemplar density ratios can detect catalog OOD arms but do not predict their score value. |
| 37 | Exploration | Curiosity-driven Exploration by Self-supervised Prediction (2017) | S3 (`INFERRED`, negative lesson) | Prediction error is confounded by irreducible/noisy outcomes; use it only after separating measurement noise from epistemic ignorance. |
| 38 | Exploration | Large-Scale Study of Curiosity-Driven Learning (2018) | S3 (`INFERRED`, adversarial) | Broad ablations warn that observation features and stochasticity dominate curiosity behavior; exact held-out ranker comparison is required. |
| 39 | Exploration | Exploration by Random Network Distillation (2018) | S3 (`INFERRED`, weak) | Fixed-target prediction error is a cheap novelty score for descriptors, but cannot outrank expected descent or information gain. |
| 40 | Exploration / unsupervised | Variational Intrinsic Control (2016) | DISCARD | Skill empowerment is irrelevant to a fixed finite measurement bank. |
| 41 | Exploration / unsupervised | Diversity is All You Need (2018) | DISCARD | Reward-free skill diversity does not identify useful witness levers. |
| 42 | Exploration / unsupervised | Variational Option Discovery Algorithms (2018) | DISCARD | Discovers skills, not safe termination of already named curriculum stages. |
| 43 | Transfer | Progressive Neural Networks (2016) | DISCARD | A new column per task defeats compact reusable pre-seeding and does not solve task-corpus custody. |
| 44 | Transfer | Universal Value Function Approximators (2015) | DISCARD | Goal-conditioned values do not define stage-exit truth for a single witness trajectory. |
| 45 | Transfer | RL with Unsupervised Auxiliary Tasks (2016) | DISCARD | Auxiliary pixel/reward predictions are generic and proxy-heavy. |
| 46 | Transfer | The Intentional Unintentional Agent (2017) | DISCARD | Multi-intention off-policy control requires a task/reward interface absent here. |
| 47 | Transfer | PathNet (2017) | S5 (`INFERRED`, weak) | Reuse a selected parameter path across corpus tasks while freezing prior paths; growth and alignment costs make it a comparator, not the first build. |
| 48 | Transfer | Mutual Alignment Transfer Learning (2017) | DISCARD | Domain-adversarial robot alignment is less direct than held-out-video meta-init validation. |
| 49 | Transfer | Learning an Embedding Space for Transferable Robot Skills (2018) | S5 (`INFERRED`) | Infer a task/clip embedding that conditions an initialization; only valid after distinct-video corpus and receiver-realized adaptation tests. |
| 50 | Transfer | Hindsight Experience Replay (2017) | DISCARD | Relabeling failed goals assumes resettable goal episodes and cannot relabel the frozen evaluator target. |
| 51 | Hierarchy | Strategic Attentive Writer for Learning Macro-Actions (2016) | S4 (`INFERRED`) | Explicit commitment length for macro-actions motivates minimum stage dwell and a cost for premature re-planning. |
| 52 | Hierarchy | FeUdal Networks for HRL (2017) | S4 (`INFERRED`) | Separate slow stage manager from fast optimizer updates; manager goals remain telemetry-checked, not reward-free authority. |
| 53 | Hierarchy | Data-Efficient HRL / HIRO (2018) | S4 (`INFERRED`, caution) | Off-policy correction for changing high-level goals highlights nonstationarity when stage semantics change; it does not define the exit gate. |
| 54 | Memory | Model-Free Episodic Control (2016) | DISCARD | Nearest-neighbor reuse of high-return states is weaker than the existing checkpoint/receipt registry. |
| 55 | Memory | Neural Episodic Control (2017) | DISCARD | Differentiable episodic value memory has no live consumer beyond already-settled artifact recall. |
| 56 | Memory | Neural Map (2017) | DISCARD | Spatial memory for navigation is task-specific. |
| 57 | Memory | Unsupervised Predictive Memory in a Goal-Directed Agent (2018) | DISCARD | Latent episodic prediction does not improve current meta-init or OPE identifiability. |
| 58 | Memory | Relational Recurrent Neural Networks (2018) | DISCARD | General relational memory adds state and resume burden without a named S1--S5 gap. |
| 59 | Model-based / learned | Imagination-Augmented Agents (2017) | S1 (`INFERRED`) | Let a learned policy interpret model rollouts rather than trust them literally; useful precedent for exposing uncertainty and rollout diagnostics to the controller. |
| 60 | Model-based / learned | Neural Network Dynamics for MBRL with Model-Free Fine-Tuning (2017) | S1 (`INFERRED`) | Use the learned model for early sample efficiency, then hand control back to exact optimization as model bias dominates. |
| 61 | Model-based / learned | Model-Based Value Expansion for Efficient Model-Free RL (2018) | S1 (`INFERRED`, high) | Fixed-depth surrogate expansion is the clean baseline for a pre-registered teacher-free trust horizon. |
| 62 | Model-based / learned | Stochastic Ensemble Value Expansion / STEVE (2018) | S1 (`INFERRED`, high) | Mix rollout horizons per state using ensemble uncertainty; exact teacher anchors remain the zero-horizon fallback. |
| 63 | Model-based / learned | Model-Ensemble Trust-Region Policy Optimization (2018) | S1 (`INFERRED`, high) | Require an update to improve across a model ensemble inside a trust region; disagreement is a refusal/query signal, not a score claim. |
| 64 | Model-based / learned | Model-Based RL via Meta-Policy Optimization (2018) | S1 (`INFERRED`) | Meta-train a policy to adapt across model errors; only relevant after distinct error regimes and held-out real-teacher validation exist. |
| 65 | Model-based / learned | Recurrent World Models Facilitate Policy Evolution (2018) | S1 (`INFERRED`, caution) | Compact latent dynamics can amortize rollouts, but open-loop latent fidelity and ES evaluation cost make it a cautionary comparator. |
| 66 | Model-based / given | Mastering Chess and Shogi by Self-Play / AlphaZero (2017) | DISCARD | Exact rules, resets, self-play, and tree search do not match a learned surrogate over one deterministic optimizer trajectory. |
| 67 | Model-based / given | Thinking Fast and Slow with Deep Learning and Tree Search / ExIt (2017) | S1 (`INFERRED`) | Distill an expensive search/teacher into an apprentice, but retain periodic expert relabeling and measure apprentice-induced action error. |
| 68 | Meta-RL | RL2: Fast RL via Slow RL (2016) | S5 (`INFERRED`, negative lesson) | A recurrent learned optimizer can adapt from history, but hidden-state resume and episode/termination assumptions make it a poor first #211 baseline. |
| 69 | Meta-RL | Learning to Reinforcement Learn (2016) | S5 (`INFERRED`) | Learn an adaptation rule across tasks; requires a real video-task distribution and exact state/checkpoint persistence. |
| 70 | Meta-RL | Model-Agnostic Meta-Learning (2017) | S5 (`INFERRED`, high) | Optimize an initialization for rapid few-step adaptation on a held-out video; differentiating through exact-teacher steps is the principal cost risk. |
| 71 | Meta-RL | A Simple Neural Attentive Meta-Learner / SNAIL (2018) | DISCARD | Attention plus temporal convolution is a heavier adaptation mechanism than the corpus and evidence justify. |
| 72 | Scaling | Accelerated Methods for Deep RL (2018) | S1/S5 (`INFERRED`, bounded) | Preserve only single-box lessons: tune batching, environment/learner balance, and update intensity under functional-parity and memory gates. |
| 73 | Scaling | IMPALA (2018) | DISCARD | Distributed actor/learner staleness and V-trace are outside the single-box MLX surface. |
| 74 | Scaling | Distributed Prioritized Experience Replay / Ape-X (2018) | DISCARD | Distributed replay throughput does not reduce frozen-teacher cost on one box. |
| 75 | Scaling | Recurrent Experience Replay in Distributed RL / R2D2 (2018) | DISCARD | Sequence replay and distributed recurrent actors add no current value. |
| 76 | Scaling | RLlib (2017) | DISCARD | General distributed infrastructure would duplicate the existing apparatus without changing control quality. |
| 77 | Real world | Benchmarking RL Algorithms on Real-World Robots (2018) | DISCARD | Robot-specific timing and reset issues are less direct than the dedicated reproducibility papers. |
| 78 | Real world | Learning Dexterous In-Hand Manipulation (2018) | DISCARD | Massive simulation/domain randomization and robotics transfer do not map to the frozen contest evaluator. |
| 79 | Real world | QT-Opt (2018) | DISCARD | Large-scale off-policy robotic grasping depends on broad exploratory data that the organ explicitly lacks. |
| 80 | Real world | Horizon: Facebook's Applied RL Platform (2018) | S2 (`INFERRED`) | Production logging, replay, feature, and evaluation separation supports a strict organ data contract, not an algorithmic OPE guarantee. |
| 81 | Safety | Concrete Problems in AI Safety (2016) | S1/S2 + containment (`INFERRED`, high) | Expensive supervision and distribution shift directly name teacher-query and replay-drift risks; reward hacking maps to proxy-only surrogate promotion. |
| 82 | Safety | Deep RL from Human Preferences (2017) | DISCARD | Preference learning replaces a missing reward; Pact already has a frozen exact objective and must not substitute taste. |
| 83 | Safety | Constrained Policy Optimization (2017) | S4 + containment (`INFERRED`) | Keep topology, pose, rate, resource, and custody constraints explicit during controller updates rather than folding them into an optimistic scalar reward. |
| 84 | Safety | Safe Exploration in Continuous Action Spaces (2018) | containment (`INFERRED`, weak) | A safety-layer projection parallels the governor/refusal surface, but the continuous actuator model is not imported. |
| 85 | Safety | Trial without Error: Safe RL via Human Intervention (2017) | DISCARD | Human intervention is an expensive and non-reproducible substitute for existing deterministic guards. |
| 86 | Safety | Leave No Trace: Learning to Reset (2017) | DISCARD | The witness process has no free episodic reset; rollback comes from preserved stage checkpoints, not a learned reset policy. |
| 87 | Imitation / IRL | Maximum Causal Entropy IRL (2010) | S2 (`INFERRED`, caution) | Inferring a reward from logged behavior cannot turn deterministic arm choices into causal counterfactuals; maximum entropy is not support. |
| 88 | Imitation / IRL | Guided Cost Learning (2016) | DISCARD | Alternating reward inference and policy optimization solves the wrong problem because Pact's score is known. |
| 89 | Imitation / IRL | Generative Adversarial Imitation Learning (2016) | DISCARD | Occupancy matching would imitate logged schedules rather than evaluate or improve them. |
| 90 | Imitation / IRL | DeepMimic (2018) | DISCARD | Motion-reference tracking and physics control are task-specific. |
| 91 | Imitation / IRL | Variational Discriminator Bottleneck / VAIL (2018) | DISCARD | Stabilizing an imitation discriminator has no identified consumer. |
| 92 | Imitation / IRL | One-Shot High-Fidelity Imitation / MetaMimic (2018) | S5 (`INFERRED`, weak) | Amortize from demonstrations to a new task, but replace demonstration fidelity with held-out video witness adaptation and exact receiver metrics. |
| 93 | Reproducibility | Benchmarking Deep RL for Continuous Control / rllab (2016) | S2/S3/S4/S5 (`INFERRED`) | Freeze common environments/configs and compare methods on identical splits; Pact analog is common-checkpoint, common-custody A/B. |
| 94 | Reproducibility | Reproducibility of Benchmarked Deep RL Tasks (2017) | S2/S3/S4/S5 (`INFERRED`) | Treat implementation, seed, and environment variance as first-class; one favorable run cannot graduate a controller. |
| 95 | Reproducibility | Deep Reinforcement Learning that Matters (2017) | S2/S3/S4/S5 (`INFERRED`, high) | Report uncertainty, significance, hyperparameters, code, and baselines; directly reinforces Pact's exact hashes, walk-forward folds, and no-borrowed-number rule. |
| 96 | Reproducibility | Where Did My Optimum Go? (2018) | S4 (`INFERRED`) | Optimizer trajectories and apparent optima can be brittle; stage changes need preserved pre-boundary checkpoints and matched continuation controls. |
| 97 | Reproducibility | Are Deep Policy Gradient Algorithms Truly Policy Gradient Algorithms? (2018) | S1 (`INFERRED`, adversarial) | Implementation details can sever the claimed gradient/objective link; validate surrogate update direction against exact teacher action, not paper identity. |
| 98 | Reproducibility | Simple Random Search Provides a Competitive Approach to RL (2018) | S3 (`INFERRED`, baseline) | Every sophisticated measure-next controller must beat transparent random/cost-only acquisition at equal exact-measurement budget. |
| 99 | Reproducibility | Benchmarking Model-Based RL (2019) | S1 (`INFERRED`, high) | Compare MBRL methods under a common implementation/data budget and expose model-error accumulation; use the same rule for surrogate horizons. |
| 100 | Classic | Policy Gradient Methods with Function Approximation (2000) | DISCARD | Foundational theorem, but no direct S1--S5 design delta survives. |
| 101 | Classic | An Analysis of Temporal-Difference Learning with Function Approximation (1997) | S2 (`INFERRED`, caution) | Off-policy bootstrapping with approximation can diverge; a fitted organ value is refused without coverage and stability evidence. |
| 102 | Classic | Reinforcement Learning of Motor Skills with Policy Gradients (2008) | DISCARD | Historical policy-gradient review adds no live mechanism. |
| 103 | Classic | Approximately Optimal Approximate Reinforcement Learning (2002) | S1 (`INFERRED`) | Performance-difference bounds motivate a conservative local update envelope; assumptions must be re-derived for deterministic optimizer states and discontinuous debt. |
| 104 | Classic | A Natural Policy Gradient (2002) | DISCARD | Natural-gradient geometry is not the current teacher-call or controller bottleneck. |
| 105 | Classic | Algorithms for Reinforcement Learning (2009) | DISCARD | Excellent general reference, but too generic to count as a build-changing transfer. |

### Corpus-count check

`DERIVED`: mapped indices total `47`; discarded indices total `58`; `47+58=105`. Off-list papers
below are not included in any of those three numbers.

## Necessary off-list supplement -- explicitly not part of the 105

The user named mechanisms that the source revision does not contain. Excluding them would answer the
page but not the live research question; including them without a boundary would falsify the census.

| Supplement | Surface | Mechanism that survives | Adversarial boundary |
|---|---|---|---|
| Dyna (Sutton, 1990/1991) | S1 | Alternate real transitions with model-generated updates from anchored states. | Generic template only; no model-error or query-real gate. |
| PETS (Chua et al., 2018) | S1 | Bootstrap ensemble plus trajectory sampling separates data uncertainty from rollout propagation. | Our optimizer is substantially deterministic; ensemble spread is mostly epistemic/numerical, not environmental aleatoric variance. |
| MBPO (Janner et al., 2019) | S1 | Use short model rollouts branched from real data and expand horizon only with measured generalization. | Its monotonic-return analysis does not bound argmax-cell `d_seg`. |
| BCQ (Fujimoto et al., 2018) | S2 | Restrict candidate actions to those plausibly generated by the logged behavior. | No support means no action, not a pessimistic numeric value. |
| BEAR (Kumar et al., 2019) | S2 | Constrain backed-up actions near the behavior distribution to limit bootstrapping error. | A distance threshold cannot create positivity for an unseen typed arm. |
| CQL (Kumar et al., 2020) | S2 | Learn a conservative value that penalizes out-of-distribution actions. | Conservatism is not causal identification; deterministic unlogged arms remain `NOT_IDENTIFIED`. |
| FQE / bootstrapped FQE | S2 | Evaluate a fixed target policy by fitted Bellman regression and report uncertainty. | Needs transition completeness, concentrability/coverage, stable rewards/dynamics, and honest run dependence. |
| FORE (van der Laan and Kallus, 2026) | S1/S2 | Fit occupancy ratios without Bellman completeness; discounted KL contraction survives deterministic kernels for `gamma<1`. | Current cache/logs lack full Markov transitions and target-action coverage. |
| Bootstrapped DQN / Thompson sampling (Osband et al., 2016) | S3 | Sample a coherent response model for a whole measurement epoch, avoiding myopic dithering. | The 72 rows are a finite bandit bank, not a long-horizon MDP; use bootstrap acquisition, not DQN. |
| Information-directed sampling | S3 | Trade expected regret against information gain when one measurement resolves many arms. | Not present on the official page; requires a calibrated posterior and named loss, currently absent. |
| Option-Critic (Bacon et al., 2016) | S4 | Type stages as options and their exits as learned termination functions. | Do not import its termination gradient without a valid return model and safety eligibility gates. |
| Deliberation cost (Harb et al., 2017) | S4 | Penalize option switching to prevent chattering and account for rewarm/checkpoint risk. | Cost must be measured/derived; it is not a guessed scalar. |
| Reverse Curriculum (Florensa et al., 2017) | S4 | Expand difficulty outward from achieved states. | `FORMULATION x PACT`: requires resettable start-state generation; does not fit one sacred optimizer trajectory. |
| Teacher-Student Curriculum Learning (Matiisen et al., 2017) | S4 | Allocate practice to tasks with high learning progress and revisit forgetting. | Raw slope is falsified as an exit signal here; use only behind topology/full-facet eligibility. |
| Reptile (Nichol et al., 2018) | S5 | First-order move of a shared initialization toward task-adapted weights. | Requires aligned parameters and independent tasks; cheaper and more resumable than MAML, but not corpus-free. |
| PEARL (Rakelly et al., 2019) | S5 | Infer a posterior over latent task context and adapt under task uncertainty. | Off-policy reward-MDP assumptions do not transfer; only the latent-context uncertainty pattern survives. |

## S1 synthesis -- 95%-kill surrogate as a short-horizon learned world model

### The five papers that change the build

1. **MVE (`INFERRED`, fixed-horizon baseline).** A learned model is used only to a declared depth,
   after which a non-model estimate resumes. Pact transfer: from an exact teacher-labeled optimizer
   anchor, permit exactly `h` surrogate updates and then query the frozen teacher. A fixed `h` is the
   required control arm; it is not presumed optimal.
2. **STEVE (`INFERRED`, horizon arbitration).** Interpolate among several rollout horizons per state
   using an ensemble's uncertainty. Pact transfer: keep `h=0` (exact teacher) in every candidate set,
   emit per-horizon predicted costate/renderer-gradient and disagreement, and mix or choose only
   horizons whose error was calibrated on held-out exact audits.
3. **MBPO (`INFERRED`, branch geometry).** Short model rollouts begin from real replay states rather
   than free-running from model states indefinitely. Pact transfer: every surrogate branch roots at a
   complete, exact-teacher-audited, resumable stage/checkpoint state. Branches never chain from an
   unaudited surrogate-only terminal state into a new “real” anchor.
4. **PETS (`INFERRED`, uncertainty custody).** Bootstrap ensembles expose epistemic disagreement and
   propagate it through trajectories. Pact transfer: frozen-stem/RFF/margin-field ensemble spread is
   a query/refusal sensor. Because the optimizer dynamics are mostly deterministic, do not label
   ensemble spread “aleatoric environment noise”; data order, numerical backend, and unmodeled state
   need separate fields.
5. **ME-TRPO (`INFERRED`, robust local step).** Optimize inside an ensemble trust region rather than
   against one model. Pact transfer: a proposed surrogate update must retain favorable direction
   across the admitted ensemble and stay inside a measured local state/feature support envelope.

MVE and STEVE are in the official corpus. MBPO and PETS are off-list but answer the literal trust-
horizon/query-real question more directly. Dyna supplies only the outer alternation skeleton. World
Models is not selected because long latent free-runs plus ES are exactly the error/call pattern to avoid.

### Concrete Pact controller object

At an exact anchor `Z_k`, let ensemble member `e` predict the exact-teacher action/costate object after
`h` surrogate updates, `g_hat[k,h,e]`. Let `g_bar[k,h]` be the ensemble mean, `u[k,h]` a registered
disagreement statistic, and `E_cal(h, region)` a held-out upper confidence bound on exact audit error
for the current feature/support region. The candidate law is intentionally only
`FORMALIZATION_PENDING`:

```text
admit horizon h only if
  anchor_custody_complete
  AND transition_state_schema_complete
  AND in_calibrated_support(Z_k, h)
  AND E_cal(h, region) <= registered_error_budget
  AND no predicted/observed argmax birth-death or stage-boundary event
  AND every full-facet containment guard passes.

h_star = largest admitted h; if none, h_star = 0 (query exact teacher).
```

`INFERRED`: a STEVE-style mixture may weight admitted horizons inversely to calibrated prediction
variance/error, but it is not canonical until the exact-teacher heldout experiment shows that the
weights predict direction better than fixed-depth MVE and `h=0`. Ensemble agreement is not enough:
all heads can share frozen-stem bias.

Query the real frozen teacher on any of these events:

- stage entry/exit and every preserved branch terminal;
- calibrated error/support refusal or non-finite prediction;
- target-class birth/death, tie-locus crossing, or margin-field event near the discrete argmax surface;
- scheduled randomized audit with recorded positive propensity, even when uncertainty is low;
- any proposed promotion from training-gradient evidence to receiver/evaluator-cell evidence.

The randomized audit is needed to observe confident shared-model failure. A deterministic
“query only when uncertain” rule makes the skipped region's error unidentifiable.

### Economics and hypothesis mismatch

For existing cached same-state differences, the settled law is

```text
C_teacher = A + c_label*D.
```

A branched controller must extend, not overwrite, it:

```text
C_teacher_total = A_sealed + A_new_anchor + A_audit + c_label*D_reuse.
```

`DERIVED`: every newly visited optimizer state that needs an exact label belongs in `A_new_anchor` or
`A_audit`; it is not made free by a surrogate rollout. A 95% call skip requires the observed exact-call
fraction to be at most 0.05 under the full hook ledger; none of the literature establishes that value
for this instance.

The central mismatch is severe. RL papers usually bound expected return under stochastic MDPs. Pact
has deterministic/near-deterministic optimizer dynamics, no natural episodic reset, and a discontinuous
argmax evaluator debt. A smooth costate error or policy KL can control a local gradient chart but cannot
certify `d_seg` across a class-cell crossing. Therefore the best admissible claim is a **teacher-gradient
trust horizon**, followed by receiver-realized audit; never a model-only score guarantee.

**S1 verdict:** `CONDITIONAL GO`, `verdict_scope=FORMULATION`: short, exact-anchor-branched,
ensemble-calibrated surrogate updates with randomized real audits. `NO-GO`,
`verdict_scope=FORMULATION x CURRENT EVIDENCE`: long free-running surrogate trajectories or any 95%
claim derived from paper sample-efficiency ratios. This FEED is advisory to
`replace_round3_fidelity_wall`; that sibling owns all measurements and code.

## S2 synthesis -- support-first organ backtest and OPE

### The five papers that change the gate

1. **BCQ (`INFERRED`).** Generate/select only actions resembling those in the fixed dataset. Pact:
   a regime-arm is evaluable only where a typed behavior-support model admits that exact arm.
2. **BEAR (`INFERRED`).** Constrain policy/actions near the behavior distribution to prevent
   bootstrapping error accumulation. Pact: prohibit multi-step value backups through schedule actions
   outside logged support; report the support distance and refusal.
3. **CQL (`INFERRED`).** Penalize high values for dataset-absent actions and seek a conservative value.
   Pact: rank supported arms by a pessimistic/lower-confidence statistic rather than a point estimate.
4. **FQE / bootstrapped FQE (`INFERRED`).** Evaluate one fixed target schedule policy by fitted
   Bellman regression, with chronological/cross-fit uncertainty. Pact: an additional estimator after
   transition and coverage gates, not a replacement for the current walk-forward forecast.
5. **Deep RL that Matters (`INFERRED`, methodology).** Standardize configuration, seeds/splits,
   implementation, uncertainty, and baselines. Pact: hashes, run strata, past-only splits, simple
   baselines, fold spread, and real-only adoption are part of the estimator, not reporting polish.

FORE sharpens the same conclusion: only-ratio realizability can remove Bellman-completeness demands,
but it does not remove Markov sufficiency, common conditional dynamics/rewards, positivity, target-
action support, or initial-state coverage.

### Concrete backtest gate

For each proposed organ target policy or regime-arm comparison:

1. **Schema gate (`DERIVED`).** Require full verdict-boundary state `Z_t`, actual typed action `A_t`,
   preregistered receiver-relevant reward `R_t`, successor `Z_(t+1)`, run/code/hardware/axis stratum,
   initial state, and every confounder needed for sequential interpretation.
2. **Support gate (`DERIVED`).** Compute per-regime target-action coverage. If a deterministic log took
   another arm and target mass is positive on the absent arm, emit `NOT_IDENTIFIED`; never clip an
   infinite/undefined ratio into a score.
3. **Estimator gate (`INFERRED`).** On supported policies only, compare the present past-only
   walk-forward GP, behavior-cloned/no-change baseline, FQE/FORE-OPE, and a BCQ/BEAR/CQL-style
   support-constrained conservative arm. Hyperparameters and estimands are fixed before the future
   fold is opened.
4. **Uncertainty gate (`INFERRED`).** Report across-run and chronological fold uncertainty, maximum
   importance/occupancy weight, effective sample size, normalization error, influential transitions,
   and sensitivity to run strata. One deterministic trajectory cannot supply an independent-run CI.
5. **Adoption gate (`DERIVED` from current contract).** A method must win on future real runs under the
   existing real-only `lambda_net_backtest.py` rule. Synthetic or same-run fit is research signal only.

`D40` is not “apply a better offline algorithm.” It is a future logging intervention: at safe stage
boundaries, log positively supported alternative schedule actions or restrict future targets to the
actions actually logged. This memo does not authorize that intervention or any launch.

**S2 verdict:** `NO-GO`, `verdict_scope=FORMULATION x CURRENT ORGAN DATA`: causal OPE of unlogged
schedule arms from the present essentially deterministic logs. `CONDITIONAL ADOPT` for a support-
constrained conservative backtest on logged arms after transition completeness and real walk-forward
validation. BCQ/BEAR/CQL reduce extrapolation; none defeats the identification boundary.

## S3 synthesis -- duty-to-measure as a finite, partially observed experimental-design problem

### The five papers that change measure-next

1. **VIME (`INFERRED`, closest).** Prefer a measurement that changes the posterior over lever-response
   dynamics, not merely one whose descriptor is novel.
2. **Bootstrapped DQN / Thompson sampling (`INFERRED`, off-list).** Sample one coherent response model
   for a whole measurement epoch, giving temporally coherent exploration of a lever family without
   maintaining a fragile exact posterior.
3. **Pseudo-count exploration (`INFERRED`).** Generalize “how often measured” through a typed lever
   descriptor so entire mechanism families do not starve.
4. **Prioritized replay (`INFERRED`, weak).** Revisit rows with large residual/surprise, but retain
   sampling propensity/importance correction when fitting the response model.
5. **RND/ICM plus the large-scale curiosity study (`INFERRED`, negative control).** Prediction error
   and fixed-target novelty are cheap descriptor-OOD signals, but can reward noise, nonstationarity,
   and irrelevant uniqueness. They are ablations, not the primary controller.

Information-directed sampling is absent from the official page. Its regret-versus-information ratio
is conceptually close, but cannot be claimed without a calibrated posterior over exact improvement.

### Concrete finite-bank policy

The state is one frozen common checkpoint/regime; each arm is one owed measurement row with a typed
descriptor and certified measurement cost; the reward is the exact receiver-realized score-component
change and bytes, never a proxy loss. A candidate, non-canonical acquisition law is:

```text
acq(i) = [ E(( -DeltaS_i )_+) + lambda_IG * I(Y_i ; response_model | history)
           + lambda_cov / sqrt(pseudocount(descriptor_i)+1) ] / certified_cost_i
```

All coefficients and probability models are `ASSUMED/FORMALIZATION_PENDING`; no value is proposed.
The P8 headroom cap and current cost-only ranker remain external governors. The law is admitted only
after real historical rows show calibrated uncertainty and the sophisticated ranker beats:

- uniform random measurement;
- cheapest-first/cost-only EIG;
- current P8 relative-significance ordering;
- family-stratified round robin.

Each measurement epoch freezes the response representation, selects one or a small preregistered
batch from the same source checkpoint, preserves every branch checkpoint, records propensity and all
failed/invalid outcomes, then updates the posterior atomically at the boundary. No within-epoch model
or loss-weight drift.

`verdict_scope`: RND/ICM as direct “remaining descent” estimators are `NO-GO` at `FORMULATION`; their
descriptor novelty remains an open anti-starvation feature. The correct family is finite contextual
pure exploration/experimental design, not a deep episodic Atari agent.

## S4 synthesis -- curriculum stages as options with guarded termination

### The five papers that change stage control

1. **Option-Critic (`INFERRED`, off-list).** Supplies the typing `(option, intra-option policy,
   termination beta, policy over options)`. Pact: a stage is an option; optimizer updates are the
   intra-option policy; a boundary controller proposes termination.
2. **Deliberation cost (`INFERRED`, off-list).** Switching is not free. Pact switch cost includes
   optimizer rewarm/reset, checkpoint/verification wall, topology risk, and potential rollback.
3. **STRAW (`INFERRED`).** Explicit macro-action commitment supports minimum dwell and discourages
   epoch-by-epoch exit chatter.
4. **FeUdal Networks (`INFERRED`).** A slow manager should consume full-facet geometry while the fast
   worker follows the fixed stage program; the manager does not rewrite loss weights per step.
5. **Teacher-Student Curriculum (`INFERRED`, off-list caution).** Learning progress can schedule
   attention and detect forgetting, but cannot serve as the exit predicate because the present
   plateau threshold fired mid-descent.

HIRO adds the warning that high-level goals become off-policy as lower-level behavior changes. Reverse
Curriculum is discarded for the live instance because there is no resettable start-state generator.

### Concrete transactional termination law

Let `G_o(Z_t)` be the conjunction of existing stage eligibility guards: class-nucleus/topology
survival, annulus/boundary state, pose trust region, rate/resource/custody constraints, and the live
unified-tau semantics. A future common-checkpoint shadow comparison produces receiver-closed losses
`L_stay` and `L_advance` over the same horizon. The candidate decision is:

```text
advance only if
  G_o(Z_t) == true
  AND minimum_dwell_and_hysteresis_pass
  AND upper_confidence(L_advance + switching_cost) < lower_confidence(L_stay)
  AND complete pre-boundary checkpoint is preserved and rollback-loadable.
otherwise stay, refuse for insufficient information, or rollback.
```

No dwell, confidence, or switching-cost number is guessed here. `#344` NCDE and learning-curve slope
are sensors; neither is the counterfactual branch outcome. Intrinsic time records how much evidence
has accumulated, not the direction of the decision. The only causal branch selector is a sacred
common-checkpoint `{stay, advance}` comparison or an estimator with explicit propensities and bias
custody.

**S4 verdict:** `NO-GO`, `verdict_scope=FORMULATION x CURRENT LIVE TRACE`, for a learned/free-running
per-step Option-Critic termination or slope-only event clock. `CONDITIONAL GO` for stage-boundary,
guarded, minimum-dwell, transactional shadow selection with full checkpoint preservation. This keeps
the existing #315/#344/adaptive-Bayes work and changes its typing, not its authority.

## S5 synthesis -- corpus-generalized witness initialization

### The five papers that change #211

1. **MAML (`INFERRED`).** Directly optimize `theta_0` so a small fixed number of new-video updates
   reduces receiver-realized debt. It is the faithful conceptual baseline but expensive because
   second-order differentiation through teacher-bearing steps may erase amortization.
2. **Reptile (`INFERRED`, preferred first build once unblocked).** First-order update toward independently
   task-adapted witness weights. It is simpler, checkpoint-native, and avoids the MAML second-order
   path while testing whether a shared initialization exists at all.
3. **PEARL (`INFERRED`).** Infer a posterior latent `z_video` from a few clip statistics and exact
   audit outcomes; uncertainty can refuse or diversify the seed. Do not import its off-policy RL loss.
4. **RL2 / Learning to Reinforcement Learn (`INFERRED`, later).** A recurrent learned optimizer could
   consume observations, updates, and exact outcomes, but every hidden state becomes resume-critical
   and the episode model is mismatched. It is not phase one.
5. **PathNet / MetaMimic (`INFERRED`, comparators).** Reuse parameter subpaths or one-shot task evidence;
   compare only after the much simpler mean/aligned/Reptile baselines.

### Concrete adoption sequence

1. **Corpus audit first (`DERIVED`).** Inventory distinct source videos/clips, independently optimized
   EMA/stage checkpoints, architecture/config compatibility, parameter symmetries/alignment, exact
   input and target hashes, and per-task teacher/evaluator custody. Same-video epochs count as one
   task lineage, not many tasks.
2. **Baselines (`INFERRED`).** Compare cold init, aligned parameter mean/medoid, nearest clip by a
   predeclared context descriptor, and Reptile initialization. MAML is next only if Reptile shows
   cross-video signal.
3. **Objective (`DERIVED`).** Optimize and evaluate time/teacher calls to a pre-registered exact
   receiver-realized fixed-quality threshold plus final score/rate non-regression. Training loss or
   distance to task weights is not the adoption outcome.
4. **Held-out split (`DERIVED`).** Split by independent video, never by checkpoint from the same video.
   Report dispersion and failure modes, not only average epoch savings.
5. **Single-box scaling (`INFERRED`).** Vectorize or accumulate meta-task gradients only within measured
   MLX memory/functional-parity limits. IMPALA/Ape-X/RLlib contribute no current design delta.

**S5 verdict:** `BLOCKED`, `verdict_scope=INSTANCE x CURRENT CORPUS CUSTODY`, until independent task
count and compatible witness checkpoints are established and the prerequisite FreSh sequence is
closed. The meta-init family remains open. S5 ranks below the top five tickets because starting an
algorithm before proving a task corpus would manufacture evidence.

## Safety / containment -- brief, not forced

- *Concrete Problems in AI Safety* names two literal S1 risks: expensive supervision and distribution
  shift. It also names reward hacking, whose Pact analogue is improving a surrogate/costate proxy while
  evaluator-cell debt worsens. The existing exact audit and no-proxy-promotion rules are the cure.
- CPO supports keeping constraints explicit. The governor, storage waterfall, no-sibling mutation,
  stage checkpoint, pose/rate/full-facet, and authority/custody constraints are not shaped rewards.
- Safe-exploration safety layers parallel fail-closed projection/refusal, but no continuous-control
  theorem transfers to Pact. “Safety” adds no new launch authority here.

## Top-five ranked deep-read / dig tickets

Ranking criterion is qualitative expected value toward (a) the 95%-kill P0 and (b) pointer-relevant
control quality. No numeric EV is guessed. All five charters are research/build-design tickets; none
authorizes training, GPU/provider actuation, or edits to live sibling files.

### 1. `DIG-S1-BRANCH-AUDIT-HORIZON` -- MBPO + MVE + STEVE

**EV: highest / direct P0.** Read the full MBPO, MVE, and STEVE proofs/ablations and derive the smallest
Pact-valid branch controller rooted at exact `replace_round3_fidelity_wall` anchors. Inputs are the
round-3 measurement receipt when it lands, frozen stage/checkpoint custody, exact teacher-call ledger,
and existing `C_teacher` equation. Output one default-off policy schema and probe charter comparing
`h=0`, fixed MVE horizons, and calibrated STEVE mixing at equal exact-call budgets; define state,
transition, audit propensity, every preserved checkpoint, and the full-facet receiver terminal. Primary
falsifier: no `h>0` horizon maintains exact teacher-gradient direction/support better than `h=0` after
call-cost accounting. Verdict scope must remain feature-chart x horizon, never surrogate-family death.

### 2. `DIG-S1-QUERY-REAL-CALIBRATION` -- PETS + ME-TRPO + confident-error audits

**EV: high / direct P0.** Derive an epistemic-disagreement and coverage calibrator for frozen-stem,
RFF, and margin-field round-3 modes without editing the sibling implementation. Separate ensemble,
seed/data-order, backend, and irreducible target variance; preregister randomized positive-propensity
audits so confident shared bias is observable. Output reliability curves for predicted disagreement
versus exact costate/renderer-gradient error, a trust-region/query/refuse policy, and exact accounting
of `A_new_anchor+A_audit`. Primary falsifier: disagreement fails to rank held-out exact error or the
audit floor alone makes a 5% call fraction impossible. No evaluator-cell claim without receiver audit.

### 3. `DIG-S2-SUPPORT-FIRST-ORGAN-OPE` -- BCQ + BEAR + CQL + FQE/FORE

**EV: high / pointer-control quality.** Specify and audit the organ's transition schema, typed action
support, behavior propensities, run strata, target estimand, and future-fold splits. On logged actions
only, compare current walk-forward GP, no-change/behavior baselines, FQE/FORE, and a conservative
BCQ/BEAR/CQL-style estimator with uncertainty. Output a machine-readable matrix assigning each
regime-arm one of `{SUPPORTED_EVALUABLE, NOT_IDENTIFIED, OUT_OF_SUPPORT, SCHEMA_INCOMPLETE}` plus the
smallest D40 logging change needed for future identification. Primary falsifier: no multi-trajectory
overlap or transition sufficiency; in that case the durable deliverable is the refusal schema, not a
numeric OPE row.

### 4. `DIG-S3-FINITE-BANK-INFORMATION-GAIN` -- VIME + bootstrap + pseudo-count controls

**EV: medium-high / cross-lane measurement velocity.** Treat the 72 owed lever rows as a fixed contextual
experimental-design bank at common checkpoints. Define a typed descriptor and exact outcome/cost row,
fit a bootstrapped response posterior only on real custody-bearing outcomes, and compare VIME-style
information gain, posterior sampling, pseudo-count anti-starvation, current P8, cheapest-first, family
round-robin, and random baselines chronologically. Output calibration, simple regret, top-k discovery,
and exact measurements to first confirmed improvement, with atomic boundary checkpoints. Primary
falsifier: descriptor geometry does not generalize response or uncertainty is uncalibrated; fall back
to stratified cheapest-first rather than RND/ICM theater.

### 5. `DIG-S4-TRANSACTIONAL-OPTION-EXIT` -- Option-Critic + deliberation + TSCL

**EV: medium-high / live control quality.** Type each live curriculum stage/tau rung as an option and
enumerate its complete state, eligibility guards, switching costs, minimum dwell/hysteresis, and
rollback checkpoint. Use existing read-only traces to test whether any proposed termination sensor
would have fired before current topology/full-facet gates; then specify the common-checkpoint
`{stay,advance}` shadow receipt needed for causal selection. Output a default-off termination policy
schema and exact stage-boundary DAG. Primary falsifier: no affordable common-horizon counterfactual or
the guards reduce the trigger to the existing fixed boundary; preserve the fixed schedule rather than
claim automatic curriculum value.

**Held next, not top five:** `DIG-S5-CORPUS-REPTILE-READINESS`. It becomes rankable only after the corpus
audit proves multiple independent compatible video tasks and the prerequisite FreSh gate closes.

## Triality, system wire-in, and authority boundary

No code, shared DAG, canonical equation registry, activation ledger, task ledger, trainer, scorer,
archive, or sibling artifact was edited. Candidate names below are routing handles, not claimed live
classes, functions, or flags.

| Surface | DSL leg (all owed/default-off) | Equation leg | DAG/consumer leg | Present status |
|---|---|---|---|---|
| S1 | `SurrogateBranchTrustPolicy` with anchor, horizons, support, audit propensity, checkpoint cadence | branch error/query law + extended `C_teacher` | round-3 receipt -> branch audit -> teacher/query/refuse -> receiver terminal | `FORMALIZATION_PENDING`; FEED sibling/main only |
| S2 | `OrganOfflineSupportGate` with state/action/estimand/schema/stratum hashes | support-constrained OPE and pessimistic selection | logs -> support status -> estimator -> real-only future fold | `FORMALIZATION_PENDING`; current unlogged arms refused |
| S3 | `DutyMeasurementAcquisitionPolicy` with frozen epoch, descriptor, posterior, propensity, cost | finite-bank information-gain acquisition | owed rows -> select -> exact measure -> posterior boundary update | `FORMALIZATION_PENDING`; no ranker win measured |
| S4 | `TransactionalOptionExitPolicy` with guards, dwell, branch horizon, switch cost, rollback hash | guarded stay/advance inequality | stage checkpoint -> shadow branches -> advance/stay/rollback | `FORMALIZATION_PENDING`; slope-only formulation refused |
| S5 | `MetaInitCorpusPolicy` with task-lineage IDs, held-out split, adaptation budget | held-out fixed-quality meta objective | corpus audit -> baseline/Reptile -> receiver validation | `BLOCKED_CURRENT_CORPUS`; no algorithm start |

Six-hook disposition:

- **Sensitivity map:** S1 consumes exact costate/renderer-gradient error by horizon; S2 consumes
  support/occupancy diagnostics; S3 consumes exact measured lever responses; S4 consumes full-facet
  stage deltas; S5 consumes held-out adaptation trajectories. All are absent until their tickets land.
- **Pareto constraints:** teacher calls and wall, exact `d_seg`, nonlinear pose contribution, archive
  bytes, storage, and uncertainty/support debt remain separate. No scalar proxy may hide a regression.
- **Bit allocator:** non-binding for S1--S4 until receiver-realized score-unit value per byte exists;
  S5 initialization is training-time and cannot claim archive savings without byte-close evidence.
- **Cathedral/autopilot:** all proposed policies are default-off and research-only. No dispatch hook.
- **Continual learning:** the corpus census, source-version gap, mapped/discarded table, and scoped
  negatives are the durable anti-rediscovery signal. Shared registration is deferred to main review.
- **Probe disambiguators:** every ticket compares the transferred method to the simplest current
  baseline at equal authority/cost. Taste never selects a mode.

## Primary literature inspected for the synthesis

Official abstracts/landing pages were inspected for the corpus and supplements; the PETS paper body
was also inspected for its uncertainty decomposition/trajectory-sampling mechanism. These are the
load-bearing primary sources, not an exhaustive duplicate of the 105 links on the OpenAI page:

- S1: [MVE](https://arxiv.org/abs/1803.00101),
  [STEVE](https://arxiv.org/abs/1807.01675),
  [MBPO](https://arxiv.org/abs/1906.08253), and
  [PETS](https://papers.nips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html).
- S2: [BCQ](https://arxiv.org/abs/1812.02900),
  [BEAR](https://arxiv.org/abs/1906.00949),
  [CQL](https://arxiv.org/abs/2006.04779), and
  [FQE analysis](https://proceedings.mlr.press/v235/wang24be.html).
- S3: [VIME](https://arxiv.org/abs/1605.09674),
  [pseudo-count exploration](https://arxiv.org/abs/1606.01868),
  [RND](https://arxiv.org/abs/1810.12894), and
  [Bootstrapped DQN](https://arxiv.org/abs/1602.04621).
- S4: [Option-Critic](https://arxiv.org/abs/1609.05140),
  [deliberation cost](https://arxiv.org/abs/1709.04571),
  [reverse curriculum](https://arxiv.org/abs/1707.05300), and
  [teacher-student curriculum](https://arxiv.org/abs/1707.00183).
- S5: [RL2](https://arxiv.org/abs/1611.02779),
  [MAML](https://arxiv.org/abs/1703.03400),
  [Reptile](https://arxiv.org/abs/1803.02999), and
  [PEARL](https://arxiv.org/abs/1903.08254).
- Rigor/containment: [Deep RL that Matters](https://arxiv.org/abs/1709.06560),
  [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565), and
  [CPO](https://arxiv.org/abs/1705.10528).

## STORES CONSULTED and custody

- Full `CLAUDE.md`, full `AGENTS.md`, and full `docs/operating_manual_craft_handoff.md`.
- Top project memory entries; latest Codex findings/session summary; latest T3 council and design memo;
  all last-24h directive files discovered during preflight.
- `reports/latest.md`; lane registry; subagent progress/ownership; master-gradient anchors; cost-band
  and continual-learning surfaces; probe-outcome API/ledger. The generated `reports/latest.md` body is
  historically stale in places, so no pointer number from it is used in this analysis.
- `.omx/research/frozen_replay_convex_head_95kill_20260713.md` and its DAG feed;
  `.omx/research/fore_occupancy_ratio_dig_20260713.md` and DAG feed;
  `.omx/research/tofupov_ranker_allocation_20260713.md` and equation/DAG feeds;
  `.omx/research/adaptivebayes_costate_intrinsictime_DAG_FEED_20260713.md`;
  `.omx/research/init_levers_fresh_metainit_20260712.md`;
  `.omx/research/fresh_run_config_adversarial_review_20260704.md`;
  `.omx/research/v9_cgauge_truly_optimal_design_20260712.md`.
- Official OpenAI page/source and primary paper surfaces linked above. The page was static and not
  JavaScript-gated, so no Vercel agent-browser fallback was needed.

No launch, training, evaluator, scorer, live run, provider, archive, submission, or sibling-owned
source/research file was touched. The only shared apparatus writes were the mandated lane registration
and crash-resume checkpoint rows. This memo is uncommitted for main review. **Pointer delta: `NONE`.**

## DAG FEED -- main-review append only

```text
FEED-SPINNINGUP-CROSSWALK-20260713

official OpenAI Spinning Up revision 038665d6
  -> exact corpus census: 105 = 47 mapped + 58 discarded
  -> source-gap guard: modern off-list supplements remain separately labeled
  -> S1 exact teacher anchors
       -> {MVE fixed horizon | STEVE calibrated horizon | MBPO short real-rooted branch}
       -> PETS/ME-TRPO disagreement + support + randomized audit
       -> {surrogate step | query teacher | refuse}
       -> receiver-realized terminal and C_teacher reconciliation
       -> FEED replace_round3_fidelity_wall / 95%-kill controller
  -> S2 organ logs
       -> transition schema + action-support gate
       -> {SUPPORTED_EVALUABLE | NOT_IDENTIFIED | OUT_OF_SUPPORT | SCHEMA_INCOMPLETE}
       -> walk-forward GP + behavior baseline + FQE/FORE + conservative offline comparator
       -> real-only future-fold adoption
  -> S3 72 owed lever rows
       -> common-checkpoint finite-bank descriptor/outcome schema
       -> VIME/bootstrap acquisition + pseudo-count anti-starvation
       -> equal-budget random/cost/P8/round-robin comparators
       -> exact receiver outcome + atomic posterior update
  -> S4 stage/tau option
       -> full-facet eligibility + minimum dwell + deliberation cost
       -> common-checkpoint {stay,advance} shadow receipt
       -> transactional {advance | stay | rollback | insufficient information}
  -> S5 independent-video corpus audit
       -> {BLOCKED_CURRENT_CORPUS | aligned baseline -> Reptile -> MAML/PEARL}
       -> held-out receiver-realized fixed-quality adaptation

BLOCKERS:
  S1 no calibrated rollout-error-to-discontinuous-d_seg bound; round-3 sibling receipt pending
  S2 current deterministic logs do not identify unlogged arms
  S3 response posterior/descriptor calibration and exact historical outcomes incomplete
  S4 no common-checkpoint receiver-closed stay/advance receipt
  S5 independent compatible task corpus and prerequisite FreSh closure absent

AUTHORITY:
  research_only=true
  score_claim=false
  promotion_eligible=false
  launch_authorized=false
  pointer_delta=NONE
  shared_DAG_append=DEFERRED_MAIN_REVIEW
```
