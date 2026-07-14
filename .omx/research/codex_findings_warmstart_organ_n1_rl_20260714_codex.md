# Codex findings — warm-start ORGAN / n=1 / RL / training dynamics (2026-07-14)

**Pointer status: UNCHANGED — `0.19108282419209976` `[contest-CPU]` submit-capable.**
The defensive-bank `0.1880443979880752` remains borrowed/non-submission. This lane is organ-facing,
`research_only=true`, `$0`, and `[macOS advisory] NON-PROMOTABLE`; no training, provider, evaluator,
archive, pointer, DSL, canonical-equation, autoconfig, or trainer actuation occurred.

## Verdict in one paragraph

The strongest new organ formulation is a block-partial-pooling posterior around the existing Q/P
physics prior, followed by disagreement-focused replay. On the single real trajectory (10 verdicts,
9 dependent intervals, 7 outer walk-forward folds), raw U reached aggregate WF MAE `0.0024959232`
and the requential warm-start reached `0.0024630519`, versus persistence `0.0027919315`. The latter
beats uniform replay by only `0.0000328714` (`1.317%`) and has `4/7` fold wins (`p=1.0`); it is an
adaptive development result, not an independent confirmation. Per-class error remains far above
persistence. The existing #436 T/persistence dispatcher remains the best aggregate composition at
`0.0015959393`, but that number belongs to the existing dispatcher: U merely supplies a per-class
decomposition constrained to its aggregate. HSE shows the broad organ society is mostly redundant,
while T and persistence are behaviorally differentiated. One-ULP perturbations flip #436 on `2/7`
folds, so the net is **REAL INSTANCE-LEVEL aggregate lever; boundary-robustness FORMULATION debt;
no promotion/adoption**.

## Custody and repeatability

- Real run: `experiments/results/levelset_v752_baseline_20260710T185913Z`
- Source log SHA-256: `7fdc44d19946121fb18e35060f5146bf1f48dea81c08891f8f4477d42b0bed82`
- Receipt: `.omx/research/warmstart_organ_n1_rl_backtest_20260714.json`
- Deterministic result digest: `a460fd0716df5f830429f17dd53a8bb97d75423ff4636d26b510651a64ba5721`
- Seed: `0`; reference: NumPy-fp32. Two independent probe invocations produced the same deterministic
  digest. MLX code exists; the focused parity tests skip honestly because this sandbox has no Metal
  device. MPS/MLX is not score authority.
- Owned source SHAs are embedded in the receipt. Shared owner surfaces were not edited.

## Measured development backtest

All rows use the same seven deployment-faithful past-only folds. `pc` is mean per-class WF MAE.

| Mechanism | Aggregate WF MAE | pc WF MAE | vs persistence | Fold signal | Scope |
|---|---:|---:|---:|---:|---|
| persistence | `0.002791932` | `0.0108225` | reference | — | incumbent |
| A ridge | `0.003902384` | `0.0227769` | worse | — | prior settled |
| Q isotropic prior | `0.003066559` | `0.0235030` | worse | — | prior settled |
| P anisotropic prior | `0.003182024` | `0.0227746` | worse | — | prior settled |
| T GP aggregate | `0.001852066` | `0.0266412` | better aggregate | prior arm `4/7` | instance; no lever field |
| U uniform block precision | `0.002961745` | `0.0298299` | worse | `3/7` | ablation |
| U P-only structural prior | `0.002624767` | `0.0549167` | better aggregate | `4/7`, `p=1.0` | instance × formulation |
| U hierarchical Q/P | `0.002495923` | `0.0559097` | `10.602%` better aggregate | `4/7`, `p=1.0` | adaptive development |
| U + VR difference clip | `0.002559264` | `0.0579513` | `8.334%` better aggregate | `4/7`, `p=1.0` | clip formulation loses to U |
| R uniform replay | `0.002495923` | `0.0559097` | same as U | `4/7`, `p=1.0` | control |
| R disagreement replay | `0.002463052` | `0.0395893` | `11.780%` better aggregate | `4/7`, `p=1.0` | adaptive development |
| closed U + existing #436 aggregate | `0.001595939` | `0.0459174` | `42.837%` better aggregate | `6/7`, `p=0.125` | existing router composition |

The initial U form is preserved in checkpoint step 6: WF `0.014095`, dominated by an ep100 error
`0.082718`. That is an **INSTANCE failure of the one-inner-fold selector**, not a family verdict.
The repaired selector freezes at Q/`0.01` until four inner folds. Because the same trajectory exposed
and repaired the instability, every repaired row above is development-WF, not a fresh test.

## Deep warm-starts, divergence forks, and mechanisms

### 1. n=1 low-data learning theory / #499 + #434

The prior arm established the relevant statistical boundary: nine dependent intervals can update a
strong low-effective-rank prior, but cannot identify a free nonlinear response field. The warm-start
therefore keeps a one-dof Q/P physics direction and learns only a block-shrunk residual. Full target-
trained Olmo/Gated-DeltaNet-scale replacement remains inadmissible; a frozen prior encoder plus a tiny
readout is the open optimal form. Synthetic #434 trajectories may pretrain/propose, but only real
prefix walk-forward can adopt.

**Built:** `costate_warmstart_cluster.py`, NumPy-fp32 posterior and optional MLX parity.  
**Measured:** U aggregate improves over persistence, but pc error worsens `5.17×`; only `6/22` lever
shares vary enough to be data-identified, and only `3/6` of those have a conditional 95% sign. The
overall `13/22` sign-resolution count is mostly fixed-prior structure and must not be called learned.  
**Verdict:** `INSTANCE_X_FORMULATION PROVISIONAL`; family open; independent trajectories owed.

### 2. VR-GHAL / #462

The full method is gradual Halpern iteration for stochastic fixed points, with recursive variance
reduction, median-of-means epoch anchors, and clipping of **operator differences** at the exact
Lipschitz radius `gamma_bar ||x-y||`; its anytime theorem assumes an unbiased stochastic oracle and
optionally same-seed multi-query coupling ([paper, §4 / Eq. 7–10](https://arxiv.org/pdf/2607.09097)).

**Divergence:** our cached trajectory is a deterministic regression table, not repeated unbiased
queries to one fixed stochastic operator.  
**Re-derivation:** transfer only `clip(delta y_t, L ||z_t-z_(t-1)||)` with `L` estimated on the past
prefix.  
**Measured:** clipped U `0.002559264` loses to unclipped U `0.002495923`; no theorem is claimed.  
**Verdict:** `FORMULATION` finding for this pathwise clip; VR family remains open only at a real
repeated-query/frozen-operator locus.

### 3. FORE / #471

FORE represents the discounted occupancy ratio in a log-linear family and applies a KL-projected
adjoint Bellman operator; discounting plus data processing yields KL contraction, then a fitted convex
objective uses offline one-step target-policy transitions and initial-state moments
([paper, §3 / Algorithm 1](https://arxiv.org/pdf/2607.05375)).

**Divergence:** there are no behavior/target propensities, visited-live density hashes, production
causal manifests, or executed decision rows.  
**Re-derivation/build:** support certificate only; no invented ratio.  
**Measured:** occupancy rank grows from `2` to `7` over an ambient dimension `10`, but this is
predictive support, not counterfactual coverage.  
**Verdict:** `BLOCKED_DISTRIBUTION_CUSTODY`; no estimator verdict and no family negative.

### 4. TOFU-POV / #463

TOFU-POV assumes action coordinates are independently masked with known Bernoulli probability, uses
a diagonal/off-diagonal missingness-corrected covariance, freezes a low-rank subspace by doubling
epochs, imputes actions, then runs OFUL in reduced coordinates
([paper, Eq. 3 and algorithm](https://arxiv.org/pdf/2607.08971)).

**Divergence:** our missing objects are counterfactual outcomes and offered slates, not randomly
masked coordinates with known reveal probability.  
**Re-derivation/build:** prefix-frozen SVD rank/condition certificate.  
**Verdict:** `BLOCKED_PARTIAL_ACTION_CUSTODY`; no OFUL/regret transfer. The current support
certificate is useful apparatus, not an organ score lever.

### 5. SPS gradient-role separation / #472

SPS decomposes current-token prediction gradients from future state-preparation gradients using
interleaved input/predict streams and persistent versus ephemeral KV state
([paper, Eq. 1–5](https://arxiv.org/abs/2607.01218)).

**Divergence:** the prior witness probe's temporal gradient was exactly zero; its negative was a
disengaged instance.  
**Re-derivation:** Q/P physics response is the persistent stream; the intercept/state residual is the
prediction stream. This changes the posterior precision blocks rather than relabeling one ridge.  
**Measured:** hierarchical blocks `0.002495923` versus uniform blocks `0.002961745`; pc trades the
other way (`0.05591` versus `0.02983`).  
**Verdict:** aggregate `INSTANCE` support for the separated formulation, with pc conflict unresolved;
SPS family remains open for engaged temporal telemetry.

### 6. HCM causal attribution / #472

HCM's hierarchical causal graphical model identifies certain **subunit-treatment** effects by
collapse/augment/marginalize plus do-calculus, while its unit-treatment theorem says hierarchy cannot
rescue a flat unidentified effect ([JMLR paper, Theorems 12–13](https://www.jmlr.org/papers/volume27/25-0899/25-0899.pdf)).

**Divergence:** one clip/run is one unit; epochs are subunits, but levers were not randomized inside
the unit.  
**Re-derivation:** Bayesian partial pooling is predictive only.  
**Built:** block posterior covariance plus explicit `causally_identified=false`.  
**Verdict:** predictive partial pooling `PROVISIONAL`; causal credit/self-activation `NOT IDENTIFIED`.

### 7. Grokking ridge bounds / #475

The ridge grokking proof separates fast training row-space modes from slow null-space/weight-decay
modes in realizable overparameterized GD; Theorems 4.4–4.6 and the appendix derive the delayed
generalization rates ([paper](https://arxiv.org/abs/2601.19791)).

**Divergence:** U is a closed-form solve over dependent intervals, not iterative GD on iid samples;
there is no legitimate “wait longer” clock.  
**Re-derivation/build:** posterior effective degrees of freedom (`1.1643` final), condition
(`5529.14`), and covariance are the honest eigensystem semantics.  
**Verdict:** late-generalization stage trigger remains a `FORMULATION` negative; eigen/precision
diagnostics transfer.

### 8. Continual-learning apparatus / #481

The paper defines a staged update operator that may alter context, weights, or memory—or do nothing—
and evaluates forgetting/forward transfer across distinct stages and mechanisms including GEPA,
SFT, RL, and test-time memory ([paper, §2–3](https://arxiv.org/abs/2607.07847)).

**Divergence:** one dependent trajectory supplies neither independent stages nor lineage/reward
custody.  
**Re-derivation:** this receipt can update an external posterior, but the organ is not promoted as a CL
system until at least three independent trajectory lineages.  
**Verdict:** `EXTERNAL_POSTERIOR_ONLY_UNTIL_3_INDEPENDENT_TRAJECTORIES`.

### 9. Spinning Up actor-critic / advantage / off-policy corpus / #473

The relevant corpus separates value baselines, advantages, actor updates, importance weighting, and
off-policy replay ([official key-papers index](https://spinningup.openai.com/en/latest/spinningup/keypapers.html)).

**Divergence:** the organ has no Markov policy, action propensities, counterfactual rewards, or
executed decision rows.  
**Re-derivation:** analytic score gradient plus T/persistence supplies the aggregate critic/baseline;
U learns a residual response field analogous to an advantage. The actor is disabled.  
**Measured:** constraining U to #436 reproduces #436's aggregate `0.001595939`, while U's pc
decomposition remains poor.  
**Verdict:** critic decomposition transfers; actor/off-policy claims are
`BLOCKED_DISTRIBUTION_CUSTODY`.

### 10. Requential Coding / inbox amendment

The exact method draws public proposals from the student's own generative distribution, uses the
teacher only at the encoder to accept an index whose marginal is the teacher distribution, and sends
a prefix-free universal code for that index. Expected length is cumulative teacher-student KL plus
logarithmic REC overhead; Appendix B controls realized-message fluctuations by a martingale variance
bound. EMA teacher smoothing and iso-loss projection shorten the trajectory. The valid prefix-free
code enters a PAC-Bayes bound, and increasing code debt under data repetition predicts overfitting
([paper, §3, Appendix A/B/D](https://arxiv.org/html/2607.11883)).

**Divergence:** our costate organ is a continuous point/posterior predictor, not a normalized
generative model with a shared-randomness proposal decoder.  
**Re-derivation:** under the declared shared-variance Gaussian posterior only,
`KL_bits = delta_S^2 / (2 variance_S ln 2)`. Protect half of each real row's replay mass, allocate the
other half proportional to past-prefix disagreement, cap at `2×`, and fit measured targets only.  
**Built:** `costate_requential_curriculum.py`, NumPy-fp32 weighted solve and MLX parity surface.  
**Measured:** disagreement replay `0.002463052` versus uniform `0.002495923`; pc improves `29.19%`
relative to uniform but is still `3.66×` persistence. Final post-birth Gaussian KL proxy is
`0.607560` bits; `4/7` evaluated variances hit the fp32 floor, the last interval contributes `68.02%`,
and late-debt slope is positive.  
**Verdict:** curriculum is a small `INSTANCE` improvement. The `0.607560` is **not** a REC code,
capacity floor, PAC-Bayes certificate, or validated overfit predictor. The rising late debt is a warning
only. A real capacity measure needs normalized organ proposal/teacher distributions, a decodable
prefix-free stream, non-floor variance custody, and independent validation.

**Rate handoff to main only (not built):** requential coding is the MDL parent of margin-conditional
flip residual coding: encode disagreement against the witness's own prediction rather than raw flip
entropy. This is the theory-compatible route from the measured ~8 bits/flip toward the published
~1–1.5 bits/flip floor; no rate claim is made by this organ receipt.

### 11. Meaningful Routing / HSE / inbox amendment

The paper forms actor behavior vectors over evaluation items, uses cosine distance, single-linkage
clustering at every threshold, and integrates Shannon cluster entropy to obtain HSE. Router robustness
is the fraction of meaning-preserving variants assigned to the original actor; a greedy max-HSE subset
exposes diminishing returns ([paper, Eq. 1–5](https://arxiv.org/pdf/2607.09197)).

**Divergence:** actors are organ mechanisms, evaluation items are seven WF folds, and variants are
one-ULP representations of the same regime state—not LLMs, prompts, or task accuracy.  
**Built:** `costate_society_diagnostics.py`. The actor society excludes the dispatcher itself; this
category error was caught in round-one review.  
**Measured:** full 11-actor normalized HSE is `0.04435` under the primary map and remains low under
reciprocal (`0.01738`) and fold-minmax (`0.13351`) representations. A/G/H/J distances from A are at
most `4.1e-8` under the primary map: those labels are behaviorally redundant on this trajectory. U
versus R-disagreement distance is `0.000861`. In contrast persistence versus T distance/HSE is
`0.108433`; the route-active society is differentiated. Diversity-only greedy max-HSE coreset is
`{T, A}` and is not a score recommendation. The operational route-active coreset is
`{persistence, T}`. Every actor is fold-wise Pareto-nondominated, so n=1 cannot collapse the
performance coreset further without a preference assumption.  
**Router stability:** 36/42 one-ULP assignments agree (`rho=0.857143`); `5/7` folds are fully stable,
while the exact-zero-margin folds at ep100 and ep150 have `rho=0.5`.  
**Net verdict:** #436 is **not inert theater**—its two actors are differentiated and its measured
aggregate WF is best. It is also **not robustly admitted**: exact-tie surface forms flip the route.
Queue an uncertainty/deadband or hysteretic tie formulation and remeasure; the current deterministic
tie law/replay certificate preserves reproducibility but does not create perturbation robustness.

**Fleet handoff only (not built):** HSE over fold/task behavior is a principled de-confliction
preflight for Codex fan-out; the recent redundant metric/basis triple-cover is the same failure class.

## Signal × EV ranking

There is no calibrated `P(real lever)` from one adaptive trajectory; inventing one would be fake. The
ranking therefore uses the observed fold-win fraction as an explicitly uncalibrated signal proxy,
then blast radius.

1. **#436 T/persistence aggregate dispatch:** `6/7` fold wins, `p=0.125`, `42.84%` mean improvement;
   highest aggregate blast radius. Existing lever, not newly owned; tie robustness debt.
2. **R disagreement curriculum:** `4/7`, `p=1.0`; only `1.317%` over uniform U, but it improves both
   aggregate and pc relative to uniform and directly targets n=1 information allocation.
3. **U hierarchical physics residual:** `4/7`, `p=1.0`; `10.60%` aggregate gain versus persistence and
   supplies a lever field/covariance, but pc harm and adaptive tuning prevent adoption.
4. **T GP alone:** prior measured `4/7`; strong aggregate mean, no per-lever field.
5. **HSE/support apparatus:** not a score lever, but high system EV: removes redundant arms and catches
   two exact-tie router folds.
6. **VR clipped difference:** `4/7`, `p=1.0`, but loses to un-clipped U; formulation queue only.
7. **FORE/TOFU/HCM actor/OPE:** no real-lever probability is estimable; custody gates are the result.

## Round-one adversarial review

- **Mechanism changes the organ:** hierarchical versus uniform precision changes aggregate WF by
  `0.000465822`; disagreement versus uniform replay changes aggregate by `0.000032871` and pc by
  `0.0163204`. These are not label-only arms.
- **No target leakage:** candidate selection and requential disagreements are computed from each outer
  training prefix. Focused mutation tests preserve a fold prediction when its held target changes.
- **No identifiability inflation:** `6/22` levers have observed share variation; fixed-prior signs are
  separately labeled. Support rank is not causal identification.
- **No borrowed score:** closed-U's aggregate equals the pre-existing #436 route by construction and is
  reported as composition, not a new U win.
- **No fake code theorem:** four Gaussian variance floors invalidate a capacity-floor claim; REC and
  PAC-Bayes names remain blocked.
- **No HSE representation monoculture:** redundancy and T/persistence differentiation persist under
  three bounded behavior maps. The exact normalized HSE value remains representation-conditional.
- **No router-as-actor category error:** corrected before final receipt. Stability is measured
  separately from actor diversity.
- **No naïve family kill:** the first-form catastrophe is INSTANCE-only; every negative below names a
  reformulation queue.

## Already-named but unfired probes

1. Independent-trajectory graduation (`>=3` lineages; `>=2` for #436 transfer boundary).
2. Frozen #434/meta encoder plus tiny Bayesian readout; no full n=1 network training.
3. Structured-GP residual/change-point U, evaluated with the same pc and aggregate gates.
4. Real FORE/OPE only after behavior/target propensities, visited-live densities, offered slates, and
   executed decisions are content-hashed.
5. Engaged SPS temporal stream; current zero-gradient instance is uninformative.
6. VR-GHAL only at a fixed repeated-query stochastic-operator locus.
7. Requential normalized Gaussian proposal/teacher distributions plus prefix-free decoder and
   non-floor variance; then test code debt on an independent run.
8. #436 deadband/hysteresis/uncertainty route at exact-zero margins, with the compiled OFF/current arm
   and real emitted telemetry compared before adoption.

## Held serial wire-in (provenance owner only)

- Factory: `costate_hierarchical_physics_residual_v1_spec()`
- LawRef: `costate_hierarchical_physics_residual_v1`
- Additional factory: `costate_requential_disagreement_curriculum_v1_spec()`
- Additional LawRef: `costate_requential_disagreement_curriculum_v1`
- Diagnostic factory/LawRef: `costate_mechanism_society_diagnostic_v1_spec()` /
  `costate_mechanism_society_hse_v1`
- Consumer: existing costate-organ tournament and #436 dispatcher, invoked after telemetry extraction
  by v9 autoconfig; never renderer/archive/inflate.
- Receipt: `costate_warmstart_cluster_backtest.v1`, with nested
  `costate_requential_curriculum_backtest.v1` and `costate_mechanism_society_diagnostic.v1`.
- Exact mathematical payloads: `warmstart_organ_n1_rl_HELD_EQUATION_SPECS_20260714.md`.
- Shared `canonical_equations/**`, `witness_dsl/**`, preflight, and v9 source closure remain held for
  `provenance_canonicalize_fix_all_fakes` after its scientific-declaration seal is stable.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; top-10 Claude memory; paper warm-start contract; latest Codex findings and
session summary; latest T3/design memos; `reports/latest.md`; lane registry; subagent progress;
master-gradient anchors; Modal ledger; cost/continual posteriors; blocking-outcomes/council helpers;
prior arm `*.last.txt` receipts; all listed prior DAG feeds; #205 daemon log; current backtest receipt;
full primary papers linked above; both live inbox files.

## Inbox cursor

Consumed per-arm directives through `2026-07-14T15:52:31Z` (Requential + HSE amendments) and
fleet-wide directives through `2026-07-14T15:56:40Z`. Later Bregman/provenance notices were respected
as shared-surface exclusions and did not alter this organ lane.
