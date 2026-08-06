# ddm_ffm1 - Functional Flow Matching Discretization Crosswalk

Date: 2026-08-06

Arm: `ffm1`

Status: COMPLETE, scorer-free, no launch, no paid dispatch, no score claim.

Paper: Lennon J. Shikhman, "Discretization and Statistical Consistency of Functional Flow Matching," arXiv:2608.04531v1, submitted 2026-08-05, 31 pages.

Sources read: arXiv abstract page `https://arxiv.org/abs/2608.04531` and arXiv HTML full text `https://arxiv.org/html/2608.04531`. No local PDF hash is claimed.

## Answer First

ffm1 does not adopt flow matching as a Pact model family. Its useful transfer is discretization discipline:

| Count | Grade | Rows |
|---:|---|---|
| 2 | ADOPT-CLASS | subset-gate strong-consistency design rule; Q3 projection-vs-conditioning endpoint prediction |
| 3 | LESSON-ONLY | finite-rank carrier refinement, Lipschitz/operator stability bookkeeping, Bernstein `n` scaling |
| 1 | N-A | training functional flow-matching generative models |

Highest-value row: **Q3 projection-vs-conditioning**. The paper gives a concrete noncommuting Gaussian example where projected restriction and exact conditioning disagree qualitatively. For Pact this is not a theorem, but it is strong enough to pre-register a `jd8q3_window` observable before endpoint harvest: Q3 should hold pose but may show a material seg-yield deficit versus jd7-OFF if linear projected-gradient descent is the wrong constrained target. Details are in `PREDICTIONS.md`.

No frozen scorer forward, no `upstream/evaluate.py`, no n600 job, no launch, no paid dispatch, and no protected-file edit were performed. `score_claim=false` for every row below.

## Continuation State

The predecessor left no ffm1 receipt in the expected directory. This arm:

- read the charter and common contract;
- ran `git log --oneline -15`;
- found no `ffm1` commit in those 15 commits;
- found no existing `.omx/research/ddm_ffm1_20260806/` before checkpoint creation;
- ran bounded filename/content searches for `ffm1`, `2608.04531`, and `Functional Flow Matching` under `.omx/research`, `.omx/tmp`, and `.omx/state`;
- committed the first continuation checkpoint as `d1af9f16d7`.

The shared worktree was already very dirty with unrelated changes. This arm touched only `.omx/research/ddm_ffm1_20260806/`.

## Paper Deep Read

### Setting And Assumptions

The paper works in a separable Hilbert space `H`. A stochastic interpolation `X_t` is almost surely absolutely continuous, with Bochner velocity `U_t`, square-integrable initial state, and square-integrable path velocity. The continuum flow-matching target is the conditional velocity

```text
v*(tau, X_tau) = E[U_tau | sigma(tau, X_tau)].
```

Finite observations are modeled by finite-rank reconstruction operators `A_m` with finite-dimensional range, `A_m x -> x` strongly for every `x`, and therefore uniformly bounded operator norms. The finite target is

```text
v_m*(tau, A_m X_tau) = E[A_m U_tau | sigma(tau, A_m X_tau)].
```

Point sensors require extra structure. The paper introduces a regularity space `V` densely and continuously embedded in `H`, sensor/reconstruction maps `S_m`, `R_m`, and `A_m = R_m S_m`. It requires bounded reconstruction into `H`, pointwise reconstruction convergence for each `f in V`, and same-information sigma-algebras between sensor values and reconstructions. Sobolev `H^s(D)` with `s > d/2` is the canonical point-evaluation example.

### Main Results Used

1. **Conditional-target consistency.** Strongly consistent finite-rank reconstructions imply `v_m*(tau,A_m X_tau) -> v*(tau,X_tau)` strongly in normalized `L2`, even when the observation sigma-algebras are not nested. This is the key anti-martingale result.

2. **Orthogonal quantitative bound.** For orthogonal projections and a Lipschitz continuum target, the finite-target error is bounded by the unresolved tail of the velocity target plus the unresolved tail of `X_t`. This is useful as a shape theorem, not as a Pact metric.

3. **Point-sensor finite-information limit.** No finite observation reconstructs the whole `L2` unit ball uniformly. Pointwise convergence on regular functions must be backed by a class-dependent reconstruction estimate. This is the paper's closest formal cousin to our prefix/subset-gate caution.

4. **Flow consistency.** Target convergence alone gives reconstructed marginal-law convergence. Deterministic flow-map convergence needs mesh-uniform ODE stability. The paper explicitly separates measure-valued superposition from unique population ODE flow maps.

5. **Uniform learning constants.** A normalized quadrature neural operator gets sensor-independent magnitude, parameter, and state constants if mass equivalence, normalized quadrature weights, globally Lipschitz activations, and explicit magnitude recurrences hold.

6. **Statistical bound.** For fixed parameter dimension and uniform envelopes, Bernstein localization gives excess risk of order `~O(n^-1)`, and in the realizable case this contributes `~O(n^-1/2)` to the generated-law bound.

7. **Noncommuting Gaussian example.** In a trace-class Gaussian model, projecting the continuum conditional field onto a boundary observation gives multiplier `0`, while exact finite conditioning gives `0.72`; finite-sample regressions converge to the nonzero value. The discrepancy does not contradict target consistency because the affected direction moves into the tail.

### Scope Boundary

The paper is about probability-flow discretization under reconstruction, realization, statistical, and stability assumptions. It does not justify arbitrary refinement, unseen-frequency recovery, point-sensor convergence without regularity, or learned-flow adoption in Pact. It controls laws, not contest archive bytes.

## Recall Evidence

| Query/source | Evidence found | What changed |
|---|---|---|
| `ffm1`, `2608.04531`, `Functional Flow Matching` in `.omx/research`, `.omx/tmp`, `.omx/state` | Queue/spawn rows and charter only; no prior ffm1 receipt found in this scope. | Treated this as a cold durable continuation, created receipt dir and checkpoint commit. |
| Prior paper drops `am1`, `bn1`, `cf1`, `sd1`, `stl1`, `vae1`, `cl1`, `coe1` | AM1 already owns measure-transport/acceleration-packet ideas; CF1 already owns conformal calibration/exchangeability; BN1 owns weakest-sufficient constraint framing; SD1 owns decoder-side CPWL packet grammar; STL/VAE/CL1/COE1 own unrelated estimator/rate/apparatus surfaces. | Narrowed ffm1 to discretization consistency and projection-vs-conditioning; no duplicate equation or transport row. |
| `.omx/state/main_hot_state.md` | Live `jd8q3_window` read is pending and explicitly compares Q3 seg descent and pose give-back against jd7-OFF. | Row 3 became time-sensitive and produced `PREDICTIONS.md`. |
| `src/tac/subset_selection.py`, `src/tac/subset_selection_gate.py`, `src/tac/canonical_anti_patterns/na3_subset_bias_builders.py` | Local code already encodes subset/prefix-bias anti-pattern discipline; NA3/NA4/CF1 receipts give axis-dependent prefix bias. | Row 1 is ADOPT-CLASS as a design rule, not a new implementation request. |
| `src/tac/losses/variable_level_waterfill_allocator.py` and token/waterfill receipts | Pact already prices coefficient choices by actual task-score/rate exchange rates, not Hilbert `L2` projection order. | Row 2 is LESSON-ONLY with a nonnested rebase warning. |
| Bounded canonical-equation scan via `tools/list_canonical_equations.py --json | head -c 20000` plus targeted text searches | Existing registry is broad; no ffm-specific equation was identified in the bounded scan. | No canonical equation registered. |

Bounded absence statement: this is not a global claim that no related work exists; it is a claim about the scopes and queries listed above.

## Ranked Crosswalk

| Rank | Surface | Grade | Honesty | Pact adjudication | Named consumer | Falsifier |
|---:|---|---|---|---|---|---|
| 1 | Q3 projected restriction vs exact constrained conditioning | ADOPT-CLASS | DERIVED analogy plus CONJECTURE on endpoint sign | The paper proves that projection and conditioning can fail to commute. Pact's Q3 is a linear projected-gradient method; exact constrained descent on the pose-null, uint8/R-surviving manifold is the conditioning analogue. Register the endpoint observable before harvest. | `jd8q3_window` endpoint read; `--seg-grad-q3-project` descendants | Q3 retains at least 90% of jd7-OFF seg descent while holding pose under the registered threshold in `PREDICTIONS.md`. |
| 2 | Non-nested sensor sigma-algebras vs subset-gate bias | ADOPT-CLASS | DERIVED | The paper's strong-consistency condition becomes a gate-design rule: a subset instrument is trustworthy only if it is a bounded, declared reconstruction of the target population and shows convergence under refinement. Prefix gates fail this unless separately calibrated by axis. | `src/tac/subset_selection_gate.py`, `src/tac/subset_selection.py`, NA3/NA4-style negative audits, OD9-style stratified projection gates | Banked n600 rows show a prefix or unweighted gate is unbiased for d_seg, d_pose, and rate under the same axis and vehicle, with stable errors across random/stratified refinements. |
| 3 | Finite-rank reconstruction consistency vs coefficient/token families | LESSON-ONLY | DERIVED | Orthogonal Hilbert `L2` bounds do not order our nonorthogonal, scorer-weighted tokens, lane coefficients, phase-field coefficients, or pose residuals. Pact's measured score-per-byte waterfill is stronger for fixed budget. The warning is procedural: when a menu is rebased mid-family, treat it as a new reconstruction sequence and reprice margins rather than appealing to nested/incremental acceptance. | tq1/tq1c token waterfill receipts; `src/tac/losses/variable_level_waterfill_allocator.py`; `src/tac/optimization/direct_description_minimizer.py` | A Hilbert projection order predicts actual task-score/rate marginal ordering better than same-object measured waterfill on a receiver-closed candidate family. |
| 4 | Quadrature neural-operator Lipschitz constants vs hosc/step activation stability | LESSON-ONLY | DERIVED | The magnitude recurrence is standard useful bookkeeping. It applies to globally Lipschitz activations, mass-equivalent nodal norms, and normalized quadrature. It does not certify our high-beta hosc/step behavior or replace the already measured fixed-beta divergence and annealed-survival history. | Future activation-stability memos only; no launch consumer | A Pact activation family satisfies the paper's bounded recurrence with sensor-independent constants and predicts a stability decision not already visible from our measured DE/receiver stability laws. |
| 5 | Bernstein `~O(n^-1)` excess risk vs gate noise floors | LESSON-ONLY | DERIVED | The paper's rate is for i.i.d. training samples, fixed parameter dimension, bounded envelopes, and excess risk. It does not justify scaling prefix gates to n600 when the subset is a different population. At most, for unbiased random sampling, standard error scales like `sqrt(600/36)=4.08` going from n600-sized accounting to n36; variance scales by 16.67. Prefix bias can dominate either number. | gate-noise receipts such as tq1b/UF1; CF1 alarm-calibration consumers | A calibrated random/stratified gate with fixed estimator and no population shift shows empirical noise scaling inconsistent with finite-population sampling but consistent with the paper's iid excess-risk bound. |
| 6 | Functional flow-matching model adoption | N-A | DERIVED | Pact does not train flow-matching generative models in this arm. The paper does not alter contest archive scoring, rule-118 payload accounting, or exact evaluator authority. | None | A future Pact branch explicitly uses finite-observation flow matching as the receiver-closed archive vehicle and charges every learned/video-derived payload byte. |

## Surface 1 - Subset-Gate Strong Consistency

The direct theorem assumes `A_m x -> x` strongly and uniformly bounded reconstruction operators. Pair-index subset gates are not literally Hilbert reconstructions unless we define a population function over pair index and a reconstruction rule. That prevents a direct theorem transfer.

The adoptable class is still concrete:

1. Declare the target population and axis before sampling.
2. Declare the reconstruction/estimator: stratified HT, seeded random mean, block bootstrap, or other typed operator.
3. Require bounded weights and no single block with uncontrolled leverage.
4. Show refinement consistency using banked n600 receipts: n32/n60/n120 should converge toward the n600 axis value under the same estimator.
5. Report axis separately because pose-prefix and seg-prefix bias can have opposite signs.

This is why OD9-style stratified projection and CF1 exchangeability discipline are the local descendants, while prefix gates remain advisory unless calibrated.

$0 falsifiable check: on banked n600 receipts with per-pair or per-block components, build a typed gate replay that compares prefix, strided, seeded-random, and stratified estimates for d_seg, d_pose, and rate. Require the stratified/seeded estimator's confidence band to cover n600 and require prefix to pass by axis before any prefix gate is called trustworthy. If per-pair components are absent, record `BLOCKED_NO_PER_PAIR_COMPONENTS`, not a theorem claim.

## Surface 2 - Coefficient And Token Refinement

The paper gives quantitative bounds for orthogonal projections under a Lipschitz target. Our live descriptions are not orthogonal Hilbert projections:

- TR1 tokens are learned categorical fields and dominate archive bytes.
- rank-4 heads, rank-1 pose residuals, lane polynomials, and phase-field coefficients are nonorthogonal parameterizations.
- the objective is scorer-space `100*d_seg + sqrt(10*d_pose) + rate`, not raw `L2`.

Therefore the theorem does not decide which coefficient to add next. It does warn against pretending a rebased nonnested menu is a martingale or nested projection. tq1c already did the right thing by making the new base explicit (`b35e7568`) and queuing resume at the next menu index; future rows should keep repricing against the current same-object base.

## Surface 3 - Q3 Projection Warning

The row is registered in `PREDICTIONS.md` because the endpoint is live. The analogy is:

| Paper object | Pact analogue |
|---|---|
| continuum conditional field | raw seg-gradient field |
| projected restriction onto observation direction | `--seg-grad-q3-project` linear projected gradient |
| exact conditioning after finite observation | constrained seg descent on pose-null, uint8/R-surviving manifold |
| noncommuting covariance | seg/pose/R/uint8 coupling, including sq1 integer leakage and curvature |

The paper proves existence of a projection-conditioning gap in a Gaussian Hilbert model. Pact only gets a conjectural diagnostic: if Q3 holds pose but under-retains seg yield, then linear projection is probably not the final constrained-descent form. If Q3 preserves seg yield while holding pose, this paper's caution is folded for the current Q3 surface.

## Surface 4 - Stability Constants

The quadrature section validates a familiar certificate: normalized weights, mass norm equivalence, bounded operators, globally Lipschitz activations, and magnitude recurrences prevent sensor-count constants from exploding. Pact should not over-read it. Our dangerous activations are dangerous because the constants are large, stage-dependent, or not globally tame. The paper is useful as a checklist paragraph, not as a launch gate or replacement for measured stability.

## Surface 5 - `n` Scaling

The Bernstein rate answers a different question from our gate-noise budget. It is about learning a bounded fixed-dimension class from iid samples. Our gate estimates are finite-population measurements over 600 ordered pairs with axis-specific skew. If the gate is random and exchangeable, standard sampling arithmetic can be used; if it is prefix or adaptive, the paper's own nonnested warning says the sample can be a different population.

Short quantitative answer: do not multiply tq1b's `~7e-7 S` noise floor by a paper rate. For a same-population random n36 estimator, a crude independent standard-error multiplier against n600 is `sqrt(600/36)=4.08`; for variance it is `600/36=16.67`. Prefix bias can be larger and sign-changing, so measured correction ratios still dominate.

## AM1 vs ffm1 Diff

AM1 was transport-flavored but about trajectory representation: acceleration/control packets, smooth `xi(t)` streams, OD8 persistence, and direct receiver equality. Its useful rows were packet-pricing tests and smoothness priors.

ffm1 is not another transport-packet row. Its useful rows are about what finite observations mean: nonnested conditioning, reconstruction consistency, projection-vs-conditioning, and when sample-size/statistical claims are legitimate. The only live overlap is philosophical: both warn that a continuum or smooth object is not enough unless the finite receiver/measurement surface is the same object. They do not share a consumer, so no AM1 row is double-counted here.

## Follow-Ons

| Status | Item | Fire order |
|---|---|---|
| FIRED | `ffm1_q3_projection_conditioning_prediction` | `PREDICTIONS.md` is written before this arm observed `jd8q3_window`. Consume it at endpoint harvest. |
| QUEUED-WITH-FIRE-ORDER | `ffm1_subset_gate_strong_consistency_replay` | When touching subset-gate calibration next, replay banked n600 per-pair/per-block rows through prefix, strided, seeded-random, and stratified estimators; report axis-specific convergence and bias. |
| FOLDED | `ffm1_flow_matching_training_adoption` | No action; out of scope for the live Pact vehicle. |
| FOLDED | `ffm1_lipschitz_activation_launch_gate` | No action; use as a checklist only. |
| FOLDED | `ffm1_noise_floor_scaling_by_paper_rate` | Do not scale gate noise floors from the paper's iid excess-risk rate. Use measured finite-population/gate statistics. |

## Boundaries

Measured in this unit: no scorer values, no archive bytes, no d_seg, no d_pose, no runtime, no contest score.

Derived in this unit: theorem-level crosswalk, ranked dispositions, and a pre-registered Q3 endpoint observable.

Not done: no local PDF hash, no arXiv source archive hash, no code implementation, no canonical-equation registration, no launch, no paid dispatch, no exact eval.

Own-vehicle frontier line remains as read from `.omx/state/main_hot_state.md`: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; contest pointer borrowed/unmoved.
