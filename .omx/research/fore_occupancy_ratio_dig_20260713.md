---
title: "FORE occupancy-ratio drift bridge for frozen replay and the costate organ"
date_utc: "2026-07-13"
lane_id: "lane_fore_occupancy_ratio_dig_20260713"
research_only: true
status: "DESIGN_ANALYSIS"
drift_bridge_verdict: "NO_GO_CURRENT_INSTANCE__CONDITIONAL_FORMULATION_OPEN"
organ_estimator: "DERIVED__CURRENT_LOGGED_POLICY_GATE_NOT_IDENTIFIED"
score_claim: false
pointer_delta: "NONE"
---

# FORE occupancy-ratio dig: replay-to-live drift bridge, organ OPE, and rate-law composition

## Answer first

1. **Drift bridge: `NO-GO` for the live round-2 cache; `CONDITIONAL GO` as a repaired
   formulation.** The correct weight orientation is
   `omega_current/replay = d_current,gamma / d_nu_replay`, and for any fixed surrogate head
   parameter `beta`,
   `L_current(beta) = E_nu[omega_current/replay * ell_beta]`. That identity is exact when the
   Radon--Nikodym derivative exists. The current frozen-replay arm does not supply the data that
   identify it: its 600 examples are isolated render/costate states assigned across three cold
   checkpoints, not full Markov optimizer state-action-next-state transitions, and they do not
   establish one-step target coverage or current-trajectory support. Its existing `GO` remains
   valid on its registered frozen-distribution axis; FORE does not retrospectively promote it to
   on-policy authority.

2. **Deterministic dynamics do not break the discounted KL contraction.** For `gamma < 1`, the
   FORE adjoint map satisfies
   `D_nu(B_gamma^pi omega || B_gamma^pi omega_tilde) <= gamma D_nu(omega || omega_tilde)`.
   The proof uses joint convexity plus data processing for a Markov kernel; a deterministic map is
   a Markov kernel. Strictness comes from the common `(1-gamma)d0` reset component, not transition
   noise. At `gamma=1`, ordinary data processing gives only nonexpansiveness. A deterministic
   injective optimizer map normally has contraction coefficient one, so strict undiscounted
   contraction would need an additional strong data-processing/mixing hypothesis that is absent.

3. **Costate-organ estimator: `DERIVED`; present causal backtest admission: `NO-GO`.** For a logged
   schedule policy, FORE yields the direct normalized-discounted estimator
   `V_hat_FORE(pi) = n^-1 sum_i omega_hat_pi(Z_i,A_i) R_i`, with the doubly robust correction also
   available. Only-ratio-realizability removes adjoint Bellman completeness and a separate critic
   class. It does not remove Markov sufficiency, common conditional dynamics/rewards, initial and
   one-step coverage, target-action positivity, or cross-run support. The current organ records one
   regime sequence with essentially deterministic schedule decisions; that is enough for its stated
   walk-forward forecasting gate, but not for causal off-policy evaluation of unlogged schedule
   arms on new runs.

4. **Composition with rate-law rung #468: outer change of measure, no new chain-rule term.** FORE
   may reweight the expected marked-chain codelength across training/control occupancies. It does
   not factor the marked conditional density, imply a conditional independence, or add a payload
   section. For the decoded witness bitstream itself, the optimizer occupancy ratio slots nowhere.

No claim in this memo is an evaluator score, a live-activation authorization, or a pointer move.

## 0. Authority, labels, and source boundary

### Primary paper

The paper was read from the official arXiv abstract, HTML, and 57-page PDF:

- Lars van der Laan and Nathan Kallus, *Fitted Occupancy-Ratio Evaluation without Bellman
  Completeness*, arXiv:2607.05375v1, 6 July 2026:
  [abstract](https://arxiv.org/abs/2607.05375),
  [HTML](https://arxiv.org/html/2607.05375),
  [PDF](https://arxiv.org/pdf/2607.05375).
- Paper anchors used here: occupancy definition and adjoint identity (equations 1--4), KL
  contraction (Lemma 3.1), single-level KL projection (Lemma 3.2 and Algorithm 1), only-ratio
  realizability (Theorem 4.1), finite-sample conditions (Theorem 4.2), target-functional
  reweighting (Corollary 5.1), doubly robust evaluation (Theorem 5.2), and the undiscounted strong
  data-processing qualification (Appendix E).

### Claim labels

- `SOURCE`: stated by the paper or a named repository artifact.
- `DERIVED`: follows algebraically from stated assumptions.
- `MEASURED-INHERITED`: copied from a sealed named receipt; not remeasured in this unit.
- `INFERRED`: a mapping from the theorem to Pact that still needs an empirical or schema gate.
- `ASSUMED-TICKET`: a proposed A/B choice that has not been executed.
- `UNKNOWN`: not identified by current evidence.

### Proactive recall honored

- `#455`: the live nonlinear on-policy surrogate failed by operator/target drift after short reuse;
  this was not re-derived.
- `#462`: the family stays open only for an explicit contraction with admissible hypotheses; this
  was used as the admission standard.
- Live round 2: the frozen-replay convex head is a narrow fixed-distribution `GO`, with
  `MEASURED-INHERITED` 600 unique states, 480/120 split, three cold checkpoints, 12x inclusive
  teacher-call amortization, and an executed-fp32 head contraction. Its files were read only.
- `#426/#431/#436`: the organ's walk-forward results are offline, instance-scoped, and
  provisional-until-accrual; no causal cross-run policy claim was imported.
- `rate_law_ladder_v1/#468`: the marked conditional chain rule is settled as a temporal source
  decomposition; it was composed, not re-derived.

## 1. What FORE actually assumes

Let `X=(S,A)` be a state-action variable, `nu` an offline state-action distribution, `pi` a target
policy, and `P_pi` the state-action transition kernel obtained by applying the environment kernel
and then drawing the next action from `pi`. The paper defines

\[
d_{\pi,\gamma}
  =(1-\gamma)\sum_{t\ge 0}\gamma^t d_0P_\pi^t,
\qquad
\omega_{\pi,\gamma}=\frac{d d_{\pi,\gamma}}{d\nu}.
\tag{1}
\]

The orientation matters: the target/current occupancy is the numerator and replay is the
denominator. The occupancy recursion and its adjoint density map are

\[
d_{\pi,\gamma}=(1-\gamma)d_0+\gamma d_{\pi,\gamma}P_\pi,
\qquad
\mathcal B_\gamma^\pi\omega
  =(1-\gamma)\omega_0
  +\gamma\frac{d\{(\omega\nu)P_\pi\}}{d\nu},
\quad
\omega_0=\frac{d d_0}{d\nu}.
\tag{2}
\]

For a normalized exponential ratio class

\[
\mathcal W
 =\{\omega_h(x)=\exp(h(x)-\Lambda_\nu(h)):h\in\mathcal H\},
\qquad
\Lambda_\nu(h)=\log E_\nu[e^{h(X)}],
\tag{3}
\]

one projected step is obtained by the single-level loss

\[
h_{k+1}\in\arg\min_{h\in\mathcal H}
\left\{
\Lambda_\nu(h)
-(1-\gamma)E_{d_0}[h(X)]
-\gamma E_\nu[\omega_k(X)h(X^+)]
\right\},
\quad X^+\sim P_\pi(\cdot\mid X).
\tag{4}
\]

`SOURCE`: FORE does not need an explicit behavior policy. It does need offline state-action
samples, their one-step environment successors, the ability to form the target-policy next action,
and the following non-negotiable assumptions:

1. `d0 << nu` and `nu P_pi << nu` (one-step target coverage);
2. `gamma < 1` for the main strict contraction;
3. a closed convex normalized log-ratio class with a well-defined KL information projection;
4. target ratio positivity and log-integrability on the working support;
5. bounded centered log class and bounded initial/one-step density ratios for the stated bounds;
6. for the finite-sample theorem, iid offline transitions, an independent initial sample, exact ERM
   per fitted step, and a uniform positive lower bound on the target ratio.

Only item 3's *function-class closure burden* is weakened to realizability/approximation of the
fixed-point ratio itself. The other items do not disappear.

## 2. Precise optimizer-as-MDP typing

The prompt's shorthand is useful but needs a type correction.

| RL object | Correct witness-training object | Why the narrower shorthand fails |
|---|---|---|
| state `Z_t` | Full crash-resumable training state: witness parameters/codes, optimizer moments, surrogate/head parameters, RNG and data-order state, stage/epoch/clock, controller memory, active typed-DSL configuration, relevant receiver/scorer fingerprints, and any history needed by a schedule rule | A rendered RGB state alone does not determine the next optimizer state and is not Markov. |
| action `A_t` | Schedule/control decision: batch/pair selection, exact-vs-surrogate teacher mode, stage/rung/lever choice, learning-rate/optimizer choice, or a fully typed update command | Calling the optimizer step map the policy conflates action selection with transition dynamics. |
| policy `pi(A|Z)` | The frozen schedule/optimizer program that selects an action from the full state | For an actionless formulation, this can be absorbed into a Markov-reward kernel; then `P_pi` is the optimizer step map. |
| transition `P(dZ'|Z,A)` | Application of the selected update plus stochastic batch/RNG/environment effects | It may be deterministic. Determinism is not itself a theorem failure. |
| target occupancy `d_pi,gamma` | Geometrically discounted distribution of full training state-actions visited by the current frozen program | A distribution over rendered pair states alone aliases optimizer/controller state. |
| offline distribution `nu_R` | A fixed replay distribution over full logged `(Z,A,Z')` transitions | Three isolated cold render states are not one-step transitions. |
| reward/cost `g(Z,A)` | Exact teacher risk, directional costate loss, evaluator-cell debt, or a typed schedule reward, depending on the declared question | The reward definition cannot change after seeing the comparison. |

`DERIVED`: an optimizer whose parameters evolve can still define a *time-homogeneous* Markov
kernel if those parameters, the clock, optimizer state, and controller memory are included in `Z`.
Thus, "the trajectory moves" is not by itself a proof that FORE is inapplicable. The actual
questions are whether the augmented state is Markov, whether the target program is held fixed for
the fitted recursion, and whether target occupancy is covered by replay.

Two formulations remain mathematically possible:

1. **State-action MDP (preferred for the organ):** schedule choice is explicit and supports policy
   comparison when logged action coverage exists.
2. **Actionless Markov reward process (possible for one fixed optimizer):** absorb the deterministic
   optimizer program into `F`, so `Z_{t+1}=F(Z_t)`. This cannot evaluate alternative schedule arms
   without returning to the state-action typing.

These are not interchangeable in a receipt. A future probe must declare which is used.

## 3. The drift-bridge identity and contraction

For the convex head `g_beta` and exact costate target `lambda(Z)`, define a fixed-state loss

\[
\ell_\beta(Z,A)
 =\|g_\beta(\phi(Z,A))-\lambda(Z,A)\|_2^2
   +\lambda_{\rm ridge}\|\beta\|_2^2.
\tag{5}
\]

If `d_pi,gamma << nu_R`, then for every fixed `beta` with integrable loss,

\[
\boxed{
\mathcal L_{\pi,\gamma}(\beta)
=E_{d_{\pi,\gamma}}[\ell_\beta]
=E_{\nu_R}[\omega_{\pi,\gamma}\ell_\beta],
\qquad
\omega_{\pi,\gamma}=\frac{d d_{\pi,\gamma}}{d\nu_R}.
}
\tag{DRIFT-BRIDGE-CANDIDATE}
\]

`DERIVED`: with nonnegative fixed weights, a linear/ridge head remains a convex optimization
problem. Its Hessian changes to the weighted covariance, so round 2's measured `1/3`-scale head
contraction cannot be inherited; it must be re-derived from the realized weighted Hessian.

### 3.1 Why deterministic transitions preserve the discounted KL contraction

For any normalized ratios `omega, omega_tilde`, the measures corresponding to the next adjoint
images are

\[
(\mathcal B_\gamma^\pi\omega)\nu
=(1-\gamma)d_0+\gamma(\omega\nu)P_\pi,
\quad
(\mathcal B_\gamma^\pi\tilde\omega)\nu
=(1-\gamma)d_0+\gamma(\tilde\omega\nu)P_\pi.
\]

Joint convexity and data processing give

\[
\begin{aligned}
D_\nu(\mathcal B_\gamma^\pi\omega
\|\mathcal B_\gamma^\pi\tilde\omega)
&\le \gamma D_{\rm KL}((\omega\nu)P_\pi
\|(\tilde\omega\nu)P_\pi)\\
&\le \gamma D_{\rm KL}(\omega\nu\|\tilde\omega\nu)
=\gamma D_\nu(\omega\|\tilde\omega).
\end{aligned}
\tag{6}
\]

A deterministic transition is the kernel `P_pi(dz'|z)=delta_{F_pi(z)}(dz')`; data processing still
holds. No transition noise appears in (6). The strict factor is the discount mixture.

`DERIVED adversarial boundary`:

- At `gamma=1`, the common reset term vanishes and (6) becomes nonexpansive at best.
- If `F_pi` is injective on the compared supports, KL can be preserved exactly, so there is no
  uniform strict contraction.
- Appendix E of the paper recovers an undiscounted strict factor only under a strong KL
  data-processing condition such as a Doeblin minorization. A deterministic nonconstant optimizer
  map does not generally satisfy such a mixing condition.
- Choosing `gamma<1` merely to obtain contraction changes the target objective. A uniform finite
  training-horizon objective is not silently equal to a geometric occupancy. An exact finite-horizon
  use must declare the time law (for example, an augmented clock with a preregistered stopping law)
  and prove that the resulting target functional is the one the A/B claims to optimize.

### 3.2 Only-ratio realizability, precisely

Under the paper's class and coverage conditions,

\[
D_\nu(\mathcal T_{\mathcal W}^{\rm KL}\omega
\|\omega_{\pi,\gamma})
\le
\gamma D_\nu(\omega\|\omega_{\pi,\gamma})
+C_{\rm app}\epsilon_{\rm KL}^2,
\quad
\epsilon_{\rm KL}
=\inf_{v\in\mathcal W}
\|\log\omega_{\pi,\gamma}-\log v\|_{L^2(\nu)}.
\tag{7}
\]

If the true ratio is realizable, `epsilon_KL=0`; the class need not contain every adjoint Bellman
image. This is the genuine FORE gain. It is a statement about the ratio recursion, not a proof that
the coupled witness/head update map contracts and not a waiver of coverage.

## 4. Why the live round-2 cache does not identify the bridge

The current instance fails before ratio-class approximation is the crux.

1. **No Markov transition tuples (`SOURCE`).** The round-2 cache assigns one rendered pair state to
   one of checkpoints 150, 251, or 275. It stores feature/target sufficient statistics and heldout
   reductions, not consecutive full `(Z,A,Z')` optimizer transitions. Checkpoint gaps are not one
   step, and the 600 pair assignments are not 600 optimizer trajectories.
2. **State is not transition-sufficient (`SOURCE + DERIVED`).** Its 31-feature chart is an excellent
   fixed supervised chart for the registered probe but omits optimizer moments, surrogate/head state,
   RNG/data order, controller memory, and typed schedule action. Therefore `P_pi` is not identified
   on that chart.
3. **Target successor moment is unavailable (`DERIVED`).** FORE step (4) needs
   `E_nu[omega_k(X)h(X+)]` for target-policy successors. The frozen cache has no `X+` with the
   required full-state typing.
4. **Coverage is unproved and likely singular (`INFERRED`).** With continuous high-dimensional
   states and a deterministic optimizer, a current trajectory may be a set of atoms/disjoint
   manifolds not visited by the three old trajectories. If `d_current(B)>0` for a set with
   `nu_R(B)=0`, the ratio does not exist. In the other direction, replay-only states give
   `omega_current=0`; although a Radon--Nikodym derivative can have zeros, this violates the
   paper's positive log-ratio and finite-sample lower-bound hypotheses on the working support.
   Coarsening states until paths overlap would hide, not solve, the Markov failure.
5. **Action support is absent for schedule comparison (`DERIVED`).** If old logs choose action
   `a_old(z)` deterministically and the current policy chooses `a_new(z) != a_old(z)`, the logged
   state-action distribution has zero mass on the target action. FORE needs no named behavior
   policy, but it still needs state-action coverage.
6. **The theorem's contraction is not the head's live contraction (`DERIVED`).** Equation (6)
   contracts ratio distributions for a fixed target kernel. It does not establish nonexpansiveness
   of the live joint witness/surrogate update that failed in #455.
7. **Inference-time feature custody remains separate (`SOURCE`).** Round 2 itself records that
   source-label/margin features require custody before any live interpretation. Reweighting does not
   solve that issue.

**Drift-bridge verdict:** `NO-GO`,
`verdict_scope=FORMULATION x CURRENT ROUND-2 INSTANCE`: the existing three-checkpoint, one-state-
per-pair frozen cache used as if it identified a full optimizer occupancy ratio. This does not kill
FORE, deterministic optimizer MRPs, occupancy reweighting, frozen replay, convex heads, or a future
transition-complete buffer.

Because the bridge's admission hypotheses are false/unproved on current bytes, no canonical-equation
module is registered in this unit. `DRIFT-BRIDGE-CANDIDATE` is deliberately
`FORMALIZATION_PENDING`, not a canonical law. Main review may register it only after the transition
schema and coverage receipt exist.

## 5. Repaired round-2+ formulation and addressed A/B ticket

### To: `replace_round2_convexhead` / main review

Your fixed-distribution `GO` is not challenged. FORE is a *successor bridge*, not a reinterpretation
of your receipt. Do not add weights to the sealed receipt or claim its 600 cached labels are
on-policy. The smallest admissible successor is the following stage-frozen design.

### 5.1 Stage-boundary schedule (`ASSUMED-TICKET`; not executed)

1. **Declare the target functional.** Choose and justify the time law before data inspection:
   geometric `gamma<1`, or a separate finite-horizon formulation. Record the target risk and
   `gamma`; no bare numeric value is proposed here.
2. **Freeze target program `pi_j`.** At a stage boundary, freeze the complete typed schedule,
   optimizer rule, surrogate version, feature schema, source hashes, and target initial
   distribution. No per-step policy mutation during a ratio fit.
3. **Build/identify a full transition corpus.** Each replay row must contain a full resumable
   `Z_i`, logged action `A_i`, environment successor `Z'_i`, target-policy next action or its exact
   expectation, reward/label hashes, run/hardware/code fingerprints, and source checkpoint custody.
   Existing isolated round-2 rows cannot be upgraded by metadata assertion.
4. **Fail closed on coverage.** Verify target initial and one-step absolute continuity on the
   declared working support; record target-action support/propensity, finite weight bounds, and
   support exclusions. Deterministic zero-support rows refuse the arm. No clipping is allowed unless
   its bias is separately bounded and preregistered.
5. **Fit the ratio.** Run the single-level objective (4) on a fixed normalized log-ratio class.
   Every fitted-ratio iteration would be an atomic, preserved, resumable stage checkpoint if this
   ticket is later built. Fit/validation splits and stopping rule are frozen before labels are read.
6. **Freeze `omega_hat_j`; then fit the head.** Use `omega_hat_j` as a fixed nonnegative weight in
   the convex ridge objective. Re-derive `mu`, `L`, step size, and the executed operator norm from
   the weighted Hessian; do not inherit the round-2 certificate.
7. **Measure on a target-policy holdout.** Compare against exact labels on independent transitions
   from the frozen `pi_j` occupancy. Report target-weighted head loss, exact costate/renderer-gradient
   direction, effective sample size, maximum weight, normalization error, heldout FORE objective,
   and all teacher calls. Costate proxy movement alone is not evaluator-cell authority.
8. **Advance only at the next stage boundary.** If the witness program changes to `pi_(j+1)`, the
   old ratio becomes a lagged object. Refit or bound the change of measure before reuse. Loss weights
   do not adapt per step.

For a bounded diagnostic `g`, deployment under a later program has the explicit drift debt

\[
|E_{d_{\pi_{j+1}}}g-E_{\nu_R}[\hat\omega_jg]|
\le
|E_{d_{\pi_j}}g-E_{\nu_R}[\hat\omega_jg]|
+2\|g\|_\infty\operatorname{TV}(d_{\pi_{j+1}} , d_{\pi_j}).
\tag{8}
\]

The second term is not paid by FORE convergence to `pi_j`.

### 5.2 Teacher-call custody

`MEASURED-INHERITED` from round 2:

\[
A=600,\qquad D=480\times15=7200,\qquad c_{\rm label}=0,
\qquad C_{\rm teacher}=A+c_{\rm label}D=600.
\tag{9}
\]

`DERIVED`: the FORE ratio loss itself uses initial/transition moments, not SegNet costate labels, so
reweighting the exact same cached labeled states adds zero teacher calls. It does **not** make the
transition corpus or target-occupancy validation free.

Let `A_new` count every fresh unique target-support, successor-construction, and validation state
that requires the exact teacher, and let `D_new` be their cached reuses. If labels are cached once,

\[
C_{\rm teacher}^{B}
=600+A_{\rm new}+0\,(7200+D_{\rm new}).
\tag{10}
\]

`A_new` is `UNKNOWN` and must remain so until the transition/validation design fixes its rows. Any
teacher call used to construct a supposedly "unlabeled" target successor is included in `A_new`.
All cache build, fit, validation, retry, recovery, and source-migration calls must reconcile to the
hook-observed ledger.

### 5.3 A/B ticket

| Field | Arm A | Arm B |
|---|---|---|
| identity | sealed unweighted frozen-replay convex head | same frozen head family with fixed FORE weights |
| labels | same exact cached label bytes | same bytes, plus separately counted target-holdout labels |
| replay/head split | inherited fixed 480/120 instance | identical where rows overlap; transition split separately preregistered |
| target | replay-distribution risk | declared frozen `pi_j` discounted occupancy risk |
| head certificate | inherited only for A | re-derived weighted Hessian certificate |
| ratio gate | N/A | Markov schema, target successors, coverage, normalization, ESS/weight diagnostics, heldout single-level loss |
| primary comparison | current registered frozen-axis metrics | exact target-occupancy heldout risk/direction versus A |
| teacher law | sealed `600` | equation (10), fully observed |
| authority | local training-gradient evidence | same until separate evaluator-cell and contest-axis gates |

No ESS, maximum-weight, improvement, or `gamma` threshold is guessed in this memo. The execution
ticket must derive/preregister them from sample-size and target-functional tolerances. Until rows
2--4 of the schedule pass, Arm B is `REFUSED_BEFORE_MEASUREMENT`, not a failed empirical treatment.

### 5.4 VR-GHAL admission readback

- `Y` at theorem/formulation level: fixed `pi_j`, full Markov transition corpus, `gamma<1`, coverage,
  normalized convex log-ratio class, ratio realizability/approximation, and a separately certified
  weighted convex head give an explicit contraction route.
- `N` for current bytes: the transition and coverage hypotheses are missing.
- The repaired formulation remains in the reformulation queue; it is not a live lever and does not
  edit the round-2 arm.

## 6. Costate-organ backtest correction

### 6.1 Offline-RL typing

For cross-run schedule evaluation, define

- `Z_t`: full organ/controller state at a verdict boundary, including measured telemetry history,
  stage, controller memory, run/code/hardware/axis fingerprints, and all confounders required for
  sequential ignorability;
- `A_t`: the typed schedule/arm decision actually taken;
- `R_t`: the preregistered realized schedule reward. This may be score-unit improvement, a full
  Lagrangian reward including rate/compute, or a narrower diagnostic, but the gate must not switch
  definitions post hoc;
- `P`: the conditional law of the next full state and reward given `(Z_t,A_t)`;
- `nu_log`: the offline distribution of logged state-action transitions across runs;
- `pi_sched`: the target schedule policy to be evaluated.

The target ratio and direct estimator are

\[
\omega_{\pi,\gamma}^{\rm organ}(z,a)
=\frac{d d_{\pi,\gamma}^{\rm organ}}{d\nu_{\rm log}}(z,a),
\qquad
\boxed{
\widehat V_{\rm FORE}(\pi)
=\frac1n\sum_{i=1}^n
\widehat\omega_{\pi,\gamma}^{\rm organ}(Z_i,A_i)R_i.
}
\tag{11}
\]

This estimates the normalized discounted value
`(1-gamma) E_pi[sum_t gamma^t R_t]`. If the organ gate targets a terminal score instead, terminal
reward and horizon must be typed explicitly; equation (11) is not silently a terminal-score
estimator.

With a fitted action-value function `Q`, the paper's doubly robust functional maps to

\[
\widehat V_{\rm DR}
=(1-\gamma)\,\mathbb P_{0,n}Q
+\mathbb P_n\!\left[
\widehat\omega(Z,A)
\{R+\gamma E_{a'\sim\pi(\cdot|Z')}Q(Z',a')-Q(Z,A)\}
\right].
\tag{12}
\]

`DERIVED`: equation (11) is the minimum new estimator. Equation (12) is preferable only after a
separate Q-function/cross-fitting design; it is not required to state the FORE correction.

### 6.2 Does only-ratio-realizability weaken the organ's assumptions?

**Yes, narrowly; no, as a blanket statement.**

| Assumption | FORE effect | Current organ consequence |
|---|---|---|
| adjoint Bellman completeness of ratio class | removed; only the fixed-point log-ratio must be realizable/approximable | material simplification for a future OPE estimator |
| separate critic richness/dual completeness | not needed by FORE fitting | material simplification versus DICE/minimax-style gates |
| behavior-policy model | explicit behavior policy not required | helpful, but logged target-action support is still required |
| Markov state and common conditional transition/reward law | unchanged | old-to-new code, hardware, controller, or unlogged state can violate it |
| state-action positivity/coverage | unchanged and boundedness is used in guarantees | deterministic historical routing cannot evaluate an unchosen arm at that state |
| target initial-distribution coverage | unchanged | new-run initialization/regime must be represented |
| ratio positivity/log regularity and empirical sample conditions | added explicitly | one trajectory with sparse dependent intervals is not the paper's iid transition regime |
| causal reward identification | unchanged | forecasting a slope is not the same estimand as the value of choosing a schedule arm |

The current organ's walk-forward forecast backtest never claimed Bellman completeness. FORE
therefore does not "weaken" its existing supervised assumptions; it defines a different, causal
policy-evaluation gate with extra data obligations. It corrects occupancy/covariate shift only under
a stable conditional law `P(R,Z'|Z,A)`. It does not repair arbitrary concept shift between old and
new training code.

### 6.3 FEED-costate verdict

`CONDITIONAL ADOPT AS AN ADDITIONAL BACKTEST GATE`; `NO-GO AS A REPLACEMENT ON CURRENT LOGS`.

- Keep the existing past-only walk-forward forecast gate as the present instance authority.
- Add FORE-OPE only after multiple trajectories log full state, actual schedule action,
  target-action propensity/support, reward, next state, run fingerprint, and initial state.
- A deterministic action with zero alternatives receives `NOT_IDENTIFIED`, never a numeric
  counterfactual estimate. Do not clip infinity into authority.
- On a new run/code/hardware stratum absent from logs, the verdict is `OUT_OF_SUPPORT`, not a
  forecast failure and not a family kill.
- If only a fixed arm's state distribution changes and the common conditional law is defensible,
  a state-only MRP ratio may correct its expected diagnostic. It still cannot compare schedule
  actions.

`verdict_scope=FORMULATION x CURRENT ORGAN DATASET`: causal OPE of new-run schedule arms from the
present single deterministic regime sequence. The costate organ, GP forecaster, regime dispatcher,
walk-forward gate, and FORE family remain open.

## 7. Composition with rate-law rung #468

Let the settled marked variables be
`Y=(X,E,Phi,Delta^E)` with context `C`, and let a declared conditional model assign the chain
surprisal. Here #468's `X` is its quantized Lie-transport datum, not FORE's state-action variable.

\[
\ell_{\rm chain}(Y,C)
=-\log p(X|C)
-\log p(E|X,C)
-\log p(\Phi|E,X,C)
-\log p(\Delta^E|\Phi,E,X,C).
\tag{13}
\]

If `(Z,A)` is a training/control context and the conditional mark channel is common given that
context, the target-occupancy expected codelength is

\[
\boxed{
R_{\rm temporal}^{\pi}
=E_{d_{\pi,\gamma}}[\ell_{\rm chain}]
=E_{\nu}[\omega_{\pi,\gamma}(Z,A)\,\ell_{\rm chain}].
}
\tag{14}
\]

This is **outer measure transport**. FORE weights how often contexts contribute; #468 factors the
conditional surprise inside a context. Consequently:

- there is no new `H(omega)` term in the marked chain rule;
- the ratio is not an archive section, event mark, phase, twist, or residual;
- no conditional independence follows from reweighting;
- if a schedule/code change alters `P(Y|Z,A)` itself, occupancy weighting alone does not transport
  the conditional channel;
- for the final decoded witness distribution, optimizer occupancy is not a receiver variable and
  slots nowhere in counted payload. It may guide training or bit allocation upstream only.

**Composition verdict:** `DERIVED-CONDITIONAL` as equation (14),
`verdict_scope=FORMULATION`. It composes only when the same state/action typing and common
conditional transition law used by FORE also generate the marked variables. It is not a fifth rung
or a modification of `rate_law_ladder_v1`.

## 8. Reformulation queue and falsifiers

1. **Transition-sufficiency audit:** prove which saved fields make `Z` Markov by replaying one step
   from a preserved state and matching the recorded successor. A render-only match is insufficient.
2. **Coverage design:** log a randomized or otherwise positively supported schedule mixture at
   stage boundaries, or restrict the target policy to logged actions. This is a design ticket, not
   permission to launch.
3. **Stage-frozen FORE arm:** fit the ratio for one fixed target program and one declared time law;
   persist every ratio-iteration checkpoint.
4. **Weighted-head certificate:** re-derive the realized weighted Hessian and run the Arm A/B target
   holdout with full teacher custody.
5. **Finite-horizon disambiguator:** compare a declared geometric occupancy with an exact
   time-indexed/terminal estimand; do not let `gamma` be a numerical tuning knob.
6. **Organ logging upgrade:** accumulate multiple trajectories with state, action, propensity,
   reward, successor, and run-stratum fingerprints before causal adoption.
7. **Outer-measure rate probe:** only after a valid ratio exists, report weighted marked
   codelengths; never encode the ratio as payload.

Falsifiers:

- Any target-positive/replay-zero region is a bridge refusal at `FORMULATION x DATA SUPPORT`.
- Failure to reproduce a one-step successor from saved state is a Markov-schema refusal at
  `INSTANCE` or `FORMULATION` depending on whether fields can be added.
- A fitted ratio failing normalization/heldout single-level risk or producing nonfinite weights is an
  `INSTANCE` negative, not family death.
- A valid ratio whose weighted head loses Arm A on the preregistered target holdout is a
  `FORMULATION x HEAD/FEATURE CHART` negative; frozen-stem, RFF, or other convex charts remain queued.
- Failure of strict contraction at `gamma=1` kills only the undiscounted/no-mixing formulation.

## 9. Triality and system wire-in

- **DAG:** standalone feed
  `.omx/research/fore_occupancy_ratio_DAG_FEED_20260713.md`; shared DAG remains untouched for main
  review.
- **Equation:** candidate equations (5)--(14) live in this memo only.
  `FORMALIZATION_PENDING_NOT_CANONICAL` because current bridge coverage and transition custody fail.
- **DSL:** no flag, trainer argv, or live policy was invented. A future typed policy must carry
  `state_schema_id`, `action_schema_id`, `target_policy_hash`, `replay_manifest_hash`, `gamma` and
  target-estimand identity, coverage receipt, ratio-class identity, split seed, checkpoint cadence,
  teacher ledger, and verdict scope.
- **Sensitivity map:** prospective contribution is target-occupancy-weighted exact costate and
  renderer-gradient risk; absent until the bridge is identified.
- **Pareto constraint:** full teacher calls, ratio ESS/max weight, transition bytes/storage, and
  evaluator-cell debt. No archive saving is claimed.
- **Bit allocator:** equation (14) may later weight marked codelength observations; currently
  non-binding.
- **Cathedral/autopilot:** no dispatch hook; `research_only=true` and current instance is refused.
- **Continual learning:** memo plus standalone DAG bank the deterministic-contraction correction and
  the support failure so later agents do not repeat either mistake.
- **Probe disambiguator:** future static modes are `full_state_action_mdp` versus
  `fixed_optimizer_actionless_mrp`; the transition-sufficiency/coverage receipt, not taste, chooses.

## 10. STORES CONSULTED and custody

- Full `CLAUDE.md`, full `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `PROGRAM.md`,
  SPEC-v7.5 including section 8/settled table, and SPEC-v8.
- Top current project memory entries, last-24h directives, `reports/latest.md`, lane registry,
  subagent-progress ownership map, latest required Codex/council/design/operator surfaces.
- `.omx/research/onpolicy_surrogate_95kill_20260713.md` (#455).
- `.omx/research/vrghal_95kill_fixedpoint_20260713.md` (#462).
- `.omx/research/frozen_replay_convex_head_95kill_20260713.md` and its standalone DAG feed.
- `.omx/research/costate_organ_capabilities_limits_envelope_20260711.md`,
  `.omx/research/closed_form_gp_costate_posterior_20260711.md`, and
  `.omx/research/organ_regime_conditional_dispatch_436_20260711.md`.
- `src/tac/canonical_equations/rate_law_ladder_20260713.py` and
  `.omx/research/condprob_homotopy_lie_dig_20260713.md` (#468).
- Official arXiv paper surfaces listed in section 0.

No trainer, scorer, evaluator, archive, live run, provider, GPU, or sibling deliverable was
mutated or launched. The only shared apparatus writes were the mandatory L0 lane registration and
checkpoint rows. This memo and its standalone DAG are uncommitted for main review.

**Pointer delta: `NONE`.**
