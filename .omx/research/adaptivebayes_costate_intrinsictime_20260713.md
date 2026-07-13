# Adaptive Bayes intrinsic time for costate DECIDE and curriculum control

**Date:** 2026-07-13 UTC  
**Status:** `DESIGN`, `research_only=true`, `MAIN_REVIEW_REQUIRED`, `UNCOMMITTED`  
**Reader lane:** `adaptivebayes_intrinsictime_reader`  
**Paper:** Akshay Balsubramani, [*Adaptive Bayes exactly tracks information over intrinsic time*](https://arxiv.org/abs/2607.08789), arXiv:2607.08789v1 (2026-06-26)  
**Authority:** paper theorem algebra + read-only repository inspection; **NO new empirical run**  
**Pointer delta:** `UNMOVED`  
**Verdict scope:** costate-organ DECIDE/controller quality only; no evaluator score, training result, launch authority, or pointer movement is claimed

## Executive verdict

| Application | Does the paper's multiplicative-weights identity apply now? | Scoped verdict |
|---|---|---|
| #436 regime arbitration | **NO** to the current landed #436 dispatcher. **YES, conditionally**, to a new fixed-index, common-state, full-loss-vector exponential-weights arm. | `WORTH-AN-ARM -> FEED-costate-controller`, subject to the gates below. |
| Curriculum stage advance | **NO** to the live single-trajectory `{stay, advance}` controller. The two actions do not expose a full same-round loss vector and `advance` changes subsequent state/action dynamics. **YES, conditionally**, only to explicit common-checkpoint shadow A/B continuations treated as fixed policy experts. | `NO-GO` as an exact live clock; `FEED-#315-#344` only as a checkpointed shadow branch selector. |

The main adversarial result is that intrinsic time alone is not directional evidence. The same intrinsic-time path can favor opposite experts after permuting their labels, and zero intrinsic time can mean either equal losses or posterior degeneracy. Therefore there is no theorem-valid rule of the form `V_T >= v_star => commit/advance`. The exact commit law uses **comparator advantage after subtracting the intrinsic-time payment**, equivalently terminal KL/posterior mass.

This is a **MEANS** design. Intrinsic-time accounting can ground a controller and diagnose when uncertainty was paid; it is not a 95%-kill lever. Only a custody-complete, exact-through-R, `n600`, NumPy-fp32-authority win can move an empirical verdict or pointer.

## Labels and source discipline

- **DERIVED:** equations and logical consequences proved below from the paper's fixed-temperature identity.
- **MEASURED (historical, not re-measured here):** facts explicitly recorded in existing Pact artifacts, always reported with their original scope.
- **INFERRED:** repository-fit conclusions obtained by comparing the theorem hypotheses against current code/memos.
- **ASSUMED:** proposed arm protocol and future loss normalization; these have no empirical authority.
- **NO-FAKE:** no paper experiment, synthetic appendix result, local proxy, or one-trajectory backtest is promoted to Pact `n600` authority.

## 1. The exact law, without the variance shortcut

Let the fixed indexed expert set be `E={1,...,K}`. At round `t`, let `p_t` be the played distribution, let `c_t in R^K` be the finite composite-loss vector, and use a fixed `eta>0`:

```text
p_{t+1,i} = p_{t,i} exp(-eta c_t(i)) / Z_t,
Z_t       = sum_i p_{t,i} exp(-eta c_t(i)),
m_t       = -(1/eta) log Z_t.
```

The paper permits `c_t = ell_t + u_t`, including positive side factors through
`u_t(i)=-(1/eta) log s_t(i)`. Define

```text
delta_t = <p_t,c_t> - m_t,
X_t     = c_t(I_t) - <p_t,c_t>,       I_t ~ p_t,
Q_t     = (1/eta^2) log E[exp(-eta X_t)],
V_T     = sum_{t=1}^T Q_t.
```

Then `delta_t = eta Q_t`. Corollary 2.3 gives, for every comparator distribution `rho`,

```text
<p_t,c_t> - <rho,c_t>
  = eta Q_t
    + [KL(rho || p_t) - KL(rho || p_{t+1})] / eta.        (1)
```

At fixed temperature this telescopes exactly:

```text
R_T(rho)
  := sum_t (<p_t,c_t> - <rho,c_t>)
   = eta V_T
     + [KL(rho || p_1) - KL(rho || p_{T+1})] / eta.       (2)
```

**DERIVED/EXACT:** `Q_t` is the finite-rate centered cumulant, not ordinary variance. The variance statement is only the small-temperature relaxation

```text
Q_t = (1/2) Var_{i~p_t}(c_t(i)) + o(1),  eta -> 0.
```

Bounded losses are sufficient for range bounds such as `Q_t <= (b_t-a_t)^2/8`; they are not needed for the algebraic identity if the exponential normalizer is finite. For partial information, the paper applies the identity to an estimated loss vector and adds action-sampling noise, estimator noise, and predictable estimator-bias terms. Those terms may not be silently discarded.

For variable temperatures, equation (2) is not valid as written. The paper's prior-retempered chain has

```text
R_T(rho) = sum_t eta_t Q_t + D_T + B_T(rho),              (3)
```

where `D_T` is explicit temperature drift and `B_T` is terminal comparator information. The local recursive update instead accumulates its Abel/KL transport terms. A future arm must choose and name one recursion; mixing their bookkeeping would fake exactness.

## 2. Application 1 — #436 regime-conditional arbitration as Hedge

### 2.1 Current repository truth

**MEASURED/ARTIFACT-SCOPED:** `.omx/research/organ_regime_conditional_dispatch_436_20260711.md` records a one-trajectory, seven-walk-forward-fold result for a fixed regime routing table. It explicitly does not establish an `n600` law. The nominal dispatcher comparison is provisional and cannot authorize promotion.

**INFERRED FROM CODE:** `src/tac/witness_control/regime_dispatch.py` implements a deterministic map from regime labels to forecast-model names. It has no probability simplex, exponential update, temperature, mix loss, composite-loss vector, or intrinsic-time state. Therefore current #436 is **not** Hedge and Balsubramani's exact identity does not describe it.

**ASSUMED/PROSPECTIVE:** the prompt's expert set

```text
{distilled surrogate,
 adversarial-boundary,
 comma10k/openpilot regime-model,
 per-class-lambda}
```

is treated here as a proposed fixed expert index. The #426 capability envelope records that several of these are designed rather than built or promotion-measured. Naming them as experts does not create the loss table the identity needs.

### 2.2 Hypothesis-fit audit

| Required object/hypothesis | Holds? | Adversarial finding |
|---|---:|---|
| Fixed, enumerable expert labels | **Conditional** | Holds only for a preregistered arm epoch. Birth/retirement requires a sleeping/shifting-expert construction and explicit prior-mass transport. |
| Predictable `p_t`, `eta`, and side information | **Can hold** | Controller state must be persisted before feedback. Fixed `eta` is the clean first arm. |
| Same scalar loss semantics for every expert | **Does not yet hold** | Raw `Delta S`, forecast error, boundary violations, and class-lambda debt are not automatically commensurate. |
| Full `c_t(i)` vector after every round | **Does not currently hold** | A selected tool normally yields one realized continuation. Unselected expert outcomes are counterfactual unless all experts are evaluated from the same frozen state. |
| Comparator is well-defined | **Conditional** | Exact comparator is the realized column sum for a fixed expert label. It is not automatically “what would have happened had expert `i` controlled training from the start.” |
| Experts may themselves train | **Algebra yes; semantics conditional** | The identity accepts an arbitrary realized loss-vector sequence. But selection-dependent expert state makes the fixed-column comparator path-dependent and defeats the intended counterfactual meaning. |
| Finite exponential normalizer | **Not instantiated** | Any finite realized `K`-vector suffices algebraically, and signed values are harmless. Non-finite entries fail; bounded/self-bounding claims and numerical controller safety additionally require a registered scale or tail guard. |
| Mixed action is the incurred loss | **Conditional** | If one expert is sampled, `<p_t,c_t>` is distribution-level loss and the realized sampled loss adds a martingale term. Persisted RNG/propensity is mandatory. |

### 2.3 Minimum valid arm protocol

At a frozen decision epoch and common pre-step checkpoint `x_t`, every eligible expert must emit an action without mutating shared state. Evaluate each action over the same receiver-closed window and define a lower-is-better composite loss

```text
c_t(i) = [DeltaS_t(i) - a_t] / b_t + u_t(i),              (4)
```

where `a_t` is a predictable common translation, `b_t>0` is a predictable common score-unit scale, and `u_t` contains only preregistered side/constraint terms. Equation (4) is **ASSUMED**, not a selected normalization. `b_t` must come from the value-provenance ladder and exact `n600` receiver-realized receipts; it must not be fitted after seeing the same round. Hard safety/topology violations should make an arm ineligible before Hedge, rather than injecting an invented infinite loss.

If evaluating every arm is too expensive, route measurement acquisition to #463. Do not call the resulting one-observation update “full-information Hedge.” A bandit estimator is possible, but then propensities, bias correction, variance, and sampling-martingale custody become first-class terms.

### 2.4 DERIVED exact commit-vs-hedge law

For a candidate expert `j`, use the point comparator `rho=e_j` and define its information distance

```text
I_t(j) := KL(e_j || p_t) = -log p_t(j).
```

Substituting into (2) gives

```text
R_T(j) - eta V_T = [I_1(j) - I_{T+1}(j)] / eta.           (5)
```

Choose a preregistered residual mixture-mass tolerance `alpha in (0,1)`. Operationally define “commit to `j`” as `p_{T+1}(j) >= 1-alpha`. This is a controller concentration rule, **not** a calibrated probability that `j` is truly optimal. Since the definition is equivalent to `I_{T+1}(j) <= -log(1-alpha)`, equation (5) yields the exact threshold

```text
COMMIT(j; alpha)
iff R_T(j) - eta V_T
    >= [I_1(j) + log(1-alpha)] / eta.                     (6)
```

For a uniform prior over `K` experts,

```text
COMMIT(j; alpha)
iff R_T(j) - eta V_T >= log(K(1-alpha)) / eta.            (7)
```

The equivalent posterior-odds audit is

```text
log[p_{T+1}(j)/p_{T+1}(k)]
 = log[p_1(j)/p_1(k)] + eta sum_t [c_t(k)-c_t(j)].         (8)
```

A sufficient all-rival condition is

```text
min_{k != j} log[p_{T+1}(j)/p_{T+1}(k)]
  >= log((K-1)(1-alpha)/alpha).                           (9)
```

Equation (6) is the clean law. It says: commit only when the candidate's accumulated advantage has paid both (i) the exact intrinsic-time uncertainty bill `eta V_T` and (ii) enough initial KL information debt to leave at most `alpha` posterior mass outside the candidate.

An intrinsic-time-indexed stopping representation is

```text
N(v) := inf{T : V_T >= v},
tau_commit := inf{N(v) : equation (6) and all safety guards hold}.  (10)
```

But (10) does **not** make `V_T` sufficient. It only uses intrinsic time as the path parameter at which the directional condition is checked.

### 2.5 Why a `V_T`-only threshold is impossible

1. Relabel two experts. `Q_t` and `V_T` are invariant, but the winning expert swaps.
2. If `c_t(i)` is constant over the support, then `Q_t=0`; this is no evidence for any expert.
3. If `p_t` is already nearly degenerate, `Q_t` can be tiny even if the off-support alternatives are poorly tested.
4. Alternating leaders can accumulate substantial intrinsic time without creating a stable fixed-expert winner.

Therefore “enough intrinsic time has elapsed” is not equivalent to “enough information favors `j`.” Terminal KL/odds is the directional object. The paper's self-bounding/low-noise theorem is also conditional: for point comparator `j` it requires a bounded common-scale loss process and a comparator-centered Bernstein-type relation. That condition is unmeasured for the costate experts and must not be assumed from low observed variance.

### 2.6 Composition with #463 TOFU-POV

`.omx/research/tofupov_ranker_allocation_20260713.md` is complementary, not interchangeable:

```text
#463: Which expensive expert/lever outcome should be measured next?
       -> acquisition/ranking under costly or partial feedback

Adaptive-Bayes arm: Given a valid common loss vector (or a fully accounted estimator),
                    how should expert mass change, and when is commitment justified?
```

The composition is

```text
fixed checkpoint/regime
  -> #463 ranks next missing outcome measurement
  -> measured outcome + propensity/custody enters the loss table
  -> full-vector arm, or explicit partial-information estimator
  -> Hedge update + Q_t/V_T + KL transport receipt
  -> equation (6) commit gate
  -> costate DECIDE advisory output
  -> exact n600 walk-forward adoption gate
```

Do not pool #463's heterogeneous lever/curriculum queues into one stationary expert game, and do not infer a Hedge expert dimension from a witness-manifold dimension. #463 buys information; equation (6) accounts for how directional expert evidence consumes it.

### 2.7 Application-1 verdict

**Verdict:** `WORTH-AN-ARM -> FEED-costate-controller`.

**Verdict scope:** a default-off, fixed-temperature, frozen-expert-epoch advisory arm with a common-checkpoint loss table. This is **not** approval to replace #436, train any prospective expert, actuate the costate organ, or claim a score effect. The current dispatcher remains outside the identity.

## 3. Application 2 — intrinsic time as curriculum stage-advance clock

### 3.1 Current controller truth

**INFERRED FROM CODE/ARTIFACTS:**

- #315's event controller uses within-stage rolling loss-slope/plateau evidence plus topology/nucleus guards and hard limits.
- Under `seg-form-unify-tau`, the old discrete CE -> tau boundary is dissolved; the corresponding curriculum-event flag is inert for that boundary, while tau-rung advancement has its own event path.
- #344's linear-NCDE surface is a shadow-only forecast/hit detector with stability and fit-quality guards. Its thresholds are not an `n600` law.
- V9/CGauge curriculum control is transactional: checkpoint, propose a tau/stage move, evaluate, accept or roll back. This is closer to a branch-selection problem than to a per-step fixed-expert game.

### 3.2 Why live `{stay, advance}` is not Hedge

At live state `x_t`, choosing `stay` produces one next state and choosing `advance` produces another. Only one continuation is realized. Moreover `advance` changes the loss geometry, optimizer/curriculum state, future action set, and the meaning of later feedback. Thus there is no observed vector

```text
c_t = (c_t(stay), c_t(advance))
```

on the live path. A fixed comparator “always advance” is not well-defined across stages, and “advance now” is an irreversible stopping policy rather than a stationary expert. Rolling slopes and NCDE forecasts are sensors; relabeling them as the two counterfactual losses does not satisfy the theorem.

**Scoped NO-GO:** Balsubramani's exact identity does not apply to the current live stage-advance process. “Advance when `V_T` says the stage is exhausted” is a metaphor, not an exact consequence.

There is also no monotone exhaustion semantics in `V_T`:

- `V_T` grows when the current mixture sees finite-rate loss dispersion.
- `V_T` can stop growing because policies agree, because the posterior collapsed, or because the loss table lost sensitivity.
- High `V_T` can coexist with alternating policy leaders.

None of these cases alone says that the present representation stage has spent its useful optimization capacity.

### 3.3 Conditional exact shadow-fork rule

An exact two-policy construction is possible only at a sacred stage checkpoint. Define two fixed, eligible continuation policies over an identical preregistered horizon `H`:

```text
pi_s: remain in the current stage for H, preserving its legal controller policy;
pi_a: advance transactionally at the checkpoint, then run H under the next-stage policy.
```

Both forks must start from byte-close-identical checkpoint state, preserve their outputs, use the same seeds/data ordering where meaningful, and be scored through the same receiver-closed `n600` NumPy-fp32 authority surface. Their complete finite loss pair is one Hedge round. Failed topology/pose/rate/checkpoint guards make a branch ineligible before the update.

For fixed `eta`, two-arm distribution `p_n`, and branch loss vector `c_n=(c_n(s),c_n(a))`, define `Q_n` and `V_N` exactly as in section 1. The DERIVED advance rule is

```text
ADVANCE(alpha)
iff all #315 topology/nucleus and transactional safety guards pass
and p_{N+1}(a) >= 1-alpha

iff R_N(a) - eta V_N
    >= [I_1(a) + log(1-alpha)] / eta.                    (11)
```

With a uniform two-policy prior,

```text
R_N(a) - eta V_N >= log(2(1-alpha)) / eta.               (12)
```

Equation (11) is exact for the **shadow branch-selection game**. It is not an exact per-training-step stage clock, and its cost may be prohibitive because it requires both continuations. Reusing approximate forecasts for the unrun branch changes the object to an estimated-loss controller with estimator/bias/martingale terms.

### 3.4 Relationship to #315 and #344

The safe composition is advisory and conjunctive:

```text
#344 forecast sensor (shadow; fit/stability receipt)
       +
#315/V9 feasibility and topology guards + transactional checkpoint/rollback
       +
conditional common-checkpoint two-policy loss receipt
       -> equation (11) directional commit check
       -> {advance | stay | insufficient-information}
```

Intrinsic-time accounting should not replace the #315 hard guards or the #344 forecasting surface. It can become a third, exact-accounting sensor only when the loss-pair custody exists. Under unified tau, route the concept to the event-native tau-rung transaction rather than pretending to restore the dissolved CE -> tau boundary.

### 3.5 Application-2 verdict

**Verdict:** `NO-GO` as an exact live curriculum clock; `FEED-#315-#344` for a common-checkpoint shadow A/B policy selector.

**Verdict scope:** this negative applies to the proposed live two-action multiplicative-weights formulation, not to the broader family of event-triggered, optimal-stopping, change-point, NCDE, or transactional branch-selection controllers. One failed formulation is not a dead controller family.

## 4. Value-provenance ladder for any later arm

| Quantity | Required authority |
|---|---|
| Expert identity and eligibility | Typed DSL/config record plus expert/checkpoint/content hashes. |
| `DeltaS_t(i)` | Exact receiver-realized component accounting on the same source/checkpoint; report Seg, nonlinear Pose term, bytes/rate, and total separately. |
| `a_t`, `b_t` in (4) | Predictable, preregistered derivation from prior authoritative receipts; no post-round fit and no guessed clip. |
| `eta` | Theorem/config rung, fixed for the first arm; any adaptive schedule names its exact recursion and drift terms. |
| `alpha` | Operator/controller risk-policy rung, preregistered; it is not learned from the evaluation rows it gates. |
| `p_t`, loss estimates, propensities | Persisted before/with each round; zero-propensity missingness fails closed. |
| `Q_t`, `V_t`, KL terms | DERIVED deterministically from persisted `p_t`, `eta`, and `c_t`; recompute residual of (1) to machine precision. |
| Empirical adoption | `n600`, exact-through-R, NumPy-fp32 authority, full provenance, full-facet non-regression; CPU/CUDA axes remain separate. |

No score normalization constant has been selected in this design. That omission is intentional: the receiver-realized range is not currently established for the proposed expert set.

## 5. Triality and apparatus wire-in

### Equation leg

Isolated candidate `adaptivebayes_fixed_eta_directional_commit_v1` is written in
`.omx/research/adaptivebayes_costate_intrinsictime_equation_feed_20260713.md`.
Canonical registry append is deferred to main because the shared canonical equation surface is live/dirty under sibling work.

### DAG leg

Standalone feeds `FEED-ADAPTIVEBAYES-COSTATE-COMMIT` and
`FEED-ADAPTIVEBAYES-CURRICULUM-SHADOW` are written in
`.omx/research/adaptivebayes_costate_intrinsictime_DAG_FEED_20260713.md`.
Shared DAG append is deferred for the same anti-collision reason.

### DSL leg

No live DSL or controller code is edited. A later main-reviewed arm owes typed fields for fixed expert epoch, loss semantics, normalization provenance, `eta`, `alpha`, full-information versus estimated-loss mode, propensity/bias terms, checkpoint hash, and default-off/advisory status. Never invent trainer flags.

### Six-hook wire-in status

1. **Sensitivity map:** consume receiver-realized per-expert component deltas; not wired in this design.
2. **Pareto constraint:** topology/pose/rate guards precede Hedge eligibility; no scalarization may hide a facet regression.
3. **Bit allocator:** no direct byte actuation; consumes byte component of expert loss only.
4. **Cathedral/autopilot:** default-off DECIDE advisory after canonical review and `n600` gate.
5. **Continual-learning posterior:** persist each authoritative loss-vector/outcome row; current design adds no empirical anchor.
6. **Probe disambiguator:** full-information shadow fork versus partial-information estimator must remain separate callable modes; a cheap probe decides whether full-vector custody is economically viable.

## 6. Minimal main-review next step

Implement no controller yet. First construct a read-only loss-table audit over already available common-checkpoint receipts and answer:

```text
Do at least two proposed experts have same-state, same-horizon, receiver-realized losses with complete custody on enough n600 units
to make one finite c_t vector without imputation?
```

If **NO**, keep the Hedge arm blocked and route missing-measurement selection to #463. If **YES**, implement the smallest fixed-`eta`, default-off ledger/replay arm and verify the one-step residual of equation (1) before any live actuation.

## STORES CONSULTED

- `CLAUDE.md` and supplied `AGENTS.md` operating contracts.
- `docs/operating_manual_craft_handoff.md`.
- Balsubramani, arXiv:2607.08789v1: abstract; Corollaries 2.3/2.4; Theorem 2.10; Proposition 2.11; Theorem 3.2; partial-information section; comparator-centered low-noise section.
- `.omx/research/organ_regime_conditional_dispatch_436_20260711.md`.
- `.omx/research/costate_organ_capabilities_limits_envelope_20260711.md`.
- `.omx/research/tofupov_ranker_allocation_20260713.md`.
- `src/tac/witness_control/regime_dispatch.py` and its tests.
- `src/tac/witness_control/ncde_trajectory.py` and #315 event-controller surfaces in the trainer.
- Latest V9/CGauge design and current sister Codex/council session memos inspected during preflight.
- Canonical lane/subagent/frontier pointer surfaces inspected read-only; no live run, paid provider, or heavy job was touched.

## Final pointer-delta honesty

`DESIGN (MEANS)`. The clean equation is conditional and the live curriculum analogy fails. No new `n600` measurement exists, no training or evaluation ran, no live controller changed, and no shared equation/DAG/costate surface was edited. Required lane/checkpoint state was updated through the canonical tools. The frontier pointer is `UNMOVED`.
