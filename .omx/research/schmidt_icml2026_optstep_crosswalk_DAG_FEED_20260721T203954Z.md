---
schema: pact.dag_feed.schmidt_icml2026_optstep_crosswalk.v1
feed_id: FEED-SCHMIDT-OPTSTEP-20260721
utc: 2026-07-21T20:39:54Z
lane_id: lane_schmidt_icml2026_optstep_crosswalk_20260721
research_only: true
execution_authority: false
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
pointer_delta: 0
axis: "[research-only; source/repository/custody audit]"
verdict: "NO-VERDICT_DATA_CUSTODY; ADOPT-INSTRUMENTATION-ONLY"
verdict_scope: "INSTANCE x CURRENT-TELEMETRY-CUSTODY; adaptive-step family open"
main_landing_review_required: true
---

# DAG FEED — Schmidt optimizer-step crosswalk

## Decision graph

```text
official 209-page tutorial + primary papers
  |  source/hash/page custody
  v
exact formula digest
  |
  +--> recall current non-PR95 optimizer/init/schedule surfaces
  |      |
  |      +--> v9/v9-CGauge schedule is an event-first hybrid
  |      |      `EventTriggeredCurriculum`
  |      |      `WitnessNativeMorseContinuationSchedule`
  |      |      `ExitEvent` / `TauAdvanceEvent` / `BirthCompletionEvent`
  |      |      `CurriculumReanchorLevers`
  |      |      fixed caps/static Muon placement/global cadence approximations
  |      |      build-owed entry/exit/repeat branches remain explicit
  |      |
  |      `--> sealed #205 hardcoded schedule remains the fixed CONTROL
  |
  v
ranked 24-row-primary-label technique crosswalk
  |  1 ADOPT probe / 8 ALREADY-HAVE-BETTER / 15 N-A-WHY
  v
D3 split question
  |  CORE: rescale fully realized incumbent delta; formulation/state fixed
  |  ADJACENT: raw-GD AdGD/Polyak and Schedule-Free formulation forks
  |
  +--> #205 trajectory: scalar loss + gnorm + events + checkpoint names
  +--> legacy mod32cap: 41 evaluator rows + 3 state-bearing boundary candidates
  |      exact deterministic fork support UNVERIFIED pending reload/one-step parity
  +--> C2: receiver strata/sensitivity, NOT optimizer trajectory
  +--> Fisher/margin/nTHn: named geometry, NOT calibrated H_loss identity
  |
  `--> missing delta-w, delta-g, raw directions, directional HVP,
       candidate loss evaluations
          |
          v
       REFUSE retrospective counterfactual superiority
       verdict_scope=INSTANCE x CURRENT-TELEMETRY-CUSTODY
          |
          v
       ADOPT exactly one pending $0 recorder/probe task
       `ADOPT_WITHIN_SEGMENT_SHADOW_CANDIDATE_STEP_PROBE`
          |
          +--> shadow one-step eligible candidate evaluations on copied full sidecar
          +--> held-out calibration + decrease-per-wall-second gate
          +--> explicit insufficient-custody/assumption refusal
          |
          v
       CONDITIONAL ONLY AFTER GREEN MEASUREMENT
          +--> typed default-off `WithinSegmentAdaptiveStep`
          +--> select/register no law until measurement; conditional candidate
          |    `within_event_segment_directional_quadratic_step_model_v1`
          +--> resumable short deterministic fork (operator GO if heavy)
          `--> exact full-facet through-R gate
```

## Verdict scopes

| Negative | Token | What remains open |
|---|---|---|
| Existing trajectories cannot establish an alternate optimizer path. | `INSTANCE x CURRENT-TELEMETRY-CUSTODY` | Instrumented AdGD, directional line search, Polyak, BB, and Schedule-Free formulations. |
| Polyak lacks a custodied within-segment optimum/target. | `INSTANCE x UNKNOWN_VALID_FSTAR` | Polyak with a measured or certified segment target. |
| AdGD/BB secants are absent. | `INSTANCE x DELTA_W_DELTA_G_ABSENT` | Source-faithful secant formulations after instrumentation. |
| Armijo/Wolfe have no candidate evaluations or wall accounting. | `INSTANCE x NO_CANDIDATE_EVALUATIONS` | Non-monotone line search with copied-state evaluation and exact cost. |
| Schedule-Free has not been compared inside fixed events. | `FORMULATION x WITHIN_SEGMENT_UNMEASURED` | Matched base-LR/momentum/EMA/resume formulation. |
| WSD has not been compared as a matched fixed-segment stable/cooldown branch. | `FORMULATION x FIXED_SEGMENT_WSD_UNMEASURED` | Source-faithful WSD inside an identical event segment. |
| D-Adaptation's convex distance estimator is not objective-local sharpness. | `FORMULATION x CONVEX_DISTANCE_ESTIMATOR_NOT_LOCAL_SHARPNESS` | A source-faithful matched formulation after target/custody gates. |
| Practical Prodigy-on-Adam efficacy is unmeasured here. | `FORMULATION x PRACTICAL_ADAM_HEURISTIC_UNMEASURED` | Equal-search/equal-compute matched fork. |
| AdaGrad-Norm still has a tuned numerator and no local-sharpness identity. | `FORMULATION x NUMERATOR_STILL_TUNED` | Matched candidate after LR sensitivity is measured as binding. |
| Coin-betting/COCOB theory is online-convex scoped. | `FORMULATION x ONLINE_CONVEX_ASSUMPTIONS` | Real-only walk-forward or matched nonconvex trainer formulation. |
| PEP/silver/long-step sequences assume fixed `L` and horizon. | `FORMULATION x FIXED_L_FIXED_HORIZON` | Measured stationary segment/local-model formulation. |
| Full-trunk BB/Chebyshev secant/spectral custody is absent. | `FORMULATION x FULL_TRUNK_SECANT_CUSTODY_ABSENT` | Instrumented secants and a measured spectral interval. |
| Polyak–Ruppert averaging exists but lacks a matched efficacy receipt. | `FORMULATION x PR_AVERAGING_UNMEASURED` | Existing `PolyakTailAverager` measured as a distinct exported candidate. |
| Full-trunk BFGS assumptions/cost are unmet; #423 head math is not argv-reachable. | `FORMULATION x FULL_TRUNK_BFGS_ASSUMPTIONS_UNMET` | Build-owed narrow head consumer; later L-BFGS only after secant/cost evidence. |
| The prior SPS instance was disengaged and interpolation is unproved. | `INSTANCE x DISENGAGED_OR_INTERPOLATION_UNPROVEN` | Engaged SPS with a valid target and matched batch order. |
| Deep large-batch variance-reduction transfer is ungrounded for this n=1 renderer. | `FORMULATION x DEEP_LARGE_BATCH_TRANSFER_UNGROUNDED` | A measured gradient-noise decomposition showing noise is binding. |
| Fisher/margin/`nᵀHn` are not automatically objective Hessian sharpness. | `FORMULATION x UNCALIBRATED_CURVATURE_LOCUS` | Out-of-sample calibration to observed directional loss curvature. |
| C2 cannot replay optimizer steps. | `INSTANCE x RECEIVER_STRATUM_NOT_OPTIMIZER_TRAJECTORY` | C2 remains valid for its receiver/stratum consumers. |
| The current probe design has no archive-byte marginal. | `CURRENT-INSTRUMENT-DESIGN x NO-BYTE_MARGINAL` | Bit-allocator hook after a trained-model byte/score effect exists. |

No negative closes a technique family or the optimization paradigm.

## Triality

### DSL

- Reuse the existing event-transition schedule unchanged.
- No new Lever lands in this research pass.
- Conditional future factory: `WithinSegmentAdaptiveStep`, default-off and additive-resume only.
- The conditional Lever is subordinate to event boundaries and may never change loss weights
  per-step.

### DAG

- This file is the standalone FEED.
- `autopilot_action=NONE`; task visibility is not dispatch authority.
- The only active edge is source/custody audit → pending recorder/probe task.

### Equations

- `law_selection=NO_LAW_SELECTED`; `registered=false`.
- Conditional candidate ID: `within_event_segment_directional_quadratic_step_model_v1` with
  `delta_f_hat(r)=r*g^T*delta_w_inc + 0.5*r^2*delta_w_inc^T*H_loss*delta_w_inc` and
  `r_Q=-(g^T*delta_w_inc)/(delta_w_inc^T*H_loss*delta_w_inc)`.
- Inputs/units: `r` is a dimensionless common multiplier (`r=1` incumbent); `g` is loss per
  parameter-unit; `delta_w_inc` is the full parameter-unit update at `r=1`, including decoupled
  weight decay while holding per-group LR ratios fixed; directional curvature is loss.
- `H_loss=nabla^2 f_batch(w_k)` at the same pre-update state, batch, and objective as `g`; require
  directional twice-differentiability/model validity, descent numerator, and positive curvature.
  Weight decay is inside `delta_w_inc` but outside `H_loss` unless explicitly part of `f_batch`.
- Wall payoff is a separate measurement, `-delta_f_actual/candidate_wall_seconds` (loss/second),
  not an equation input.
- Producer if built: `pact.within_segment_candidate_step_probe.v1`; immediate measured consumer:
  `tac.probe_outcomes_ledger.register_probe_outcome`; conditional downstream consumer:
  `WithinSegmentAdaptiveStep`.
- Registration requires held-out one-step calibration plus deterministic path-level fork benefit
  with exact resume and protected-facet custody.
- Existing event/Morse, Fisher/margin, viscosity/CFL, and optimizer-geometry laws are reused without
  claiming equivalence.

## Six-hook wire-in

1. **Sensitivity:** reuse Fisher/margin/`nᵀHn` as covariates only; no new constant.
2. **Pareto:** track objective decrease per wall-second; no score/pointer constraint mutation.
3. **Bit allocator:** `N-A-WHY — CURRENT-INSTRUMENT-DESIGN x NO-BYTE_MARGINAL` until an
   archive-byte/score effect exists.
4. **Cathedral/autopilot:** pending task is queryable; no actuation or dispatch edge.
5. **Continual learning:** no empirical posterior row before a measurement receipt.
6. **Probe disambiguator:** candidate-law matrix plus explicit refusal when inputs/assumptions are
   missing.

## Consumers and falsifiers

| Object | Named consumer | Falsifier |
|---|---|---|
| Recorder/probe | `tac.probe_outcomes_ledger.register_probe_outcome` consuming `pact.within_segment_candidate_step_probe.v1` | Cannot reproduce the incumbent one-step digest from copied state; candidate evaluation perturbs RNG/source state; or required inputs remain absent. |
| AdGD mode | raw-gradient branch only | Held-out secant proposal gives no positive paired candidate-loss-per-wall benefit or violates safeguards. |
| Directional local-quadratic mode | full realized AdamW/Muon update at common multiplier `r=1` | `delta_w_incᵀH_loss delta_w_inc≤0/unstable`, local model residual fails calibration, or incumbent wins. |
| Polyak mode | shadow probe with valid target | No valid target; overshoot; or incumbent wins. |
| Armijo mode | copied-state non-monotone evaluator | Extra evaluations erase wall-clock benefit or protected facets regress. |
| Schedule-Free mode | formulation-selection gate, then separately preregistered deterministic fork | No variant is selected now. Basic SGD owns `x_t,z_t,z_1=x_1`; AdamW v4 Algorithm 1 additionally owns moments, hyperparameters, warmup/bias-corrected step, cumulative weighting, epsilon, and decay-at-`y` semantics. Any incomplete state/resume schema or lack of matched benefit falsifies that formulation. |

## Stores consulted

Official tutorial/PDF extraction; AdGD, Polyak, D-Adaptation, Prodigy, Schedule-Free, MiniCPM/WSD,
sign-gradient momentum/Signum/Lion, coin-betting, PEP, BFGS, and pinned Muon primary sources;
v7.5/v8 specs; current v9 design and sister audits;
`curriculum_dsl.py`; `spec_v9_cgauge.py`; `witness_control/`; `optimization/`;
`canonical_equations/`; #205 trajectory/diagnostic snapshot; mod32cap launch/result/resume files;
C2 per-class-stratum and witness-own decomposition artifacts; canonical task/lane/checkpoint ledgers.

## Authority boundary

Research only; **$0**; no launch, scorer execution, archive mutation, provider dispatch, equation
registration, DSL actuation, or score promotion occurred. MAIN must review the complete branch diff
before landing. Pointer **0.1910828242 [contest-CPU] UNMOVED**.
