---
title: "Support-first organ OPE audit against pact.causal_manifest.v1"
date_utc: "2026-07-13"
lane_id: "lane_organ_ope_support_first_20260713"
checkpoint_id: "organ_ope_support"
research_only: true
review_status: "UNREVIEWED_MAIN_REVIEW_OWED"
ope_verdict: "REFUSAL_SCHEMA_ONLY__NO_NUMERIC_OPE_ROW"
score_claim: false
pointer_delta: "NONE"
---

# Organ OPE support first: the causal manifest unblocks refusal, not a value estimate

## Answer first

**Per-arm matrix (`DERIVED`):** `T_gp_costate_posterior` is
`transient=SCHEMA_INCOMPLETE`, `plateau=NOT_IDENTIFIED`, and
`uncertain=NOT_IDENTIFIED`. `persistence` is `transient=NOT_IDENTIFIED`,
`plateau=SCHEMA_INCOMPLETE`, and `uncertain=SCHEMA_INCOMPLETE`. Totals are
`SUPPORTED_EVALUABLE=0`, `NOT_IDENTIFIED=3`, `OUT_OF_SUPPORT=0`, and
`SCHEMA_INCOMPLETE=3` across the six reachable regime-arm cells.

**Smallest D40 deltas (`DERIVED`):** join the actual executed regime decision to its following
transition; create positive logged propensity only under a separately operator-authorized safe
stage-boundary policy (otherwise restrict targets to behavior); log adjacent complete
realized-through-R score-gain transitions; emit regime/stratum coverage receipts; and accrue
prospectively split independent real runs. Pair outcomes, loss closure, negative controls, and the
frozen/no-update positive control are an additional logging delta when HCM-L4 participates in
clearance.

**Verdict:** `REFUSAL_SCHEMA_ONLY__NO_NUMERIC_OPE_ROW`. The primary falsifier fired. The landed
`pact.causal_manifest.v1` makes the missing evidence explicit, but the audited organ has one
non-actuating predictive trajectory and **MEASURED 0** production `causal_manifest.jsonl` files.
No FQE, FORE, DR, BCQ, BEAR, CQL, or hierarchical prior may turn those bytes into an action-value
row.

Machine-readable authority for every status and delta:
`.omx/research/organ_ope_support_first_20260713.json`.

## 1. Authority and exact audit boundary

`MEASURED` — The landed source is `src/tac/causal_manifest.py`, schema
`pact.causal_manifest.v1`, SHA-256
`65abcf9213926c1f7ddd5ea333f5a19d0cdc2d4341231d1004ea308a8b03f6fa`, landed by commit
`b6783d45dc`. It represents run manifests, boundaries, transitions, exploration decisions,
propensities, and coverage receipts; it also exposes the structural `check_fore_support` and the
HCM-L4 skeleton.

`MEASURED-INHERITED` — The sealed predictive artifact is
`experiments/results/costate_organ_backtests/costate_organ_backtest_20260711T164017Z.json`, SHA-256
`35e79ca420672c7769377a8b96e18de8459e0cbae4c732c8278f613bc3a2ab61`. It contains one real
trajectory, nine intervals, and seven past-only folds: three transient, three plateau, and one
uncertain. The current deterministic routing law is:

```text
transient -> T_gp_costate_posterior
plateau   -> persistence
uncertain -> persistence
```

`MEASURED-INHERITED` — On that predictive metric only, the dispatcher MAE is
`0.0015959393896760557`, the fixed GP MAE is `0.00185206618604584`, and persistence MAE is
`0.002791931483929152`. Every value is `[macOS advisory] NON-PROMOTABLE`; none is an action effect,
OPE value, evaluator score, or pointer row.

`MEASURED` — A workspace inventory found zero `causal_manifest.jsonl` files, including none under
the sealed legacy run. Running the landed consumers on the empty current causal corpus returned:

```text
FORE: NOT_IDENTIFIED
  missing_run_treatment_manifest
  missing_state_action_reward_successor_transitions
  missing_explicit_coverage_receipt
  target_action_not_supported:{persistence,T_gp_costate_posterior}
  no_executed_decision_rows_for_target_policy

HCM-L4: NO_ROWS
  no_transition_rows
```

This is a read-only `$0` audit. No trainer, scorer, evaluator, archive, provider, GPU, or live run
was invoked or changed.

## 2. Why the predictive arms are not logged actions

`DERIVED` — The present GP and persistence rows predict the same later measured slope. Selecting a
forecaster does not change the optimizer state, schedule, or witness. They are therefore valid
paired predictive comparators but not `do(A)` interventions.

`MEASURED` — The new causal-manifest organ producer is attached to
`shadow_controller.write_shadow_row`, not to `regime_dispatch.dispatch_decision`. It writes policy
`costate_shadow_rank_v1`, chooses the first ranked recommendation, logs deterministic `1/0`
propensities, and fixes `executed=false`, `actuation=NONE`. Those recommendation actions differ in
type from the GP/persistence forecasting tools audited here.

`DERIVED` — Consequently, calling the three GP-routed folds “treated transitions” or the four
persistence-routed folds “behavior transitions” would commit three errors at once: change a
forecaster label into an actuator, treat one trajectory's folds as independent treatment units,
and invent a decision-to-successor join that the rows do not contain.

## 3. Per-regime action-support matrix

Status semantics are intentionally stricter than the generic FORE checker's single status:

- `SUPPORTED_EVALUABLE`: joined executed action/transition, positive propensity, initial and
  one-step coverage, transition-sufficient state, observed preregistered reward, and whole-run
  future-fold support all close.
- `NOT_IDENTIFIED`: the deterministic policy assigns zero propensity to the target alternative or
  the treatment contrast otherwise remains causally unidentified.
- `OUT_OF_SUPPORT`: the target regime or exact run/code/hardware/scorer stratum is absent from
  logged initial/one-step support.
- `SCHEMA_INCOMPLETE`: the current route can reach the cell predictively, but the actual evidence
  lacks a joined executed decision and sufficient transition/reward/fold custody.

| regime | arm | routed folds | status | decisive reason |
|---|---|---:|---|---|
| transient | `T_gp_costate_posterior` | `MEASURED-INHERITED 3` | `SCHEMA_INCOMPLETE` | routed predictor only; no executed joined transition or outer run fold |
| transient | `persistence` | `MEASURED-INHERITED 0` | `NOT_IDENTIFIED` | deterministic alternative propensity is zero |
| plateau | `T_gp_costate_posterior` | `MEASURED-INHERITED 0` | `NOT_IDENTIFIED` | deterministic alternative propensity is zero |
| plateau | `persistence` | `MEASURED-INHERITED 3` | `SCHEMA_INCOMPLETE` | routed predictor only; no executed joined transition or outer run fold |
| uncertain | `T_gp_costate_posterior` | `MEASURED-INHERITED 0` | `NOT_IDENTIFIED` | meta-lambda defer policy excludes the GP arm |
| uncertain | `persistence` | `MEASURED-INHERITED 1` | `SCHEMA_INCOMPLETE` | routed predictor only; one same-run fold cannot establish support |

`DERIVED` — There are no `OUT_OF_SUPPORT` cells only because this matrix is restricted to regimes
observed in the current sealed trajectory. A future run, code, hardware, backend, scorer/cache, or
initial-state stratum without an exact coverage receipt is `OUT_OF_SUPPORT`; it must not borrow
support from this macOS-advisory run.

`DERIVED` — `E_prototype`, `E_prototype_bregman`, and `F_bsf` remain legitimate predictive
comparators but are excluded from the action matrix. They occur in `DEFAULT_ARM_POOL`, not in
`DISPATCH_POLICY`. Adding them as OPE actions would invent an action interface and propensity.

## 4. Audit of the landed schema and checkers

The causal manifest is a real improvement, but “representable” is not “identified.”

| needed surface | landed state | audit verdict |
|---|---|---|
| run treatment/stratum | `RunTreatmentManifest` | sufficient representation; no rows for the legacy organ run |
| state `Z` | `StateSummary` + optional checkpoint hashes | compact candidate only; Markov replay unproved, RNG/controller hashes null in current producer |
| action `A` | `ActionSummary` | transition action exists, but the trainer's arm is stage and the shadow's recommendation is a separate row |
| behavior propensity | `ExplorationDecisionRow` | exact 1/0 or authorized randomized values supported, but no validated decision-to-transition join |
| reward `R` | `RewardObservation` | trainer logs next-state `negative_implied_score`; target one-boundary gain/time law is not frozen |
| successor `Z'` | `TransitionRow` | ordered boundary chain exists prospectively; no complete organ corpus or replay certificate exists |
| coverage | `CoverageReceiptRow` | generic target-policy receipt exists; no producer and no regime/stratum-specific checker gate |
| future folds | none | whole-run and chronological split identities are not represented |
| HCM apparatus | partial | aggregate pair sentinel, no typed term closure/negative controls/positive control |

### Fresh-eyes checker finding

`MEASURED-FROM-CODE` — `check_fore_support` is correctly documented as structural admission, but it
does not enforce the ticket's complete bar. Its positive unit test admits one run with one
transition and one randomized decision. The function does not require:

1. multiple independent trajectories;
2. regime-specific action overlap;
3. equality between an `ExplorationDecisionRow.decision_id/chosen_arm` and the following
   `TransitionRow.action`;
4. a one-step replay proof that `StateSummary` is Markov-sufficient; or
5. a preregistered future-fold/time-law identity.

`DERIVED` — Therefore `ADMISSIBLE_STRUCTURAL_INPUT` must remain below
`SUPPORTED_EVALUABLE`. Main review should not use the former as a numeric-OPE authorization token.
This is an apparatus-scope finding, not a condemnation of FORE or the manifest family.

## 5. Target estimand, strata, and future folds

### Estimand

`DERIVED` — The smallest evaluator-relevant regime-arm target is:

\[
\psi_{g,a}
=\mathbb E\!\left[S(Z_t)-S(Z_{t+1})
\mid G_t=g,\operatorname{do}(A_t=a),\sigma\right],
\]

where adjacent `t` and `t+1` are complete, preserved, realized-through-R organ decision boundaries
in exact stratum `sigma`. Larger is better. The estimand ID is
`organ_one_boundary_full_score_gain_v1`; its components remain `d_seg`, the nonlinear pose term,
and exact archive bytes/rate.

`DERIVED` — This one-step definition avoids silently choosing a FORE discount. A future multi-step
FQE/FORE row must additionally freeze a finite-horizon or geometric `time_law_id`. `gamma<1` is not
a tuning convenience: it changes the target.

### Exact run strata

`DERIVED` — The future join key must include git/treatment/base-checkpoint hashes; seed;
machine/backend/axis; data-order, scorer, and cache hashes; regime-schema and behavior-policy hashes;
and estimand/time-law identities. No CPU/CUDA/MLX or old/new-code equivalence is inferred.

### Future folds

`DERIVED` — Outer folds are whole independent real runs, with the policy, estimand, estimator, and
hyperparameters frozen before the held-out future run begins. Inner folds are past-only expanding
chronological decision boundaries within training runs. Random pair/epoch splits and repeated
snapshots as pseudo-runs are forbidden.

`MEASURED-FROM-CODE` — HCM-L4 requires at least two runs merely to enter its whole-run calibration
path. That is not an adoption threshold. The existing stricter real-only organ accrual/graduation
rules remain binding; synthetic rows and same-run folds do not count as trajectories.

## 6. Estimator comparison: predictive numbers retained, OPE numbers refused

| estimator | current evidence | present authority | OPE disposition |
|---|---|---|---|
| regime dispatcher | `MEASURED-INHERITED MAE 0.0015959393896760557` | predictive, one trajectory | no action-value row |
| fixed GP | `MEASURED-INHERITED MAE 0.00185206618604584` | predictive | no action-value row |
| persistence/no-change | `MEASURED-INHERITED MAE 0.002791931483929152` | predictive incumbent | no behavior-policy value row |
| behavior policy value | none | none | refused: no executed actions/rewards |
| bootstrapped FQE | none | none | refused before fit: no joined transition corpus/time law/outer folds |
| FORE direct or DR | none | none | refused before fit: current support blockers plus no cross-run/regime overlap |
| BCQ/BEAR/CQL-style conservative | none | none | refused before fit: no supported causal action set |

`DERIVED` — BCQ/BEAR/CQL-style pessimism is useful only after support exists. It may restrict
selection or lower confidence inside logged support; it cannot assign a value to the three
zero-propensity cells. FQE/FORE likewise add no authority until the transition and fold gates close.

`DERIVED` — Seven chronological folds quantify predictive error on one sequence. They do not supply
an across-run confidence interval. No OPE standard error, ESS, maximum weight, normalization error,
or influential-transition number is emitted because no ratio or value estimator was fit.

## 7. Smallest remaining D40 logging deltas

1. **`D40-L1`, decision-transition join.** At each preserved stage decision boundary, use the
   existing `ActionSummary.parameters` to carry `decision_id`, `regime_id`, behavior-policy hash,
   chosen schedule arm, and actual actuation; validate equality with the matching decision row.
   The action must change the schedule, not merely choose a forecaster. No schema-version change is
   necessary for the fields, though a strict join checker is owed.
2. **`D40-L2`, positive support or behavior restriction.** Only a separately operator-authorized
   typed policy may randomize safe eligible arms. It must record the actual seed/draw and nonzero
   propensities. Without that authority, keep deterministic 1/0 rows and restrict all targets to
   behavior; do not manufacture epsilon.
3. **`D40-L3`, reward and replay sufficiency.** At adjacent complete decision checkpoints, emit
   `organ_one_boundary_full_score_gain_v1`, with both through-R outcomes and complete
   checkpoint/resume/RNG/controller/data-order/history/scorer/cache custody sufficient to replay one
   step.
4. **`D40-L4`, coverage producer.** Emit initial-state and one-step coverage per regime, exact run
   stratum, target policy, and arm. Extend the checker so a receipt cannot be borrowed across
   regimes or unrelated runs. This is the one change that likely needs a versioned typed extension.
5. **`D40-L5`, future-run accrual.** Pre-freeze the outer whole-run split and collect independent
   real trajectories. Two runs only unlock the current HCM-L4 code path; adoption still obeys the
   existing stricter real-only organ gate.
6. **`D40-L6`, HCM-L4 apparatus.** If L4 clearance is required, log pair-custodied outcomes, typed
   loss terms/weights, preregistered negative controls, and a frozen/no-update positive control.

This is the smallest *logging* route. It does not authorize a training launch, randomization, or
action actuation.

## 8. Refusal schema and admission rule

The durable output for a zero-supported-cell audit is:

```json
{
  "verdict": "REFUSAL_SCHEMA_ONLY__NO_NUMERIC_OPE_ROW",
  "matrix_summary": "required",
  "cell_statuses": "required",
  "blockers": "required",
  "smallest_delta_ids": "required",
  "numeric_value": null,
  "confidence_interval": null,
  "importance_weight_diagnostics": null
}
```

`DERIVED` — A numeric row becomes admissible only after at least one cell is
`SUPPORTED_EVALUABLE` and the estimator was frozen before its whole-run future fold. A quiet HCM-L4
result is never an unconfounded certificate.

**Verdict scope:** `FORMULATION x CURRENT SINGLE-TRAJECTORY NON-ACTUATING ORGAN EVIDENCE`. The
regime dispatcher, GP forecaster, persistence baseline, FORE, FQE, HCM, and conservative offline-RL
families remain open after real transition/support repair.

## 9. Triality, apparatus wire-in, and pointer honesty

- **DAG:** standalone collision-free feed
  `.omx/research/organ_ope_support_first_DAG_FEED_20260713.md`; shared hot DAG untouched for main.
- **Equation:** `N/A-with-reason`. The estimand is typed here, but no OPE identification law or
  empirical anchor closes; no canonical equation is registered.
- **DSL:** no lever or behavior-changing flag was added. A future randomized schedule policy is
  default-OFF and requires separate operator GO plus typed policy review.
- **Sensitivity map:** no numeric update; all six cells are refused.
- **Pareto:** independent-run count, overlap, transition storage, scorer custody, and estimator
  uncertainty are binding; no score/byte or compute benefit is claimed.
- **Bit allocator:** non-binding; this work evaluates control decisions, not archive payload.
- **Cathedral/autopilot:** consume the four-status refusal token before any estimator fit. No
  dispatch hook is added.
- **Continual learning:** the machine-readable matrix preserves both the manifest progress and the
  remaining decision-transition/multi-run gaps.
- **Probe disambiguator:** `predictive_forecaster_selection` versus
  `executed_schedule_action_ope`; actual actuation, joined rows, and coverage receipts decide.

**Pointer delta: `NONE`.** No score, archive, evaluator result, launch state, or promotion state
changed. Files remain uncommitted for main review.

## 10. STORES CONSULTED

Full `CLAUDE.md`, full `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, top project memory,
latest Codex/council/design/directive surfaces, current lane/subagent state, `reports/latest.md`, D40
in `.omx/state/deferral_ledger.md`, the Spinning Up crosswalk ticket, committed FORE/HCM memos and
standalone DAGs, the causal-manifest build memo/source/tests and producer wiring, the sealed organ
dispatcher memo/source/backtest JSON, the costate-organ envelope/trajectory ledger, and current git
state on `main`.

