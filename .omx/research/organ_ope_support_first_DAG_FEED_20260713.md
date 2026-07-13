# Standalone DAG FEED — support-first organ OPE refusal matrix

- Date: 2026-07-13
- Lane: `lane_organ_ope_support_first_20260713`
- Checkpoint: `organ_ope_support`
- Node: `FEED-ORGAN-OPE-SUPPORT-FIRST-20260713`
- Status: `REFUSAL_SCHEMA_ONLY__NO_NUMERIC_OPE_ROW`, `research_only=true`
- Shared-DAG append: `DEFERRED_MAIN_OWN_FILES_ONLY`
- Pointer delta: `NONE`

## Parent edges consumed

```text
FEED-436-regime-dispatch
  -> one trajectory / 9 intervals / 7 predictive folds
  -> transient -> GP; plateau/uncertain -> persistence

FEED-FORE-occupancy-ratio-drift-bridge
  -> full Markov (Z,A,R,Z') + initial/one-step/action coverage
  -> deterministic zero support remains NOT_IDENTIFIED

FEED-HCM-causal-attribution-20260713
  -> unit-level treatment needs independent runs and overlap
  -> FORE transport before whole-run HCM pooling

FEED-CAUSAL-MANIFEST-20260713
  -> pact.causal_manifest.v1 representation + fail-closed structural consumers
  -> current shadow decisions deterministic, advisory, non-actuating

D40
  -> actual exploration/propensity logging remains OPEN
```

## Support-first edge

```text
legacy organ predictive folds
  -> forecaster choice is not schedule actuation
  -> workspace causal manifest inventory = 0 files / 0 transitions / 0 executed decisions
  -> current manifest checker = NOT_IDENTIFIED
  -> HCM-L4 = NO_ROWS
  -> primary falsifier FIRES
  -> REFUSAL SCHEMA, no numeric OPE row
```

Per-regime matrix:

```text
transient: GP=SCHEMA_INCOMPLETE, persistence=NOT_IDENTIFIED
plateau:   GP=NOT_IDENTIFIED, persistence=SCHEMA_INCOMPLETE
uncertain: GP=NOT_IDENTIFIED, persistence=SCHEMA_INCOMPLETE

SUPPORTED_EVALUABLE=0
OUT_OF_SUPPORT=0 only because the matrix is restricted to observed regimes;
any unseen exact run/code/hardware/scorer stratum is OUT_OF_SUPPORT.
```

## Fresh-eyes apparatus edge

```text
check_fore_support = necessary structural gate
  -X-> multi-trajectory overlap
  -X-> regime-specific support
  -X-> decision_id/chosen_arm == following transition action
  -X-> one-step Markov replay certificate
  -X-> preregistered future-fold/time-law identity

therefore:
ADMISSIBLE_STRUCTURAL_INPUT < SUPPORTED_EVALUABLE
```

This is an apparatus-scope correction, not a FORE-family or manifest-family negative.

## Estimator edge

```text
current GP / persistence / dispatcher numbers
  -> predictive walk-forward MAE only
  -X-> behavior-policy value

joined supported future rows
  -> behavior/no-change baseline
  -> bootstrapped FQE
  -> FORE direct and optional DR
  -> BCQ/BEAR/CQL-style support restriction + pessimistic selection
  -> whole-run future fold
  -> numeric row only for SUPPORTED_EVALUABLE cells
```

Conservative offline RL cannot cross a zero-propensity edge. One trajectory cannot produce an
independent-run confidence interval.

## Smallest D40 successor chain

```text
D40-L1 actual executed regime decision <-> transition join
  -> D40-L2 authorized positive propensity OR behavior-only target restriction
  -> D40-L3 adjacent complete through-R score-gain transition + replay custody
  -> D40-L4 regime/stratum initial + one-step coverage receipt
  -> D40-L5 prospectively frozen independent whole-run folds
  -> D40-L6 pair/apparatus controls when HCM-L4 clearance is requested
```

No edge in this feed authorizes randomization, launch, training, or actuation.

## Triality and consumers

- **DAG:** this standalone collision-free feed; shared append deferred to main review.
- **Equation:** none registered; identification did not close.
- **DSL:** no behavior-changing policy added. Future exploration is default-OFF and operator-GO'd.
- **Sensitivity:** no numeric regime-arm effect update.
- **Pareto:** overlap, independent runs, transition storage, scorer custody, and variance remain
  separate debts.
- **Bit allocator:** non-binding.
- **Cathedral/autopilot:** require matrix status before estimator fit; refuse every non-supported
  cell.
- **Continual learning:** matrix JSON is the durable anti-rediscovery artifact.
- **Probe disambiguator:** predictive forecaster selection versus executed schedule-action OPE.

## Authority and pointer

Machine-readable node payload:
`.omx/research/organ_ope_support_first_20260713.json`.

No trainer, scorer, evaluator, provider, GPU, archive, live run, shared DAG, or pointer was mutated.
Pointer delta: `NONE`. Files are uncommitted for main review.
