# Standalone DAG FEED — HCM causal attribution, CONFOUND-L4, and organ pooling

- Date: 2026-07-13
- Lane: `lane_hcm_causal_attribution_dig_20260713`
- Node: `FEED-HCM-causal-attribution-20260713`
- Status: `DESIGN_ANALYSIS`, `research_only=true`, uncommitted for main review
- Shared-DAG append: `DEFERRED_MAIN_OWN_FILES_ONLY`
- Pointer delta: `NONE`

## Parent edges consumed

`MEASURED` — The following named repository and paper surfaces were consumed read-only:

```text
72-item lever duty-to-measure queue
  -> top 3: DsegAwareTaper / HorizonWeightedMargin / StepNativeActivation

CONFOUND immune system L1/L2/L3
  -> runtime alarms / strict gates / verdict clearance

#426/#431/#436 costate organ
  -> one real trajectory / nine intervals / three regimes / predictive walk-forward gate

#319 campaign outcome credit
  -> MC-return credit family; cross-run causal assignment deferred

FORE occupancy-ratio sibling
  -> target/logged distribution transport; current transition/action support refused

Weinstein & Blei HCM
  -> collapse + hierarchical do-calculus + within-unit identification advantage
```

## Inner-plate typing

`ASSUMED` — This is the smallest faithful Pact graph used for the identification audit:

```text
run r (unit)
  Z_r,U_r -> A_r                     A_r = run-level lever, outside inner plate
  Z_r,U_r,A_r,history -> H_r,t       H = shared longitudinal optimizer/EMA/guard state

  pair p inner plate:
    B_p,H_r,t,A_r -> Y_r,p,t -> Q_r

  apparatus:
    G_r,t,Y_r,p,t -> Y_tilde_r,p,t

  epoch t is ordered, not exchangeable;
  pairs interact through shared H_r,t/trunk.
```

## Identification edge and refusal

`DERIVED` — Treatment placement and the open run-selection path give the following edge:

```text
HCM hierarchy advantage
  requires treatment variation inside the subunit plate
  + subunit positivity/instrument/backdoor conditions

Pact lever assignment
  A_r constant over all pairs and epochs
  + unblocked A_r <- U_r -> Y_r,p,t
  + no observed adjustment/instrument/front-door route
  + exact-arm positivity absent for HorizonWeightedMargin and StepNativeActivation

therefore
  identified existing-corpus run-level lever effects = 0
```

`MEASURED + DERIVED` — Current duty-state plus graph disposition:

| query | state | refusal | verdict scope |
|---|---|---|---|
| `DsegAwareTaper -> Delta S` | `NOT_IDENTIFIED` | run-level backdoor; no compatible isolated completed A/B | existing heterogeneous corpus only |
| `HorizonWeightedMargin -> Delta S` | `NOT_IDENTIFIED` | same backdoor + exact treatment support zero | never-fired exact arm only |
| `StepNativeActivation -> Delta S` | `NOT_IDENTIFIED` | same backdoor + exact registered formulation support zero | exact registered arm only |

`DERIVED` — No canonical equation is registered. The equation leg is `N/A-with-reason`: identification did not
close. Matched/randomized run A/B and a genuine interference-aware subunit intervention remain open.

## CONFOUND-L4 edge

`DERIVED` — The declared graph entails a falsification check, not a no-confounding certificate:

```text
typed causal graph + apparatus-validity precondition
  -> leave-one-run-out outcome model m_hat^(-r)(A,Z,B,H)
  -> held-out residual R = Y_tilde - m_hat^(-r)
  -> preregistered negative-control moments E[R h(N)] = 0
  -> whole-run wild/permutation calibration
  -> loss-term closure C = L_total - sum_k w_k L_k
  -> frozen/no-update positive-control sentinel

violation -> GRAPH_OR_APPARATUS_INVALID; withhold L3 clearance
quiet     -> NOT_REJECTED, never "no hidden confounder"
```

`DERIVED` — Verdict: `YES`, a graph-falsification L4 is derived. Current execution is refused on the common
telemetry schema because treatment, pair outcomes, run parents, and apparatus tags are not jointly
captured. Scope: current schema, not the L4 family.

## Organ partial-pooling edge

`DERIVED` — The current predictive gate admits shrinkage but not causal credit:

```text
current predictive gate
  paired regime x forecaster errors on one trajectory
  -> robust hierarchical model
  -> shrink theta_regime,arm toward arm/family mean
  -> posterior ranking + prior sensitivity
  -> NO causal/self-activation authority at n=1 run
```

`DERIVED` — The eight-schools lesson is split correctly:

```text
within-unit treatment variation -> identification
hierarchical Bayes              -> partial pooling after identification
```

`DERIVED` — Pooling alone cannot supply the first arrow.

## FORE composition edge

`DERIVED` — FORE transport and HCM pooling compose sequentially:

```text
HCM graph/schema gate
  -> logged transition/action support
  -> FORE omega_pi = d_pi,gamma / d_nu_log
  -> cross-fitted DR pseudo-outcome
  -> aggregate by independent run/regime
  -> HCM partial pooling over supported regime-arm cells
  -> support/ESS/prior-sensitivity decision gate
```

`DERIVED` — FORE corrects occupancy shift; HCM represents causal hierarchy and heterogeneity. Neither repairs
the other's missing condition. Structural zero support yields `NOT_IDENTIFIED`, never a pooled
prior-only estimate.

## Triality and wire-in

`DERIVED` — Triality disposition and prospective six-hook wire-in:

- **DAG:** this collision-free standalone feed; shared DAG deferred to main.
- **Equation:** none registered; candidate diagnostics remain in the parent memo only.
- **DSL:** no flag or live controller. A future typed causal-attribution spec is default-OFF and
  needs graph, treatment, outcome, clustering, negative-control, support, and scope identities.
- **Sensitivity:** no numeric lever update; current effect is `NOT_IDENTIFIED`.
- **Pareto:** independent-run accrual, overlap, custody, telemetry/storage, and variance are binding.
- **Bit allocator:** non-binding.
- **Cathedral/autopilot:** future refusal tokens for unpaired unit treatments, zero support,
  apparatus invalidity, and row-level pseudo-replication; no dispatch now.
- **Continual learning:** parent memo plus this feed preserve the theorem-to-corpus boundary, L4
  falsification check, and FORE-before-pooling composition.
- **Probe disambiguator:** `run_randomized_unit_treatment` versus
  `interference_aware_subunit_treatment` versus `predictive_only_partial_pooling`.

## Pointer and collision custody

`MEASURED` — No trainer, scorer, evaluator, provider, GPU, archive, live run, shared DAG, canonical equation,
DSL, controller, or sibling deliverable was mutated or launched. The parent memo and this feed are
the only writes and remain uncommitted for main review. Pointer delta: `NONE`.

## Primary source

`MEASURED` — Weinstein and Blei, *Hierarchical Causal Models*, JMLR 27(37), 2026:
https://www.jmlr.org/papers/v27/25-0899.html . See the parent memo for the complete store and theorem
audit.
