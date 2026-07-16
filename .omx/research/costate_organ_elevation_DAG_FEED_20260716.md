# DAG FEED — task #516 costate-organ exact-factorized elevation

**Date:** 2026-07-16 UTC  
**Feed:** `FEED-516-costate-organ-exact-factorized-adjoint`  
**Status:** `BUILT`, `BACKTESTED-PASS-DEVELOPMENT[#205 scalar/binding only]`, `advisory_only=true`  
**Pointer delta:** `UNMOVED`  
**Actuation:** `NONE`

```text
LawRef segnet_head_rank4_linear_flipdist_v1
  + LawRef realization_necessity_preimage_per_stratum_v1
  + LawRef lane_gain_chain_composed_v1
  -> exact K = p_visible B^T diag(||dw_pair|| / G_pair) B
  -> rank-4 / all-class gauge-null structural checks
  -> differentiated-RBF temporal posterior (existing T arm)
  -> five-scalar non-negative temporal/event amplitude residual
  -> past-only inner amplitude-admission gate
  -> lambda_net LOO + past-only walk-forward + binding-AUROC tri-gate
       | PASS -> existing shadow-controller recommendation ranker
       | FAIL -> visible factorization + blocker; no recommendation authority
  -> identified-lever-only Hamiltonian DECIDE
       | unvaried lever -> duty-to-measure, never recommendation authority
  -> existing never-regress / predicted-DeltaS-per-cost ordering
  -> costate_shadow.jsonl
  -> {costate digest, witness-run introspection, live dashboard}
```

Event edge:

```text
MEASURED Lane persistence + n600 birth/death upper-bound rates
  -> DERIVED Morse-Smale saddle-node warning proxy
  + #344 NCDE basin/asymptote row
  -> STAGE_BOUNDARY_MORSE_SMALE_NCDE_ADVISORY
  -> predicted_dS=null + actuation=NONE
  -> next governed stage-boundary watch only
```

Backtest edge:

```text
#205 v752: 9 compatible intervals
  -> scalar LOO 0.003246 < persistence 0.003698
  -> scalar past-only WF 0.001852 < persistence 0.002792
  -> binding AUROC 0.82 >= 0.80
  -> BACKTESTED-PASS-DEVELOPMENT (ridge=10 selected on #205; independent validation owed)
  BUT per-class WF 0.026641 > persistence 0.010823
  -> per-class generalization remains BLOCKED / verdict scope stays instance

mod32cap: 41 verdicts + 1000 loss rows but no interval-aligned d_seg_by_class
  -> UNAVAILABLE_INSUFFICIENT_INTERVAL_SCHEMA (no #205 equivalence inference)

c2 current analysis surfaces: no run.log rows
  -> PENDING_NO_RUN_LOG (harvest when rows arrive)
```

## Triality

- **DSL:** `costate_agent_dsl.py` registers the `factorized_adjoint` sensor; the existing expert panel registers `V_exact_factorized_residual`. No new witness lever or trainer flag was invented.
- **DAG:** this FEED closes the production consumption chain above.
- **Equations:** `hybrid_exact_factorized_costate_adjoint_v1` is appended to `.omx/state/canonical_equations_registry.jsonl`; it composes the three existing LawRefs and carries the #205/per-class scope split.

## Six-hook wire-in

1. Sensitivity map: exact pair operator consumes head normals, resize-visible support, and inverse composed gain.
2. Pareto/never-regress: candidate enters the existing positive-DeltaS refusal and per-cost ranker.
3. Bit allocator: no byte actuator; critical pair pressure is exposed to the existing duty queue only. No fabricated rate effect.
4. Cathedral/autopilot: always-on observer call path consumes the overlay; actuation remains `NONE`.
5. Continual learning: backtest receipt + canonical equation are durable recalibration anchors.
6. Probe disambiguator: exact structure and learned amplitude are separately surfaced; incompatible trajectories fail closed instead of borrowing #205 authority.

## Remaining blockers

- Independent compatible trajectory (especially C2) is required before any cross-run/general law.
- Residual ridge `10.0` was selected post hoc during the #205 build; this makes #205 a development-set pass, not an independent validation.
- Per-class walk-forward must beat its persistence baseline before per-class promotion language is allowed.
- The current in-run shadow sidecar predates this landing; MAIN must review/land before the next observer refresh can durably emit the new fields.
- Any schedule/config change remains operator-GO at a governed stage boundary; this feed authorizes none.
