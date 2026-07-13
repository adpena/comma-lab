# Standalone DAG FEED — causal/transition manifest apparatus

- Date: 2026-07-13
- Lane: `lane_causal_manifest_build_20260713`
- Node: `FEED-CAUSAL-MANIFEST-20260713`
- Status: `LOCAL_BUILD`, `research_only=false`, uncommitted for main review
- Shared-DAG append: `DEFERRED_MAIN_OWN_FILES_ONLY`
- Pointer delta: `NONE`

## Parent edges consumed

```text
FEED-FORE-occupancy-ratio-drift-bridge
  requires full ordered (Z,A,R,Z') rows + explicit initial/one-step target coverage

FEED-HCM-causal-attribution-20260713
  requires typed treatment/run custody + pair outcomes + apparatus tags
  + whole-run residual moments + loss closure + frozen/no-update positive control

D40 organ causal-OPE deferral
  requires logged arm alternatives and actual behavior propensities/randomization custody
```

No convergence premise was re-derived. These three already-settled reader requirements are the
input contract for this build.

## Build edge

```text
exact run treatment manifest
  {run id, typed argv treatment, base checkpoint/hash, seed, machine/backend/axis,
   data-order identity, stage plan, scorer/cache hashes}
        |
        v
ordered boundary summaries Z_t
  {epoch/stage/policy, data-order cursor, checkpoint/resume custody,
   liveness/apparatus state, realized-through-R outcome}
        |
        +-- actual interval action A_t
        +-- preregistered reward R_t or explicit unobserved reason
        v
append-only transition (Z_t, A_t, R_t, Z_t+1)

costate shadow recommendation
        |
        v
typed exploration decision
  {chosen arm, all alternatives, logged propensities, policy hash,
   exploration authorization state, actual seed/draw if randomized, executed, actuation}
```

`MEASURED` — the implementation is `tac.causal_manifest` with schema id
`pact.causal_manifest.v1`. Every row is a frozen dataclass, validates before append, and persists via
the canonical `tac.jsonl_store.append_locked_jsonl` fcntl/fsync path.

`DERIVED` — asynchronous late verdicts are retained as boundary observations but do not create a
backward transition. This preserves evidence without claiming a false temporal edge.

## Consumer edges and current refusals

```text
causal manifest rows
  -> check_fore_support(...)
       requires treatment manifest + observed transitions + explicit coverage receipt
       + executed decisions with positive target-arm propensity
       missing condition -> NOT_IDENTIFIED

causal manifest rows + caller-supplied leave-one-run-out predictions
  -> hcm_l4_residual_check(...)
       requires per-pair custody + captured scorer/cache hashes + >=2 whole runs
       + exact weighted-loss closure + preregistered negative controls
       + frozen/no-update positive control that actually triggers
       invalid custody -> INVALID_INPUT
       apparatus break -> REFUSED_APPARATUS
       moment violation -> FIRED_GRAPH_FALSIFICATION
       quiet -> QUIET_NOT_CERTIFIED, never an unconfounded certificate
```

`MEASURED` — current trainer boundary transitions use `pair_id=__aggregate_all_pairs__` and do not
invent loss-term decomposition at checkpoint-only rows. Therefore they seed ordered transition
custody but intentionally fail the strict pair-level HCM gate and cannot by themselves admit a FORE
cache.

`MEASURED` — the current costate-shadow decision policy is deterministic, advisory only,
`executed=false`, `actuation=NONE`, and logs 1/0 propensities. Therefore unchosen arms remain
`NOT_IDENTIFIED`. The schema contains a validated randomization hook, but randomization requires
`exploration_hook=externally_authorized` plus the actual seed and draw; no operator GO was assumed.

## Triality and six-hook wire-in

- **DAG:** this collision-free standalone FEED; shared DAG append is deferred to main.
- **Equation:** no new canonical equation. The read-only HCM stub implements the custody/skeleton of
  parent equations (2)--(5); the FORE checker implements only parent admission hypotheses, not the
  estimator or a claimed contraction certificate.
- **DSL:** `N/A-with-reason`. This schema is default-ON, score-neutral apparatus, not a score lever or
  treatment. Adding a new trainer flag would create an orphanable off-state and parallel config
  vocabulary. A future randomized organ policy is behavior-changing and must enter the typed DSL in
  its own operator-GO'd landing.
- **Sensitivity map:** non-binding now; future identified per-arm effects may feed it only after the
  support and HCM gates clear.
- **Pareto:** logging/storage/hash startup cost and transition coverage are explicit constraints;
  no score/Pareto claim is made.
- **Bit allocator:** non-binding; no archive section or decode payload is created.
- **Cathedral/autopilot:** `NOT_IDENTIFIED`, `INVALID_INPUT`, and `REFUSED_APPARATUS` are strict
  read-only refusal surfaces. No dispatch/actuation hook is added.
- **Continual learning:** causal rows are the reusable typed signal. Deterministic 1/0 propensity is
  preserved as a negative support fact rather than smoothed into fake exploration.
- **Probe disambiguator:** future policy admission compares deterministic walk-forward-only versus an
  externally authorized randomized schedule. The logged propensity/coverage receipt arbitrates; this
  build does not choose the behavior-changing branch.

## Pointer and execution custody

`MEASURED` — `$0` local only. No trainer was launched, no evaluator/scorer/GPU/provider was invoked,
no run directory was modified, no archive was produced, and no shared DAG/DSL/equation/deferral row
was appended. Source and standalone research artifacts remain uncommitted for main review. Pointer
delta: `NONE`.

