# DAG FEED — R1b6 admissible carrier

`FEED-R1B6-ADMISSIBLE-CARRIER-20260720`

## Node update

- `R1B6_RECEIVER_BOUND_SINGLETON_PREFIX`: `MEASURED_NEGATIVE_STOP`.
- Evidence: n16, 512 requested Fisher/Road–Lane-first cells, 498 exact uint8
  endpoints, sealed R1B4 double decode, hard CPU-Torch batch16.
- Delta: 7 additional Seg flips, slight pose regression, combined
  `Delta S_recovery=-0.00066067449`; formulation break-even `0 B`.
- `verdict_scope=n16 source-closest-sign same-rounded-bin absolute replay only;
  no n600 or carrier-family negative`.

## Dependency changes

- `realization_breakeven_bytes_v1` -> `DOMAIN_REFINED`: preserve the existing
  1,852.091296-byte R2b n600 anchor; forbid transfer to R1B4 absolute replay
  without fresh hard-oracle evidence; forbid prefix-as-n600 use.
- `compact_binary_v2_1273_projection` ->
  `KILLED_CURRENT_FORMULATION_PARSEBACK_FALSE_AND_OPERATIONAL_PREFIX_NEGATIVE`.
  Reactivation requires a counted parser/receiver grammar with positive hard
  realization and exact marginal bytes.
- `rank4_secant_production_custody` remains `BLOCKED`: prefix endpoint evidence
  is not the required typed tangent plus realized-secant n600 tensor custody.
- `full_kernel_compact_replay` -> `PARTIAL`: search-free absolute replay and
  n16 runtime proved; n600 MDL selection/compact replay/positive hard admission
  remain absent.
- `vjp_full_sidecar_rehash` remains `DEFERRED_FAIL_CLOSED` until producer inputs
  exist.

## Solver-stack wire-in

- Sensitivity map: consumes the SHA-bound Fisher/margin ordering; no Euclidean
  or Fourier rank was introduced.
- Pareto constraint: measured marginal value is below the contest byte price,
  so reverse-waterfill stops before other-edge/nonedge strata.
- Bit allocator: current absolute replay costs `22,891 B` incremental for 498
  sites and has negative value; allocate zero bytes to this formulation.
- Cathedral/autopilot: no n600 compile or dispatch; reactivation requires a new
  receiver-bound cell grammar or secant/QP formulation, not a larger sweep of
  the same sign rule.
- Continual learning: machine receipt plus equation domain-refinement row 765.
- Probe disambiguator: source-closest sign was measured and lost; a future
  secant/QP-selected sign is a distinct formulation and remains open.

Pointer `0.1910828242 [contest-CPU]` unchanged.  MAIN landing review required.
