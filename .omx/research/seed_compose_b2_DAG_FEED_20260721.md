# seed_compose_b2 DAG FEED — 2026-07-21

```text
frozen CPU-Torch n600 cache cf8d8360...
  + S2 finite event seed df4c0534... (17,926 events / 39,836 B)
  + G1 PoseNet->xi LawRefs (s_t=-0.00143, s_r=0)
  + canonical movable-site correspondence
  -> five-site compatibility chart [native MS blocked]
  -> 3 nested PPCS curve points
  -> n16 -> n64 -> n600 cache-replay description oracle
       |-- cell exact: PASS
       |-- Pose tube: PASS
       |-- deterministic double decode: PASS
       `-- camera RGB / uint8 factor-2: FAIL CLOSED
             -> terminal rungs 2..9 NOT REACHED
```

- DSL leg: strict `predict_project_constraint_seed.v0`, exact seed 1234/batch16 callback contract.
- DAG leg: all 600 pair rows are preserved and exposed as a block-diagonal vector; shared chart/trajectory/rate are the only cross-pair couplings.
- Equation leg: G1 constants resolve through `dsl_custodied_scalar_identity_v1`; measured allocation marginals are compared to `25/37,545,489` and yield `BOUNDARY_NO_INTERIOR_KNEE`.
- Exact blocker: `MS_ARC_TO_CELL_RASTERIZATION_SEMANTICS_UNMEASURED`.
- Next authorizing edge: bind this object to a camera-RGB inverse-R realization with realized-entry telemetry. Only then may #400/#396/#553/JRD/#557/R6 execute.
- Measurement artifact: `.omx/research/seed_compose_b2_measurements_20260721.json` SHA-256 `5108ae6ab4febf0c1d8f22c5f978224a803d25b332159c48a7e2e74130509205`.
- Pointer delta: none.
