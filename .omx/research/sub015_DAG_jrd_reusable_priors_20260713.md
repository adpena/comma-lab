# FEED-jrd-priors-20260713 — JRD reusable-prior routing DAG

`research_only=true` · `score_claim=false` · `pointer_moved=false` · `$0 LOCAL` · no #336 or V9-spec edits

## Executable dependency graph

```text
PR110 exact archive + JRD response JSON
  ├─ n=1 component-split rows ──> DORMANT tensor-rank prior
  │                                └─[real-GT n600 NumPy-fp32 exact-R gate]─> measurement initializer
  ├─ pose-only n=1 rows ─────────> R1 dxi composition CANDIDATE
  │                                └─[n600 joint receiver-closed proof]────> admit or reject
  └─ exact component-safe gain=0B ─> STOP same PR110 uniform/Laplace rerun
                                   └─> TRAIN-TIME structure route (#110/#242/#311)
                                       └─> witness response instrument (#336 owner; route only)
```

## Canonical task rows

| Task row | State | Producer | Consumer | Hard gate / verdict scope |
|---|---|---|---|---|
| `jrd_n600_tensor_prior_reactivation_20260713` | `PENDING_DORMANT` | `tac.witness_dsl.jrd_priors` | Agent A / #336 measurement order only | Real-GT n600, NumPy-fp32 bit-identical, exact R, split d_seg/d_pose, exact bytes. n=1 is SCREEN only. |
| `jrd_pose_decoupling_r1_composition_20260713` | `PENDING_CANDIDATE` | n=1 pose-only row classifier | R1 dxi witness carrier | Must prove the sidecar/carrier is actually read and absorbs pose debt jointly at n600; no post-hoc absorption assumption. |
| `jrd_training_time_entropy_edge_route_20260713` | `ROUTED` | `jrd_component_safe_entropy_edge_stop_v1` | #110 latent structure + #242 MDL/entropy + #311 low-tau TropNNC | Strengthens training-time mechanism; does not reopen or overwrite any existing measured-negative operating point. |
| `jrd_witness_rate_instrument_route_20260713` | `ROUTED_NOT_EDITED` | existing JRD response-curve tool | Agent A / #336 | Owner must run on witness and n600; this lane does not edit #336. |
| `jrd_v9_cgauge_entropy_route_20260713` | `ROUTED_NOT_EDITED` | entropy-edge law + DSL policy | ideal-config owner | Spec delta candidate: compile training-time rate/structure arm plus n600 response receipt; this lane does not edit `spec_v9_cgauge.py`. |
| `jrd_pr110_exact_rerun_stop_20260713` | `COMPLETE_STOP` | PR110 n600 receipt | scheduler | `FORMULATION x INSTANCE`: same archive + post-hoc uniform/Laplace prefixes only. |

## Triality

- DSL: `tac.witness_dsl.jrd_priors.JrdReusablePriorPolicy`.
- Equation: `jrd_component_safe_entropy_edge_stop_v1`.
- DAG: this FEED.
- Durable decision memo: `.omx/research/jrd_reusable_priors_harvest_20260713.md`.

## Pointer-delta honesty

No candidate bytes were created, no contest evaluation was run, and no frontier pointer moved.
