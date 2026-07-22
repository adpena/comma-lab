---
task: 603
feeds_task: 613
lane_id: lane_ddm_v4_stratum_structured_members_20260722
research_only: true
triality_leg: DAG
main_landing_review_required: true
---

# FEED-603 -> 613: structured-stratum membership DAG

```text
S4 0.bin + runtime + target receipt
  -> typed DDM v4 config compile
  -> per-stratum deterministic encoders
       Road/Undrivable: PXQ1 + class-filtered PCE3/PCOMP3
       Lane: exact LBND2 decode/raster + class-filtered events
       MyCar: PXQ1 static hood
       Movable: class-filtered S4 events
       bulk: cheap baseline
  -> exact deterministic ZIP bytes
  -> receiver parse-back + canonical recompile equality
  -> frozen-SegNet membership matrix + Pose completeness
  -> immutable candidate-stage checkpoints
  -> receipt with hashes, byte homes, custody, and scoped verdict
  -> FEED-613: Lane is a positive representation rung; Road/MyCar/Movable stay blockers
```

## Executable readiness state

| node | state | evidence / reason |
|---|---|---|
| source-boundary hygiene | PASS | S4 archive/runtime/target hashes bound in receipt |
| typed DSL compile | PASS | deterministic compiled-program hash in receipt |
| receiver closure | PASS | parse-back plus canonical recompile equality for all six archives |
| per-stage resumability | PASS | distinct immutable candidate checkpoints, stop/resume supported |
| n64 structured membership | PASS | Lane target membership `0.686570162333` at 84,918 bytes |
| Road semantic routing | BLOCKED | Road role maps primarily to MyCar, target Road remains zero |
| MyCar/Movable efficacy | BLOCKED | target-class membership remains zero at n64 |
| exact evaluator replay | NOT RUN | research-only local instrument; no d_seg/d_pose authority |
| n256/n600 closure | NOT RUN | bounded arm stopped after required n64 |
| promotion/frontier move | REFUSE | no contest-CPU/CUDA exact replay or custody-complete score |

## Unified-stack wire-in

1. Sensitivity-map contribution: the class-by-member membership matrix is the reusable local
   stratum sensitivity signal; no score sensitivity is inferred.
2. Pareto constraint: archive bytes and target-class membership are jointly reported; Pose
   completeness is a hard feasibility constraint.
3. Bit allocator: the event-prefix curve exposes exact marginal bytes for each class subset.
4. Cathedral/autopilot: #613 receives only the positive Lane representation rung and explicit
   Road/MyCar/Movable blockers.
5. Continual learning: the dated receipt/register row is the empirical anchor; no posterior is
   promoted from a non-authority scorer.
6. Probe disambiguator: Road palette routing versus carrier geometry remains a two-hypothesis
   blocker and requires a future paired probe before design choice.

