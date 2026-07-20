# DAG FEED — shared receiver R1

`feed_id=FEED-shared-receiver-r1-20260720` · `lane_id=lane_r1_shared_receiver_20260720T151631Z` · `research_only=true` · pointer unchanged

## Typed nodes

| node | evidence type | current state |
|---|---|---|
| `P` | canonical PDW2 coefficient packet | exact `138 B` raw / `133 B` Brotli; coefficient-space only |
| `Q_dense` | explicit n600 quotient field | exact `1,887,436,928 B` source; SHA-bound |
| `Z_dense` | deterministic ZIP section | exact `561,502,227 B`; `561,502,059 B` member + `168 B` overhead |
| `L_pdw1` | PDW1 label/fill carrier | n24 measured, n600 bytes derived: `496,067 B`; `d_A=0`, `d_B=0.0080695682` |
| `H_sparse` | prior EV-ranked repair prefix | n48 measured: `3` net pixels / `3,374 B`; marginal below waterline; selected `0` |
| `R_prod` | production archive receiver | refuses `receiver_consumed` with `SHARED_RECEIVER_COUNTED_SPATIAL_HARD_ORACLE_INTERSECTION_EMPTY` |
| `A_hard` | canonical admission evaluator | always refuses in R1; its conjunction is necessary but cannot grant authority until canonical contest-CPU replay and a trusted production-receiver parser derive every term |
| `G_boundary` | next reformulated generator | OPEN: curvelet/shearlet boundary generator + active-set hard-oracle preimage + `xi` |

## Edges and authority transformations

```text
P --conditioning only--> Q_dense --deterministic ZIP measurement--> Z_dense
Z_dense --561,215,547 B over gate-------------------------------> measured rate blocker

L_pdw1 --decode + real R + hard SegNet--> d_A/d_B + class/boundary split
H_sparse --Fisher/margin rank + measured prefix--> reverse-waterfill selects zero

{P, Z_dense, L_pdw1, H_sparse}
    --typed decomposition; NO post-hoc gain composition-->
    shared_receiver_admission.v1
    --fail closed--> R_prod refusal

{P as conditioning, boundary necessity, Fisher/margin, resize cells, secant/QP, xi}
    --joint solve inside one receiver-->
    G_boundary
    --actual archive + counted payload; no caller booleans-->
    canonical contest-CPU evaluator + trusted production-receiver parser [OWED]
    --derived hashes/sample count/parse/R/causality/rate/distortion-->
    A_hard
    --future enabling change only after MAIN review--> candidate admission
```

## Solver-stack hooks

- **Sensitivity:** consume winner-rival Fisher/margin and the measured residual mechanism/class/boundary splits; no Euclidean or Fourier ranking.
- **Pareto/rate:** exact gate `B<=286,680`, `d_seg<=3.39e-4`; reverse-waterfill stops below `25/37,545,489` score/byte.
- **Bit allocator:** current dominant terms are dense field (`561,502,059 B` compressed member) then PDW1 label stream (`19,386 / 19,859 B` at n24). The next allocator spends only on necessary boundary/active-set coordinates.
- **Autopilot:** no dispatch. Merely supplying artifact paths is insufficient. Future eligibility requires a canonical contest-CPU evaluator invocation and trusted production-receiver parser over durable workspace/SSD custody.
- **Continual learning:** `.omx/research/shared_receiver_r1_20260720.json` is the machine-readable decomposition/decision record; its verdict scope preserves the open family.
- **Probe disambiguation:** dense field, label/fill carrier, and sparse repair are separate formulations. Their bytes and distortions cannot be combined after the fact; the joint solve is the disambiguator.

## Reactivation predicate

Reactivate when a scorer-free boundary-coordinate generator produces an actual durable n600 archive and counted payload, and the canonical contest-CPU evaluator plus trusted production-receiver parser can derive the exact archive/inflate-output hashes, sample count, packet-mutation replay, through-R metrics, and axis. Do not reactivate merely for self-authored JSON, a smaller target packet, an affine-only uint8 witness, a proxy/MPS metric, or a target-space partition.
