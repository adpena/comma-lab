---
title: "Research-only DAG FEED: VR-GHAL to #455/#454"
date_utc: "2026-07-13"
research_only: true
actuation_authority: false
---

# Research-only DAG FEED: VR-GHAL to #455/#454

## Nodes

| Node | Type | Status | Authority |
|---|---|---|---|
| `PAPER_2607_09097_ABSTRACT` | source | `MEASURED` abstract only | rate exponents and stated assumptions |
| `PAPER_2607_09097_THEOREMS` | source | `BLOCKED/UNKNOWN` | no exact constants or recursion |
| `EQ_VRGHAL_455_MOVING_OPERATOR_DEBT` | equation | `DERIVED` | applicability guard |
| `EQ_VRGHAL_455_QUERY_TO_TEACHER` | equation | `DERIVED` | teacher-call accounting guard |
| `EQ_VRGHAL_454_CLIPPED_TAIL_DEBT` | equation | `DERIVED` | reuse safety guard |
| `FEED_455_FIXED_REPLAY_CONVEX_HEAD` | feed | `PLAUSIBLE/NOT-ADMITTED` | design prior only |
| `FEED_454_MONITOR_ONLY` | feed | `PLAUSIBLE/NOT-SAFETY` | observability prior only |
| `LIVE_455_ARM` | actuator | `NO EDGE` | sibling-owned |
| `LIVE_454_CERTIFICATE` | actuator | `NO EDGE` | sibling-owned |
| `FRONTIER_POINTER` | result | `NO EDGE` | byte-closed exact row required |

## Directed edges

`PAPER_2607_09097_ABSTRACT -> EQ_VRGHAL_455_MOVING_OPERATOR_DEBT`

`PAPER_2607_09097_ABSTRACT -> EQ_VRGHAL_455_QUERY_TO_TEACHER`

`PAPER_2607_09097_ABSTRACT -> EQ_VRGHAL_454_CLIPPED_TAIL_DEBT`

`EQ_VRGHAL_455_MOVING_OPERATOR_DEBT -> FEED_455_FIXED_REPLAY_CONVEX_HEAD`

`EQ_VRGHAL_455_QUERY_TO_TEACHER -> FEED_455_FIXED_REPLAY_CONVEX_HEAD`

`EQ_VRGHAL_454_CLIPPED_TAIL_DEBT -> FEED_454_MONITOR_ONLY`

No edge reaches a live actuator because the fixed-map, geometry, oracle, and call-accounting gates
have not passed. No edge reaches the frontier pointer because there is no `n600` measured reduction
or byte-closed exact row.

## Rejection routing

- `verdict_scope=instance`: current live nonlinear moving-distribution #455 map does not fit.
- `verdict_scope=formulation`: direct VR-GHAL wrapper around that map is `NO-GO`.
- `verdict_scope=family`: fixed-replay convex-head stochastic fixed-point training remains open.
- `verdict_scope=paradigm`: stochastic fixed-point methods are not rejected.
- `verdict_scope=#454 formulation`: clipping as the safety certificate is `NO-GO`; monitoring-only
  use remains open.

## Deferred executable wire-in

The DSL leg is deliberately deferred. The live trainer/certificate files are owned by sibling lanes,
and this feed has no admission authority. Main may register these equations and attach typed,
default-off gates only after the paper theorem body and live constants are in custody.

