---
schema: ddm_ms2rp_partial_coverage_box_fit_dag_feed.v1
feed_id: FEED-613-ms2rp-partial-coverage-box-fit
lane_id: ddm_ms2rp_partial_coverage_box_fit
research_only: true
pointer_moved: false
main_landing_review_required: true
verdict_scope: PRECONDITION/INSTANCE
---

# Trajectory

```text
RG4 cb34bbb0f1
  |-- 25/25 production-alphabet obstructions typed
  |-- partial nonexcluded coverage proven
  `-- 34/1200 event-incidence joins
          |
          |  DOES NOT CREATE
          v
MS4D/PF3 materialization conjunction
  |-- receiver builder                 0
  |-- realized uint8 quantum           0
  |-- candidate delta                  0
  |-- same-object dimension rate home  0
  `-- coder payload owner              0
          |
          v
EV2/RD1 finite prices                 0/162
MS2R-R3 measured typed rungs          0
          |
          v
PARTIAL MEMBER                        ABSENT
n600 receiver/R/uint8/scorer/coder    NOT INVOKED
IN_BOX / OUT_OF_BOX                    NOT REACHED
```

# Feed disposition

- Preserve RG4's 25 exclusions as real instance signal; do not rerun the
  exhausted production alphabet and do not build RG5 in this lane.
- Route the next build to PF3 same-object materialization: one
  scorer-recursive coordinate, deterministic receiver builder, realized
  uint8 quantum, candidate delta, rate home, and coder owner.
- After a finite RD1 price exists on that same object, resume this exact
  config, set the 25 RG4 blocks to zero allocation, materialize once, and run
  the n600 byte-close.
- Do not use the exact C1 or finite q4/q8 controls as the missing typed rung.
- Do not derive excluded-block d_seg from PF2 event counts. The error ownership
  field remains null until measured on the candidate object.

# Triality

- DSL/config:
  `.omx/research/configs/ddm_ms2rp_partial_coverage_box_fit_20260725.json`
- DAG/FEED: this document
- Equations: no new law; reuse the #613 box and existing PF3 materialization
  conjunction. Canonical-equations debt is explicitly zero until a same-object
  rate/reach law is measured.
- Receipt:
  `.omx/research/ddm_ms2rp_partial_coverage_box_fit_20260725T185945Z/receipt.json`

Pointer delta: **NONE**. This is a strict apparatus precondition, not a
describe-line family verdict.
