# FEED — DDM MS6b receiver-support resume

`research_only=true` · `score_claim=false` ·
`main_landing_review_required=true`

## Executable trajectory

```text
SHA-bound V19C archive + PF2 atlas + MS5 assignment vocabulary + frozen scorer
  -> governed four-thread SSD resume
  -> 748 atomic v2 signed-probe checkpoints
  -> loader-schema PF2 assignment table
  -> per-actuator support + per-bucket join + sign-asymmetry summary
  -> exact G3 top24 pair-by-bucket conjunction
       false: 106 missing blocks, 1/24 pairs complete
  -> MS4 HOLD (not invoked)
  -> pointer 0.1910828242 [contest-CPU] UNMOVED
```

The producer gate is executable through
`tools/summarize_ddm_ms6_receiver_support.py`; it does not infer causal joins.
It admits a G3 block only when the hard pair is both a PF2 member of a bucket
and present in that bucket's measured `pair_ids`.

## Triality

- DSL/code leg:
  `tools/measure_ddm_ms6_receiver_support.py`,
  `tools/summarize_ddm_ms6_receiver_support.py`, and the MS5 typed loader
  schema.
- DAG leg: this feed, with the false G3 conjunction visibly preventing MS4.
- Equation leg:
  `ddm_ms6b_receiver_support_resume_canonical_equations_20260724T063800Z.md`.

## Unified-solver wire-in

This lane is research-only and does not dispatch or mutate the frontier. Its
durable output is the typed assignment table and exact missing-block set. Those
are the sensitivity-map/probe-disambiguation inputs for a future separately
claimed receiver-extension lane. Pareto, bit allocation, autopilot actuation,
and posterior promotion remain fail-closed until G3 coverage and real MS4
artifacts exist.
