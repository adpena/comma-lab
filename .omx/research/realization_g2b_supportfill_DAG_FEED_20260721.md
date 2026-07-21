---
schema: realization_g2b_supportfill_dag_feed.v1
task_id: "578"
lane_id: lane_realization_g2b_supportfill_578_20260721
research_only: true
status: D2_ZERO_BYTE_SEMANTIC_CELLS_TO_RGB_ADMISSION_FALSE
main_landing_review_required: true
---

# DAG FEED: realization G2b support-fill

```text
seed_compose_b2 bytes
  -> scorer-free seed parser
  -> HxW class-ID cells                         [MEASURED n600]
  -> cells-to-RGB receiver                      [MISSING / BLOCKER]
  -> two HxWx3 RGB scorer planes
  -> canonical support-fill / factor-2 lattice [MEASURED exact source control]
  -> two camera uint8 frames                    [MEASURED double-decode exact]
  -> frozen SegNet + PoseNet                    [MEASURED macOS-CPU advisory]
  -> predict_project_realization_admissibility_v1
                                                 [MEASURED FALSE]
```

The source-RGB control bypasses the missing edge with encoder-supplied, counted dense planes. It proves the nodes below that edge and nothing above it. At n600 it has factor-2 exact fraction 1.0, 600/600 deterministic double decodes, 600/600 pose-tube rows, 0/600 semantic-exact rows, and 707,788,800 added raw RGB bytes.

## Consumers

- Sensitivity map: retain per-class/per-stratum survival counts; do not treat target d_seg as seed efficacy.
- Pareto constraint: hard-refuse zero-byte realization unless the registered conjunction is true.
- Bit allocator: charge any RGB payload; current source control is not a buy.
- Cathedral/autopilot: route to construction of a real receiver-side cells-to-RGB map, not another lattice audit.
- Continual learning: consume the registered empirical false anchor and the premise-falsification memo.
- Probe disambiguator: two interpretations are now explicit—receiver-derived RGB versus encoder-supplied counted RGB—and the admission predicate arbitrates them.

Pointer `0.1910828242 [contest-CPU]` remains unchanged. MAIN review is required before landing.
