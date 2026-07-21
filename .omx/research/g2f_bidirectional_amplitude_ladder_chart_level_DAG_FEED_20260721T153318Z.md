---
schema: g2f_bidirectional_amplitude_ladder_chart_level_dag_feed.v1
task_id: "578"
lane_id: lane_g2f_chart_bidirectional_amplitude_ladder_578_20260721
research_only: true
status: MEASURED_CHART_TRUST_RESCUES_PIXEL_EMPTY_N64_FAMILY_OPEN
authority: "[macOS-CPU advisory]"
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
main_landing_review_required: true
---

# DAG FEED: G2f level-attributed amplitude ladder

## Measured path

```text
preserved pixel-level n64
  -> 5/64 selected pairs
       pixel-only = {24,31,63}
       overlap = {22,30}

coherent centerline-intercept chart n64
  -> same numeric rungs [0.5,1,2,4,8,16]
  -> full AA-SDF raster + frozen Lane/Road palette + exact R + CPU SegNet
  -> chart knee 0.5 native centerline pixels
       17/116 usable trust rows at knee
       28/696 usable trust rows overall
  -> 6/64 selected pairs
       chart-only = {0,34,37,46}
       overlap = {22,30}

level partition
  -> chart-only 4 / pixel-only 3 / both 2 / neither 55
  -> pair-conditional alphabet recommendation
  -> chart packet NOT BUILT
  -> receiver admission NOT RUN
  -> 0 correction bytes / pointer unchanged
```

## Fail-closed routing

```text
nonempty chart trust without receiver packet
  -> do not call chart rows admitted
  -> do not append a false hard-admission anchor
  -> do not spend +95,094-byte headroom
  -> refuse n600
  -> build counted chart-symbol packet at n16 pair 0
  -> require parse-back + hard Seg/Pose + rate admission
  -> only positive n16 may authorize n64 packet replay
  -> only positive n64 may authorize n600
```

## Six-hook wire-in

1. Sensitivity map: keep pixel and chart response fields separate. Attach chart
   knee `0.5`, pixel knee `1.0`, their physical units, and the exact four-way
   pair partition; never pool unequal-dimensional directions.
2. Pareto constraint: allocation stays at zero until counted chart-symbol bytes
   reconstruct the coefficient and pass full receiver closure. Trust alone is
   not Pareto admission.
3. Bit allocator: use chart symbols only on chart-only rescues, retain pixel
   candidates on pixel-only pairs, and price both packet forms on overlap pairs.
   The 55 neither pairs receive no symbol from these measured alphabets.
4. Cathedral/autopilot dispatch: rank n16 chart-packet parse-back for pair 0
   ahead of any n600 chart measurement. This feed emits no launch authority.
5. Continual-learning posterior: consume the strict chart receipt as a typed
   measurement record. Defer hard-admissibility anchoring until the packet and
   receiver predicates are actually evaluated.
6. Probe disambiguator: compare counted chart versus pixel packet value on
   overlap pairs `22,30`, while preserving chart-only and pixel-only controls;
   stop each stream below `25/37,545,489` marginal score units per byte.

## Triality delta

- DSL/code: typed chart custody plus `--chart-amplitude-ladder` makes level
  explicit inside the existing resumable G2 runner.
- DAG: four disjoint pair sets route to chart, pixel, packet A/B, or stop.
- Equations: `predict_project_realization_admissibility_v1` remains unchanged;
  no chart anchor is fabricated before receiver evaluation.

## Custody

- receipt:
  `/Volumes/VertigoDataTier/pact/evidence/g2f_chart_amplitude_20260721/receipt.json`
- file SHA-256:
  `47d3ca538f1b876f7639223a1a9a7714b7db2083eaa0971936b9a43a1e6d0d04`
- canonical receipt SHA-256:
  `2e0ccc5a6822b584ad66d7dc522551684ef20c358628b20ebdbb9bdd02cfe120`
- config SHA-256:
  `8b357c1d9c7c7ac5257cc67e996851597d3989f3c2921e9c60e28b319550e055`
- as-run source patch:
  `.omx/research/g2f_chart_measurement_runtime_sources_20260721T153318Z.patch`
  at SHA-256
  `e2dca7957ee3d96b9d5a116077ee3c2fcf7ede52e28c5c640b62cfaf35a16d5d`,
  verified against base `c9abc61b2e` and both receipt implementation hashes
- verdict scope: exact contiguous n64, one support-maximizing lane line and
  centerline intercept per pair, coherent chart raster through exact R and
  native CPU SegNet; no packet, admission, n600, score, promotion, other-chart,
  pixel-family, quotient-family, or integer-lattice-family claim.

MAIN landing review is required before this DAG feed becomes canonical.
