# DAG FEED — G1 worldsheet / G3 cell code — 2026-07-20T21:00Z

`axis=[macOS-CPU advisory]` · `score_claim=false` · `pointer=0.18804 UNMOVED` · `research_only=true` · `MAIN_REVIEW_REQUIRED=true`

## FEED-G1-WORLDSHEET-N600-20260720

- producer: `tools/measure_g1_worldsheet_g3_cellcode.py`
- artifact: `.omx/research/g1_worldsheet_g3_cellcode_measurements_20260720T210000Z.json#g1`
- equation: `worldsheet_transport_residual_event_rate_v1`
- consumer: `#574 post-row worldsheet elevation`
- measured state: 600 exact-pose within transitions + 599 proxy-pose cross transitions; 10 class-pair strata each; median of transition medians `0.2792 / 0.2798 px`; `E_4=8.3977% / 8.2497%`.
- verdict: `SPARSE_EVENT_GO`
- verdict_scope: `single global ground-plane-homography realization using exact within-pair poses and nearest-target-pair proxy cross poses; not the worldsheet object/family`
- blocker carried forward: `WORLDsheet_EVENT_GRAMMAR_OWED` — Lane–MyCar, Lane–Undrivable, Lane–Movable birth/death and Road–Lane jitter must be explicit states; exact cross-pair pose targets are absent.
- next admissible edge: build scorer-free event grammar + exact cross-pair target bank, then compile through typed DSL; no launch authorization in this feed.

## FEED-G3-CELLCODE-N600-20260720

- producer: exact 38-stage live batch-16 flip inventory + causal prior measurement
- artifact: `.omx/research/g1_worldsheet_g3_cellcode_measurements_20260720T210000Z.json#g3`
- equation: `argmax_cell_identity_ideal_bytes_v1`
- consumers: `#572`, `r1b5 GAP-3`
- measured state: 17,926 flips / 16,319 moderate; best joint spatial+temporal identity floor `2,724.8733 / 2,474.8396 B`; raw coordinates `62,741 / 57,116.5 B`; Road–Lane stratum `810.0446 B`.
- comparators: `realization_breakeven_bytes_v1=1,852.091296 B` (consumed by ID); deterministic production fixture `2,114 B`.
- verdict: `CELL_ID_PAYS_VS_RAW_COORDINATES_BUT_CHEAP_PRIORS_MISS_LIVE_BYTE_GATES`
- verdict_scope: `known-site ideal cell-identity stream under the measured uniform, local Potts, and same-site temporal priors; excludes site-location, candidate-set, coder-header, receiver, and realized-flip costs`
- blocker carried forward: `RECEIVER_CLOSED_SITE_PLUS_CELL_STREAM_OWED`; the optimistic floor exceeds both all-site gates and cannot authorize r1b5.
- next admissible edge: reverse-waterfill Road–Lane first, charge site grammar + candidate alphabet + headers + receiver, and stop at the canonical rate price.

## Triality closure

- DAG: this feed.
- equations: `src/tac/canonical_equations/g1g3_successor_measurements_20260720.py`.
- DSL: deliberately absent because this is `research_only=true`; future event/cell-code actuation must add a typed, provenance-bound DSL lever before launch.

