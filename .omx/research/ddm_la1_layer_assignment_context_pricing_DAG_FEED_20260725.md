# DDM LA1 layer assignment and context pricing — DAG / FEED

`research_only=true` · `score_claim=false` · `pointer_moved=false` ·
`main_landing_review_required=true`

## Trajectory

```text
C1 exact 133,941-byte ZIP
  + J2 exact seeded reconstruction (134,211 B; Lane home +270 B)
  + LP1 typed allocation (134,211 B)
  + EV2 seven-home partition and stream/cell firewall
  + G4 zero-parameter decoder context
  + CC2/CC3 Bellard-KT implementation
  + E4 Brotli-Q11 / raw-LZMA1 discipline
  + GA1 DOMINATED disposition (not reopened)
        |
        v
seven exact whole-stream payloads
        |
        +--> explicit frames: raw / Brotli-Q11 / raw-LZMA1
        |
        `--> context frames: G4 / Bellard-KT
                    |
                    v
           uniform 46-byte framing
           exact parse-back + deterministic re-encode
                    |
                    +--> #669(b) RESIDUAL vs CONTEXT table
                    |       `--> CONTEXT mass 355 / 134,211 < 1%
                    |
                    `--> #669(c) deepest-home assignment table
                            `--> 128,254 B LA1 alternative
                                      |
                                      v
                    min(overlapping alternatives, never sum deltas)
                    min(130,789 CC3, 128,254 LA1) = 128,254 B
                                      |
                                      v
FEED-603-la1 --> E5 receiver-closed export composition
                + payload-consumption bijection
                + exact receiver output identity
                + only then composition-accounting replacement
```

## FEED-603-la1

| field | value |
|---|---|
| feed id | `FEED-603-la1` |
| producer | `lane_ddm_la1_layer_assignment_context_pricing_20260725` |
| consumer | E5 export composition |
| payload | seven selected codec/frame identities, exact raw SHA-256 values, deepest layer homes, and payload-cleanliness rows |
| current confirmed accounting | 130,789 B post-CC3 |
| prospective accounting | 128,254 B |
| prospective delta | -2,535 B |
| required admission | rebuild receiver-closed export; prove selected-frame decode, receiver-consumption bijection, and exact output identity |
| prohibited inference | do not add CC3 and LA1 deltas; do not allocate rate to EV2 cells; do not claim a score from lossless stream parse-back |
| verdict_scope | `INSTANCE: exact SHA-bound C1/EV2 seven-home accounting object x current coder implementations` |
| disposition | `QUEUE_E5_RECEIVER_INTEGRATION; CURRENT_130789_UNCHANGED_UNTIL_CLOSED` |

## Unified-action wire-in

| consumer surface | LA1 contribution | authority boundary |
|---|---|---|
| sensitivity map | two CONTEXT-winning whole-stream rows and five RESIDUAL-winning rows | stream rate only; no cell sensitivity inferred |
| Pareto constraint | lossless coding fixes `delta d_seg = delta d_pose = 0` only after receiver identity; rate candidate is -2,535 B versus current plan | prospective until E5 |
| bit allocator | replaces per-cell allocation with exact seven-stream price rows | EV2 stream/cell firewall binding |
| cathedral autopilot | E5 queue row with selected codec and frame SHA per stream | no dispatch or launch authority |
| continual learning | CONTEXT `<1%` falsifier at INSTANCE scope plus reopener | no family closure |
| disambiguator | CC3 and LA1 are competing overlapping layouts; `min`, never additive delta | exact coordination law |

## Triality

- DSL/data:
  `.omx/research/configs/ddm_la1_layer_assignment_context_pricing_20260725.json`
  and
  `.omx/research/ddm_la1_layer_assignment_context_pricing_20260725T112908Z/receipt.json`
- DAG: this memo.
- Equations:
  `.omx/research/ddm_la1_layer_assignment_context_pricing_canonical_equations_20260725.md`

MAIN landing review is required before this feed is consumed.
