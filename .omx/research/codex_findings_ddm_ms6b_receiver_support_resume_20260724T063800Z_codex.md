# Codex findings — DDM MS6b receiver-support resume

Date: 2026-07-24
Lane: `lane_ddm_ms6b_receiver_support_resume_20260724`
Verdict scope: `INSTANCE_V19C_ENDPOINT_ONE_QUANTUM_SWEEP`
Evidence axis: `[macOS-CPU frozen-scorer advisory]`
`score_claim=false`; `research_only=true`; `main_landing_review_required=true`

## Verdict

The governed, SSD-first v2 sweep is **COMPLETE: 748/748 signed one-quantum
probes**. The resulting causal receiver assignment remains **PARTIAL**:
34/1,200 PF2 buckets have a measured assignment and 1,166 do not. Exact G3
top-24 pair-by-bucket coverage is **NOT PROVEN**: only pair 21 is complete and
106 required pair/bucket blocks remain missing. Therefore MS4 was not invoked,
Pose remains COMPLETE without remeasurement, and the pointer remains
`0.1910828242 [contest-CPU]` UNMOVED.

## Stores consulted

- Delegated authority:
  `ddm_ms6b_receiver_support_resume_20260724T063800Z.wrapped.prompt.txt`,
  SHA-256 `e1149f5b9d785bfe34668809a7aac3b6acd12f0f1309c810c2152cd1188f044e`,
  5,971 bytes.
- Predecessor findings and receipt named by the authority.
- Base archive SHA-256
  `dc767b59c9e8671b6870e0f9f17a24cfe900dd0f2ae2a251825e41566b52e4c9`.
- PF2 receipt SHA-256
  `85084f7bd3a03dbd1b9f04fe6a9b84df4948a6caf64620beef42da8924345f73`.
- MS5 source table/receipt SHA-256
  `20fa2b2ce2bd96b91c64d4e1342109dd7dab399d4769cd372dbf67fbcdf97d8d`
  and
  `3d0b9fcc738a1092bad495b0dbce2b022451e1442814a7cc274da41e43d455d6`.
- Frozen SegNet/modules SHA-256
  `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`
  and
  `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa`.
- G3 registry SHA-256
  `0c9ce6d0ce2b2c0830400f096438355242527d40f682fc1b201f67d8d951a4e4`.
- Final v2 checkpoint digest-chain SHA-256
  `27f6c56928068b383fa930095e3b4d3f6faa2433c40a0dc81c39e8d3cf9dabdc`.
- Final assignment table file/content SHA-256
  `e34e15e19e2181ceb2dea2c2e69ba46e9ac8c2900df90ccb2cb1fc6c7a27b7c6`
  and
  `368a133ce347362369d7d2843498618f6699102f39376266faac8637fd5632e3`.
- Resume summary file/content SHA-256
  `6963b79e075530739ed7a956e3ae0892ca5c16ab3c7d3df968c10a2f89e87ffa`
  and
  `62005cdd91f8f0f55eee6acb298b8cfe9a416ee9848eb203a8b55ee8d1ab3640`.

## Finding 1 — the signed receiver sweep is complete

The final status decomposition is:

| Status | Probe count |
|---|---:|
| `MEASURED_ARGMAX_PERTURBATION` | 663 |
| `MEASURED_EMPTY_ARGMAX_INVARIANT` | 1 |
| `MEASURED_EMPTY_RASTER_SUPPORT` | 14 |
| `INFEASIBLE_RECEIVER_QUANTUM` | 70 |

The 70 infeasible rows remain explicit, never coerced to zero: 48 Lane grammar
rows, 12 G2CS1 grammar rows, and 10 island `center_x` receiver-geometry rows.
The authority's initial state named one known geometry row; the completed sweep
measured nine additional instance-scoped geometry failures. This does not
promote to a formulation/family negative.

The 15 measured-empty rows are also explicit. One is
`j2.island.track117.center_y` in the positive direction with argmax invariance.
Fourteen template directions have empty raster support: both directions for
row0 RGB; row2 green negative; both directions for row3 RGB; and row4 green
negative.

## Finding 2 — exact causal assignment remains sparse

The loader-schema assignment table recovers PF2 membership for all 1,200
buckets and conserves the 4,011,236-event atlas mass, but only 34 buckets have
measured causal assignments. Of those, 33 are multi-actuator. The summary
artifact contains all 374 actuator rows, both direction supports, and the exact
per-bucket assignment-row, actuator, direction, joined-pair, and labeled probe
event-incidence counts.

`probe_event_incidence_count` is deliberately not called unique-event
cardinality: it sums incidence across actuator-direction assignments and may
count the same raw PF2 event more than once.

## Finding 3 — track0 sign asymmetry generalizes in intensity, not spatial reach

For `j2.island.track0.center_x`, positive versus negative changes camera-value
support from 10,398 to 10,572 and composite-R cells from 566 to 579, while
bucket hits fall from 15 to 13 and perturbed events from 545 to 493. The
positive-minus-negative normalized asymmetries are therefore `+0.00830`,
`+0.01135`, `-0.07143`, and `-0.05010`, respectively. Raster/scorer pair reach
is tied at 35/35.

Across 334 actuator coordinates with two measured directions, the qualitative
pattern persists:

| Metric | Negative / tie / positive dominant | Mean asymmetry |
|---|---:|---:|
| Bucket hits | 125 / 95 / 114 | -0.00520 |
| Camera values | 144 / 47 / 143 | +0.00728 |
| Composite-R cells | 134 / 50 / 150 | +0.00935 |
| Perturbed events | 151 / 17 / 166 | -0.00488 |
| Raster pairs | 0 / 332 / 2 | +0.00599 |
| Scorer pairs | 4 / 324 / 6 | +0.00572 |

Thus signed quantization can alter support intensity substantially for
individual coordinates, but the set-size reach across raster/scorer pairs is
overwhelmingly symmetric. This statement is scoped to this V19C endpoint and
one-quantum formulation.

## Finding 4 — G3 top-24 coverage fails closed

Exact pair-by-bucket coverage is:

| G3 pair | Required / joined / missing buckets |
|---:|---:|
| 523 | 20 / 9 / 11 |
| 54 | 16 / 14 / 2 |
| 1 | 13 / 9 / 4 |
| 90 | 18 / 9 / 9 |
| 21 | 10 / 10 / 0 |
| 446 | 18 / 14 / 4 |
| 0 | 13 / 11 / 2 |
| 14 | 13 / 10 / 3 |
| 18 | 16 / 11 / 5 |
| 327 | 31 / 21 / 10 |
| 7 | 13 / 10 / 3 |
| 60 | 19 / 13 / 6 |
| 49 | 13 / 9 / 4 |
| 41 | 12 / 11 / 1 |
| 323 | 25 / 18 / 7 |
| 44 | 14 / 10 / 4 |
| 38 | 13 / 8 / 5 |
| 42 | 11 / 9 / 2 |
| 4 | 10 / 9 / 1 |
| 36 | 12 / 9 / 3 |
| 320 | 20 / 14 / 6 |
| 55 | 13 / 8 / 5 |
| 56 | 15 / 11 / 4 |
| 16 | 14 / 9 / 5 |

The exact 106 missing `{pair_id, bucket_id}` blocks are machine-readable in
`ddm_ms6_receiver_support_resume_summary.json`. The MS4 prerequisite is false,
so `tools/produce_ddm_ms4_metric_custody.py` was not run.

## Required next action

Open a separately claimed build lane for the receiver grammar/geometry
extension and alternative receiver-effective coordinates. Preserve the 60
grammar and 10 geometry rows as the current instance evidence. Re-run only
newly admissible signed probes, rebuild the assignment, and require all 106
current G3 blocks to close before MS4 becomes eligible.

## Review disposition

Round 1 found a HIGH-severity summary-consumer error: recovered bucket rows do
not expose a bucket-level `perturbed_event_count`, because that would imply a
false unique-event count. The helper now sums explicitly labeled per-assignment
probe incidence and has a regression test. The clean-pass counter was reset;
the three post-fix passes and commands are recorded in the round-1 review
receipt. MAIN must review the branch diff and these custody claims before merge.
