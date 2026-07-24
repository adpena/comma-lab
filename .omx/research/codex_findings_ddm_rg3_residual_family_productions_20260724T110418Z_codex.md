---
title: Codex findings — DDM RG3 terminal residual-family productions
utc: 2026-07-24T11:45:00Z
lane_id: lane_ddm_rg3_residual_family_productions_20260724
verdict: PARTIAL_INSTANCE_RG3_11_OF_36_CLOSED
research_only: true
score_claim: false
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
pointer_moved: false
main_landing_review_required: true
canonical_equation_id: ddm_rg3_residual_family_productions_v1
---

# Outcome

RG3 is a real, additive, counted receiver grammar and the full local advisory
sweep completed. It closed **11 of RG2's 36 exact pair/bucket residual
obligations**. Twenty-five remain, so MS4 is not eligible and the producer was
not rerun. This is an `INSTANCE_EXTENDED_GRAMMAR_RG3` partial result, not a
formulation/family/paradigm negative. The contest pointer is unchanged.

Custody:

- assignment file SHA-256 `40d4150eb3c1bee1197b4023a1e2986e498f429d85677e992a8464ac7acab82e`;
- measurement receipt SHA-256 `29faac211c6442f9462ba25f66c9e1ef7e2caedd325d190568ae25e5d640971e`;
- merged table SHA-256 `57d3954bc4661f5da48aae943433a7c5f611639b2d5a24854a01d658fd52aebd`;
- support summary SHA-256 `3d4c4fb635ec37668cbf6037cefca63fe7c08a9ad950e6724ae023deb0473fd2`;
- exact V19C base SHA-256 `dc767b59c9e8671b6870e0f9f17a24cfe900dd0f2ae2a251825e41566b52e4c9`.

## Finding 1 — the RG3 grammar is additive and receiver-effective

`RG3ResidualCoordinateV1` adds one versioned `RG3RF` counted member to the
existing RG1/RG2 wrapper. Canonical address validation, signed magnitude
validation, sorted uniqueness, CRC, strict member parsing, and decode/re-encode
identity are enforced. Empty RG3 compilation returns the exact nested-carrier
bytes (`7990fce7...`), so the inactive extension is byte-identical. The sealed
RG1/RG2 base is not mutated.

The three families are represented exactly as assigned:

| Family | Residual rows | Magnitudes | Signed probes |
|---|---:|---:|---:|
| event-local class birth | 10 | `{1}` | 20 |
| finer event-local boundary amplitude | 9 | `{1,2}` | 36 |
| Fisher-margin per-stratum cell amplitude | 17 | `{1,2}` | 68 |
| total | 36 | 62 coordinate magnitudes | 124 |

The honest stream home is `SKELETON/L3_raster`. Fisher margins are consumed
only by the offline assignment builder through
`0.5*sech²(top1_top2_margin/2)`; neither margins nor scorer state are shipped.

## Finding 2 — 124 new probes preserve 870 prior rows

The SSD-backed sweep completed 124 new per-coordinate checkpoints under four
Torch threads. New-only status counts were:

- 68 `MEASURED_ARGMAX_PERTURBATION`;
- 30 `MEASURED_EMPTY_NO_OCCUPIED_BUCKET_OVERLAP`;
- 26 `MEASURED_EMPTY_RASTER_SUPPORT`.

All 124 compiled candidates had distinct archive SHA-256 values. Merging them
with the exact 870 prior authoritative rows produced 994 bound probe rows.
The 372-required iterative obligation invariant held: the input and output
tables each contain 1200 PF2 rows and 9892 pair/bucket membership incidences,
with identical membership digest `995f72c1...`.

## Finding 3 — closure is family-specific, not a headline-only 11/36

| Family | Entered | Closed | Remaining |
|---|---:|---:|---:|
| class birth | 10 | 0 | 10 |
| finer boundary | 9 | 3 | 6 |
| Fisher cell | 17 | 8 | 9 |
| total | 36 | 11 | 25 |

The 11 closed rows were:

`(1, road_movable cell static)`;
`(7, road_movable boundary transient)`;
`(7, road_movable cell static)`;
`(7, road_movable cell transient)`;
`(16, lane_movable cell static)`;
`(16, road_movable cell static)`;
`(60, lane_mycar boundary static)`;
`(90, lane_movable cell static)`;
`(90, road_movable cell static)`;
`(523, road_mycar boundary transient)`;
`(523, road_mycar cell transient)`.

The complete G3 top-24 state moved from 7 to 9 fully joined hard pairs. That
is meaningful apparatus progress, but it is not the all-36 closure required
for MS4.

## Finding 4 — all 25 remaining blockers are exact and terminal for this arm

Every remaining row has the same measured blocker class:
`NO_TARGET_BUCKET_EVENT_CHANGED_BY_ANY_COUNTED_RG3_MAGNITUDE_OR_SIGN`.
The machine-readable summary retains every constituent checkpoint SHA,
direction, status, target-bucket-hit bit, target-pair-join bit, and event
count. The exact residual inventory is:

| Pair | Bucket | RG3 family | Advisory derived next family |
|---:|---|---|---|
| 523 | lane_movable__cell__static_in_image | Fisher cell | Fisher-margin site-local per-stratum |
| 523 | lane_undrivable__boundary__static_in_image | class birth | worldsheet-event-indexed typed interface arc |
| 523 | lane_undrivable__cell__static_in_image | Fisher cell | Fisher-margin site-local per-stratum |
| 54 | road_mycar__cell__static_in_image | Fisher cell | Fisher-margin site-local per-stratum |
| 90 | lane_mycar__boundary__static_in_image | finer boundary | curvelet/shearlet boundary arc |
| 90 | undrivable_movable__boundary__transient | finer boundary | curvelet/shearlet boundary arc |
| 446 | lane_undrivable__boundary__static_in_image | class birth | worldsheet-event-indexed typed interface arc |
| 446 | lane_undrivable__cell__static_in_image | Fisher cell | Fisher-margin site-local per-stratum |
| 0 | road_undrivable__boundary__transient | class birth | worldsheet-event-indexed typed interface arc |
| 14 | lane_undrivable__cell__static_in_image | Fisher cell | Fisher-margin site-local per-stratum |
| 327 | road_mycar__cell__static_in_image | Fisher cell | Fisher-margin site-local per-stratum |
| 60 | lane_movable__boundary__static_in_image | class birth | worldsheet-event-indexed typed interface arc |
| 60 | road_mycar__cell__static_in_image | Fisher cell | Fisher-margin site-local per-stratum |
| 323 | lane_mycar__boundary__static_in_image | finer boundary | curvelet/shearlet boundary arc |
| 323 | undrivable_movable__boundary__transient | finer boundary | curvelet/shearlet boundary arc |
| 38 | lane_mycar__boundary__static_in_image | class birth | worldsheet-event-indexed typed interface arc |
| 42 | lane_movable__boundary__static_in_image | class birth | worldsheet-event-indexed typed interface arc |
| 4 | lane_undrivable__boundary__static_in_image | class birth | worldsheet-event-indexed typed interface arc |
| 55 | lane_undrivable__boundary__static_in_image | class birth | worldsheet-event-indexed typed interface arc |
| 55 | road_undrivable__boundary__transient | class birth | worldsheet-event-indexed typed interface arc |
| 56 | lane_movable__boundary__static_in_image | finer boundary | curvelet/shearlet boundary arc |
| 56 | lane_movable__cell__static_in_image | Fisher cell | Fisher-margin site-local per-stratum |
| 56 | lane_mycar__boundary__static_in_image | finer boundary | curvelet/shearlet boundary arc |
| 16 | road_mycar__cell__static_in_image | Fisher cell | Fisher-margin site-local per-stratum |
| 16 | undrivable_movable__boundary__transient | class birth | worldsheet-event-indexed typed interface arc |

These next-family names are first-rung derivations only. They do not authorize
RG4, do not admit bytes, and do not justify a launch. MAIN must explicitly
authorize any successor lane.

## Finding 5 — own round-1 caught two receipt-schema defects

The first completed sweep preserved all checkpoints but receipt assembly
failed because the consumer referenced a nonexistent
`fisher_margin_input` key instead of the assignment's sealed
`input_custody.fisher_margin_sha256`. The first summary attempt likewise
referenced `family` instead of the assignment row's
`selected_coordinate_family`. Both were fixed through typed helpers and
regression tests before any receipt was admitted. No measurement bytes were
lost or recomputed.

The same review also caught a canonical-equation mismatch: the implementation
uses four 16-row subbands inside each 64-row RG2 band, while the initial
equation draft described sixteen 4-row subbands. The equation and tests now
match the executable receiver.

## No-orphan routing

- Sensitivity map: exact PF2 incidence and pair joins are carried in the merged table.
- Pareto/bit allocator: non-binding; score-unit value per byte remains owed.
- Cathedral/autopilot: MS4 stays fail-closed; no producer rerun occurred.
- Continual learning: `ddm_rg3_residual_family_productions_v1` and the probe-outcome row consume this result.
- Probe disambiguation: the 124 signed probes selected the measured disposition of all three RG3 families.
- Mission contribution: `frontier_protecting`—it prevents a false BUNDLE-COMPLETE claim while preserving 11 real closures.

## Triality and review disposition

- DSL: RG3 packet/compiler/receiver in `ddm_rg1_receiver_grammar.py`.
- DAG: `ddm_rg3_residual_family_productions_DAG_FEED_20260724.md`.
- Equation: registered `ddm_rg3_residual_family_productions_v1`.

Three post-fix clean passes are recorded separately. This branch is not
self-promoting: `main_landing_review_required=true`, and MAIN must inspect the
base-to-branch diff before merge.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; latest
Claude MEMORY top entries; RG2 findings 1-5; exact RG2 summary; lane registry;
subagent progress; canonical equation registry; both live inboxes. The
2026-07-19 Fisher-margin and reverse-waterfill directives were consumed; no
Fourier residual basis was introduced.
