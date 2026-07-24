# Codex findings: DDM MS7 receiver edges and 25-bucket reach

Captured: 2026-07-24T18:00:27Z
Lane: `lane_ddm_ms7_receiver_edges_20260724`
Authority: `ddm_ms7_receiver_edges_and_25bucket_reach_20260724T172249Z`
Evidence axis: `[macOS-CPU frozen-scorer advisory]`
Research only: `true`
Score claim: `false`
MAIN landing review required: `true`

## Verdict

The mandatory R0 table is complete: `0/25` terminal rows can pay the cheapest
currently measured guaranteed-reach price, so `25/25` are
`UNREACHABLE-AND-IGNORED`.

Verdict scope:
`INSTANCE(exact MS4D terminal rows × G3 pair mass × DM4 guaranteed-reach prices)`.
This does not prove that a future dynamic-coordinate or T-residual family cannot
be cheaper. Their byte prices remain exactly `NULL`.

The independent PF3 control closes all five previously missing receiver edges
for one scorer-recursive coordinate. It is a measured pricing control, not an
R0 admission and not a contest score.

## Mandatory R0 table

The durable 25-row table is
`.omx/research/ddm_ms7_receiver_edges_and_25bucket_reach_20260724T172249Z/r0_25_bucket_reach_table.json`,
SHA-256
`bfdc19a56206dfc920a9d0b25f38b1d59b73b18aaa5fb864ab354b4d00e310f6`.

Every row contains:

- exact `bucket_id` and `pair_id`;
- event mass `support_count / pair_flip_count`;
- flip-weighted S leverage
  `pair_distortion_score_mass * event_mass`;
- the measured DM4 corrected-J/shearlet sparse-pixel guaranteed-reach byte
  bound;
- rate break-even `25 * reach_bytes / 37,545,489`;
- explicit `NULL` R1 dynamic-coordinate and R2 T-residual prices;
- a scoped `UNREACHABLE-AND-IGNORED` verdict.

R1, R2, and R3 execution were correctly not run because the R0 mass-paying
subset is empty. No fixed-amplitude ladder, fake cheaper price, or blank-to-zero
coercion was used.

## PF3 five-edge closure

The mandated control is pair `523`,
`lane_undrivable__boundary__static_in_image`, using the already measured
receiver-derived actuator
`rg3.class_birth.pair523.class1_2.boundary.static_in_image.band03.fine00.mag1`.

1. Receiver-object builder: the RG3 symbol is inserted into the nested carrier,
   then rewrapped through the frozen coupled-margin and pre-uint8 V19C
   receiver. Candidate SHA-256:
   `483df80b3ec3ebfd4a3afd6f9d5b8810f6365d35f2d9a339ccce5d42d22765a4`.
2. Realized uint8 quantum: `90` channel values changed on pair `523` only;
   minimum nonzero absolute change `45`, maximum `159`.
3. Same-object candidate delta: batch-16 replay changed `3` Seg argmax cells,
   moved global Seg errors `2,923,991 -> 2,923,992`, and changed global Pose
   SSE by `-0.0011531194868439343`. All other pair outputs were byte-identical.
4. Dimension rate home: typed `SKELETON / L3_RASTER`,
   `production/residual_family_coordinates.rg3rf`.
5. Coder payload owner: `DDM_MS7_SAME_OBJECT_RECEIVER_ARCHIVE_PAYLOAD`.

The registered `ddm_tolerance_capped_min_score_waterfill_v1` callable prices
the winning exact-coded object at `129,797` bytes, `S=42.945963466535154`, and
rejects it because `2,923,992 > 136,839` Seg errors. This is the expected
`MEASURED_PRICED_CONTROL_NONADMISSIBLE_R0_AND_ERROR_CAP` result.

## Dynamic quantum calibration

`dynamic_quantum_calibration_v1` replaces the superseded fixed
`±{2,3,4}` ladder:

`k_i* = snap_up(ceil(q_i / (2 |g_i|)))`, validity-gated by the measured family
radius.

For the control, composite-R gain `g_i=2.908951903163341`, deadzone `q_i=1`,
and class-birth validity radius `1` produce `k*=1`. The single realized probe
crossed the deadzone with minimum nonzero uint8 change `45`; the binary
predicted-vs-realized residual is `0`. The equation is registered append-only
with this empirical anchor. Outside a measured validity radius, calibration
returns `NULL` rather than clipping.

## Exact same-object coder race

All available rows decoded byte-for-byte to the same `141,835`-byte receiver
object:

| Codec | Counted bytes | Result |
|---|---:|---|
| Brotli Q11 | 129,797 | winner |
| raw LZMA1 | 131,562 | exact |
| zlib9 | 132,360 | exact |
| zstd19 + trained dictionary | 132,390 | exact; dictionary bytes counted |
| raw compact | 141,835 | exact identity |
| order-1 context arithmetic | 144,003 | exact |
| constriction order-1 context ANS | 144,130 | exact |
| G4 decoder-derived spatial context | `NULL` | no spatial payload home for the flat archive object |

Thus constriction is usable and deterministic here, but it is not competitive
with Brotli on this flat full-archive object. The measured result does not
negative the constriction family on parsed grammar streams or a future genuine
G4 decoder-derived spatial payload.

## Triality and system feed

- DSL:
  `.omx/research/configs/ddm_ms7_receiver_edges_20260724.json`
- DAG/feed:
  `.omx/research/ddm_ms7_receiver_edges_and_25bucket_reach_20260724T172249Z/ddm_ms7_receiver_edges_DAG_FEED.json`
- equations:
  `dynamic_quantum_calibration_v1` and
  `ddm_tolerance_capped_min_score_waterfill_v1`

The DAG records the sensitivity-map, Pareto, bit-allocator, autopilot,
continual-learning, and probe-disambiguator hooks. Because R0 is empty,
autopilot receives a stop result rather than a construction or dispatch.

## Directive consumption

| Directive UTC | Disposition |
|---|---|
| 2026-07-24T14:45:16Z | Consumed: used the MS6 receiver-derived RG3 coordinate and MS4D post-R rank-4 gain; no generic spatial menu. |
| 2026-07-24T17:26:45Z | Consumed: dynamic per-coordinate `k*` superseded the fixed ladder; one probe verified predicted versus realized. |
| 2026-07-24T17:39:13Z | Consumed: dependency-count cap removed; deterministic decode and counted payload retained. |
| 2026-07-24T17:42:58Z | Consumed: constriction and trained-dictionary zstd raced on exact bytes; G4 spatial home stayed `NULL`. |

## STORES CONSULTED

- delegated authority file, SHA-256
  `7545a6c6e9f1c5a072051b6b02eb256be7cfc9c08adbe1f6f55cd9b8e4772d3f`;
- full `CLAUDE.md`, full `AGENTS.md`,
  `docs/operating_manual_craft_handoff.md`, and recent Claude memory;
- canonical lane, subagent, frontier, gradient, dispatch, cost-band, and
  continual-learning state surfaces;
- latest sister findings/session/design/council memos;
- MS4D direct metric and complete bundle, G3 n600 atlas, DM4 reach receipt,
  V19C endpoint/archive, G2f amplitude law, v16/v17 validity receipts, frozen
  scorer config/weights, and exact target cache;
- both live inboxes through `2026-07-24T17:42:58Z`.

No training, paid dispatch, Modal work, contest eval, candidate promotion, or
frontier mutation occurred. The pointer remains exactly
`0.1910828242 [contest-CPU]`.

MAIN must independently review the R0 leverage definition, NULL-price
preservation, nested receiver composition, scorer-batch splice, coder framing,
canonical-equation registry row, and non-promotion scope before landing.
