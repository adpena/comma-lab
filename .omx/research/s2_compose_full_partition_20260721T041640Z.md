# Task #595 — S2 full-partition composition, finite component landing

- Captured: 2026-07-21 UTC
- Lane: `s2_compose_full_partition_20260721T035922Z`
- Axis: **[macOS-CPU advisory]**
- Score claim: **false**
- Promotion eligible: **false**
- Pointer delta: **none**; `0.19108 [contest-CPU]` is unchanged
- Content lineage: from-scratch/our-solve; inherited candidate bytes = **0**
- Verdict: **PARTIAL_GAP1_MEASURED_GAP2_FINITE_BASELINE_GAPS3_TO_5_OPEN**
- Verdict scope: real n600 source/cache instance. This is neither a complete S2 partition nor an
  S3/S4 receiver row, and it does not close the task-space level-set witness family.

## Outcome first

Two durable stage measurements landed.

1. The coherent Lane polynomial chart now has a true-mask, finite-byte control curve. The
   finite dash chart is 41,303 B and has pixel F1 0.573397; switching the same chart to a
   continuous decoder raises recall from 0.543964 to 0.811379, but precision falls from
   0.606196 to 0.425998 and F1 falls to 0.558675. More recall is not closure: dash-gap false
   positives dominate, especially on Road.
2. The G3 inventory now has a strict finite packet. All 17,926 sites, target/baseline cell
   identities, headers, coder bytes, and CRC occupy **39,836 B** and parse back exactly. It stores
   zero RGB/YUV values. This replaces the 2,724.873 B ideal/no-site/no-header number as the
   measured literal-event baseline, not as the optimal Morse–Smale description.

The strict spine audit admits both component rows while keeping complete S2, S3, S4, score, and
promotion authority false.

## Gap 1 — true Lane-mask control curve

Receipt: `s2_lane_true_mask_curve_20260721T042500Z.json`, SHA-256
`273d7ef28b9312973831403c57552274a8fef57f53a1eb4517c1e3551d76ef94`.

The tool fits from the real `lstars` n600 memmap, serializes/de-serializes LBND2, renders the
decoded chart, and compares it directly with class-1 pixels. These are mask-fidelity numbers,
not through-R scores.

| finite decoded chart | bytes | precision | recall | F1 | mean pair F1 |
|---|---:|---:|---:|---:|---:|
| coherent slot, range-dependent dash | 41,303 | 0.606196 | 0.543964 | 0.573397 | 0.581197 |
| coherent slot, continuous | 41,298 | 0.425998 | 0.811379 | 0.558675 | 0.570782 |

Selected strata explain the tradeoff:

| stratum | dash | continuous |
|---|---:|---:|
| Lane boundary recall | 0.523794 | 0.758904 |
| Lane interior recall | 0.604615 | 0.969169 |
| Road false-positive rate | 0.005884 | 0.018634 |
| other-non-Lane false-positive rate | 0.000921 | 0.002719 |

The genuine finite-polar-curvelet/compact-shearlet structural proof remains clean at SHA
`677a2252c43c1272ec0e2e83d65ce1b82d23b8ddb089d73a111a5f0b26d46d25`. It does not supply the
missing custodied inverse from this Lane true-mask residual into finite selected atoms. Therefore
the exact blocker is `BLOCKED_TARGET_BOUNDARY_INVERSE_CUSTODY`. No Fourier or generic wave-packet
substitute was used, and gap 1 remains open.

## Gap 2 — finite literal event packet

Receipt: `s2_compose_full_partition_20260721T041640Z.json`, SHA-256
`dd67fbe8d5aa67ea80e3d2dc689281547a5a7dd3a42c12d5c426214f68f3b023`.

| counted component | bytes |
|---|---:|
| packet prefix | 16 |
| canonical header | 497 |
| finite compressed event body | 39,319 |
| CRC | 4 |
| **counted video-derived seed** | **39,836** |

The packet is 2.222247 B/event, 14.6194x the old ideal entropy-only floor, and 0.634928x the
prior 62,741 B raw-coordinate baseline. Its score-rate term alone is 0.0265252. The decoder
interpreter is 452 free LOC in `src/tac/optimization/s2_partition_seed.py`; no video-derived
table is embedded in code.

Measured standalone strata include:

| event stratum | events | standalone finite bytes |
|---|---:|---:|
| Road–Lane edge | 5,193 | 12,891 |
| other edge | 12,468 | 27,413 |
| non-edge | 265 | 1,170 |
| tight margin `<1e-3` | 1,607 | 5,407 |
| moderate margin `[1e-3,1)` | 16,319 | 36,584 |

This packet is intentionally strict but not optimal-form authority. The later operator directives
require the full partition to be one spacetime Morse–Smale object: critical points, separatrix
arcs, canonical cell adjacency traversal, and vineyard lifecycle symbols. Labels are conditional
exceptions and decoder-recomputable sites are top-k/arc-length exceptions, not raw coordinates.
Accordingly, 39,836 B is the literal-event R0 baseline that the sibling predict→project receiver
must beat; it is not a competing `constraint-seed.v0` schema.

## Rule-118 boundary and the five-gap status

Generic algorithms belong in the free interpreter: semantic detection, Morse–Smale traversal,
coder contexts, deterministic tie breaks, AA-SDF, gauge/range(A) projection, integer projection,
and predict→project selection. The counted seed carries only video-specific constraints.

| gap | status after this landing |
|---|---|
| 1 Lane closure | **MEASURED CONTROL, OPEN** — genuine residual inverse missing |
| 2 full partition | **FINITE LITERAL-EVENT BASELINE, OPEN** — baseline predictor/MS 1-skeleton not receiver-composed |
| 3 ξ stream | **NOT BUILT** — bulk transport must be free; counted stream is phase-conditioned boundary jitter/events |
| 4 S3 realization | **NOT COMPOSED** — gauge → range(A) → integer lattice and survived-flip waterfill still owed |
| 5 S4 byte-close | **NOT BUILT** — no archive, d_seg, d_pose, or score row |

For gap 3, `partition_temporal_transport_amortization_jitter_bound_v1` forbids the naive
“ξ innovations of labels” interpretation (measured ratio 0.71). The next receiver must measure
R0 raw arc-normal offsets, R1 phase-conditioned residuals, then R2 ground appearance chart + ξ +
generic response + exceptions. Jitter is modeled cause, not irreducible noise.

The eventual allocator is global, not one waterfill per stream. Chart precision, cell/sites,
jitter, event symbols, pose-tube radius, and response parameters compete for the same realized
flips at `realization_breakeven_bytes_v1` (`6.6585895e-7 S/B`). It must full-decode every joint
composition, publish the pairwise interaction matrix (joint delta minus summed single deltas),
account for composition order, and retain an explicit eat-the-flip action when bytes cost more
than the distortion. No independent marginal curves may be added as though opportunity pools were
disjoint.

The pose description also cannot defer checking to decode: PoseNet is absent from the interpreter,
so the encoder must tighten pixel-space pose boxes and conditioning through the real PoseNet. G3's
frame asymmetry means frame 0 is pose-only at the measured Seg obligation; it must not receive
Seg-grade description merely for schema symmetry.

## Acceptance and reproducibility

- GT cache SHA-256:
  `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`
- Finite packet SHA-256:
  `df4c0534537a9919681509a0b44a392d7d4b46c812d7570c534e6b823adae7fc`
- Strict composed audit: `s2_compose_full_partition_audit_20260721T042200Z.json`, SHA-256
  `4a8b358ed8201f4e082ed96babc27bce4edbcc6ae57d0329b3c92876145d68ea`
- Packet: exact double encode and exact parse-back event identity.
- Tests: 14 focused tests pass; ruff and py_compile pass.
- Rust receiver primitives: AA-SDF positive parity, negative-control bit flip, range decoder, and
  ξ column-delta parity all execute and pass. Scope remains n96 primitive proof, not n600 closure.
- SSD custody:
  `/Volumes/VertigoDataTier/pact/evidence/s2_compose_20260721/partition_seed/`.

No new universal measured law is claimed here, so no new canonical equation was registered. The
two receipts are instance-level empirical anchors and consume the existing equation IDs listed in
the DAG feed.

## Cathedral triality and MAIN landing review

- DSL/schema leg: strict `s2_partition_event_seed.v1` component packet only. MAIN must reconcile
  this decoder with the sibling-owned single-spacetime `constraint-seed.v0`; do not merge it as a
  second full-seed schema.
- DAG leg: `s2_compose_full_partition_DAG_FEED_20260721T041640Z.md` records the admitted and
  deliberately absent edges.
- Equation leg: existing Fisher/margin, Morse–Smale cell, shearlet, transport-jitter, range(A),
  and break-even laws are consumed by ID; no equation is re-derived.

MAIN review is mandatory for schema reconciliation, source-lineage custody, strict-auditor
admission scope, and whether the literal-event baseline should be retained as a regression fixture.

## STORES CONSULTED

- `reports/latest.md`
- `.omx/state/lane_registry.json`
- `.omx/state/subagent_progress.jsonl`
- `.omx/state/master_gradient_anchors.jsonl`
- `.omx/state/modal_call_id_ledger.jsonl`
- `.omx/state/cost_band_posterior.jsonl`
- `.omx/state/continual_learning_posterior.jsonl`
- `.omx/state/canonical_task_status.jsonl`
- `.omx/research/joint_planes_direct_strike_20260721T034248Z.{md,json}`
- `.omx/research/m1_quant_aware_retrain_20260721T034000Z.md`
- `.omx/research/g1_worldsheet_g3_cellcode_measurements_20260720T210000Z.json`
- `.omx/research/genuine_curvelet_shearlet_structural_proof_v2_polar_frequency_wedge_20260714.json`
- `.omx/research/flicker_transform_geometry_term_design_20260710.md`
- both delegated-task inboxes through 2026-07-21T04:31:31Z
