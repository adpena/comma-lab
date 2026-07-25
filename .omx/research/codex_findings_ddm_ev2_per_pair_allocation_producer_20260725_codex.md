# Codex findings — DDM EV2 per-pair allocation producer

Date: 2026-07-25 UTC  
Lane: `ddm_ev2_per_pair_allocation_producer`  
Authority: delegated build, research-only, MAIN-review-required  
Evidence axis: `[macOS-CPU frozen-scorer advisory]`  
Pointer: `0.1910828242 [contest-CPU]` UNMOVED  
Score claim: false  
Promotion eligible: false

## Verdict

`FORMULATION_MISPOSED_FOR_CURRENT_C1_COMPOSITION` at `FORMULATION` scope.

The preregistered falsifier fired. Exact construction-lineage tracing found no
counted final-byte interval with an exclusive `{source_pair_id,
stratum × scorer_visibility × G4_temporal_class}` owner. Lawfully separable C1
mass is `0 / 134,211 B = 0.0%`; typed `UNALLOCATED` mass is
`134,211 / 134,211 B = 100.0%`, above the 30% falsifier threshold. This is not a
negative verdict on waterfilling, Fisher geometry, the scorer-derived
partition, or the representation family. It is a rejection of this
fine-grained rate-home formulation for the current jointly coded C1 object.

## Fresh construction-lineage proof

The exact C1 archive is 133,941 bytes, SHA-256
`759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df`.
LP1 adds a separately measured 270-byte lane-program-seed home, giving its
settled 134,211-byte total.

The outer archive partitions exactly into the following non-overlapping homes:

| LP1 typed stream home | Counted bytes | Exact outer range |
|---|---:|---|
| `manifest` | 3,345 | `[0, 3345)` |
| `v15_predictor_zip_outer_home` | 100,099 | `[3345, 103444)` |
| `g1_movable_worldsheet_outer_home` | 29,878 | `[103444, 133322)` |
| `receiver_realization_profile` | 85 | `[133322, 133407)` |
| `solved_template_outer_home` | 151 | `[133407, 133558)` |
| `central_directory_and_eocd` | 383 | `[133558, 133941)` |
| `lane_program_seed` | 270 | separate LP1 measured home |

These seven rows are the coarsest lawful partition. Every row records
`derivation_method=EXACT_CONSTRUCTION_LINEAGE` and
`assignment_status=UNALLOCATED_NO_EXCLUSIVE_FINAL_BYTE_PAIR_AND_CELL_FOREIGN_KEY`.

The two large archive homes prove why finer ownership is unavailable:

- `predictor.zip` contains 11 coded members spanning all 600 pairs. Its
  construction exposes no exclusive final-byte pair boundaries.
- the G1S1 worldsheet contains three globally coded productions spanning all
  600 pairs. Its envelope likewise exposes no exclusive final-byte pair
  boundaries.
- the remaining homes are shared metadata, shared receiver/template state, or
  ZIP container structure. The lane seed is a shared measured delta, not a
  per-pair archive section.

A receiver-effect probe was not run because exact construction lineage already
proves the required final-byte foreign key is absent. Perturbing a shared
compressed stream can reveal receiver effects, but cannot manufacture an
exclusive final-byte owner. The 600 per-pair rows therefore record zero
exclusive archive-section bytes and preserve all mass as `UNALLOCATED`.

## Same-object and conservation firewalls

LP1 C1 bytes and EV1 endpoint-edge accounting bytes are different objects.
EV1's 162 exclusive accounting homes reconcile exactly within their own object:

- dual 1: 16 bytes;
- dual 2: 962 bytes;
- dual 3: 409,388,124 bytes.

No EV1 byte was smeared into C1. The typed receipt records
`same_object_as_c1_allocation=false` and `cross_object_byte_smearing=0`.

The consumer-facing allocation table has 162 cell rows and 600 pair rows, all
with zero assigned bytes. It also emits a strictly validated
`ddm_ms5_pf2_bucket_assignment_table.v1` projection: the original 1,200 MS5
rows remain intact and an additive `ev2_rate_home_extension` binds the 162-cell
refusal by content hash. This avoids falsely relabeling a 162-row cube as the
sealed 1,200-row PF2 schema.

## Costates and headline replay

RD1 backfill result: `0 / 162` finite lambda values; `162 / 162` remain null.
Value-filling would be imputation.

The canonical minimum-description headline builder was replayed through the
MS3 `BUNDLE-COMPLETE` custody loader. This edge clears none of the four
blockers; the exact remaining set is:

1. `POSE_TUBE_NOT_ACTIVE_IN_SOLVE`
2. `TYPED_SUBPROBLEM_ALTERNATION_NOT_ACTIVE`
3. `TYPED_BLOCK_ATLAS_NOT_ACTIVE`
4. `PER_DIMENSION_EFFECTIVE_QUANTA_NOT_ACTIVE`

## Registered waterfill preflight

The registered `tools/run_ddm_ms4d_waterfill.py` preflight admitted
`BUNDLE-COMPLETE` and then intentionally refused before solve with exit code 2:

`BLOCKED_PF3_RECEIVER_OBJECT_AND_TYPED_RATE_HOME_ABSENT`

Its scope is
`INSTANCE(current MS4D direct bundle) × FORMULATION(receiver-object tolerance-waterfill)`.
It measured zero materialized receiver/rate-home columns, ran no receiver,
coder, scorer, training, paid dispatch, or full solve, and preserved all 162
prices as null.

## Durable artifacts

- `allocation_table.json` — SHA-256
  `db9e5153d18f2f8c149080f57b608d9c74245d400a0c39fee0570dec467d7056`
- `ms5_loader_table.json` — SHA-256
  `69394cf983c0cfe3d9898476ceaaeef78b1ce5be9a378a1e3bd7e94ea577e9de`
- `rd1_dual_backfill.json` — SHA-256
  `aebbfec7baa1fb321a20ede380e140cb53d4df938dfa0f2c9a269d13e60ca11c`
- `headline_replay.json` — SHA-256
  `24b808c41d336c419393cd2f04db4e79cdb14fbaf0baca86b51c3bba16df60ca`
- `receipt.json` — SHA-256
  `1d76934d9b52d61ac7eca4b4f95d6eb76a31e8c7b354e2913fab3dca991d3fcc`
- `waterfill_post_admission_receipt.json` — SHA-256
  `68d9f581d4b02b2b896da07fd9f8a9062203eb989f7ddb5302936ec6a688e37c`
- `rd1_dual_backfill_post_admission.json` — SHA-256
  `c918c96bb242573821a44d81118281fe64fe802bc1ce613aeec11c0fece0e714`

Focused verification: 6 tests passed; Ruff and `py_compile` passed. Repeated
materialization is byte-identical through immutable checkpoint comparison.

## Triality and next lawful formulation

- Typed config: SHA-bound, local-only materializer with score/promotion/pointer
  authority set false.
- DAG: `LP1 C1 homes + exact archive lineage + EV1 cube + RD1 nulls + MS5
  schema + R3 headline + MS3 bundle → EV2 typed refusal → registered preflight`.
- Equations: `assigned + UNALLOCATED = 134,211`; observed unallocated fraction
  `1.0 ≥ 0.30`; no same-object rate home implies no finite per-cell costate.

The lawful successor is either (a) a coarser seven-home stream-level
waterfill, or (b) a newly constructed C1 object with independently coded,
parse-back-stable per-pair/per-cell sections and exact byte offsets. The
current 162-cell object must not be populated by proportional allocation.

## Stores consulted

LP1 layer-pricing receipt; EV1 campaign-evidence join; RD1 dimension-dual
source and frontier; C1 composed-candidate ledger and exact v15 archive; MS5
assignment table; MS6/RG receiver-support methodology; R3 box receipt; MS3
`BUNDLE-COMPLETE`; canonical headline builder; registered MS4D waterfill gate;
CLAUDE.md; AGENTS.md; program manual; live inbox broadcasts through
2026-07-24T23:09:25Z.

MAIN must independently review the same-object firewall, exact outer-byte
ranges, strict MS5-loader validation, falsifier scope, and the decision not to
run a full solve before merging.
