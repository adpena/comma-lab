# G89 — the semantic correction layer must be class-complete

Date: 2026-07-27  
Axis: `[macOS-CPU encoder-only exact frozen-scorer-cell census]`  
Verdict scope: the fresh G78 target/base coordinate and a correction family whose
only semantic actuators alter the Road and Undrivable masks. This is not a
negative on V9/V10, the selected-preimage codec, shearlets, or analytic
factorization as families.

## Pointer-delta honesty

No archive was promoted and the authoritative pointer remains `0.172`. The G85
base archive is 129,392 bytes with SHA-256
`b9c8ab2af8886c5b26bba63e02b7c5fe9951bb42a871c5e8472483977788d9fd`.
Its exact receiver object is useful custody, but its target-vs-described semantic
distance is `0.027470304701063368`, so it is not a frontier candidate.

## Exact observation

The exact n600 G78 target/description confusion matrix is below. Rows are target
classes and columns are described classes in canonical order
`Road, Lane, Undrivable, Movable, MyCar`.

| target \ described | Road | Lane | Undrivable | Movable | MyCar |
|---|---:|---:|---:|---:|---:|
| Road | 25,196,274 | 674,071 | 1,169,783 | 113,438 | 253,477 |
| Lane | 201,517 | 390,076 | 94,371 | 2,129 | 2,546 |
| Undrivable | 180,466 | 20,008 | 58,176,385 | 34,621 | 1,802 |
| Movable | 94,834 | 1,176 | 239,588 | 1,034,472 | 90,255 |
| MyCar | 65,029 | 1,396 | 22 | 0 | 29,927,064 |

The 3,240,529 mismatches over 117,964,800 cells imply:

`d_seg = 3,240,529 / 117,964,800 = 0.027470304701063368`

`100*d_seg = 2.7470304701063366`

At 129,392 bytes the rate term is `0.08615682166238399`, giving a
pose-independent lower bound of `2.8331872917687204` for the uncorrected base.

## Structural lower bound on a Road/Undrivable-only correction family

A Road/Undrivable-only mask actuator can potentially correct a mismatch when
either the target or described class is Road/Undrivable. Even granting it every
such cell for free, 97,502 errors remain where both labels lie in
`{Lane, Movable, MyCar}`. This deliberately optimistic support bound is:

`d_seg_floor = 97,502 / 117,964,800 = 0.0008265346950954861`

`seg_score_floor = 0.08265346950954862`

Together with the unchanged 129,392-byte base:

`seg_score_floor + rate = 0.1688102911719326`

Thus a perfect Road/Undrivable correction leaves only
`0.003189708828067378` score units for both pose and every added operand byte.
With zero added bytes, Pose distortion would have to be below
`1.0174242407850967e-06`. Using the sealed low-distortion anchor's
`d_pose = 0.0001018434704747051` instead gives a total floor of
`0.2007232149606902`. Even the unphysical zero-pose case permits at most
134,182 archive bytes, only 4,790 bytes above the base.

This is the missing macro composition constraint: G72's Road/Undrivable
shearlets are one carrier in the codec, not the semantic correction layer.
Selecting or pricing 303,528 G72 proposals alone cannot be the next
frontier-producing action.

## Required class-complete factorization

The existing original V9/V10 type system already contains the pieces that match
the measured error geometry:

1. Road/Undrivable: shared level-set or boundary-shearlet bulk field.
2. Lane: ground-chart curve/strip with births, deaths, and thin-class offset.
3. Movable: sparse sites plus temporally persistent worldsheet tracks and knots.
4. MyCar: camera-attached static hood component plus a sparse exception stream.
5. Conditional Y0 given exact Y1: pose fibre, evaluated jointly rather than by
   independent component thresholds.
6. Irreducible remainder: the only part eligible for learned representation.

Logical roles need not map one-to-one to physical streams. The encoder should
induce shared spatiotemporal programs across roles when doing so shortens the
actual serialized archive.

## Executable next edge

Compile the complete G78 mismatch field into all five actuator families, project
each realized-through-R intervention through the current-base scorer costates,
then induce population-global repeated programs before exact archive
arbitration. Admission is on complete n600 archive states:

`S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489`

No fixed per-component acceptance thresholds are valid. The learned arm receives
only the exact residual left after the four analytic class carriers and the
conditional pose fibre have been optimized jointly.

## Triality

DSL:

`SEMANTIC = BULK_0_2 + LANE_1 + MOVABLE_3 + HOOD_4 + EXCEPTIONS`

`PAIR = EXACT_Y1(SEMANTIC) + CONDITIONAL_Y0_GIVEN_Y1(POSE)`

DAG:

`G78 exact cells -> typed class actuators -> actual R realization -> scorer costates -> shared-program induction -> exact ZIP -> G83 whole-state arbitration`

Equations:

`theta* = argmin_theta S(R(decode(P, theta)), archive(P, theta))`

`LEARN = target / span(BULK, LANE, MOVABLE, HOOD, POSE)`

## Stores consulted

- G78 aggregate receipt:
  `/Volumes/VertigoDataTier/pact/taskspace_batch16_margin_base_scorer_cache_n600_20260726_r4/aggregate_receipt.json`,
  file SHA-256
  `f2422488fb8a3158d191b9a5fbc1150ce6e24a9c6bd7cace80b57845f86f7fb4`,
  sealed self-hash
  `fc6a2de90de0c8f8037c88bc4ae9853ab3bbffb9cb7f5a42b0e849098a15f3b7`.
- G46 target labels:
  `/Volumes/VertigoDataTier/pact/taskspace_fresh_teacher_batch16_20260726/11_target_labels/target_labels_n600_or_bounded.u8`,
  SHA-256
  `6d2ca48ac07323c7fc3a5299023bc291363192e10130eb3bc63d446bb8e65b85`.
- G85 exact receiver archive and double-decode custody.
- `src/tac/v2_compose/store_learn_split.py`,
  `src/tac/v2_compose/residual_compose.py`,
  `src/tac/boundary_math/lane_ground_factorization.py`,
  `src/tac/boundary_math/movable_site_coder.py`,
  `src/tac/boundary_math/hood_static_component.py`,
  `src/tac/witness_dsl/generative_taskspace_correction.py`.
- Current G72 exact-component compiler source observed at SHA-256
  `d9bd328fe87c03bed3d302c3a2085f6e2e97e63bb7b33a0ed719d0d22d7e5c09`.
