# DDM JO3 entrypoint and final reseal receipt — 2026-08-21

## Outcome

Seal r6 is **BLOCKED**.  The requested ready state would have been a fake:
the real fx5 carrier contains nine int12 endpoint coordinates across eight
pairs, while the pinned fresh Schur solver requires an in-domain central
difference at every coordinate.  Even after an endpoint-safe derivative cure,
the landed solver's mandatory uncompressed payload retention has a
2,907,449,989,536-byte lower bound for three stages plus reserve, against
603,076,071,424 free bytes measured by the real-config preflight.

The final authoritative seal is
`.omx/research/ddm_jo3_entrypoint_and_final_reseal_20260821/seal_r6/`:

- `compiled_config.json`: 16,028 bytes, SHA-256
  `3af9848e29b607675c96da33fc440094d376d0c07934d3ce256b67f49065c5a6`;
- workload identity:
  `b0a7713cd68ddc3cc8d57bd36787763ddda689ec57fbf5129fbfacb4495a99f1`;
- `READINESS.json`: SHA-256
  `79b571c9715fc78a4f2e52cfa6ad81c24f7adbc36c28e7acad8634c288d5d21a`;
- `FIRE_ORDER.json`: SHA-256
  `c93e26c40117650acbb6f316c271c2ff652c16fb2a6231fdc69bf5c9ba36e488`.

`FIRE_ORDER.json` disposition is `BLOCKED`; ordinal 3 has `argv: null`.  MAIN
must not launch training from this seal.

## Blocker delta

| r5 blocker | r6 disposition |
|---|---|
| `JO2_REMOTE_TRAINER_ENTRYPOINT_NOT_IMPLEMENTED` | Cleared by the local CPU JO3 entrypoint. |
| `RC2_BASE_ARGMAX_FIELD_MISSING` | Cleared by exact local custody and triple binding. |
| `FX5_BASE_POSE6_MISSING` | Cleared by exact local custody and triple binding. |
| `MEMORY_PREFLIGHT_BLOCKED:receipt absent` | Replaced by a real failed receipt carrying the two scale blockers below. |
| — | `FRESH_SCHUR_ENDPOINT_CENTRAL_DIFFERENCE_BLOCKED:endpoint_coordinates=9,endpoint_pairs=8`. |
| — | `RETAINED_FRESH_SCHUR_STORAGE_BLOCKED:all_stage_plus_reserve_minimum_bytes=2907449989536,free_bytes=603076071424`. |
| — | `MEMORY_PREFLIGHT_BLOCKED:memory preflight did not pass`, the aggregate gate. |

No heavy training or exact evaluation was launched.  The frontier pointer did
not move.

## Real entrypoint

`experiments/ddm_jo3_joint_objective_entrypoint.py` is a real local CPU target,
68,109 bytes, SHA-256
`92d2a2ab2a927d15dcdc1b97642edfdd4ceaf414113a3ad342b3423760c1f4a6`.
It implements the three required stages, exact float16 residual receiver,
through-R frozen SegNet/PoseNet objective, live and EMA state, optimizer/RNG/
dual/cursor checkpoints, full-field materialization, the pinned
`solve_fresh_compensation`, real coder race, shipped receiver execution,
decoded identity, scorer retention, and exact stage admission.  Every
materialized payload is persisted with path, byte count, and SHA-256.  A stage
cannot pass on proxy loss or a partial field.

The actual full-scale receiver call is fail-closed behind
`receiver_scale_preflight`; it was not invoked after the preflight proved that
the pinned mechanism would abort and could not retain its required outputs on
any admitted local tier.  The pinned sources remain:

- receiver-close:
  `f391b71963f6cd69611edac10df44408a49aa824942c31c3305d7971386edf5a`;
- residual runtime:
  `455b1b2ddce6ad1e9f3c5134f6d3bdb5e6f9c4d7cad582e76acce18b1ab32dea`.

The local wall-clock projection was derived before route selection.  It is
20.9–35.4 hours using one measured 1.3895-second full training step, JG1's
measured 39–64 seconds per pair for three carrier solves, conservative
full-field scoring, and a receiver/coder upper allowance.  Local compute is
feasible; local retained storage in the pinned representation is not.

## Payload custody

Both chartered payloads were independently rehashed and triple-bound without
a download or scorer rerun:

- `fx5_e1_argmax_n600.npy`: 117,964,928 bytes, SHA-256
  `e89e1ac083e5964975a1b4121cd1bc8bd91236256d6922f66c650246d7783c34`,
  `uint8 (600,384,512)`;
- `fx5_e1_first6_n600.npy`: 14,528 bytes, SHA-256
  `71f7d2639eb624f4d0eb89e40ac5956a74b1f72951dc7f07424468769af8350f`,
  `float32 (600,6)`.

Both use fx5 archive SHA-256
`4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841`
as `source_object_sha256`.  The fx5 body Pose6 table remains distinct from the
DALI source target.

## Real-config preflight

The retained receipt is
`experiments/.scratch/ddm_jo2_joint_objective_solve/ddm_jo3_entrypoint_final_reseal_20260821_r6_final/memory_preflight/MEMORY_PREFLIGHT.json`,
7,604 bytes, SHA-256
`09e5affa4b7224b2bbea09bc3bd1927b1e22da7ac1f8ba78c3a8b8b4a0a30a3d`.
Measured on `[macOS-CPU real-config preflight; no score authority]`:

- the real one-pair residual/R/SegNet/PoseNet forward and backward took
  1.3895 seconds and produced gradient norm 0.0012942249;
- peak RSS was 2,858,516,480 bytes; the streamed n600 projection was
  5,363,136,328 bytes under a 17,179,869,184-byte cap, leaving
  11,816,732,856 bytes; memory geometry itself passed;
- nine endpoint coordinates were measured at
  `(63,10), (67,10), (150,0), (150,7), (162,6), (214,8), (252,11),
  (450,9), (543,4)`;
- 589/600 rows require at least 176 retained candidates and the other 11
  require at least 28 under an endpoint-safe lower-bound formulation, for at
  least 103,972 candidates per stage;
- each candidate makes the landed solver retain one slave camera and one
  two-frame PoseNet input, three 874x1164 RGB camera payloads or 9,156,024
  bytes; this is at least 951,970,127,328 bytes per stage and
  2,855,910,381,984 bytes for three stages, before extra coordinate-descent
  passes and non-camera payloads;
- adding the 48 GiB non-solver reserve gives the typed
  2,907,449,989,536-byte storage blocker.

The one-pair probe retained all nine exact training surfaces and its YUV6
patch receipt with content records.  It did not consume the full-n600 scorer
slot.

## RECALL EVIDENCE

The recall sweep searched `.omx/research/` memos and arm receipts, final-arm
messages, canonical research index and `sub015_DAG_*` surfaces, live hot state,
lane/task ledgers, the fx5 runtime, and the canonical equation registry.
Content queries included `joint objective`, `fresh Schur`, `same-object`,
`candidate_object`, `carrier re-solve`, `Pose6`, `JG1`, `BU1`, `QS4`, `QS5`,
`PK4`, `fx5_e1`, `rc2`, and both payload hashes.  The equation command was
`.venv/bin/python tools/list_canonical_equations.py --json`.

Beyond the charter seeds, JG1 measured 98.7–100 percent recovery of tested
frame-1 pose damage through the native 12 int12 coordinates at 39–64 seconds
per pair, and BU1 measured a fresh joint solve beating naive stacking by
3.705x.  Those findings kept fresh cumulative-object recompilation in the
entrypoint and supplied the local wall-clock anchor.  The later exact endpoint
census and retention denominator changed the plan decisively: they invalidated
the provisional ready seal and required this blocked reseal.

## Verification and boundaries

- Focused tests: 30 passed; only the existing Pydantic `schema` shadow
  warnings remained.
- `ruff`, `py_compile`, and `git diff --check` passed on the changed surfaces.
- Two genuine review passes were recorded for every changed `.py` file.
- The real preflight exited 2 by design after persisting its failed receipt and
  all materialized payloads.

Measured: base payload identity, exact one-pair scorer gradient and RSS,
endpoint census, minimum retained-candidate denominator, storage lower bound,
float16 receiver equality, checkpoint restoration, and blocked r6 readiness.

Not measured: a trained stage, full n600 fresh solve, full decoded candidate,
stage B/H/Pose economics, an exact candidate score, contest-CPU, or
contest-CUDA.  The storage figure is a lower bound for this landed uncompressed
retention form, not a price for all possible lossless retention designs.

Own-vehicle frontier: **fx5_e1 S 0.14823186109359 @ 180,386 B
[contest-CUDA T4 n600]**, archive
`4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841`;
**UNMOVED**.

## NEXT_IF_RESUMED

- **BLOCKED** — Owner: receiver-close implementation owner; consumer store:
  the next JO3 seal and its real-config preflight receipt; fire trigger: land
  and review endpoint-safe one-sided finite differences for all nine measured
  endpoint coordinates without weakening the fresh same-object solve.
- **BLOCKED** — Owner: retained-payload representation owner; consumer store:
  the next receiver-close retained solve root; fire trigger: prove a lossless,
  byte-decodable deduplicated representation for every candidate payload whose
  measured three-stage bound fits an admitted storage tier, while retaining
  path, bytes, SHA-256, and deterministic reconstruction provenance.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN; consumer store:
  `experiments/.scratch/ddm_jo2_joint_objective_solve/<new-run-id>/`; fire
  trigger: both cures land, a fresh real-config scale receipt passes, r7
  reseals with zero blockers, and MAIN holds the unique local lane; then
  execute only the new seal's non-null training ordinal.

## LIVE-HYPOTHESES

- One-sided finite differences at the nine int12 endpoints may preserve the
  local Schur model because they change derivative sampling at only 9/7,200
  carrier coordinates while keeping the same exact candidate objective.
- Candidate cameras and PoseNet inputs may admit large exact deduplication or
  lossless delta coding because the landed solver currently stores repeated
  full arrays for small int12 coordinate changes; decoded byte identity, not a
  scalar-only surrogate, is the required test.
- The joint residual route remains worth testing after these apparatus cures
  because JG1 recovered 98.7–100 percent of tested pose damage through the
  same carrier family and BU1 measured a 3.705x gain from fresh joint solving.

## DEAD-ENDS

- The current f391b719 receiver at full fx5 n600: instance-closed because its
  central-difference Jacobian steps out of int12 range at nine coordinates.
- The current uncompressed per-candidate retention form on local/AP/Vertigo
  storage: instance-closed because its 2.907 TB lower bound exceeds every
  admitted tier; moving the same bytes between tiers does not cure it.
- The earlier r6 `READY` result and 48 GiB storage projection: invalidated by
  inspecting the real retention loop and measuring the full endpoint census.
- Modal training as a storage workaround: closed for this mechanism because
  the charter permits Modal only when local compute is impossible and short,
  while the payload-retention obligation remains regardless of provider.
- Carrying QS4 compensation across objects and reopening PK4 linear overlays:
  formulation-closed by their prior measured failures; JO3 preserves fresh
  cumulative-object recompilation and nonlinear joint descent.
- Re-downloading or re-scoring the two base payloads: closed because their
  exact local bytes and COMPLETE provenance receipt were verified.
