# DDM JO4 certified-retention reseal receipt — 2026-08-21

## Outcome

Seal r7 is **READY_TO_FIRE_UNDER_STANDING_GO** with zero blockers. The two r6
scale blockers are cured in the existing fresh same-object solve path; no
weakened solve, retention skip, heavy training, full-n600 scorer job, or exact
evaluation ran in this arm.

The authoritative seal is
`.omx/research/ddm_jo4_certified_retention_reseal_20260821/seal_r7/`:

- `compiled_config.json`: 16,047 bytes, SHA-256
  `5ba2a9b8f01d0295fc3a96e49256cf7b503c45d31d69fce958d3bb2d8881c69b`;
- workload identity:
  `8f8a319a10771e99763e596760b91a67f265fb0f0bb3ef5d52ac519749f70241`;
- `READINESS.json`: 469 bytes, SHA-256
  `41824130099247239a510f4725f6ebb1f168e692175b2849ed26898f2beb6b85`;
- `FIRE_ORDER.json`: 6,385 bytes, SHA-256
  `50c1d74a31fac7dfeadad5370a29c11ec58aba57b5acd02df11ea38fc3350a2b`.

`FIRE_ORDER.json` is owned by MAIN, has `current_disposition=READY`, and has a
non-null argv at every ordinal, including the governed local training command
at ordinal 3. This arm did not execute that command.

## Blocker delta

| r6 blocker | r7 disposition |
|---|---|
| `FRESH_SCHUR_ENDPOINT_CENTRAL_DIFFERENCE_BLOCKED:endpoint_coordinates=9,endpoint_pairs=8` | **CLEARED.** The nine endpoint coordinates use matched unit-step inward one-sided differences; all other 7,191 coordinates retain central differences. The real census executed both-probe in-domain assertions and found 0 blocked coordinates. |
| `RETAINED_FRESH_SCHUR_STORAGE_BLOCKED:all_stage_plus_reserve_minimum_bytes=2907449989536,...` | **CLEARED.** Explored non-winner camera buffers now receive fail-closed certified rebuild records; each pair winner is regenerated, checked against its exploration hashes, and retained in full. The re-derived three-stage projection plus reserve is 68,243,679,333 bytes against 602,862,194,688 bytes free. |
| `MEMORY_PREFLIGHT_BLOCKED:memory preflight did not pass` | **CLEARED.** The refreshed real-config receipt passed and is bound into r7. |

## Endpoint-safe derivative cure

`experiments/ddm_jo2_receiver_close.py` keeps the original second-order central
stencil at every interior int12 coordinate. At `-2048` it uses probes
`(-2048,-2047)`; at `2047` it uses `(2046,2047)`. Both endpoint stencils use
the same unit step and are explicitly labeled first-order `O(h)`, rather than
silently inheriting the central stencil's `O(h^2)` truncation claim.

The real fx5 census measured:

- 7,191 central second-order coordinates;
- 6 forward one-sided first-order coordinates;
- 3 backward one-sided first-order coordinates;
- 9 endpoint coordinates across 8 pairs;
- 0 blocked coordinates and 0 out-of-domain probe points.

The exact endpoint list remains `(63,10), (67,10), (150,0), (150,7),
(162,6), (214,8), (252,11), (450,9), (543,4)`.

## Certified two-tier retention cure

`experiments/ddm_jo3_joint_objective_entrypoint.py` now supplies the retention
layer to the pinned receiver-close solve. Every explored candidate is still
materialized on the real receiver path. Before a non-winner camera buffer can
be released, an atomic compact manifest records both materialized camera
payload hashes and byte counts plus the exact regeneration tuple:

`entrypoint SHA + workload identity + base archive SHA + stage id + pair id + candidate coordinate delta`.

The certificate write is fail-closed. Missing, drifting, unwritable, escaped,
or stale certificates refuse the candidate. Codes and Pose6 vectors remain
full-byte NPY payloads. After selection, the pair winner is regenerated once;
its camera hashes and Pose6 must match the explored candidate exactly, then its
carrier codes, slave camera, two-frame PoseNet input, and Pose6 output are
retained in full. A completed stage solve requires an inventory covering 600
winner receipts plus every rebuild manifest. Existing stage checkpoints,
admission receipts, scorer surfaces, receiver repeats, and coder-race
artifacts remain unchanged and full-byte retained.

## Re-derived storage projection

The passed preflight receipt is
`experiments/.scratch/ddm_jo2_joint_objective_solve/ddm_jo4_certified_retention_reseal_20260821_r7_final/memory_preflight/MEMORY_PREFLIGHT.json`,
11,025 bytes, SHA-256
`f28d5471d5dd523fa66f9f4311058739a56c58ce2c97313b01c70f25528b0174`.
Axis: `[macOS-CPU real-config preflight; no score authority]`.

| Component | Re-derived bytes |
|---|---:|
| Minimum explored candidates per stage | 104,104 candidates |
| Compact certificate row bound | 488 B/candidate |
| Certificate manifest context bound | 721 B/manifest |
| Minimum manifests per stage | 4,167 manifests |
| Certified rebuild records per stage | 53,807,159 B |
| Full explored code/Pose small state per stage | 19,987,968 B |
| Full pair winners per stage | 5,494,228,800 B |
| One-stage retained projection | 5,568,023,927 B |
| Three-stage retained projection | 16,704,071,781 B |
| Fields/checkpoints/extra-pass/coder reserve | 51,539,607,552 B |
| **Three stages plus reserve** | **68,243,679,333 B** |
| Measured free bytes | **602,862,194,688 B** |
| Measured headroom over projection | **534,618,515,355 B** |

The 68.24 GB figure is a baseline projection with a separate 48 GiB reserve,
not a measured full-run high-water mark. Extra coordinate-descent passes are
not counted as a fixed denominator; every later write remains fail-closed if
the reserve proves insufficient. The cured projection is 8.83 times below the
measured free tier.

## Real-config preflight

Measured on the same non-authority local CPU axis:

- one real residual/R/SegNet/PoseNet forward and backward: 1.7786169 s;
- nonzero gradient norm: 0.0012942249;
- peak RSS: 2,855,567,360 B;
- streamed n600 projection: 5,360,187,208 B under 17,179,869,184 B;
- memory headroom: 11,819,681,976 B;
- derived governed local wall range: 76,603.0–128,803.0 s, or
  21.28–35.78 h.

All nine one-pair training payloads and the YUV6 patch receipt were retained.
The preflight consumed no full-n600 scorer slot.

## RECALL EVIDENCE

The recall sweep searched the full `.omx/research/` memo/receipt corpus,
arm-final messages, `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*`, the canonical
equation registry via
`.venv/bin/python tools/list_canonical_equations.py --json`, live hot state,
task/P0/lane ledgers, the r6 seal and receipt, and the actual JO1/JO2/JO3 source
and tests. Content queries included `fresh Schur`, `same-object`,
`solve_fresh_compensation`, `central difference`, `one-sided`, `endpoint`,
`certified rebuild`, `retention mode`, `candidate coordinate delta`,
`READY_TO_FIRE_UNDER_STANDING_GO`, `FIRE_ORDER`, `JG1`, `BU1`, `QS5`, `PK4`,
and `fx5_e1`.

Beyond the charter seeds, the search confirmed that the existing receiver
already inventories full real coder/receiver artifacts and that the current
JO2 freshness fingerprints bind the cumulative semantic object, camera field,
base Pose6, pair, and archive. That kept this cure inside the same solve path
and limited certification to the repeated exploration-camera bytes. No
current-vehicle memo, equation, DAG row, or task-ledger row in the searched
scope supplied a conflicting endpoint-safe implementation or a cheaper
already-landed JO retention mode; the implementation plan otherwise remained
unchanged.

## Verification and review

- Focused tests: 38 passed; only the two pre-existing Pydantic `schema`
  shadow warnings remained.
- `ruff`, `py_compile`, and `git diff --check` passed on every changed Python
  surface.
- The bounded P0 payload-retention scan returned 0 findings on all changed
  Python files.
- Two genuine post-fix review passes were recorded for all four changed
  Python files. The first adversarial pass found and fixed missing strict
  storage-receipt arithmetic and made completed FIRE_ORDER ordinals carry
  reproducible argv; the second pass was clean.
- Assumption challenge: the certified-rebuild tier assumes the SHA-pinned
  receiver is deterministic from the recorded tuple. Violating that assumption
  would require retaining additional hidden state or full bytes for every
  candidate, not a weaker certificate. This assumption is supported by the r8
  dual-decode identity receipt; r7 additionally repeats every winner and fails
  closed on any camera or Pose6 mismatch.

## Boundaries

Measured: exact source and payload identities; all 7,200 derivative stencil
domains; one real scorer-gradient/RSS probe; serialized certificate-row size;
candidate and manifest denominators; free storage; r7 readiness and complete
argv custody.

Not measured: a trained stage, a full-n600 fresh solve, realized certificate
volume at the full-run endpoint, full decoded candidate, stage B/H/Pose/rate
economics, exact candidate score, contest-CPU, or contest-CUDA. This arm moved
no frontier pointer; it sealed the governed run that MAIN now owns.

Own-vehicle frontier: **fx5_e1 S 0.14823186109359 @ 180,386 B
[contest-CUDA T4 n600]**, archive
`4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841`;
**UNMOVED**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN; consumer store: `experiments/.scratch/ddm_jo2_joint_objective_solve/ddm_jo4_certified_retention_reseal_20260821_r7_final/`; fire trigger: MAIN holds the unique local lane, confirms r7 `READINESS.json` still has zero blockers and all source/input triples still match, then executes only ordinal 3 from `.omx/research/ddm_jo4_certified_retention_reseal_20260821/seal_r7/FIRE_ORDER.json`.

## LIVE-HYPOTHESES

- The nine first-order endpoint columns will preserve useful local Schur
  direction quality because they change only 9 of 7,200 Jacobian coordinates,
  keep the same unit step, and remain inside the exact realized objective.
- The full run will remain below the admitted storage tier because winner
  bytes dominate the 16.70 GB three-stage baseline while compact certificates
  are only 53.81 MB/stage and the separate reserve is 48 GiB; actual extra-pass
  volume is still unmeasured and therefore watched fail-closed.
- The governed JO1 run remains worth firing because JG1's 98.7–100 percent
  recovery and BU1's 3.705x fresh-joint advantage support the same native
  carrier route, while r7 now removes the two apparatus blockers that prevented
  measuring its real stage economics.

## DEAD-ENDS

- Central differences at int12 endpoints are closed for this solver because
  one probe necessarily leaves the legal carrier domain at nine measured
  coordinates.
- Full uncompressed camera retention for every explored candidate is closed
  for this run because its measured r6 lower bound was 2.907 TB, over 4.8 times
  the available tier.
- Renaming scalar-only disposal as retention is closed: r7 requires an atomic
  per-candidate camera hash/byte certificate and refuses the candidate if that
  record cannot be written.
- Reopening stale cross-object QS4 compensation or PK4 linear overlays remains
  closed by their prior measured failures; r7 preserves fresh cumulative-object
  recompilation and nonlinear native-carrier descent.
- Launching the 21.28–35.78 h training job from this arm is closed by the
  charter; the complete ordinal-3 command is sealed for MAIN only.
