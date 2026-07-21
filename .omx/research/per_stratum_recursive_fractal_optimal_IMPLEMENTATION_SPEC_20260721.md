# Per-stratum recursive-fractal optimal-treatment custody audit — implementation spec

UTC: 2026-07-21T19:12:17Z

Lane: `lane_per_stratum_recursive_fractal_optimal_20260721`

Authority: research-only, local `$0`, MAIN landing review required. Pointer
`0.1910828242 [contest-CPU]` stays unchanged.

## Outcome to implement

Build one deterministic, fail-closed measurement tool that answers the byte-verdict
question supported by the bytes that actually exist. It must compose no imaginary
payloads and must never translate v8/v9 design estimates into measured archive bytes.
It will:

1. validate the settled n64/n600 BEV receipts and all 600 persisted pose stages;
2. prove whether the stored trajectory contains independently observed rotation;
3. exercise `CalibratedGeometry` only with an explicit scorer-resolution `K` identity
   canary, and refuse causal `R`/`t` attribution when observed homographies are absent;
4. report OpenPilot pitch/yaw spread and validity windows as bounded vanishing-point
   sensitivity, never as an explained fraction of the 39/47-pixel residual;
5. validate exact M1 and S4 receiver/archive custody and surface their measured n600
   hard-oracle rows as controls;
6. bind the c2 per-class residual taxonomy and settled v8/v9 treatments into a
   per-stratum table, leaving per-stratum bytes `null` when no parser-consumed section
   establishes unique-home attribution;
7. emit `NO_VERDICT_RECEIVER_RATE_CUSTODY` for the requested v9 composition unless a
   real encoder, parser, standalone receiver, exact archive, n600 decode, and hard
   scorer receipt are all present.

## Owned files

- `tools/measure_per_stratum_recursive_fractal_optimal.py`
- `tools/tests/test_measure_per_stratum_recursive_fractal_optimal.py`
- `.omx/research/per_stratum_recursive_fractal_optimal_20260721T191217Z_receipt.json`
- `.omx/research/per_stratum_recursive_fractal_optimal_20260721T191217Z.md`
- `.omx/research/per_stratum_recursive_fractal_optimal_DAG_FEED_20260721.md`
- `.omx/research/per_stratum_recursive_fractal_optimal_REUSE_MANIFEST_20260721.md`

The lane registry/audit files are already owned operational state. Do not edit other
code, canonical equations, scorers, receivers, or prior research artifacts.

## Inputs and exact custody

Default read-only inputs:

- BEV root:
  `/Volumes/VertigoDataTier/pact/evidence/bev_staticity_v2_20260721/canonical_v1`
  - `receipt_n64.json` SHA-256
    `94a7d7b5635e04d5da6f22e1d4f2e5b8d170a9dc95923e3835b9421aedb8bbba`
  - `receipt_n600.json` SHA-256
    `c3ec847ba5ca43246f01af12f7bd650b14aba2784eb1878c29c16f8a4469ab96`
  - 600 `measurement_stages_n600/frame_*.json` stages.
- M1 archive:
  `/Volumes/VertigoDataTier/pact/evidence/m1_byteclose_20260721/m1_candidate_archive.zip`,
  90,566 bytes, SHA-256
  `a386a854e2483f839191f6c9da781f60b49774b71830b9baccee259be85edf8c`.
- M1 exact harness receipt and chunked hard-oracle decomposition under the same SSD
  root. The decomposition must retain seed 1234, 38 completed chunks, exact through-R
  `d_seg=0.0035157945421006942`, official `d_pose=127.36588287353516`, and the five
  per-class mismatch rows.
- S4 repo receipts plus SSD archive:
  `/Volumes/VertigoDataTier/pact/evidence/s4_composer_20260721/canonical_s4_20260721/archive.zip`,
  451,191 bytes, SHA-256
  `d84f2fe053239d1542ba381420e9569d431ed2015e22e60e49ef48f1321696ed`.
- v9 measurement receipt and the required module paths named in the v9 build spec.
- c2 canonical-equation constants from
  `tac.canonical_equations.perclass_stratum_carrier_taxonomy_20260716` plus the dated
  taxonomy memo for the complete residual table.

Every input file used must be SHA-256 bound in the output receipt. Missing inputs,
schema drift, hash drift, non-600 stage counts, pointer mutation, seed drift, or
authority promotion flags fail closed.

## Measurement rules

### A. Road/Lane calibration custody

- Read every n600 stage's `absolute_f1_pose`, `calibrated_cross_xi`, and
  `calibrated_within_xi`.
- Measure maximum rotation-matrix deviation from identity, maximum rotation-vector
  magnitude, and count of nonzero-rotation frames/transitions.
- Confirm the source receipt's `s_r=0` and record that `pitch_rad=-0.05` is custody
  metadata but is not used by `xi_from_pose_calibration` to construct rotation.
- Record `observed_pixel_homography_count=0`. Generated homographies from the same
  stored pose are circular and must not be admitted as independent evidence.
- Instantiate `CalibratedGeometry(fx=400.3, fy=399.5, pp=(256,192), width=512,
  height=384)` and decompose identity `H`; require exact-near identity `R`, zero `t`,
  and zero pose. Do not use the module defaults because their native/scorer units are
  inconsistent per settled #326.
- Compute OpenPilot vanishing-point sensitivity with
  `u=cx+fx*tan(yaw)`, `v=cy-fy*tan(pitch)/cos(yaw)` using the reconciled horizon 174
  as nominal. Emit:
  - spread thresholds: pitch 4 degrees, yaw 2 degrees;
  - validity windows: pitch `[-0.09074,0.17]` rad, yaw `[-0.06912,0.06912]` rad;
  - exact pixel shifts and the minimum one-axis angular equivalents for Road/Lane
    p50 residuals.
- `calibration_explained_fraction` and `genuine_geometry_fraction` remain `null`.
  Verdict: `UNIDENTIFIABLE_FROM_CURRENT_CUSTODY`; the bounded VP shifts are
  sensitivity only. Exact attribution requires independent observed H/flow or
  cameraOdometry/liveCalibration custody and held-out R-only/t-only counterfactuals.

### B. Byte-closed controls and requested composition

- Rehash both existing archives. Copy no large artifact.
- M1 row: receiver-closed 90,566-byte control, n600 hard CPU-torch/numpy, seed 1234,
  chunked hard-oracle decomposition; full-richness exactness is false because
  mismatch pixels are nonzero and Pose is catastrophic.
- S4 row: receiver-closed 451,191-byte control with exact n600 deterministic decode;
  full-richness exactness is false from its measured hard scorer.
- Cap accounting:
  - `cap_bytes=154600`;
  - `rate=25*cap_bytes/37545489`;
  - remaining sub-0.15 distortion budget
    `0.15-rate`;
  - archive ratios use exact integer bytes.
- Requested v9 row must require an actual candidate archive, parser-consumed section
  registry, n600 decode hash, full scorer receipt, and per-stratum unique-home byte
  attribution. Because those are absent, total and per-dimension/per-stratum bytes are
  `null`, not zero or an estimate.

### C. Per-stratum table

Exactly five self-detected class rows: Road, Lane, Undrivable, Movable, MyCar. Each row
contains:

- settled v8 carrier and v9 dimensional treatment (recall, not a new derivation);
- c2 residual buckets/weights with measurement scope;
- M1 hard-oracle GT pixels, mismatch pixels, conditional error, and all-pixel
  `d_seg` contribution;
- coordinate/basis/temporal/quantization/boundary/composition status;
- `measured_unique_home_bytes=null` and a precise blocker unless an exact
  parser-consumed payload establishes the number;
- verdict scope that rejects only the current formulation/custody, never the family.

The table must explicitly correct the OpenPilot representation claim: v0.9.7/current
lane lines are four sampled 33-point `(x,y,z)` curves with probabilities/stds and two
sampled road edges; `LaneLine` cubic/polynomial is this repo's compression abstraction,
not an OpenPilot-native polynomial carrier.

## Output and hygiene

- Tool writes one atomic canonical JSON receipt to an operator-selected path.
- Production run writes the full receipt under
  `/Volumes/VertigoDataTier/pact/evidence/per_stratum_recursive_fractal_20260721/`
  and copies only a compact deterministic receipt into `.omx/research/`.
- No raw frames, caches, scorer loads, or large artifacts are created. The receipt
  declares zero bulk bytes created and records the SSD free-space preflight.
- The durable memo, DAG FEED, and REUSE MANIFEST must distinguish MEASURED, DERIVED,
  SETTLED-RECALL, and NO_VERDICT claims, record STORES CONSULTED, triality, pointer
  delta, and MAIN review boundary.

## Tests and acceptance

Focused tests must cover:

1. cap arithmetic and exact archive ratios;
2. VP sensitivity and equivalent-angle math;
3. scorer-K identity homography canary;
4. rotation audit detects identity-only and nonzero-rotation fixture stages;
5. missing observed-homography custody forces null attribution;
6. per-class hard-oracle accounting sums exactly to aggregate mismatch pixels and
   exact rational `d_seg`;
7. missing v9 module/archive/receiver surfaces cannot become zero-byte measurements;
8. deterministic receipt bytes on repeated runs with fixtures.

Verification: Ruff on owned Python files, `py_compile`, focused pytest, two-pass
`review_tracker` for both Python files, and a production run against the exact SSD
inputs. Do not run a new 20-minute hard scorer; consume the already-settled exact
receipts after rehashing their archive/output custody. Commit only after review and
hash checks. MAIN must independently review the branch before merge.

## Do not touch

- `src/tac/calibrated_geometry.py` or its tests;
- canonical equation registries;
- v8/v9 carrier modules;
- existing scorer/receiver/archive bytes;
- other worktrees, main, or shared provider state;
- any paid/remote/GPU surface.
