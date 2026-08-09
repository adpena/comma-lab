# DDM PK2 pose-carrier representation implementation spec

Tags: [no-triality] [p0-ledger-ok]

## Objective

Build one deterministic, resumable experiment runner that executes the current
`ddm_pk2_pose_carrier_representation.md` ladder against the reproduced PR130
CPR1 archive.  The runner must produce real parse-back arrays and real archive
bytes, then optionally score a seeded stratified-random `n>=120` subset through
the frozen CPU-torch SegNet and PoseNet.  This is `[macOS-CPU advisory]`,
`score_claim=false`, never a contest score.

## Governing inputs

- Charter: `.omx/research/charters/ddm_pk2_pose_carrier_representation.md`.
- Common contract: `.omx/tmp/codex_runs/_common_contract.md`.
- Archive:
  `/Volumes/VertigoDataTier/pact/ddm_pr130_reproduce_20260809/reproduction/archive.zip`,
  required SHA-256
  `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`.
- Read-only source runtime:
  `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code/carrier_codec.py`
  and `inflate.py`.  Require the chartered hashes and schema before work.
- Frozen Ada DALI target cache:
  `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/caches/gt_cache_600_official_ada.pt.xz`.
  The cache `seg` tensor SHA-256 must equal the published decoded-token SHA
  `c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`.
- Reusable prior method:
  `src/tac/torch_vehicle/pose_film.py`; record repository HEAD and file hash,
  but do not modify `src/tac`.

## Correction that the runner must prove

The reproduced CPR1 header contains 600 x 12 = 7,200 coefficient symbols, not
1,200 x 12 = 14,400.  Therefore 79,076 bits is 10.98 bits/value, not 6.59.
The basis tensor is stored in an int8 container but its deployed source
precision is already signed int5, range -15..15.  Treat int7/int6/int5 as the
same deployed precision row; do not falsely claim an int8-to-int5 reduction.

## OPTIMAL FORM

The reference family is a counted low-rank or smooth-trajectory packet with a
residual, per-dimension quantizers, and a real receiver dispatch. CPR1 rows
that merely store a projected full array are explicitly toy brackets: they can
measure distortion response and bound archive behavior, but cannot close the
low-rank, spline, or transform-packet family. Signed-int4/int3 basis rows are
real unchanged-receiver representation changes because they alter the source
alphabet and scales before the incumbent CPR1 encoder.

## Files and mutation boundary

- New runner: `experiments/ddm_pk2_pose_carrier_representation.py`.
- New focused tests:
  `experiments/tests/test_ddm_pk2_pose_carrier_representation.py`.
- Durable research outputs are written by the primary agent after measurement
  under `.omx/research/ddm_pk2_pose_carrier_representation_20260809/`.
- Bulk candidates, checkpoints, rendered caches, and stage receipts go only to
  `/Volumes/VertigoDataTier/pact/ddm_pk2_20260809/`.
- Never edit the intake clone, `upstream/`, `tools/`, `src/tac`, or the three
  protected common-contract files.

## CLI and resumability

Provide explicit `analyze`, `score`, and `finalize` or equivalent stages.  A
long stage must require an SSD output root and `--resume-from`.  State is an
atomic JSON file; every completed stage has a distinct stage receipt and all
candidate packets/archives are preserved under stable names.  A rerun verifies
input and output hashes and skips valid stages.  Run storage preflight before
rendering/scoring and fail closed when free space is insufficient.  Never cite
or persist `/tmp`.

## Baseline extraction and real receiver checks

1. Extract the single stored `p` member from the exact archive in memory.
2. Decompress the model bundle, isolate semantic, CPR1 carrier, HPAC, and token
   streams, and reproduce their exact sizes and hashes.
3. Import the read-only `carrier_codec.py`; decode the carrier with
   `basis_count=12*3*24*32`, `frames=600`, `dimensions=12`.
4. Re-encode the decoded arrays with the real `encode_compact_carrier`; demand
   byte identity.  Import the read-only `inflate.py` and demand that
   `unpack_semantic_pose` returns the same basis and coefficient tensors.
5. Materialize a deterministic one-member candidate archive for every row by
   replacing only the carrier section, recompressing the same raw model bundle
   with the incumbent XZ filter, and preserving the HPAC/token bytes.  Parse the
   resulting archive back through the same extraction and real inflate parser.
6. Record both carrier-section bytes and full archive bytes.  The joint score
   arithmetic uses full archive bytes.

## Ladder rows

Every scorer-bearing row must report full archive bytes, delta bytes, d_pose,
d_seg, and
`delta_S = 100*delta_d_seg + sqrt(10*d_pose_candidate) - sqrt(10*d_pose_baseline) + 25*delta_archive_bytes/37_545_489`.
All output numbers carry `[macOS-CPU advisory]`, `score_claim=false`.

### A: coefficients

- Measure per-dimension lag autocorrelation, power-spectrum summaries,
  cross-dimension singular-value energy, and the exact incumbent Rice bit
  count.
- Race exact reversible predictors on the same absolute int12 lattice:
  incumbent first difference, second difference, fitted integer AR(1..4), and
  piecewise-linear/cubic-knot prediction plus exact modulo residual.  Predictor
  parameters and knots are counted.  These are representation rows because the
  residual model changes; never rerun a bare outer-coder race on the unchanged
  bytes.
- Implement a genuine low-rank-plus-residual family, not rank-1/no-residual.
  Search ranks 4..11 and multiple per-factor quantizer levels/scales.  Decode to
  a full 600x12 coefficient tensor before the real render.  If a new packet
  magic is needed, stage an experiment-owned receiver overlay that dispatches
  only that magic and delegates all unchanged work to the pinned intake
  runtime.  Parse-back equality to the packet's declared reconstructed arrays
  is mandatory.

### B: basis

- Measure spatial spectrum/correlation and cross-plane singular-value energy.
- Race reversible integer spatial transforms (at least Haar or lifting
  wavelet) with counted transform metadata against incumbent CPR1.  A transform
  plus entropy coding is a new representation; do not relabel compression of
  unchanged bytes as a representation win.
- Race lossy spatial DCT/low-pass and cross-plane low-rank reconstructions at
  multiple ranks/retained-coefficient levels.
- The deployed precision baseline is signed int5.  Race int4/int3 effective
  precision with per-dimension scale/outlier-clip search through the unchanged
  real CPR1 codec.  If per-plane scale support is added, count all 36 scales and
  parse them through the experiment receiver overlay.  Include int5 identity as
  the correction/control row.

### C: capacity

- Score each single-dimension drop first, rank dimensions by measured joint
  value per byte, then test nested drops in that order.  Encoding zeros through
  a nominal 12D CPR1 section is permitted as an unchanged-runtime rung; a truly
  smaller-D packet must count its dimension map and pass experiment receiver
  parse-back.
- Label this only as response of the existing trained carrier.  Retraining a
  smaller carrier is out of scope and may be recommended only with a fire
  trigger derived from the measured curve.

### Composition and beyond-seed gauge row

- Race A and B independently first.  Compose the best measured A and B rows and
  re-score; never add their deltas arithmetically.
- Include a small, seeded gauge-rotation search across the 12 carrier
  dimensions because `C @ B` is invariant before re-quantization.  Measure only
  materialized CPR1 rows; label it beyond-seed, not a chartered prerequisite.

## Scoring

- Use `tac.subset_selection.MODE_STRATIFIED`, seed `20260809`, 10 blocks, and
  the target-cache pose-center energy governing statistic.  Require `n>=120`;
  never prefix.
- Render the semantic master exactly from the archive-unpacked semantic model
  and the frozen cache's token maps using the pinned intake runtime functions.
  Render carrier slaves with the pinned `normalized_basis`, amplitude 64,
  camera bicubic path, clamp, and uint8 round used by `inflate.py`.
- Run frozen upstream CPU-torch PoseNet and SegNet.  Cache the baseline master
  frames and target scorer outputs under the SSD root with hashes and a stage
  receipt.  Candidate frame1 bytes must equal baseline frame1 bytes; still run
  and report the baseline selected-sample d_seg, then report every candidate's
  measured identical d_seg and delta zero.
- Score all shortlisted rows in batches in one governed scorer stage.  Do not
  launch a full n600 stage while another live n600 scorer owner exists.  The
  n120 sweep may run only when the primary agent confirms it will not violate
  the live scorer ledger.  Full n600 is reserved for the winning composed row
  and must be left queued if the scorer slot remains occupied.

## Required outputs

- Machine-readable analysis receipt with input hashes, actual CPR1 counts,
  selection provenance, temporal/spatial diagnostics, every candidate packet
  and archive hash, parse-back results, scorer rows, and exact delta-S terms.
- Ranked scorer table and separate byte-only table.  No byte-only row may be
  described as a score row.
- Explicit unrun rungs and scoped verdicts.
- Tests must mutation-catch a fake encoder that ignores its input, a decoder
  that does not reconstruct declared arrays, a prefix selector, an uncounted
  metadata path, and arithmetic that uses section bytes instead of archive
  bytes.

## Verification

Run focused tests, compile the runner, run `--help`, execute the exact baseline
round-trip, execute the scorer-free ladder, then the n120 scorer stage if
ledger-safe.  Do not commit.  Return changed paths, exact commands, receipt
paths/hashes, test results, and any blocker to the primary agent for review and
serializer landing.
