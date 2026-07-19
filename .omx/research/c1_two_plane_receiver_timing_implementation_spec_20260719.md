# C1 two-independent-plane receiver timing — implementation specification

Date: 2026-07-19 UTC  
Lane: `lane_c1_two_plane_receiver_timing_20260719`  
Task: #561 / SPEC_v10 §8 C1  
Authority: local train-free BUILD + MEASURE only; MAIN landing review required

## Outcome

Build an additive scorer-free receiver/timing path for one full-n600 production
archive whose every pair contains two independently described exact `uint8`
scorer planes under `description-frame0.v1`.  Run that same archive through a
single-worker attribution baseline and through two fresh, individually timed
four-worker process-pool invocations.  C1's local receiver
surface clears only if both planes of every pair satisfy exact factor-2
integer-numerator equality, strict parse/re-encode is byte-identical, the two
complete decoded outputs are byte-identical, and all required component timings
are present.

Operator amendments received at `2026-07-19T14:07:02Z` and
`2026-07-19T14:11:02Z` supersede the original per-invocation budget reading.
The 1,800-second contest constant applies to the complete official evaluation,
`T_inflate + T_scoring`, on contest hardware (`upstream/README.md:114`), not to
either local inflate.  Local timing must never be converted to PASS/FAIL with an
invented margin.  The final timing verdict is exactly one of
`CLEARLY_UNDER`, `CLOSE -> MODAL_MEASUREMENT_OWED`, or `CLEARLY_OVER`, supported
only by a measured paired local/contest timing spread.  If no admissible paired
anchor exists, default to `CLOSE -> MODAL_MEASUREMENT_OWED` and emit a ready,
unfired full-`evaluate.sh` Modal ticket specification under the #381 envelope.

The hardware-exploitation amendment received at `2026-07-19T14:18:32Z`
requires the contest-facing CPU receiver to use at least a four-worker process
pool across independent pairs, with fixed-order assembly.  The serial number is
only an attribution baseline; the three-way timing verdict is scoped to the
exploited configuration.  Double-decode byte identity must hold on that
parallel path.  A deterministic T4 integer-lattice solve remains a derived
follow-on estimate/path, not a measured CUDA row in this local-only task.

The local-twin addendum received at `2026-07-19T14:21:03Z` additionally requires
an MLX/Metal-optimized developer twin.  It is admitted only by per-plane,
per-pair byte parity against the NumPy/CPU integer authority on at least six
real pairs.  Every twin timing/output row is tagged
`[macOS-MLX research-signal]`, fixes `score_claim=false`, and is excluded from
the contest budget verdict.  A Metal integer-op divergence is preserved as a
first-class scoped finding and does not invalidate the contest CPU receiver.

This task does not authorize training, paid/provider work, contest evaluation,
score/promotion claims, submission, or movement of pointer
`0.1910828242 [contest-CPU Linux x86_64]`.

## Frozen inputs and derivation

- Canonical real cache:
  `/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz`,
  SHA-256 `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.
- Required fields and shapes: `n_pairs=600`; `gt_f0` and `gt_f1` are independent
  `uint8[600,874,1164,3]`; `lstars[600,384,512]` and `gt_poses[600,6]` support
  the encode-side hard-oracle spot check.
- Derive each scorer plane exactly as the already-settled Task #541 rung-E
  control: apply `DisjointResizeOperator(874,1164 -> 384,512)` in integer
  numerator arithmetic, then nonnegative round-half-up to `uint8`.  Derive Y0
  from `gt_f0[pair]` and Y1 independently from `gt_f1[pair]`.
- Fresh read-only streaming re-derivation fixes the aggregate raw-C plane
  digests: Y0 SHA-256
  `5e86e419cdd5bd41c9482cabc78cf27cec22281098b64c715d91f1f067d11566`;
  Y1 SHA-256
  `6a731946e3d9de82089c90de9784c5a5bc72c607c963fb6f79dac16f00ac89bc`.
- Pair IDs are exactly `0..599`, in order.  Refuse shared memory, equal per-pair
  plane digests, equal per-pair plane bytes, or any path that selects
  `repeat-frame1`.
- The description codec is the closed production
  `predictor-residual-u8.v1`; use the measured Task #541
  `spatial-smooth-121.v1` mode.  Both the charged frame-0 bootstrap and the
  exact frame-1 residual remain counted.
- Frozen hard-oracle weights: SegNet SHA-256
  `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`;
  PoseNet SHA-256
  `0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576`.

The sacred donor tree
`experiments/results/levelset_n600_witness_20260717T113932Z/` is read-only.
Snapshot it before and after the measurement and fail if metadata changes.

## Owned implementation surface

Implementation worker owns only these additive or append-only paths:

- `src/tac/witness_dsl/v10_two_plane_timing_receiver.py`;
- `src/tac/tests/test_v10_two_plane_timing_receiver.py`;
- `tools/measure_v10_two_plane_receiver_timing.py`;
- `tools/tests/test_measure_v10_two_plane_receiver_timing.py`.

If a small additive public helper in
`src/tac/witness_dsl/v10_production_receiver.py` is strictly necessary to
package an already encoded predictor payload without recompressing it, stop and
route the proposed signature to the parent before editing that established
file.  Do not touch PDW1, PDW2, trainers, scorer code, the source cache, the
sacred donor, canonical pointer files, or evaluator paths.

## Receiver ABI

The receiver module must expose a typed result and a callable equivalent to:

```python
timed_inflate_two_plane_archive(
    archive_dir,
    output_dir,
    video_names_file,
    *,
    timing_receipt_path,
    worker_count=4,
    resume=False,
    stop_after_pairs=None,
) -> TwoPlaneTimedInflateResult
```

It must:

1. read exactly one stored `0.bin` member; call the production parser; refuse
   every codec/policy except `predictor-residual-u8.v1` paired with
   `description-frame0.v1`; refuse quotient residuals for this C1 control;
2. reserialize the fully parsed packet, including prefix, canonical header,
   every section length, and every section payload, and require byte identity
   with the original packet (and deterministic ZIP identity where rebuilt);
3. expand the predictor description once into independently owned Y0 and Y1
   arrays; require exact n600 geometry/pair IDs and non-alias/non-copy checks;
4. solve Y0 and Y1 independently with the existing factor-2 integer primitive;
   use one worker only for the attribution baseline or a fixed four-worker
   process pool for the exploited run; no helper or branch may assign/copy X1
   to X0 or vice versa;
5. preserve one write-once pair-stage binary plus a deterministic, timing-free
   pair manifest after each pair; the manifest binds archive, packet, pair ID,
   Y0/Y1 digests, X0/X1 digests, stage bytes/hash, policy IDs, and exact
   numerator counts;
6. preserve deterministic timing-free chunk manifests (canonical 12-pair
   chunks) that bind the pair-manifest hashes; consume worker results and
   assemble the final raw file in pair-ID order, atomically, from reopened
   stages regardless of completion order;
7. in a separate verification stage, reopen every pair stage, recompute exact
   numerators against both decoded targets, validate all manifests, final raw
   size/hash, and strict packet re-encode;
8. support crash resume from the write-once stages.  Resume must revalidate
   scientific inputs and preserved bytes.  The two authoritative timing runs
   must be fresh and report `resumed_pairs=0`.

Scorer/Torch/cache/source imports are forbidden in the decoder module.
The full exact proof count is `707,788,800` numerator values and the assembled
raw output is exactly `3,662,409,600` bytes.  Build the resize operator once per
invocation.  The production parser itself performs predictor validation and
therefore incurs some decompression inside `parse_seconds`; report that overlap
honestly while retaining a separate explicit expansion stage.

## Required timing schema

Each single-inflate invocation emits a write-once canonical JSON receipt outside
the deterministic decoded-output tree.  It records monotonic seconds for these
six nonempty named components:

- `parse_seconds`;
- `expansion_seconds`;
- `solve0_seconds`;
- `solve1_seconds`;
- `assembly_io_seconds`;
- `verification_seconds`.

Also record `component_sum_seconds`, receiver `total_seconds`, outer-process
wall seconds, any explicitly named unclassified overhead, per-pair solve0/solve1
seconds for later C9 intervention pricing, pair/chunk counts, host/platform,
thread/process environment and worker count, command argv, source/input hashes,
archive/packet hashes,
Y0/Y1 aggregate and per-chunk digests, raw/stage/chunk tree digests, numerator
counts, storage preflight, and authority booleans fixed false.  Total accounting
must be internally consistent; missing, negative, nonfinite, or zero required
components refuse.  These are `[macOS-CPU local timing]` measurements only and
carry no contest-budget verdict by themselves.

## Preparation and composition tool

The tool has three explicit phases/subcommands so the two timing rows are two
real process invocations:

1. `prepare`: validate an SSD storage plan, materialize Y0/Y1 in canonical
   12-pair resumable chunks, build independently parseable predictor chunks,
   combine those preserved records into one full-n600 payload/archive without
   rerunning prior chunk compression, parse it back, and emit source/chunk
   custody.  Partial chunks are not authoritative; completed chunks are
   write-once and hash-bound.
2. `inflate`: perform exactly one timed receiver invocation and write one
   invocation receipt.  It requires an explicit worker count, and C1 runs one
   fresh serial baseline plus two fresh four-worker exploited invocations.  A
   fresh authoritative call refuses an existing output; `--resume` is explicit
   and never qualifies as one of the two exploited timing rows.
3. `compose`: require two fresh full-n600 invocation receipts for the identical
   archive; require raw bytes, deterministic stage manifests, and chunk
   manifests byte-identical; run an encode-side native-f32
   frozen SegNet + PoseNet hard-oracle spot check on at least pairs
   `90,175,277,381,424,573`; preserve per-pair `d_seg`, `d_pose`, Seg mismatch
   counts and Pose6 results; cite
   `f32_receiver_arithmetic_exactness_admissibility_v1`; and emit a compact
   content-addressed n600 receipt suitable for `.omx/research/`.  Separately
   ingest only real paired local/contest timing anchors, classify each as
   `inflate_only` or `full_official_evaluation`, and issue the mandated
   three-way timing verdict.  When calibration is absent or the result is
   close, emit—but do not dispatch—a full `evaluate.sh` Modal ticket binding
   the exact archive bytes, inflate entrypoint, expected CPU/T4 instance class,
   #381 resource envelope, single-flight key, and both dispatch/call-id ledgers.
4. `mlx-parity`: on an MLX-capable M5 Max host, run the integer-only local twin
   over at least the six frozen real pairs, compare Y0/X0 and Y1/X1 bytes and
   exact-numerator proofs pair by pair against the NumPy/CPU authority, and
   emit only `[macOS-MLX research-signal]` rows.  An unavailable MLX runtime is
   an honest host-custody result; it must not be represented as parity.

The hard oracle consumes both full decoded camera frames and the official
PoseNet 2x2/YUV6 path.  Rational equality is labeled DERIVED; decoder timings,
digests, numerator counts, and hard-oracle observations are MEASURED.  No
hard-oracle state enters the archive or decoder.

## Storage and hygiene

All bulk belongs under
`/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719/`.
Before materialization, run the canonical storage waterfall with at least
24 GiB requested and require the selected root to equal that exact path.  The
prepare state must bind the storage plan.  Retain the two decoded outputs,
stage manifests, archive, and source-plane/chunk custody through review.  Any
later cleanup is lossless/certified and must record original path, bytes,
SHA-256/tree hash, deterministic rebuild command/config/input hashes, and
reason; no operator-facing evidence may cite a temporary path.

## Behavioral tests

Tests must use small certified factor-2 fixtures and include:

- distinct two-plane happy path with both exact numerator counts;
- byte-equal and memory-aliased planes refuse;
- `repeat-frame1`, legacy/raw/Brotli policy, residual section, wrong pair IDs,
  wrong geometry/dtype, trailing bytes, and packet re-encode drift refuse;
- no implementation branch assigns/copies one solved frame to the other;
- every required timing field exists and is finite/nonnegative/nonzero on a
  real smoke;
- stop/resume preserves prior pair bytes and refuses edited stage or manifest;
- deterministic second fresh decode has identical raw/stage/chunk digests;
- four-worker completion order cannot change fixed-order raw/stage/chunk bytes,
  and serial versus four-worker outputs are byte-identical;
- compose refuses resumed timing rows, missing component timing, archive
  mismatch, output mismatch, fewer than six hard-oracle pairs, an invented
  local-to-contest margin, or a bare `local<1800` timing verdict;
- decoder source contains no Torch, SegNet, PoseNet, DistortionNet, source
  cache, or scorer import path.
- MLX twin parity is tested per plane/per pair, keeps all authority booleans
  false, is excluded from contest verdict inputs, and reports rather than hides
  an integer-op divergence.

Run focused tests, pycompile, Ruff/format checks, `git diff --check`, and an
actual small end-to-end CLI smoke.  Then the parent performs independent
adversarial review before serializer commit.

## What success does not mean

C1 can establish only local receiver viability and timing custody.  Even a
structurally clean local result does not prove rate viability, native-f32 exact
score, contest-axis runtime, promotion, or pointer movement.  The honest memo
must report the serial attribution total plus both MEASURED four-worker local
totals without calling any of them a 1,800-second PASS, classify the
3.775-minute `repeat-frame1` datum as inflate-only, state whether two distinct
solves remain in the exploited local runtime class, and issue the mandated
calibrated three-way timing verdict only for the exploited path.  MLX rows
remain research-signal only.  Only a full official contest-hardware evaluation
can measure `T_inflate + T_scoring < 1800 s`.

Stores consulted: `CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`;
`SPEC_v10_integer_plane_vehicle_20260719.md`; Task #541 receiver/predictor
receipt and code; Task #543 production receiver receipt and code; the canonical
f32 receiver-arithmetic law; current pointer/lane/ownership/checkpoint state;
and both delegated inboxes.
