# M1 deep decomposed checkpoint facet observer build spec

## Objective

Build `tools/observe_m1_banded_checkpoints.py`, a detached read-only observer for
the live M1 checkpoint stream.  For every previously unseen
`*__ipe_stage*_ep*_step*.json` checkpoint, it must decode the EMA state through
the real `integer_plane_emitter` implementation, independently parse/apply the
exported R1b4 packet through the production receiver primitives, require exact
agreement, realize the fixed n24 sample to camera uint8, and score it with the
frozen upstream CPU `DistortionNet` at batch 16.

## Fixed authority and inputs

- Axis: `[macOS-CPU advisory]`; `score_claim=false`,
  `subsample_advisory=true`; this never supplies an n600 verdict.
- Pair sample: 24 unique indices selected by NumPy `PCG64(1234)` without
  replacement from `[0,600)`, persisted with its SHA-256 in every row.
- Live checkpoint directory: the live materializer output directory supplied
  by CLI; glob only `*__ipe_stage*_ep*_step*.json` and never write there.
- Band manifest SHA-256 must equal
  `2fd10841dc0cb344454e4af55bd8d27e5e1d819a97df3fc03307604dfffcc367`.
- R1b4 binding is loaded via `C2R1B4CurveletBinding.load`.
- GT labels, poses, and native camera frames are mapped from the sealed n600 GT
  cache without loading the 5 GB NPZ into memory.
- Base camera pairs are snapshotted once into the observer's own SSD directory
  from a complete, stable live base raw scratch file before certified trainer
  cleanup.  Snapshot only n24 and persist source/sample custody.  If that
  source is gone, allow a pre-existing valid n24 cache; otherwise wait without
  fabricating pose telemetry.

## Required row fields

Each canonical JSONL row includes checkpoint path/name/SHA and
stage/epoch/global-step; fixed pair IDs and seed; git and scorer/band/binding/
base/GT provenance; authority labels; overall d_seg; per-class d_seg in the
canonical order `Road`, `Lane`, `Undriv`, `Movable`, `MyCar` with counts,
conditional error, and total-d_seg contribution; frozen PoseNet d_pose; changed
pixel residency split inside the realizable band versus outside it; d_seg
contribution for realizable-band, structurally dead candidate, and outside-
candidate pixels; and a clearly labelled zlib level-9 compressed-byte estimate
for the **live** float32 `pair_plane_codes` (EMA is used for decoding).

Every row must also record emitter/R1b4 equality, exact factor-2 receiver proof,
batch size 16, n24 sample size, `score_claim=false`,
`subsample_advisory=true`, and an explicit verdict scope.

## Memory, durability, and polling

- Peak observer footprint must remain below 6 GiB.  Operate on at most 16
  sampled pairs at a time, release tensors/arrays between batches and
  checkpoints, use CPU Torch with one thread, and never retain full-n600 dense
  arrays in RAM.
- Output is append-only canonical JSONL at the requested SSD path.  On restart,
  read existing valid rows and skip already processed checkpoint SHA-256s.
- Existing checkpoints are processed in stage/epoch/step order before polling
  every 120 seconds.  Incomplete checkpoint writes are retried; malformed or
  custody-invalid checkpoints fail closed and are written to a small observer
  error log, never converted into facet rows.
- The observer does not touch the live PID, trainer, run files, pointer, or
  evaluator.  It creates files only below its own output directory.

## Tests and acceptance

Add focused tests for deterministic pair selection, strict checkpoint envelope
parsing, restart deduplication, per-class/stratum accounting, code-byte
estimation, and fail-closed band/binding custody.  Tests may use tiny pure
fixtures/stubs but production execution must call the real emitter, R1b4 packet
parser/application, factor-2 realization, and frozen scorer.

Commands that must pass:

```text
/Users/adpena/Projects/pact/.venv/bin/python3 -m py_compile tools/observe_m1_banded_checkpoints.py
/Users/adpena/Projects/pact/.venv/bin/python3 -m ruff check tools/observe_m1_banded_checkpoints.py src/tac/boundary_math/tests/test_observe_m1_banded_checkpoints.py
PYTHONPATH=src:. /Users/adpena/Projects/pact/.venv/bin/python3 -m pytest -q src/tac/boundary_math/tests/test_observe_m1_banded_checkpoints.py
```

After two clean review-tracker passes for the Python files, commit the tool,
tests, dated memo, and DAG feed without a co-author trailer.  Launch through
`tools/launch_detached_process.py` at low priority, verify the detached PID and
first real row(s), and leave MAIN a mandatory full-diff landing review.

## Binding scope amendment — 2026-07-21T01:20:52Z

`SCOPE_AMENDMENT_ACK`: the operator inbox directive at the timestamp above
supersedes the random-n24 cohort in the original mission and the earlier sample
section of this spec.

### Bootstrap and recurring cohort

- Preserve a read-only, hash-bound camera-base snapshot while the trainer's
  ephemeral complete raw exists. A full n600 snapshot is permitted because its
  3,662,409,600 raw bytes plus observer products stay below the 6 GiB footprint
  ceiling. Treat it as certified rebuildable scratch and automatically remove
  it only after the one-time rank and recurring-cohort snapshot are durable.
- At the first processable checkpoint, stream all 600 pairs in batches no larger
  than 16 through the parsed R1b4 receiver and frozen CPU scorers. Emit exactly
  one canonical row per pair to `facets_perpair_rank.jsonl`, including pair ID,
  d_seg, d_pose, checkpoint SHA, receiver/emitter parity, and advisory labels.
- Rank by d_seg descending with pair ID as the deterministic tie break. Freeze
  the recurring cohort as the top 32 hardest pairs plus 16 unique background
  pairs drawn by PCG64 seed 1234 from the complement. Persist the ordered 48 IDs
  and hash. The recurring cohort never changes after bootstrap.
- If and only if full n600 cannot be preserved within the 6 GiB preflight, use a
  seeded n128 bootstrap population and label that fallback explicitly. Do not
  silently reduce coverage.

### Every checkpoint row

- Write the mandated main stream as `facets.jsonl`.
- In addition to aggregate and per-class metrics, include per-pair d_seg/d_pose
  for the recurring cohort and the top eight worst pairs under each component.
  A mean is never the sole pose report.
- Execute the unquantized NumPy emitter and the parsed R1b4 receiver. Score the
  receiver. Record exact equality, differing uint8-value count, and maximum
  absolute uint8 difference; quantization mismatch is measured telemetry, not a
  reason to discard a receiver-valid row. Packet parse/re-encode and factor-2
  receiver proofs remain fail closed.
- Compute out-of-band excursion against the hash-bound source/radius planes over
  realizable candidate pixels. Report total fraction and rows keyed by canonical
  GT-class to emitted-class pair, with pixel and excursion counts. Preserve the
  structurally-dead candidate d_seg contribution separately.
- Compute temporal argmax instability between the two consecutive frames inside
  every sampled pair, plus aggregate and worst-pair tails; name that definition
  in the row so it cannot be confused with non-adjacent pair-index comparisons.

### Stage panels

- `SCOPE_AMENDMENT_ACK` also covers operator amendments received at
  `2026-07-21T01:24:41Z`, `2026-07-21T01:25:35Z`, and
  `2026-07-21T01:26:01Z`. They supersede the underived hardest-six and
  fixed-mass-cut panel rules below while preserving the read-only/advisory
  boundary.
- Derive the visual cohort from the complete bootstrap population. Build a
  mechanism signature per pair from class-flip composition,
  boundary-versus-bulk flip fraction, and an explicitly defined temporal
  flicker flag. Fit deterministic clusters for every k in `[2,20]` that the
  population supports and choose k by the reported BIC curve; no hand-picked
  cluster count is admitted.
- Allocate depth by Neyman allocation, proportional to cluster d_seg mass times
  within-cluster d_seg standard deviation, with at least one exemplar from
  every selected cluster. Rank exemplars deterministically inside each cluster.
  Stop through a Good-Turing unseen-mechanism-mass rule. Consume
  `realization_breakeven_bytes_v1` by ID, invoking its registered callable to
  derive (not retype) the exact non-rate score floor corresponding to the panel
  fix-payload comparator. Persist equation/callable/anchor custody in the
  derivation row.
- Emit the complete panel-budget derivation in every required boundary/final
  facet row: k and BIC curve, per-cluster mass/std/Neyman depth, Good-Turing
  unseen-mass estimate, derived total, envelope cap, and emitted total. Freeze
  an observer-owned base snapshot for the derived panel IDs before deleting the
  full bootstrap scratch so later stage boundaries remain reproducible. If the
  derived total exceeds the byte envelope, cap deterministically and report
  both totals; never silently drop pairs.
- On each distinct stage-boundary checkpoint (`stage_complete=true`) and the
  terminal/final checkpoint, render a full multipane panel for every admitted
  derived exemplar. Also render a cluster-overview contact sheet with one row
  per cluster and exemplars in rank order.
- Store panels only under the observer evidence directory's `panels/` subtree.
  Bind each panel and contact sheet to checkpoint SHA and pair ID/cluster in row
  provenance. Use only lossless PNG. Scorer-plane maps are direct native
  `512x384` arrays at one array pixel per image pixel; indexed class maps use
  nearest-neighbor only. No lossy codec, fractional resampling, antialiasing,
  matplotlib, or GUI is allowed.
- Preserve every signed RGB plane delta as an exact raw `.npy` sidecar beside
  its panel. Reload one written panel/map and assert the persisted argmax-map
  pixels are bit-exact. Contact sheets may use only integer-factor nearest
  downscaling or native-resolution crops, and must record the transform.

### Restart and cleanup

- Existing canonical `facets.jsonl`, `facets_perpair_rank.jsonl`, cohort receipt,
  base snapshot receipts, and panel names are the resume state. Never duplicate
  checkpoint or pair-rank rows.
- Complete/stable source disappearance is retryable while bootstrapping. Never
  hash the live `.partial` decoder file. The observer must remain alive while it
  waits for the final raw and first checkpoint.
- If the materializer has already success-cleaned the final camera raw, accept
  its retained stable `base_scorer_planes.npy` only after exact shape/dtype/hash
  validation against the adjacent materialization receipt. Deterministically
  factor-2 realize both planes into observer-owned camera bases in batches no
  larger than 16, verify each scorer-plane round trip exactly, and record this
  fallback source/proof. Prefer the complete final camera raw when both exist.
- Delete only the observer-owned full n600 scratch after a hash-bound recurring
  48-pair snapshot and complete rank table exist. Record original path, bytes,
  SHA-256, source custody, derived artifacts, command/config, and reason before
  unlinking. No user/run artifact is ever removed.

## Do not touch

- The live trainer process or PID.
- Any path inside the live run except read-only checkpoint/base-scratch reads.
- The trainer implementation, canonical frontier pointer, score ledgers, or
  exact evaluator.
- Main or any sibling worktree.
