# WD3 scorer-aware width-distillation implementation specification

## Objective

Land the build-only apparatus required by the sealed WD3 charter.  The apparatus must be capable of
materializing the shared teacher/scorer cache, training the WD2 student family through the real paired
receiver and both frozen scorers, exporting an adaptively quantized receiver-consumed student packet,
retaining every payload, and compiling a fail-closed fire order.  This arm must not invoke a scorer,
start Metal work, launch training, or move the frontier.

The governing sources are:

- `.omx/research/charters/ddm_wd3_scorer_aware_width_distillation_20260815.md`
- `.omx/research/ddm_wd3_scorer_aware_width_distillation_20260815.json`
- `.omx/tmp/codex_runs/_common_contract.md`
- the existing WD2 builder/receiver, which remain unchanged

## Files and ownership

Create only these Python files:

- `experiments/ddm_wd3_student_receiver.py`
- `experiments/ddm_wd3_scorer_aware_width_distillation.py`
- `experiments/tests/test_ddm_wd3_scorer_aware_width_distillation.py`

Do not edit any existing file.  In particular do not touch `src/tac/**`, `upstream/**`, the WD2
builder/receiver, protected common-contract files, or lane ledgers.  Land only the named WD3 files and
their receipts through the governed serializer, without staging unrelated dirty work.

## Pinned real inputs

The code must pin and verify the bytes and SHA-256 identities from the charter/design for the base
receipt, teacher master cache, WD2 ep60 checkpoint, WD2 train receipt, WD2 archive, GT cache, scorer
weights, and fixed frame-0 raw stream.  The fixed frame-0 source is the retained WD2 advisory raw:

`/Volumes/APDataStore/pact/ddm_wd2_width_distillation/primary_flattened_d4_w64/retained/candidates/flattened_d4_w64_epoch_0060/attempt_0000/advisory_n600_cpu/work/inflated/0.raw`

Its manifest pins 3,662,409,600 bytes, SHA-256
`7a065f110f0b8202f098cec9dc2267d6be7e99a179c911e404226d6a289f2c56`.  Only the even frames are
the fixed carrier; the student/teacher frame 1 comes from the semantic renderer.  The original-target
cache is `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`, 5,078,017,610 bytes, SHA-256
`cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`, with `lstars` int64
`[600,384,512]` and `gt_poses` float64 `[600,6]`.  Scorer hashes are PoseNet
`0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576` and SegNet
`68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`.

## Receiver and packet

`ddm_wd3_student_receiver.py` may reuse WD2 topology and archive patch helpers, but it must implement a
real new packet format, not relabel WD2 bytes.  Required behavior:

1. A typed adaptive allocation maps every quantized tensor group to a signed bit depth in 2..8.
   Embeddings allocate per column, matrix/conv weights per output row, and other learned vectors remain
   fp16.  The default uniform-int4 allocation is a valid degenerate rung.
2. Fake quantization and packing use the same per-group scale, clipping, signed code range, and bit
   allocation.  Forward fake quantization is STE; packed parse-back values equal the forward grid.
3. The counted packet includes magic/version, canonical student spec, a canonical allocation/selection
   map, scales, bit-packed signed codes, and all fp16 vectors.  Parse rejects truncation, trailing bytes,
   invalid bit depths, shape/spec drift, or noncanonical allocation.
4. `pack_student -> unpack_student -> pack_student` is byte-idempotent.  A one-bit/payload mutation in a
   learned code changes a realized frame or is rejected; inactive dispatch must remain byte-identical.
5. Runtime patching must make the exact retained submission parse and consume this WD3 packet.  Copy the
   new receiver into the retained `cpr1` runtime and add a narrowly dispatched WD3 branch without changing
   the old WD2/inactive branches.  Full-container parse-back must return the exact WD3 packet.
6. Expose exact serialized-byte accounting and allocation telemetry by tensor/group/bit depth.  Never
   claim a projected archive size as measured.

## Real paired scorer chain

The scorer-aware builder must import `load_differentiable_scorers` and actually call the scorers in the
cache/train paths.  A student batch is:

1. render at 384x512 with the adaptive packet quantizer in-loop;
2. bilinear resize frame 1 to 874x1164, clamp, and uint8 STE;
3. read the corresponding retained fixed frame 0 bytes from the exact raw memmap;
4. form `[B,2,3,874,1164]` in chronological order;
5. use the scorers' real preprocessing; PoseNet consumes first six outputs, SegNet consumes frame 1.

The teacher cache uses the retained teacher frame 1 and the same fixed frame 0.  It runs the exact frozen
scorers in chunks at most 120 and atomically retains all six charter fields in stable declared dtypes:
teacher Seg logits f16, argmax u8, top1-runner-up margin f16, teacher Pose6 f32, original target argmax
u8, and original target Pose6 f32.  Preserve immutable per-chunk checkpoints plus aggregate files and a
second independent repeat; finalization requires byte-identical payloads.  Receipts bind every file,
input/hash, scorer/hash, command, environment, shapes/dtypes, upstream snapshot, source archive and
aggregate SHA.  An incomplete cache cannot be consumed.

## Objective, cells, regions, selectivity

Implement the actual differentiable objective, returning named components and gradients:

- primary Seg term: mean `1 - softmax(student_logits)[original_target]`, multiplied by a stage-frozen
  calibration scale and exactly 100; calibration is hard d_seg divided by mean soft disagreement on the
  same fixed controller subset, and may only change at a stage boundary;
- impostor-complete constraint: for the teacher winner, hinge against every other student class using
  the teacher top1-runner-up margin;
- auxiliary `T^2 * KL(teacher/T || student/T)` with T=2;
- Pose term: `sqrt(10 * mean((student_pose6-original_pose6)^2))`, with teacher Pose6 deviation reported
  separately;
- decode trust region: uint8 MSE versus retained teacher master, ceiling 50.6728233448345;
- nonnegative margin/KL/decode/teacher-Pose duals start at zero and update only by positive constraint
  violation; teacher Pose matching is therefore adaptive rather than a guessed fixed-weight penalty.

No fixed `kd_w` and no linearized Pose coefficient may drive training/admission.  Loss telemetry must not
call a proxy `d_seg` or `d_pose`.

Implement deterministic cell/region treatment:

- derive, never top-k, a stage-frozen selective mask from actual teacher/student/GT argmax mismatch plus
  a one-cell codimension-1 boundary band; unchanged interiors are skipped;
- identify unordered class-pair edges and report per-edge hard flips, with Road-Lane explicit;
- report per-target-cell flips and per-pair counts;
- use pair/class flip burden to derive a deterministic adaptive quant allocation (frame-embedding rows and
  class-sensitive parameter groups first), retaining the selection/allocation maps; uniform int4 remains a
  comparison rung;
- validation n60 is fixed evenly strided; negative-confirmation n120 is fixed seeded stratified random and
  cannot be a prefix.  A negative cannot be emitted from n60.

Expose a typed, fail-closed surgical-finish handoff only after a retained candidate is near admission.  It
must name the real QS2/QS5 Schur-compensation producers and require their pinned receiver-consumed receipt,
exact candidate archive, residual Road-hub edge map, and pose/base custody.  It may not fabricate edits or
declare QS completion from configuration alone.

## Training and resumability

Support the sealed sequential arms: W0 preserved state, W0 fresh optimizer with a magnitude-matched ramp,
D56, F64, conditional W96, and fresh only after both W0 n120 failures.  Compile/execute order is enforced;
dense W96 is refused unless real coder evidence overrides the projection.  W0 must load the real WD2 ep60
checkpoint, including model/optimizer/EMA/cursor for preserved state.  The reset control retains weights
but creates a fresh optimizer and uses an explicit ramp bounded to remove the documented 3.16x--6.57x
zero-moment excursion.

Every run uses one recorded seed for Python/NumPy/Torch/sampling and deterministic algorithms.  A
scorer-free typed `prepare-arm-birth` action must persist a complete initial model/EMA/optimizer/scheduler/
RNG checkpoint for every non-W0 topology; W0 alone migrates the pinned WD2 checkpoint.  Register a
real WD3 controller with `ResumeRegistry`.  Checkpoints are atomic and contain live model, EMA, optimizer,
scheduler, scaler (or explicit disabled scaler state), RNG, generator, exact subset IDs and hashes,
selection/allocation maps, batch/epoch cursor, stage-frozen calibration, duals, exact compiled config, and
history.  Resume must reproduce the next batch/order.  Save distinct immutable stage-end checkpoints and
at most every five epochs.  Retain every evaluated render, scorer bundle, exact packet, archive,
repeat archive, runtime, parse-back receipt, and manifest.

## Typed compiler and CLI

The only heavy entrypoints are a canonical compiled JSON config, never invented free-form flags.  A small
CLI may expose `compile`, `prepare-arm-birth`, `prepare-teacher-scorer-cache`, `train`, `inventory`, and `verify-build`, but
cache/train accept only `--compiled-config` plus source SHA pins.  The compiler rejects before scorer/model
loading:

- unknown/invented fields;
- missing or drifted base receipt/cache/source pins;
- prefix subsets or wrong n60/n120 population rules;
- `retain_all_payloads != true`;
- chunk >120;
- absent resume checkpoint/state for every training arm;
- unclaimed or duplicate Metal/scorer lanes;
- arm-order/reset/fresh/W96 violations;
- checkpoint cadence >5;
- score-aware objectives not explicitly active and nonzero;
- output outside `/Volumes/APDataStore/pact/ddm_wd3_scorer_aware_width_distillation/`;
- any attempt to launch while charter authorization remains false.

Compilation without active lane claims must emit a machine-readable `BLOCKED_NOT_LAUNCHABLE` fire order,
not runnable argv.  A separate pure verifier accepts injectable file/lane facts so unit tests do not touch
the real scorer or bulk files.  Heavy functions call `assert_governed_admission` and revalidate pins.

## Acceptance tests

The test suite must use tiny deterministic dummy models/data and temporary files; it must not load frozen
scorers, hash multi-GB inputs, invoke Metal, or write outside pytest temp roots.  Behavioral tests cover:

1. adaptive packet exact pack/unpack/repack and allocation changes bytes/grid;
2. a packet code mutation changes a dummy realized output or is rejected;
3. paired receiver uses fixed frame 0 and student frame 1 in correct order with uint8 STE gradients;
4. both frozen dummy scorers participate and Pose gradients reach student frame 1;
5. primary soft disagreement, complete-impostor hinge, T=2 KL, nonlinear Pose, trust region, and one-sided
   dual updates match hand calculations;
6. cell/boundary selection is mismatch-derived, Road-Lane edge telemetry is correct, no top-k/prefix;
7. n60/n120 subsets are deterministic, cover the population, and n120 is stratified/nonprefix;
8. ResumeRegistry round-trip plus checkpoint reload reproduces selection/allocation/duals/cursor/RNG;
9. typed compiler rejects every G5 case and unknown fields before heavy loader callbacks run;
10. arm order, W0 preserved/reset rules, conditional W96/fresh rules, cache repeat finalization, retention,
    parse-back, and surgical-handoff blockers are fail-closed.

Run exactly:

```bash
.venv/bin/python -m pytest -q experiments/tests/test_ddm_wd3_scorer_aware_width_distillation.py
.venv/bin/python -m pytest -q experiments/tests/test_ddm_wd2_width_distillation_build.py experiments/tests/test_ddm_wd3_scorer_aware_width_distillation.py
.venv/bin/python -m ruff check experiments/ddm_wd3_student_receiver.py experiments/ddm_wd3_scorer_aware_width_distillation.py experiments/tests/test_ddm_wd3_scorer_aware_width_distillation.py
```

Do not run a cache build, training, scorer, full render, Metal task, or candidate evaluation.
