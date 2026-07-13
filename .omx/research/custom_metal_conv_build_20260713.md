# Custom Metal frozen-SegNet convolution build — 2026-07-13

`research_only=true` · `$0 local` · `score_claim=false` · `pointer_moved=false` ·
`training_launched=false` · lane `lane_custom_metal_conv_20260713`

## Outcome first

**The source build is complete, but device execution is BLOCKED-NOT-MEASURED in this process.**
The pointwise speedup versus MLX at real shapes is **NOT MEASURED**; the depthwise fp16 bandwidth
thesis is **UNADJUDICATED**; Metal parity and real-frame argmax flips are **NOT MEASURED**; a
measured composed-forward Amdahl estimate is therefore **REFUSED**; and fp16/int8/int4 device
timing/fidelity are **NOT MEASURED**. The kernel sources, deterministic NumPy-fp32 references,
weight packers, default-OFF adapter, real-shape inventory, tests, and resumable benchmark command
are built. Their honest status is `BUILT-NEVER-FIRED`, not a performance verdict.

The exact local blocker is:

```text
MLX default device labels GPU, but an evaluated allocation raises:
[metal::load_device] No Metal device available
```

The independently owned precision-backend sibling first reached the same blocker, then obtained a
Metal-visible main-local receipt after our benchmark attempt. That receipt measures only the
EfficientNet-B2 3x3 stride-2 **stem**, not any 1x1 pointwise geometry: at B=1 its W4/W6/W8/W8A8
im2col paths are respectively `1.3478409x`, `1.1031949x`, `1.1107175x`, and `1.0068950x` versus
native fp32; at B=8 they are `0.5479686x`, `0.5512115x`, `0.5533905x`, and `0.4160392x` (all
losses). The n16 full-logit cosine/flip rows are also preserved in that sibling receipt. This is
useful lowering evidence but does **not** answer custom-pointwise versus im2col at matched 1x1 real
shapes, so there is still no direct winner. An independent offline compile here was also
unavailable: `xcrun metal --version` returned 1 because this Xcode installation lacks the optional
Metal Toolchain. Therefore our source is built but its device compilation remains unverified.

`verdict_scope: source construction, NumPy-fp32 reference behavior, frozen-weight packet
determinism, and current-process Metal accessibility for the frozen-SegNet local forward
throughput formulation only; no negative transfer to a Metal-visible process, other hardware,
backward/VJP, full training, contest CPU/CUDA, evaluator, d_seg, d_pose, archive, score, or
promotion`

## What was built

`src/tac/local_acceleration/metal_segnet_conv.py` adds a standalone member of the #212/#260/#443
custom-kernel suite:

- NHWC pointwise 1x1 implicit GEMM with 8x8x8 tiles, `simdgroup_half8x8` inputs and
  `simdgroup_float8x8` accumulation; fixed traversal and no atomics;
- frozen-weight variants `fp16`, symmetric per-output-channel `int8`, and packed symmetric
  per-output-channel `int4`; int8/int4 dequantize on load into half threadgroup tiles;
- NHWC depthwise 3x3/5x5 for stride 1/2 with fp16 traffic and sequential fp32 accumulation;
- fixed-order NumPy-fp32 pointwise and depthwise references plus deterministic int8/int4 packet
  producers;
- a forward-only frozen-Conv2d adapter and a scoped converter patch that restores the shared
  converter under a lock in `finally`;
- default-OFF enablement through `TAC_MLX_CUSTOM_SEGNET_CONV=1`, with pointwise storage selected by
  `TAC_MLX_CUSTOM_SEGNET_POINTWISE_WEIGHT={fp16,int8,int4}`;
- a hard fail when the opt-in is requested without an evaluated Metal device; and
- explicit `vjp=not-implemented-forward-only-fail-closed`. A deterministic training VJP is still
  `NEEDS-BUILD`; this forward source does not authorize a training launch.

`experiments/bench_custom_metal_segnet_conv.py` derives the convolution inventory from the pinned
frozen model, uses real receiver bytes, benchmarks each unique real pointwise/depthwise geometry
against native MLX when Metal exists, compares full logits/argmax, and composes a time-domain
Amdahl substitution from measured isolated native/custom times against the directly measured
native full-forward wall. It refuses that estimate if isolated native times exceed the direct wall;
the MAC-share calculation remains explicitly structural. The harness writes a fail-closed receipt
on preflight or kernel failure and writes no timing row when Metal is absent.

## Frozen-model geometry

The harness re-derived 125 full-SegNet Conv2d calls at batch 1 from the pinned model:

| scope | calls | pointwise | depthwise | MLX-native remainder | MACs |
|---|---:|---:|---:|---:|---:|
| EfficientNet-B2 encoder | 114 | 90 calls / 89.5365703373% | 23 / 8.7493451277% | 1 / 1.7140845350% | 2,477,551,552 |
| full SegNet | 125 | 90 / 22.3861129188% | 23 / 2.1875288193% | 12 / 75.4263582619% | 9,909,333,952 |

The full per-call inventory, including B/H/W/C, kernel, stride, padding, scope, and MACs, is in the
receipt. The large decoder/head 3x3 dense convolutions intentionally remain MLX-native in this arm.

## Evidence that exists, and evidence that does not

### MEASURED locally without Metal

- tests: `23 passed, 5 skipped`; all five skips are device-execution tests under the recorded Metal
  blocker;
- ruff: clean on the module, tests, and benchmark harness;
- fixed-order pointwise NumPy-fp32 versus NumPy matmul: max abs
  `9.5367431640625e-07`, mean abs `6.113296979748384e-09` on the seeded static probe;
- fixed-order pointwise and depthwise reference repeat equality: `true`;
- deterministic int8 packet max frozen-weight reconstruction error: `0.009281158447265625`;
- deterministic int4 packet max frozen-weight reconstruction error: `0.1572265625`.

Those packet errors are a seeded format smoke, not frozen-SegNet fidelity. Metal source compilation
itself is also unverified because the device compiler could not be reached.

### BLOCKED-NOT-MEASURED

| requested result | receipt value | honest conclusion |
|---|---|---|
| pointwise fp16 speedup vs MLX at real shapes | 0 timing rows | NOT MEASURED |
| pointwise int8/int4 speedup and parity | 0 timing/parity rows | NOT MEASURED |
| depthwise fp16 speedup | 0 timing rows | bandwidth thesis UNADJUDICATED |
| Metal max-abs / real-frame argmax flips | null | NOT MEASURED |
| direct full-forward timing | null | NOT MEASURED |
| composed measured Amdahl | null | REFUSED-INCOMPLETE-MEASUREMENTS |
| sibling im2col comparison | stem-only timing/fidelity measured; no matched 1x1 row | NO DIRECT WINNER |

## Structural ceiling, not a measurement

If both implemented families became free while all MLX-native remainder stayed fixed, the
MAC-share-only ceiling would be

`1 / 0.7542635826186354 = 1.3257964762506795x`.

Using the operator-provided teacher-forward share of 0.78, the corresponding idealized training-wall
ceiling is

`1 / (0.22 + 0.78 / 1.3257964762506795) = 1.23712524615734x`.

Both are **DERIVED structural upper bounds**, not latency models and not measured Amdahl claims.
Memory traffic, launch overhead, fusion loss, and the MLX-native decoder can make reality materially
worse. The benchmark will replace these bounds with weighted measured rows plus a direct full-forward
check when it can run.

## Promotion gate

The named gate is `custom-metal-segnet-conv-n600-fidelity-gate`. It remains
`NEEDS-MEASUREMENT` and must fail closed unless all of the following are present on one fingerprinted
Metal runtime:

1. every unique real pointwise and depthwise geometry has native/custom warm timing with a positive
   matched speed sign; no losing geometry may be averaged away;
2. the full frozen teacher has matched n600 receiver-realized timing and the lower 95% paired
   speedup bound exceeds 1.0;
3. n600 full-logit max/mean absolute drift and final argmax flip count are reported for each weight
   variant; flips are training-tolerance evidence only, never score/verdict authority;
4. after a deterministic VJP is built, global and worst-pair input-gradient cosine are each at
   least `0.99`, matching the repository's precision-training bars, with all 600 pair rows present;
5. two fresh-process replays reproduce packet hashes, ordered argmax hashes, and per-stage timing
   custody; and
6. the candidate remains default OFF until the governed launcher consumes the gate receipt.

An exact argmax requirement is deliberately not substituted for gradient fidelity: Task #456 showed
that stable razor-tie forward flips can coexist with a large training-throughput prize, while also
showing why this regime cannot carry evaluator or score authority.

## Exact continuation

```bash
.venv/bin/python experiments/bench_custom_metal_segnet_conv.py \
  --seed 0 --real-pairs 4 --warmup 5 --repeats 30 \
  --out experiments/results/custom_metal_conv_20260713/receipt.json
```

The command is deterministic and idempotently replaces the small receipt. A Metal-visible rerun
must first clear the four-pair build/timing smoke; the n600 gate is a separate stage and must preserve
its own stage receipt. No training, paid dispatch, or long job was launched.

## Custody and disk hygiene

- git HEAD at receipt creation: `41f1ff008f642975123579d4b6c8c7cc43012b29` on `main`, dirty
  shared worktree recorded by status hash;
- seed `0`; MLX `0.31.2`; Python `3.13.12`; macOS arm64;
- weights: `upstream/models/segnet.safetensors`, 38,502,892 bytes, SHA-256
  `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`;
- receiver raw: `experiments/results/levelset_packet_20260708T221700Z/inflated/0.raw`,
  3,662,409,600 bytes, SHA-256
  `3819479cf6afc44b0366b01a1f1babfd25cd8fcc180825a24097e10b10d98975`;
- receipt: `experiments/results/custom_metal_conv_20260713/receipt.json`, SHA-256
  `0bbf0a3de97611f952d937a49799170baff41da71cae9f7783470d09bb0b16d1`;
- kernel module SHA-256:
  `0c967ec29c41810c9be936fa8869baa53b4737f0ccf53d81db8e17ce8930689e`;
- test module SHA-256:
  `23aa9fda884da02a37b66151cb41ea1e22e8f3511d6867bbeb7531c8aee2425c`;
- benchmark harness SHA-256:
  `20beee17678f41ba4d283ac4c121c98e30522fa7396c5748c38a74806daf66b9`.

The existing raw is read in place. This build creates only source, tests, Markdown, and a small JSON
receipt; it creates no bulky rebuildable artifact, so no move or deletion is authorized or needed.

## Triality and system wiring

- DSL: no score/trainer lever is activated. The two typed environment selectors are default OFF;
  a governed typed training policy is owed after VJP and n600 admission.
- DAG: `.omx/research/custom_metal_conv_DAG_FEED_20260713.md` records the source-built/device-blocked
  state and all refused downstream claims.
- Equations: the only current equation is the explicitly DERIVED structural ceiling above; no
  empirical latency law was registered from missing measurements.
- Sensitivity/Pareto: the n600 gate consumes full-logit, argmax, and input-gradient sensitivity
  against paired latency. Compute-only work has no archive-bit allocation hook.
- Autopilot: candidate pool status is `built-never-fired`; dispatch must refuse without Metal timing,
  deterministic VJP, and the n600 gate.
- Continual learning: this memo and the fail-closed receipt preserve the environment-scoped blocker;
  they do not turn it into a family negative.
- Probe disambiguation: the sibling stem receipt establishes strong batch dependence but does not
  share a pointwise geometry. Custom pointwise versus im2col, and custom depthwise versus native MLX,
  are decided only by matched real-shape measurements when this build is Metal-visible.

The canonical frontier pointer is unchanged.

## STORES CONSULTED

Full `CLAUDE.md`; full `AGENTS.md`; full `docs/operating_manual_craft_handoff.md`; `PROGRAM.md`;
v7.5 §8 and v8 specifications; top Claude memory entries; latest Codex findings/session summary;
latest council/design/directive memos; canonical frontier, lane, task, and subagent state; prior
#212/#260/#356/#435/#443 Metal kernel modules, tests, and memos; Task #456 terminal static-forward
memo; frozen SegNet converter/model source and pinned weights; real receiver bytes; sibling
`precision_backend_matrix_20260713.md`, its original blocked receipt, and the later main-local
MEASURED stem receipt (SHA-256
`123f3a928f6341f229351bed0c10e3ab37cdb9b2351185709f734bb41c17486d`). Paid/cloud/provider state,
protected live runs, contest CPU/CUDA, evaluator, and training were not actuated.
