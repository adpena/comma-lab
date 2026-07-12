# Task 438 — V9 CGauge Torch/CUDA maximum-throughput landing (2026-07-12)

Status: **COMPLETE_1_TO_1; CUDA THROUGHPUT
UNMEASURED-pending-CUDA-dispatch**. Authority: `[macOS-CPU/Torch advisory]
NON-PROMOTABLE`. No provider was contacted and no paid resource was dispatched.
No `d_seg`, `d_pose`, score, archive, or frontier-pointer claim is made. The live
MLX arm was not read, written, signalled, or used for verification.

This landing supersedes the 2026-07-11 *current-state verdict* in
`cloud_launcher_cuda_port_438_20260711.md`; that file remains an immutable
historical receipt of the originally stripped port.

## Governing override and stores consulted

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`.
- `.omx/research/cloud_launcher_cuda_port_438_20260711.md` and the operator's
  `max_throughput_over_bit_identity_operator_override_20260712.md` directive.
- `experiments/train_levelset_witness_realized_through_R_mlx.py` as semantic
  authority; `src/tac/witness_dsl/spec_v9_cgauge.py` as sole scientific-config
  owner. No trainer-only scientific flag was added.
- Canonical lane/task/subagent ledgers and the existing Task 438 lane.

The operator waived bit identity and deterministic floating-point ordering for
the **training loop only**. `compile_identity_probe` / `backend_fp_reorder_probe`
is therefore advisory for bit identity and adoption-gating for functional parity:
`argmax_equal == true && cosine_phi >= 0.9997`. Byte-closed archive evaluation
through `upstream/evaluate.py` remains the only score authority.

## Eleven score-affecting surfaces closed

The machine-readable `cuda_v9_port_coverage.v1` receipt now has status
`COMPLETE_1_TO_1`, an empty blocker list, and a named primitive for each twin.

1. Generated/table pose carrier frame-0 dispatch and learnable `dxi`: typed
   `TorchPoseCarrier`, pose parameters in optimizer/checkpoint state, and real
   motion-stratified `J_xi` probes.
2. Structured scorer-SDF prefit: `structured_sdf_prefit`, fresh-start-only
   actuation, typed target/method/step contract, and resume suppression.
3. `accum-pairs=8`: vectorized fixed-shape chunks, one chunk-atomic optimizer
   decision, and accepted-fraction accounting over actual optimizer updates.
4. Curriculum sensors: scorer-derived island/dseg/pose/sigma observations,
   event latches, and atomic controller resume state.
5. Ladder island homotopy: eased supports, refreshed per-class lambdas/rungs,
   and static-address tensor updates compatible with compiled regions.
6. Seed/birth stack: protected training-only island seed with its own AdamW,
   containment and resume state, plus classwise birth-completion ramps driven by
   witness formation/persistence rather than epoch-only emulation.
7. AdamW-to-Muon: event-driven parameter split, native Torch Muon actuation,
   inherited-moment warm start, rewarmup, and optimizer state persistence.
8. Pose finish: `sigma_min_plateau` conditioning gate using real scorer Jacobian
   probes and the degenerate banked-R1 path.
9. Polyak finisher: GPU-resident tail average, resumable state, and an additional
   candidate export rather than overwriting EMA/live candidates.
10. Update geometry: typed dseg-aware taper and per-group gradient clipping before
    optimizer actuation.
11. Governed tail cycles: controller-owned tau/LR overrides, per-cycle preserved
    checkpoints, powerplay/stop latches, and exact resumed stop state.

## Throughput stack landed

- BF16 CUDA autocast when supported, fp16 plus `GradScaler` fallback, TF32 matmul,
  cuDNN benchmarking, and nondeterministic kernels under the explicit waiver.
- `torch.compile(mode="max-autotune")` with fixed shapes and Inductor/Triton
  regional fusion for FiLM/HOSC fields, contest R, frozen SegNet/PoseNet, and the
  score-bearing level-set loss regions. AOTAutograd owns their backward graphs.
- CUDA Graph Trees requested through Inductor, with explicit step boundaries;
  the typed AdamW/Muon optimizer transition remains outside graph capture.
- Device-resident targets, margins, pose targets, and pre-resized chroma frames;
  vectorized lane composition removes a former CUDA-to-CPU synchronization.
- Controller counters transfer once per epoch, loss telemetry reduces only on the
  epoch-final chunk, and Polyak accumulation remains on device until checkpoint.
- Epoch receipts record synchronized seconds, pairs/s, optimizer updates/s, peak
  allocated bytes, AMP dtype, compiled-region adoption, graph request, and warmup
  status. Every row is `promotion_eligible=false` and `score_claim=false`.

This is a compiler-generated **regional Triton megakernel stack**, not a claim
that one hand-written monolithic forward+R+scorer+backward kernel exists. A custom
kernel would need CUDA profiling evidence to justify replacing Inductor output.

## Functional-parity and local verification evidence

Final command:

```text
.venv/bin/python experiments/train_levelset_witness_realized_through_R_torch.py \
  --verify-only --compile-probe --device cpu --num-pairs 8 \
  --out-dir experiments/results/codex_v9_cuda_verify_20260712
```

Measured result: RGB max absolute delta `3.0517578125e-05`; phi max absolute
delta `4.172325134277344e-07`; `argmax_equal=true`; `cosine_phi=1.0`; compiled
loss max delta `0.0`; compiled gradient max delta `0.0029296875`; functional
adoption `true`; coverage `COMPLETE_1_TO_1`. The output directory was not
created by verify-only. Final focused suite: **73 passed**; `py_compile` and
`git diff --check` passed. A non-failing headless Metal atexit warning is outside
the Torch/CUDA path.

## Throughput-design ledger

| Optimization | Dispatch/data-motion target | Expected win class | CUDA evidence |
|---|---|---|---|
| Vectorized `accum-pairs=8` | amortize Python and scorer launches | launch-bound | UNMEASURED-pending-CUDA-dispatch |
| Device-resident/pre-resized targets | eliminate repeated H2D/resize work | transfer + bandwidth | UNMEASURED-pending-CUDA-dispatch |
| BF16/fp16 autocast + TF32 | Tensor Core use and lower activation traffic | compute + bandwidth | UNMEASURED-pending-CUDA-dispatch |
| Inductor max-autotune FiLM/HOSC | fuse pointwise/linear activation regions | launch + compute | UNMEASURED-pending-CUDA-dispatch |
| Compiled R/scorers/losses/backward | fuse surrounding ops and AOT backward | launch + memory traffic | UNMEASURED-pending-CUDA-dispatch |
| CUDA Graph Trees | replay steady fixed-shape regions | CPU launch overhead | UNMEASURED-pending-CUDA-dispatch |
| Epoch-only telemetry transfers | remove per-chunk synchronization | synchronization | UNMEASURED-pending-CUDA-dispatch |
| GPU Polyak accumulation | avoid recurrent device-to-host copies | transfer | UNMEASURED-pending-CUDA-dispatch |

No expected-win entry is a speedup estimate. macOS cannot measure CUDA, and the
operator did not authorize a paid dispatch in this task.

## Exact first authorized CUDA measurement plan

1. Under a separate dispatch authorization, stage the exact GT cache with its
   local SHA-256 and run the existing provider plan on an H100 first (A100 is the
   fallback). Keep the typed V9 config unchanged. Do not use a short timing run
   as scientific evidence.
2. Run a fixed-shape `num_pairs=600`, at least three-epoch timing smoke so epoch
   one absorbs compilation/autotune. Preserve every per-stage/intra-stage
   checkpoint and the provider/call-ID custody receipts. The launcher remains
   plan-only unless separately given `--execute` and the canonical go token.
3. Before accepting timing, require remote `--verify-only --compile-probe` to
   report `argmax_equal=true`, `cosine_phi>=0.9997`, and
   `COMPLETE_1_TO_1`; require the runtime receipt to identify actual GPU model,
   AMP dtype, adopted compiled regions, and requested graph trees.
4. Harvest the detached call through the canonical Modal ledger/harvester.
   Report epoch-one compile/autotune separately; steady-state authority is the
   median of later `training_throughput_epoch` pairs/s and updates/s, with epoch
   seconds and peak allocated bytes. Record any graph break/capture fallback.
5. Only CUDA profiler evidence may motivate a custom hand-written Triton/CUDA
   kernel. The first profile should rank FiLM/HOSC, R, scorer, loss, backward,
   optimizer, and host-gap time before another fusion landing.

## Triality and pointer delta

- **DSL:** `compile_v9_cgauge_432_launch_config` owns every scientific value;
  backend policy owns only hardware execution choices.
- **DAG:** `FEED-438-cloud-cuda-port` advances from stripped-port blocker to
  complete functional vehicle, while paid execution remains separately gated.
- **Equations:** the NumPy-fp32 forward oracle and functional argmax/cosine law
  gate adoption; the fp-reorder bit-identity law remains recorded as advisory
  evidence under the training-only waiver.

Pointer delta: **none**. Durable delta: eleven real functional twins, a resumable
complete Torch vehicle, a functionally gated maximum-throughput CUDA execution
stack, and machine-readable timing receipts awaiting authorized CUDA evidence.
