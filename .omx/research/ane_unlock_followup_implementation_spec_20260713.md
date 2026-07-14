# Implementation spec — ANE unlock follow-up (local-only MEANS lane)

Date: 2026-07-13  
Lane: `lane_ane_unlock_followup_20260713`  
Checkpoint: `ane_unlock_followup`  
Authority: operator directive 2026-07-13; local build/bench/measure and git only  
Research-only: `true`

## Outcome and authority boundary

Build and measure three orthogonal questions without mutating the witness trainer:

1. Probe the direct private `AppleNeuralEngine.framework` / `ANECompiler.framework`
   execution surface, distinguishing framework presence, runtime class/method reachability,
   a directly executed ANE forward, and any exposed backward/autodiff/VJP surface. A direct
   `_ANEModel` route may guarantee residency only after a model actually executes through that
   API; framework or selector discovery alone must not be called residency.
2. Run a matched ABBA local concurrency experiment between a real frozen-SegNet CoreML
   `CPU_AND_NE` forward loop and a representative MLX Metal load. Measure both solo and
   concurrent medians and compute degradation as `T_concurrent/T_solo-1`; accept only when both
   upper estimates are below 5%. Record `powermetrics` counters when permission permits and a
   typed blocker otherwise.
3. Convert the real frozen EfficientNet-B2 SegNet to CoreML, compress weights only (per-channel
   symmetric W8; LUT4/LUT6 as bounded follow-ups), compile `fwd_b1`, `fwd_b8`, and `fwd_b32`
   functions with shared-weight deduplication where the installed SDK supports it, and measure
   package/blob bytes, batch latency, latency per pair, and argmax fidelity against the
   NumPy-fp32/Torch-fp32 reference over the real `gt_n600.npz` frame-1 states. Also measure the
   exact `ct.transform.FP16ComputePrecision(op_selector)` logit-head T4 arm over n600.

This is compute/teacher throughput MEANS. MPS, MLX, ANE, and CoreML remain local advisory axes.
No measurement in this lane is a score row; only a byte-closed exact n600
`upstream/evaluate.py` row on a contest axis can move the pointer.

## Settled inputs (consume, do not reopen)

- `.omx/research/ane_ecosystem_survey_20260713.md`
- `.omx/research/ane_unlock_correction_20260713.md`
- `.omx/research/ane_unlock_directive_20260713.md`
- `.omx/research/GO_PACKET_inloop_component_timer_20260713.md`
- `.omx/research/throughput_fresh_eyes_measurements_20260713.json`
- Real frozen weights: `upstream/models/segnet.safetensors`
- Real full state cache: `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`

The W8A8 result (45.8368% held-out flips) kills only the W8A8 formulation. It does not answer
weight-only W8 or palettization. The prior requested `CPU_AND_NE` 9.098 ms result does not prove
ANE placement. The prior full-float32 CoreML row is advisory-only and backward remains unmeasured.

## Owned new files

- `src/tac/local_acceleration/ane_unlock_followup_20260713.py`
- `src/tac/local_acceleration/tests/test_ane_unlock_followup_20260713.py`
- `src/tac/canonical_equations/ane_unlock_followup_20260713.py`
- `src/tac/canonical_equations/tests/test_ane_unlock_followup_20260713.py`
- `src/tac/witness_dsl/ane_unlock_followup_policy_20260713.py`
- `src/tac/witness_dsl/tests/test_ane_unlock_followup_policy_20260713.py`
- `tools/native/ane_direct_residency_probe_20260713.m`
- `tools/probe_ane_direct_residency_20260713.py`
- `tools/bench_ane_unlock_followup_20260713.py`
- `tools/prepare_ane_full_trainer_concurrency_ab_20260713.py`
- `.omx/research/ane_unlock_followup_DAG_FEED_20260713.md`
- `.omx/research/GO_PACKET_ane_full_trainer_concurrency_ab_20260713.md`
- `.omx/research/ane_unlock_followup_20260713.md`
- `experiments/results/ane_unlock_followup_20260713/**`

If a listed file collides before creation, stop and choose a new dated filename rather than
editing another lane's bytes.

## Files that are read-only for this lane

Do not edit the trainer, `tools/launch_witness_run.py`, `witness_control/**`,
`curriculum_dsl.py`, `resume_registry.py`, any v9/#432 run file, any file owned by
`throughput_fresh_eyes`, `custom_sparse_adjoint_kernel`, `quant_tail_reliability`, or
`pre_se_reopen_a`, or shared hot registries other than CLI-owned lane/checkpoint/task surfaces.
Do not build a rival adjoint kernel. If a true ANE backward surface is found, route the evidence
to `custom_sparse_adjoint_kernel` in the final memo.

## Required implementation behavior

### Direct private-API probe

- Compile a small Objective-C binary with `clang`, `Foundation`, `dlopen`, and Objective-C runtime
  enumeration. Load both private frameworks directly, enumerate ANE-prefixed classes and selectors,
  and record exact candidate execution and backward/autodiff/VJP selectors.
- Attempt only locally safe construction/introspection. Never forge entitlements or weaken system
  security. A missing entitlement, model artifact, compile service, or executable selector becomes
  `BLOCKED_NOT_MEASURED` with `verdict_scope` and `req_R`.
- Emit a durable JSON receipt including framework hashes/paths, OS/build, architecture, compile
  command, binary hash, exit status, class/selector inventory, and the narrow backward verdict.

### CoreML/MLX measurement harness

- Use a success-cleaned scratch virtualenv under the storage waterfall. Prefer the connected SSD;
  if the managed sandbox refuses SSD writes, record that fact and use `/private/tmp` scratch.
  Install from the local `uv` cache offline. Delete success-only scratch automatically. Never cite
  scratch as evidence; copy small JSON receipts into the owned result directory atomically.
- Fix seeds, `torch.set_num_threads(1)`, record all package versions and source/model/cache hashes,
  warm up each arm, use ABBA order, and keep raw sample arrays.
- The representative MLX load must execute real Metal kernels with a workload-shape manifest
  grounded in the throughput sibling's n24 per-step evidence. It must not invoke the governed
  trainer. If no Metal device is accessible, emit `BLOCKED_NOT_MEASURED`; do not substitute CPU.
- `powermetrics` failure is non-fatal for timing but must be recorded with its exact permission
  blocker. Never call a requested CoreML compute unit proof of placement.

### Weight compression, batches, and n600 fidelity

- Build the real frozen model once from exact safetensors custody. W8 means weight-only: no
  activation quantization. Record original fp32/fp16 tensor bytes, compressed weight blob/package
  bytes, and the derived 32 MiB-cliff comparison. Do not call package size literal on-chip SRAM.
- Compile/benchmark b1/b8/b32 with identical weights and preprocessing. For a multifunction
  package, use CoreML's supported descriptor/save API and verify function names by parse-back.
- Stream the stored `gt_f1.npy` member inside `gt_n600.npz`; do not materialize a second 1.8 GiB
  frame tree. Report aggregate and worst-pair argmax flip fractions, with exactly 600 real states.
- Compare W8/LUT/T4 outputs to the same fp32 reference. A smaller sample may be used only for a
  smoke and must be labeled `n24-extrap`, never n600.
- T4 uses `ct.transform.FP16ComputePrecision(op_selector)` in one graph and records the selected
  MIL ops. It is not a hand-split pipeline.

### Held full-trainer concurrency AB

- Generate, but do not launch, a bounded n24 two-arm packet. Reuse the throughput owner's typed
  component-timer config and governed launcher path. The only treatment is the direct/CoreML ANE
  forward sidecar; the trainer configuration is byte-identical across arms.
- The packet must be operator-GO-only, per-stage checkpointed, resumable, storage-preflighted,
  source-hash bound, and command-ready. It must explicitly label any n600 payoff as DERIVED from
  n24 per-pair timing rather than measured.

## Canonical laws and gates

Land pure, tested equations for:

- `d_teacher = T_teacher_concurrent/T_teacher_solo - 1`
- `d_mlx = T_mlx_concurrent/T_mlx_solo - 1`
- concurrency admission iff both degradations are `<0.05` and placement is independently proved
- `t_pair(B)=T_batch(B)/B` and throughput `B/T_batch(B)`
- weight-fit headroom `H=32*2^20-W_payload`; package/blob bytes are measured, SRAM residency is
  derived/unknown unless a hardware counter proves it
- forward-only Amdahl bound using the throughput sibling's measured in-loop split only after that
  split lands; until then report a parameterized law, not a guessed speedup

Every negative must carry the narrowest `verdict_scope` and a concrete `req_R` reactivation.

## Acceptance and verification

- Direct probe builds and emits a parseable receipt even when execution is blocked.
- Unit tests cover degradation math, 5% strict boundary, batch accounting, SRAM labeling,
  n600 count enforcement, typed operator-GO refusal, and receipt authority flags.
- All owned Python files compile and focused tests pass.
- Measurement receipts are atomic, deterministic where applicable, hash-custodied, and contain
  `score_claim=false`, `promotion_eligible=false`, `pointer_moved=false`.
- Final memo leads with one headline and updates advisory, training-gradient, and label-grade
  tiers separately. It names what is MEASURED, DERIVED, INFERRED, or ASSUMED.
- Triality lands as isolated DSL policy, DAG FEED, and equation module. No shared registry append is
  required while shared registries are sibling-dirty.
- Commit only owned new files via `tools/subagent_commit_serializer.py` using post-edit hashes and
  required review-tracker passes. Preserve all inherited dirty files.

