# Task #494 implementation specification — compute-substrate authority ladder

**UTC:** 2026-07-14T01:07:47Z  
**Lane:** `throughput_authority_ladder` (L0, `research_only=true`)  
**Pointer:** submittable `0.19108282419209976 [contest-CPU]`; borrowed non-submission bank
`0.1880443979880752 [contest-CPU]`; this work is MEANS and cannot move either pointer.  
**Containment:** local `$0` build and measurement only. Do not run the governed in-loop timer,
`upstream/evaluate.py`, CUDA/provider work, a live-run/config mutation, a run stop, or a training launch.

## Objective

Build four independently testable surfaces that turn the settled one-axis Q15/int32 result into:

1. a full four-stage render-R adjoint, cross-process n600 real-frame bit-identity measurement; and
2. a calibrated fixed-point/QDQ frozen-SegNet forward precision ladder over the real n600 frame set,
   with exact aggregate and worst-pair argmax-flip accounting and a measured tie-margin budget;
3. actual default-OFF custom Metal integer-MAC and CoreML/ANE W8A8 verdict candidates with parity,
   residency, and timing host harnesses; and
4. an authority-retaining typed training-loop assignment policy plus an integer render-R adjoint backend.

The first answers whether integer, order-independent accumulation transfers beyond the single
`384->874` transpose instance. The second answers the distinct forward-verdict feasibility question.
No result may be transferred to MPS, ANE residency, CUDA, PoseNet, contest score authority, or a
training-loop speedup without its own receipt.  MAIN owns every Metal/ANE execution; this arm builds
the kernels and emits exact host commands without waiting on unavailable hardware.

## Settled inputs — consume, do not re-derive

- Existing one-axis probe and 10-process Metal receipt:
  `tools/probe_pythagorean_exact_arithmetic_bitident.py` and
  `.omx/research/pythagorean_exact_arithmetic_bitident_probe_20260713.json`.
- Full NumPy/fixed-order R-VJP structure:
  `src/tac/local_acceleration/metal_fused_r_operator.py`.
- Live R forward geometry:
  `src/tac/local_acceleration/pr95_hnerv_mlx_training.py`.
- Real pair cache:
  `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`, SHA-256
  `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`, ZIP_STORED
  `gt_f0.npy` and `gt_f1.npy`, each `(600,874,1164,3)` uint8. Stream/memmap it; never inflate or copy it.
- Frozen SegNet weights:
  `upstream/models/segnet.safetensors`, SHA-256
  `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`.
- #456 exact-forward result: one-thread CPU is the training teacher; six-thread speed transfer failed
  exactness on 15/600 pairs. Do not reopen it.
- #482/#490 correction results: naive/CoreML weight-only W8 is a scoped NO-GO (n600 aggregate
  `38.291378%`, worst pair `88.775126%`); a calibrated operand ladder is a distinct formulation.
- ANE is forward-only on the exposed runtime and remains local advisory. MPS remains never-authority.

## Work unit A — full R-adjoint n600

### Owned files

- Modify `tools/probe_pythagorean_exact_arithmetic_bitident.py`.
- Modify `tools/run_pythagorean_exact_arithmetic_bitident_host.command` only to forward an explicit
  full-scope argument set without changing the default settled one-axis command.
- Modify `src/tac/tests/test_probe_pythagorean_exact_arithmetic_bitident.py`.

Do not edit the trainer, fused-R implementation, DSL, shared DAG, registry, or any run directory.

### Required behavior

- Preserve the existing default v1 one-axis CLI, receipt schema, and tests byte-for-behavior.
- Add an explicit scope such as `--scope full-r-n600`, a distinct v2 receipt schema, `--gt-cache`,
  `--pair-start`, `--pair-count` (authority requires 600), `--n`, and resumable atomic checkpoints.
- Full chain order must match the NumPy authority exactly:
  `Down-W^T(512->1164) -> Down-H^T(384->874) -> uint8-STE clip mask ->
   Up-W^T(1164->512) -> Up-H^T(874->384)`.
- Exercise both `gt_f0` and `gt_f1` for all 600 pairs (1,200 real frames), one frame at a time.
- Derive a deterministic real-image cotangent from each frame and record the formulation literally.
  It is an `INSTANCE: real-0.mkv n600 R-chain residual-cotangent`, not a scorer/training cotangent.
- Float cell uses the current MLX float32 duplicate-index atomic adds. Fixed cell uses the same indices
  and Q-weight operands with bounded int32 atomic adds and deterministic signed round-to-nearest
  requantization between stages. Never rely on overflow, saturation, or implementation-defined shifts.
- Before every integer stage, compute/record an int64 sum-of-absolute-contributions bound, maximum fan-in,
  int32 headroom, and fail closed before MLX execution if unsafe.
- Build a single streaming NumPy fixed-order authority pass. Record canonical corpus digests, exact
  NumPy-int parity, max/RMSE error versus NumPy-fp32, and a conservative accumulated quantization plus
  fp32-reorder bound propagated across all four stages. Do not emit tensors or bulk caches.
- Each of ten child processes streams the same corpus and returns only custody, digests, error aggregates,
  stage bounds, and completion counts. Parent persists after every child; interruption loses at most one child.
- A missing Metal device is `BLOCKED_NOT_MEASURED`, never a zero-valued negative.

### Rung-A acceptance

- Coverage is exactly 600 pairs / 1,200 frames and every cache/source hash matches.
- `float_full_r_vjp`: ten completed children and more than one corpus hash (reproduces the fp-reorder wall).
- `fixed_full_r_vjp`: ten completed children, exactly one corpus hash, exact NumPy-int digest parity for
  every child, no overflow at any stage, and final dequantized error within the derived full-chain bound.
- If any clause fails, verdict_scope names the first failing axis/stage/bit allocation and stays at
  `INSTANCE` or `FORMULATION`; it must not kill integer lowering as a family.

### Required tests

- Existing one-axis tests remain green.
- Four-stage order/shape and forward-index direction are tested on a small non-square fixture.
- Signed requantization tests cover positive/negative half-way cases and forbid implementation-defined bias.
- Overflow preflight has passing and failing fixtures.
- Streamed memmap coverage proves both frame members and all requested pair indices are consumed once.
- Resume rejects source/probe/contract hash drift and does not repeat completed children.
- Summary verdict tests cover positive, float-not-reproduced, integer-divergent, overflow, incomplete,
  and no-Metal cases.

Exact checks:

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_probe_pythagorean_exact_arithmetic_bitident.py
.venv/bin/ruff check tools/probe_pythagorean_exact_arithmetic_bitident.py \
  src/tac/tests/test_probe_pythagorean_exact_arithmetic_bitident.py
.venv/bin/python tools/probe_pythagorean_exact_arithmetic_bitident.py --numpy-only
```

## Work unit B — calibrated forward fixed-point feasibility

### Owned files

- Add `tools/probe_fixedpoint_scorer_forward_n600.py`.
- Add `src/tac/tests/test_probe_fixedpoint_scorer_forward_n600.py`.

Do not edit scorer weights, upstream code, the trainer, CoreML/ANE files, custom Metal-conv files, DSL,
the shared DAG, or existing receipts.

### Required behavior

- CPU Torch, eager NCHW, batch 1, `torch.set_num_threads(1)` and inter-op 1 in a fresh process. MPS is
  forbidden. Exact fp32 logits/argmax are the local reference.
- Stream `gt_f1.npy` for all 600 real pairs from the ZIP_STORED cache. Resize exactly as upstream SegNet.
- Freeze a preregistered calibration split before results: pairs `0..119`; held-out pairs `120..599`.
  Report calibration, held-out, and full-n600 separately; only held-out/full-n600 gate a rung.
- Build a real calibrated operand ladder, not weight-only quantization: symmetric per-output-channel
  weight scales and fixed per-operator activation scales derived only from the calibration split.
  Biases and normalization parameters remain fp32 and are named in the receipt. Use QDQ/fp32 accumulation
  as a feasibility emulation and say explicitly that it is not an integer-MAC or speed receipt.
- Evaluate signed bit budgets at least `{8,10,12,14,16}` plus the fp32 control. A deterministic
  calibration-chosen mixed arm may retain the segmentation head in fp32, but held-out labels may not
  choose scales/arms. Preserve all chosen scales or their canonical digest in the receipt.
- Resume after a bounded pair interval with a fingerprint over tool, weights, cache, runtime, calibration
  policy, thread contract, bit ladder, and split. Persist no logits or frames.
- Per rung record: total flips/pixels/fraction, full ordered argmax digest, per-pair flips/fraction,
  worst-pair index/fraction, top1-top2 baseline margin quantiles, flipped-pixel margin quantiles,
  maximum/RMSE logit error, and the count/fraction not certified by `margin > 2*max_class_abs_error`.
- Report two gates without conflation:
  1. `ARGMAX_EXACT`: zero flips on held-out and full n600;
  2. `TRAINING_TOLERANCE`: aggregate and worst-pair flip fractions each `<=3.3e-5`.
  The minimum admitted precision is the lowest bit rung satisfying the named gate; otherwise `NONE`.
- PoseNet has continuous pose outputs, not argmax. This tool may leave PoseNet `OWED` with that reason;
  it must not claim PoseNet preservation from SegNet labels. A later PoseNet rung must compare first-six
  pose coordinates and nonlinear pose-score debt on both frames.

### Rung-B acceptance

- Exactly 600 full rows and 480 held-out rows per completed rung; worst-pair metrics present.
- Calibration state is derived only from indices 0..119 and cannot change during held-out evaluation.
- A toy-network test proves the activation path is instrumented (an input-only/weight-only fake must fail).
- Positive controls reproduce fp32 argmax exactly; a forced-low-bit negative produces measured flips.
- Receipt and summary distinguish `MEASURED`, `DERIVED`, and `OWED`; every negative has narrow scope.

Exact checks:

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_probe_fixedpoint_scorer_forward_n600.py
.venv/bin/ruff check tools/probe_fixedpoint_scorer_forward_n600.py \
  src/tac/tests/test_probe_fixedpoint_scorer_forward_n600.py
.venv/bin/python tools/probe_fixedpoint_scorer_forward_n600.py --help
```

## Cross-piece review and landing

- Run both focused suites together, Ruff, `py_compile`, and `git diff --check`.
- Adversarially re-derive R-stage order, cache coverage, all counts/denominators, calibration isolation,
  worst-pair selection, overflow proof, and the distinction between QDQ feasibility and integer authority.
- Do not modify the hot trainer, shared DAG, or shared equation registry.  Land a standalone typed policy,
  canonical-equation module, and `_DAG_FEED_*.md`; MAIN merges them after host receipts.
- Host Metal/ANE execution is delegated to MAIN by live directive.  The governed in-loop timer remains a
  separately governed measurement even though MAIN reports that it is firing it under operator GO.

## Work unit C — actual fixed-point verdict candidates

### Owned files

- Add `src/tac/local_acceleration/metal_fixedpoint_verdict.py` and focused tests.
- Add `tools/bench_fixedpoint_authority_kernels.py` and
  `tools/run_fixedpoint_authority_kernels_host.command`.
- Add `tools/build_ane_fixedpoint_verdict.py` and
  `tools/run_ane_fixedpoint_verdict_host.command`.

### Required behavior

- Metal pointwise and depthwise primitives quantize signed activations and per-output-channel weights,
  perform integer multiply with int32 accumulation in an explicit traversal, add a separately scaled
  int32 bias where representable, and dequantize once at the output.  They are not QDQ-float kernels.
- Every packet records input/weight/output scales, bit widths, exact maximum fan-in, a worst-case int64
  accumulator bound, int32 headroom, and fails closed before dispatch if the proof does not fit.
- The NumPy int32 reference is byte-level authority for each primitive.  The host harness first proves
  exact integer accumulator parity, then measures dequantized/logit error and SegNet argmax flips.
- The adapter is frozen-forward-only and default OFF.  Unsupported convolutions or non-finite calibration
  state must fail closed; native MLX fallback is forbidden after explicit opt-in.
- CoreML builder consumes the same preregistered calibration split and emits W8A8 plus mixed-head models.
  It must request `CPU_AND_NE`, record model/package hashes, compare first-six PoseNet outputs where used,
  and label ANE residency `UNPROVED` unless an Instruments/compute-plan receipt proves placement.
- Metal and ANE host receipts report warmup, synchronized batch-1 latency, aggregate/worst-pair Seg flips,
  continuous Pose max-abs/MSE/`sqrt(10*d_pose)` debt, and separate exact/tolerance gates.  No speed claim
  is admitted without both parity and placement/residency custody.

### Work-unit-C acceptance

- Pure-NumPy packet/reference tests prove signed rounding, per-channel scale alignment, overflow refusal,
  deterministic accumulator bytes, and a forced-low-bit argmax negative.
- CPU-only import/help/build-plan paths work without Metal/CoreML; missing optional runtimes are explicit
  `BLOCKED_NOT_MEASURED`, never silent fallbacks.
- MAIN can run both `.command` files from a clean shell and obtain resumable, atomically written receipts.

## Work unit D — integer R-adjoint and final authority-retaining loop

### Owned files

- Add `src/tac/local_acceleration/metal_integer_r_adjoint.py` and focused tests.
- Add `src/tac/witness_dsl/throughput_authority_policy_20260714.py` and focused tests.
- Add `tools/bench_integer_r_adjoint_backend.py` and
  `tools/run_integer_r_adjoint_backend_host.command`.

### Required behavior

- Implement all four render-R transpose axes as one reusable Metal integer-adjoint family.  Each axis uses
  precombined transpose CSR taps, signed Q weights, int32 accumulation, deterministic signed requantization,
  and no float atomic or ordering contract.  The clip mask is applied at the same chain boundary as the
  NumPy authority.  A custom-function wrapper is provided only behind an explicit typed policy.
- Preserve the exact float fused-R path as the default.  Integer enablement requires a host receipt matching
  the current kernel/tool/policy hashes, full 1,200-frame coverage, one cross-process digest, exact NumPy-int
  parity, error within the derived budget, and a measured positive speed result.  Stale receipts fail closed.
- The typed assignment policy names each op family, substrate, precision, authority grade, fallback, and
  evidence gate.  It must keep MPS research-only, ANE forward-only, CPU/CUDA terminal score axes separate,
  and forbid inferred equivalence between macOS local results and contest authority.
- The final loop is train-fast/retain-authority: fast Metal/MLX gradient and candidate-verdict stages can
  reject obviously bad candidates, but archive selection and pointer movement remain exact CPU/CUDA replay
  on exact bytes.  A candidate-authority scorer is a filter until its own contest-axis equivalence receipt.
- The float megakernel negative is scoped to float/fixed-order formulation.  Integer order-independent fusion
  is a distinct formulation whose host benchmark reports launch count, bytes moved, latency, and parity.

### Work-unit-D acceptance

- Tests prove policy compilation is deterministic, rejects illegal substrate/authority combinations, rejects
  missing/stale receipts, and preserves the exact CPU/CUDA terminal fallback.
- NumPy and CPU-only tests cover four-axis shapes, integer overflow proofs, clip-boundary behavior, signed
  requantization, and no-atomic kernel source registration.
- MAIN host command produces the decisive kernel receipt; this arm does not label the backend admitted before it.
