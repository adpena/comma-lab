# Codex Findings - MLX SegNet Backend Batch Crux

utc: 2026-05-22T01:26:21Z
agent: codex
topic: mlx-scorer-port-segnet-backend-batch-crux
research_only: false
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false

## Scope

Adversarial follow-up on the MLX port of the canonical auth upstream scorer,
focused on the residual SegNet argmax drift, CUDA determinism questions, and
whether auth-eval hardware can explain the batch behavior.

## Online Research Summary

Primary-source research says this is a known numerical class, not an exotic MLX
failure mode:

- PyTorch numerical accuracy docs explicitly warn that floating-point operations
  need not be bitwise identical across platforms/backends and that batched
  computations can differ from equivalent sliced computations:
  https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html
- PyTorch reproducibility docs say CUDA convolution benchmarking can select
  different cuDNN algorithms and that disabling benchmarking is separate from
  forcing deterministic convolution algorithms:
  https://docs.pytorch.org/docs/2.12/notes/randomness.html
- PyTorch CUDA semantics docs record TF32 controls for CUDA matmul/cuDNN
  convolution; cuDNN TF32 is a relevant auth-hardware policy surface:
  https://docs.pytorch.org/docs/2.12/notes/cuda.html
- NVIDIA cuDNN reproducibility docs say bitwise reproducibility is not
  guaranteed across GPU architectures, and tensor-core/scalar paths can produce
  slightly different numerical results:
  https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-870/developer-guide/index.html

Conclusion: batch/layout/backend drift is a solved-and-documented class. Exact
MLX-vs-PyTorch-CUDA scorer parity remains workload-specific and must be measured
on the exact auth-eval axis.

## Local Evidence

Full bounded sweep over the FEC6/PR101 600-pair cache:

- command: `tools/audit_mlx_scorer_torch_parity_sweep.py --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs --repo-root . --device cpu --start-pair 0 --max-pairs 600 --window-pairs 16 --run-id fec6_pr101_full600_cpu_window16_20260522 --progress-every 8`
- verdict: `FAIL_MLX_TORCH_SCORER_PARITY_SWEEP`
- failed windows: 8 of 38
- every failed window: exactly one SegNet argmax-different pixel
- max SegNet logit delta: 0.0008352920413017273
- max SegNet argmax diff fraction: 3.178914388020833e-07
- PoseNet deltas stayed tiny.

Focused failing slice 48:80:

- windows 48:64 and 64:80 each have exactly one mismatched SegNet pixel.
- mismatch margins are extremely small:
  - 48:64 sample 6 y=183 x=299: PyTorch class 0, MLX class 1,
    PyTorch top-2 margin 9.5367431640625e-07, MLX top-2 margin
    7.62939453125e-06.
  - 64:80 sample 11 y=177 x=286: PyTorch class 2, MLX class 0,
    PyTorch top-2 margin 7.152557373046875e-06, MLX top-2 margin
    3.6716461181640625e-05.

Layer trace on 48:64:

- command: `tools/trace_mlx_segnet_layer_parity.py --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs --output experiments/results/mlx_segnet_trace_fec6_pr101_pairs48_64_affinebn_20260522.json --repo-root . --device cpu --start-pair 48 --max-pairs 16 --cliff-threshold 1e-4 --run-id fec6_pr101_pairs48_64_segnet_trace_affinebn_20260522`
- trace count: 218
- SegNet argmax diff pixels: 1
- first >1e-4 drift cliff: `encoder.stage_0.block_0.bn2`
- `encoder.stage_0.block_0.conv_pw` max delta: 6.198883056640625e-05
- `encoder.stage_0.block_0.bn2` max delta: 0.000423431396484375
- BN2 eval scale max was measured separately at 9.627345085144043.

Interpretation: BN2 is the first large amplifier, not the root operation. The
root delta enters through earlier convolution/layout math and crosses an
extremely low SegNet class margin.

## Engineering Changes

Landed/validated diagnostic tooling:

- `tools/trace_mlx_segnet_layer_parity.py` traces PyTorch-vs-MLX SegNet layer
  boundaries down to EfficientNet block internals.
- `tools/trace_torch_segnet_batch_invariance.py` traces upstream PyTorch SegNet
  whole-batch vs per-sample-loop behavior on CPU/CUDA/MPS and records backend
  metadata: torch version, CUDA/cuDNN state, TF32 flags, deterministic flags,
  MKLDNN state, thread count, and device identity.
- A local eval-mode BatchNorm2d affine-conversion probe was tested against the
  same failing window. It did not remove the failing pixel; that confirms BN is
  an amplifier, not the underlying source. The durable change from this pass is
  the backend batch-invariance trace, not an unproven BN rewrite.

## Backend Batch-Invariance Result

New CPU backend trace on the same failing 48:64 window:

- command: `tools/trace_torch_segnet_batch_invariance.py --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs --output experiments/results/torch_segnet_batch_invariance_fec6_pr101_pairs48_64_cpu_20260522.json --repo-root . --device cpu --start-pair 48 --max-pairs 16 --run-id fec6_pr101_pairs48_64_torch_cpu_batch_invariance_20260522`
- verdict: `FAIL_TORCH_SEGNET_BATCH_INVARIANCE`
- PyTorch CPU batch-vs-per-sample-loop argmax diff pixels: 1
- logit max delta: 0.00011539459228515625
- mismatched pixel: sample 6 y=183 x=299, batch class 0 vs loop class 1
- batch top-2 margin: 9.5367431640625e-07
- loop top-2 margin: 3.814697265625e-06
- backend metadata: torch 2.11.0, CPU, 6 threads, MKLDNN enabled,
  deterministic algorithms false.

Negative controls:

- `--use-deterministic-algorithms --disable-cudnn-benchmark` on CPU still has
  the same one-pixel flip.
- disabling MKLDNN still has the same one-pixel flip.
- forcing one CPU thread still has the same one-pixel flip.
- all controls preserve the same pixel coordinate and near-tie class flip.

Interpretation: local PyTorch CPU itself is not batch-invariant for this
near-boundary SegNet pixel. This is consistent with PyTorch's documented
batched-vs-sliced numerical behavior. The MLX failure is therefore not simply
"MLX bad"; it is a cross-backend/backends-and-batch-size near-tie problem.

## CUDA/Auth-Eval Question

Upstream `evaluate.py` uses CUDA when available, DALI for CUDA video loading,
`DistortionNet().eval().to(device)`, and `torch.inference_mode()`. It does not
set `torch.use_deterministic_algorithms`, `torch.backends.cudnn.deterministic`,
`torch.backends.cudnn.benchmark`, or TF32 flags in the upstream scorer file.
The local wrapper sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, but does not rewrite
scorer math.

Answer: CUDA is not guaranteed bit-exact deterministic by upstream policy. The
contest CUDA axis is authoritative because it is the contest runtime, not
because it is a mathematical ground truth. We should match or calibrate against
that axis, not assume CPU or MLX is the reference.

Next CUDA hardware probe:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/trace_torch_segnet_batch_invariance.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --output experiments/results/torch_segnet_batch_invariance_fec6_pr101_pairs48_64_cuda_20260522.json \
  --repo-root . \
  --device cuda \
  --start-pair 48 \
  --max-pairs 16 \
  --run-id fec6_pr101_pairs48_64_torch_cuda_batch_invariance_20260522
```

Run a paired strictness variant:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/trace_torch_segnet_batch_invariance.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --output experiments/results/torch_segnet_batch_invariance_fec6_pr101_pairs48_64_cuda_strict_20260522.json \
  --repo-root . \
  --device cuda \
  --start-pair 48 \
  --max-pairs 16 \
  --run-id fec6_pr101_pairs48_64_torch_cuda_batch_invariance_strict_20260522 \
  --use-deterministic-algorithms \
  --disable-cudnn-benchmark \
  --enable-cudnn-deterministic \
  --disable-tf32
```

## Engineering Path Forward

1. Keep MLX scorer outputs on a strict advisory axis until cache-window parity
   and exact CUDA auth-eval calibration are green.
2. Use the new PyTorch backend batch-invariance trace on auth-eval hardware to
   determine whether CUDA batch mode flips the same near-tie pixels, different
   pixels, or no pixels.
3. Treat low SegNet top-2 margin pixels as ambiguous scorer-response signal.
   MLX can be used for fast candidate generation, but rows with low-margin
   SegNet predictions need PyTorch/CUDA confirmation before training priority,
   rank/kill, or promotion.
4. If production MLX exactness is still required after CUDA calibration, the
   likely engineering route is not "turn on determinism"; it is either
   canonical-backend emulation at the sensitive convolution/BN boundary or a
   margin-aware fallback path to canonical PyTorch/CUDA for ambiguous windows.

## 2026-05-22 Prefix-Reset Follow-Up

Additional local diagnostic tooling was added after the batch-invariance trace:

- `src/tac/local_acceleration/mlx_segnet_prefix_reset_probe.py`
- `tools/probe_mlx_segnet_prefix_reset.py`
- `src/tac/tests/test_mlx_segnet_prefix_reset_probe.py`

The probe cumulatively resets MLX SegNet prefixes to exact PyTorch tensors and
then runs the remaining MLX suffix to final logits. This converts "first drift
cliff" into "earliest boundary whose reset actually clears final argmax."

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/probe_mlx_segnet_prefix_reset.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --output experiments/results/mlx_segnet_prefix_reset_fec6_pr101_pairs48_64_20260522.json \
  --repo-root . \
  --device cpu \
  --start-pair 48 \
  --max-pairs 16 \
  --run-id fec6_pr101_pairs48_64_mlx_segnet_prefix_reset_20260522
```

Result:

- schema: `mlx_segnet_prefix_reset_probe.v2`
- baseline argmax diff pixels: 1
- earliest zero-argmax boundary: `segmentation_head.logits`
- verdict: `PREFIX_RESET_FIXES_ARGMAX_AT:segmentation_head.logits`

Selected rows:

| reset boundary | argmax diff pixels | logit max delta |
| --- | ---: | ---: |
| `input` | 1 | 0.0006306171417236328 |
| `encoder.stage_0.block_0.bn2` | 1 | 0.0004792213439941406 |
| `encoder.stage_0` | 1 | 0.0002384185791015625 |
| `encoder.stage_6` | 1 | 0.0000820159912109375 |
| `encoder.all_features` | 1 | 0.0000820159912109375 |
| `decoder.block_4` | 1 | 0.00002288818359375 |
| `decoder.output` | 1 | 0.00002288818359375 |
| `segmentation_head.logits` | 0 | 0.0 |

Interpretation: repairing only the first drift cliff is insufficient. Even with
all encoder and decoder tensors reset to PyTorch values, the MLX segmentation
head can still flip the same near-tie pixel from a 2.29e-5 logit delta. The
final head is a sensitive contributor, but a local explicit spatial-convolution
head did not clear the full 48:80 scorer window; the full failure remains a
cumulative encoder/decoder/head drift problem crossing a sub-1e-6 PyTorch
margin.

Post-head-repair 48:80 sweep:

- command: `tools/audit_mlx_scorer_torch_parity_sweep.py --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs --output experiments/results/mlx_torch_parity_fec6_pr101_pairs48_80_cpu_window16_explicit_head_20260522.json --repo-root . --device cpu --start-pair 48 --max-pairs 32 --window-pairs 16 --run-id fec6_pr101_pairs48_80_cpu_window16_explicit_head_20260522 --progress-every 1`
- verdict: `FAIL_MLX_TORCH_SCORER_PARITY_SWEEP`
- failed windows: 2 of 2
- each failed window still has exactly one SegNet argmax-different pixel
- mismatch coordinates remain sample 6 y=183 x=299 for 48:64 and sample 11
  y=177 x=286 for 64:80

Actionable conclusion: exact full-argmax parity is not going to be achieved by
one local op replacement. The production-safe path is a margin-aware fallback:
let MLX generate fast advisory scorer-response priors, but route low-margin
SegNet windows through canonical PyTorch/CUDA before any rank/kill, training
priority, or promotion decision.

## 2026-05-22 SE Pool Variant Follow-Up

Additional local diagnostic tooling was added for the stage-0 squeeze-excite
pooling reduction order:

- `src/tac/local_acceleration/mlx_segnet_se_pool_variants.py`
- `tools/probe_mlx_segnet_stage0_se_pool_variants.py`
- `src/tac/tests/test_mlx_segnet_se_pool_variants.py`

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/probe_mlx_segnet_stage0_se_pool_variants.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --output experiments/results/mlx_segnet_stage0_se_pool_variants_fec6_pr101_pairs48_64_20260522.json \
  --repo-root . \
  --device cpu \
  --start-pair 48 \
  --max-pairs 16 \
  --run-id fec6_pr101_pairs48_64_mlx_segnet_stage0_se_pool_variants_20260522
```

Result:

- best pool delta variant: `mean_h_then_w`
- best SE output delta variant: `mean_w_then_h`
- best output max_abs_delta: 2.384185791015625e-06

The SE adapter now uses the `mean_w_then_h` reduction order, which is the best
forced-output local variant on this probe. The 48:80 full scorer sweep remains
unchanged after combining explicit spatial head convolution plus this SE pool
order:

- verdict: `FAIL_MLX_TORCH_SCORER_PARITY_SWEEP`
- failed windows: 2 of 2
- total SegNet argmax mismatches: 2
- mismatch coordinates unchanged.

This is useful local numerical tightening, not a parity completion. The
remaining production boundary is still the same: low-margin SegNet windows need
canonical PyTorch/CUDA confirmation.

Regression coverage also now verifies that the EfficientNet SE adapter uses
the same sequential width-then-height pool order:
`src/tac/tests/test_mlx_efficientnet_se_pool_repair.py`.

## 2026-05-22 Reusable Adapter State Fix

The broad adapter suite exposed a production-critical repeated-call bug in the
MLX scorer adapter: reusing one `torch_distortion_net_to_mlx(...)` adapter for
reference and candidate scorer calls could corrupt the second PoseNet output.
The failing symptom was a PoseNet component delta of `1.884989857673645` against
an expected `1.4442425e-09` in
`test_distortion_scorer_responses_and_components_match_torch_on_mlx_cpu`.

Root class: stateful MLX eval BatchNorm reuse. Durable fix:

- `torch_batchnorm2d_to_mlx` now returns a stateless precomputed affine adapter
  for eval-mode `BatchNorm2d` with running stats.
- The existing MLX `nn.BatchNorm` path remains only for unsupported non-eval or
  non-running-stat modes.
- Regression: `test_batchnorm2d_adapter_uses_eval_affine_for_running_stats`.

Verification:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  'src/tac/tests/test_mlx_scorer_adapters.py::test_distortion_scorer_responses_and_components_match_torch_on_mlx_cpu' \
  src/tac/tests/test_mlx_scorer_adapters.py
```

Result: `37 passed in 6.74s`.

After affine BN plus explicit spatial head plus SE pool order, the same 48:80
full scorer sweep still fails closed:

- command: `tools/audit_mlx_scorer_torch_parity_sweep.py --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs --output experiments/results/mlx_torch_parity_fec6_pr101_pairs48_80_cpu_window16_affinebn_head_sepool_20260522.json --repo-root . --device cpu --start-pair 48 --max-pairs 32 --window-pairs 16 --run-id fec6_pr101_pairs48_80_cpu_window16_affinebn_head_sepool_20260522 --progress-every 1`
- verdict: `FAIL_MLX_TORCH_SCORER_PARITY_SWEEP`
- failed windows: 2 of 2
- total SegNet argmax mismatches: 2
- max SegNet logit delta: 0.0006306171417236328

This is the current honest state: repeated-call correctness is fixed, local op
drift is tightened, but exact SegNet argmax parity is still blocked by two
near-tie pixels on this slice.

## 2026-05-22 SE 1x1 Conv Variant Follow-Up

Additional local diagnostic tooling was added for native-vs-explicit 1x1 MLX
convolutions inside the stage-0 squeeze-excite module:

- `src/tac/local_acceleration/mlx_segnet_se_conv_variants.py`
- `tools/probe_mlx_segnet_stage0_se_conv_variants.py`
- `src/tac/tests/test_mlx_segnet_se_conv_variants.py`

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/probe_mlx_segnet_stage0_se_conv_variants.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --output experiments/results/mlx_segnet_stage0_se_conv_variants_fec6_pr101_pairs48_64_20260522.json \
  --repo-root . \
  --device cpu \
  --start-pair 48 \
  --max-pairs 16 \
  --run-id fec6_pr101_pairs48_64_mlx_segnet_stage0_se_conv_variants_20260522
```

Result:

- verdict: `EXPLICIT_1X1_IMPROVES:conv_reduce`
- worst native row: `conv_reduce`, max_abs_delta 1.430511474609375e-06
- worst explicit row: `conv_reduce`, max_abs_delta 7.152557373046875e-07

The SE adapter now uses explicit ordered 1x1 convolution for `fc1` and `fc2`.
Combined with affine BN, explicit spatial segmentation-head convolution, and
the SE pool order, the 48:80 full scorer sweep still fails closed:

- command: `tools/audit_mlx_scorer_torch_parity_sweep.py --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs --output experiments/results/mlx_torch_parity_fec6_pr101_pairs48_80_cpu_window16_affinebn_explicit_convs_sepool_20260522.json --repo-root . --device cpu --start-pair 48 --max-pairs 32 --window-pairs 16 --run-id fec6_pr101_pairs48_80_cpu_window16_affinebn_explicit_convs_sepool_20260522 --progress-every 1`
- verdict: `FAIL_MLX_TORCH_SCORER_PARITY_SWEEP`
- failed windows: 2 of 2
- total SegNet argmax mismatches: 2
- max SegNet logit delta: 0.0006306171417236328

This further supports the same conclusion: we can and should tighten local
operator drift, but these near-tie argmax pixels remain backend-sensitive and
need canonical-axis fallback.

## 2026-05-22 Repaired-SE Full-Logit Probe

Additional local diagnostic tooling was added to run repaired stage-0 SE
variants through final SegNet logits:

- `src/tac/local_acceleration/mlx_segnet_repaired_se_probe.py`
- `tools/probe_mlx_segnet_repaired_stage0_se.py`
- `src/tac/tests/test_mlx_segnet_repaired_se_probe.py`

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/probe_mlx_segnet_repaired_stage0_se.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --output experiments/results/mlx_segnet_repaired_stage0_se_fec6_pr101_pairs48_64_20260522.json \
  --repo-root . \
  --device cpu \
  --start-pair 48 \
  --max-pairs 16 \
  --run-id fec6_pr101_pairs48_64_mlx_segnet_repaired_stage0_se_20260522
```

Result:

- verdict: `REPAIRED_SE_IMPROVES_LOGITS:cpu_pool_repair`
- native argmax diff pixels: 1
- best variant: `cpu_pool_repair`
- best variant logit max delta: 0.0004544258117675781
- best variant argmax diff pixels: 1

This confirms the SE pool-order repair can reduce final-logit drift for the
48:64 window, but it still does not clear the near-tie argmax pixel.
