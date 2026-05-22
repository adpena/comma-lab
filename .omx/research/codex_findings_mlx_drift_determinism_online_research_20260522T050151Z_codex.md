# Codex Findings: MLX Drift And Determinism Research

timestamp_utc: 2026-05-22T05:01:51Z
lane_id: lane_codex_mlx_auth_scorer_hardening_20260522
author: codex
verdict: PROCEED_WITH_TRACE_AND_CUSTODY_GUARDS

## Scope

Adversarial online/source check for whether the MLX scorer drift is a solved
engineering problem, an expected hardware/software numerical difference, or an
open portability surface.

## External Sources Checked

- MLX docs: MLX is lazy, multi-device, and graph-optimizing; compiled functions
  are documented as matching regular functions only up to numerical precision.
  Sources:
  <https://ml-explore.github.io/mlx/build/html/index.html>,
  <https://ml-explore.github.io/mlx/build/html/usage/compile.html>.
- MLX random docs: PRNG state can be global or explicit-keyed, so exact probes
  should record seed/key mode rather than relying on ambient state.
  Source: <https://ml-explore.github.io/mlx/build/html/python/random.html>.
- Apple Metal docs: Metal exposes fast vs precise FP32 math function modes, and
  Apple-silicon shader compilation can optimize or merge floating-point
  operations for performance, changing results slightly.
  Sources:
  <https://developer.apple.com/documentation/metal/mtlmathfloatingpointfunctions>,
  <https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon>.
- PyTorch reproducibility docs: PyTorch does not guarantee identical results
  across releases/platforms or CPU vs GPU. CUDA convolution benchmarking can
  choose different algorithms, and deterministic algorithms should be enabled
  when reproducibility is required.
  Source: <https://docs.pytorch.org/docs/2.12/notes/randomness.html>.
- cuDNN docs: most routines are bitwise reproducible run-to-run on the same
  GPU architecture, but exceptions exist, and cross-architecture bitwise
  reproducibility is not guaranteed. Tensor Core opt-in can change numerical
  sequencing.
  Source:
  <https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-870/developer-guide/index.html>.
- NVIDIA CCCL docs/blog: reproducible reductions require explicit determinism
  contracts; fixed reduction order / reproducible accumulators are the portable
  solution class.
  Source:
  <https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/>.

## Interpretation

The observed MLX drift is plausibly an engineering-portability problem, not a
single obvious bug. The likely mixed causes are:

- operation-order differences from MLX graph compilation/laziness;
- Metal fast-vs-precise FP32 behavior, FMA/reassociation, and shader-level
  optimization;
- reduction-order differences in mean/std/pooling/convolution-adjacent paths;
- batch-shape/layout-sensitive kernels;
- possible PyTorch CUDA reference behavior that is deterministic on the same
  hardware stack but still not cross-platform bit-exact authority.

This means local MLX can be a high-value research substrate only after each
artifact carries explicit evidence semantics and auth-axis custody. The correct
engineering response is not to pretend MLX is exact CUDA; it is to isolate the
first divergent primitive, make drift local and bounded, and allow local MLX
only to triage candidates when strict auth/cache identity gates pass.

## Engineering Actions Already Landed

- `src/tac/local_acceleration/mlx_score_calibration.py` now requires strict
  contest auth-eval payloads for calibration anchors; scalar-only CPU/CUDA
  fields are not score authority.
- `src/tac/local_acceleration/mlx_segnet_trace_compare.py` now fails closed
  unless trace manifests carry false authority fields, MLX evidence labels,
  device metadata, and explicit GPU research-signal allowance.
- Hash/cache identity gates keep MLX transfer calibration tied to archive,
  inflated-output, tensor-shape, and scorer-input hash agreement.

## Next Debug Crux

Build a primitive-level trace ladder:

1. Lock inputs/weights and record MLX CPU, MLX GPU, PyTorch CPU, and PyTorch
   CUDA outputs for each SegNet preprocessing, convolution, normalization,
   pooling, activation, resize/interpolation, and reduction primitive.
2. Separate run-to-run nondeterminism from cross-backend deterministic drift by
   repeating each primitive N times under a fixed seed/key and recording both
   max ULP drift and aggregate output hashes.
3. Use reference NumPy/PyTorch CPU implementations for the first divergent
   primitive, then replace only that MLX primitive with a deterministic/precise
   custom implementation if it materially reduces end-to-end drift.
4. Preserve the existing rule: primitive parity improvements remain
   `[macOS-MLX research-signal]` evidence until paired auth-axis eval proves
   transfer.

## Non-Authority

This memo does not claim an MLX score, promotion, rank/kill verdict, or exact
CUDA equivalence. It is a portability engineering roadmap for making MLX useful
without weakening contest-score custody.
