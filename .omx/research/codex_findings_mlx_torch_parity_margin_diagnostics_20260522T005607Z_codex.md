# Codex Findings: MLX Torch Parity Margin Diagnostics

utc: 2026-05-22T00:56:07Z
lane_id: lane_mlx_auth_scorer_training_signal_fidelity_20260521
evidence_axis: [macOS-MLX research-signal]
verdict: FAIL_CLOSED_SEGNET_ARGMAX_BOUNDARY_SENSITIVITY

## Finding

The strict PyTorch-vs-MLX scorer parity sweep failure is localized to sparse
SegNet `argmax` flips, not broad logit or PoseNet drift. The upstream contest
scorer computes SegNet distortion from exact class-label disagreement, so these
single-pixel flips are scorer-component deltas and must remain fail-closed for
promotion authority.

The likely mechanism is near-decision-boundary numerical sensitivity: at the
observed failing pixels, the PyTorch top-2 class margins are smaller than the
MLX-vs-PyTorch logit delta at the same pixel.

## What changed

- Added `segnet_argmax_mismatch_detail` to the parity manifest.
- Added sweep-level summaries for total mismatch pixels, mismatch top-2
  margins, and mismatch-pixel logit deltas.
- Added concrete mismatch examples with sample index, pixel coordinate,
  PyTorch class, MLX class, top-2 margins, and logit delta.
- Added unit coverage for the mismatch-detail fields.

## FEC6 Margin Diagnostic Anchor

Command:

```bash
tmpdir=$(mktemp -d)
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/audit_mlx_scorer_torch_parity_sweep.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --repo-root . \
  --device cpu \
  --start-pair 48 \
  --max-pairs 32 \
  --window-pairs 16 \
  --run-id fec6_pr101_pairs48_80_cpu_window16_margin_20260522 \
  --progress-every 1 \
  --output "$tmpdir/fec6_pairs48_80_window16_margin_sweep.json"
```

Result:

- verdict: `FAIL_MLX_TORCH_SCORER_PARITY_SWEEP`
- covered_pair_window: `[48, 80]`
- window_count: `2`
- failed_windows: `2`
- segnet_argmax_mismatch_pixels_total: `2`
- segnet_argmax_diff_fraction max: `3.178914388020833e-07`
- mismatch_min_top2_margin max: `7.152557373046875e-06`
- mismatch_min_top2_margin mean: `4.0531158447265625e-06`
- mismatch_logit_abs_delta_max max: `0.0001277923583984375`

Failed window `[48, 64]`:

- mismatch pixel: sample `6`, `y=183`, `x=299`
- PyTorch class: `0`
- MLX class: `1`
- PyTorch top-2 margin: `9.5367431640625e-07`
- MLX top-2 margin: `7.62939453125e-06`
- mismatch-pixel max logit delta: `5.14984130859375e-05`

Failed window `[64, 80]`:

- mismatch pixel: sample `11`, `y=177`, `x=286`
- PyTorch class: `2`
- MLX class: `0`
- PyTorch top-2 margin: `7.152557373046875e-06`
- MLX top-2 margin: `3.6716461181640625e-05`
- mismatch-pixel max logit delta: `0.0001277923583984375`

## Full-300 Blocker Window Anchor

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/audit_mlx_scorer_torch_parity_sweep.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --repo-root . \
  --device cpu \
  --start-pair 156 \
  --max-pairs 4 \
  --window-pairs 4 \
  --run-id fec6_pr101_pair156_160_cpu_window4_margin_20260522 \
  --progress-every 1 \
  --output experiments/results/mlx_scorer_torch_parity_sweep_fec6_pr101_pair156_160_20260522T_margin.json
```

Result:

- verdict: `FAIL_MLX_TORCH_SCORER_PARITY_SWEEP`
- covered_pair_window: `[156, 160]`
- window_count: `1`
- failed_windows: `1`
- segnet_argmax_mismatch_pixels_total: `1`
- segnet_argmax_diff_fraction: `1.2715657552083333e-06`
- segnet_logit_abs_max: `0.00039064884185791016`
- posenet_output_abs_max: `3.814697265625e-06`
- posenet_component_abs_max: `2.433423905087717e-12`
- mismatch_min_top2_margin: `9.5367431640625e-07`
- mismatch_logit_abs_delta_max: `3.147125244140625e-05`

Failed window `[156, 160]`:

- mismatch pixel: sample `1`, `y=180`, `x=335`
- PyTorch class: `3`
- MLX class: `2`
- PyTorch top-2 margin: `1.0013580322265625e-05`
- MLX top-2 margin: `9.5367431640625e-07`
- mismatch-pixel max logit delta: `3.147125244140625e-05`

## Layerwise Trace Anchor

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/trace_mlx_segnet_layer_parity.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --repo-root . \
  --device cpu \
  --start-pair 156 \
  --max-pairs 4 \
  --run-id fec6_pr101_pair156_160_cpu_layer_trace_20260522 \
  --output experiments/results/mlx_segnet_layer_trace_fec6_pr101_pair156_160_20260522T.json
```

Result:

- schema: `mlx_segnet_layer_trace.v1`
- pair_window: `[156, 160]`
- trace_count: `218`
- segnet_argmax_diff_pixels: `1`
- segnet_argmax_diff_fraction: `1.2715657552083333e-06`
- first drift cliff above `1.0e-4`: `encoder.stage_0.block_0.bn2`
- `encoder.stem` max_abs_delta: `1.33514404296875e-05`
- `encoder.stage_0.block_0.conv_pw` max_abs_delta: `3.6716461181640625e-05`
- `encoder.stage_0.block_0.bn2` max_abs_delta: `0.00026702880859375`
- `encoder.stage_0.block_0.bn2` p95_abs_delta: `6.151199340820312e-05`
- `encoder.stage_0.block_0.bn2` p99_abs_delta: `8.0108642578125e-05`

Interpretation: the first cliff appears immediately after the first
EfficientNet stage-0 block's second BatchNorm adapter, after a small stem and
conv drift. That points the next debugger at BatchNormAct2d/BatchNorm eval
semantics and EfficientNet stage-0 residual propagation rather than decoder
upsampling or the segmentation head.

Interpretation: MLX logits are close, but exact `argmax` parity is not
guaranteed when upstream PyTorch logits are effectively tied at a pixel. The
MLX scorer port remains useful for local research signal and candidate
generation priors; it is not a scorer replacement and not promotion/rank
authority.

## CUDA Determinism Notes

Upstream `evaluate.py` does not set deterministic PyTorch flags, CUDNN flags,
or TF32 policy. It runs `DistortionNet().eval().to(device)` under
`torch.inference_mode()`. CUDA uses `DaliVideoDataset` for input video decode
and then PyTorch CUDA kernels for interpolation, PoseNet, SegNet, and exact
SegNet `argmax` scoring. The local `experiments/contest_auth_eval.py` wrapper
sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `DALI_DISABLE_NVML=1`, but it still
delegates scoring to the upstream evaluator.

Therefore CUDA should be treated as the contest-authoritative axis when run
through the exact auth-eval path, not as a bitwise mathematical reference for
MLX. CUDA/CPU/MLX can differ at low-margin SegNet pixels; paired exact CUDA
auth eval remains required for score claims and promotion.

## Recommended Fix Path

- Keep strict zero-argmax-diff parity as the default gate for any cache window
  consumed as a hard scorer signal.
- Add a research-only override path that records mismatch-margin summaries and
  keeps `score_claim=false`, `promotion_eligible=false`,
  `rank_or_kill_eligible=false`, and
  `requires_exact_eval_before_promotion=true`.
- For candidate generation, discount or route to PyTorch/contest eval when
  SegNet top-2 margins are below the observed implementation-delta envelope.
- Next structural debug: layerwise SegNet parity on failing samples to separate
  expected backend floating-point accumulation drift from an adapter-specific
  mismatch in interpolation, padding, convolution, activation, or normalization.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_mlx_scorer_torch_parity.py
# 10 passed in 3.27s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check \
  src/tac/local_acceleration/mlx_scorer_torch_parity.py \
  src/tac/tests/test_mlx_scorer_torch_parity.py \
  tools/audit_mlx_scorer_torch_parity.py \
  tools/audit_mlx_scorer_torch_parity_sweep.py \
  tools/trace_mlx_segnet_layer_parity.py
# All checks passed!

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/tests/test_plan_ll_scorer_response_next_cli.py \
  src/tac/tests/test_mlx_execution_plan.py \
  src/tac/tests/test_mlx_profile_stability.py \
  src/tac/tests/test_profile_mlx_scorer_response_cache.py \
  src/tac/tests/test_mlx_scorer_response.py \
  src/tac/tests/test_mlx_batch_invariance.py \
  src/tac/tests/test_mlx_scorer_fidelity.py \
  src/tac/tests/test_mlx_scorer_torch_parity.py \
  src/tac/tests/test_mlx_production_contract.py \
  src/tac/tests/test_mlx_scorer_adapters.py \
  src/tac/tests/test_mlx_preprocess.py \
  src/tac/tests/test_mlx_cache_audit.py \
  src/tac/tests/test_mlx_scorer_state_map.py \
  src/tac/tests/test_mlx_scorer_port_inventory.py \
  src/tac/tests/test_mlx_to_pytorch_export.py
# 154 passed in 19.84s
```
