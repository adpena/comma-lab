# Codex Premise Falsification: MLX BatchNorm Affine Repair

- UTC: 2026-05-22T01:15:39Z
- Lane: `lane_mlx_auth_scorer_training_signal_fidelity_20260521`
- Evidence axis: `[macOS-MLX research-signal]`
- Verdict: `FALSIFIED_AS_REPAIR`
- Score claim: false
- Promotion eligible: false
- Ready for exact eval dispatch: false

## Premise Tested

Hypothesis: replacing `mlx.nn.BatchNorm` with a deterministic eval-mode affine
BatchNorm transform would reduce the first SegNet drift cliff at
`encoder.stage_0.block_0.bn2` and remove or shrink the remaining full-300
strict parity blocker at pair window `[156, 160]`.

## Result

Falsified. The candidate still failed strict parity on `[156, 160]` with one
SegNet argmax pixel mismatch.

Generic MLX BatchNorm baseline:

- drift cliff: `encoder.stage_0.block_0.bn2`
- drift-cliff max_abs_delta: `0.00026702880859375`
- SegNet argmax diff pixels: `1`
- mismatch-pixel logit_abs_delta_max: `0.00003147125244140625`

Deterministic affine BatchNorm candidate:

- drift cliff: `encoder.stage_0.block_0.bn2`
- drift-cliff max_abs_delta: `0.000274658203125`
- SegNet argmax diff pixels: `1`
- mismatch-pixel logit_abs_delta_max: `0.00004291534423828125`

The deterministic affine path made the drift cliff slightly worse, so it was
not landed.

## Tooling Landed

Added a reusable comparison surface:

- `tac.local_acceleration.mlx_segnet_trace_compare.compare_mlx_segnet_layer_traces`
- `tools/compare_mlx_segnet_layer_traces.py`

This compares two `mlx_segnet_layer_trace.v1` manifests, remains
non-promotional by construction, and reports argmax-pixel change plus improved
and worsened trace rows.

## Comparison Anchor

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/compare_mlx_segnet_layer_traces.py \
  --baseline experiments/results/mlx_segnet_layer_trace_fec6_pr101_pair156_160_20260522T.json \
  --candidate experiments/results/mlx_segnet_layer_trace_fec6_pr101_pair156_160_20260522T_bn_affine.json \
  --baseline-label generic_mlx_batchnorm \
  --candidate-label deterministic_affine_batchnorm \
  --output experiments/results/mlx_segnet_layer_trace_compare_fec6_pr101_pair156_160_bn_affine_20260522T.json
```

Result:

- verdict: `TRACE_CANDIDATE_WORSENED_DRIFT`
- segnet_argmax_diff_pixels_change: `0`
- baseline drift cliff: `encoder.stage_0.block_0.bn2`, max_abs_delta `0.00026702880859375`
- candidate drift cliff: `encoder.stage_0.block_0.bn2`, max_abs_delta `0.000274658203125`

## Verification

```bash
.venv/bin/python -m ruff check \
  src/tac/local_acceleration/mlx_segnet_trace_compare.py \
  src/tac/tests/test_mlx_segnet_trace_compare.py \
  tools/compare_mlx_segnet_layer_traces.py
# All checks passed

.venv/bin/python -m pytest -q src/tac/tests/test_mlx_segnet_trace_compare.py
# 2 passed
```

Focused BatchNorm candidate checks also passed mechanically before the patch
was rejected as a repair:

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_mlx_scorer_adapters.py
# 37 passed
```

## Next Repair Target

Do not pursue the deterministic affine BatchNorm replacement as a parity fix.
The next useful probe is finer stage-0 arithmetic localization across
`conv_pw -> bn2`, especially Conv2d accumulation/layout behavior and the exact
input distribution into `bn2`, because the generic BN path is already slightly
closer than the affine replacement.
