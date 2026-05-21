# Codex Findings: MLX Scorer State-Map Conformance Scaffold

timestamp_utc: 2026-05-21T21:57:13Z
agent: codex
lane_id: lane_mlx_auth_scorer_training_signal_fidelity_20260521
evidence_grade: macOS-MLX-research-signal
score_claim: false
promotion_eligible: false

## Summary

Codex landed a key-level PyTorch-to-MLX state-dict layout map for the upstream contest scorer. This is the next conformance scaffold after the module/op inventory: a faithful MLX scorer port must map every upstream safetensor/state-dict key into an MLX tensor name with shape, dtype, layout, and transform policy.

New reusable surfaces:

- `src/tac/local_acceleration/mlx_scorer_state_map.py`
- `tools/plan_mlx_scorer_state_map.py`
- `src/tac/tests/test_mlx_scorer_state_map.py`

Generated state maps:

- `experiments/results/mlx_scorer_state_map_20260521.json`
- `experiments/results/mlx_scorer_state_map_with_weights_20260521.json`

## Empirical State-Key Result

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/plan_mlx_scorer_state_map.py \
  --repo-root . \
  --load-weights \
  --output experiments/results/mlx_scorer_state_map_with_weights_20260521.json
```

Result:

```json
{
  "state_key_count": 1160,
  "mapped_or_intentionally_unused_key_count": 1160,
  "hard_unmapped_key_count": 0,
  "requires_module_adapter_key_count": 891,
  "status_counts": {
    "mapped_tensor_transform": 103,
    "requires_module_adapter": 891,
    "unused_eval_buffer": 166
  },
  "transform_policy_counts": {
    "batchnorm_1d_vector_identity": 184,
    "conv2d_weight_oihw_to_ohwi": 203,
    "drop_eval_only_num_batches_tracked": 166,
    "identity": 593,
    "linear_weight_identity_out_in": 14
  }
}
```

Interpretation: every upstream state key now has a declared layout policy, but 891 keys still require module-adapter parity before the MLX scorer can be called a 1:1 scorer. `num_batches_tracked` is explicitly classified as an eval-only PyTorch BatchNorm training counter and intentionally omitted only under eval-mode parity tests.

## Conformance Impact

This moves the MLX scorer path from vague architecture coverage to concrete state custody:

- Conv2d weights are declared `OIHW -> OHWI`, matching MLX Conv2d layout.
- Linear weights are declared identity `out,in`.
- BatchNorm vectors are declared identity, but blocked on eval running-stat parity.
- All mappings remain non-authoritative: `score_claim=false`, `promotion_eligible=false`, and `full_mlx_port_claim_allowed=false`.

## Next Action

Implement the first adapter-parity layer tests in this order:

1. Conv2d grouped/depthwise OIHW-to-OHWI parity on fixed tensors.
2. BatchNorm eval running-stat parity.
3. `torch.nn.functional.interpolate(..., mode="bilinear")` scorer-preprocess parity against the hash bridge.
4. PoseNet stem block parity, then SegNet first encoder block parity.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_scorer_state_map.py \
  src/tac/tests/test_mlx_scorer_port_inventory.py \
  -q
```

Observed: `5 passed`.
