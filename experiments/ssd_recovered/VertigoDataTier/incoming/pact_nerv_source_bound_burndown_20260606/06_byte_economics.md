# 06 — Byte economics: required value-per-byte telemetry

## Byte price

`p_byte = 25/37,545,489 = 6.658589531221714e-7`.

Every byte added must buy more score reduction than this, with uncertainty.

## Required actuator telemetry

Each actuator must emit:

```json
{
  "section": "head_rgb_1.weight",
  "actuator": "hinerv_target_region_birth",
  "delta_bytes_estimate": 0,
  "delta_bytes_measured": null,
  "delta_seg_score_units": -0.24,
  "delta_pose_score_units": 0.03,
  "delta_rate_score_units": 0.0,
  "delta_total_score_units": -0.21,
  "value_per_byte_score_units": null,
  "authority": "train_time_proxy_false_authority",
  "parseback_required": true
}
```

If `delta_bytes_measured` is unknown, the actuator is not byte-closed and cannot authorize promotion/long continuation beyond the short gate.

## Section byte ledger

Sections:

- `receiver_runtime`
- `header`
- `decoder_state`
- `latents_coarse`
- `latents_mid`
- `latents_fine`
- `feature_grids`
- `head_rgb_0`
- `head_rgb_1`
- `snerv_lf_payload`
- `snerv_hf_payload`
- `snerv_mfu_hfr_tub_weights`
- `codebooks`
- `entropy_model`
- `target_region_residual_atoms`
- `pose_yuv6_atoms`

## Rule

A section survives if:

`-Delta S_distortion / Delta bytes > p_byte * safety_factor`.

Recommended `safety_factor = 2.0` for local false-authority smokes.

## Patch target

`src/tac/analysis/nerv_section_value_ledger.py`

Add reusable schema and helper:

```python
def compose_section_value_row(
    *,
    section: str,
    bytes_before: int,
    bytes_after: int,
    seg_before: float,
    seg_after: float,
    pose_before: float,
    pose_after: float,
    pose_marginal: float,
) -> dict:
    ...
```
