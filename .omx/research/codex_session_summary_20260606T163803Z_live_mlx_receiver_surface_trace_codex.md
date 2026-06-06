# Codex Session Summary - Live MLX Receiver-Surface Trace

UTC: 2026-06-06T16:38:03Z

## Landing

Implemented producer-side receiver-surface trace emission from the live MLX
score-aware step guard in
`src/tac/substrates/_shared/mlx_score_aware/adapter.py`.

Accepted guarded updates now emit:

- `receiver_surface_source_live_mlx_step_guard`
- `receiver_surface_update_accepted`
- `receiver_surface_loss_delta`
- `receiver_surface_nonrate_score_unit_movement`
- `receiver_surface_pose_output_delta`
- `receiver_surface_pose_score_unit_movement`
- `receiver_surface_segnet_score_debt_delta`
- `receiver_surface_worst_region_margin_proxy_delta`
- direct-live Pose score marginal pre/post aliases when present

Rejected or ineligible steps do not emit successful receiver evidence.

## Fail-Closed State

The live adapter still does not compute exact float-RGB delta, uint8 changed
pixels, SegNet input delta, argmax flip counts, fakequant survival, parse-back
survival, or inflate survival inside the train step. Accepted updates therefore
emit explicit `*_evidence_missing` flags for those required receiver surfaces.

The existing `tac.analysis.nerv_crux_trace` consumer sees
`receiver_surface_loss_delta < 0` without exact uint8 evidence as:

`receiver_surface_loss_improved_without_uint8_evidence`

This preserves launch blocking until exact receiver-surface projection is wired.

## Verification

- `uv run ruff check src/tac/substrates/_shared/mlx_score_aware/adapter.py src/tac/substrates/_shared/mlx_score_aware/tests/test_loss_adapter_harness.py`
- `uv run pytest src/tac/substrates/_shared/mlx_score_aware/tests/test_loss_adapter_harness.py -k "prices_target_region_score_debt_before_argmax_proxy" -q`
- `uv run pytest src/tac/substrates/_shared/mlx_score_aware/tests/test_loss_adapter_harness.py -q`
- `uv run pytest src/tac/tests/test_trace_nerv_crux.py -q`
- Synthetic `build_trace_rows(...)` probe confirmed
  `receiver_surface_loss_improved_without_uint8_evidence`.

## Next Concrete Step

Wire exact receiver projection into the adapter trace:

`float_rgb -> clamp/round STE -> official scorer preprocess -> SegNet/PoseNet`

Then fill exact `receiver_surface_uint8_changed_pixels`,
`receiver_surface_segnet_input_delta_linf`,
`receiver_surface_argmax_flipped_pixels`, and
`receiver_surface_posenet_input_delta_linf` before the first HiNeRV hard-birth
smoke is admitted.
