# Codex Session Summary - Exact Receiver Projection Trace

UTC: 2026-06-06T16:47:24Z

## Landing

Extended the live MLX score-aware step guard in
`src/tac/substrates/_shared/mlx_score_aware/adapter.py` from receipt-only
telemetry to exact local receiver projection.

Accepted guarded updates now capture a pre-update receiver snapshot and compare
it with the final accepted model state after full-step or backtracking
acceptance. Rejected and ineligible steps still do not emit successful receiver
motion.

Exact local fields now emitted when available:

- `receiver_surface_float_rgb_delta_linf`
- `receiver_surface_uint8_changed_pixels`
- `receiver_surface_uint8_changed_channel_values`
- `receiver_surface_segnet_input_delta_linf`
- `receiver_surface_argmax_flipped_pixels` when a live SegNet teacher is wired
- `receiver_surface_fakequant_argmax_flipped_pixels` when eval-roundtrip STE and
  a live SegNet teacher are wired
- `receiver_surface_worst_region_margin_p50_delta` from exact 4-connected
  target-component margins, with class id and component-size metadata
- `receiver_surface_posenet_input_delta_linf`
- `receiver_surface_posenet_output_delta_linf` when a live PoseNet teacher is wired
- `receiver_surface_posenet_output_delta_l2_mean` when a live PoseNet teacher is wired

The trace still preserves score-unit movement from the step guard:

- `receiver_surface_loss_delta`
- `receiver_surface_nonrate_score_unit_movement`
- `receiver_surface_pose_score_unit_movement`
- direct-live Pose marginal pre/post aliases

`src/tac/analysis/nerv_crux_trace.py` now consumes only canonical
`receiver_surface_*` fields for this contract. Short aliases and pair-local
HiNeRV smoke fields do not satisfy `receiver_surface_trace_present`.

## Fail-Closed State

This closes the previous local-MLX `uint8_changed_pixels` evidence gap. The
trace consumer now moves an improving accepted step with zero byte motion from:

`receiver_surface_loss_improved_without_uint8_evidence`

to:

`receiver_surface_loss_improved_without_uint8_motion`

The following surfaces remain intentionally blocked until wired:

- parse-back survival producers
- inflate survival producers
- archive-byte/value-per-byte authority

The trace consumer already fails closed for missing fakequant, parse-back, or
inflate survival once an upstream receiver argmax stage reports motion.

## Lane Registry

Registered `lane_live_mlx_exact_receiver_projection_20260606` at L0 and marked
`impl_complete`, producing L1 only. No exact archive, CPU, CUDA, or strict
preflight gates were claimed.

## Verification

- `uv run ruff check src/tac/analysis/nerv_crux_trace.py src/tac/tests/test_trace_nerv_crux.py src/tac/substrates/_shared/mlx_score_aware/adapter.py src/tac/substrates/_shared/mlx_score_aware/tests/test_loss_adapter_harness.py`
- `uv run pytest src/tac/tests/test_trace_nerv_crux.py -q`
- `uv run pytest src/tac/substrates/_shared/mlx_score_aware/tests/test_loss_adapter_harness.py -q`
- Synthetic `build_trace_rows(...)` probe confirmed the blocker now becomes
  `receiver_surface_loss_improved_without_uint8_motion` when exact uint8
  evidence is present but zero.
- Synthetic `build_trace_rows(...)` probe confirmed
  `receiver_surface_fakequant_lost_argmax_motion` when live argmax flips are
  lost by fakequant.
- Focused adapter coverage forces disconnected same-class target regions so the
  worst-region p50 path cannot regress to class-aggregate margin math.
- Focused trace coverage rejects short receiver-surface aliases and pair-local
  smoke fields as mandatory receiver trace evidence.

## Next Concrete Step

Wire parse-back/inflate survival producers, then feed the same canonical
receiver receipt into campaign admission/preflight so a long MLX row cannot
launch on local loss movement alone.
