# Codex Session Summary — NeRV Receiver-Surface Trace

UTC: 2026-06-06T16:25:48Z

## Scope

Converted the PR95-vs-HiNeRV/SNeRV "loss moved but receiver surface did not"
diagnosis into an executable crux-trace contract. The existing
`tools/trace_nerv_crux.py` path now emits receiver-surface rows alongside
SegNet, PoseNet, and rate rows.

## Landed Contracts

- Added receiver-surface trace extraction for:
  `float_rgb_delta_linf`, `uint8_changed_pixels`,
  `segnet_input_delta_linf`, `worst_region_margin_p50_delta`,
  `argmax_flipped_pixels`, `posenet_input_delta_linf`, `pose_output_delta`,
  `fakequant_argmax_flipped_pixels`, `parseback_argmax_flipped_pixels`, and
  `inflated_argmax_flipped_pixels`.
- Made missing receiver-surface trace evidence fail closed by default, with an
  explicit `--allow-missing-receiver-surface-trace` forensic opt-out for old
  artifacts.
- Added the two hard blockers from the PR95 control-loop diagnosis:
  `receiver_surface_loss_improved_without_uint8_motion` and
  `receiver_surface_uint8_motion_without_argmax_or_margin_motion`.
- Added survival blockers for fakequant, parse-back, and inflate surfaces when
  earlier argmax motion disappears.
- Registered lane `lane_nerv_receiver_surface_crux_trace_20260606` and marked
  `impl_complete` with `src/tac/analysis/nerv_crux_trace.py`.

## Verification

- `uv run ruff check src/tac/analysis/nerv_crux_trace.py tools/trace_nerv_crux.py src/tac/tests/test_trace_nerv_crux.py`
- `uv run pytest src/tac/tests/test_trace_nerv_crux.py -q`
- `uv run python tools/lane_maturity.py validate`
- `uv run pytest src/tac/tests/test_trace_nerv_crux.py src/tac/tests/test_pr95_distortion_practices_guard.py -q`

Results: `ruff` clean, `11 passed`, lane registry `1692 lane(s) validated
cleanly`, and combined trace plus PR95 guard suite `31 passed`.

## Authority Boundary

This is local diagnostic/control evidence only. It is not a score claim, not an
exact-eval replay, and not receiver-proof closure. It prevents a future
HiNeRV/SNeRV run from treating smooth loss movement as scorer-visible
actuation unless the update crosses the uint8/preprocess/argmax/parse-back
surfaces.

## Next Work

Wire producer-side receiver-surface trace emission into the live MLX update
path so accepted HiNeRV pair-adapter/birth-overlay updates and SNeRV MFU/HFR/TUB
modulation updates populate these rows automatically.
