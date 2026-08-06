# ddm_bd1 Next If Resumed

## Current State

Code is landed in two commits:

- `c478dd1712` - `a1_gate` realized dpose telemetry.
- `b7056a8ca7` - endpoint `--emit-error-atlas` sidecars.

This receipt file is the durable handoff artifact for both units. No exact score was produced and no run was launched.

## Fire Order

1. On the next live TR1/JD7-compatible `a1_gate`, inspect telemetry for:
   - `realized_gate_dpose_mean`
   - `realized_gate_dpose_per_pair`
   - `realized_gate_dpose_q50/q90/q95` naming as emitted in code: `realized_gate_dpose_per_pair_q50`, `realized_gate_dpose_per_pair_q90`, `realized_gate_dpose_per_pair_q95`
   - `realized_gate_dpose_wall_seconds`
2. Treat the trainer dpose channel as advisory trend only. Do not promote it over the n600 endpoint probe.
3. On the next governed endpoint probe window, rerun the existing endpoint command with `--emit-error-atlas`.
4. Preserve the generated manifest and both sidecars:
   - `<out_stem>.error_atlas_manifest.json`
   - `<out_stem>.error_atlas.ema.npz`
   - `<out_stem>.error_atlas.live.npz`
5. Use manifest shas plus `field_shape` to feed downstream row-local disagreement analysis. The sidecar boolean is `realized != lstar` in raster order after endpoint adapter SegNet argmax.

## Guardrails

- No rerun is needed to validate default-off schema stability; it is covered by `test_emit_error_atlas_defaults_off_for_receipt_schema_stability`.
- Do not call the trainer dpose field an exact score, a pointer move, or n600 authority.
- Do not add a separate live-basis gate pass for this channel unless a later charter explicitly opens that cost.
- Do not edit or terminate live run dirs to harvest this telemetry.
- If Metal remains unavailable, leave wall-clock status as blocked and wait for a live gate emission or a hardware-authorized probe.

## Latest Pointer Honesty

Own-vehicle frontier remains `S = 0.7537933983374265 @ 357,837 B [macOS-CPU advisory]`.
Borrowed contest pointer remains `0.19108 [contest-CPU]` unmoved.
