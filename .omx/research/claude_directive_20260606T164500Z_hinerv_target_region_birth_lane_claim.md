# Claude directive — lane claim: HiNeRV target-region birth actuator

UTC: 2026-06-06T16:45:00Z
Agent: claude (main session, Opus 4.8)
Lane: `lane_hinerv_target_region_score_debt_smoke_20260606` (registered L0)
Backlog item: `hinerv_target_region_birth_actuator` (priority 2,
`/Volumes/VertigoDataTier/pact/incoming/pact_nerv_source_bound_burndown_20260606/implementation_backlog.json`)

## Scope claimed by claude

- NEW `src/tac/substrates/hi_nerv/target_region_birth.py` — torch-free
  connected-region score-debt selection + scoped-update-name predicate +
  receipt builders (crux-trace-compatible receiver-surface keys).
- NEW method `fit_target_region_birth_from_segnet(...)` appended to
  `src/tac/substrates/hi_nerv/mlx_renderer.py` (additive only; no edits to
  existing methods).
- NEW `src/tac/substrates/hi_nerv/tests/test_target_region_birth.py`.

## Explicitly NOT claimed (sister codex owns, in flight 2026-06-06)

- `src/tac/analysis/nerv_pair_local_distortion_servo.py` + its test
  (untracked, actively edited by codex ~16:36Z).
- `src/tac/analysis/nerv_crux_trace.py`, `src/tac/score_geometry.py`,
  `src/tac/contest_eval_contract.py`, `tools/trace_nerv_crux.py`,
  `tools/run_compact_renderer_mlx_spine_runner.py`,
  `tools/cathedral_autopilot.py`, `src/tac/analysis/nerv_source_boundary_audit.py`
  and all currently-modified test files (codex uncommitted working set).
- SNeRV TUB source-forward burndown
  (`lane_snerv_source_forward_proof_burndown_20260606`, codex at v61).

## Interaction contract

The new renderer method emits receiver-surface keys using the alias names
`receiver_surface_uint8_changed_pixels`, `receiver_surface_argmax_flipped_pixels`,
`receiver_surface_worst_region_margin_p50_delta`,
`receiver_surface_float_rgb_delta_linf` so the sister crux-trace consumer and
pair-local servo admission kernel can ingest the rows without adaptation.
No imports from the sister's uncommitted module are introduced.

## Authority

Local $0 MLX/CPU work only. No paid dispatch. Non-promotable
`[macOS-MLX research-signal]` / planning-control evidence only.
