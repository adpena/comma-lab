# L5 v2 Measurement Schedule And Dispatch Plan Refresh

- date: `2026-05-16`
- agent: `codex`
- score_claim: `false`
- promotion_eligible: `false`
- dispatch_attempted: `false`

## Why

After the probe-intake custody normalization, the L5-v2 probe gate no longer reports stale TT5L CUDA blockers for missing CUDA device, CUDA log path, contest evidence grade, or absent-axis runtime SHA. The measurement schedule and paired dispatch plan are derived control-plane artifacts, so they were regenerated to preserve no-signal-loss alignment.

## Refreshed Artifacts

- `.omx/research/l5_v2_lattice_measurement_schedule_20260516_codex.json`
- `.omx/research/l5_v2_paired_measurement_dispatch_plan_20260516_codex.json`
- `.omx/research/l5_v2_paired_measurement_dispatch_plan_20260516_codex.md`

The paired dispatch plan remains planning-only with `ready_work_unit_count=0`.

## TT5L Remaining Measurement Blockers

- `l5_v2_probe_predicate_failed`
- `l5_v2_probe_paired_exact_axes_missing`
- `l5_v2_probe_sideinfo_consumption_missing`
- `l5_v2_probe_axis_evidence_missing:contest_cpu`
- `l5_v2_probe_axis_score_delta_missing:contest_cuda`

These are now the actionable TT5L blockers for the L5-v2 staircase, plus the dispatch-template blockers requiring a byte-closed archive path, archive SHA-256, submission runtime, and explicit operator execution flag before spend.
