# Codex Session Summary - HiNeRV Wall-Normal Archive Birth Support - 2026-06-13T13:01:06Z

## Scope

Materialized archive-closed target-region birth support from the true wall-normal teacher receipt:

`/Volumes/VertigoDataTier/pact/experiments/results/hinerv_target_region_birth_real_smoke_20260606/hinerv_witness_readiness_short_smoke_v41_lateall_wall_normal_forced_region_20260607T094500Z/hi_nerv_mlx_training/target_region_wall_normal_lift_receipt.json`

## Code Landed

- `tools/run_compact_renderer_mlx_spine_runner.py`
  - Derived legacy wall-normal archive-executable support from the true wall-normal telemetry candidate instead of inheriting generic sidecar/mask support.
  - Preserved `direct_teacher_mask_support_sha256` separately from archive-executable support.
  - Added target-region action inflate materialization and threaded inflated raw output into parse-back survival so the producer can emit `parseback_survived` and `inflate_survived`.
- `src/tac/tests/test_compact_renderer_mlx_spine_runner.py`
  - Added a legacy v41 selector regression for true wall-normal action support derivation.

## False-Authority Artifacts

- Storage preflight:
  `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_wall_normal_archive_birth_support_20260613T124608Z_storage_preflight.json`
- Materialization summary:
  `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_wall_normal_archive_birth_support_materialized_20260613T125428Z/wall_normal_archive_birth_support_materialization_summary.json`
- Archive:
  `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_wall_normal_archive_birth_support_materialized_20260613T125428Z/archive_closed_wall_normal_action/archive.zip`
- Parse-back survival:
  `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_wall_normal_archive_birth_support_materialized_20260613T125428Z/archive_closed_wall_normal_action/hi_nerv_target_region_action_parseback_survival.json`
- Inflate materialization:
  `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_wall_normal_archive_birth_support_materialized_20260613T125428Z/archive_closed_wall_normal_action/hi_nerv_target_region_action_inflate_materialization.json`

Key receipt fields:

- Archive bytes: `115993`
- Archive SHA-256: `633ba4e4e13c17b22d4e84293eacb4ac093cbb09f5b3c7c2c806db136fd6ab13`
- Archive-executable support SHA-256: `c7363bfe996759d9309b8f6029aaa881aa76e8483fd2f8812a6752866f77d8ca`
- Direct teacher mask support SHA-256: `2265a2140bfbccc24881b0c9d2e32150cf8c617cab872fc41cfbb5bb99dd8933`
- Parse-back survived: `true`
- Inflate survived: `true`
- Inflated raw SHA-256: `9addb9b7eb27803fec8fb266d1ce212f72fa40ea03a6734eb63295b9e2a01341`
- Score claim: `false`
- Promotion eligible: `false`
- Ready for exact eval dispatch: `false`

## Bounded Rerun Status

The bounded MLX rerun was attempted through the runner, but the foreground attempt was interrupted before export and the detached retry exited without a report. The archive support was then materialized through the bounded archive-side producer path using the v41 true wall-normal evidence and false-authority planning control fields.

Dispatch claim terminal status was recorded as `completed_materialized_archive_support_false_authority` for `lane_hinerv_wall_normal_archive_birth_support_20260613`.

## Verification

Passed:

```bash
.venv/bin/python -m pytest src/tac/tests/test_compact_renderer_mlx_spine_runner.py -k 'hinerv_action_program_selection'
.venv/bin/python -m pytest src/tac/tests/test_hinerv_target_region_action_comparison.py -k 'archive_executable_direct_support or cli_materializes_inflate'
.venv/bin/python -m pytest src/tac/substrates/hi_nerv/tests/test_target_region_birth.py -k 'fakequant_survival_requirement_controls_acceptance'
```

## Remaining Blocker

This is archive-closed parse-back/inflate survival only. Promotion remains blocked by scorer-effect/fakequant/exact-eval authority surfaces; no score movement or rank claim is admitted from this materialization.
