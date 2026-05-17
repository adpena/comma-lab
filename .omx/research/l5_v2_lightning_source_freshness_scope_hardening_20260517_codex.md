# L5 v2 Lightning source freshness scope hardening - 2026-05-17

Scope: follow-up from the xhigh L5-v2 adversarial review. The review found that
the TT5L Lightning side-info paired-axis plan freshness check covered the
Lightning plan builder and deploy layer, but not the readiness interpreter that
decides whether the plan is current.

Evidence axis: custody/readiness hardening only. This is not a score claim, not
promotion evidence, and not dispatch authorization.

## Finding

The stale-plan freshness path did not include:

- `src/tac/optimization/l5_staircase_v2.py`
- `tools/build_l5_v2_architecture_lock_packet.py`
- `src/tac/exact_eval_custody.py`
- `src/tac/optimization/l5_v2_sideinfo_effect_curve.py`

That meant changes to the interpreter or exact-eval custody code could leave a
previous Lightning dry-run plan marked current if only the narrower plan-builder
path set was unchanged.

## Fix

`_TT5L_LIGHTNING_PAIRED_AXIS_STATIC_SOURCE_PATHS` now includes the readiness
interpreter, packet builder, exact-eval custody validator, and side-info
effect-curve validator. The architecture packet therefore treats the existing
Lightning side-info plan as structurally valid but stale for execution until it
is rebuilt under the current interpreter.

Live packet after regeneration:

- `artifact_valid=true`
- `source_relevant_paths_match=false`
- `source_relevant_diff_paths=['src/tac/optimization/l5_staircase_v2.py']`
- `source_custody_current_for_execution=false`
- `all_cells_dry_run_structurally_valid=true`
- `all_cells_dry_run_ready=false`

## Validation

Focused regression:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_readiness_surfaces_current_lightning_paired_axis_plan \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_lightning_paired_axis_plan_status_blocks_relevant_source_drift
```

Result: `2 passed`.
