# L5 v2 TT5L side-info effect-curve harvest cell builder

Date: 2026-05-17

## Classification

- Lane: `l5_v2_tt5l_sideinfo_effect_curve`
- Axis: paired `[contest-CPU]` + `[contest-CUDA]`
- Score claim: `false`
- Promotion eligible: `false`
- Dispatch attempted: `false`
- Artifact advanced: post-harvest cell builder for TT5L side-info effect curve

## Gap

The Lightning paired-axis plan already carries the identity needed for every
TT5L side-info effect-curve cell: `axis`, `variant`, `archive_sha256`,
`pair_group_id`, `run_id`, `local_artifact_dir`, and the source variant
manifest. The final effect-curve builder already validates exact-eval custody,
paired CPU/CUDA identity, side-info liveness, and the trained-vs-control
predicate.

The missing operator-safe step was the adapter between those surfaces. Without
it, a human or future agent could hand-assemble `--cell-json` and accidentally
drop or remix `pair_group_id` / `run_id` before the aggregate effect curve.

## Landing

- Added `src/tac/optimization/l5_v2_tt5l_sideinfo_effect_curve_harvest.py`.
- Added `tools/build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan.py`.
- Reused `exact_eval_evidence_from_auth_eval_artifact()` from
  `src/tac/optimization/l5_v2_probe_intake.py` so side-info effect-curve
  harvests share the same exact-eval custody extraction semantics as L5 v2
  probe intake.
- Added focused tests in
  `src/tac/tests/test_l5_v2_tt5l_sideinfo_effect_curve_harvest.py`.

## Current artifact

Generated:

`.omx/research/l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.json`

Summary:

- `cell_count=10`
- `harvested_exact_eval_artifact_count=0`
- `missing_exact_eval_artifact_count=10`
- Each missing cell still preserves `axis`, `variant`, `archive_sha256`,
  `pair_group_id`, `run_id`, and source side-info liveness.

This is intentionally fail-closed. It is not a negative TT5L result; it records
that the paired exact-eval artifacts have not yet been harvested into this
effect-curve surface.

## Validation

Commands:

```bash
.venv/bin/python -m pytest src/tac/tests/test_l5_v2_tt5l_sideinfo_effect_curve_harvest.py -q
.venv/bin/python -m ruff check src/tac/optimization/l5_v2_probe_intake.py src/tac/optimization/l5_v2_tt5l_sideinfo_effect_curve_harvest.py tools/build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan.py src/tac/tests/test_l5_v2_tt5l_sideinfo_effect_curve_harvest.py
.venv/bin/python tools/build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan.py --lightning-plan-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_20260517_codex.json --output-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.json --repo-root .
```

Observed:

- `3 passed`
- `All checks passed!`
- Builder output: `cell_count=10`, `harvested_exact_eval_artifact_count=0`,
  `missing_exact_eval_artifact_count=10`, `score_claim=false`,
  `promotion_eligible=false`.

## Next gate

After paired exact-eval artifacts exist under the Lightning plan
`local_artifact_dir` values, rerun the harvest-cell builder and then feed the
generated cell JSON into:

```bash
.venv/bin/python tools/build_l5_v2_sideinfo_effect_curve.py \
  --cell-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.json \
  --output-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json \
  --repo-root .
```

The architecture lock remains blocked until the aggregate effect curve validates
all 10 paired cells and the trained variant beats or ties every control on both
contest axes.
