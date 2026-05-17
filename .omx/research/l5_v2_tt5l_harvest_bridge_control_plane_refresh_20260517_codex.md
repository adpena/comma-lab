# L5 v2 TT5L harvest bridge control-plane refresh - 2026-05-17

## Summary

Refreshed the TT5L side-info harvest-cell Markdown and the L5 v2
architecture-lock packet after the landed harvest bridge.

## Finding

The committed harvest bridge implementation is sound under focused tests, but
the generated Markdown status in the landing commit had been produced from a
test fixture plan:

- source plan: `.omx/research/lightning_paired_axis_plan.json`
- ready cells: `1/10`
- artifact root: `experiments/results/lightning_batch/test_tt5l_paired_axes`

That is not the live TT5L Lightning paired-axis plan. The live control-plane
refresh now records:

- source plan:
  `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_20260517_codex.json`
- ready cells: `0/10`
- harvested exact-eval artifacts: `0`
- missing exact-eval artifacts: `10`
- artifact root:
  `experiments/results/lightning_batch/l5_v2_tt5l_sideinfo_effect_curve_paired_axes`

## Architecture Lock Packet

The refreshed architecture-lock packet remains `architecture_lock_allowed=false`.
It also correctly marks the paired-axis Lightning plan source custody as not
current for execution because a source-relevant path changed after the plan's
recorded source commit:

- changed relevant path: `scripts/launch_lightning_batch_job.py`
- blocker added:
  `l5_v2_tt5l_lightning_paired_axis_plan_source_relevant_paths_changed`

This is conservative and correct: the side-info effect-curve dispatch plan must
be refreshed or explicitly revalidated before non-dry-run execution.

## Root-Cause Fix

The harvest-cell CLI now writes its Markdown report beside a noncanonical
`--output-json` path unless `--output-md` is explicitly provided. This prevents
tests or ad hoc temporary invocations from writing fixture-derived Markdown into
the canonical `.omx/research` report path.

## Verification

- `.venv/bin/python tools/build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan.py --lightning-plan-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_20260517_codex.json --output-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.json --output-md .omx/research/l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.md --repo-root .`
- `.venv/bin/python tools/build_l5_v2_architecture_lock_packet.py --repo-root . --output-json .omx/research/l5_v2_architecture_lock_packet_20260516_codex.json --output-md .omx/research/l5_v2_architecture_lock_packet_20260516_codex.md`
- `.venv/bin/python -m pytest src/tac/tests/test_l5_v2_tt5l_sideinfo_effect_curve_harvest.py src/tac/tests/test_l5_v2_sideinfo_effect_curve.py src/tac/tests/test_l5_v2_tt5l_lightning_route_unblock_packet.py src/tac/tests/test_l5_v2_tt5l_sideinfo_lightning_execution_bundle.py src/tac/tests/test_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run.py src/tac/tests/test_l5_v2_tt5l_sideinfo_lightning_execution_preflight.py -q`
  -> `44 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py -q`
  -> `134 passed`
- `.venv/bin/ruff check src/tac/optimization/l5_v2_tt5l_sideinfo_effect_curve_harvest.py tools/build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan.py src/tac/tests/test_l5_v2_tt5l_sideinfo_effect_curve_harvest.py tools/build_l5_v2_architecture_lock_packet.py`
  -> clean

## Score And Dispatch Status

- No provider dispatch was launched.
- No archive was built.
- No score claim is made.
- Next score-lowering gate remains the real paired CPU/CUDA TT5L side-info
  harvest after source-custody revalidation.
