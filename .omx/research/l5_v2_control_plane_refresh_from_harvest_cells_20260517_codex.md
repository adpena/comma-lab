# L5 v2 control-plane refresh from TT5L harvest cells

Date: 2026-05-17

## Classification

- Lane: `l5_v2_tt5l_sideinfo_effect_curve`
- Axis: paired `[contest-CPU]` + `[contest-CUDA]`
- Score claim: `false`
- Promotion eligible: `false`
- Dispatch attempted: `false`
- Purpose: refresh canonical L5 v2 control-plane artifacts through the new
  harvest-cell adapter instead of the stale partial/manual side-info surface.

## Finding

The canonical side-info effect-curve artifact was stale after the harvest-cell
builder landed. It preserved only a partial old surface, while the newer harvest
artifact represented all ten required cells:

- 5 variants: `zero`, `random_lsb`, `shuffled`, `trained`, `ablated`
- 2 axes: `[contest-CPU]`, `[contest-CUDA]`
- preserved per-cell `archive_sha256`, `pair_group_id`, `run_id`, and
  side-info liveness

This was a no-signal-loss gap, not a score result.

## Landing

- Refreshed `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json`
  from `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.json`.
- Refreshed `.omx/research/l5_v2_lattice_measurement_schedule_20260516_codex.json`
  and `.md`.
- Refreshed `.omx/research/l5_v2_architecture_lock_packet_20260516_codex.json`
  and `.md`.
- Wired the architecture-lock packet to surface
  `sideinfo_effect_curve_harvest_cells_status`.
- Updated the `measure_tt5l_sideinfo_effect_curve` command template to run:
  harvest-cell builder -> side-info effect-curve builder -> measurement schedule
  refresh.

## Current state

- Harvest cells: `cell_count=10`
- Harvested exact-eval artifacts: `0`
- Missing exact-eval artifacts: `10`
- Canonical effect curve: `observed_cell_count=10`, `predicate_passed=false`
- Architecture lock: `architecture_lock_allowed=false`
- Remaining architecture-lock blockers:
  - `requires_all_l5_v2_gate_evidence_valid`
  - `requires_c1_z5_tt5l_probe_gate_evidence`
  - `requires_paired_cpu_cuda_sideinfo_effect_curve`

This is intentionally fail-closed. It upgrades the control plane from
"partial stale evidence" to "complete 5x2 missing-exact-eval target map."

## Validation

Commands:

```bash
.venv/bin/python tools/build_l5_v2_sideinfo_effect_curve.py --cell-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.json --output-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json --repo-root .
.venv/bin/python tools/audit_l5_v2_probe_observations.py --output-json .omx/research/l5_v2_probe_observation_intake_20260516_codex.json --output-md .omx/research/l5_v2_probe_observation_intake_20260516_codex.md --probe-gate-out .omx/research/l5_v2_probe_gate_artifact_20260516_codex.json
.venv/bin/python tools/build_l5_v2_lattice_measurement_schedule.py --probe-intake-json .omx/research/l5_v2_probe_observation_intake_20260516_codex.json --sideinfo-effect-curve-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json --output-json .omx/research/l5_v2_lattice_measurement_schedule_20260516_codex.json --output-md .omx/research/l5_v2_lattice_measurement_schedule_20260516_codex.md --repo-root .
.venv/bin/python tools/build_l5_v2_architecture_lock_packet.py --repo-root . --output-json .omx/research/l5_v2_architecture_lock_packet_20260516_codex.json --output-md .omx/research/l5_v2_architecture_lock_packet_20260516_codex.md
.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py -q
.venv/bin/python -m pytest src/tac/tests/test_l5_v2_tt5l_sideinfo_effect_curve_harvest.py src/tac/tests/test_l5_v2_sideinfo_effect_curve.py src/tac/tests/test_l5_v2_probe_intake.py -q
.venv/bin/python -m ruff check src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py src/tac/optimization/l5_v2_probe_intake.py src/tac/optimization/l5_v2_tt5l_sideinfo_effect_curve_harvest.py tools/build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan.py src/tac/tests/test_l5_v2_tt5l_sideinfo_effect_curve_harvest.py
git diff --check
```

Observed:

- `129 passed` for L5 staircase v2
- `31 passed` for harvest, side-info effect-curve, and probe-intake tests
- `ruff`: all checks passed
- `git diff --check`: clean

## Next concrete gate

The side-info effect curve now has a complete fail-closed 5x2 target map. The
next non-retread action is to resolve the TT5L provider blocker or use the
alternate provider path, run the paired exact-eval cells, harvest them into the
Lightning `local_artifact_dir` paths, then rerun the harvest-cell builder and
effect-curve builder.
