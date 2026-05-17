# L5 v2 Schedule Refresh After Side-info Custody Hardening - 2026-05-17

## Purpose

Refresh the committed L5 v2 measurement schedule and paired-measurement
dispatch plan after the stricter TT5L side-info effect-curve custody validator
landed. This is a planning-artifact update only: no score claim, no promotion,
and no provider dispatch.

## Source Inputs

- Probe intake:
  `.omx/research/l5_v2_probe_observation_intake_20260516_codex.json`
- Side-info effect curve:
  `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json`
- Schedule builder:
  `tools/build_l5_v2_lattice_measurement_schedule.py`
- Dispatch-plan builder:
  `tools/build_l5_v2_paired_measurement_dispatch_plan.py`

## Finding

The current L5 v2 lattice still routes to
`fill_missing_c1_z5_tt5l_probe_observations`. The stricter side-info
effect-curve validator now surfaces additional exact-eval custody blockers for
the one existing `contest_cuda:trained` TT5L side-info cell:

- `tt5l_sideinfo_effect_curve_cell_exact_eval_hardware_missing:contest_cuda:trained`
- `tt5l_sideinfo_effect_curve_cell_exact_eval_inflate_device_missing:contest_cuda:trained`
- `tt5l_sideinfo_effect_curve_cell_exact_eval_eval_device_missing:contest_cuda:trained`
- `tt5l_sideinfo_effect_curve_cell_exact_eval_auth_eval_command_missing:contest_cuda:trained`

Those blockers are correct: the side-info effect curve remains incomplete and
cannot advance architecture lock or promotion. The paired-measurement dispatch
plan was regenerated so its source schedule SHA and `plan_id` match the
refreshed schedule artifact.

## Commands

```bash
.venv/bin/python tools/build_l5_v2_lattice_measurement_schedule.py \
  --probe-intake-json .omx/research/l5_v2_probe_observation_intake_20260516_codex.json \
  --sideinfo-effect-curve-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json \
  --repo-root .

.venv/bin/python tools/build_l5_v2_paired_measurement_dispatch_plan.py
```

## Updated Artifacts

- `.omx/research/l5_v2_lattice_measurement_schedule_20260516_codex.json`
- `.omx/research/l5_v2_lattice_measurement_schedule_20260516_codex.md`
- `.omx/research/l5_v2_paired_measurement_dispatch_plan_20260516_codex.json`
- `.omx/research/l5_v2_paired_measurement_dispatch_plan_20260516_codex.md`

## Current Next Action

Do not treat this as a readiness promotion. The next L5 v2 work remains filling
the missing byte-closed, axis-labelled paired evidence for C1, Z5, TT5L, and
the TT5L side-info effect-curve variants, with dispatch claims and paired
CPU/CUDA custody before any score or architecture-lock status changes.
