# L5 v2 Schedule Refresh After Side-info Custody Hardening - 2026-05-17

## Purpose

Refresh the committed L5 v2 measurement schedule, TT5L side-info effect curve,
and paired-measurement dispatch plan after the stricter TT5L side-info
effect-curve custody validator landed. This is a planning-artifact update only:
no score claim, no promotion, and no provider dispatch.

## Source Inputs

- Probe intake:
  `.omx/research/l5_v2_probe_observation_intake_20260516_codex.json`
- Side-info effect curve:
  `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json`
- Side-info effect-curve seed cell:
  `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_seed_cells_20260516_codex.json`
- Side-info effect-curve builder:
  `tools/build_l5_v2_sideinfo_effect_curve.py`
- Schedule builder:
  `tools/build_l5_v2_lattice_measurement_schedule.py`
- Dispatch-plan builder:
  `tools/build_l5_v2_paired_measurement_dispatch_plan.py`

## Finding

The current L5 v2 lattice still routes to
`fill_missing_c1_z5_tt5l_probe_observations`. The stricter side-info
effect-curve validator initially exposed that the committed effect-curve
artifact had preserved TT5L liveness but had not preserved the exact CUDA
custody fields that were already present in the seed cell. Rebuilding directly
from the seed then exposed the inverse no-signal-loss issue: exact custody was
preserved, but side-info liveness was lost because the seed cell lacked
`provenance.per_pair_side_info_liveness`.

The seed cell now carries both:

- exact CUDA custody: hardware, inflate/eval device, auth-eval command, log,
  artifact, inflated-output manifest, raw-output aggregate SHA, runtime SHAs;
- TT5L side-info liveness: checked all-zero `trained` side-info with
  `nonzero_values=0`.

The rebuilt effect curve therefore removes the stale exact-custody missing
blockers while retaining the correct blocker:

- `tt5l_sideinfo_effect_curve_cell_sideinfo_nonzero_missing:contest_cuda:trained`

The side-info effect curve remains incomplete and cannot advance architecture
lock or promotion because it still has only one observed cell and the trained
side-info is all-zero. The paired-measurement dispatch plan was regenerated so
its source schedule SHA and `plan_id` match the refreshed schedule artifact.

## Commands

```bash
.venv/bin/python tools/build_l5_v2_sideinfo_effect_curve.py \
  --cell-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_seed_cells_20260516_codex.json \
  --output-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json \
  --repo-root .

.venv/bin/python tools/build_l5_v2_lattice_measurement_schedule.py \
  --probe-intake-json .omx/research/l5_v2_probe_observation_intake_20260516_codex.json \
  --sideinfo-effect-curve-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json \
  --repo-root .

.venv/bin/python tools/build_l5_v2_paired_measurement_dispatch_plan.py
```

## Updated Artifacts

- `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_seed_cells_20260516_codex.json`
- `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json`
- `.omx/research/l5_v2_lattice_measurement_schedule_20260516_codex.json`
- `.omx/research/l5_v2_lattice_measurement_schedule_20260516_codex.md`
- `.omx/research/l5_v2_paired_measurement_dispatch_plan_20260516_codex.json`
- `.omx/research/l5_v2_paired_measurement_dispatch_plan_20260516_codex.md`
- `.omx/research/l5_v2_architecture_lock_packet_20260516_codex.json`

## Current Next Action

Do not treat this as a readiness promotion. The next L5 v2 work remains filling
the missing byte-closed, axis-labelled paired evidence for C1, Z5, TT5L, and
the TT5L side-info effect-curve variants, with dispatch claims and paired
CPU/CUDA custody before any score or architecture-lock status changes.
