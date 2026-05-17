# L5 v2 Pair-Identity Control-Plane Refresh

Date: 2026-05-17
Author: Codex
Scope: TT5L L5 v2 side-info effect-curve custody and dispatch planning

## Reason

After the TT5L side-info effect-curve validator began carrying
`pair_group_id` and `run_id`, the committed control-plane artifacts needed to
be regenerated so they did not point at a pre-guard source manifest. This is a
no-score, no-dispatch refresh.

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

.venv/bin/python tools/build_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan.py \
  --repo-root .

.venv/bin/python tools/build_l5_v2_architecture_lock_packet.py --repo-root .
```

## Result

- `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json`
  now preserves empty `pair_group_id` and `run_id` fields on the stale recovered
  single-cell TT5L CUDA observation. That old cell remains diagnostic only.
- `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_20260517_codex.json`
  now records source commit `fe65c9ec71ef568ea42f4c0edc38b34408489f83`, the
  commit that includes the stricter pair/run validator.
- `.omx/research/l5_v2_architecture_lock_packet_20260516_codex.json` now sees
  the Lightning paired-axis dry-run plan as source-current again:
  `source_relevant_paths_match=true` and
  `source_custody_current_for_execution=true`.

The architecture lock remains false. Current blockers remain:

- `requires_all_l5_v2_gate_evidence_valid`
- `requires_c1_z5_tt5l_probe_gate_evidence`
- `requires_paired_cpu_cuda_sideinfo_effect_curve`

## Next Artifact-Producing Action

The next frontier action is not another local-basin PR106/HNeRV pass. It is
unblocking and dispatching the current TT5L 10-cell side-info effect curve on
paired `[contest-CPU]` and `[contest-CUDA]` axes after provider identity,
source-manifest staging, runtime probe, and per-axis dispatch claims are real.

No score claim, promotion claim, or architecture-lock authority is made by this
refresh.
