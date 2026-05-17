# L5 v2 TT5L materialized output identity hardening

- date: 2026-05-17
- scope: TT5L-first L5 v2 staircase paired CPU/CUDA materialized measurement
- authority: engineering/custody hardening only; no score claim
- axis labels: `[contest-CPU]` and `[contest-CUDA]` kept separate

## Finding

The TT5L `random_lsb` materialized paired work unit selected a new byte-closed
archive:

- archive: `experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_20260517_codex/random_lsb/archive.zip`
- sha256: `b6a5b63c0ea8acd582d8f273a1ee9e00f74becc9d1993a2f3085f2f89d64b1c7`

But the generated run/output identity reused the prior
`measure_tt5l_autonomy_paired_exact` source-archive result directories. Those
directories already contain recovered paired artifacts for source archive
`2b05b7351b690b0b2251ddc620d80dd9a1833051cfa07e679106d00fbc70024a`.

That created two signal-loss risks:

1. A fresh `random_lsb` dispatch could overwrite or be confused with the old
   source-archive paired result.
2. Probe intake could combine axes from different archive SHAs if one new axis
   existed and one stale axis remained.

## Fix

- `src/tac/optimization/l5_v2_tt5l_materialized_work_unit.py` now emits a
  variant/archive-specific `run_id`, e.g.
  `l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_random_lsb_b6a5b63c0ea8`.
- The materialized JSON and Markdown now expose that `run_id`.
- `src/tac/optimization/l5_staircase_v2.py` validates output directories,
  per-axis `--output-dir`, and per-axis `--instance-job-id`; it also refuses an
  existing output result whose archive SHA mismatches the materialized archive.
- The synthesized operator execute command now mirrors the materialized
  `run_id` and `output_root`, and no longer adds
  `--skip-axis-if-promotable-anchor-exists` unless the plan explicitly opts in.
- `src/tac/optimization/l5_v2_probe_intake.py` reads materialized plan outputs
  before legacy TT5L paths and groups candidate axes by exact archive SHA before
  forming a paired observation.

## Verification

- `13 passed`: focused TT5L materialized builder, probe-intake, and L5 v2 action
  tests after the code change.
- `14 passed`: focused tests after refreshing live `.omx` artifacts, including
  `test_l5_v2_architecture_lock_packet_artifact_tracks_live_payload`.
- `ruff check` passed on touched Python files.
- `py_compile` passed on touched Python modules/tools.
- `git diff --check` passed.

## Current State

The refreshed materialized work-unit packet remains non-promotional:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatch_attempted=false`

The next material L5 v2 action is still to review and execute the corrected
TT5L `random_lsb` paired CPU/CUDA measurement through the canonical paired
Modal dispatcher, then harvest both axes through `tools/recover_modal_auth_eval.py`.
