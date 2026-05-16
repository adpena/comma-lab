# L5 v2 Probe Intake Custody Normalization Hardening

- date: `2026-05-16`
- agent: `codex`
- scope: L5-v2 staircase probe/custody intake
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Change

The L5-v2 probe-observation intake now normalizes already-present Modal auth-eval custody from recovered exact-eval artifacts instead of leaving those fields blank when they live under nested `provenance` or sibling log files.

Recovered fields:

- `provenance.sys_argv` as the full auth-eval command when no higher-level custody command is present.
- `provenance.gpu_model`, `provenance.device`, and `provenance.inflate_device_policy`.
- local sibling `contest_auth_eval.stdout.log` / `contest_auth_eval.stderr.log` for recovered Modal auth-eval logs.
- contest evidence-grade wording for rows with exact contest axes, without making the row architecture-lock eligible.

The disambiguator also no longer reports `l5_v2_probe_runtime_tree_sha_by_axis_invalid:contest_cpu` when the CPU axis is absent. The true blocker remains `l5_v2_probe_axis_evidence_missing:contest_cpu`.

## TT5L Current Gate State

After regeneration, TT5L has one accepted exact axis:

- axis: `contest_cuda`
- archive_sha256: `2b05b7351b690b0b2251ddc620d80dd9a1833051cfa07e679106d00fbc70024a`
- runtime_tree_sha256: `ed41e941b624b00412c680c56dc5b9b23db32c70ce1008d03f5bca939917b6cd`
- score: `3.9007398365396795`
- log_path: `experiments/results/modal_auth_eval/time_traveler_recovered_tt5l_25ep_exact_cuda_20260514T105300Z/contest_auth_eval.stdout.log`

Remaining blockers are now the real next actions:

- `l5_v2_probe_predicate_failed`
- `l5_v2_probe_paired_exact_axes_missing`
- `l5_v2_probe_sideinfo_consumption_missing`
- `l5_v2_probe_axis_evidence_missing:contest_cpu`
- `l5_v2_probe_axis_score_delta_missing:contest_cuda`

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/l5_v2_probe_disambiguator.py src/tac/optimization/l5_v2_probe_intake.py src/tac/tests/test_l5_v2_probe_disambiguator.py src/tac/tests/test_l5_v2_probe_intake.py`
- `.venv/bin/python -m pytest src/tac/tests/test_l5_v2_probe_disambiguator.py src/tac/tests/test_l5_v2_probe_intake.py -q`
- `.venv/bin/python tools/audit_l5_v2_probe_observations.py --output-json .omx/research/l5_v2_probe_observation_intake_20260516_codex.json --output-md .omx/research/l5_v2_probe_observation_intake_20260516_codex.md --probe-gate-out .omx/research/l5_v2_probe_gate_artifact_20260516_codex.json`

The audit exits nonzero by design while `architecture_lock_allowed=false`.
