# L5 v2 Full-Frame Axis Custody Hardening

- date: `2026-05-16`
- agent: `codex`
- trigger: readonly L5-v2 adversarial review P1 finding
- score_claim: `false`
- promotion_eligible: `false`

## Finding

The L5-v2 probe gate could validate paired score JSON/log/artifact fields without requiring full-frame inflated-output custody. That allowed a theoretical architecture-lock path where the scored archive/runtime identity was present, but the raw output manifest and aggregate SHA were absent.

## Fix

`validate_exact_eval_evidence()` now supports optional full-frame custody requirements:

- `require_inflated_outputs_manifest`
- `require_raw_output_aggregate_sha256`

When enabled, validation requires a local inflated-output manifest path, manifest SHA-256, raw-output aggregate SHA-256, manifest file existence, manifest SHA match, and manifest aggregate match.

The L5-v2 probe disambiguator enables both requirements for every required axis. The TT5L CUDA intake now carries:

- `inflated_outputs_manifest_path`
- `inflated_outputs_manifest_sha256`
- `raw_output_aggregate_sha256`

## Current TT5L State

TT5L contest-CUDA remains recognized and full-frame custody-complete. It is still not eligible for architecture lock because the remaining blockers are real:

- `l5_v2_probe_predicate_failed`
- `l5_v2_probe_paired_exact_axes_missing`
- `l5_v2_probe_sideinfo_consumption_missing`
- `l5_v2_probe_axis_evidence_missing:contest_cpu`
- `l5_v2_probe_axis_score_delta_missing:contest_cuda`

## Verification

- `.venv/bin/python -m ruff check src/tac/exact_eval_custody.py src/tac/optimization/l5_v2_probe_disambiguator.py src/tac/optimization/l5_v2_probe_intake.py src/tac/tests/test_exact_eval_custody.py src/tac/tests/test_l5_v2_probe_disambiguator.py src/tac/tests/test_l5_v2_probe_intake.py`
- `.venv/bin/python -m pytest src/tac/tests/test_exact_eval_custody.py src/tac/tests/test_l5_v2_probe_disambiguator.py src/tac/tests/test_l5_v2_probe_intake.py -q`
- `.venv/bin/python tools/audit_l5_v2_probe_observations.py --output-json .omx/research/l5_v2_probe_observation_intake_20260516_codex.json --output-md .omx/research/l5_v2_probe_observation_intake_20260516_codex.md --probe-gate-out .omx/research/l5_v2_probe_gate_artifact_20260516_codex.json`

The audit still exits nonzero by design while `architecture_lock_allowed=false`.
