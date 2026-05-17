# L5 v2 Exact-Eval Artifact Hash Binding

Date: 2026-05-17

## Finding

The shared exact-eval custody validator proved that `artifact_path` existed,
but it did not require the metrics row to be hash-bound to the JSON artifact
at that path. That left a custody gap: a row could carry copied score,
component, archive, runtime, and manifest fields while pointing at a different
or later-mutated exact-eval JSON file.

This is especially risky for L5-v2 because probe intake and TT5L side-info
effect-curve cells decide architecture-lock readiness from structured rows
derived from exact-eval artifacts. Path existence is not enough for
contest-grade provenance; the row must bind to the artifact bytes.

## Fix

- `src/tac/exact_eval_custody.py`
  - added `artifact_sha256` to `ExactEvalEvidenceValidation`;
  - added `require_artifact_sha256`;
  - validates `artifact_sha256` / `exact_eval_artifact_sha256`;
  - compares the declared artifact hash to the resolved `artifact_path` bytes
    when `artifact_base_dir` is supplied.
- `src/tac/optimization/l5_v2_probe_intake.py`
  - records the source exact-eval JSON SHA in each axis evidence row;
  - requires artifact SHA custody for L5-v2 probe intake validation.
- `src/tac/optimization/l5_v2_probe_disambiguator.py`
  - requires artifact SHA custody for paired CPU/CUDA axis evidence consumed
    by the C1/Z5/TT5L probe gate;
  - exposes axis-specific missing/mismatched artifact-hash blockers in the
    public gate artifact.
- `src/tac/optimization/l5_v2_sideinfo_effect_curve.py`
  - binds each normalized cell to an exact-eval artifact hash;
  - rejects a supplied artifact hash that does not match the cell artifact.
- `src/tac/optimization/l5_v2_measurement_schedule.py`
  - revalidates side-info effect-curve cells with artifact SHA custody before
    allowing the schedule to advance past the side-info curve gate.

## Regression Coverage

- `test_validate_exact_eval_evidence_can_require_artifact_sha256`
  checks valid, alias, missing, and mismatched artifact hashes.
- `test_l5_v2_probe_requires_axis_artifact_hash_binding`
  proves probe-gate axis evidence fails closed on missing or mismatched
  exact-eval artifact hashes.
- `test_l5_v2_probe_intake_recovers_direct_auth_eval_provenance_and_local_log`
  now asserts probe intake carries the exact SHA-256 of the source
  `contest_auth_eval.json`.
- `test_sideinfo_effect_curve_rejects_artifact_sha_mismatch`
  proves TT5L side-info cells fail closed when the declared artifact hash does
  not match the pointed-to exact-eval JSON.
- `test_l5_v2_schedule_requires_sideinfo_cell_artifact_hash_binding`
  proves the measurement schedule does not accept stale hand-written
  side-info cells without artifact hash binding.

## Validation

```text
.venv/bin/python -m pytest src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_l5_v2_measurement_schedule.py src/tac/tests/test_l5_v2_probe_disambiguator.py src/tac/tests/test_l5_v2_sideinfo_effect_curve.py src/tac/tests/test_l5_v2_probe_intake.py src/tac/tests/test_exact_eval_custody.py -q
219 passed in 2.58s

.venv/bin/python -m py_compile src/tac/exact_eval_custody.py src/tac/optimization/l5_v2_probe_disambiguator.py src/tac/optimization/l5_v2_probe_intake.py src/tac/optimization/l5_v2_sideinfo_effect_curve.py src/tac/optimization/l5_v2_measurement_schedule.py src/tac/tests/test_l5_staircase_v2.py
pass

.venv/bin/python -m ruff check src/tac/tests/test_l5_staircase_v2.py src/tac/tests/test_l5_v2_measurement_schedule.py src/tac/tests/test_l5_v2_probe_disambiguator.py src/tac/tests/test_l5_v2_probe_intake.py src/tac/tests/test_exact_eval_custody.py src/tac/tests/test_l5_v2_sideinfo_effect_curve.py src/tac/exact_eval_custody.py src/tac/optimization/l5_v2_measurement_schedule.py src/tac/optimization/l5_v2_probe_disambiguator.py src/tac/optimization/l5_v2_probe_intake.py src/tac/optimization/l5_v2_sideinfo_effect_curve.py
All checks passed!

.venv/bin/python tools/audit_l5_v2_probe_observations.py --output-json .omx/research/l5_v2_probe_observation_intake_20260516_codex.json --output-md .omx/research/l5_v2_probe_observation_intake_20260516_codex.md --probe-gate-out .omx/research/l5_v2_probe_gate_artifact_20260516_codex.json
wrote intake and probe-gate artifacts; exited nonzero as expected because architecture lock remains blocked

.venv/bin/python tools/build_l5_v2_sideinfo_effect_curve.py --cell-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_seed_cells_20260516_codex.json --output-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json
predicate_passed=false; contract_blockers=7; effect_blockers=2

.venv/bin/python tools/build_l5_v2_lattice_measurement_schedule.py --probe-intake-json .omx/research/l5_v2_probe_observation_intake_20260516_codex.json --sideinfo-effect-curve-json .omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json --output-json .omx/research/l5_v2_lattice_measurement_schedule_20260516_codex.json --output-md .omx/research/l5_v2_lattice_measurement_schedule_20260516_codex.md
active_rule_id=fill_missing_c1_z5_tt5l_probe_observations

.venv/bin/python tools/build_l5_v2_architecture_lock_packet.py --output-json .omx/research/l5_v2_architecture_lock_packet_20260516_codex.json --output-md .omx/research/l5_v2_architecture_lock_packet_20260516_codex.md
architecture_lock_allowed=false; blockers=[requires_all_l5_v2_gate_evidence_valid, requires_c1_z5_tt5l_probe_gate_evidence, requires_paired_cpu_cuda_sideinfo_effect_curve]
```

## Status

No score claim and no promotion claim. This only hardens custody semantics. L5
v2 remains fail-closed pending complete paired CPU/CUDA side-info effect-curve
evidence and C1/Z5/TT5L probe gate evidence.
