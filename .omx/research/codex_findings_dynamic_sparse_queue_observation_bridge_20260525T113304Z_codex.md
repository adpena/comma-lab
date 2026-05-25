# Codex Findings: Dynamic Sparse Queue Observation Bridge

- UTC: 2026-05-25T11:33:04Z
- Lane: `codex_dynamic_sparse_queue_observation_bridge_20260525`
- Status: integrated; planning-only; not score authority

## Finding

`operation_set_compiler_hint_from_observation_feedback(...)` could consume
normalized inverse-steganalysis observations, but queue-owned materializer
results and direct materializer manifests still needed a manual conversion step
before they could steer dynamic sparse gate selection. That kept realized
archive deltas and receiver proof signals one step away from the compiler
surface.

## Landed Integration

- Added `tools/build_dynamic_sparse_gate_compiler_hint.py --queue-observation`
  input for queue-owned materializer feedback.
- Added shared materializer feedback normalization in
  `tac.optimization.materializer_feedback` and wired
  `--materializer-feedback` into the dynamic sparse gate compiler CLI.
- Reused the existing inverse-steganalysis queue converters, including required
  runtime/cache identity inputs, candidate-map handling, source-path recording,
  axis labeling, and false-authority rejection.
- Compiled queue-derived materializer archive-delta observations directly into
  dynamic sparse observation-feedback hints.
- Added regressions proving successful materializer queue observations and
  direct materializer manifests select the source unit, preserve saved bytes,
  carry provenance, and remain non-authoritative.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/dynamic_sparse_gate_oracle.py src/tac/optimization/__init__.py src/tac/optimization/materializer_feedback.py tools/build_dynamic_sparse_gate_compiler_hint.py src/tac/tests/test_dynamic_sparse_gate_oracle.py`
- `.venv/bin/python -m pytest src/tac/tests/test_dynamic_sparse_gate_oracle.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_dynamic_sparse_gate_oracle.py src/tac/tests/test_byte_shaving_campaign_queue.py::test_dynamic_sparse_observation_feedback_hint_reaches_materializer_work_queue -q`
- `.venv/bin/python -m ruff check tools/run_family_agnostic_materializer_sweep.py src/tac/optimization/materializer_feedback.py src/tac/tests/test_family_agnostic_materializer_sweep.py`
- `.venv/bin/python -m pytest src/tac/tests/test_family_agnostic_materializer_sweep.py -q`

## Remaining Work

- Have the byte-shaving materializer campaign runner invoke this bridge after
  queue observation harvest so gate hints are generated as a standard artifact.
- Feed calibrated interaction deltas from repeated queue observations back into
  the water-fill allocator.
- Continue to require byte-closed archive materialization and exact contest auth
  eval before score or promotion authority.
