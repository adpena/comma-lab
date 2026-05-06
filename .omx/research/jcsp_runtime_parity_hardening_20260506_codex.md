# JCSP Runtime Parity Hardening

Date: 2026-05-06
Author: codex
Evidence grade: empirical local guard/test
Score claim: false
Dispatch attempted: false

## Scope

Focused `joint_admm_balle_arithmetic_stack` readiness hardening in
`src/tac/joint_codec_stack_orchestrator.py`.

## Change

- Added deterministic JSON-ready metadata manifests for
  `model_to_jcsp_streams` output, including stream count, stream records,
  stable manifest SHA-256, no-score flags, and promotion blockers.
- Added fail-closed codec-kind to payload-magic validation for JCSP pack and
  unpack paths. `KIND_ARITHMETIC_STATIC` accepts `AQv1` and `AQc1`; Ballé
  hyperprior accepts `BHv1`; raw passthrough must be non-empty.
- Added a runtime-loader parity summary helper that records payload magics and
  the dispatch kind a future inflate-side loader would trust.

## Evidence

- [empirical:local-test] `.venv/bin/python -m pytest
  src/tac/tests/test_jcsp_model_streams.py
  src/tac/tests/test_joint_codec_stack_orchestrator.py -q` passed with
  12 tests.
- [empirical:local-test] `.venv/bin/python -m pytest
  src/tac/tests/test_arithmetic_qint_codec.py
  src/tac/tests/test_jcsp_model_streams.py
  src/tac/tests/test_joint_codec_stack_orchestrator.py -q` passed with
  35 tests.
- [empirical:local-test] `.venv/bin/python -m pytest
  src/tac/tests/test_stack_compositions.py -q` passed with 12 tests.
- [empirical:local-lint] `.venv/bin/python -m ruff check
  src/tac/joint_codec_stack_orchestrator.py
  src/tac/tests/test_jcsp_model_streams.py
  src/tac/tests/test_joint_codec_stack_orchestrator.py` passed.

## Remaining Blockers

- No byte-closed contest archive member consumes JCSP yet.
- No canonical `submissions/robust_current` inflate/runtime loader consumes a
  JCSP archive member yet.
- No lane claim was opened and no remote/GPU dispatch was attempted.
- No exact CUDA auth eval exists for a stacked JCSP archive, so there is no
  score, rank, promotion, or sub-0.15 claim.
