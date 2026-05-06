# JCSP Stream Decomposition Hardening - 2026-05-06 Codex

## Scope

Focused PARADIGM-gamma hardening in
`src/tac/joint_codec_stack_orchestrator.py` for `model_to_jcsp_streams`.

## Change

- Model/state-dict tensor stream decomposition is now sorted by stream name,
  independent of mapping or module iteration order.
- Duplicate stream names fail closed before any planning rows are emitted.
- Codec overrides that reference unknown streams fail closed instead of being
  silently ignored.
- Score marginals and default marginals must be finite.
- Each `JCSPTensorStreamSpec` now carries a deterministic `stream_id` plus
  `decomposition_index` for stable downstream manifests.

## Evidence

- [empirical:src/tac/tests/test_jcsp_model_streams.py] Added focused tests for
  order invariance, stable stream ids, duplicate-name rejection, stale override
  rejection, and non-finite marginal rejection.
- [empirical:local-test] `.venv/bin/python -m pytest
  src/tac/tests/test_jcsp_model_streams.py
  src/tac/tests/test_joint_codec_stack_orchestrator.py
  src/tac/tests/test_stack_compositions.py -q` passed with 20 tests.
- [empirical:local-lint] `.venv/bin/python -m ruff check
  src/tac/joint_codec_stack_orchestrator.py
  src/tac/tests/test_jcsp_model_streams.py` passed.

## Dispatch Status

No GPU, remote job, exact eval, archive build, or score claim was attempted.
This remains a compress-time planning/readiness improvement only; JCSP archive
dispatch remains blocked on byte-closed archive-member construction, runtime
loader parity, lane claim, and exact CUDA auth eval.
