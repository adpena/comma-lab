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

## Worker B Dispatchability Hardening Addendum

Date: 2026-05-06
Score claim: false
Dispatch attempted: false

Focused hardening stayed inside the JCSP/model-to-streams path:

- Model stream ordering is now assigned after UTF-8 wire-name sorting, with
  1..255 byte stream-name validation before manifest emission.
- Per-stream byte annotations now distinguish `byte_estimate_source` from
  `bytes_charged_source`; raw passthrough overrides charge raw tensor bytes.
- Wet streams fail closed unless every matched stream has an explicit
  `KIND_RAW_PASSTHROUGH` codec override. Once proven raw, the wet stream gets a
  derivation marginal and no stale missing-marginal blocker.
- Score marginal and codec override maps are normalized by stream name and
  stale entries fail closed.
- Stream manifests now carry a no-sidecar policy and single-`jcsp.bin` archive
  requirement.
- Added `jcsp_model_stream_archive_readiness(...)`, a scoreless readiness
  artifact that consumes a deterministic single-member JCSP archive, rejects
  sidecars through the runtime archive-member loader, reconciles archive stream
  order/name/codec kind with the model-stream manifest, and records per-stream
  archive payload bytes plus remaining blockers.

Evidence:

- [empirical:local-test] `.venv/bin/python -m pytest
  src/tac/tests/test_jcsp_model_streams.py
  src/tac/tests/test_joint_codec_stack_orchestrator.py -q` passed with
  19 tests.
- [empirical:local-test] `.venv/bin/python -m pytest
  src/tac/tests/test_stack_compositions.py -q` passed with 12 tests.
- [empirical:local-lint] `.venv/bin/python -m ruff check
  src/tac/joint_codec_stack_orchestrator.py
  src/tac/tests/test_jcsp_model_streams.py` passed.
- [empirical:syntax] `.venv/bin/python -m py_compile
  src/tac/joint_codec_stack_orchestrator.py
  src/tac/tests/test_jcsp_model_streams.py` passed.

Remaining blockers:

- The readiness artifact is still scoreless and always reports
  `ready_for_exact_eval_dispatch=false`.
- `submissions/robust_current` does not yet consume a JCSP member.
- Exact stream `bytes_charged` must be reconciled to archive payload bytes
  before dispatch; the readiness helper marks mismatches explicitly.
- No lane claim, GPU job, remote dispatch, or CUDA auth eval was attempted.
