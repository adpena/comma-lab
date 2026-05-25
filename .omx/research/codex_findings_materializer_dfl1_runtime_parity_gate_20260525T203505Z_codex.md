# Codex Findings - Materializer DFL1 Runtime Parity Gate

UTC: 2026-05-25T20:35:05Z

## Scope

Adversarial follow-up on the family-agnostic materializer queue after the
renderer-payload DFL1 lane was identified as executable-looking without the
same receiver/runtime/full-frame proof obligations expected from byte-closed
semantic rewrites.

## Finding

`renderer_payload_dfl1` queue postconditions required a manifest and
false-authority fields, but did not require:

- receiver contract satisfaction;
- runtime adapter readiness;
- full-frame inflate parity;
- renderer-payload DFL1 inflate parity.

That allowed a DFL1 manifest to satisfy queue-local completion while still
lacking evidence that the source runtime would consume the transformed payload
and reproduce contest frames.

## Fix

The materializer queue postcondition builder now treats DFL1 as a
receiver-proven materializer family and requires runtime/full-frame/Dfl1 parity
fields before the queue can mark the generated candidate as passing.

Regression coverage:

- `test_renderer_payload_dfl1_postconditions_require_runtime_and_full_frame_parity`

Verification:

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_campaign_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py::test_renderer_payload_dfl1_postconditions_require_runtime_and_full_frame_parity -q`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py -q`

## Remaining Work

The next materializer hardening pass should apply the same scrutiny to tensor
factorization and section recoding semantic rewrites: proof declarations are
not enough until runtime consumption and full-frame parity are mechanically
checked for the rewritten archive/runtime pair.
