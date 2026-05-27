# Codex Findings - Runtime Expected Tree Identity

Generated: 2026-05-27T19:39:43Z

## Summary

The materializer harvest and runtime-adapter identity boundary now treats
`expected_runtime_tree_sha256` as a distinct fail-closed receiver/runtime
contract, rather than letting an observed runtime tree hash silently stand in as
the expected identity.

## Engineering Outcome

- Ready runtime adapters with a live runtime directory must declare an expected
  runtime tree hash.
- Harvested materializer chains propagate expected runtime identity into source
  rows.
- Mismatches between expected identity, observed manifest identity, and live
  runtime tree identity block receiver readiness instead of promoting a stale
  archive/runtime pair.
- Scheduler tests now cover missing expected identity and stale expected runtime
  identity.

## Validation

- `.venv/bin/ruff check src/tac/optimization/runtime_adapter_identity.py src/tac/optimizer/materializer_chain_harvest.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_runtime_adapter_identity.py`
- `.venv/bin/python -m py_compile src/tac/optimization/runtime_adapter_identity.py src/tac/optimizer/materializer_chain_harvest.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_runtime_adapter_identity.py`
- `.venv/bin/python -m pytest src/tac/tests/test_runtime_adapter_identity.py src/tac/tests/test_materializer_chain_harvest_scheduler.py -q`

Final focused result: `69 passed`.

## Remaining Work

The next queue-owned final-rate run should consume this stricter identity
contract when promoting receiver-closed materializer chains. Any materializer
that still emits only an observed runtime tree hash should be treated as
runtime-custody incomplete until its expected receiver identity is declared and
live-verified.
