# Materializer Chain Runtime Proof False-Authority Hardening

Generated: 2026-05-27T13:58:19Z
Author: Codex

## Finding

Family-agnostic materializer harvest could previously treat manifest declarations
as receiver/runtime proof. A manifest could set `receiver_contract_satisfied=true`
and `receiver_verification.proof_present=true` while the proof file was missing or
invalid, and the harvested optimizer row would surface as runtime-adapter ready.

Legacy chain harvest also trusted declared runtime tree identity when runtime
adapter readiness was true. That allowed a stale or declaration-only runtime tree
to enter the source queue without re-hashing a live runtime directory.

## Fix

- Family-agnostic harvest now loads runtime-consumption proof fail-closed.
- Missing, symlinked, invalid, non-object, authority-bearing, blocker-bearing,
  archive-mismatched, or verifier-failing proof files demote receiver/runtime
  readiness and add durable readiness/dispatch blockers.
- Chain harvest now requires live runtime adapter identity when runtime readiness
  is claimed, including a real runtime dir with `inflate.sh` and a matching live
  `tree_sha256`.
- The old silent optional-proof loader was removed to avoid a second weak
  interpretation staying in the module.

## Regression Coverage

- `test_harvest_rejects_chain_manifest_stale_runtime_tree_identity`
- `test_harvest_family_agnostic_declaration_only_proof_is_not_receiver_ready`

## Verification

```
.venv/bin/python -m pytest src/tac/tests/test_materializer_chain_harvest_scheduler.py -q
# 60 passed in 26.34s

.venv/bin/ruff check src/tac/optimizer/materializer_chain_harvest.py src/tac/tests/test_materializer_chain_harvest_scheduler.py
# All checks passed

.venv/bin/python -m py_compile src/tac/optimizer/materializer_chain_harvest.py
```

## Remaining Connected Work

The same bug class still needs closure in queue observation and exact-follow-up
bundle readiness surfaces. Those are being handled as separate, scoped patches
so this verified materializer-chain guard can land without mixing unrelated
operational state.
