# Codex Findings: Byte-Shaving Queue Runtime Identity Greenup

UTC: 2026-05-31T02:42:06Z

## Scope

The prior MLX runtime-bridge migration exposed a current-tree blocker in the
byte-shaving queue regression suite:

- `test_byte_shaving_campaign_queue.py` had 8 failures.
- The failures were in fixture/readiness expectations, not in the migrated MLX
  bridge files.

This follow-up keeps the stricter queue contracts and updates the tests to
exercise them honestly.

## Fix

The byte-shaving queue tests now provide real runtime identity/proof custody
where they claim receiver or runtime readiness:

- valid runtime directory with `inflate.sh`;
- observed and expected runtime-tree SHA-256;
- runtime-consumption proof JSON with `passed=true`,
  `runtime_consumption_proof_passed=true`, false authority fields, empty
  blockers, candidate-archive SHA, and runtime identity manifest;
- source-native adapter-file identity where DFL1 uses an unpacker file rather
  than a runtime tree.

The registry expected-target list was also refreshed for the already-registered
FP11 source-brotli recode and FECA selector-stream materializers.

The DFL1 follow-up assertions now match the exact-readiness handoff directory
currently produced outside the materializer output directory, preserving the
intent: local reruns must not erase downstream exact-readiness artifacts.

## Validation

Passed:

- `ruff src/tac/tests/test_byte_shaving_campaign_queue.py`
- `pytest src/tac/tests/test_byte_shaving_campaign_queue.py -q`
  - 94 passed.

## Verdict

This is a test/custody greenup, not a score claim. It protects the archive and
entropy automation surface by ensuring tests no longer bless receiver-ready or
runtime-ready manifests without concrete identity and proof artifacts.
