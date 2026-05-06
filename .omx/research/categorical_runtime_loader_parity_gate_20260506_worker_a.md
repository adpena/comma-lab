# Categorical Runtime Loader Parity Gate - 2026-05-06

Worker: A
Evidence grade: pre-dispatch readiness hardening
Score claim: false
Dispatch attempted: false

## Context

`categorical_qma9_clade_spade_openpilot` remains planning-only until it has a
real byte-closed archive candidate. The current blockers include PR91/HPM1 full
decode/reencode parity and runtime loader parity. A candidate manifest could
previously declare a repo runtime consumer and a charged archive member set
without proving that the archive actually carries the declared loader bytes.

## Change

`audit_categorical_candidate_manifest()` now requires
`runtime_loader_parity` before readiness can pass. The parity report must:

- use `categorical_runtime_loader_parity_v1`;
- keep `score_claim=false` and `dispatch_attempted=false`;
- name a safe charged archive member with role `decoder_or_runtime_consumer`;
- prove the loader member SHA-256 matches the declared runtime source SHA-256;
- declare byte identity, sidecar-free loading, no fallback, and the charged
  members loaded by the runtime.

The deterministic categorical fixture now embeds the bytes of
`src/tac/qma9_range_mask_contract.py` as `runtime_decoder.py`, records the
runtime source hash, and still stays non-dispatchable through
`fixture_only_candidate_not_dispatchable`.

## Verification

```text
.venv/bin/python -m pytest src/tac/tests/test_categorical_candidate_readiness.py src/tac/tests/test_build_categorical_candidate_fixture.py -q
```

Result: `21 passed, 1 warning` (duplicate ZIP member warning is from the
existing duplicate-member regression test).

## Remaining Blockers

- PR91/HPM1 still lacks full 600-frame decode/reencode parity.
- No real categorical archive candidate has replaced the deterministic fixture.
- HPAC CPU/CUDA runtime contract remains unresolved for PR91-derived HPM1 work.
- No lane claim, exact CUDA auth eval, or score claim was made.
