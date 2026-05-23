# Codex Findings - Byte-Range Entropy Receiver Proof Bridge

UTC: 2026-05-23T16:06:07Z
Lane: `byte_range_entropy_recode_materializer_contract`
Authority axis: local runtime-consumption proof / false score authority

## Landing

- Added `build_byte_range_entropy_recode_receiver_proof(...)` to convert an
  existing PR103 LC-AC runtime-adapter manifest into
  `byte_range_entropy_recode_receiver_proof_v1`.
- Added `tools/build_byte_range_entropy_recode_receiver_proof.py` so this proof
  can be generated from operator flows and queue steps.
- The proof binds:
  - runtime adapter manifest path and sha256;
  - PR103 candidate manifest path and sha256;
  - candidate archive sha256;
  - candidate member sha256;
  - changed candidate byte ranges;
  - runtime-consumption probe pass;
  - decoder-state parity pass.
- The byte-range materializer now removes
  `candidate_runtime_adapter_missing` only when this receiver proof verifies.
  It still keeps inflate/full-frame/exact-eval blockers.
- The materializer registry now advertises the receiver proof builder callable
  alongside plan/materialize/verify functions.

## Verification

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_byte_range_entropy_recode_materializer.py \
  src/tac/tests/test_byte_shaving_campaign_queue.py \
  src/tac/tests/test_pr103_lc_ac_runtime_adapter.py
```

Result: `27 passed`.

```bash
.venv/bin/ruff check \
  src/comma_lab/scheduler/byte_shaving_materializer_registry.py \
  src/comma_lab/scheduler/byte_shaving_campaign_queue.py \
  src/tac/tests/test_byte_shaving_campaign_queue.py \
  src/tac/optimization/byte_range_entropy_recode_materializer.py \
  src/tac/tests/test_byte_range_entropy_recode_materializer.py \
  tools/build_byte_range_entropy_recode_receiver_proof.py
```

Result: clean.

## Authority Boundary

This clears only the runtime-consumption receiver blocker for byte-range
entropy recode work. It does not clear:

- `candidate_inflate_output_parity_missing`
- full-frame parity
- strict pre-submission compliance
- lane dispatch claim
- exact CPU/CUDA auth eval
- score claim, promotion, or rank/kill authority

## Next Integration

The queue can now chain:

1. PR103 arithmetic candidate materialization.
2. PR103 LC-AC runtime adapter generation.
3. Byte-range entropy receiver-proof generation.
4. Byte-range materializer verification with receiver proof supplied.
5. Full inflate parity / exact auth gates.
