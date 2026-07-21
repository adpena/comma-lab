# REUSE MANIFEST — per-stratum recursive-fractal optimal treatment

UTC: 2026-07-21T19:12:17Z

## Reusable landing

- `tools/measure_per_stratum_recursive_fractal_optimal.py`
  - deterministic, read-only custody audit;
  - exact archive/receipt rehash;
  - n64/n600 BEV stage rotation audit;
  - scorer-`K` `CalibratedGeometry` identity canary;
  - OpenPilot VP sensitivity math with causal-attribution refusal;
  - M1/S4 control binding, cap arithmetic, and five-row per-stratum table;
  - atomic JSON output, no scorer launch and no bulk creation.
- `tools/tests/test_measure_per_stratum_recursive_fractal_optimal.py`
  - fixture tests for fail-closed hashes/schemas, cap and sensitivity arithmetic,
    identity/nonidentity rotation custody, exact per-class accounting, and null v9
    byte behavior.
- Full production receipt (SSD):
  `/Volumes/VertigoDataTier/pact/evidence/per_stratum_recursive_fractal_20260721/`
- Compact repository receipt:
  `.omx/research/per_stratum_recursive_fractal_optimal_20260721T191217Z_receipt.json`.

## Inputs reused read-only

- BEV-v2 SSD n64/n600 receipts and 600 resumable stages.
- M1 90,566-byte archive, exact n600 harness receipt, and 38-chunk hard-oracle
  decomposition.
- S4 451,191-byte archive plus standalone parity and hard-scorer receipts.
- #503 measurement receipt/build spec, SPEC_v8 carrier table, and c2 canonical equation.
- `src/tac/calibrated_geometry.py` only through explicit scorer-resolution arguments;
  its defaults are not reused.

## Explicit non-reuse / blockers

- The historical untracked #503 module hashes are not executable source.
- 65,172-byte `0.bin` and zero-group deltas are rate diagnostics, not an archive.
- M1/S4 controls are not per-stratum v9 candidates.
- BEV-v2 generated homographies are not independent observations and may not be
  decomposed into a causal calibration fraction.
- OpenPilot sampled lane points may not be described as native polynomial DOF.
- No current result is promotion or score authority; pointer remains unchanged.

## Command

After MAIN review, reproduce with:

```bash
PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python \
  tools/measure_per_stratum_recursive_fractal_optimal.py \
  --output /Volumes/VertigoDataTier/pact/evidence/per_stratum_recursive_fractal_20260721/per_stratum_recursive_fractal_optimal_receipt.json \
  --compact-output .omx/research/per_stratum_recursive_fractal_optimal_20260721T191217Z_receipt.json
```

Two consecutive post-binding runs were byte-identical. Expected outputs are:

- full receipt: 124,744 bytes, SHA-256
  `bbab41d92c0cc05a88a0c107f04d4b9112b24ca63a39d329dfaa660549a9af2c`;
- compact receipt: 23,427 bytes, SHA-256
  `b45c80c06d146b92808cf50ad1d6b98e4be699e4d8ee14472dc36c88a1becb49`.

The command keeps all large inputs and the full output on the SSD tier and creates no
raw decode/scorer scratch.

## Review boundary

MAIN should re-run focused tests and the read-only production command, compare output
bytes/hash, inspect every `null` measured-byte field, and verify that no missing source
is silently coerced to zero before merge.
