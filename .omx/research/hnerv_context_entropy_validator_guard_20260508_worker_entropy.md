# HNeRV Context Entropy Validator Guard - 2026-05-08

## Scope

Worker HNeRV-Entropy tightened the HNeRV entropy candidate-packet validator
after reviewing the untracked HDM/HDC work products under
`experiments/results/hnerv_hdm3_entropy_packet_20260507_codex/`.

No raw experiment artifact was modified. No dispatch was attempted. No score is
claimed.

## Finding

The HDC1/HDC2 context fixtures are useful parity and byte-accounting probes,
but they are not exact-evaluable archive lanes by themselves:

- HDC1/HDC2 are byte-negative versus the current HNeRV decoder section in the
  inspected work products.
- HDC2 lacks a deterministic runtime consumer contract that would let public
  inflate consume the candidate decoder section directly.
- A candidate packet must not treat an HDC archive contract as valid unless the
  contract proves strict byte improvement and runtime consumption.

## Guard Added

`src/tac/hnerv_entropy_candidate_packet.py` now requires the HDC2 archive
candidate contract to include and satisfy:

- `candidate_rate_positive=true`
- `candidate_archive_bytes < source_archive_bytes`
- `candidate_decoder_section_bytes < source_decoder_section_bytes`
- source payload and source decoder-section SHA-256 custody

If any of those fail, the requirement artifact is marked invalid and the packet
remains `ready_for_archive_preflight=false` and
`ready_for_exact_eval_dispatch=false`.

## Regression Tests

Added focused tests in
`src/tac/tests/test_hnerv_entropy_candidate_packet.py`:

- byte-negative HDC2 archive contracts are rejected with explicit validation
  blockers;
- strict byte-winning HDC2 archive contracts can validate locally while still
  remaining non-dispatchable.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_hnerv_entropy_candidate_packet.py
# 16 passed in 3.95s

.venv/bin/ruff check src/tac/hnerv_entropy_candidate_packet.py src/tac/tests/test_hnerv_entropy_candidate_packet.py
# All checks passed.

.venv/bin/python -m pytest src/tac/tests/test_hnerv_hdc2_combined_entropy.py src/tac/tests/test_hnerv_entropy_candidate_packet.py
# 19 passed in 3.98s
```

## Remaining State

This is a validator/readiness guard only. It does not promote HDC1/HDC2, does
not build a candidate archive, and does not alter HDM3 runtime-adapter custody.
The next HDC-family archive path still needs an actual smaller runtime-consumed
stream, candidate archive manifest, runtime parity, strict compliance, Level-2
lane claim, and exact CUDA auth eval before any score use.
