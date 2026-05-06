# Entropy-rate decomposition surface

Date: 2026-05-06
Operator: codex
Evidence grade: empirical planning surface

## Scope

Implemented a focused rate-decomposition surface for HNeRV, categorical, and
pose-like streams. The surface consumes explicit stream labels, actual charged
bytes, symbol counts, and optional conditional count groups. It computes:

- empirical entropy bits per symbol from symbol counts;
- entropy floor bits and bytes;
- actual bytes versus entropy-floor byte gap;
- conditional entropy floors by named grouping model;
- best conditional floor and conditional gain over the unconditional model;
- per-stream evidence grade and dispatch blockers.

## Safety contract

- planning_only: true
- score_claim: false
- score_evidence_grade: invalid
- dispatch_attempted: false
- gpu_required: false
- ready_for_exact_eval_dispatch: false

Required blockers remain attached to the top-level manifest, each stream, each
conditional model, and each opportunity row:

- planning_only_entropy_rate_decomposition
- requires_byte_equivalent_codec_transform
- requires_archive_manifest_preflight
- requires_runtime_parity_proof
- requires_lane_dispatch_claim_before_gpu
- requires_exact_cuda_auth_eval

This ledger records no score claims and no GPU dispatch. The output is suitable
only for choosing which byte streams deserve a future byte-equivalent codec
transform and archive preflight.

## Files

- `src/tac/optimization/entropy_rate_decomposition.py`
- `tools/audit_entropy_rate_decomposition.py`
- `src/tac/tests/test_entropy_rate_decomposition.py`

## Verification

Commands run:

```text
.venv/bin/python -m pytest src/tac/tests/test_entropy_rate_decomposition.py
.venv/bin/ruff check src/tac/optimization/entropy_rate_decomposition.py src/tac/tests/test_entropy_rate_decomposition.py tools/audit_entropy_rate_decomposition.py
```

Results:

- pytest: 5 passed
- ruff: all checks passed

The focused tests cover deterministic output, conditional grouping, invalid
counts, CLI artifact writing, and no dispatch readiness.
