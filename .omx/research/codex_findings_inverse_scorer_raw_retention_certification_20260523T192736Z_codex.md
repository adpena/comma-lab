# Codex Findings - Inverse Scorer Raw Retention Certification

UTC: 2026-05-23T19:27:36Z
Lane: `lane_artifact_retention_inverse_scorer_raw_parity_20260523`

## Finding

Full-frame IAS1 inflate parity creates a safe opportunity to reclaim raw
inflated output trees, but only if the proof is stronger than a local byte
match. The retention layer needs to know that the output tree is full-frame,
that source and candidate outputs are byte-identical, that the proof and
runtime/archive references survive deletion, and that the parity artifact is
not score or promotion authority.

## Fix

- `comma_lab.artifact_retention` now recognizes
  `inverse_scorer_inflate_parity_raw_output` candidates.
- Certification requires the strict
  `inverse_scorer_cell_inflate_parity_probe_v1` schema, full-frame scope,
  byte-identical source/candidate output trees, empty parity blockers, retained
  workdir proof, descriptor hashes, archive/runtime rebuild records, and false
  authority flags.
- Retention revalidation re-runs the inverse-scorer certifier immediately
  before delete/move.
- The certifier fails closed on proof/tree blockers, symlinked references,
  references that would be deleted with the raw tree, and missing archive or
  runtime rebuild records.

## Verification

- `src/tac/tests/test_artifact_retention.py`
  `src/tac/tests/test_exact_dispatch_authority.py`
  `src/tac/tests/test_inverse_scorer_cell_materializer.py`
  `src/tac/tests/test_optimizer_exact_readiness.py`: 96 passed.
- Integrated focused slice
  `src/tac/tests/test_inverse_scorer_cell_materializer.py`
  `src/tac/tests/test_artifact_retention.py`
  `src/tac/tests/test_optimizer_exact_readiness.py`
  `src/tac/tests/test_exact_dispatch_authority.py`
  `src/tac/tests/test_byte_shaving_campaign_queue.py`: 127 passed.
- `git diff --check`: passed.

## Authority

This is storage custody only. It can reclaim certified rebuildable raw output
bulk, but it does not claim score, promotion eligibility, rank/kill authority,
or exact-eval dispatch readiness.
