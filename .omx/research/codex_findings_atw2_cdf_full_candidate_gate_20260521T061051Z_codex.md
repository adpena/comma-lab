# Codex Findings - ATW2 CDF Full-Candidate Gate

**Timestamp (UTC)**: 2026-05-21T06:10:51Z  
**Scope**: Prevent the ATW2 CDF batch compactor from treating smoke/small archives as full contest candidates.  
**Verdict**: FALSE_AUTHORITY_GATE_LANDED

## Summary

Codex added candidate classification to the ATW2 CDF scanner and batch compactor. Candidate reports now include parsed `num_pairs`, `candidate_class`, and `full_candidate` fields derived from the ATW2 payload itself.

The batch compactor now supports:

```bash
--full-candidate-only
```

That gate compacts only payloads with `num_pairs >= 600`, while still reporting skipped smoke/small candidates in the scan report.

This is a false-authority fix: local smoke archives remain useful for proof and CI, but they are now mechanically distinguishable from full contest candidate archives before any exact-eval routing.

## Code Change

- `tools/scan_atw2_cdf_compaction_candidates.py`
  - Parses candidate payloads with `parse_archive`.
  - Adds `num_pairs`, `candidate_class`, and `full_candidate` to every candidate row.
  - Markdown reports now show class and pair count.
- `tools/compact_atw2_cdf_candidates.py`
  - Adds `--full-candidate-only`.
  - Adds `scan_candidates_found`, `full_candidates_seen`, `non_full_candidates_seen`, and `skipped_non_full_candidate_count` to reports.
  - Adds candidate class and pair count to each compacted row.
- `src/tac/substrates/atw_codec_v2/tests/test_cdf_dead_section.py`
  - Verifies smoke/small candidate classification.
  - Verifies `--full-candidate-only` skips smoke/small candidates.

## Verification

Focused test command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/substrates/atw_codec_v2/tests/test_atw_codec_v2.py \
  src/tac/substrates/atw_codec_v2/tests/test_cdf_dead_section.py
```

Result:

- `42 passed in 16.83s`

Unrestricted batch command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/compact_atw2_cdf_candidates.py \
  experiments/results/atw2_cdf_one_command_smoke_20260521T060022Z/source \
  --output-dir experiments/results/atw2_cdf_batch_classification_20260521T061051Z/all \
  --device cpu \
  > experiments/results/atw2_cdf_batch_classification_20260521T061051Z.all.stdout.json
```

Full-candidate-only command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/compact_atw2_cdf_candidates.py \
  experiments/results/atw2_cdf_one_command_smoke_20260521T060022Z/source \
  --output-dir experiments/results/atw2_cdf_batch_classification_20260521T061051Z/full_only \
  --device cpu --full-candidate-only \
  > experiments/results/atw2_cdf_batch_classification_20260521T061051Z.full_only.stdout.json
```

## Empirical Results

Unrestricted batch:

- `scan_candidates_found`: `1`
- `candidates_seen`: `1`
- `full_candidates_seen`: `0`
- `non_full_candidates_seen`: `1`
- `skipped_non_full_candidate_count`: `0`
- `compacted_count`: `1`
- `total_archive_zip_bytes_saved`: `2,398`
- Candidate `num_pairs`: `8`
- Candidate `candidate_class`: `smoke_or_small_candidate`
- Candidate `full_candidate`: `false`
- Raw equal: `true`
- Max raw byte delta: `0`

Full-candidate-only batch:

- `scan_candidates_found`: `1`
- `candidates_seen`: `0`
- `full_candidates_seen`: `0`
- `non_full_candidates_seen`: `0`
- `skipped_non_full_candidate_count`: `1`
- `compacted_count`: `0`
- `total_archive_zip_bytes_saved`: `0`
- Scan candidate `num_pairs`: `8`
- Scan candidate `candidate_class`: `smoke_or_small_candidate`
- Scan candidate `full_candidate`: `false`

Ignored proof artifact hashes:

- `experiments/results/atw2_cdf_batch_classification_20260521T061051Z/all/batch_compaction_report.json`: `0104da3f261582a3e7f6f03d22266a62a0497466954dbd05d7366b2a4442d734`
- `experiments/results/atw2_cdf_batch_classification_20260521T061051Z/all/batch_compaction_report.md`: `6cda9eb9fde0929905d688d0e67b815ae8a2265096d7e159f01731730721f809`
- `experiments/results/atw2_cdf_batch_classification_20260521T061051Z/full_only/batch_compaction_report.json`: `6feb2a6d224522f293ef432e4bf7e74e42f95e53b38a9aaf01b5ddb7b06e7281`
- `experiments/results/atw2_cdf_batch_classification_20260521T061051Z/full_only/batch_compaction_report.md`: `17721a89f5880c8f9bff7e406d94682618c70468158fbd847c5d890b81f6f919`

## Interpretation

The ATW2 CDF removal lane can now safely scan large artifact roots without creating false authority from smoke archives. Full-candidate compaction is mechanically gated on parsed payload structure, not path names or operator memory.

This still does not prove contest score movement; it hardens the path that will be used once a full ATW2 archive exists.
