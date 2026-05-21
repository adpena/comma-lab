# Codex Findings - ATW2 CDF Batch Candidate Compactor

**Timestamp (UTC)**: 2026-05-21T06:05:38Z  
**Scope**: Add scanner-driven batch compaction for parseable ATW2 `archive.zip` candidates under arbitrary roots.  
**Verdict**: BATCH_ACTUATOR_LANDED_AND_VERIFIED

## Summary

Codex added `tools/compact_atw2_cdf_candidates.py`, a full-candidate actuator for the ATW2 CDF removal lane. It scans roots for parseable ATW2 `archive.zip` candidates, compacts each candidate's current-runtime-dead `cdf_table_blob`, writes compacted ZIPs under a deterministic output directory, and emits JSON/Markdown custody reports.

The tool is intentionally non-promotional:

- `score_claim`: false
- `promotion_eligible`: false
- `ready_for_exact_eval_dispatch`: false

## Code Change

- `tools/compact_atw2_cdf_candidates.py`
  - Scans arbitrary roots via `scan_atw2_cdf_candidates`.
  - Compacts each discovered candidate through `compact_atw2_cdf_table_in_archive_zip`.
  - Writes outputs as `candidate_<index>_<source_sha16>/archive.zip`.
  - Reports total ZIP bytes saved, per-candidate raw parity, failures, and scan details.
- `src/tac/substrates/atw_codec_v2/tests/test_cdf_dead_section.py`
  - Adds CLI coverage using a synthetic stored `archive.zip` candidate.

## Verification

Focused test command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/substrates/atw_codec_v2/tests/test_atw_codec_v2.py \
  src/tac/substrates/atw_codec_v2/tests/test_cdf_dead_section.py
```

Result:

- `41 passed in 17.15s`

Batch actuator run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/compact_atw2_cdf_candidates.py \
  experiments/results/atw2_cdf_one_command_smoke_20260521T060022Z/source \
  --output-dir experiments/results/atw2_cdf_batch_compaction_20260521T060538Z \
  --device cpu \
  > experiments/results/atw2_cdf_batch_compaction_20260521T060538Z.stdout.json
```

## Empirical Results

Batch report:

- Candidates seen: `1`
- Compacted count: `1`
- Failure count: `0`
- Total ZIP bytes saved: `2,398`

Source candidate:

- Path: `experiments/results/atw2_cdf_one_command_smoke_20260521T060022Z/source/archive.zip`
- Bytes: `90,664`
- SHA-256: `dbde1713566a06538a6e718fca8d855b190cb3581d3ddf05d4bfa205c94a4f0f`
- Member: `0.bin`

Compacted candidate:

- Path: `experiments/results/atw2_cdf_batch_compaction_20260521T060538Z/candidate_0000_dbde1713566a0653/archive.zip`
- Bytes: `88,266`
- SHA-256: `dff8e3f2ecfb06fcb787bfcc2954ceb314971469639a4d188b6cb5c8c65fb57f`

Raw parity:

- Raw equal: `true`
- Max absolute raw byte delta: `0`
- Source raw SHA-256: `c2c6e18c3ca3706d437126a5fd632be9d8e36eb88b8259353db17ddafc4c88fd`
- Compact raw SHA-256: `c2c6e18c3ca3706d437126a5fd632be9d8e36eb88b8259353db17ddafc4c88fd`

Ignored proof artifact hashes:

- `experiments/results/atw2_cdf_batch_compaction_20260521T060538Z/batch_compaction_report.json`: `e386fa380cbf41f40cbddc2f22ef72386c32cebc39892a603f0bbc135d9019a6`
- `experiments/results/atw2_cdf_batch_compaction_20260521T060538Z/batch_compaction_report.md`: `734dda12d91a0aa937ddb1492c8fc9aff46e46d5ee6f5de9b7708dcf584ab200`
- `experiments/results/atw2_cdf_batch_compaction_20260521T060538Z.stdout.json`: `e386fa380cbf41f40cbddc2f22ef72386c32cebc39892a603f0bbc135d9019a6`

## Interpretation

The ATW2 CDF removal lane now has two operator surfaces:

1. `tools/materialize_atw2_cdf_compaction_smoke.py` for one-command smoke materialization and proof.
2. `tools/compact_atw2_cdf_candidates.py` for scanner-driven compaction over any existing ATW2 candidate roots.

This closes the remaining hand-run gap for future full ATW2 artifacts. It still does not prove contest score movement because the empirical run here uses the materialized smoke archive, not a full 600-pair contest candidate.
