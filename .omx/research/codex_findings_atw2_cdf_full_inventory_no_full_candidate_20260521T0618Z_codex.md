# Codex Findings - ATW2 CDF Full-Candidate Inventory

**Timestamp (UTC)**: 2026-05-21T06:18:00Z  
**Scope**: Run the ATW2 CDF scanner and full-candidate-only batch compactor over the local artifact inventory.  
**Verdict**: BLOCKED_NO_FULL_ATW2_CDF_CANDIDATE_FOUND

## Summary

Codex ran the classified ATW2 CDF inventory over:

- `experiments/results`
- `submissions`

The scan found **6 parseable ATW2 `archive.zip` artifacts** across **3,786** ZIPs, but all six are smoke/small artifacts produced by the current ATW2 compaction work. The full-candidate gate found **0** payloads with `num_pairs >= 600`, compacted **0** candidates, and saved **0** bytes.

This is a hard inventory blocker for full-candidate ATW2 CDF compaction. The tooling is ready; the missing object is a full ATW2 candidate archive.

## Commands

Full-candidate-only batch compaction:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/compact_atw2_cdf_candidates.py \
  experiments/results submissions \
  --output-dir experiments/results/atw2_cdf_full_inventory_20260521T0618Z/full_only \
  --device cpu --full-candidate-only \
  > experiments/results/atw2_cdf_full_inventory_20260521T0618Z.full_only.stdout.json
```

Unrestricted classified scan:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/scan_atw2_cdf_compaction_candidates.py \
  experiments/results submissions \
  --json-output experiments/results/atw2_cdf_full_inventory_20260521T0618Z/scan_all.json \
  --md-output experiments/results/atw2_cdf_full_inventory_20260521T0618Z/scan_all.md
```

## Empirical Results

Full-candidate-only report:

- `archives_seen`: `3,786`
- `scan_candidates_found`: `6`
- `candidates_seen`: `0`
- `full_candidates_seen`: `0`
- `non_full_candidates_seen`: `0`
- `skipped_non_full_candidate_count`: `6`
- `compacted_count`: `0`
- `failure_count`: `0`
- `total_archive_zip_bytes_saved`: `0`
- `score_claim`: `false`
- `promotion_eligible`: `false`
- `ready_for_exact_eval_dispatch`: `false`

Unrestricted scan:

- `archives_seen`: `3,786`
- `candidates_found`: `6`
- `zip_errors`: `1`
- `skipped_after_limit`: `false`

All six parseable ATW2 artifacts are classified:

- `candidate_class`: `smoke_or_small_candidate`
- `num_pairs`: `8`
- `full_candidate`: `false`

The single bad ZIP path remains:

- `experiments/results/top_submission_delta_reverse_engineering_20260503/pr_heads/pr65/submissions/henosis_qz_n3z_r25_clean/archive.zip`

## Artifact Hashes

- `experiments/results/atw2_cdf_full_inventory_20260521T0618Z/full_only/batch_compaction_report.json`: `dee17bd94a6249e921546d16717ecf95fe1728bdb15366f8c9bf957bce2c5588`
- `experiments/results/atw2_cdf_full_inventory_20260521T0618Z/full_only/batch_compaction_report.md`: `ea37b0f291a0be5ff3f7d351cdbee0d4011b7f4bbbb6db4481c3e17f6d1ae9dd`
- `experiments/results/atw2_cdf_full_inventory_20260521T0618Z.full_only.stdout.json`: `dee17bd94a6249e921546d16717ecf95fe1728bdb15366f8c9bf957bce2c5588`
- `experiments/results/atw2_cdf_full_inventory_20260521T0618Z/scan_all.json`: `3ea08e0e8fd64565386d35e17bc7d8fa05ed79d04aef254a8f27498df78f25b3`
- `experiments/results/atw2_cdf_full_inventory_20260521T0618Z/scan_all.md`: `1a47b214b963b82406462bcaff04295c17111ca393e1d3cf8fe210c42fd955ff`

## Interpretation

The current ATW2 CDF compaction stack is structurally ready:

1. smoke materialization,
2. scanner classification,
3. unrestricted batch compaction,
4. full-candidate-only gate,
5. exact raw-parity proof on compacted candidates.

There is no local full ATW2 candidate archive to apply it to. The next frontier-moving action is therefore not more CDF tool hardening; it is to produce or harvest a 600-pair ATW2 candidate archive, then rerun:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/compact_atw2_cdf_candidates.py \
  <candidate-root> --device cpu --full-candidate-only
```

Until that archive exists, ATW2 CDF compaction remains blocked at candidate availability, not implementation.
