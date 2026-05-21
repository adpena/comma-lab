# Codex Findings — ATW2 CDF Real-Archive Compaction Scan

**Timestamp (UTC)**: 2026-05-21T05:47:01Z  
**Scope**: Structural scan for existing `archive.zip` artifacts containing parseable ATW2 payloads eligible for the compact-CDF removal scaffold.  
**Verdict**: BLOCKED_NO_REAL_ATW2_ARCHIVE_FOUND

## Summary

Codex added and ran a structural scanner for ATW2 CDF compaction candidates across local contest artifact roots. The scanner opens each `archive.zip`, checks member names `0.bin` and `x`, and accepts a candidate only when the member parses through the ATW2 archive parser and the `cdf_table_blob` analysis succeeds.

The full scan found **0 parseable ATW2 `archive.zip` members** across **3,780** archives under:

- `experiments/results`
- `submissions`

This means the compact-CDF implementation is ready at the reusable tool level, but there is no current local real ATW2 candidate archive to compact and send toward exact-eval custody.

## Evidence

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/scan_atw2_cdf_compaction_candidates.py \
  experiments/results submissions \
  --json-output experiments/results/atw2_cdf_compaction_scan_20260521T054701Z/scan.json \
  --md-output experiments/results/atw2_cdf_compaction_scan_20260521T054701Z/scan.md
```

Result:

- `archives_seen`: 3780
- `candidates_found`: 0
- `zip_errors`: 1
- `skipped_after_limit`: false
- `score_claim`: false
- `promotion_eligible`: false
- `ready_for_exact_eval_dispatch`: false

Bad ZIP path:

- `experiments/results/top_submission_delta_reverse_engineering_20260503/pr_heads/pr65/submissions/henosis_qz_n3z_r25_clean/archive.zip`

Artifact hashes:

- `experiments/results/atw2_cdf_compaction_scan_20260521T054701Z/scan.json`: `6969c224ef57b20804bbeb6ea72eccc463d868c395a0ef4d4c3d680b556a9049`
- `experiments/results/atw2_cdf_compaction_scan_20260521T054701Z/scan.md`: `274b44f22975f2424e8884f4e54bbc937bbec2b82b5a5e505df1d7e2a4867f1b`

## Interpretation

The prior compact-CDF scaffold remains valid for ATW2 inner bytes and archive.zip rewriting, but this scan blocks a real-candidate compaction landing because no existing local `archive.zip` contains the ATW2 grammar. This is a real inventory blocker, not a method negative.

The next frontier-moving action is therefore one of:

1. Build or harvest a real ATW2 `archive.zip` from the ATW2 trainer path, then run `tools/probe_atw2_cdf_dead_section.py --compact-cdf --output-archive-zip ...`.
2. If ATW2 remains deferred, redirect the removal-paradigm compaction machinery to another substrate with an existing real archive and a parser-visible decode-opaque section.
3. Keep this scanner as the guardrail so future ATW2 archives are auto-discoverable before dispatch or exact-eval promotion.

