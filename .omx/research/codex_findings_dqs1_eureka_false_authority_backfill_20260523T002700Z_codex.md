# Codex Findings: DQS1 Eureka False-Authority Backfill

UTC: 2026-05-23T00:27:00Z

## Context

Commit `2251ba904` made DQS1 local-first eureka signals strict: every eureka
payload must explicitly set the full false-authority surface to `false`.
Historical observe-only eureka artifacts predated that expanded surface, so the
queue builder correctly failed closed on the first stale matching artifact
instead of silently rerouting past it.

## Action

Appended new timestamped eureka artifacts for historical completed local
candidates. No old eureka artifact was overwritten.

Backfilled candidates:

- `pairset_drop_one_rank001_pair0501`
- `pairset_drop_one_rank002_pair0109`
- `pairset_drop_one_rank003_pair0479`
- `pairset_drop_one_rank004_pair0098`
- `pairset_drop_one_rank005_pair0467`
- `pairset_drop_one_rank006_pair0544`
- `pairset_drop_one_rank014_pair0492`
- `pairset_drop_one_rank015_pair0068`
- `pairset_drop_one_rank016_pair0229`
- `pairset_drop_one_rank017_pair0242`
- `pairset_drop_one_rank018_pair0588`
- `pairset_drop_one_rank023_pair0440`
- `pairset_drop_one_rank024_pair0112`
- `pairset_drop_one_rank025_pair0026`

Each new artifact is named:

`local_cpu_contest_drift_eureka_<candidate>_20260523T002700Z.json`

## Verification

- Rebuilt the local-first queue from the top-32 action summary without writing;
  it now routes `pairset_drop_one_rank029_pair0259`.
- Checked the latest eureka artifact per completed candidate with
  `tac.optimization.local_cpu_contest_drift.eureka_false_authority_violations`;
  every latest artifact returned no violations.

## Authority

These are `[contest-CPU drift-projected; false authority]` spend-triage
artifacts only. They do not claim scores, promote candidates, rank/kill
candidates, or dispatch exact eval.
