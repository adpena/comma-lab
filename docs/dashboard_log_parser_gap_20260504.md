# Dashboard log-parser gap — PR106 frontier score off-dashboard (2026-05-04)

## Discovery via operator briefing

Ran `tools/operator_briefing.py --top 10` for the first state-of-stack
snapshot since the dispatch tooling stack was completed. Phase 2 (score
dashboard) showed top-10 contest scores but the **PR106 frontier score
(0.20945673) was missing**.

## Root cause

PR106's contest_auth_eval was run on Lightning Studio. The result JSON
was written to a Lightning-local path (`/teamspace/studios/this_studio/pact/...`)
and never synced back to the local repo. What IS local:

```
experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_adapter_t4_20260504T1330Z/
├── adjudication.log    ← has SCORE_RECOMPUTED=0.209456736805712
├── archive.zip
└── auth_eval.log       ← has full embedded RESULT_JSON: {...}
```

The dashboard's glob (`experiments/results/**/contest_auth_eval*.json`)
correctly skips `.log` files. The canonical JSON only exists on Lightning.

## What's actually in auth_eval.log

The `auth_eval.log` from `contest_auth_eval.py` runs contains a single
line `RESULT_JSON: {...}` with the full canonical JSON serialized. Example
from the PR106 run:

```
RESULT_JSON: {"schema_version": 1, "final_score": 0.21,
  "avg_posenet_dist": 3.351e-05, "avg_segnet_dist": 0.00067142,
  "rate_unscaled": 0.00496036,
  "score_recomputed_from_components": 0.20945673680571203,
  "canonical_score": 0.20945673680571203,
  "archive_size_bytes": 186239, "n_samples": 600, ...}
```

The `adjudication.log` companion has a stripped-down key=value form:

```
SCORE_RECOMPUTED=0.209456736805712
SCORE_REPORTED_ROUNDED=0.21
LANE_STATUS=IN_PREDICTED_BAND
ARCHIVE_SHA256=3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58
```

## Affected dispatches (off-dashboard)

Lightning-batch dispatches whose JSON output didn't sync back:

```
experiments/results/lightning_batch/exact_eval_public_pr100_*/auth_eval.log
experiments/results/lightning_batch/exact_eval_public_pr105_*/auth_eval.log
experiments/results/lightning_batch/exact_eval_public_pr106_*/auth_eval.log
... and 1 .adjudicated.json variant per dispatch
```

These ARE the most important scores in the repo (the public frontier).
Their absence from the dashboard is a real observability gap.

## Fix options

| # | Option | LOC | Risk | When to do |
|---:|---|---:|---|---|
| 1 | Add `auth_eval.log` parser to score_dashboard.py | ~30 | low | Next /loop tick if shipping |
| 2 | Write `tools/extract_log_to_json.py` to materialize RESULT_JSON | ~50 | low | Pre-process step before dashboard |
| 3 | Sync JSONs from Lightning to local | 0 LOC, manual | none | When operator next visits Lightning |
| 4 | Document the gap (this memo) | — | none | NOW |

Option #4 is the immediate $0 / zero-risk move. Options #1-3 are queued
for explicit decision on the next polish-tick if the operator wants
the dashboard to be authoritative.

## Decision: DOCUMENT (option #4)

Pattern recognition: the dashboard is INFORMATIONAL, not AUTHORITATIVE.
The reports/latest.md is the operator's canonical source-of-truth for the
public frontier; the dashboard is for tracking *new* dispatches that produce
proper JSON. Adding log-parsing complexity to the dashboard would conflate
its design intent (find new scores) with archival lookup (find the canonical
PR106 frontier).

Better: extend the operator briefing's preamble to explicitly link
reports/latest.md as the canonical frontier, complementary to the dashboard's
"recent JSON dispatches" view. This is still option #4 (documentation), just
more proactively surfaced.

## Cross-refs

- score_dashboard.py source: `tools/score_dashboard.py`
- operator_briefing.py source: `tools/operator_briefing.py`
- canonical PR106 score: `reports/latest.md` "Current Floor — NEW EXACT PUBLIC FRONTIER" section
- PR106 auth_eval.log: `experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_adapter_t4_20260504T1330Z/auth_eval.log`
- archive bytes/SHA: 186,239 / `3fefbe5d...` (verified in adjudication.log SHA line)
