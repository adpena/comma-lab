# Dashboard log-parser gap — PR106 frontier score off-dashboard (2026-05-04)

> **STATUS: RESOLVED** (option #1 implemented at commit dbb0032d; regression
> tests pinned at commit b3e07b24). Dashboard now shows 357 scores (up from
> 64), with PR106 frontier at #1-2. The "Decision: DOCUMENT" section below
> reflects the original tick's decision; the next tick reversed it after
> seeing the operator briefing actually use the dashboard. Keeping the
> original analysis for the cost-benefit reasoning trail.


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

## Decision: DOCUMENT (option #4) — REVERSED next tick to option #1

Original tick (this memo, commit 9a0d1652): chose option #4 (document the
gap, defer fix) on the reasoning that "dashboard is INFORMATIONAL, not
AUTHORITATIVE" — adding log-parsing would conflate design intent.

Next-tick reversal (commit dbb0032d): implemented option #1 (extend
score_dashboard.py to parse RESULT_JSON from auth_eval.log) after running
the operator briefing for real and seeing PR106 frontier missing. The
30-LOC change unblocked 293 additional scores (5.6× visibility increase).
The empirical impact dwarfed the implementation cost, justifying the
reversal.

Lesson on the reversal: "informational not authoritative" was a *theory*;
running the tool against real data revealed it WAS being implicitly used
authoritatively (the operator briefing's Phase 2 is treated as the score
ranking). When the gap between intent and use is visible at runtime, fix
the use rather than rationalizing the intent.

Regression coverage (commit b3e07b24): 4 tests in
`src/tac/tests/test_score_dashboard_log_fallback.py` pin behavior:
  - test_dashboard_picks_up_log_files
  - test_dashboard_pr106_frontier_surfaces (asserts 0.20945673 + 186,239b)
  - test_dashboard_canonical_json_takes_precedence
  - test_dashboard_log_parser_extracts_components

## Cross-refs

- score_dashboard.py source: `tools/score_dashboard.py`
- operator_briefing.py source: `tools/operator_briefing.py`
- canonical PR106 score: `reports/latest.md` "Current Floor — NEW EXACT PUBLIC FRONTIER" section
- PR106 auth_eval.log: `experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_adapter_t4_20260504T1330Z/auth_eval.log`
- archive bytes/SHA: 186,239 / `3fefbe5d...` (verified in adjudication.log SHA line)
