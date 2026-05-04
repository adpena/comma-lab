# Replay Observability Signal Accelerator - 2026-05-03

## Scope

Built an offline observability layer for archive byte streams and exact
component-trace artifacts. This is a planning/signal accelerator only:

- no remote GPU dispatch
- no archive inflate
- no scorer invocation
- no score promotion
- no method retirement

## Artifacts

- Code: `src/tac/archive_signal.py`
- CLI: `experiments/build_replay_observability_signal.py`
- Tests: `src/tac/tests/test_archive_signal.py`
- Default output path: `experiments/results/replay_observability_signal_20260503_codex/`

## Signal Contract

The table joins:

- PR79/S2 exact baseline components from `contest_auth_eval.json`
- PR81 QMA9 static public profile rows
- PR82 Henosis static public profile rows
- optional ZIP member byte profiles
- optional diagnostic `component_trace.json` rows

Every row records `score_claim=false`, `promotion_eligible=false`, evidence
grade, bytes at stake, rate-equivalent priority, and the exact-eval gate note.
Component-trace pair rows convert first-order component contribution into
rate-equivalent break-even bytes using the contest rate coefficient
`25 / 37545489`.

## Dispatch Guidance

The output is allowed to guide which archive stream or pair atom should be
studied next. It is not dispatch authority. Before any exact CUDA run, the
operator must still claim the lane with `tools/claim_lane_dispatch.py claim`
and run identical archive bytes through the canonical auth-eval path.
