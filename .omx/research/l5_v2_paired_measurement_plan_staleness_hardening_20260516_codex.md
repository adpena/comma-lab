# L5 v2 Paired Measurement Plan Staleness Hardening

- date: `2026-05-16`
- agent: `codex`
- scope: L5 v2 paired CPU/CUDA measurement dispatch plan, operator briefing
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Why

The paired measurement dispatch plan is a planning-only artifact derived from
the L5 v2 lattice measurement schedule. If the schedule changes but an older
dispatch plan remains on disk, operator briefing must not silently present the
stale plan as current work. That would recreate the local-minimum failure mode:
valid-looking downstream artifacts whose upstream assumptions have drifted.

## Landed

`tools/operator_briefing.py` now recomputes the referenced source schedule hash
when loading the L5 v2 paired measurement dispatch plan.

It adds blockers when:

- `source_schedule_sha256` is missing;
- the referenced source schedule file is missing;
- the source schedule path is absolute;
- the stored source schedule SHA differs from the current source schedule file.

The briefing payload now includes:

- `paired_measurement_dispatch_plan_source_schedule_sha256`
- `paired_measurement_dispatch_plan_current_source_schedule_sha256`
- `paired_measurement_dispatch_plan_source_schedule_stale`

This keeps paired CPU/CUDA work units tied to the exact schedule that produced
them.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_operator_briefing.py::test_l5_v2_briefing_blocks_stale_paired_measurement_plan src/tac/tests/test_operator_briefing.py::test_briefing_json_skip_pareto_still_surfaces_exact_ready_audit -q` -> `2 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_operator_briefing.py -q` -> `20 passed`
- `.venv/bin/python -m ruff check tools/operator_briefing.py src/tac/tests/test_operator_briefing.py` -> clean
- `git diff --check` -> clean
