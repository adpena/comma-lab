# Default Materializer Discovery Priority Finding

Generated at: 2026-05-26T03:03:22Z

## Summary

The default frontier feedback cycle now discovers materializer feedback from
`.omx/research` when the operator does not provide an explicit
`--frontier-artifact-root`. The operation-materializer bridge also prioritizes
empirical materializer rows ahead of abstract registered backlog seeds, so live
cycles consume measured receiver/materializer signal first.

## Engineering Changes

- `discover_materializer_feedback_payloads(...)` now uses `.omx/research` as
  the default root, matching the other feedback discovery paths.
- Materializer feedback discovery caps candidate feedback files rather than
  every file below the research root, so ordinary memos do not exhaust the cap.
- The operation-materializer bridge sorts selected rows by concrete feedback,
  queue/follow-up signal, executable adapter support, saved bytes, and priority.
- Bridge summaries now expose `selected_source_operation_ids` and
  `selected_materializer_targets` for auditability.
- Regression coverage verifies default research-root discovery with unrelated
  memo files present and verifies empirical materializer feedback is selected
  before generic backlog rows.

## Live Artifact

The live cycle artifact is:

`.omx/research/frontier_rate_attack_feedback_cycle_20260526T_materializer_discovery_priority/`

The default cycle discovered 1 materializer feedback payload from `.omx/research`
after scanning 253 candidate feedback paths. The bridge selected:

- `archive_section_entropy_recode_v1`
- `byte_range_entropy_recode_v1`
- `archive_section_header_elide_v1`
- `archive_section_reorder_v1`

The concrete entropy-recode row is now first. It remains blocked on missing
`archive_path` and `section_manifest`, which is the next queue-owned context
binding gap rather than an orphaned advisory signal.

## Verification

- `ruff check` passed on the touched feedback scheduler, cycle writer, runner,
  and feedback tests.
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q` passed:
  18 tests.
- Live DQS1, receiver-repair, and targeted component-correction queues
  validated successfully.
- `tools/lane_maturity.py validate` passed: 1372 lanes clean.

## Authority Boundary

This patch changes discovery and planning priority only. The materializer work
queue stays fail-closed when required archive/runtime context is absent. No
score, promotion, rank/kill, exact auth, GPU, or dispatch authority is created.
