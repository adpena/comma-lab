# Codex Findings: Repair MLX Custody Gate

## Summary

Codex tightened the repair campaign scorer's executable-local path. A row is no
longer `ready_for_local_mlx_advisory_execution` merely because a couple of
paths exist. The gate now treats MLX advisory custody as real custody:

- `local_mlx_response_path` and `reference_local_mlx_response_path` must resolve
  to live files, not directories or symlinks;
- JSON local-custody artifacts must parse as objects and carry no truthy score,
  promotion, rank/kill, dispatch, or exact-eval authority fields;
- non-path `required_local_artifacts` from the repair-family prior are now
  checked as first-class evidence values, so family-specific requirements such
  as `segnet_class_region_mask_ids` cannot be silently skipped;
- the execution gate emits `local_mlx_custody_values` alongside path custody so
  downstream queues can see exactly which non-path artifacts were required;
- missing local evidence is named precisely as
  `<artifact_key>:missing_or_empty`, `<path_key>:path_is_symlink`,
  `<path_key>:path_not_file`, or `<path_key>:false_authority_violation:*`.

## Why It Matters

Week 2 needs repair optimization to be executable, not advisory prose. Before
this landing, the default campaign scorer carried the action-functional terms,
entropy position, interaction scope, receiver proof status, and legal/runtime
constraints, but the "ready" local path could still be under-specified. A
SegNet class-region row could omit the class-region mask IDs and still be
selected if the MLX response files existed.

That was too weak for queue-owned repair optimization. The scorer now enforces
the family prior's artifact contract before the optimizer can select an
allocation for stackability probing.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/repair_campaign_scorer.py src/tac/tests/test_repair_campaign_scorer.py src/tac/tests/test_repair_campaign_stackability_queue.py src/tac/tests/test_repair_campaign_score_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_campaign_scorer.py src/tac/tests/test_repair_campaign_stackability_queue.py src/tac/tests/test_repair_campaign_score_queue.py src/tac/tests/test_repair_cascade_mlx_probe_queue.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_campaign_*.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`
- `.venv/bin/python tools/review_gate_hook.py`
- `.venv/bin/python tools/lane_maturity.py validate`

## Commit

- `98d4a255e Require real MLX custody for repair scorer`

## Remaining Work

The next closure target is the materialization side: selected repair allocations
should not only prove local MLX advisory custody, but also name or emit the
byte-closed candidate archive, archive-bound runtime consumption proof,
component-response replay manifest, and exact-axis handoff/refusal packet from
the same row lineage.
