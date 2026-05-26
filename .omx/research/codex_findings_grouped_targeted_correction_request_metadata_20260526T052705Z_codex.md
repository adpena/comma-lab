# Codex Findings: Grouped Targeted Correction Request Metadata

UTC: 2026-05-26T05:27:05Z

## Verdict

A sibling edit expanded targeted component response harvesting so one queue
experiment can carry a `correction_requests` list instead of forcing one
experiment per correction family. That is aligned with the operator steer away
from leaf-only/manual flows: one candidate can now preserve multiple correction
requests across region, boundary, pair, batch, and full-video levels.

I kept the patch and added regression coverage rather than reverting it. Missing
work-order or local component artifacts still emit blocked false-authority rows;
they do not become materialization, budget-spend, score, promotion, rank/kill,
or dispatch authority.

## Verification

- Added a regression that a grouped metadata list expands into two response
  harvest rows while preserving operation levels and targeted dimensions.
- The rows remain blocked with
  `targeted_component_correction_response_artifacts_missing`.
- Regenerated live artifact:
  `.omx/research/frontier_rate_attack_feedback_refresh_20260526T_grouped_component_requests/`.
  Its targeted component correction queue now has 1 grouped candidate experiment
  with 5 correction requests, 14 steps, and 4 deduped full local CPU evals.
- The broader targeted feedback suite remains the owning gate.
