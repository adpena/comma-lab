# Codex Findings: Targeted Chain Context Closure Portfolio

Date: 2026-05-26T09:44:00Z

## Verdict

The targeted correction loop was still too manual after accepted local
component-response rows entered operation-chain work orders. Registered
materializer targets were present, but the queue did not emit a durable
per-target closure plan naming the exact receiver/runtime fields needed before
execution, replay, or budget spend.

## Landed Change

- Expanded targeted correction chains from packet-only rate targets into a
  registered portfolio spanning packet-member, archive-section, tensor, and
  high-level inverse-steganalysis materializer targets.
- Added per-target context closure plans with required, provided, and missing
  fields plus receiver proof requests. Parser-only proof remains explicitly
  rejected.
- Preserved those closure plans through final-byte contexts and materializer
  work rows.
- Added fail-closed queue handoff support for packet-member reorder and
  tensor quantize/prune/shared-codebook receiver contracts, so those registered
  targets no longer fall through as generic adapter-missing rows.

## Authority

All new surfaces remain local queue/planning artifacts:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `budget_spend_allowed=false`

The only rows that may execute locally are those whose materializer context is
already byte-closed. Receiver consumption proof, component replay, exact
readiness, and exact auth eval remain required before any score, rank, kill,
promotion, or budget-spend authority.

## Verification

- `ruff` on touched scheduler/test files: passed
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py`: 32 passed
- `pytest src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_campaign.py`: 118 passed

## Remaining Work

Next highest-EV closure is to generate the missing contracts automatically:
merge contracts, archive section manifests/contracts, tensor sensitivity
contracts, and inverse-action operation-set compilers should be produced from
the same accepted chain budget and scorer/geometry priors instead of filled by
an operator.
