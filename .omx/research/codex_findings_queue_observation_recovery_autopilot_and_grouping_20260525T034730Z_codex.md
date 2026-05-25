# Codex findings: queue observation recovery autopilot and grouping

UTC: 2026-05-25T03:47:30Z

## Scope

Reviewed and hardened the queue-observation recovery tranche that sits between
materializer campaign observation, queue-health recovery, inverse-steganalysis
queue-health feedback, and the next water-bucket planning pass.

## Landed signal

- Recovery plans now preserve per-step planning metadata instead of reducing
  failures to anonymous queue mutations: target kind, materializer id, receiver
  contract kind, resource kind, candidate/work/backlog/source ids, and expected
  artifact paths.
- Recovery plans now emit grouped blocker summaries under `grouped_blockers`
  and `queue_health_groups`, including repeated-group counts and a
  deterministic planning effect. This gives inverse-steganalysis acquisition a
  structured queue-health prior rather than a flat blocker string.
- Recovery queues now have an explicit validation surface:
  `queue_observation_recovery_queue_validation.v1`. Validation fails closed for
  non-paused controls, non-local concurrency, truthy authority fields, missing
  source custody paths, non-local steps, forbidden wrappers/flags, and source
  queue/state command mismatches.
- The materializer campaign runner can execute the emitted recovery queue only
  through an explicit CLI flag or strict local-autopilot flag. Execution remains
  local-only, bounded by max steps/parallelism/idle limits, records command
  payloads, observes the child queue, and re-observes the source queue after
  mutation.
- Operator briefing now surfaces recovery execution request/execution/success
  counts and stops recommending a recovery queue once the latest recovery
  execution succeeded.

## Safeguards

- Recovery queues are generated paused by default and marked no-score-authority.
- Execution payloads are false-authority records only:
  `score_claim=false`, no promotion authority, and no paid dispatch authority.
- Local autopilot is gated by the same validation helper used by tests and the
  runner, so non-local or malformed queues are refused before worker execution.
- A noncanonical child recovery state path requires an explicit rationale.
- Source queue observation after worker completion is part of the success
  condition; unhealthy source state remains a blocker.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_queue_feedback_replan_policy.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_operator_briefing.py -q`
  - Result: 109 passed.
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/__init__.py src/comma_lab/scheduler/queue_feedback_replan_policy.py tools/run_byte_shaving_materializer_campaign.py tools/operator_briefing.py src/tac/tests/test_queue_feedback_replan_policy.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_operator_briefing.py`
  - Result: all checks passed.
- `git diff --check`
  - Result: clean.

## Remaining work

- Wire the higher-level controller to decide when
  `--queue-observation-recovery-policy-local-autopilot` should be enabled during
  unattended local campaigns.
- After a recovery execution succeeds, immediately feed the source observation
  and grouped queue-health priors into the next acquisition/materializer pass so
  the queue repair changes the next water-bucket decision without operator
  mediation.
- Extend the same recovery grouping to MLX learned-sweep queues once local
  scorer-response campaigns produce queue-observation blockers with materializer
  metadata.
