# Codex Findings - Queue Feedback Continuation Queue

UTC: 2026-05-24T21:45:16Z

## Scope

Codex extended the queue feedback replan loop so a successful local materializer
campaign can emit a paused `experiment_queue.v1` continuation queue for the
next materializer campaign iteration. The artifact is not score, promotion,
rank/kill, or paid-dispatch authority; it is a queue-owned local continuation
proposal that must still be resumed deliberately by operator or autopilot
policy.

## Landed Behavior

- Added `queue_feedback_replan_continuation_metadata.v1` and
  `build_queue_feedback_replan_continuation_queue(...)` in
  `src/comma_lab/scheduler/queue_feedback_replan_policy.py`.
- The continuation builder only emits a queue when the policy is a valid
  `run_next_materializer_campaign_iteration` decision, has no blockers, has no
  truthy authority fields, and its next-iteration command is a local Python
  invocation of `tools/run_byte_shaving_materializer_campaign.py` with matching
  plan/action-functional/iteration/max-iteration arguments.
- `tools/run_byte_shaving_materializer_campaign.py` now writes
  `queue_feedback_replan_continuation_queue.json`, emits a child staircase DAG
  and dependent queue reference for the continuation queue, and surfaces the
  path/blockers in the run summary.
- Continuation lane identity is derived in descending authority order:
  explicit runner `--lane-id`, source campaign plan `lane_id`, source queue
  experiment lane, then queue-id fallback.
- `tools/operator_briefing.py` now surfaces continuation-queue presence and
  blocker counts in the high-level byte-shaving acquisition summary.

## Safeguards

- The continuation queue is `controls.mode=paused`, `local_first=true`, and
  constrained to `max_concurrency.local_cpu=1`.
- Continuation metadata carries false-authority fields and dispatch blockers:
  explicit resume required, exact auth eval required before score claim, and
  lane dispatch claim required before paid or remote eval.
- Unsafe command wrappers, remote/provider flags, shell commands, non-local
  Python commands, and policy/command path mismatches all fail closed.
- The reusable builder still requires a lane id; the runner is responsible for
  deriving one from the already-authored campaign artifacts.

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/queue_feedback_replan_policy.py src/comma_lab/scheduler/__init__.py tools/run_byte_shaving_materializer_campaign.py tools/operator_briefing.py src/tac/tests/test_queue_feedback_replan_policy.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_operator_briefing.py`
- `.venv/bin/python -m pytest src/tac/tests/test_queue_feedback_replan_policy.py -q`
  - `13 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_operator_briefing.py -q`
  - `72 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_staircase_dag.py -q`
  - `135 passed`
- `.venv/bin/python tools/review_tracker.py scan`
  - `78235 entities across 5222 files`
- `.venv/bin/python tools/review_tracker.py policy-check src/comma_lab/scheduler/queue_feedback_replan_policy.py`
  - `17 entities compliant, 0 violations`
- `.venv/bin/python tools/review_tracker.py policy-check tools/run_byte_shaving_materializer_campaign.py`
  - `76 entities compliant, 0 violations`
- `.venv/bin/python tools/review_tracker.py policy-check tools/operator_briefing.py`
  - `78 entities compliant, 0 violations`

## Independent Audit Signal

The xhigh read-only audit found that inverse steganalysis is implemented as a
real planning/control plane with false-authority hygiene: scorer inverse
surfaces, discrete action functionals, water-fill, queue performance feedback,
MLX advisory rows, exact-auth calibration hooks, materializer registry
boundaries, and staircase DAG handoffs are all present. The audit also found the
same main gap this landing targets: the loop is not yet a full inverse-scorer
optimum because high-level action cells still too often stop at compiler
requirements or greedy/local materializers.

Ranked next work from the audit:

1. Finish queue-owned feedback continuation so materializer results
   automatically become performance observations, action-functional inputs, and
   next materializer queues.
2. Implement inverse-action operation-set compilers for the top executable
   families: archive-section entropy recode, packet-member recompress, and
   tensor factorize.
3. Replace scalar greedy water-fill with a portfolio optimizer that respects
   conflicts, interactions, materializer feasibility, runtime budgets, and
   exact-auth calibration residuals.
4. Upgrade inverse scorer surface modeling from median buckets to a learned
   multiscale response model trained from MLX rows plus exact CPU/CUDA anchors.
5. Promote MLX/Metal/Accelerate from advisory triage into local
   candidate-training/materializer lanes that emit compiler hints or byte-closed
   candidate packets while remaining false-authority until exact auth.
6. Add a global calibration posterior consumed by acquisition priority, risk,
   and lambda-rate selection.

## Residual Gaps

- The continuation queue is paused by design; a future autopilot policy must
  decide when to resume it.
- This landing does not broaden materializer coverage. It removes loop latency
  and custody ambiguity so compiler/materializer/training work has a queue-owned
  place to land.
- Water-fill remains greedy and local. A portfolio optimizer is the next
  mathematical step if the goal is to approach the full-information theoretical
  floor rather than keep shaving isolated cells.
