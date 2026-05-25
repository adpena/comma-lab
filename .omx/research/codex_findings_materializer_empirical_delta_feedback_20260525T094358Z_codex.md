# Codex Findings - Materializer Empirical Delta Feedback

Timestamp: 2026-05-25T09:43:58Z
Agent: Codex
Lane: `codex_archive_section_receiver_safe_signal_20260525`

## Finding

The queue/runner could now emit receiver-negative family-agnostic materializer
observations, but the planner bridge still had two signal-loss edges:

- `section_recode` byte deltas were preserved in candidate manifests but not
  normalized into the queue artifact's serialized archive-delta fields.
- Generated `family_agnostic_materializer_empirical_observation` rows were
  usable as best-observation signal, but not fully equivalent to
  `materializer_chain_archive_delta` rows for water-bucket blocking and global
  archive-delta feedback.

During wider runner verification, a second bug class surfaced: duplicate
chain/candidate archive-delta rows with negative `saved_bytes` crashed the
feedback action-functional merge path because the merge helper applied a
nonnegative guard to all numeric fields. Negative `saved_bytes` is valid and
important: it records a successful local proof that increased archive bytes.

## Landing

- Queue observation artifact records now derive serialized archive-delta fields
  from `section_recode` when explicit `serialized_archive_delta_*` fields are
  absent.
- Inverse-steganalysis feedback now treats
  `family_agnostic_materializer_empirical_observation` as materializer
  archive-delta feedback for target/materializer/receiver matching,
  water-bucket blocking, and global feedback summaries.
- Materializer archive-delta merge semantics now preserve negative
  `saved_bytes` as rate-cost evidence, while keeping nonnegative guards for
  archive byte counts and observed gain fields.
- Regression coverage proves:
  - positive receiver-satisfied materializer sweeps feed expected gain without
    blocking;
  - receiver-negative materializer sweeps block the matching bucket;
  - queue-observed `section_recode` deltas become serialized archive-delta
    planner signal;
  - duplicated negative chain/candidate deltas merge without crashing and keep
    the rate-cost blocker.

## Authority Contract

All rows remain false authority and planning-only:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

Negative/receiver-blocked materializer feedback may demote, repair, or suppress
local replanning buckets. It is not a score claim, promotion claim, or exact-eval
dispatch authorization.

## Verification

```bash
.venv/bin/python -m ruff check \
  src/comma_lab/scheduler/experiment_queue_observer.py \
  src/tac/optimization/inverse_steganalysis_acquisition.py \
  src/tac/tests/test_experiment_queue_observer.py \
  src/tac/tests/test_inverse_steganalysis_acquisition.py
```

Result: passed.

```bash
PYTHONPATH=. .venv/bin/pytest \
  src/tac/tests/test_experiment_queue_observer.py \
  src/tac/tests/test_inverse_steganalysis_acquisition.py \
  -q
```

Result: 57 passed.

```bash
PYTHONPATH=. .venv/bin/pytest \
  src/tac/tests/test_inverse_steganalysis_acquisition.py::test_materializer_chain_realized_cost_blocks_matching_water_bucket \
  src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_executes_no_paid_inverse_scorer_chain_and_handoff \
  -q
```

Result: 2 passed.

```bash
PYTHONPATH=. .venv/bin/pytest \
  src/tac/tests/test_experiment_queue_observer.py \
  src/tac/tests/test_queue_feedback_replan_policy.py \
  src/tac/tests/test_inverse_steganalysis_acquisition.py \
  src/tac/tests/test_byte_shaving_materializer_campaign_runner.py \
  -q
```

Result: 147 passed.

## Next Integration

The next queue-owned step is to use the repaired empirical-delta feedback as
the demotion/repair signal for packet-level and tensor-level materializer
families, starting with `packet_member_recompress_v1` and `tensor_factorize_v1`.
Those families should emit the same receiver/rate feedback shape so the action
planner can compare byte, section, tensor, and receiver costs in one
water-bucket surface.
