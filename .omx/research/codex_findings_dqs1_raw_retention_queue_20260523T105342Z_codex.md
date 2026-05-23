# Codex Findings - DQS1 Raw Retention Queue Hardening - 2026-05-23T10:53:42Z

## Scope

Adversarial review and hardening pass for the DQS1 local-first queue after the
MLX scorer-response bridge work exposed repeated multi-GB raw inflation/cache
artifacts under:

`experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/`

## Finding

The local-first queue already generated useful local CPU and MLX signal, but raw
inflation scratch was not part of the queue contract. This let successful
candidates leave behind rebuildable `inflated/`, `control.raw`, and scorer-cache
bulk until a manual cleanup pass noticed disk pressure.

Failure class: result signal was canonicalized, but artifact-retention signal was
still operator-memory dependent.

## Landing

`src/comma_lab/scheduler/dqs1_local_first_queue.py` now inserts
`plan_raw_artifact_retention` after `local_cpu_advisory` by default. The step
runs `tools/compact_experiment_artifacts.py <materialized_root> --min-bytes 1`
and requires:

- `plan.blocked_candidate_count == 0`
- `plan.score_claim == false`
- `plan.promotion_eligible == false`
- `plan.ready_for_exact_eval_dispatch == false`

`tools/build_dqs1_local_first_queue.py` exposes
`--skip-raw-retention-plan` only for explicit operator/debug override.

## Empirical Anchors

Two completed candidates were harvested observe-only and their rebuildable bulk
was moved to `/Volumes/APDataStore/pact-tertiary/artifact_cold_store` with
certified journals:

- `pairset_drop_two_r029_020_p0259_0430`: local CPU advisory
  `[macOS-CPU advisory] 0.19204061709818365`, no eureka trigger, 17,480,986,070
  bytes moved.
- `pairset_drop_two_r028_020_p0257_0430`: local CPU advisory
  `[macOS-CPU advisory] 0.19204161709818363`, no eureka trigger, 14,649,816,858
  bytes moved.

Queue rerouted to `pairset_drop_two_r029_023_p0259_0440`.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_artifact_retention.py src/tac/tests/test_dqs1_local_first_autopilot.py`
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/dqs1_local_first_queue.py tools/build_dqs1_local_first_queue.py src/tac/tests/test_dqs1_local_first_queue_builder.py`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`
- `git diff --check`

All passed.

## Remaining Work

The queue now canonicalizes retention planning, but retention execution is still
manual. Next hardening pass should route successful completed candidates through
the same `comma_lab.artifact_retention` execution surface used by
`tools/compact_experiment_artifacts.py`, with cold-store path policy and journal
verification preserved. This should replace the older local-CPU-only scratch
cleanup path in the DQS1 autopilot so locality raw, local CPU raw, extracted
scratch, and MLX scorer caches share one fail-closed artifact-retention contract.
