# Codex Findings: DQS1 Parallel Batch Autopilot

Date: 2026-05-23T14:37:00Z
Author: Codex
Status: landed candidate for commit

## Scope

Advanced the DQS1 local-first byte-shaving loop from a single-candidate
manual outer loop toward bounded batch execution:

- learned pairset combo candidates can now be synthesized from local advisory
  component marginals while preserving false-authority semantics;
- DQS1 harvest can select a specific candidate from a multi-candidate queue;
- autopilot can let one worker round advance multiple independent experiments;
- tranche defaults now build two-candidate local CPU batches with concurrency 2;
- checked-in queue now validates as 3 experiments and 16 steps;
- post-harvest retention defaults to cold-store move, not delete.

## Findings

1. The existing generic experiment queue already has the core DAG/concurrency
   substrate. The bottleneck was DQS1-specific single-candidate orchestration
   and harvest/reroute assumptions.
2. Multi-candidate harvest must not rewrite the queue in place. Batch harvest
   records should be consumed into canonical observations, then the queue should
   be rebuilt from the updated portfolio.
3. Local advisory pairset observations are useful planning signal, but they are
   not score authority. All generated combo candidates remain
   `score_claim=false`, `promotion_eligible=false`, and
   `ready_for_exact_eval_dispatch=false`.
4. Retention defaults matter. A no-signal-loss autopilot should move certified
   rebuildable bulk to tiered cold storage by default and require explicit
   operator intent for delete paths.

## Verification

- `py_compile` passed for touched Python entry points.
- `ruff check` passed for touched implementation and tests.
- `pytest` focused bundle passed: 56 passed.
- `git diff --check` clean.
- `tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`
  returned valid with local max parallel 2.
- Dry tranche smoke:
  `.omx/research/dqs1_local_first_tranche_parallel_batch_smoke_20260523T143652Z.json`.

## Frontier

No score movement in this turn. Canonical frontier scan used by the smoke:

- contest CPU: `0.19202828295713675`
- contest CUDA: `0.20533002902019143`

The change is infrastructure for faster byte-shaving search, not a new
promotion claim.

## Next Integration

The next high-EV step is to compile broader `byte_shaving_campaign_plan.v1`
rows into executable queue experiments beyond DQS1 pairsets, while keeping the
same batch harvest, storage waterfall, MLX/local advisory calibration, and
exact-auth eureka gates.
