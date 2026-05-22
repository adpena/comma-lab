# Codex Findings - Proxy Queue + DQS1 Eureka Hardening - 2026-05-22

## Scope

Adversarial review follow-up on the MLX/local-training optimizer queue and DQS1
local-first scorer loop. Inputs were the Sagan/Averroes/Russell sister-agent
findings and the live rank015 local advisory worker.

## Bugs extincted

1. Non-comparable auth bridge scores from `[macOS-CPU advisory]` and
   `[macOS-MLX research-signal]` can no longer become queue `rank_score`.
   Adapters now only rank from an auth bridge when it is explicitly
   `score_comparable=true` on `contest-CPU` or `contest-CUDA`; advisory scores
   remain visible as advisory metadata.
2. Consumer payload authority validation is recursive, so nested optimizer
   recipes or payload arrays cannot smuggle `ready_for_exact_eval_dispatch`,
   `promotable`, or score-claim flags into Cathedral/autopilot consumers.
3. Candidate queue merge identity now includes family/schema fields so generic
   representation manifests with the same human `candidate_id` do not collapse
   unrelated HNeRV, NeRV-family, non-NeRV, or signal-processing candidates.
4. Score-affecting booleans in merged queue rows are OR-preserved. A
   byte-closed `true` cannot be erased by a proxy `false` that arrives first or
   later.
5. PR95/generic local-training plan schemas route through the same queue
   adapters as their manifest schemas.
6. DQS1 local-first queue completion now requires both a valid
   `local_cpu_advisory.json` and a matching
   `local_cpu_contest_drift_eureka_signal.v1` record. A local advisory alone is
   not enough to advance the queue.

## Rank015 harvest

- Candidate: `pairset_drop_one_rank015_pair0068`
- Archive SHA:
  `2776804063ab15e0a760a0d8c525d3b0828113bf1a8774e310a96baf62152f4e`
- Local score `[macOS-CPU advisory]`: `0.19204028295713674`
- Current contest-CPU frontier used for drift projection:
  `0.19202828295713675`
- Conservative projected contest score: `0.19203328295713673`
- Eureka margin: `-0.000004999999999977245`
- Verdict: `observe_only`, no exact CPU/CUDA spend trigger.

Artifacts:

- `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank015_pair0068_20260522T222011Z.json`
- `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank015_pair0068/local_cpu_advisory.json`
- `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank015_pair0068/locality_controls.json`

## Queue state

After recording rank015's eureka signal, the local-first queue was regenerated.
It skipped ranks 023, 024, 018, 017, 016, 025, and 015 because each now has both
local advisory custody and an eureka record, and selected:

- Next candidate: `pairset_drop_one_rank001_pair0501`
- Queue: `configs/experiment_queues/dqs1_pairset_local_first.yaml`

## Verification

- `pytest` proxy/cathedral/optimizer/training queue suite: passed, 37 tests.
- `pytest src/tac/tests/test_dqs1_local_first_queue_builder.py`: passed, 7 tests.

Remaining high-EV follow-up: make the DQS1 worker emit the eureka signal as an
explicit queue step with a deterministic artifact path instead of relying on the
post-worker harvest loop.
