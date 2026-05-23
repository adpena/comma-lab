# Codex Findings - Inverse Scorer Exact-Eval Queue Bridge

UTC: 2026-05-23T19:47:03Z
Lane: `lane_inverse_scorer_exact_eval_queue_bridge_20260523`

## Finding

The IAS1 chain had cleared receiver and full-frame inflate parity custody, but
the next gate still required manual JSON assembly before the normal exact-eval
readiness machinery could consume it. That made the chain easy to inspect but
not yet a reusable dispatch-preparation artifact.

## Fix

- Added `tac.optimization.inverse_scorer_exact_eval_queue`, which converts a
  verified `inverse_scorer_cell_candidate_chain_v1` manifest into an
  `optimizer_candidate_queue_v1` source queue plus an archive manifest.
- Added `tools/build_inverse_scorer_exact_eval_queue.py` as the operator-facing
  bridge.
- The bridge refuses chains without receiver success, strict full-frame parity,
  a hashed parity-probe artifact, parity JSON content proving nonempty
  byte-identical full-frame output, runtime binding to the parity-probed
  `inflate.sh`, false authority fields, a source archive diff, runtime
  `inflate.sh`, and `report.txt`.
- The produced source row keeps IAS1 score authority false while recording
  archive SHA/byte deltas as exact-readiness change proof.
- Adversarial review found four pre-dispatch authority holes: dummy parity JSON
  could pass, runtime submission dirs were not bound to the parity proof,
  unrelated readiness blockers could survive into promotion, and repo-external
  paths could enter custody records. The bridge and exact-readiness gate now
  fail closed on those cases.

## Real Artifact

- Source queue:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/inverse_scorer_exact_eval_queue_20260523T195836Z/source_queue.json`
- Archive manifest:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/inverse_scorer_exact_eval_queue_20260523T195836Z/archive_manifest.json`
- Exact-ready queue:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/inverse_scorer_exact_eval_queue_20260523T195836Z/exact_ready_queue.json`
- Readiness report:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/inverse_scorer_exact_eval_queue_20260523T195836Z/exact_readiness_report.json`
- Exact-ready queue SHA-256:
  `d472bee7d2bf8283b9f40d524272818c0025160b28ece1f2e7a7af69134dad99`

## Verification

- `src/tac/tests/test_inverse_scorer_exact_eval_queue.py`
  `src/tac/tests/test_optimizer_exact_readiness.py`
  `src/tac/tests/test_byte_shaving_campaign_queue.py`
  `src/tac/tests/test_dqs1_local_first_queue_builder.py`
  `src/tac/tests/test_experiment_queue.py`: 144 passed.
- `tools/promote_optimizer_candidate_for_exact_eval.py` promoted
  `ias1_runtime_parity_top4_20260523T1930Z` to `dispatch_ready_count=1` with
  `score_claim=false`.
- `tools/audit_exact_ready_queues.py --format json` passed with
  `stale_ready_row_count=0`.

## Authority

This bridge prepares a candidate for exact-eval dispatch after a lane claim. It
does not launch Modal/Lightning, claim score, promote, rank, or kill.
