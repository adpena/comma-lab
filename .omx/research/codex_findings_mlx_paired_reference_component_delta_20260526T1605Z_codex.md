# Codex Findings - MLX Paired Reference Component Delta - 2026-05-26T16:05Z

## Scope

Bounded adversarial review and repair of the frontier targeted-component
correction lane:

- Queue/artifact root:
  `.omx/research/frontier_mlx_paired_reference_cache_reuse_20260526T154742Z/`
- Runtime result root:
  `experiments/results/frontier_mlx_paired_reference_cache_reuse_20260526T154742Z/`
- Candidate:
  `packet_member_merge_cbe7d79124ba`
- Correction family:
  `repair_dynamics_frame0_palette_interaction_waterfill`

## Findings

The stale failure mode was a signal-loss bug at the paired-reference boundary:
older artifacts could reach the response-harvest stage without the
receiver-closed source archive/runtime context needed to compute candidate vs
source component deltas.

Current code recovers receiver-closed source-reference context from the
submission-closure report and closed-source queue before building the queue or
work order. The regenerated queue therefore has:

- `reference_component_eval_available=true`
- `source_archive_path=submissions/robust_current/archive_correct.zip`
- `source_inflate_sh_path=submissions/robust_current/inflate.sh`
- `source_archive_sha256=4dd46fed78ed064bc97c9b3205088e82838c03667394f7936c8ae8d20f9837ab`

The executed queue completed `10/10` steps with no blockers. The paired MLX
and local CPU deltas agree exactly for the receiver-closed rate-only candidate:

- SegNet delta: `0.0`
- PoseNet delta: `0.0`
- Archive byte delta vs source reference: `-258`
- Receiver-closed rate delta: `-0.0001717916099055202`
- Receiver-closed total delta: `-0.0001717916099055202`
- MLX-vs-local-CPU paired lagrangian drift: `0.0`

This is local acquisition signal only. The row remains blocked for exact score
or budget-spend authority by:

- `exact_auth_eval_required_before_score_or_promotion_claim`
- `component_response_harvest_is_local_acquisition_signal_only`
- `exact_axis_component_response_required_before_budget_spend`
- `receiver_runtime_materialization_required_before_budget_spend`

## Code Changes

Added explicit `targeted_component_score_delta_summary.v1` rows to the response
harvest and propagated them into the materializer basis/request surface:

- `local_cpu_score_delta_summary`
- `local_mlx_score_delta_summary`
- `best_local_cpu_score_delta_summary`
- `best_local_mlx_score_delta_summary`

These are false-authority summaries. They make the rate-vs-distortion delta
machine-readable for downstream receiver/materializer work without changing any
promotion, dispatch, or score-claim gate.

Follow-up adversarial audit found that request-level propagation was not enough:
operation-chain budgets and materializer handoff operation params also need the
same paired-delta basis. The summaries are now self-marking false-authority
objects, and the chain budget carries:

- `best_local_cpu_score_delta_summary`
- `best_local_mlx_score_delta_summary`
- `paired_delta_basis`

The materializer handoff preserves those fields in both portfolio evidence and
operation params so receiver-runtime materializers consume the same
rate-vs-distortion budget signal.

## Verification

Commands:

```bash
ruff check src/comma_lab/scheduler/frontier_rate_attack_feedback.py src/tac/tests/test_frontier_rate_attack_feedback.py
.venv/bin/python -m py_compile src/comma_lab/scheduler/frontier_rate_attack_feedback.py src/tac/tests/test_frontier_rate_attack_feedback.py
.venv/bin/python -m pytest -q src/tac/tests/test_frontier_rate_attack_feedback.py -k 'targeted_component_response_harvest_derives_paired_local_mlx_deltas or targeted_component_queue_carries_receiver_closed_reference_eval or targeted_component_queue_recovers_reference_eval_from_closure_report or targeted_component_queue_imports_false_authority_component_response_cache'
.venv/bin/python tools/experiment_queue.py --queue .omx/research/frontier_mlx_paired_reference_cache_reuse_20260526T154742Z/targeted_component_correction_queue.json run-worker --execute --max-steps 10 --max-experiments 1 --max-parallel 2
.venv/bin/python tools/harvest_frontier_targeted_component_correction_response.py --targeted-component-correction-queue .omx/research/frontier_mlx_paired_reference_cache_reuse_20260526T154742Z/targeted_component_correction_queue.json --output .omx/research/frontier_mlx_paired_reference_cache_reuse_20260526T154742Z/targeted_component_correction_response_harvest.json --materialization-requests-output .omx/research/frontier_mlx_paired_reference_cache_reuse_20260526T154742Z/targeted_component_correction_materialization_requests.json --materialization-queue-output .omx/research/frontier_mlx_paired_reference_cache_reuse_20260526T154742Z/targeted_component_correction_chain_materializer_work_queue.json --operation-chain-work-orders-output .omx/research/frontier_mlx_paired_reference_cache_reuse_20260526T154742Z/targeted_component_correction_operation_chain_work_orders.json --operation-chain-queue-output .omx/research/frontier_mlx_paired_reference_cache_reuse_20260526T154742Z/operation_chain_compiler_queue.json --operation-chain-materializer-handoff-output .omx/research/frontier_mlx_paired_reference_cache_reuse_20260526T154742Z/targeted_component_correction_chain_materializer_handoff.json --operation-chain-queue-id frontier_mlx_paired_reference_cache_reuse_20260526t154742z_operation_chain_compiler --materialization-queue-id frontier_mlx_paired_reference_cache_reuse_20260526t154742z_chain_materializer --results-root experiments/results/frontier_mlx_paired_reference_cache_reuse_20260526T154742Z --materialization-candidate-limit 1 --materialization-family-limit-per-candidate 8 --overwrite --repo-root .
.venv/bin/python tools/scan_best_anchor_per_axis.py --format json
```

Focused test result: `4 passed, 36 deselected`.

Frontier scan after this lane is unchanged:

- `[contest-CPU]`: `0.19202828295713675`, archive SHA-256
  `7a0da5d0fc327cba3f7d1387a544fd5ce5f05bc56ecc8e12cd5097141672f4fe`
- `[contest-CUDA]`: `0.20533002902019143`, archive SHA-256
  `9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4`

## Next Action

The next executable step is not another paired-reference diagnosis. It is the
receiver-runtime materializer for the accepted local acquisition row, followed
by full-frame inflate parity, exact-axis component response, and exact auth eval
only after those gates are true.
