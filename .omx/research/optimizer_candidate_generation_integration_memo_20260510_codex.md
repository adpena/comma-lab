# Optimizer Candidate Generation Integration Memo

generated_at: 2026-05-10T15:10:00Z
operator_scope: score-lowering optimizer/candidate-selection integration, local only
research_only: true

## Verdict

Do not create another optimizer. The canonical integration path already exists:

1. Local proxy search emits planning artifacts:
   - `tools/codec_op_optuna_search.py` (`codec_op_optuna_search_report_v1`)
   - `tools/codec_op_cma_search.py` (`codec_op_cma_search_report_v1`)
   - `tools/meta_lagrangian_search_cli.py` (`meta_lagrangian_search_v1`)
   - `tools/constrained_coord_search_pr101_bias_sidecar.py` rollups
2. `src/tac/optimizer/candidate_queue.py` normalizes those artifacts into
   `optimizer_candidate_queue_v1`.
3. `tools/build_optimizer_candidate_queue.py` is the operator CLI for that
   adapter.
4. `tools/parallel_dispatch_top_k.py` is the paid exact-eval actuator, and it
   correctly refuses every proxy/planning row until a separate exact-readiness
   gate proves byte-closed archive/runtime custody.

## Existing Files

- `src/tac/optimizer/meta_lagrangian.py` - deterministic CMA-ES planning
  generator and Lagrangian ranker.
- `tools/meta_lagrangian_search_cli.py` - local proxy/predictor search CLI;
  output is explicitly non-promotable.
- `tools/codec_op_cma_search.py` - CMA-ES/random-search over CodecOp params.
- `tools/codec_op_optuna_search.py` - Optuna TPE over integer-heavy CodecOp
  params.
- `src/tac/optimizer/candidate_queue.py` - canonical fail-closed queue adapter.
- `tools/build_optimizer_candidate_queue.py` - queue build CLI.
- `tools/parallel_dispatch_top_k.py` - fail-closed exact-eval fan-out actuator.
- `src/tac/tests/test_optimizer_candidate_queue.py`,
  `src/tac/tests/test_meta_lagrangian.py`,
  `src/tac/tests/test_codec_op_cma_search.py`, and
  `src/tac/tests/test_codec_op_optuna_search.py` - current guard coverage.

## Current Refreshed Queue

Regenerated:

`experiments/results/optimizer_candidate_queue_20260510_codex/next_candidate_queue.json`

Inputs:

- `experiments/results/constrained_coord_search_pr101_bias_20260509T142645Z/rollup.json`
- `experiments/results/m5max_sweep_constrained_coord_search_resume_20260509T152200Z/sweep_manifest.json`
- `experiments/results/optuna_pr101_known_best_probe_20260507_codex/optuna_search_report.json`
- `experiments/results/optuna_pr101_real_substrate_20260507T230716Z/optuna_search_report.json`
- `experiments/results/optuna_pr101_real_substrate_hardened_20260507_codex/optuna_search_report.json`
- `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/codec_op_replacement_manifest.json`

Summary:

- `n_candidates=124`
- `top_k_count=30`
- `dispatch_ready_count=0`

Top current candidate rows are A1/PR101 inflate-time bias coordinate variants
ranked by `predicted_contest_cpu_gha`. They are concrete archive/runtime
variant directories, but still planning-only because exact CUDA readiness has
not been proven.

## Gap

The missing score-lowering bridge is not Optuna/CMA-ES itself. The gap is a
canonical exact-readiness promoter that takes one queue row, verifies:

- byte-closed archive path, bytes, and SHA-256;
- runtime tree SHA / `inflate.sh` closure;
- no proxy-only or macOS-only evidence promotion;
- static pre-submission compliance;
- active lane claim availability before any remote spend;
- terminal output shape acceptable to `tools/parallel_dispatch_top_k.py`.

Until that bridge marks `ready_for_exact_eval_dispatch=true`, the paid actuator
must continue to refuse the queue.

## Next Patch

Add a narrow helper such as `tools/promote_optimizer_candidate_for_exact_eval.py`
backed by `tac.optimizer.candidate_queue` or a sibling module. It should accept
one `candidate_id` from `optimizer_candidate_queue_v1`, run local custody and
pre-submission checks only, and emit a new queue/report with exactly one
promoted row or a fail-closed blocker list. It must not dispatch.

## Exact Next Candidate-Generation Command

```bash
.venv/bin/python tools/build_optimizer_candidate_queue.py \
  --source experiments/results/constrained_coord_search_pr101_bias_20260509T142645Z/rollup.json \
  --source experiments/results/m5max_sweep_constrained_coord_search_resume_20260509T152200Z/sweep_manifest.json \
  --source experiments/results/optuna_pr101_known_best_probe_20260507_codex/optuna_search_report.json \
  --source experiments/results/optuna_pr101_real_substrate_20260507T230716Z/optuna_search_report.json \
  --source experiments/results/optuna_pr101_real_substrate_hardened_20260507_codex/optuna_search_report.json \
  --source experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/codec_op_replacement_manifest.json \
  --output experiments/results/optimizer_candidate_queue_20260510_codex/next_candidate_queue.json \
  --top-k 30
```

This command is local candidate generation only. It feeds the exact CUDA lane
stack by producing the queue shape that `tools/parallel_dispatch_top_k.py`
consumes after exact-readiness promotion.

## Solver Wire-In Hooks

- Sensitivity-map contribution: N/A, `research_only=true`; no empirical anchor
  or tensor sensitivity changed.
- Pareto constraint: enforced as planning blockers and deterministic ranking;
  no row is promotion-eligible.
- Bit-allocator hook: N/A; no allocation policy changed.
- Cathedral autopilot dispatch hook: the queue uses the existing `top_k` shape
  consumed by `tools/parallel_dispatch_top_k.py`, with `dispatch_ready=[]`.
- Continual-learning posterior update: N/A; no exact CUDA anchor was harvested.
- Probe-disambiguator: N/A; no competing optimizer integration mode was chosen.
