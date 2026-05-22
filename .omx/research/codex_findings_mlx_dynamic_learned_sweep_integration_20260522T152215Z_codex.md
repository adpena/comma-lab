# Codex Findings - MLX Dynamic Learned Sweep Integration

Date: 2026-05-22T15:22:15Z

## Summary

Landed a planning-only dynamic sweep surface for local MLX/CPU scorer-response
work. The goal is to remove hand-picked arbitrary configs from bolt-ons,
stackables, and substrate sweeps by representing every candidate/config/pass as
one row with expected-improvement, uncertainty, cost, orthogonality, component
axis, master-gradient, canonical-equation, freezing, waterbucket, and portable
trainer context.

This is explicitly not score authority:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- exact CPU/CUDA rows keep lane-claim, locality/control, and canonical auth
  harvest blockers.

## Integration Contract

Producer inputs:

- LL/scorer-response datasets and OOF rows via explicit `candidates[]`
  payloads.
- DQS1 selector Pareto plans via
  `decoder_q_selective_selector_pareto.v1`.
- Waterbucket/sign-calibration context via candidate
  `waterbucket_context`.
- SegNet/PoseNet/rate component metadata via `component_axis_context`,
  `segnet_context`, and `posenet_context`.
- Master-gradient and per-pair metadata via `master_gradient_priority` and
  `master_gradient_provenance`.
- Orthogonal-stack metadata via `orthogonality_score` and
  `orthogonality_contract`.
- Canonical equation metadata via `canonical_equation_provenance`.
- Freezing/portable-trainer metadata via `freezing_provenance` and
  `portable_trainer_provenance`.

Consumer outputs:

- Local MLX/CPU sweep queues for high-throughput in-vivo search.
- Exact-eval candidate queues only as blocked rows requiring materialization,
  controls, lane claim, auth eval, and harvest.
- Freeze-candidate rows after macro pass only; these are allowed to seed
  baselines or become archive/runtime constants only after reproducible
  observations and exact auth-axis proof.

## Recursive Sweep Policy

Default pass ladder:

1. `smoke`: liveness/sign probe, high exploration, tiny cost.
2. `micro`: local candidate/config learning.
3. `intermediate`: family confirmation and posterior update.
4. `macro`: freeze-candidate pass for configs that survived recursion.

Each observation must be appended with candidate/config/pass ids, component
deltas, axis tag, archive/runtime/cache hashes, and scorer output. The planner
is rerun after each pass so later rows are informed by live evidence.

The CLI also supports `--per-pass-top-k` to preserve a stratified queue across
the pass ladder. This prevents the cheapest smoke rows from crowding out
micro, intermediate, and macro rows when the operator wants the first planned
batch to cover all recursive scales.

## Files

- `src/tac/optimization/mlx_dynamic_learned_sweep.py`
- `tools/plan_mlx_dynamic_learned_sweep.py`
- `src/tac/tests/test_mlx_dynamic_learned_sweep.py`
- `tools/plan_decoder_q_signed_waterbucket.py`
- `src/tac/tests/test_plan_decoder_q_signed_waterbucket.py`

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_mlx_dynamic_learned_sweep.py src/tac/tests/test_plan_decoder_q_signed_waterbucket.py src/tac/tests/test_decoder_q_selective_runtime_packet.py src/tac/tests/test_decoder_q_selective_selector_pareto.py -q
.venv/bin/python -m ruff check src/tac/optimization/mlx_dynamic_learned_sweep.py tools/plan_mlx_dynamic_learned_sweep.py src/tac/tests/test_mlx_dynamic_learned_sweep.py tools/plan_decoder_q_signed_waterbucket.py src/tac/tests/test_plan_decoder_q_signed_waterbucket.py src/tac/optimization/decoder_q_selective_runtime_packet.py tools/plan_decoder_q_selective_runtime_packet.py src/tac/optimization/decoder_q_selective_selector_pareto.py tools/plan_decoder_q_selective_selector_pareto.py src/tac/tests/test_decoder_q_selective_runtime_packet.py src/tac/tests/test_decoder_q_selective_selector_pareto.py
```

Both passed at landing time; the targeted pytest command reports `19 passed`.

## Next Integration Hooks

- Feed strict MLX scorer-response selection rows into
  `tools/plan_mlx_dynamic_learned_sweep.py --candidate-payload`.
- Add an observation-appender helper so local sweep results can update the
  same row schema without ad hoc JSON edits.
- Add a graph builder for sweep rows, calibration uncertainty, exact-eval
  confirmations, and freeze-candidate transitions.
- Add a minimal MLX portable trainer smoke for `grayscale_lut` that emits
  `portable_trainer_provenance` in this schema.
