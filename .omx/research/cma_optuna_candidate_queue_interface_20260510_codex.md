# CMA/Optuna candidate queue interface landing

generated_at: 2026-05-10T13:40:57Z  
operator_scope: CMA-ES/Optuna-style score-lowering sweep infrastructure, no GPU/remote dispatch  
research_only: true

## Finding

Existing A1/PR101 bias, sidecar, CMA, Optuna, and meta-Lagrangian sweep tools
already emit useful local candidates, but their outputs are heterogeneous:

- A1 constrained coordinate search emits per-variant runtime directories and a
  rollup.
- M5/macOS sweeps emit calibrated ranking queues, not contest-CUDA evidence.
- CodecOp CMA/Optuna tools emit CPU planning reports and optional raw section
  payloads, not byte-closed archives.
- `tools/parallel_dispatch_top_k.py` already has the correct paid-dispatch
  fail-closed checks, but there was no small adapter from these sweep artifacts
  into its `top_k` queue shape.

## Landing

Added a reusable fail-closed adapter:

- `src/tac/optimizer/candidate_queue.py`
- `tools/build_optimizer_candidate_queue.py`
- `src/tac/tests/test_optimizer_candidate_queue.py`

The adapter merges supported sources by `candidate_id`, ranks deterministically,
and emits `optimizer_candidate_queue_v1` with `top_k`, `top_k_forensic`, and
`dispatch_ready`. All locally ranked rows default to:

- `ready_for_exact_eval_dispatch=false`
- `score_claim=false`
- `promotion_eligible=false`
- `target_modes=["contest_exact_eval_planning"]`
- explicit blockers for exact readiness, lane claim, non-proxy evidence, and
  archive/runtime custody.

This is an interface landing only. It does not dispatch, does not modify
preflight, and does not touch T1 Modal files.

## Generated Queue

Generated:

`experiments/results/optimizer_candidate_queue_20260510_codex/next_candidate_queue.json`

Inputs:

- `experiments/results/constrained_coord_search_pr101_bias_20260509T142645Z/rollup.json`
- `experiments/results/m5max_sweep_constrained_coord_search_resume_20260509T152200Z/sweep_manifest.json`
- `experiments/results/optuna_pr101_real_substrate_hardened_20260507_codex/optuna_search_report.json`
- `experiments/results/cma_meta_lagrangian_smoke_20260507_codex/report.json`

Queue summary:

- `n_candidates=126`
- `top_k_count=10`
- `dispatch_ready_count=0`

Top next rows by current local ranking:

1. `v_n1_00_n1_00_n1_00` — A1 baseline, macOS/M5-ranked, requires exact eval readiness.
2. `v_n1_00_n0_50_n1_00` — A1 coord variant, macOS/M5-ranked, requires exact eval readiness.
3. `v_n1_00_p0_00_n1_00` — A1 coord variant, macOS/M5-ranked, requires exact eval readiness.
4. `v_n1_00_p0_50_n1_00` — A1 coord variant, macOS/M5-ranked, requires exact eval readiness.
5. `v_p0_00_n1_00_n1_00` — A1 coord variant, macOS/M5-ranked, requires exact eval readiness.

The existing dispatch actuator refused the queue as expected:

`tools/parallel_dispatch_top_k.py --ranked-input experiments/results/optimizer_candidate_queue_20260510_codex/next_candidate_queue.json --top-k 3 --dry-run`

Refusal class: non-dispatch-ready candidates; proxy/macOS evidence blocked;
abstract proxy rows remain behind concrete A1 runtime candidates until a byte-
closed archive/runtime gate promotes them.

## Solver Wire-In Hooks

- Sensitivity-map contribution: N/A, `research_only=true`; this landing only
  normalizes candidate metadata.
- Pareto constraint: explicit planning blockers preserve the Pareto boundary;
  no candidate is promotion-eligible.
- Bit-allocator hook: N/A; no atom weights or bit allocation changed.
- Cathedral autopilot dispatch hook: output is in the existing `top_k` shape
  consumed by `tools/parallel_dispatch_top_k.py`, but `dispatch_ready=[]`.
- Continual-learning posterior update: N/A; no new empirical anchor.
- Probe-disambiguator: N/A; no competing design mode was selected.

## Tests

- `.venv/bin/python -m pytest src/tac/tests/test_optimizer_candidate_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_optimizer_sweep_plugin.py src/tac/tests/test_codec_op_cma_search.py src/tac/tests/test_codec_op_optuna_search.py tests/test_constrained_coord_search_pr101_bias_sidecar.py tests/test_harvest_a1_bias_correction_sweep.py`
