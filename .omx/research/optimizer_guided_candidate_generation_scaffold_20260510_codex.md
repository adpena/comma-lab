# Optimizer-guided candidate generation scaffold

generated_at: 2026-05-10T00:00:00Z
operator_scope: bounded offline score-lowering candidate generation for A1/PR101 bias and sidecar prefilters
research_only: true

## Landing

Added a reusable dry-run generator for low-dimensional optimizer-guided
candidate queues:

- `src/tac/optimization/optimizer_guided_candidate_generation.py`
- `tools/build_optimizer_guided_candidate_queue.py`
- `src/tac/tests/test_optimizer_guided_candidate_generation.py`

The scaffold emits `optimizer_guided_candidate_queue_v1`, compatible with the
existing `top_k`/`top_k_forensic` planning queue shape. Built-in profiles:

- `pr101_bias_refine`: three runtime-consumed PR101 bias parameters
  (`bias_b`, `bias_g`, `bias_r`) using the existing bias runtime packet schema.
- `pr101_bias_sidecar`: the same PR101 bias anchor plus one bounded valid
  sidecar coordinate on `up[:, 1, 0]`.

Supported proposal strategies are stdlib-only and deterministic with seed:
`grid`, `random`, `cmaes`, and `optuna` (TPE-style fallback, no Optuna required).

## Evidence Boundary

Every row is forced through `tac.optimization.proxy_candidate_contract`:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `target_modes=["contest_exact_eval_planning"]`

The queue contains proxy objectives only. It emits no archive, no inflate
runtime, no dispatch claim, and no exact CUDA evidence. A candidate can affect
score only after a builder consumes `candidate_params`, emits byte-closed
archive/runtime custody, proves runtime consumption, passes exact-readiness
promotion, receives a lane claim, and then lands exact CUDA auth eval.

## Score-Lowering Connection

This replaces ad hoc constants for A1/PR101 bias and sidecar searches with a
bounded ranked queue. Modal, Kaggle, and M5 prefilters can consume
`top_k[*].candidate_params` deterministically and report back proxy evidence
without becoming score authority. The highest-EV path is to materialize only
top-ranked rows through the existing PR101 bias runtime packet path or a future
sidecar-aware builder.

## Solver Wire-In Hooks

- Sensitivity-map contribution: N/A; `research_only=true`, no empirical anchor
  or tensor sensitivity changed.
- Pareto constraint: enforced through the proxy false-authority contract and
  dispatch blockers; no row is promotion-eligible.
- Bit-allocator hook: N/A; this ranks runtime/search parameters, not bit
  allocation policy.
- Cathedral autopilot dispatch hook: queue uses the existing `top_k` shape, but
  `dispatch_ready=[]`; actuators must require exact-readiness promotion.
- Continual-learning posterior update: N/A; no exact CUDA anchor harvested.
- Probe-disambiguator: N/A; all strategies share the same callable profile
  interface and proxy boundary.

## Commands

```bash
.venv/bin/python tools/build_optimizer_guided_candidate_queue.py \
  --profile pr101_bias_sidecar \
  --optimizer cmaes \
  --seed 20260510 \
  --max-candidates 64 \
  --top-k 16 \
  --output experiments/results/optimizer_guided_candidate_queue_20260510_codex/pr101_bias_sidecar_queue.json
```

```bash
.venv/bin/python -m pytest src/tac/tests/test_optimizer_guided_candidate_generation.py
```
