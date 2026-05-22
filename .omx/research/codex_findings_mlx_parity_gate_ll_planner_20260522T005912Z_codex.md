# Codex Findings: MLX Parity Gate for LL Planner

- UTC: 2026-05-22T00:59:12Z
- Lane: `mlx_parity_gate_ll_planner`
- Verdict: `PROCEED_RESEARCH_SIGNAL_ONLY`
- Score claim: false
- Promotion eligible: false
- Ready for exact eval dispatch: false

## Finding

The LL scorer-response planner was willing to prioritize normalized
`mlx_scorer_response` rows from the dataset without requiring an attached
PyTorch-vs-MLX scorer parity sweep. That was too permissive for the intended
use of MLX as a local training accelerator: the rows are useful only if their
scorer implementation is explicitly gated against upstream PyTorch, and they
must never become score, rank, promotion, or exact-dispatch authority.

## Fix Landed

Added `build_mlx_torch_parity_sweep_gate(...)` in
`tac.optimization.scorer_response_dataset`.

The gate requires:

- schema `mlx_scorer_torch_parity_sweep.v1`
- explicit false `score_claim`, `score_claim_valid`, `promotion_eligible`,
  `ready_for_exact_eval_dispatch`, and `rank_or_kill_eligible`
- `candidate_generation_only=true`
- `requires_exact_eval_before_promotion=true`
- MLX evidence tags matching `macOS-MLX-research-signal`

Planner behavior is now fail-closed:

- MLX rows without a parity sweep emit
  `do_not_use_mlx_rows_without_torch_parity_sweep` and route to
  `ll_mlx_torch_parity_sweep_required`.
- MLX rows with a failed strict sweep emit
  `do_not_use_mlx_rows_after_failed_strict_parity_sweep` and route to
  `ll_mlx_torch_parity_repair_or_override`.
- A failed sweep can be used only with
  `--allow-mlx-parity-research-signal-override`, and remains non-promotional.
- A strict pass enables `ll_mlx_cpu_stable_response_harvest` as local LL
  surrogate training signal only.

## Empirical Context

The current full-300 FEC6 parity sweep remains research-signal, not
promotion-signal:

- 75 windows checked
- 74 passed
- 1 failed at pair window `[156, 160]`
- failure mode: 1 SegNet argmax pixel mismatch out of 786432 pixels
- PoseNet component max remained approximately `9.71e-12`

The gate records these mismatch summaries in the planner output so the planner
can use them for observability while still blocking unqualified promotion.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/scorer_response_dataset.py tools/plan_ll_scorer_response_next.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_mlx_scorer_response.py src/tac/tests/test_mlx_scorer_torch_parity.py`

Result: all targeted checks passed.

## Residual Risk

The current MLX scorer is not yet a strict full-window replacement for upstream
PyTorch because the full-300 sweep still has one SegNet argmax mismatch. This
landing does not claim to fix MLX numerical conformance. It prevents that
non-strict parity state from silently driving LL training or spend filters
without an explicit research-only override.
