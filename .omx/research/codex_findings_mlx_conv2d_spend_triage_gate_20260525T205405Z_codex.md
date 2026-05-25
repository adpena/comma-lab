# Codex Findings: MLX Conv2d Spend-Triage Gate

Captured at: 2026-05-25T20:54:05Z

## Finding

The MLX Conv2d accumulation audit signal had become executable diagnostic
evidence, but LL scorer-response spend triage could still accept a strict MLX
production contract that did not require that numerical-mitigation probe. That
left Kahan/fp64/MLX-determinism evidence at risk of becoming a sidecar result
instead of a planner-owned gate.

## Landing

The LL scorer-response production-contract gate now requires
`required_gates.conv2d_accumulation_probe=true` before MLX rows can filter
exact-eval spend. The gate preserves the probe summary in planner output, and
missing Conv2d accumulation evidence routes the next-probe plan to
`ll_mlx_conv2d_accumulation_probe_required` instead of silently prioritizing
more MLX response harvest.

`tools/plan_ll_scorer_response_next.py` now documents that repeated
`--mlx-production-contract` inputs must include the Conv2d accumulation gate
for spend-triage use.

## Authority Boundary

This is still local `[macOS-MLX research-signal]` only. It does not claim score,
promote, rank/kill, or authorize exact-eval dispatch. The only added authority
is negative planner authority: MLX rows cannot be used as exact-eval spend
filters until the numerical-mitigation proof is present and strict.

## Verification

- `.venv/bin/ruff check src/tac/optimization/scorer_response_dataset.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_mlx_production_contract.py`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_mlx_production_contract.py -q`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_mlx_scorer_torch_parity.py -q`
- `PYTHONPATH=. .venv/bin/python -m pytest src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_mlx_production_contract.py src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_mlx_scorer_torch_parity.py -q`

Focused result: 142 scorer-response/production-contract tests passed and 36
MLX spend-triage/parity tests passed; the combined affected MLX gate suite was
178 passed.

## Next Gap

The next higher-EV MLX follow-up is downstream scorer-response measurement:
run full PR95 decoder/score-surface probes with Kahan/fp64 substitutions where
the Conv2d accumulation probe predicts scale-conditional improvement, then feed
that result into the same effective spend-triage gate.
