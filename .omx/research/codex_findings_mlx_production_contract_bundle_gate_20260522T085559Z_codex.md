# Codex Findings: MLX Production Contract Bundle Gate

UTC: 2026-05-22T08:55:59Z
Lane: `lane_mlx_production_contract_bundle_planner_gate_20260522`

## Verdict

PROCEED. The LL scorer-response planner can now consume more than one strict
MLX production contract, but only as non-authoritative local MLX research signal
for exact-eval spend triage. Every MLX row must be covered by a strict child
contract bound to the same archive SHA-256, inflated-output aggregate SHA-256,
batch-pair count, sample count, pair window, and, when present, response/cache
array/component identity.

## Structural Fixes

- Added `mlx_scorer_production_contract_bundle.v1` handling to
  `build_mlx_production_contract_gate`.
- Added bundle-level false-authority checks and explicit `passed=true` plus
  `PASS_MLX_SCORER_PRODUCTION_CONTRACT_BUNDLE` enforcement.
- Changed `tools/plan_ll_scorer_response_next.py --mlx-production-contract` to
  accept repeated contract manifests and synthesize a strict bundle.
- Bound bundle child contracts to MLX planner rows by broad identity:
  archive SHA-256, inflated-output aggregate SHA-256, batch pairs, sample count,
  and pair window.
- Strengthened row coverage with rich identity where present:
  response run id, candidate/reference scorer-input cache array SHA-256 maps,
  PosNet component SHA-256, and SegNet component SHA-256.
- Extended MLX production-contract response summaries to emit those rich
  identity fields so future manifests can satisfy the stricter planner gate.
- Updated markdown rendering so bundle gates show contract count, strict-child
  count, row coverage, and unmatched row ids instead of singleton-only fields.

## Adversarial Review Closure

The xhigh reviewer found three issues:

1. Bundle verdict ignored: fixed by requiring `passed=true` and the bundle pass
   verdict before `strict_pass`.
2. Coarse row identity: fixed by propagating response/cache/component identity
   through production summaries and comparing it for each row.
3. CLI schema drift / missing repeated-contract tests: fixed by importing
   canonical constants and adding repeated-contract pass/fail CLI coverage.

## Verification

- `.venv/bin/ruff check src/tac/local_acceleration/mlx_production_contract.py src/tac/optimization/scorer_response_dataset.py tools/plan_ll_scorer_response_next.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py src/tac/tests/test_mlx_production_contract.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py src/tac/tests/test_build_scorer_response_dataset_cli.py src/tac/tests/test_mlx_production_contract.py`

Result: 108 passed.

## Remaining Non-Authority Boundary

This gate does not make MLX a score authority. It only permits local MLX rows to
rank or filter exact-eval spend candidates after strict parity, calibration, and
production-contract coverage. Contest CPU/CUDA auth eval remains mandatory for
score claims, promotion, rank/kill, and dispatch readiness.
