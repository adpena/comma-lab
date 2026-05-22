# Codex Findings: MLX Parent Contract Closure Terminal Check

Timestamp: 2026-05-22T12:54:11Z

## Scope

Terminal check for the MLX auth-scorer local-acceleration lane after the
decoder-q and FEC6 full-parent response contracts were rebuilt on the same
600-pair contest CPU axis. This memo records the refreshed closure plan and
the live adversarial-review disposition. It does not create score, promotion,
rank/kill, or dispatch authority for MLX artifacts.

## Refreshed Closure Artifacts

- Closure plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_plan_20260522T1113Z/decoderq_parent_contract_closure_plan.json`
- Closure plan markdown:
  `experiments/results/mlx_decoderq_parent_contract_closure_plan_20260522T1113Z/decoderq_parent_contract_closure_plan.md`
- Production bundle:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1113Z/mlx_parent_contract_bundle_with_candidate.json`
- Parent production plan:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1113Z/parent_production_contract_plan_after_candidate.json`
- Same-axis window dataset:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1113Z/candidate_same_axis_window_response_dataset.json`
- Score calibration:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1113Z/candidate_score_calibration_cpu.json`

## Empirical State

- Refreshed closure plan status: `ready_for_refresh`
- Refreshed closure plan next blocker: `null`
- Production bundle verdict: `PASS_MLX_SCORER_PRODUCTION_CONTRACT_BUNDLE`
- Parent production plan status: `strict_pass`
- Same-axis window dataset row count: `1200`
- Same-axis window dataset skipped row count: `0`
- Score calibration row count: `2`
- Score calibration rank inversions: `0`
- Conservative recommended MLX spend-triage gap:
  `7.375772066442465e-06`

Full-parent MLX/auth residuals:

- FEC6: MLX `0.1920527920355189` vs contest CPU auth
  `0.1920513168811056`; residual `1.475154413288493e-06`
- Decoder-q: MLX `0.1924459939299716` vs contest CPU auth
  `0.19244523120613244`; residual `7.627238391705315e-07`

## Adversarial Review Disposition

Read-only review agents surfaced several historical MLX conformance risks:

- Bundle-level `passed`/`verdict` ignored by LL production gate.
- Coarse row identity could let one contract cover a different row.
- Scorer cache arrays could drift after audit stamping.
- Score calibration could bind to a different response.
- Reference-cache PyTorch parity could be absent.
- Profile stability could certify metadata without value/component binding.
- Partial-window artifacts could masquerade as full-parent calibration.

Current `main` already contains fail-closed protections for these cases:

- `build_mlx_production_contract_gate()` and bundle-gate paths reject failed
  bundle verdicts and failed child contracts.
- Row/contract matching checks archive SHA, inflated aggregate SHA, pair
  window, batch/sample count, candidate/reference cache array hashes, and
  component hashes.
- `load_scorer_input_cache()` recomputes array SHA-256 and artifact SHA-256
  before scoring; stale or mutated cache files fail before MLX inference.
- Score calibration requires a full response row matching archive identity,
  inflated output identity, response family, window, sample count, score,
  component means, component hashes, and both cache identities.
- Production contracts require candidate and reference parity evidence.
- Profile stability binds the recommended row back to response score,
  component means, device, batch size, and component hashes.
- Closure planning requires full 600-pair parent calibration unless an explicit
  research override is passed, and the calibration CLI fails closed on partial
  planner rows by default.

## Verification

Commands run:

```bash
ruff check src/tac/local_acceleration/mlx_scorer_response.py \
  src/tac/local_acceleration/mlx_production_contract.py \
  src/tac/optimization/scorer_response_dataset.py \
  tools/plan_mlx_parent_contract_closure.py \
  tools/build_mlx_window_response_dataset.py \
  tools/calibrate_mlx_scorer_response_scores.py \
  src/tac/tests/test_mlx_score_calibration.py \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/tests/test_mlx_scorer_response.py \
  src/tac/tests/test_mlx_production_contract.py \
  src/tac/tests/test_plan_ll_scorer_response_next_cli.py \
  tests/test_plan_mlx_parent_contract_closure.py
```

Result: `All checks passed!`

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/tests/test_mlx_score_calibration.py \
  src/tac/tests/test_mlx_scorer_response.py \
  src/tac/tests/test_mlx_production_contract.py \
  src/tac/tests/test_plan_ll_scorer_response_next_cli.py \
  tests/test_plan_mlx_parent_contract_closure.py -q
```

Result: `165 passed in 5.40s`

## Remaining Risk

The MLX artifacts are strong local spend-triage and training-acceleration
signals, not contest-authoritative scores. Exact contest CPU/CUDA auth eval
remains required for promotion, ranking, kill decisions, and public claims.

The worktree also contains unrelated dirty partner/operator files. This memo
intentionally does not stage or absorb those changes.
