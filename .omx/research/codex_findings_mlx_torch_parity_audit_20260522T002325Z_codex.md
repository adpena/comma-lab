# Codex Findings: MLX Torch Parity Audit

utc: 2026-05-22T00:23:25Z
lane_id: lane_mlx_auth_scorer_training_signal_fidelity_20260521
evidence_axis: [macOS-MLX research-signal]
verdict: PROCEED_WITH_LOCAL_PARITY_GATE

## What changed

- Added `tac.local_acceleration.mlx_scorer_torch_parity`, a reusable PyTorch-vs-MLX scorer parity audit over fixed scorer-input cache windows.
- Added `tools/audit_mlx_scorer_torch_parity.py` as the operator-facing CLI.
- Added `src/tac/tests/test_mlx_scorer_torch_parity.py` with:
  - a real upstream-weight CPU cache-window parity pass,
  - a synthetic failure case for over-tight output thresholds,
  - a CLI invalid-window fail-fast check.

## Signal exposed

The new manifest records:

- PoseNet output max-absolute delta by head and aggregate max.
- SegNet logit max-absolute delta.
- SegNet argmax-difference pixel count.
- PoseNet component distortion when treating PyTorch as reference and MLX as candidate.
- Non-authoritative MLX axis labels and false-authority flags.

This gives the local MLX training stack a cheap gate before using scorer-response signals to prioritize local candidate generation or paid exact-eval spend.

## Operator command

```bash
.venv/bin/python tools/audit_mlx_scorer_torch_parity.py \
  --cache-dir <scorer_input_cache_dir> \
  --output <parity_manifest.json> \
  --repo-root . \
  --device cpu \
  --start-pair 0 \
  --max-pairs 1
```

## Verification

```bash
.venv/bin/python -m ruff check src/tac/local_acceleration/mlx_scorer_torch_parity.py tools/audit_mlx_scorer_torch_parity.py src/tac/tests/test_mlx_scorer_torch_parity.py
.venv/bin/python -m pytest -q src/tac/tests/test_mlx_scorer_torch_parity.py
.venv/bin/python -m pytest -q src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py src/tac/tests/test_mlx_execution_plan.py src/tac/tests/test_mlx_profile_stability.py src/tac/tests/test_profile_mlx_scorer_response_cache.py src/tac/tests/test_mlx_scorer_response.py src/tac/tests/test_mlx_batch_invariance.py src/tac/tests/test_mlx_scorer_fidelity.py src/tac/tests/test_mlx_scorer_torch_parity.py
```

Results:

- Ruff: passed.
- New parity tests: 3 passed.
- Broader MLX/scorer-response suite: 71 passed.

## FEC6 PR101 cache-window anchor

Command:

```bash
.venv/bin/python tools/audit_mlx_scorer_torch_parity.py --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T205900Z_pairs16 --output experiments/results/mlx_scorer_torch_parity_fec6_pr101_pairs16_20260522T002325Z_window0_1.json --repo-root . --device cpu --start-pair 0 --max-pairs 1 --run-id fec6_pr101_pairs16_window0_1
```

Result:

- verdict: `PASS_MLX_TORCH_SCORER_PARITY`
- pair_window: `[0, 1]`
- n_samples: `1`
- PoseNet output max abs delta: `4.76837158203125e-07`
- PoseNet component abs max: `1.6375789613221059e-15`
- SegNet logit max abs delta: `0.0002993345260620117`
- SegNet argmax diff pixels: `0`

Second command:

```bash
.venv/bin/python tools/audit_mlx_scorer_torch_parity.py --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T205900Z_pairs16 --output experiments/results/mlx_scorer_torch_parity_fec6_pr101_pairs16_20260522T002325Z_window0_16.json --repo-root . --device cpu --start-pair 0 --max-pairs 16 --run-id fec6_pr101_pairs16_window0_16
```

Result:

- verdict: `PASS_MLX_TORCH_SCORER_PARITY`
- pair_window: `[0, 16]`
- n_samples: `16`
- PoseNet output max abs delta: `7.62939453125e-06`
- PoseNet component abs max: `9.70575442932331e-12`
- SegNet logit max abs delta: `0.0007464885711669922`
- SegNet argmax diff pixels: `0`

## Remaining work

Run this audit across the FEC6 full-600 scorer-input cache and record the per-window delta distribution. If CPU parity is stable, the next frontier-moving build is to route MLX-local scorer-response training through this parity gate plus the existing fidelity gate before exact-eval dispatch selection.
