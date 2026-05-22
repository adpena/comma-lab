# Codex Findings: MLX Torch Parity Audit Greenup

**UTC:** 2026-05-22T00:25:44Z
**Lane:** `lane_mlx_auth_scorer_production_hardening_20260521`
**Scope:** PyTorch-vs-MLX scorer parity on fixed scorer-input cache windows.

## Finding

The next production-hardening gap after MLX response/profile/window contracts was direct implementation parity against the upstream PyTorch `DistortionNet`. Batch-invariance and profile-stability catch local consistency, but they do not directly compare the MLX scorer port against upstream scorer outputs on a fixed cache window.

## Fix

- Added `src/tac/local_acceleration/mlx_scorer_torch_parity.py`.
- Added `tools/audit_mlx_scorer_torch_parity.py`.
- Added `src/tac/tests/test_mlx_scorer_torch_parity.py`.
- The parity manifest records PoseNet output deltas, SegNet logit deltas, SegNet argmax pixel drift, and recomputed component-distance deltas.
- The manifest keeps all score/promotion/rank/dispatch authority fields false and requires exact auth eval before promotion.
- GPU parity audits require explicit `--allow-gpu-research-signal`; otherwise they fail closed with the existing MLX GPU research-signal blocker.
- The CLI catches validation/runtime errors and returns rc=2 without writing misleading manifests.

## Real FEC6 CPU Window Evidence

```bash
tmpdir=$(mktemp -d)
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/audit_mlx_scorer_torch_parity.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --repo-root . \
  --device cpu \
  --start-pair 16 \
  --max-pairs 4 \
  --run-id fec6_pr101_cpu_pairs16_20 \
  --output "$tmpdir/parity.json"
```

Observed:

- `passed=true`
- `verdict=PASS_MLX_TORCH_SCORER_PARITY`
- `score_claim=false`
- `requires_exact_eval_before_promotion=true`
- `pair_window=[16,20]`
- `n_samples=4`
- `posenet_output_abs_max=0.00000762939453125`
- `posenet_component_abs_max=9.702655345833477e-12`
- `segnet_logit_abs_max=0.0004553794860839844`
- `segnet_argmax_diff_pixels=0`
- `segnet_component_diff_samples=0`

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_mlx_scorer_torch_parity.py
# 5 passed in 1.63s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check \
  src/tac/local_acceleration/mlx_scorer_torch_parity.py \
  src/tac/tests/test_mlx_scorer_torch_parity.py \
  tools/audit_mlx_scorer_torch_parity.py
# All checks passed!

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_mlx_scorer_torch_parity.py \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/tests/test_plan_ll_scorer_response_next_cli.py \
  src/tac/tests/test_mlx_execution_plan.py \
  src/tac/tests/test_mlx_profile_stability.py \
  src/tac/tests/test_profile_mlx_scorer_response_cache.py \
  src/tac/tests/test_mlx_scorer_response.py \
  src/tac/tests/test_mlx_batch_invariance.py \
  src/tac/tests/test_mlx_production_contract.py \
  src/tac/tests/test_mlx_scorer_adapters.py \
  src/tac/tests/test_mlx_preprocess.py \
  src/tac/tests/test_mlx_scorer_fidelity.py \
  src/tac/tests/test_mlx_cache_audit.py \
  src/tac/tests/test_mlx_scorer_state_map.py \
  src/tac/tests/test_mlx_scorer_port_inventory.py \
  src/tac/tests/test_mlx_to_pytorch_export.py
# 143 passed in 19.91s
```
