# Codex Findings: MLX Scorer Production Contract Greenup

**UTC:** 2026-05-22T00:01:12Z  
**Lane:** `lane_mlx_auth_scorer_production_hardening_20260521`  
**Scope:** MLX port of the canonical auth upstream contest scorer, local acceleration and research-helper authority only.

## Finding

The MLX scorer-response surface needed one more fail-closed production boundary before it could be safely used as a durable local accelerator:

- GPU scorer responses already required explicit research-signal allowance, but non-singleton GPU batches still could be requested after allowance even though FEC6 calibration showed batch-shape drift.
- Profile-stability and batch-invariance manifests were separate artifacts; there was no single production contract checker that refused score authority and required those gates before a local MLX scorer-response artifact could be called production-safe.
- New local outputs needed explicit ignore coverage so cache/profile/contract artifacts remain rebuildable state rather than accidental committed payloads.

## Fix

- Added `GPU_BATCH_SHAPE_BLOCKER` to `src/tac/local_acceleration/mlx_scorer_response.py`.
- `tools/run_mlx_scorer_response_cache.py` now refuses `--device gpu --batch-pairs != 1` even with `--allow-gpu-research-signal`.
- `tools/profile_mlx_scorer_response_cache.py` now refuses GPU profile grids containing non-singleton batch sizes until a reviewed invariance override exists.
- Added `src/tac/local_acceleration/mlx_production_contract.py` and `tools/check_mlx_scorer_production_contract.py`.
- The production contract requires:
  - `schema_version=mlx_scorer_response.v1`
  - MLX evidence grade/tag/score axis
  - all score/promotion/rank/dispatch authority fields explicit false
  - `candidate_generation_only=true`
  - `requires_exact_eval_before_promotion=true`
  - recomputed-score provenance
  - finite component metrics
  - component array shapes matching `n_samples`
  - cache identity custody by default
  - passing profile-stability and batch-invariance manifests by default
- Updated `.gitignore` with MLX batch-invariance, production-contract, and execution-plan output roots.

## Authority

Passing the production contract means **production-safe local MLX scorer acceleration signal only**. It does not create a contest score claim, promotion decision, rank/kill decision, or exact-eval dispatch readiness. Paired contest CPU/CUDA auth eval remains mandatory for those surfaces.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_production_contract.py \
  src/tac/tests/test_mlx_execution_plan.py \
  src/tac/tests/test_mlx_scorer_response.py \
  src/tac/tests/test_profile_mlx_scorer_response_cache.py \
  src/tac/tests/test_mlx_batch_invariance.py \
  src/tac/tests/test_mlx_profile_stability.py -q
# 34 passed in 4.18s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_production_contract.py \
  src/tac/tests/test_mlx_execution_plan.py \
  src/tac/tests/test_mlx_batch_invariance.py \
  src/tac/tests/test_mlx_scorer_adapters.py \
  src/tac/tests/test_mlx_scorer_response.py \
  src/tac/tests/test_profile_mlx_scorer_response_cache.py \
  src/tac/tests/test_mlx_profile_stability.py \
  src/tac/tests/test_mlx_preprocess.py \
  src/tac/tests/test_mlx_scorer_fidelity.py \
  src/tac/tests/test_mlx_cache_audit.py \
  src/tac/tests/test_mlx_scorer_state_map.py \
  src/tac/tests/test_mlx_scorer_port_inventory.py \
  src/tac/tests/test_mlx_to_pytorch_export.py -q
# 104 passed in 19.61s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check \
  src/tac/local_acceleration/mlx_execution_plan.py \
  src/tac/local_acceleration/mlx_production_contract.py \
  src/tac/local_acceleration/mlx_profile_stability.py \
  src/tac/local_acceleration/mlx_scorer_response.py \
  src/tac/optimization/scorer_response_dataset.py \
  tools/check_mlx_scorer_production_contract.py \
  tools/plan_mlx_scorer_response_execution.py \
  tools/plan_ll_scorer_response_next.py \
  tools/profile_mlx_scorer_response_cache.py \
  src/tac/tests/test_mlx_execution_plan.py \
  src/tac/tests/test_mlx_production_contract.py \
  src/tac/tests/test_mlx_profile_stability.py \
  src/tac/tests/test_mlx_scorer_response.py \
  src/tac/tests/test_profile_mlx_scorer_response_cache.py
# All checks passed!

git diff --check
# clean
```
