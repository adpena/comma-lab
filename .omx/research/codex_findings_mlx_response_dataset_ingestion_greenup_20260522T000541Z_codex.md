# Codex Findings: MLX Response Dataset Ingestion Greenup

**UTC:** 2026-05-22T00:05:41Z  
**Lane:** `lane_mlx_auth_scorer_production_hardening_20260521`  
**Scope:** direct MLX scorer-response artifacts as scorer-response dataset rows.

## Finding

The direct MLX scorer-response ingestion path is useful for the research-helper side of the scorer port: it lets local MLX response artifacts become LL/scorer-response dataset rows without wrapping them in a legacy candidate envelope.

The first cut needed stricter authority and custody gates. Without them, a malformed direct MLX payload could enter the response dataset if it carried the right schema and false score-claim fields but omitted exact-eval-required provenance, score-axis tagging, recomputed-score provenance, or cache identity.

## Fix

- `src/tac/optimization/scorer_response_dataset.py` now accepts `schema_version=mlx_scorer_response.v1` as a direct candidate source.
- Direct MLX response ingestion now requires:
  - canonical MLX evidence grade and tag from `tac.local_acceleration`
  - `score_axis=[macOS-MLX research-signal]`
  - score/promotion/rank/dispatch authority fields explicit false
  - `candidate_generation_only=true`
  - `requires_exact_eval_before_promotion=true`
  - `canonical_score_source=score_recomputed_from_components`
  - finite `canonical_score`, `score_recomputed_from_components`, PoseNet, and SegNet metrics
  - exact equality between canonical and recomputed scores
  - `device_contract.forbidden_uses` including `score_claim`
  - cache identity with equal pair indices, candidate archive/raw SHA-256 custody,
    and reference archive/raw or array SHA-256 custody
- Normalized rows preserve MLX source schema, evidence grade/tag, substrate, pair window, batch size, sample count, elapsed seconds, and component hashes.

## Authority

These rows remain training/research-helper rows only. They are not score claims, rank/kill rows, promotion-ready rows, or exact-eval dispatch rows.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/tests/test_mlx_production_contract.py \
  src/tac/tests/test_mlx_execution_plan.py -q
# 35 passed in 0.48s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_scorer_response_dataset.py \
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
# 128 passed in 16.93s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check \
  src/tac/optimization/scorer_response_dataset.py \
  src/tac/tests/test_scorer_response_dataset.py
# All checks passed!
```
