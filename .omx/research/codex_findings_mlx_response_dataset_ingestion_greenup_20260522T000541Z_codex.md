# Codex Findings: MLX Response Dataset Ingestion Greenup

**UTC:** 2026-05-22T00:05:41Z
**Lane:** `lane_mlx_auth_scorer_production_hardening_20260521`
**Scope:** direct MLX scorer-response artifacts as scorer-response dataset rows.

## Finding

The direct MLX scorer-response ingestion path is useful for the research-helper side of the scorer port: it lets local MLX response artifacts become LL/scorer-response dataset rows without wrapping them in a legacy candidate envelope.

The first cut needed stricter authority and custody gates. Without them, a malformed direct MLX payload could enter the response dataset if it carried the right schema and false score-claim fields but omitted score-axis tagging, recomputed-score provenance, or cache identity.

## Fix

- `src/tac/optimization/scorer_response_dataset.py` now accepts `schema_version=mlx_scorer_response.v1` as a direct candidate source.
- Direct MLX response ingestion now requires:
  - canonical MLX evidence grade and tag from `tac.local_acceleration`
  - `score_axis=[macOS-MLX research-signal]`
  - score/promotion/rank/dispatch authority fields explicit false
  - `candidate_generation_only=true`
  - `requires_exact_eval_before_promotion=true` when that optional field is present
  - `canonical_score_source=score_recomputed_from_components`
  - finite `canonical_score`, `score_recomputed_from_components`, PoseNet, and SegNet metrics
  - exact equality between canonical and recomputed scores
  - `device_contract.forbidden_uses` including `score_claim`
  - cache identity with equal pair indices, candidate archive/raw/inflated SHA-256 custody, and scorer-input array hashes when present
- Normalized rows preserve MLX source schema, evidence grade/tag, substrate, pair window, batch size, sample count, elapsed seconds, and component hashes.
- The LL next-probe planner now promotes MLX-only datasets into a specific `ll_mlx_cpu_stable_response_harvest` probe instead of falling through to an empty decoder-q recommendation.

## Authority

These rows remain training/research-helper rows only. They are not score claims, rank/kill rows, promotion-ready rows, or exact-eval dispatch rows.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check \
  src/tac/optimization/scorer_response_dataset.py \
  src/tac/tests/test_scorer_response_dataset.py \
  tools/build_scorer_response_dataset.py \
  tools/plan_ll_scorer_response_next.py
# All checks passed!

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/tests/test_plan_ll_scorer_response_next_cli.py \
  src/tac/tests/test_mlx_execution_plan.py -q
# 31 passed in 0.56s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/tests/test_plan_ll_scorer_response_next_cli.py \
  src/tac/tests/test_mlx_execution_plan.py \
  src/tac/tests/test_mlx_profile_stability.py \
  src/tac/tests/test_profile_mlx_scorer_response_cache.py \
  src/tac/tests/test_mlx_scorer_response.py \
  src/tac/tests/test_mlx_batch_invariance.py -q
# 54 passed in 3.89s
```

## Real FEC6 local-window artifact

Input response:

`experiments/results/mlx_scorer_response_profile_fec6_20260521T2340Z_pairs16_4_cpu_gpu/next_response_cpu_b2_pairs16_20.json`

Generated scorer-response dataset:

- `experiments/results/mlx_scorer_response_profile_fec6_20260521T2340Z_pairs16_4_cpu_gpu/scorer_response_dataset_cpu_b2_pairs16_20.json`
- `experiments/results/mlx_scorer_response_profile_fec6_20260521T2340Z_pairs16_4_cpu_gpu/scorer_response_dataset_cpu_b2_pairs16_20.md`

Generated LL plan:

- `experiments/results/mlx_scorer_response_profile_fec6_20260521T2340Z_pairs16_4_cpu_gpu/ll_scorer_response_next_probe_plan_mlx_cpu_b2_pairs16_20.json`
- `experiments/results/mlx_scorer_response_profile_fec6_20260521T2340Z_pairs16_4_cpu_gpu/ll_scorer_response_next_probe_plan_mlx_cpu_b2_pairs16_20.md`

The real dataset has `row_count=1`, `family_counts.mlx_scorer_response=1`, `score_claim=false`, and the LL plan's top probe is `ll_mlx_cpu_stable_response_harvest`.
