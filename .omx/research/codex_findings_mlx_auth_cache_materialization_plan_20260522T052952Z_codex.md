# Codex Findings: MLX Auth-Cache Materialization Plan

utc: 2026-05-22T05:29:52Z
lane: lane_mlx_auth_cache_materialization_plan_20260522
status: LANDED
evidence_grade: macOS-MLX-research-signal
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false

## Finding

The FEC6 PR101 local MLX cache still cannot be used as an auth-axis transfer
calibration target. The archive identity matches the Modal/Linux contest-CPU
auth eval, but the decoded raw surface and scorer-input tensor surface do not:

- archive SHA-256 matches:
  `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- local macOS inflated aggregate/raw:
  `dbc67c898ecb158912f86c920f09bf2c68307b77c1cec3c1baa27a845d3850f1` /
  `d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c`
- Modal/Linux contest-CPU inflated aggregate/raw:
  `10c68e4266e79fc3e878fd20136e8aaa56262b3a2ff45eed7b8d5a4b1e1ee66d` /
  `fef02ccd53ad4355f2dbb8e0b9cd4efb847daa243bd35a8411c5260d584fda8b`
- pair indices hash matches, but SegNet and PoseNet scorer-input hashes differ.

This is not a PyTorch-vs-MLX parity failure. It is an auth-surface
materialization blocker: local MLX training targets must be rebuilt from the
Modal/Linux raw bytes or from a Linux-exported tensor cache before local MLX can
serve as production training signal.

## Landing

Added:

- `src/tac/local_acceleration/mlx_auth_cache_materialization.py`
- `tools/plan_mlx_auth_cache_materialization.py`
- `src/tac/tests/test_mlx_auth_cache_materialization.py`

The helper consumes the existing cache/auth audit and emits a fail-closed plan
with:

- false-authority fields preserved at the plan surface;
- archive/raw/scorer-input identity classification;
- required auth artifacts;
- machine-readable `next_materialization_action`.

Live FEC6 plan:

`experiments/results/mlx_production_contract_fec6_pr101_singleton_window_20260522T051339Z/auth_cache_materialization_plan_v1.json`

Plan verdict:

- `passed=false`
- `verdict=AUTH_CACHE_MATERIALIZATION_REQUIRED`
- `next_materialization_action=materialize_auth_axis_tensor_cache_from_modal_linux_raw_or_export_linux_tensor_cache`

## Verification

- `.venv/bin/ruff check src/tac/local_acceleration/mlx_auth_cache_materialization.py tools/plan_mlx_auth_cache_materialization.py src/tac/tests/test_mlx_auth_cache_materialization.py`
- `.venv/bin/python -m pytest src/tac/tests/test_mlx_auth_cache_materialization.py src/tac/tests/test_mlx_cache_audit.py src/tac/tests/test_mlx_production_contract.py -q`

Result: 42 passed.

## Next Action

Do not train local MLX scorer-response targets from the current macOS advisory
cache. The next production-enabling step is one of:

1. recover/materialize the Modal/Linux inflated raw file with SHA-256
   `fef02ccd53ad4355f2dbb8e0b9cd4efb847daa243bd35a8411c5260d584fda8b`,
   then rebuild the local MLX scorer-input cache from that raw surface; or
2. add a Modal/Linux tensor-cache export path that writes the auth-side
   `segnet_last_rgb`, `posenet_yuv6_pair`, and `pair_indices` tensors to a
   durable artifact store, then verify local hashes against the auth hash
   manifest.

Only after `PASS_CACHE_AUTH_EVAL_IDENTITY` should local MLX training targets be
eligible for production local-acceleration use.
