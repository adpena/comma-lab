# Codex Findings - MLX Upstream Scorer Contract Gate

UTC: 2026-06-01T16:46:44Z

## Landing

- Added a fail-closed MLX scorer fidelity contract in `tac.local_acceleration.mlx_upstream_scorer_contract`.
- Wired `tools/build_mlx_scorer_input_cache.py` so every emitted cache manifest records upstream scorer contract validation and exits nonzero on mismatched shapes or false-authority flags.
- Hardened malformed cache manifests: missing shape keys now return blockers instead of crashing validation.
- Grounded joint P18/P19 waterfill in the contest-fixed byte water level, `25 / source_video_bytes`, and added a DeepFool-style SegNet top-2 margin weight bridge for boundary/class-region allocation.

## Contest Authority

This landing does not promote MLX outputs to contest score authority. The contract preserves:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

MLX remains an advisory acceleration lane until paired CPU/CUDA contest payloads, receiver proof, and exact-axis custody are present.

## Verification

- `.venv/bin/ruff check src/tac/local_acceleration/mlx_upstream_scorer_contract.py src/tac/tests/test_mlx_upstream_scorer_contract.py tools/build_mlx_scorer_input_cache.py src/tac/tests/test_contest_eval_contract.py src/tac/optimization/joint_p18_p19_waterfill.py src/tac/tests/test_joint_p18_p19_waterfill.py`
- `.venv/bin/pytest src/tac/tests/test_mlx_upstream_scorer_contract.py src/tac/tests/test_contest_eval_contract.py src/tac/tests/test_joint_p18_p19_waterfill.py -q`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_mlx_preprocess.py -q`

## Remaining Work

- Replace any remaining static SegNet class-region masks in active acquisition rows with measured margin/VJP surfaces when those bundles are available.
- Attach full-video exact-reduced P18/P19 cache validation to compact-base and HPRC runner outputs before exact-gate spend.
- Keep PR95/RNeRV/PACT-NeRV compact-base training as the highest-EV score-lowering path; use this contract as the guardrail, not as an endpoint.
