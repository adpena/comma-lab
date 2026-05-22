# Codex Findings: MLX Strict Production Contract V2

timestamp_utc: 2026-05-22T08:28:12Z
agent: codex
lane: mlx_strict_production_contract_pr101_pose_axis_20260522
evidence_grade: [macOS-MLX research-signal] calibrated against [contest-CPU]
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false

## Summary

Converted the PR101 pose-axis MLX local scorer path from advisory-only to a
strict non-authoritative production signal contract for local candidate
generation and spend triage.

Final artifact:

`experiments/results/mlx_strict_score_calibration_pr101_pose_axis_20260522/candidate_production_contract_strict_v2.json`

Verdict:

- `PASS_MLX_SCORER_PRODUCTION_CONTRACT`
- `score_authority=false`
- `contest_authority=false`
- exact CPU/CUDA auth eval still required for score claims, promotion, rank/kill,
  and leaderboard decisions

## Structural Fixes

- Required reference-side PyTorch-vs-MLX parity in addition to candidate-side
  parity.
- Required parity sweep window shape and stride to match response `batch_pairs`;
  stale 4-pair sweeps can no longer certify singleton responses.
- Added a bounded parity-threshold guard: at most 1 SegNet argmax pixel per
  singleton window, with existing PoseNet/logit tolerances not loosenable beyond
  the canonical limits.
- Bound score calibration rows to the exact response identity: archive SHA,
  inflated-output aggregate SHA, response family, pair window, component hashes,
  and reference/candidate cache tensor hashes.
- Bound profile stability to the actual response values and component hashes;
  single-row/vacuous stability no longer passes.
- Kept non-singleton MLX production use blocked until a tiled batch-invariance
  sweep exists.
- Added complete false-authority fields to scorer-input cache manifests and
  auth-identity audit stamps.

## Empirical Evidence

Candidate-side parity:

- Artifact: `candidate_torch_parity_sweep_cpu_singleton_full600.json`
- Result: 600/600 singleton windows passed at strict 0 argmax-pixel tolerance.

Reference-side parity:

- Strict 0-pixel artifact: `reference_torch_parity_sweep_cpu_singleton_full600.json`
- Result: 598/600 passed; two one-pixel SegNet argmax flips at near-zero top-2
  margin.
- Bounded artifact: `reference_torch_parity_sweep_cpu_singleton_full600_argmax1.json`
- Result: 600/600 passed with `max_segnet_argmax_diff_pixels=1`.

Score calibration:

- Artifact: `score_calibration_cpu_v2.json`
- `mlx_cpu_rank_inversions=0`
- `mlx_minus_cpu_max_abs=1.8768130750634882e-6`
- `recommended_min_mlx_gap_for_spend_triage=9.38406537531744e-6`

Profile stability:

- Artifact: `candidate_profile_stability_cpu_singleton_full600_repeat2.json`
- Result: `PASS_MLX_PROFILE_STABILITY`
- Throughput: fastest row `1.1877396567999097` pairs/sec on local MLX CPU.

## Production Contract

Contract artifact:

`experiments/results/mlx_strict_score_calibration_pr101_pose_axis_20260522/candidate_production_contract_strict_v2.json`

Required gates:

- cache identity: true
- cache/auth audit: true
- candidate torch parity: true
- reference torch parity: true
- profile stability: true
- score calibration: true
- batch invariance: false for singleton response only

Warning retained:

- `batch_invariance_not_required_for_singleton_response`

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_mlx_production_contract.py src/tac/tests/test_mlx_profile_stability.py src/tac/tests/test_mlx_score_calibration.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_mlx_production_contract.py src/tac/tests/test_mlx_profile_stability.py src/tac/tests/test_mlx_score_calibration.py src/tac/tests/test_mlx_cache_audit.py src/tac/tests/test_materialize_mlx_scorer_cache_from_auth_eval.py`
- `.venv/bin/ruff check src/tac/local_acceleration/mlx_production_contract.py src/tac/local_acceleration/mlx_profile_stability.py src/tac/local_acceleration/mlx_score_calibration.py src/tac/local_acceleration/mlx_preprocess.py tools/check_mlx_scorer_production_contract.py tools/audit_mlx_scorer_input_cache.py tools/materialize_mlx_scorer_cache_from_auth_eval.py src/tac/tests/test_mlx_production_contract.py src/tac/tests/test_mlx_profile_stability.py src/tac/tests/test_mlx_score_calibration.py`

All passed.

## Next Action

Use this strict MLX contract as the local CPU scorer-response gate for PR101
pose-axis candidate-generation and spend triage. A local MLX candidate should
only affect paid exact-eval ordering when the predicted MLX gap exceeds
`9.38406537531744e-6`; every promoted candidate still needs byte-closed
contest CPU/CUDA auth eval.
