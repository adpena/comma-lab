# Codex Findings: MLX Strict PR101 Pose-Axis Calibration

timestamp_utc: 2026-05-22T07:10:08Z
agent: codex
lane: mlx_strict_calibration_pr101_pose_axis_20260522
evidence_grade: [macOS-MLX research-signal] calibrated against [contest-CPU]
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false

## Summary

Built a strict PR101 pose-axis MLX calibration pair using Modal/Linux CPU auth-eval tensor exports as the candidate-cache source of truth. Local macOS re-inflate was explicitly falsified for this PR101 runtime: it produced raw aggregate SHA mismatches against Modal CPU, so it remains invalid as a tensor-cache materialization bridge for PR101/HNeRV public runtimes.

## Artifacts

- Calibration root: `experiments/results/mlx_strict_score_calibration_pr101_pose_axis_20260522/`
- Baseline Modal tensor auth eval: `experiments/results/modal_auth_eval_cpu/mlx_strict_calibration_pr101_pose_baseline_tensors_cpu_20260522/contest_auth_eval.json`
- Candidate Modal tensor auth eval: `experiments/results/modal_auth_eval_cpu/mlx_strict_calibration_pr101_pose_candidate_tensors_cpu_20260522/contest_auth_eval.json`
- Baseline downloaded cache audit: `experiments/results/mlx_strict_score_calibration_pr101_pose_axis_20260522/baseline_cache_vs_auth_audit.json`
- Candidate downloaded cache audit: `experiments/results/mlx_strict_score_calibration_pr101_pose_axis_20260522/candidate_cache_vs_auth_audit.json`
- MLX response rows: `baseline_mlx_response_cpu_full600.json`, `candidate_mlx_response_cpu_full600.json`
- Score calibration: `experiments/results/mlx_strict_score_calibration_pr101_pose_axis_20260522/score_calibration_cpu.json`
- Advisory production contract: `experiments/results/mlx_strict_score_calibration_pr101_pose_axis_20260522/candidate_production_contract_advisory_with_score_calibration.json`

## Empirical Results

Modal CPU auth eval rows:

- Baseline archive `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`: CPU score `0.1928510127024255`.
- Candidate archive `30826b37093ee3af9512a1b46bd0b569fecbc4ccf75b8ff2dd746de113a5144a`: CPU score `0.19454975105500705`.

MLX CPU full-600 singleton response rows:

- Baseline MLX score `0.19285167228570535`; MLX minus CPU `+6.595832798550472e-7`.
- Candidate MLX score `0.1945516278680821`; MLX minus CPU `+1.8768130750634882e-6`.

Calibration summary:

- `mlx_cpu_rank_inversions`: `0`.
- `mlx_minus_cpu_max_abs`: `1.8768130750634882e-6`.
- `recommended_min_mlx_gap_for_spend_triage`: `9.38406537531744e-6`.
- Certified pairwise spend-triage comparisons: `1/1`.

## Guardrails Landed

- `tools/materialize_mlx_scorer_cache_from_auth_eval.py` now verifies runtime content custody, refuses non-default inflate policy/env replay, fails closed on raw aggregate mismatch, deletes generated caches after failed audits, and stamps passing cache manifests with audit path/hash.
- `tools/audit_mlx_scorer_input_cache.py --stamp-cache-manifest-on-pass` stamps downloaded Modal tensor caches only after `PASS_CACHE_AUTH_EVAL_IDENTITY`.
- `tools/run_mlx_scorer_response_cache.py` now rejects unaudited candidate caches by default.
- `tac.local_acceleration.mlx_score_calibration` now rejects MLX response rows whose candidate cache lacks a passing zero-residual auth-identity audit.

## Open Production Gates

The advisory production contract passes with no blockers only because torch parity, profile stability, and batch-invariance requirements were explicitly bypassed for this calibration artifact. Strict production deployment still needs:

- PyTorch-vs-MLX parity covering the response window.
- Profile-stability gate for the same candidate/reference cache pair and window.
- Batch-invariance gate only if production use moves beyond singleton batches.

## Recommendation

Use `score_calibration_cpu.json` for local spend-triage only when candidate MLX score gaps exceed `9.38406537531744e-6`. Do not use this MLX row for score claims, rank/kill, promotion, or leaderboard claims. For PR101/HNeRV public runtimes, prefer Modal/Linux tensor export or hash export over local macOS re-inflate because local raw reconstruction is not byte-identical.
