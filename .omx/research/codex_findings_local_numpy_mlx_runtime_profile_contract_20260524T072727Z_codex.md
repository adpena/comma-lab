# Codex Findings: Local NumPy/MLX Runtime Profile Contract

UTC: 2026-05-24T07:27:27Z

## Scope

The acquisition layer should not be locked to DQS1 or to one representation
family. Local HNeRV, BoostNeRV/NeRV, and non-NeRV representation-training smokes
need a shared runtime profile contract so the solver can compare local NumPy and
MLX throughput before expensive archive/export/exact-eval work.

## Landing

`tac.optimization.local_training_runtime_profile` now recognizes NumPy backends:

- `numpy`, `np`, `local_numpy`, and `macos_numpy` normalize to
  `training_backend="local_numpy"`.
- `local_numpy` maps to scheduler resource `local_cpu`.
- `mlx`, `local_mlx`, and `macos_mlx` normalize to `training_backend="mlx"` and
  scheduler resource `local_mlx`.
- Runtime profile rows carry `scheduler_resource_kind`, `evidence_grade`, and
  `evidence_tag`.

`tac.optimization.representation_training_probe_integration` now propagates the
best runtime profile's scheduler resource into candidate params as
`best_scheduler_resource_kind`, so representation-training probe candidates keep
the local execution hint when they enter optimizer queues.

## Authority Boundary

This remains cost/scheduling signal only. Local NumPy and MLX runtime profiles do
not become score, promotion, rank/kill, or dispatch authority. Additional
backend-specific blockers are emitted:

- `local_numpy_training_profile_not_score_authority`
- `local_mlx_training_profile_not_score_authority`

## Verification

- `src/tac/tests/test_local_training_runtime_profile.py`
- `src/tac/tests/test_representation_training_probe_integration.py`
- `src/tac/tests/test_optimizer_candidate_queue.py`

Focused run passed: `45 passed in 0.45s`.
Ruff passed on the runtime-profile and representation-training integration
files plus their focused tests.

## Remaining Gaps

1. The follow-up local-training queue compiler now exists in
   `comma_lab.scheduler.local_training_queue`, but it still needs concrete MLX
   adapter producers for PR95/HNeRV, BoostNeRV, broader NeRV variants, and
   non-NeRV representations.
2. MLX scorer-response acquisition still needs continuous batch execution and
   harvest back into the action functional, not just candidate/result adapters.
3. Family-specific archive materializers remain the narrow production gap:
   HNeRV section recodes, BoostNeRV tensor overlays, broader NeRV variants, and
   non-NeRV packet/member operations need concrete adapters before water-filled
   operation sets can become byte-closed candidates.
