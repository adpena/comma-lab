# Codex Findings: MLX Local Training Runtime Profile

Generated: 2026-05-23T02:15:36Z

## Scope

The local MLX/CPU training stack needed a reusable way to turn timing,
operator-mix, kernel-fusion, memory, drift, and archive/export gate evidence
into planner signal without leaking score authority. This closes the first
generic bridge for using local MLX as a cloud-GPU replacement substrate for
BoostNeRV, PR95/HNeRV variants, broader NeRV-family variants, non-NeRV learned
codecs, and future representation classes.

## Landing

- Added `trainer_runtime_profile_observation.v1` in
  `src/tac/optimization/local_training_runtime_profile.py`.
- Wired standalone runtime profiles into `tac.optimizer.candidate_queue`.
- Wired embedded runtime profiles into both generic
  `representation_training_probe_manifest_v1` and the PR95/HNeRV local-training
  adapter.
- Extended optimizer/scheduler telemetry to carry measured backend/kernel
  context: backend, kernel-fusion strategy, backend kernel contract, operator
  mix, numerical drift profile, and ineligible reason.
- Extended optimizer-training variant axes with backend/kernel/drift dimensions.

## Authority Boundary

All new rows remain false-authority planning signal:

- no score claim;
- no rank/kill authority;
- no exact-eval dispatch readiness;
- no promotion eligibility;
- byte-closed archive export, PacketIR/archive compiler gate, runtime
  consumption proof, and exact CPU/CUDA auth eval remain required.

## Verification

```bash
.venv/bin/python -m ruff check src/tac/optimization/local_training_runtime_profile.py src/tac/optimization/representation_training_probe_integration.py src/tac/optimization/pr95_muon_local_training_integration.py src/tac/optimization/optimizer_scheduler_registry.py src/tac/optimization/optimizer_training_signal_bridge.py src/tac/optimizer/candidate_queue.py src/tac/tests/test_local_training_runtime_profile.py src/tac/tests/test_representation_training_probe_integration.py src/tac/tests/test_pr95_muon_local_training_integration.py src/tac/tests/test_optimizer_scheduler_registry.py
.venv/bin/python -m pytest -q src/tac/tests/test_local_training_runtime_profile.py src/tac/tests/test_representation_training_probe_integration.py src/tac/tests/test_pr95_muon_local_training_integration.py src/tac/tests/test_optimizer_scheduler_registry.py src/tac/tests/test_optimizer_candidate_queue.py
```

Result: `46 passed`; ruff clean.

## Next Integration

Feed measured BoostNeRV/PR95/HNeRV/NeRV/non-NeRV local MLX timing smokes into
`trainer_runtime_profile_observation.v1`, then let learned sweep and candidate
queues treat those rows as cost/acquisition signal. Exact score authority stays
on claimed contest CPU/CUDA anchors.
