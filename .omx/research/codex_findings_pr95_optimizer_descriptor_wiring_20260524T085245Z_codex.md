# Codex Findings: PR95 MLX Optimizer Descriptor Wiring

UTC: 2026-05-24T08:52:45Z

## Landing

- Added PR95-specific optimizer scheduler descriptors for Stage 1 AdamW,
  Stage 5 AdamW, Stage 8 Muon+AdamW, plus descriptor-only Muon-all-stages and
  Langevin-polish alternatives.
- Lowered executable PR95 descriptors into `Pr95MlxOptimizerConfig` while
  refusing descriptor-only recipes at execution time.
- Added MLX parameter-shape fingerprints using the canonical
  `embedding_theta1_hidden_muon_adamw` policy and carried descriptor/config/
  policy hashes through runtime profiles, manifests, queue plans, and optimizer
  candidate rows.
- Preserved optimizer descriptor identity in local-training queue metadata so
  queue telemetry cannot collapse distinct optimizer recipes into one apparent
  candidate.

## Authority Boundary

All descriptor rows are planning/local-training evidence only. Descriptor-only
alternatives carry `optimizer_backend_missing`; executable PR95 descriptors still
carry the exact-readiness blockers for source checkpoint parity, runtime
consumption proof, byte-closed contest export, and exact CPU/CUDA auth eval.

## Verification

- `.venv/bin/ruff check src/tac/local_acceleration/pr95_hnerv_mlx.py tools/run_pr95_mlx_timing_smoke.py src/tac/optimization/optimizer_scheduler_registry.py src/tac/optimization/representation_training_probe_integration.py src/tac/optimization/pr95_muon_local_training_integration.py src/comma_lab/scheduler/local_training_queue.py src/tac/tests/test_optimizer_scheduler_registry.py src/tac/tests/test_pr95_hnerv_mlx.py src/tac/tests/test_run_pr95_mlx_timing_smoke_plan.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py src/tac/tests/test_local_training_execution_queue.py src/tac/tests/test_representation_training_probe_integration.py src/tac/tests/test_pr95_muon_local_training_integration.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_optimizer_scheduler_registry.py src/tac/tests/test_pr95_hnerv_mlx.py src/tac/tests/test_run_pr95_mlx_timing_smoke_plan.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py src/tac/tests/test_local_training_execution_queue.py src/tac/tests/test_representation_training_probe_integration.py src/tac/tests/test_pr95_muon_local_training_integration.py`
- Manual plan-only CLI smoke for Stage 8 with
  `--optimizer-descriptor-id pr95_stage8_muon_adamw_mlx`.

Result: 50 focused tests passed; ruff passed; manual plan emitted descriptor ID,
64-hex config hash, policy ID, and descriptor-bearing execution command.

## Remaining Gap

The next optimizer tranche is a plan-matrix emitter for stages x optimizer
descriptors, followed by executable MLX/NumPy backends for the highest-EV
descriptor-only alternatives that survive cheap local timing and parity probes.
