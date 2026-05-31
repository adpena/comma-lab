# Codex Findings: MLX Archive Emitter Contract Closure

UTC: 2026-05-31T09:49:18Z

## Verdict

Closed the remaining live MLX/archive emitter gaps found in the contract-first
audit without absorbing unrelated generated research artifacts.

## Landed changes

- `pact_nerv_ia3` now emits a shared archive-bound runtime adapter package,
  top-level archive-bound contract, deterministic replay bundle, receiver gate,
  exact-axis blocker, and posterior hook from its byte-closed materializer.
- PR95 local-training candidate rows now attach the shared archive-bound
  contract whenever they carry archive bytes, so acquisition can route from the
  common schema instead of loose archive/readiness fields.
- PR95 MLX timing-smoke `pr95_public_archive_export.json` now refreshes an
  archive-bound contract before initial export write, after runtime-consumption
  proof, and after full-frame parity proof.

## Confirmed not missing

The live audit found Z5, PACT-NeRV selectors v2/v3/v4, DQS1, byte-shaving, and
public-frontier intakes already using the shared bridge or contract surface.

## Verification

- `ruff check` on touched code and tests passed.
- `pytest src/tac/tests/test_pact_nerv_ia3_archive_candidate.py src/tac/tests/test_pr95_muon_local_training_integration.py -q`: 10 passed.
- `pytest src/tac/tests/test_run_pr95_mlx_timing_smoke.py::test_run_pr95_mlx_timing_smoke_cli_writes_queueable_manifests -q`: 1 passed.
- `pytest src/tac/tests/test_archive_bound_runtime_bridge_remaining_mlx_emitters.py src/tac/tests/test_pr95_mlx_pytorch_archive_package.py -q`: 11 passed.
- `pytest src/tac/tests/test_run_pr95_mlx_timing_smoke.py -q`: 1 passed.

## Remaining score-lowering work

This closes archive/MLX emitter custody for these surfaces. The next frontier
work is acquisition and bounded-runner actuation: consume the new contracts as
selection inputs, demote exact/receiver/preclaim failures into posterior budget
routing, and spend only on byte-closed candidates with receiver/runtime custody.
