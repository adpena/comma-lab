# Codex Findings: Local Training Queue NumPy Resource Mapping

UTC: 2026-05-24T07:29:00Z

## Scope

The local representation-training queue must support HNeRV variants,
BoostNeRV/NeRV bolt-ons, and non-NeRV training probes without treating MLX or
NumPy observations as score authority. NumPy should use local CPU capacity while
MLX uses local MLX capacity.

## Landing

`comma_lab.scheduler.local_training_queue` now maps local NumPy execution plans
to `experiment_queue.v1` resource `local_cpu`:

- `training_backend in {"numpy", "np", "local_numpy", "macos_numpy"}`
- `device in {"cpu", "numpy", "local_numpy"}`
- explicit `scheduler_resource_kind` is honored first

Focused tests in `src/tac/tests/test_local_training_execution_queue.py` cover
local NumPy mapping and explicit scheduler resource hints.

## Verification

Passed:

- `src/tac/tests/test_local_training_execution_queue.py`
- `src/tac/tests/test_local_training_runtime_profile.py`
- `src/tac/tests/test_representation_training_probe_integration.py`

Focused result: `17 passed in 0.62s`.
Ruff passed on local-training queue, local-training CLI, runtime-profile, and
representation-training integration files plus focused tests.

## Remaining Gap

The queue compiler can schedule concrete local training plans, but family
adapters still need to emit runnable `recommended_execution` blocks for PR95
HNeRV, HNeRV variants, BoostNeRV overlays, broader NeRV-family models, and
non-NeRV representations. The highest-EV next step is a measured PR95/HNeRV
MLX Stage 1/5/8 smoke that writes `trainer_runtime_profile_observation.v1` and
feeds the optimizer candidate queue.
