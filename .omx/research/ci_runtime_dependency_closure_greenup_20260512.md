# CI Runtime Dependency Closure Greenup (2026-05-12)

## Scope

Fixed the hosted CI collection failure that appeared after preflight moved
under the 30s DX budget.

## Finding

The CI job installed `tac[dev]` plus a hand-maintained tool list, but the
collected unit tests import runtime-backed modules:

- `tac.data` imports `av`.
- upstream scorer modules import `safetensors`, `timm`, `einops`, and
  `segmentation_models_pytorch`.
- wavelet residual tests use `PyWavelets` as an oracle.

This made CI depend on whatever happened to be present locally and failed on a
clean GitHub runner.

## Patch

- `runtime` now declares `safetensors` alongside the upstream scorer deps.
- `dev` now declares the CI tooling packages and `PyWavelets` test oracle.
- CI installs CPU Torch first, then `tac[dev,runtime]`, so pytest collection
  uses the package metadata instead of a duplicated ad hoc dependency list.

## Evidence

- GitHub run `25768201263` passed Ruff, `ty`, and preflight on commit
  `64d22d71`; pytest then failed only on missing dependency imports.
- Follow-up validation command: `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/ --ignore=src/tac/tests/test_scheduler_cli.py -q --no-header -m "not slow" --collect-only`.
