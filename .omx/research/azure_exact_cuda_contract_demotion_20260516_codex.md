# Azure Exact-CUDA Contract Demotion - 2026-05-16

## Scope

Patch class: provider contract hardening / false-authority prevention.

Finding: the Azure provider registry advertised `exact_cuda_eval_supported=true`
while the launcher and deploy module only own VM lifecycle scaffolding. The
provider does not yet enforce a pre-provision lane claim, terminal claim,
runtime custody manifest, CUDA import probe, and harvest custody path end to
end.

## Change

- Demoted Azure from `status=implemented` to `status=scaffold`.
- Set `execution_flag=None` and `exact_cuda_eval_supported=false`.
- Added explicit setup blockers for lane-claim lifecycle, runtime custody,
  CUDA import probe, and harvest custody.
- Added a regression test that keeps Azure scaffold-only until those contracts
  land.

## Evidence

- `.venv/bin/python -m pytest -q src/tac/tests/test_provider_deploy_contracts.py`
- `.venv/bin/ruff check src/tac/deploy/provider_contracts.py src/tac/tests/test_provider_deploy_contracts.py`

## Evidence Boundary

No provider dispatch. No score claim. No Azure launch. This only removes false
exact-CUDA authority from the static provider registry.
