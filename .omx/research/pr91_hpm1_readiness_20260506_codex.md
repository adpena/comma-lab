# PR91 HPM1 Readiness - 2026-05-06

This is a custody/readiness ledger, not a score claim.

- tool: `tools/audit_pr91_hpm1_readiness.py`
- artifact: `experiments/results/pr91_hpm1_readiness_20260506_codex/readiness.json`
- runtime-contract tool: `tools/audit_pr91_hpm1_runtime_contract.py`
- runtime-contract artifact: `experiments/results/pr91_hpm1_runtime_contract_20260506_codex/runtime_contract.json`
- score_claim: `false`
- dispatch_attempted: `false`
- ready_for_exact_eval_dispatch: `false`
- evidence_grade: `static_custody_plus_blocked_replay`

## Custody

The local public PR91 archive matches the expected archive custody:

- archive bytes: `222404`
- archive sha256: `4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f`
- member `x` sha256: `5c213c61cc4d29b62286063bfdcb97e812af6b06c0021aeaecc8bc46644e17bf`
- HPM1 mask segment bytes: `145087`
- HPM1 mask segment sha256: `a4ed57ff0af1d8c914f004de165aeead50ec8dd61e99b0afdfbfa2d1e7fd9fcc`
- HPM1 tokens sha256: `541016d83852a5bb3e0738caa3b44d7b2b0f7372f1841085cf9554f039c6cf6b`
- HPM1 HPAC sha256: `de7638c531c9dafa06148602cf784bf3ae9997f326f85cc25b9f3646b536abdd`

## Dispatch Blockers

PR91/HPM1 remains fail-closed for exact eval dispatch:

- `full_hpm1_decode_600_frames`
- `byte_exact_hpm1_reencode`
- `runtime_hpm1_loader_without_sidecars`
- `exact_cuda_auth_eval_after_parity`

The runtime-contract audit also blocks dispatch on:

- `hpac_device_contract_resolved`
- `runtime_consumer_sidecar_free_hpm1`

It records two ambient HPAC device call sites and one contradiction between a
CPU-determinism comment and an ambient `device` argument in the public runtime
sources. Resolve this before treating PR91/HPM1 as a byte-savings stacker.

Next safe work is to recover full 600-frame HPM1 probability/range decode,
prove byte-exact decode/reencode parity, pin the CPU/CUDA runtime contract,
then wire a sidecar-free contest inflate runtime before any lane claim or exact
CUDA auth eval.
