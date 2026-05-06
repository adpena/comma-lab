# PR91 HPM1 Runtime-Contract Gate

Date: 2026-05-06
Agent: Codex
Evidence grade: empirical

## Finding

Gate #19 in `tools/all_lanes_preflight.py` correctly kept PR91/HPM1
fail-closed, but it required `hpac_device_contract_resolved` to remain present
as a dispatch blocker. That would make a future artifact with a pinned literal
CPU HPAC contract fail the preflight even though it should remain useful
non-dispatch evidence while `runtime_consumer_sidecar_free_hpm1` is unresolved.

## Change

The PR91 gate now accepts exactly two runtime-device states:

- current fail-closed evidence: ambient/contradictory HPAC device usage is
  recorded and `hpac_device_contract_resolved` remains a blocker.
- future resolved evidence: every visible HPAC call is pinned CPU-only, no
  ambient/contradictory calls remain, and only downstream runtime-consumption
  blockers keep the bundle non-dispatchable.

CUDA-only or otherwise ambiguous runtime-device contracts still fail closed.

## Guard

`src/tac/tests/test_all_lanes_pr91_gate.py` now covers the future resolved
literal-CPU state and rejects a resolved-CUDA state. The gate still requires
`score_claim=false`, `dispatch_attempted=false`,
`ready_for_exact_eval_dispatch=false`, manifest hash consistency, static source
inventory, PR91 archive/member custody, HPM1 mask custody, and ZIP wire-contract
evidence.

## Dispatch Status

No GPU dispatch or exact eval was attempted. This is preflight/operator-DX
hardening only; PR91/HPM1 remains blocked until full HPM1 decode/reencode,
sidecar-free runtime consumption, and exact CUDA auth eval are proven.
