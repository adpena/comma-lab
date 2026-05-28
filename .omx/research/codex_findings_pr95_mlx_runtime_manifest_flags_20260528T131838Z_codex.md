# Codex Findings: PR95 MLX Runtime Manifest Receiver Flags

UTC: 2026-05-28T13:18:38Z

## Scope

Small schema hardening after the PR95 MLX runtime-consumption proof.

## Finding

The runtime proof path set `receiver_contract_satisfied=true` and `receiver_proof_present=true` on the timing manifest, and nested packet-compiler metadata recorded proof state, but the top-level manifest did not expose `runtime_consumption_proof_present` or `runtime_consumption_proven`.

## Fix

`tools/run_pr95_mlx_timing_smoke.py` now writes both top-level fields after a runtime proof is produced:

- `runtime_consumption_proof_present=true`
- `runtime_consumption_proven=<proof runtime_consumption_proven>`

The existing runtime-consumption test now asserts these fields plus the receiver flags. This gives downstream observers and queue planners a stable shallow manifest surface instead of requiring nested packet-compiler inspection.

## Authority

Schema/discoverability hardening only; it does not change score authority or promotion eligibility.
