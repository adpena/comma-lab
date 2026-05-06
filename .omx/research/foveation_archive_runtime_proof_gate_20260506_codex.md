# Foveation Archive Runtime Proof Gate

Date: 2026-05-06
Author: codex
Evidence grade: pre-dispatch readiness hardening
Score claim: false
Dispatch attempted: false

## Context

Telescopic/hyperbolic foveation is useful only if the scored inflate runtime
actually consumes charged `foveation_params.bin` bytes. Payload geometry custody
alone is not enough.

## Change

`audit_foveation_params` now accepts optional:

- `candidate_archive`
- `runtime_consumer`

When supplied, the audit verifies that the archive contains
`foveation_params.bin` with matching bytes/SHA-256 and that the runtime consumer
references both the charged member name and the foveation loader. Without those
proofs, the readiness report remains blocked by:

- `foveation_charged_member_not_proven`
- `foveation_runtime_consumer_not_proven`

The CLI `tools/audit_hyperbolic_foveation_readiness.py` exposes the same inputs
and records them in the tool manifest.

## Boundary

This is not a score claim. Even with archive/runtime proofs, exact CUDA auth
eval remains required before dispatch, promotion, or ranking.
