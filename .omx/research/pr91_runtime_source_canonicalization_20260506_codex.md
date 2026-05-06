# PR91 Runtime Source Canonicalization

Date: 2026-05-06
Author: codex
Evidence grade: static custody hardening
Score claim: false
Dispatch attempted: false

## Context

PR91/HPM1 readiness used a replay-submission directory whose inventory had
only `__pycache__` bytecode files. The separate PR91 runtime-contract audit
already used the release-view source tree with `inflate.py`, `pr86_hpac.py`,
`inflate.sh`, and `range_mask_codec.cpp`. Those two custody views should not
diverge.

## Change

Readiness now uses the release-view PR91 runtime source directory by default.
The runtime-source inventory requires at least:

- `inflate.py`
- `pr86_hpac.py`

Pycache-only inventories fail closed with
`failed_closed_missing_required_runtime_sources`. The all-lanes PR91 gate also
checks live and committed readiness artifacts for required runtime sources and
rejects pycache-only custody.

## Boundary

This does not alter PR91 score status. The readiness report remains
non-dispatchable until HPM1 full decode/reencode parity, sidecar-free runtime
consumption, and exact CUDA auth eval are proven.
