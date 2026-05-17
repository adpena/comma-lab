# L5 v2 provider route blocker hardening - 2026-05-17

## Summary

This landing tightens the provider-readiness control plane for the L5 v2 TT5L
side-info effect-curve path. The previous provider inventory reported
Lightning as SDK-ready but left route blockers as generic
`credits_or_quota_not_checked` and `studio_route_not_checked`. That was too
coarse for autonomous continuation: it let future agents retread provider
unblock work and made it harder to distinguish a route-missing state from a
quota-checked but dispatch-unclaimed state.

No provider work was launched. No score claim, promotion claim, or
architecture-lock claim is made here.

## Fix

- `tools/cloud_provider_readiness.py` now inspects the Lightning route
  environment and reports explicit blockers for missing `LIGHTNING_TEAMSPACE`,
  missing `LIGHTNING_SDK_USER`/`LIGHTNING_ORG`, and missing
  `LIGHTNING_SSH_TARGET`.
- When the Lightning route is declared, the same provider row now requires the
  machine inventory, source manifest staging, remote CUDA runtime probe, quota
  check, and lane claim before dispatch.
- The provider readiness Markdown now preserves durable context and the
  "no exact-CUDA provider currently authorized" implication instead of writing
  only a table.
- The refreshed L5 v2 provider readiness ledger records Lightning as
  `ready_sdk_missing_lightning_route`, with exact-CUDA dispatch still false.

## Validation

- `.venv/bin/python -m ruff check tools/cloud_provider_readiness.py tests/test_cloud_provider_readiness.py`
- `.venv/bin/python -m pytest tests/test_cloud_provider_readiness.py -q` -> 13 passed
- `.venv/bin/python tools/cloud_provider_readiness.py --timeout-s 10 --output .omx/research/l5_v2_provider_readiness_20260517_codex.json --markdown-output .omx/research/l5_v2_provider_readiness_20260517_codex.md`
- `git diff --check`

## Current State

The next TT5L side-info effect-curve action is still provider unblock and lane
claims, not a new source-custody review. Lightning needs a declared teamspace,
owner, SSH target, source manifest staging, remote CUDA/runtime probe, machine
inventory check, quota/credit check, and per-axis lane claims before non-dry-run
exact eval.
