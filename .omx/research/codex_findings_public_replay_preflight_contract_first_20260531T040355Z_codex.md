# Codex Findings: Public Replay Preflight Contract-First Fix

- UTC: 2026-05-31T04:03:55Z
- Lane: `lane_codex_contract_first_adversarial_review_20260531`
- Commit target: Codex-owned follow-up after `f60743022`

## Finding

`experiments/preflight_public_replay_intake.py` could report
`ready_for_exact_eval_dispatch=true` when only static archive/runtime linting
had passed. That conflated a useful public-frontier intake preclaim with exact
dispatch authority.

## Fix

The preflight now emits a shared archive-bound candidate contract for the
public replay archive/runtime pair and keeps `ready_for_exact_eval_dispatch`
false. A clean static pass is exposed as `public_replay_preclaim_ready`, while
the contract remains blocked on receiver runtime consumption proof and contest
CPU/CUDA authority.

## Protection

Added `src/tac/tests/test_public_replay_intake_contract.py` to lock the
contract behavior:

- archive file custody can be complete;
- runtime adapter lint can be clean;
- exact dispatch remains false;
- missing receiver proof is a durable contract blocker;
- advisory/static-preflight anti-patterns are routed into acquisition penalties.

## Remaining Gap

Public PR audit result objects still carry only their historical audit fields.
They should be migrated next to embed the same contract surface when a local
downloaded archive path exists.
