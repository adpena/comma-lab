# Codex Findings - Catalog #204 ITEM_4 dispatch refused by predecessor probe

**UTC:** 2026-05-19T21:13:13Z  
**Task:** `codex_routing_directive_session_20260519_paid_dispatch_batch_C6_plus_204_followon_20260519T071500Z::ITEM_4`  
**Directive:** `.omx/research/codex_routing_directive_session_20260519_paid_dispatch_batch_C6_plus_204_followon_20260519T071500Z.md`  
**Prior hardening commit:** `7eea2d620`  
**Verdict:** BLOCKED, `catalog_313_refused`

## What Happened

The pre-dispatch guard patch landed and the focused tests passed, but the real
operator-authorize path still refused the Modal recovery dispatch before any
provider spawn, lane claim, or spend.

Command:

```text
OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=0.50 \
STACK_OF_STACKS_LANGEVIN_T_INIT_CAP=1.0 \
PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python tools/operator_authorize.py \
  --recipe substrate_stack_of_stacks_sgld_convergence_diagnostic_modal_t4_dispatch \
  --target modal \
  --yes \
  --agent codex:019de465 \
  --label-suffix _catalog204_a1_passthrough_recovery
```

Result:

```text
exit code: 1
[operator-authorize] FATAL: probe-disambiguator predecessor verdict DEFER blocks dispatch
substrate=stack_of_stacks
probe_id=harvest_e8_sgld_1_instant_crash_20260519
adjudicated_at=2026-05-19T06:10:00Z
metric=elapsed_seconds_before_crash=2.11
```

The dry-run path completes because it does not exercise the paid-dispatch
predecessor gate. The real `--yes` path exercises Catalog #313 and fails
closed.

## Blocking Authority

Canonical probe rows still active for this recipe include:

- `harvest_e8_sgld_1_instant_crash_20260519` — DEFER, blocking, 14-day window.
- `sgld_convergence_dispatch_trainer_only_single_arm_passthrough_not_real_sgld_DEFER_20260519` — DEFER, blocking, 30-day window.

The prior commit `7eea2d620` hardened the Catalog #204 durable-output and
auth-eval authority path, but it did not supersede the instant-crash predecessor
or prove that the stack-of-stacks recipe now runs a true SGLD convergence path
rather than the single-arm A1 passthrough path.

## Adversarial Review

Do not set `OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_VERDICT=1` from this
Codex loop. The goal cap explicitly says `#313 refuse predecessor
INDEPENDENT/KILL/DEFER <14d`; the current blocker is a same-day DEFER.

Do not mark ITEM_4 complete from the pre-dispatch hardening commit alone. The
directive's deliverable was a recovered Modal T4 inflate + auth-eval anchor; no
provider job was created and no `contest_auth_eval.json` was recovered.

Do not treat the dry-run as dispatch evidence. It validated recipe readability
and required-input resolution only.

## Next Valid Paths

1. Run a fresh sister probe that addresses the instant-crash root cause with
   captured output and updates the original probe outcome via the canonical
   ledger if it genuinely supersedes the blocker.
2. Build the dedicated SGLD-only trainer path requested by the older blocking
   probe, then reroute this recipe to that path.
3. Have council/operator ratify a fresh-evidence bypass and set the paired
   `OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_*` env vars explicitly. This
   Codex turn did not invoke that path.

## Authority

No score claim. No dispatch claim. No provider spend. No promotion, rank, or
retirement authority. ITEM_4 should be `blocked` with blocker
`catalog_313_refused:harvest_e8_sgld_1_instant_crash_20260519`.
