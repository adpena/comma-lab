# Provider canonicalization and score-path hardening (2026-05-11)

## Scope

This pass migrated dispatch-claim construction toward a single reusable
contract and made all current provider surfaces visible in the canonical
provider registry/readiness probe.

Score-lowering relevance: this reduces provider drift between candidate
generation, exact-eval dispatch, harvest, and promotion. A score win is not
usable if the provider wrapper silently skips a lane claim, loses custody, or
misclassifies proxy hardware as score authority.

## Landed implementation

- Added `tac.deploy.claims.DispatchClaimSpec` plus canonical claim-command,
  active-claim, terminal-claim, UTC timestamp, and predicted-ETA helpers.
- Migrated Modal auth-eval claim handling to the shared helper while retaining
  the Modal-specific `ClaimSpec(platform="modal")` compatibility surface.
- Migrated `scripts/launch_lane_on_vastai.py` active and terminal claim command
  construction to the same helper.
- Migrated the active T1 Modal Ballé dispatcher plus the Phase A1 and Phase A4
  Modal lane wrappers to the same helper, eliminating their lane-local
  `claim_lane_dispatch.py` argv construction.
- Expanded `tac.deploy.provider_contracts` from 5 providers to the full active
  provider set: Modal, Kaggle, Lightning, Vast.ai, AWS, Azure, and GCP.
- Added a registry guard that detects any new `src/tac/deploy/<provider>/`
  package without a canonical provider contract.
- Expanded `tools/cloud_provider_readiness.py` to probe Lightning and Vast.ai,
  and to emit readiness in provider-contract order.

## Read-only provider readiness snapshot

Command:

```bash
.venv/bin/python tools/cloud_provider_readiness.py --timeout-s 2 --output experiments/results/cloud_provider_readiness_latest.json --markdown-output experiments/results/cloud_provider_readiness_latest.md
```

Snapshot at `2026-05-11T07:29:38Z`:

| provider | status | score-lowering use now |
|---|---|---|
| modal | `ready_cli_check_runtime_probe_next` | candidate exact-eval path, but billing/CUDA import probe still required before new dispatch |
| kaggle | `ready_proxy` | free Optuna/CMA-ES/proxy sweeps only; no score authority |
| lightning | `ready_sdk_check_credit_quota_next` | SDK present; credit/quota/studio route unchecked |
| vastai | `blocked_credentials_missing` | blocked locally until `vastai set api-key` |
| aws | `blocked_auth` / `aws_session_expired` | blocked until AWS login and quota/budget checks |
| azure | `blocked_auth` | blocked until `az login` and quota/budget checks |
| gcp | `blocked_billing` | selected project has billing disabled; blocked until billed project/quota checks |

## Active dispatch state

`tools/claim_lane_dispatch.py summary` reports exactly one active dispatch:

- `t1_balle_128k_endtoend`
- job `t1_balle_modal_phase1_ab2d0f6_20260510T1437Z`
- platform `modal`
- status `active_dispatching`

No duplicate T1 dispatch should be launched while this claim remains active.

## Verification

Focused tests:

```bash
.venv/bin/python -m pytest src/tac/tests/test_deploy_claims.py src/tac/tests/test_provider_deploy_contracts.py tests/test_cloud_provider_readiness.py src/tac/tests/test_launch_lane_on_vastai_create_instance.py src/tac/tests/test_modal_auth_eval.py -q
```

Result: `56 passed`.

Syntax:

```bash
.venv/bin/python -m py_compile src/tac/deploy/claims.py src/tac/deploy/provider_contracts.py src/tac/deploy/modal/auth_eval.py tools/cloud_provider_readiness.py scripts/launch_lane_on_vastai.py experiments/modal_t1_balle_endtoend.py experiments/modal_phase_a1_score_gradient_pr101.py experiments/modal_phase_a4_charm_50k_toy_substrate.py
```

Result: passed.

## Next provider cleanup targets

1. Replace legacy provider packagers that still hand-roll repo layout with the
   canonical bundle/bootstrap surfaces.
2. Add a preflight guard that flags new direct `tools/claim_lane_dispatch.py`
   subprocess construction outside the canonical helper unless explicitly
   waived.
