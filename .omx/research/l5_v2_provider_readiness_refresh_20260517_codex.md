# Cloud provider readiness

Generated: `2026-05-17T10:48:40Z`

This is a read-only provider inventory. It is not a dispatch, score claim, or promotion artifact.

Context: provider readiness is an execution-unblocker ledger for byte-closed candidate work units. It does not override lane claims, source-custody checks, exact-eval adjudication, or architecture-lock gates.

| provider | status | exact CUDA allowed now | proxy only | blockers | next action |
|---|---|---:|---:|---|---|
| modal | ready_cli_check_runtime_probe_next | no | no | modal_billing_not_checked, cuda_runtime_import_probe_not_run | Harvest active Modal A1 calls before refiring. |
| kaggle | ready_proxy | no | yes | - | Use Kaggle for Optuna/CMA-ES/proxy curves only. |
| lightning | ready_sdk_missing_lightning_route | no | no | credits_or_quota_not_checked, no_dispatch_claim, lightning_teamspace_missing, lightning_owner_missing, lightning_ssh_target_missing | Set LIGHTNING_TEAMSPACE plus LIGHTNING_SDK_USER or LIGHTNING_ORG before Lightning dispatch. |
| vastai | blocked_credentials_missing | no | no | vastai_api_key_missing | Run `vastai set api-key <key>` before any Vast.ai launch. |
| aws | blocked_auth | no | no | aws_session_expired | Run `aws login`, then re-run this readiness tool. |
| azure | blocked_auth | no | no | azure_not_logged_in | Run `az login`, then re-run this readiness tool. |
| gcp | blocked_billing | no | no | gcp_billing_not_enabled_or_not_readable, gpu_quota_not_checked | Enable billing on the selected GCP project or switch to a billed project. |

## Implication

- No provider row currently authorizes exact-CUDA dispatch when `exact CUDA allowed now` is `no` for every provider.
- Lightning route blockers are explicit: configure `LIGHTNING_TEAMSPACE`, `LIGHTNING_SDK_USER` or `LIGHTNING_ORG`, and `LIGHTNING_SSH_TARGET`, then run the Lightning doctor with required SSH, machine inventory, and remote supply-chain checks before dispatch.
- Kaggle remains proxy-only and cannot promote or rank contest candidates.
- Provider readiness is not a score claim; harvested exact CPU/CUDA artifacts still need custody, adjudication, and lane-claim closure.
