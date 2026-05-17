# Cloud provider readiness

Generated: `2026-05-17T09:05:06Z`

This is a read-only provider inventory. It is not a dispatch, score claim, or promotion artifact.

Context: TT5L side-info effect-curve work units are byte-closed, the Lightning
paired-axis dry-run plan is source-current, and architecture lock still refuses
score/promotion authority. Provider execution remains blocked. This inventory
records the live provider surface after refreshing
`l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_20260517_codex`.

Command:

```bash
.venv/bin/python tools/cloud_provider_readiness.py --timeout-s 10 --output .omx/research/l5_v2_provider_readiness_20260517_codex.json --markdown-output .omx/research/l5_v2_provider_readiness_20260517_codex.md
```

| provider | status | exact CUDA allowed now | proxy only | blockers | next action |
|---|---|---:|---:|---|---|
| modal | ready_cli_check_runtime_probe_next | no | no | modal_billing_not_checked, cuda_runtime_import_probe_not_run | Harvest active Modal A1 calls before refiring. |
| kaggle | ready_proxy | no | yes | - | Use Kaggle for Optuna/CMA-ES/proxy curves only. |
| lightning | ready_sdk_check_credit_quota_next | no | no | credits_or_quota_not_checked, studio_route_not_checked, no_dispatch_claim | Run `scripts/launch_lightning_batch_job.py doctor` before any Lightning dispatch. |
| vastai | blocked_credentials_missing | no | no | vastai_api_key_missing | Run `vastai set api-key <key>` before any Vast.ai launch. |
| aws | blocked_auth | no | no | aws_session_expired | Run `aws login`, then re-run this readiness tool. |
| azure | blocked_auth | no | no | azure_not_logged_in | Run `az login`, then re-run this readiness tool. |
| gcp | blocked_billing | no | no | gcp_billing_not_enabled_or_not_readable, gpu_quota_not_checked | Enable billing on the selected GCP project or switch to a billed project. |

Implication for L5 v2:

- No provider row currently authorizes exact-CUDA dispatch.
- Modal CLI exists, but billing/runtime import probes are not cleared.
- Lightning SDK exists, but quota/studio route/claim prerequisites are not
  cleared.
- Kaggle is proxy-only and cannot promote or rank TT5L.
- The next material TT5L action is provider unblock plus lane claims, not a
  new source-custody review.
